import os
if os.name == 'nt':
    try:
        import ctypes
        awareness = ctypes.c_int()
        ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
        if awareness.value != 2:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

import threading
import queue
import time
import numpy as np
import soundcard as sc
from vispy import app, gloo

# Quality and Performance Settings
FRAME_RATE = 200  # Reduced frame rate for more stability
QUALITY_SCALE = 1.2  # Balanced quality scale
SMOOTHING_FACTOR = 0.9  # Smoothing between frames

# Shader definitions
VERT_SHADER = """
#version 120
attribute vec2 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texcoord = a_texcoord;
}
"""

FRAG_SHADER = """
#version 120
uniform sampler2D u_texture;
varying vec2 v_texcoord;

void main() {
    vec2 texcoord = vec2(v_texcoord.y, v_texcoord.x);
    float intensity = texture2D(u_texture, texcoord).r;
    
    intensity = pow(intensity, 4.5);
    intensity *= 0.7;
    
    vec3 color;
    const float BLACK_THRESHOLD = 0.01;
    const float PURPLE_MAX = 0.05;
    const float YELLOW_MAX = 0.2;
    const float ORANGE_MAX = 0.4;
    
    if(intensity < BLACK_THRESHOLD) {
        color = vec3(0.0);
    }
    else if(intensity < PURPLE_MAX) {
        float t = intensity / PURPLE_MAX;
        color = mix(vec3(0.5, 0.0, 0.5), vec3(0.8, 0.0, 0.8), t);
    }
    else if(intensity < YELLOW_MAX) {
        float t = (intensity - PURPLE_MAX) / (YELLOW_MAX - PURPLE_MAX);
        color = mix(vec3(0.8, 0.0, 0.8), vec3(1.0, 1.0, 0.0), t);
    }
    else if(intensity < ORANGE_MAX) {
        float t = (intensity - YELLOW_MAX) / (ORANGE_MAX - YELLOW_MAX);
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), t);
    }
    else {
        float t = (intensity - ORANGE_MAX) / (1.0 - ORANGE_MAX);
        color = mix(vec3(0.9, 0.2, 0.0), vec3(1.0, 0.0, 0.5), t);
    }
    
    color = pow(color, vec3(1.0/2.2));
    gl_FragColor = vec4(color, 1.0);
}
"""

class AudioVisualizer(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, title='WaterFall Audio Visualizer', 
                          size=(800, 1000), 
                          keys='interactive',
                          vsync=True)

        # Audio processing parameters
        self.chunk_size = int(2048 * QUALITY_SCALE)
        self.n_history = int(500 * QUALITY_SCALE)
        self.distortion_reduction = 0.9
        self.dynamic_range = 60
        self.samplerate = 48000

        # Initialize visualization
        self.n_freq_bins = self.chunk_size // 2 + 1
        self.texture_data = np.zeros((self.n_history, self.n_freq_bins), dtype=np.float32)
        
        # Add buffers for smoothing
        self.previous_spectrum = None
        self.smoothing_factor = SMOOTHING_FACTOR
        self.texture_data_buffer = np.zeros_like(self.texture_data)
        
        # Add frequency smoothing
        self.freq_smoothing = np.ones(3) / 3
        self.freq_smoothing_enabled = True
        
        # Add temporal smoothing
        self.temporal_smoothing_buffer = np.zeros((5, self.n_freq_bins))
        self.temporal_smoothing_weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])

        # Setup texture
        self.texture = gloo.Texture2D(
            self.texture_data,
            interpolation='linear',
            internalformat='r32f',
            format='luminance'
        )

        # Program setup
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = gloo.VertexBuffer(np.array([
            [-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32))
        self.program['a_texcoord'] = gloo.VertexBuffer(np.array([
            [0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
        self.program['u_texture'] = self.texture

        # Enable antialiasing
        gloo.set_state(clear_color='black', blend=True)
        gloo.set_blend_func('src_alpha', 'one_minus_src_alpha')

        # Audio processing setup
        self.window = np.hanning(self.chunk_size)
        self.audio_queue = queue.Queue(maxsize=int(50 * QUALITY_SCALE))

        # Start audio capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_audio, daemon=True)
        self.processor_thread = threading.Thread(target=self.process_audio, daemon=True)
        
        # Start threads
        self.capture_thread.start()
        self.processor_thread.start()

        # Update timer
        self.timer = app.Timer(interval=1.0/FRAME_RATE, connect=self.on_timer, start=True)
        self.show()

    def capture_audio(self):
        try:
            speaker = sc.default_speaker()
            with sc.get_microphone(id=str(speaker.name), include_loopback=True).recorder(samplerate=self.samplerate) as mic:
                while self.running:
                    try:
                        data = mic.record(numframes=self.chunk_size)
                        self.audio_queue.put(data[:, 0], timeout=0.1)
                    except queue.Full:
                        continue
                    except Exception as e:
                        print(f"Error capturing audio: {e}")
                        time.sleep(0.1)
        except Exception as e:
            print(f"Error in capture thread: {e}")

    def process_audio(self):
        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
                
                # Apply window
                windowed = chunk * self.window
                
                # Compute FFT
                spectrum = np.fft.rfft(windowed)
                magnitude = np.abs(spectrum)
                
                # Apply smoothing
                if self.previous_spectrum is None:
                    self.previous_spectrum = magnitude
                else:
                    magnitude = (self.smoothing_factor * magnitude + 
                               (1 - self.smoothing_factor) * self.previous_spectrum)
                    self.previous_spectrum = magnitude
                
                # Convert to dB
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
                
                # Normalize
                magnitude_db = np.clip(magnitude_db, -self.dynamic_range, 0)
                row = (magnitude_db + self.dynamic_range) / self.dynamic_range
                
                # Apply distortion reduction
                row = np.clip(row * self.distortion_reduction, 0.0, 1.0)
                
                # Apply frequency smoothing
                if self.freq_smoothing_enabled:
                    row = np.convolve(row, self.freq_smoothing, mode='same')

                # Apply temporal smoothing using multiple frames
                self.temporal_smoothing_buffer[1:] = self.temporal_smoothing_buffer[:-1]
                self.temporal_smoothing_buffer[0] = row
                row = np.sum(self.temporal_smoothing_buffer * 
                           self.temporal_smoothing_weights[:, np.newaxis], axis=0)
                
                # Update texture data using double buffering
                self.texture_data_buffer[1:, :] = self.texture_data_buffer[:-1, :]
                self.texture_data_buffer[0, :] = row.astype(np.float32)
                
                # Apply temporal smoothing
                self.texture_data = (self.smoothing_factor * self.texture_data_buffer + 
                                   (1 - self.smoothing_factor) * self.texture_data)
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
                continue

    def on_timer(self, event):
        try:
            self.texture.set_data(self.texture_data)
            self.update()
        except Exception as e:
            print(f"Error in timer event: {e}")

    def on_draw(self, event):
        try:
            gloo.clear(color=(0.0, 0.0, 0.0, 1.0))
            self.program.draw('triangle_strip')
        except Exception as e:
            print(f"Error in draw event: {e}")

    def on_resize(self, event):
        try:
            gloo.set_viewport(0, 0, *event.physical_size)
        except Exception as e:
            print(f"Error in resize event: {e}")

    def close(self):
        self.running = False
        time.sleep(0.2)
        super().close()

def main():
    try:
        vis = AudioVisualizer()
        app.run()
    except Exception as e:
        print(f"Error running visualizer: {e}")
        raise

if __name__ == '__main__':
    main()
