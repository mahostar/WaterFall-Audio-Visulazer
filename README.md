# Waterfall Visualizer
A Python-based tool for visualizing audio, with room for enhancements.

### Project Description 
This project provides a visual representation of output device audio, the source code partly generated using AI. It is designed to be extensible and can be implimented in other softweres.

## **Demo Video 🎥**
[![Watch the demo video](https://img.youtube.com/vi/dePnt0WHOk8/0.jpg)](https://youtube.com/shorts/dePnt0WHOk8?feature=share)


## if you want to improve this project (I recomand creating a virtual envirment): 

**Creating a python virtual envirment :**
```bash
python -m venv WFV
```

**activating on windows :**
```bash
WFV\Scripts\activate
```

**activating on mac/Linux :**
```bash
source WFV/bin/activate
```

**installing requarments :**
```bash
pip install -r requirements.txt
```


## if you want the build the app : 

```bash
pip install pyinstaller
```

```bash
pyinstaller --onefile --windowed --hidden-import=vispy.app.backends._pyqt5 --icon=image.ico WaterFall_Visulazer.py
```
