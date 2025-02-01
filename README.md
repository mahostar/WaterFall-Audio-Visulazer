# Waterfall Visualizer
A Python-based tool for visualizing waterfalls, with room for enhancements.

### ** Project Description **
This project provides a visual representation of waterfall data, partly generated using AI. It is designed to be extensible and can be implimented in other softweres.


## if you want to improve this project ( I recomand creating a virtual envirment ):

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
