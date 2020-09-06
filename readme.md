# Bokeh Server - Complimentary to Miaas

### 1.0 - Prerequisites
1. [Python v3.7.5](https://www.python.org/downloads/release/python-375/)
2. [Git](https://git-scm.com/downloads)

### 1.1 - Clone this repo
1. Create a folder of where this project's folder will be downloaded
2. On Windows, press `Shift` and then `Right click` on an empty area
3. Select "Open Terminal Here"
4. Run the command `git clone HDD`

### 1.2 - Install the virtual environment library
```bash
pip install virtualenv 
```

### 1.3 - Create The Virtual Environment
```bash
python -m venv venv
```

### 1.4 - Run The Virtual Environment
```bash
source venv/bin/activate # Linux
.\venv\Scripts\activate # Windows
```

### 1.5 - Download Dependencies (After Activating the virtual environment)
```bash
pip install -r requirements.txt
```

### 2.0 - Run The Server
Done after the virtual environment is activated (Step #1.4)
```bash
python bokeh-server.py
```