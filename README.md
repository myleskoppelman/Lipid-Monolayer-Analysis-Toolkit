# Domain Analysis Programs

**Author:** Myles Koppelman  
**Date:** 08/08/2025  

These programs use **Python 3.12** and several libraries that may conflict with locally installed versions on your machine.  
To avoid issues, it is recommended to use **Anaconda**, which allows you to manage environments and library versions without affecting your system installation.

---

## Quick Start (One Command)

### macOS / Linux
Open a terminal and run (you can change 'domain-env' to any name you'd like for your virtual environment):

```bash
conda create -n domain-env python=3.12
```
```bash
conda activate domain-env
```
```bash
conda install -y \ numpy=1.26.4 \ pandas \ matplotlib \ scikit-image \ tifffile \ openpyxl \ imageio \ pyarrow \ scipy \ bottleneck \ tqdm \ pillow \ ipykernel \ seaborn
```
```bash
pip install easygui==0.98.3 matplotlib==3.10.5
```




### Windows
Open Anaconda Prompt (not Command Prompt) and run:

```bash
conda create -n domain-env python=3.12
```
```bash
conda activate domain-env
```
```bash
conda install -y numpy=1.26.4 pandas openpyxl tifffile scikit-image matplotlib tqdm scipy pillow ipykernel seaborn
```
```bash
pip install easygui==0.98.3 matplotlib==3.10.5
```





Once the environment is set up, you can run any of the provided programs.

If you encounter a missing package error, simply install it with:

```bash
conda install [package-name]
```

If you run into problems, copy and paste your error message into ChatGPT for troubleshooting.

### Exit the Environment
When you are finished, deactivate the environment with:
```bash
conda deactivate
```

To see existing virtual environments (to get back to your existing environment), run:
```bash
conda env list
```
