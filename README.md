STLCG
======

A toolbox to compute the robustness of STL formulas using computations graphs (PyTorch).

## Installation
This was done on a linux machine running Python 3.5. You may need to change some of the packages in requirements to make it work on your setup. If you are running Python3.6, you may need to remove `importlib==1.0.4`.

First create a virtual environment
```
python3 -m venv stlcg-env
source stlcg-env/bin/activate
pip3 install -r requirements.txt
```
Then, since this project uses IPython notebooks, we'll install this conda environment as a kernel.
```
python3 -m ipykernel install --user --name stlcg-env --display-name "Python 3.5 (stlcg)"
```

The vizualization code here is constructed from https://github.com/szagoruyko/pytorchviz but with modifications to represent STL operators.


## Usage

`stlcg demo new.ipynb` is an IPython jupyter notebook that showcases the basic functionality of the toolbox.

