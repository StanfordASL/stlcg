STLCG
======

A toolbox to compute the robustness of STL formulas using computations graphs (PyTorch).

The `wafr20` branch reproduces the results from our [WAFR2020 paper](http://asl.stanford.edu/wp-content/papercite-data/pdf/Leung.Arechiga.Pavone.WAFR20.pdf). Note that the stlcg code in the `master` and `dev` branches have been updated since. So the `wafr20` branch works in isolation. 

## Installation
This was done on a linux machine running Python 3.5. You may need to change some of the packages in requirements to make it work on your setup. If you are running Python3.6 or higher, you may need to remove `importlib==1.0.4`.

First create a virtual environment
```
python3 -m venv stlcg-env
source stlcg-env/bin/activate
pip3 install -r requirements.txt
```
Then, since this project uses IPython notebooks, we'll install this conda environment as a kernel.
```
python3 -m ipykernel install --user --name stlcg-env --display-name "Python 3.X (stlcg)"
```
where X is the python3 version number.

The vizualization code here is constructed from https://github.com/szagoruyko/pytorchviz but with modifications to represent STL operators.


## Usage

`demo.ipynb` is an IPython jupyter notebook that showcases the basic functionality of the toolbox.

## Publications
Here are a list of publications that use stlcg. Please file an issue, or pull request to add your publication to the list.

[K. Leung, N. Aréchiga, and M. Pavone, "Back-propagation through STL specifications: Infusing logical structure into gradient-based methods," in Workshop on Algorithmic Foundations of Robotics, Oulu, Finland, 2020.](https://arxiv.org/abs/2008.00097)

[J. DeCastro, K. Leung, N. Aréchiga, and M. Pavone, "Interpretable Policies from Formally-Specified Temporal Properties,"" in Proc. IEEE Int. Conf. on Intelligent Transportation Systems, Rhodes, Greece, 2020.](http://asl.stanford.edu/wp-content/papercite-data/pdf/DeCastro.Leung.ea.ITSC20.pdf)

[K. Leung, N. Arechiga, and M. Pavone, "Backpropagation for Parametric STL," in IEEE Intelligent Vehicles Symposium: Workshop on Unsupervised Learning for Automated Driving, Paris, France, 2019.](http://asl.stanford.edu/wp-content/papercite-data/pdf/Leung.Arechiga.ea.ULAD19.pdf)

When citing stlcg, please use the following Bibtex:
```
@Inproceedings{LeungArechigaEtAl2020,
  author       = {Leung, K. and Ar\'{e}chiga, N. and Pavone, M.},
  title        = {Back-propagation through signal temporal logic specifications: Infusing logical structure into gradient-based methods},
  booktitle    = {{Workshop on Algorithmic Foundations of Robotics}},
  year         = {2020},

}
```


## Feedback

If there are any issues with the code, please make file an issue, or make a pull request.

