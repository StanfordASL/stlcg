STLCG
======

A toolbox to compute the robustness of STL formulas using computation graphs. Since STLCG uses PyTorch, it provides a very easy way to incorporate STL formulas into neural network models (e.g., adding an STL term in the training loss).

## Installation

You need Python3 and PyTorch installed.

The vizualization code here is constructed from https://github.com/szagoruyko/pytorchviz but with modifications to represent STL operators.


## Usage

`demo.ipynb` is an IPython jupyter notebook that showcases the basic functionality of the toolbox.

The `examples` folder contains example usage of STLCG in a number of applications. These are the examples investigated in the WAFR 2020 publication (see below).

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

