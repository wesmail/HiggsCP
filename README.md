# HiggsCP
Graph Neural Networks for CP
### How to use
#### Install packages  
1. [PyTorch](https://pytorch.org)
2. [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)  
3. [PyTorch Lightning](https://pypi.org/project/pytorch-lightning/)
4. [PyTorch Metrics](https://torchmetrics.readthedocs.io/en/stable/pages/quickstart.html)

#### How to run  
1. For Homogenous GNN `ipython create_graphs` -- -input_file=kinematics_lowlevel_ttA.csv (same for other two files) and move created files inside the `GNN` directory.  
2. run `ipython main.py -- -n=3` for multiclass classification or `ipython main.py -- -n=2` for binary classification.  
