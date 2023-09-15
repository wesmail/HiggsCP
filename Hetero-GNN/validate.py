# generic imports
import sys
import argparse
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm.auto import tqdm
from collections import namedtuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, confusion_matrix
from torchmetrics import ROC

# torchmetrics
from torchmetrics.classification import ConfusionMatrix, BinaryConfusionMatrix

# torch imports
import torch

# pyg imports
import torch_geometric

# pytorch lightning imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

# framework imports
from data_handling import HiggsPyGDataset
from models import GCN

class GraphLevelGNN(pl.LightningModule):
    def __init__(self, module, batch_size):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.model = module()

    def test_step(self, data, batch_idx):
        graphs, labels = data["graphs"].cuda(), data["labels"]
        linear_out = self.model(graphs)
        
        return linear_out
    
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=False, help='binary or multiclass classification', default=3)
    args = parser.parse_args()     
    # some variables
    BATCH_SIZE = 256
    N_WORKERS = 16
    ACTIVATION = "softmax"
    CHECKPOINT = 'lightning_logs/version_0/checkpoints/epoch=4-step=2395.ckpt'
    MODEL = GCN
    num_classes = args.n

    start = datetime.now()
    background = pd.read_csv("../kinematics_polarisation_ttbb.csv")
    cp_even = pd.read_csv("../kinematics_polarisation_ttH.csv")
    cp_odd = pd.read_csv("../kinematics_polarisation_ttA.csv")
    # assign labels
    background = pd.read_csv("../kinematics_polarisation_ttbb.csv")
    cp_even = pd.read_csv("../kinematics_polarisation_ttH.csv")
    cp_odd = pd.read_csv("../kinematics_polarisation_ttA.csv")
    
    # assign labels
    if num_classes == 3:
        background['label'] = np.zeros(background.shape[0])
        cp_even['label'] = np.ones(cp_even.shape[0])
        cp_odd['label'] = 2*np.ones(cp_odd.shape[0])

        background = background.sample(n=9531)
        cp_even = cp_even.sample(n=9531)
        cp_odd = cp_odd.sample(n=9531)        
    elif num_classes == 2:
        cp_even['label'] = np.zeros(cp_even.shape[0])
        cp_odd['label'] = np.ones(cp_odd.shape[0])     

    df = None
    # combine both datasets
    if num_classes == 3:
        df = pd.concat([background, cp_even, cp_odd], ignore_index=True)
    elif num_classes == 2:
        df = pd.concat([cp_even, cp_odd], ignore_index=True)

    dataset = HiggsPyGDataset(indata=df, shuffle=True)
    
    # randomly split the data into (train/val/test) (60/10/30)
    # Set the random seed
    torch.manual_seed(42)     
    frac_list=[0.7, 0.1, 0.2]
    dataset_size = len(dataset)
    train_size = int(dataset_size * frac_list[0])
    valid_size = int(dataset_size * frac_list[1])
    test_size = dataset_size - train_size - valid_size

    # Randomly split the dataset into training, validation and test sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])    

    # create dataloaders
    graph_test_loader = torch_geometric.loader.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, drop_last=True)    

    # seed for everything
    pl.seed_everything(42)

    model = GraphLevelGNN(module=MODEL, batch_size=BATCH_SIZE)
    model = model.load_from_checkpoint(CHECKPOINT)
    print(model)

    start = datetime.now()
    
    predictions, targets = [], []

    i = 0
    for data in tqdm(graph_test_loader):
        #if i > 20: break
        linear_out = model.test_step(data, None)
        yhat = None
        if ACTIVATION == "softmax":
            yhat       = torch.nn.functional.softmax(linear_out, dim=1).detach().cpu().numpy()
        elif ACTIVATION == "sigmoid":
            yhat       = torch.nn.functional.sigmoid(linear_out).flatten().detach().cpu().numpy()

        targets.extend(data["labels"].cpu().flatten().numpy())
        predictions.extend(yhat)

        i += 1

    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    # --------------------------------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------------------------------
    if ACTIVATION == "softmax":
        predictions = np.argmax(predictions, axis=1)
        print(predictions)
    
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize="pred")
    cm = confmat(torch.tensor(predictions), torch.tensor(targets)).numpy()
    #cm = confusion_matrix(targets, np.round(predictions))
    #cm = np.round((cm.astype('float') / cm.sum(axis=1)), decimals=3)
    figcm, ax = plt.subplots(figsize=(7,5))

    #plt.figure(figsize=(7,5))
    cm = pd.DataFrame(cm)
    #cm.columns =['Signal(true)', 'Background(true)']
    #cm.index =['Signal(pred)', 'Background(pred)']
    sns.heatmap(cm, annot=True, fmt='0.2f', square=True, annot_kws={"size": 20}, linewidth=5, cmap=sns.cubehelix_palette(as_cmap=True)); 
    sns.set(font_scale=1.5)
    #sns.set(font_scale=2.0)
    #sns.heatmap(cm, square=True, annot=True, annot_kws={"size": 22}, cmap='Blues')
    #classes=['$\pi$','   $K$','   $p$']
    #tick_marks = np.arange(len(classes))
    if num_classes == 2:
        plt.xticks(np.arange(num_classes)+0.5, ["ttA(true)", "ttH(true)"], rotation=0., fontsize=15, va="center")
        plt.yticks(np.arange(num_classes)+0.5, ["ttA(pred)", "ttH(pred)"], rotation=90., fontsize=15, va="center")
    elif num_classes == 3:
        plt.xticks(np.arange(num_classes)+0.5, ["ttbb(true)", "ttH(true)", "ttA(true)"], rotation=0., fontsize=15, va="center")
        plt.yticks(np.arange(num_classes)+0.5, ["ttbb(pred)", "ttH(pred)", "ttA(pred)"], rotation=90., fontsize=15, va="center")                  
    #plt.xticks(np.arange(n_classes)+0.5, ["Signal(true)", "Background(true)"], rotation=0., fontsize=15, va="center")
    #plt.yticks(np.arange(n_classes)+0.5, ["Signal(pred)", "Background(pred)"], rotation=90., fontsize=15, va="center")
    ax.set_xticks(np.arange(num_classes), minor=True)
    ax.set_yticks(np.arange(num_classes), minor=True)
    #plt.xlabel('', horizontalalignment="center",  fontsize=22,)
    #plt.ylabel('True Species',  fontsize=22)

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------------------------------     

    end = datetime.now()
    print("Train completed in: {}".format(end-start))

if __name__ == "__main__":
    main(sys.argv[1:])   