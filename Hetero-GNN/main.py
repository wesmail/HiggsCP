# generic imports
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# torch imports
import torch
import torch_geometric

# pytorch lightning imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics

# framework imports
from data_handling import HiggsPyGDataset
from models import GCN


class GraphLevelGNN(pl.LightningModule):
    def __init__(self, module, batch_size, num_classes=2):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.model = module()
        # initialize metric
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_auc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=0.0)
        return optimizer

    def training_step(self, data, batch_idx):
        graphs, labels = data["graphs"], data["labels"]
        linear_out = self.model(graphs)
        pred = torch.nn.functional.softmax(linear_out, dim=1)
        loss = torch.nn.functional.cross_entropy(pred, labels)
        self.log("train_loss", loss, batch_size=self.batch_size, prog_bar=True)
        self.log("train_acc", self.train_metric(pred, labels),
                 batch_size=self.batch_size, prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx):
        graphs, labels = data["graphs"], data["labels"]
        linear_out = self.model(graphs)
        pred = torch.nn.functional.softmax(linear_out, dim=1)
        val_loss = torch.nn.functional.cross_entropy(pred, labels)
        self.log("val_loss", val_loss, sync_dist=True,
                 batch_size=self.batch_size, prog_bar=True)
        self.log("val_acc", self.val_metric(pred, labels), sync_dist=True,
                 batch_size=self.batch_size, prog_bar=True)

    def test_step(self, data, batch_idx):
        graphs, labels = data["graphs"], data["labels"]
        linear_out = self.model(graphs)
        pred = torch.nn.functional.softmax(linear_out, dim=1)
        self.log("test_acc", self.test_metric(pred, labels),
                 batch_size=self.batch_size, prog_bar=True)
        self.log("test_auc", self.test_auc(pred, labels),
                 batch_size=self.batch_size, prog_bar=True)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=False, help='binary or multiclass classification', default=3)
    args = parser.parse_args()       
    # some variables
    BATCH_SIZE = 256
    N_WORKERS = 8
    MAX_EPOCHS = 5
    DEVICE_ID = [0]
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

    torch.set_float32_matmul_precision('medium')

    # Randomly split the dataset into training, validation and test sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])    

    # create dataloaders
    graph_train_loader = torch_geometric.loader.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS, drop_last=True, pin_memory=True)   
    graph_val_loader = torch_geometric.loader.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, drop_last=True, pin_memory=True)
    graph_test_loader = torch_geometric.loader.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, drop_last=True)
    
    model = GraphLevelGNN(module=MODEL, batch_size=BATCH_SIZE)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    ) 

    print("Training from scratch ... ")
    trainer = pl.Trainer(callbacks=[ModelCheckpoint(save_weights_only=False, mode="max", monitor="val_acc",
                                                    every_n_train_steps=0, every_n_epochs=1, train_time_interval=None), 
                                                    early_stop_callback],
                                                    devices=DEVICE_ID, accelerator="gpu", max_epochs=MAX_EPOCHS)
    

    trainer.fit(model=model, train_dataloaders=graph_train_loader, val_dataloaders=graph_val_loader)
    test_result = trainer.test(model, dataloaders=graph_test_loader, verbose=False)
    result = {"acc": test_result[0]["test_acc"], "auc": test_result[0]["test_auc"]}    
    
    end = datetime.now()
    print("Test performance:  %4.2f%%" % (100.0 * result["acc"]))
    print("Test ROC AUC: {}".format((100.0 * result["auc"])))
    print("Train completed in: {}".format(end-start))

if __name__ == "__main__":
    main(sys.argv[1:])