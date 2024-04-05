# generic imports
import numpy as np
import pandas as pd
from datetime import datetime

# sklean imports
from sklearn.model_selection import train_test_split

# torch imports
import torch

# PyTorch Geometric
import torch_geometric

# pytorch lightning imports
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# framework imports
from models import HomoGNN
from data_handling import HiggsPyGDataset


def create_dataloader(df, cfg):
    # Shuffle the data
    df = df.sample(frac=1.0)

    # Split the array into 70% training and 30% temporary (validation + test) sets
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=cfg.seed
    )

    # Further split the 30% temporary set into 10% validation and 20% testing sets
    # To get 10% and 20% of the total data, the split ratio within the temporary set should be 1/3 (10% out of 30%)
    val_df, test_df = train_test_split(
        temp_df, test_size=2 / 3, stratify=temp_df["label"], random_state=cfg.seed
    )

    train_dataset = HiggsPyGDataset(indata=train_df, shuffle=True)
    val_dataset = HiggsPyGDataset(indata=val_df, shuffle=True)
    test_dataset = HiggsPyGDataset(indata=test_df, shuffle=True)

    # create dataloaders
    train_loader = torch_geometric.loader.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    val_loader = torch_geometric.loader.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    test_loader = torch_geometric.loader.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


class Config:
    pass


config = Config()
config.data_path = "../Files/"
config.classes = "CPOdd"
config.seed = 42
config.batch_size = 128
config.learning_rate = 0.0001
config.num_workers = 16
config.max_epochs = 1
config.patience = 3
config.accelerator = "gpu"
config.device_id = [0]

# MLP model
config.mlp_feature_dim = 7
config.mlp_hidden_dim = 16
config.mlp_dropout = 0.1


def main():
    # Seed everything
    L.seed_everything(config.seed)

    torch.set_float32_matmul_precision("medium")

    start = datetime.now()
    df_bkg = pd.read_csv(config.data_path + "graphs_qcd.csv")
    df_sig = None

    if config.classes == "CPOdd":
        df_sig = pd.read_csv(config.data_path + "graphs_cpodd.csv")
    elif config.classes == "CPeven":
        df_sig = pd.read_csv(config.data_path + "graphs_cpeven.csv")

    df_bkg["label"] = np.zeros(df_bkg.shape[0])
    df_sig["label"] = np.ones(df_sig.shape[0])
    df = pd.concat([df_bkg, df_sig], ignore_index=True)

    dataloader = create_dataloader(df=df, cfg=config)

    model = HomoGNN(
        in_feat=config.mlp_feature_dim,
        h_feat=config.mlp_hidden_dim,
        dropout=config.mlp_dropout,
        learning_rate=config.learning_rate,
        num_classes=len(config.classes),
        batch_size=config.batch_size,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        min_delta=0.00,
        patience=config.patience,
        verbose=False,
        mode="max",
    )

    print("Training from scratch ... ")
    trainer = L.Trainer(
        callbacks=[
            ModelCheckpoint(
                save_weights_only=False,
                mode="max",
                monitor="val_acc",
                every_n_train_steps=0,
                every_n_epochs=1,
                train_time_interval=None,
            ),
            early_stop_callback,
        ],
        devices=config.device_id,
        accelerator=config.accelerator,
        max_epochs=config.max_epochs,
    )

    trainer.fit(
        model=model,
        train_dataloaders=dataloader["train"],
        val_dataloaders=dataloader["val"],
    )
    test_result = trainer.test(model, dataloaders=dataloader["test"], verbose=False)
    result = {"acc": test_result[0]["test_acc"], "auc": test_result[0]["test_auc"]}

    end = datetime.now()
    print("Test performance:  %4.2f%%" % (100.0 * result["acc"]))
    print("Test ROC AUC: {}".format((100.0 * result["auc"])))
    print("Train completed in: {}".format(end - start))


if __name__ == "__main__":
    main()
