# Torch imports
import torch

# PyTorch Lightning imports
from lightning.pytorch.cli import LightningCLI

# Framework imports
from models.models import HeteroGNN
from data.data_handling import HiggsHeteroDataModule


def cli_main():
    cli = LightningCLI(HeteroGNN, HiggsHeteroDataModule)


if __name__ == "__main__":
    # Uncomment if you don't have a GPU with tensor cores
    torch.set_float32_matmul_precision("medium")
    cli_main()
