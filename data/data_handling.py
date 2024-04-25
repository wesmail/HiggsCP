# Generic imports
import h5py

# PyTorch imports
import torch
from torch.utils.data import random_split

# PyTorch Geometric imports
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

# PyTorch Lightning imports
from lightning.pytorch import LightningDataModule


class HiggsHeteroDataModule(LightningDataModule):
    def __init__(
        self,
        h5_file: str,
        seed: int,
        batch_size: int = 32,
        num_workers: int = 16,
    ) -> None:
        super().__init__()

        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        # save the parameters
        self.save_hyperparameters()

        self.h5_file = h5py.File(h5_file, mode="r")
        self.data = self.h5_file["data"]
        self.labels = self.h5_file["labels"]

    def __getitem__(self, item) -> tuple:
        # 0,1,2 pions indices from tau1 (index 6) decay
        # 3,4,5 pions indices fram tau2 (index 7) decay
        # higgs index is 8, higgs decays to tau (idnex 6), tau (index 7)
        data, y = self.data[item], self.labels[item]
        # Initialize a HeteroData object
        G = torch_geometric.data.HeteroData()

        G["pion"].x = torch.tensor(data[0:6], dtype=torch.float32)
        G["tau"].x = torch.tensor(data[[6, 7]][:, :5], dtype=torch.float32)

        higgs_feat = data[8:9, 0:6]  # data[8:9, 0:7]
        # higgs_feat[:, -1] = data[-1, 0]
        G["higgs"].x = torch.tensor(higgs_feat, dtype=torch.float32)

        # Graph convectivity:
        # Higgs to taus edges
        # Here Higgs at index 0 (its own scope) decays to the two taus at indices 0 and 1 (in 'tau' scope)
        G["higgs", "decays_to", "tau"].edge_index = torch.tensor(
            [[0, 0], [0, 1]], dtype=torch.long
        )

        # Tau to pions edges
        # Edges for tau 0 to its pions
        G["tau", "decays_to", "pion"].edge_index = torch.tensor(
            [[0, 0, 0], [0, 1, 2]], dtype=torch.long
        )
        # Edges for tau 1 to its pions
        G["tau", "decays_to", "pion"].edge_index = torch.cat(
            [
                G["tau", "decays_to", "pion"].edge_index,
                torch.tensor([[1, 1, 1], [3, 4, 5]], dtype=torch.long),
            ],
            dim=1,
        )

        # Tau2Tau edge and edge feature
        G["tau", "connects_to", "tau"].edge_index = torch.tensor(
            [[0], [1]], dtype=torch.long
        )

        G["tau", "connects_to", "tau"].edge_attr = torch.tensor(
            data[-1, 0], dtype=torch.float32
        ).view(1, 1)

        G = T.AddSelfLoops()(G)
        G = T.ToUndirected()(G)

        return (G, torch.tensor(y, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.labels)

    def setup(self, stage: str) -> None:
        dataset = self

        # TODO: remove hardcoded numbers: fractions
        self.train, self.val, self.test = random_split(
            dataset, [0.7, 0.1, 0.2], generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=False,
        )
