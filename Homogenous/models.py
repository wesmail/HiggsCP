# torch imports
import torch

# torch_geometric imports
import torch_geometric

# PyTorch Lightning imports
from lightning.pytorch import LightningModule

# torchmetrics
from torchmetrics import Accuracy, AUROC


class BackBoneGNN(torch.nn.Module):
    def __init__(self, in_feat=7, h_feat=8, num_classes=2, dropout=0.5):
        super().__init__()

        self.gnn = torch_geometric.nn.GAT(
            in_channels=in_feat,
            hidden_channels=h_feat,
            out_channels=h_feat,
            num_layers=3,
            dropout=dropout,
            jk="cat",
            norm="BatchNorm",
            act=torch.nn.PReLU(),
        )
        self.classifier = torch_geometric.nn.MLP(
            [h_feat, h_feat, num_classes],
            norm="batch_norm",
            dropout=dropout,
            act=torch.nn.PReLU(),
        )

    def forward(self, graph: torch.Tensor) -> torch.Tensor:
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.gnn(x=x, edge_index=edge_index)
        x = torch_geometric.nn.global_add_pool(x, batch)

        return self.classifier(x)


class HomoGNN(LightningModule):
    def __init__(
        self,
        in_feat: int = 7,
        h_feat: int = 32,
        dropout: float = 0.1,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        num_classes: int = 2,
    ):
        super().__init__()

        self.in_feat = in_feat
        self.h_feat = h_feat
        self.dropout = dropout
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = learning_rate
        # Saving hyperparameters
        self.save_hyperparameters()

        # initialize metric
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_auc = AUROC(task="multiclass", num_classes=num_classes)

        self.model = BackBoneGNN(
            in_feat=in_feat, h_feat=h_feat, num_classes=num_classes, dropout=dropout
        )

    def training_step(self, graph, batch_idx):
        graphs, labels = graph[0], graph[1]
        linear_out = self.model(graphs)
        pred = torch.nn.functional.softmax(linear_out, dim=1)
        loss = torch.nn.functional.cross_entropy(pred, labels)
        self.log("train_loss", loss, batch_size=self.batch_size, prog_bar=True)
        self.log(
            "train_acc",
            self.acc(pred, labels),
            batch_size=self.batch_size,
            prog_bar=True,
        )
        return loss

    def validation_step(self, graph, batch_idx):
        graphs, labels = graph[0], graph[1]
        linear_out = self.model(graphs)
        pred = torch.nn.functional.softmax(linear_out, dim=1)
        val_loss = torch.nn.functional.cross_entropy(pred, labels)
        self.log(
            "val_loss",
            val_loss,
            sync_dist=True,
            batch_size=self.batch_size,
            prog_bar=True,
        )
        self.log(
            "val_acc",
            self.acc(pred, labels),
            sync_dist=True,
            batch_size=self.batch_size,
            prog_bar=True,
        )

    def test_step(self, graph, batch_idx):
        graphs, labels = graph[0], graph[1]
        linear_out = self.model(graphs)
        pred = torch.nn.functional.softmax(linear_out, dim=1)
        self.log(
            "test_acc",
            self.acc(pred, labels),
            batch_size=self.batch_size,
            prog_bar=True,
        )
        self.log(
            "test_auc",
            self.test_auc(pred, labels),
            batch_size=self.batch_size,
            prog_bar=True,
        )

    def predict_step(self, graph):
        return self.model(graph)

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0)
        return optimizer
