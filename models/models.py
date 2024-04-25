# torch imports
import torch
import torch.nn.functional as F

# PyTorch Geometric imports
import torch_geometric

# PyTorch lightning imports
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

# TorchMetrics
from torchmetrics import Accuracy, AUROC

ACTIVATION = "softmax"


class HeteroGCN(torch.nn.Module):
    def __init__(self, h_feat=16, num_classes=2):
        super(HeteroGCN, self).__init__()
        self.conv1 = torch_geometric.nn.GraphConv((-1, -1), h_feat)
        self.conv2 = torch_geometric.nn.GraphConv((-1, -1), h_feat)
        self.conv3 = torch_geometric.nn.GraphConv((-1, -1), h_feat)

        self.pool = torch_geometric.nn.SumAggregation()
        self.clf = torch.nn.Linear(h_feat, num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = F.leaky_relu(h)
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h)
        h = self.conv3(h, edge_index)

        h = F.dropout(h, p=0.5, training=self.training)
        h = self.pool(h, batch)
        h = self.clf(h)

        return h


class HeteroTConv(LightningModule):
    def __init__(self, h_feat, num_classes, n_heads, dropout) -> None:
        super(HeteroTConv, self).__init__()
        self.conv1 = torch_geometric.nn.TransformerConv(-1, h_feat, heads=n_heads)
        self.batch_norm1 = torch.nn.BatchNorm1d(h_feat * n_heads)
        self.conv2 = torch_geometric.nn.TransformerConv(-1, h_feat, heads=n_heads)
        self.batch_norm2 = torch.nn.BatchNorm1d(h_feat * n_heads)
        self.conv3 = torch_geometric.nn.TransformerConv(-1, h_feat, heads=n_heads)
        self.batch_norm3 = torch.nn.BatchNorm1d(h_feat * n_heads)

        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.PReLU()

        self.pool = torch_geometric.nn.SumAggregation()
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(h_feat * n_heads, h_feat),
            torch.nn.BatchNorm1d(h_feat),
            torch.nn.PReLU(),
            torch.nn.Linear(h_feat, num_classes),
        )

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.batch_norm1(h)
        h = self.activation(h)
        h = self.conv2(h, edge_index)
        h = self.batch_norm2(h)
        h = self.activation(h)

        h = self.conv3(h, edge_index)
        h = self.batch_norm3(h)
        h = self.activation(h)

        h = self.dropout(h)
        h = self.pool(h, batch)
        h = self.clf(h)

        return h


class HeteroGNN(LightningModule):
    def __init__(
        self,
        h_feat=32,
        num_classes=2,
        n_heads=8,
        dropout=0.1,
        batch_size=32,
        learning_rate=0.0001,
    ) -> None:
        super().__init__()
        self.h_feat = h_feat
        self.n_heads = n_heads
        self.dropout = dropout
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = learning_rate

        self.metadata = (
            ["pion", "tau", "higgs"],
            [
                ("higgs", "decays_to", "tau"),
                ("tau", "decays_to", "pion"),
                ("tau", "connects_to", "tau"),
                ("tau", "rev_decays_to", "higgs"),
                ("pion", "rev_decays_to", "tau"),
            ],
        )

        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = HeteroTConv(
            h_feat=self.h_feat,
            num_classes=self.num_classes,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )
        self.model = torch_geometric.nn.to_hetero(
            self.model, metadata=self.metadata, aggr="sum"
        )
        # initialize metric
        self.train_metric = Accuracy(task="binary")
        self.val_metric = Accuracy(task="binary")
        self.test_metric = Accuracy(task="binary")
        self.test_auc = AUROC(task="binary")
        if ACTIVATION == "softmax":
            self.train_metric = Accuracy(
                task="multiclass", num_classes=self.num_classes
            )
            self.val_metric = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.test_metric = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.test_auc = AUROC(task="multiclass", num_classes=self.num_classes)

        # for predictions
        self.test_predictions = []
        self.test_targets = []
        self.test_theta = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.001
        )
        return optimizer

    def training_step(self, data, batch_idx) -> STEP_OUTPUT:
        hetero, labels = data[0], data[1]
        if ACTIVATION == "sigmoid":
            labels = labels.float()
        linear_out = self.model(
            hetero.x_dict, hetero.edge_index_dict, hetero.batch_dict
        )
        pred, loss = None, None
        if ACTIVATION == "softmax":
            pred = torch.nn.functional.softmax(linear_out, dim=1)
            loss = torch.nn.functional.cross_entropy(pred, labels)
        elif ACTIVATION == "sigmoid":
            pred = torch.nn.functional.sigmoid(linear_out).flatten()
            loss = torch.nn.functional.binary_cross_entropy(pred, labels)

        self.log("train_loss", loss, batch_size=self.batch_size, prog_bar=True)
        self.log(
            "train_acc",
            self.train_metric(pred, labels),
            batch_size=self.batch_size,
            prog_bar=True,
        )
        return loss

    def validation_step(self, data, batch_idx) -> STEP_OUTPUT:
        hetero, labels = data[0], data[1]
        if ACTIVATION == "sigmoid":
            labels = labels.float()
        linear_out = self.model(
            hetero.x_dict, hetero.edge_index_dict, hetero.batch_dict
        )

        val_pred, val_loss = None, None
        if ACTIVATION == "softmax":
            val_pred = torch.nn.functional.softmax(linear_out, dim=1)
            val_loss = torch.nn.functional.cross_entropy(val_pred, labels)
        elif ACTIVATION == "sigmoid":
            val_pred = torch.nn.functional.sigmoid(linear_out).flatten()
            val_loss = torch.nn.functional.binary_cross_entropy(val_pred, labels)

        self.log(
            "val_loss",
            val_loss,
            sync_dist=True,
            batch_size=self.batch_size,
            prog_bar=True,
        )
        self.log(
            "val_acc",
            self.val_metric(val_pred, labels),
            sync_dist=True,
            batch_size=self.batch_size,
            prog_bar=True,
        )

    def test_step(self, data, batch_idx) -> STEP_OUTPUT:
        hetero, labels = data[0], data[1]
        if ACTIVATION == "sigmoid":
            labels = labels.float()
        linear_out = self.model(
            hetero.x_dict, hetero.edge_index_dict, hetero.batch_dict
        )
        test_pred = None
        if ACTIVATION == "softmax":
            test_pred = torch.nn.functional.softmax(linear_out, dim=1)
        elif ACTIVATION == "sigmoid":
            test_pred = torch.nn.functional.sigmoid(linear_out).flatten()

        self.log(
            "test_acc",
            self.test_metric(test_pred, labels),
            batch_size=self.batch_size,
            prog_bar=True,
        )
        self.log(
            "test_auc",
            self.test_auc(test_pred, labels),
            batch_size=self.batch_size,
            prog_bar=True,
        )

        # for predictions
        self.test_predictions.extend(test_pred.detach().cpu().numpy())
        self.test_targets.extend(labels.cpu().numpy())
        self.test_theta.extend(
            list(hetero.edge_attr_dict.values())[0].detach().cpu().numpy()
        )

    def predict_step(self, hetero) -> STEP_OUTPUT:
        return self.model(hetero.x_dict, hetero.edge_index_dict, hetero.batch_dict)
