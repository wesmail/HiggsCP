# torch imports
import torch

# torch_geometric imports
import torch_geometric

class EdgeConv(torch.nn.Module):
    def __init__(self, in_feat=7, h_feat=8, num_classes=2, dropout=0.5):
        super().__init__()

        self.gnn = torch_geometric.nn.EdgeCNN(in_feat, h_feat, 3, dropout=dropout, jk='cat', norm="BatchNorm")
        self.classifier = torch_geometric.nn.MLP([h_feat, h_feat*2, num_classes], norm="batch_norm", dropout=dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gnn(x=x, edge_index=edge_index)
        x = torch_geometric.nn.global_add_pool(x, batch)
        x = self.classifier(x)
        return x
      
class GCN(torch.nn.Module):
    def __init__(self, in_feat=7, h_feat=8, num_classes=3):
        super().__init__()

        self.conv1 = torch_geometric.nn.GraphConv(in_feat, h_feat)
        self.conv2 = torch_geometric.nn.GraphConv(h_feat, h_feat)
        self.conv3 = torch_geometric.nn.GraphConv(h_feat, h_feat)
        self.lin = torch.nn.Linear(h_feat, num_classes)

        # Optional: Activation and BatchNorm
        self.act = torch.nn.LeakyReLU()
        self.batch_norm = torch.nn.BatchNorm1d(h_feat)        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.conv1(x=x, edge_index=edge_index)
        h = self.act(h)  # Activation function
        #h = self.batch_norm(h)  # Batch normalization        
        h = self.conv2(x=h, edge_index=edge_index)
        h = self.act(h)  # Activation function
        #h = self.batch_norm(h)  # Batch normalization        
        h = self.conv3(x=h, edge_index=edge_index)
        #h = self.act(h)  # Activation function
        #h = self.batch_norm(h)  # Batch normalization

        out = torch_geometric.nn.global_mean_pool(h, batch)
        out = self.lin(out)
        return out