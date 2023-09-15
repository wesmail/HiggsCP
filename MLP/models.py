# torch imports
import torch

# torch_geometric imports
import torch_geometric

class SimpleMLP(torch.nn.Module):
    def __init__(self, in_feat=35, h_feat=8, num_classes=3, dropout=0.0):
        super().__init__()
        self.mlp = torch_geometric.nn.MLP([in_feat, h_feat, h_feat, num_classes], \
                                          norm="batch_norm", dropout=dropout, act='leaky_relu')

    def forward(self, x):
        return self.mlp(x=x)