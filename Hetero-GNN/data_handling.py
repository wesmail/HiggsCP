# generic imports
import numpy as np
import pandas as pd
import scipy
import sklearn.preprocessing as sklp

# torch imports
import torch

# pyg
import torch_geometric
import torch_geometric.data as PyGData
from torch_cluster import knn_graph

class HiggsPyGDataset(torch.utils.data.Dataset):
    """
    PyTorch class to generate graph data
    """
    def __init__(self, indata, shuffle=False, start=0, end=-1):
        # read in the ascii file
        self.data = indata
        if shuffle:
            self.data = self.data.sample(frac=1.0)
        # split into events
        gb = self.data.groupby('event_id')
        dfs = [gb.get_group(x) for x in gb.groups]
        self.graphs = dfs[start:end]

    def load_edges(self, nodes):
        # complete graph (fully connected without self loop)
        edge_index = torch.ones(
            [nodes, nodes], dtype=torch.int32) - torch.eye(nodes, dtype=torch.int32)
        self.edge_index = edge_index.to_sparse()._indices()

    def __getitem__(self, item):
        graph = self.graphs[item]
        n_nodes = graph.shape[0]
        # edge index (use complete graph without edge weighting)
        self.load_edges(n_nodes)
        # node features
        x = graph[['I1', 'I2', 'I3', 'It', 'pT', 'E', 'Eta', 'Phi']].to_numpy()
        scale = np.array([1., 1., 1., 1., 100., 100., 1., 1.])
        x /= scale
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(np.unique(graph['label']).item(), dtype=torch.int64)

        # Create edge indices for a fully connected graph
        num_nodes = graph.shape[0]
        edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()
        
        # Initialize edge attributes with zeros
        edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.float)
        
        # Find the indices of the two leptons and two tops in this event
        lepton_indices = graph.index[graph['I1'] == 1].tolist()
        top_indices = graph.index[graph['It'] == 1].tolist()
        
        # Assuming the two leptons and two tops are the first four nodes in the graph
        lepton1_idx, lepton2_idx = lepton_indices
        top1_idx, top2_idx = top_indices
        
        # Find the edge index corresponding to the edge between the two leptons and two tops
        edge_idx_leptons = (edge_index[0] == lepton1_idx) & (edge_index[1] == lepton2_idx)
        edge_idx_tops = (edge_index[0] == top1_idx) & (edge_index[1] == top2_idx)
        
        # Set the edge attribute for these edges to the value of ll_angle and tt_angle
        edge_attr[edge_idx_leptons] = graph.loc[lepton1_idx, 'll_angle']
        edge_attr[edge_idx_tops] = graph.loc[top1_idx, 'tt_angle']
        
        # Create a PyTorch Geometric graph
        g = PyGData.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return {"graphs": g, "labels": y}

    def __len__(self):
        return len(self.graphs)    