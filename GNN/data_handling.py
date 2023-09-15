# generic imports
import numpy as np

# torch imports
import torch

# pyg
import torch_geometric.data as PyGData

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
        x = graph[['I1', 'I2', 'I3', 'pT', 'E', 'Eta', 'Phi']].to_numpy()
        scale = np.array([1., 1., 1., 100., 100., 1., 1.])
        x /= scale
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(np.unique(graph['label']).item(), dtype=torch.int64)
        g = PyGData.Data(x=x, y=y, edge_index=self.edge_index)
        
        return {"graphs": g, "labels": y}

    def __len__(self):
        return len(self.graphs)    