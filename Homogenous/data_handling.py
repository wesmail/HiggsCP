# generic imports
import numpy as np

# torch imports
import torch

# PyTorch Geometric
import torch_geometric.data as PyGData


class HiggsPyGDataset(torch.utils.data.Dataset):
    """
    Initialize the HiggsPyGDataset class for Homogenous Graphs.

    Parameters
    ----------
    indata : pandas.DataFrame
        The input data.
    shuffle : bool, optional
        Whether to shuffle the data. The default is False.
    start : int, optional
        The index of the first event to use. The default is 0.
    end : int, optional
        The index of the last event to use. If -1, all events are used. The default is -1.
    """

    def __init__(self, indata, shuffle=False, num_features=7):
        # read in the ascii file
        self.data = indata
        self.num_features = num_features
        self.num_nodes = 8  # number of nodes in the "homogenous" graph
        if shuffle:
            self.data = self.data.sample(frac=1.0)

        self.scale = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 1.0, 1.0])

    def load_edges(self, nodes):
        """
        This function creates a complete undirected graph with `nodes` nodes.
        The adjacency matrix is represented as a sparse tensor with shape `[nodes, nodes]`.
        The diagonal elements are set to zero and the off-diagonal elements are set to 1.
        The function returns the indices of the non-zero elements of the sparse tensor.

        Parameters
        ----------
        nodes : int
            The number of nodes in the graph.

        Returns
        -------
        edge_index : torch.LongTensor
            The indices of the non-zero elements of the adjacency matrix.
        """
        edge_index = torch.ones([nodes, nodes], dtype=torch.int32) - torch.eye(
            nodes, dtype=torch.int32
        )
        self.edge_index = edge_index.to_sparse()._indices()

    def __getitem__(self, item):
        graph = self.data.iloc[item].to_numpy()
        x, y = graph[:-1].reshape(self.num_nodes, self.num_features), graph[-1]

        # edge index (use complete graph without edge weighting)
        self.load_edges(self.num_nodes)

        x /= self.scale
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        g = PyGData.Data(x=x, y=y, edge_index=self.edge_index)

        return (g, y)

    def __len__(self):
        return len(self.data)
