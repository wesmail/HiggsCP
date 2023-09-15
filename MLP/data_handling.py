# torch imports
import torch

class HiggsMLPDataset(torch.utils.data.Dataset):
    """
    PyTorch class to generate tabular data
    """
    def __init__(self, indata, shuffle=False):
        # read in the ascii file
        self.data = indata
        if shuffle:
            self.data = self.data.sample(frac=1.0)

    def __getitem__(self, item):
        inputs = self.data.iloc[item, 0:-1]

        x = torch.tensor(inputs, dtype=torch.float32)
        y = torch.tensor(self.data.iloc[item, -1].item(), dtype=torch.int64)
        
        return {"inputs": x, "labels": y}

    def __len__(self):
        return self.data.shape[0]    