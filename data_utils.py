import torch
from torch.utils.data import Dataset

class FloodDataset(Dataset):
    """
    Dataset for flattened grid data.
    Each row in X_flat corresponds to a single (sample, lat, lon) point.
    
    X: (n_samples*n_lat*n_lon, T_obs)
    y: (n_samples*n_lat*n_lon,)
    """
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('float32')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])
