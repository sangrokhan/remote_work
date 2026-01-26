import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ToyDataset(Dataset):
    def __init__(self, size=1000, input_dim=10):
        self.x = torch.randn(size, input_dim)
        # Target function: y = sum(x) + noise, a simple regression task
        self.y = torch.sum(self.x, dim=1, keepdim=True) + 0.1 * torch.randn(size, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloader(batch_size=32, size=1000):
    dataset = ToyDataset(size=size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
