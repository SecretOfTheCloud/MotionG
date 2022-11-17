import torch
import numpy as np
import torch.utils.data
from normalize import normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class FrameDataSet(Dataset):
    def __init__(self, x, y, p, z):
        self.x, self.y = x, y
        self.number, self.p = len(self.x), p
        self.z = z
        
    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.p[idx], self.z[idx]

def paired_collate_fn(insts):
    x, y, p, z = list(zip(*insts))
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)
    p = torch.FloatTensor(p)
    z = torch.FloatTensor(z)
    return x, y, p, z

def dataloader(batch, x, y, p, z):
    dataset = FrameDataSet(x, y, p, z)
    return DataLoader(
        dataset,
        num_workers = 0,
        batch_size = batch,
        shuffle = True,
        collate_fn = paired_collate_fn,
        pin_memory = True)