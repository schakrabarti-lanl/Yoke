"""Custom dataset class specialized for a new data source (PDEBench).
Currently, it only handles compNS dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from yoke.datasets.hdf5_datasets import CompNSDataset

class CompNSLodeRunnerDataset(Dataset):
    def __init__(self, root_dir, split='train', n_steps=2, dt=1, transform=None):
        """Args:
        root_dir (str): Path to folder containing HDF5 files.
        split (str): 'train', 'val', or 'test'
        n_steps (int): Number of input steps (T).
        dt (int): Time delta between steps.
        transform (callable): Optional transform (e.g., ResizePadCrop)
        """
        self.dataset = CompNSDataset(
            path=root_dir,
            include_string='',  # Match all
            n_steps=n_steps,
            dt=dt,
            split=split,
            train_val_test=(0.8, 0.1, 0.1)
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_seq, _, target = self.dataset[idx]  # [T, C, H, W], target: [C, H, W]

        # Combine input sequence and target into a longer image sequence
        # img_seq: [T, C, H, W] → [T+1, C, H, W]
        full_seq = torch.cat([img_seq, target.unsqueeze(0)], dim=0)

        if self.transform:
            full_seq = self.transform(full_seq)

        return full_seq[:-1], full_seq[-1]  # input_seq [T, C, H, W], target [C, H, W]



class CompNSDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        self.train_set = CompNSLodeRunnerDataset(self.data_dir, split='train', transform=self.transform)
        self.val_set = CompNSLodeRunnerDataset(self.data_dir, split='val', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

