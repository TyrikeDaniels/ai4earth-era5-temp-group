"""
Dummy ERA5 data loader for testing your ConvLSTM-U-Net pipeline WITHOUT MSI access. No normalization.

Mimics the interface of UnifiedERA5Dataset / get_data_loader from data_loader.py,
but generates random synthetic data matching your real config:
  - 53 x 97 spatial grid
  - 27 input channels
  - zero-inflated precipitation-like output (mostly zeros, occasional bursts)

Key difference from the real dataset: this one returns SEQUENCES,
shape (seq_len, C, H, W), since your ConvLSTM needs a time window as input,
not a single timestep. When you swap in the real MSI data later, you'll need
to wrap UnifiedERA5Dataset with similar windowing logic (grab idx..idx+seq_len).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.convlstmunet import UNetConvLSTM  
from models.weighted_mse_loss import WeightedMSELoss  

class DummyERA5Dataset(Dataset):
    def __init__(
        self,
        num_samples=200,
        seq_len=6,             # input timesteps fed to ConvLSTM
        pred_len=1,            # output timesteps to predict
        n_input_channels=27,
        n_output_channels=1,   # e.g. precipitation only
        height=53,
        width=97,
        seed=0,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.height = height
        self.width = width
        self.rng = np.random.default_rng(seed)

        # fake metadata, mirrors what UnifiedERA5Dataset exposes
        self.lat = np.linspace(30, 50, height).astype(np.float32)
        self.lon = np.linspace(-100, -80, width).astype(np.float32)
        self.channels = [f"channel_{i}" for i in range(n_input_channels)]
        self.input_channels = self.channels
        self.output_channels = self.channels[:n_output_channels]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_seq = self.rng.normal(
            size=(self.seq_len, self.n_input_channels, self.height, self.width)
        ).astype(np.float32)

        # zero-inflated target: mostly zero, sparse exponential bursts (mimics precip)
        output_seq = np.zeros(
            (self.pred_len, self.n_output_channels, self.height, self.width),
            dtype=np.float32,
        )
        rain_mask = self.rng.random(output_seq.shape) < 0.1
        output_seq[rain_mask] = self.rng.exponential(scale=2.0, size=rain_mask.sum())

        return {
            "input": torch.from_numpy(input_seq),      # (seq_len, C, H, W)
            "output": torch.from_numpy(output_seq),     # (pred_len, C_out, H, W)
            "timestamp": f"dummy_{idx}",
            "global_idx": idx,
        }

def get_dummy_data_loader(batch_size=4, num_samples=200, seq_len=6, num_workers=0, **kwargs):
    dataset = DummyERA5Dataset(num_samples=num_samples, seq_len=seq_len, **kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return loader, dataset

if __name__ == "__main__":
    # 1. Get a dummy loader
    loader, dataset = get_dummy_data_loader(batch_size=2, num_samples=20, seq_len=6)

    # 2. Grab a batch, pull out ONE sequence to sanity-check shapes
    batch = next(iter(loader))
    print("input batch shape: ", batch["input"].shape)   # (B, T, C, H, W)
    print("output batch shape:", batch["output"].shape)  # (B, T_out, C_out, H, W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    single_input = batch["input"][0:1].to(device)    # (1, seq_len, 27, 53, 97)
    single_output = batch["output"][0:1].to(device)  # (1, 1, 1, 53, 97)

    model = UNetConvLSTM(input_channels=27, hidden_channels=[16, 32, 64], output_channels=1).to(device)
    criterion = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(50):
        optimizer.zero_grad()
        pred = model(single_input)
        target = single_output[:, 0]
        loss = criterion(pred, target, weights=(target > 0).float() + 1.0)  # weight non-zero targets more
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"step {step:02d} | loss {loss.item():.4f}")