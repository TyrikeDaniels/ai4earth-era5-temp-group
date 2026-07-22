import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.YParams import YParams

from models.convlstmunet import UNetConvLSTM  
#from models.weighted_mse_loss import WeightedMSELoss  
from data_loader import get_data_loader  # adjust class name to match yours

def train(
    model: nn.Module,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 50
    ):
        
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            features = batch["input"].to(device)
            target = batch["output"].to(device)
            pred = model(features)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        print(f"epoch {epoch:02d} | avg loss {avg_loss:.4f}")

def simple_visualization(dataloader: DataLoader, dataset: object):
        
    sample_batch = next(iter(dataloader))
    input_data = sample_batch['input']
    output_data = sample_batch['output']
    
    print(f"\nData shapes:")
    print(f"  Input: {input_data.shape}")  # [batch, channels, lat, lon]
    print(f"  Output: {output_data.shape}") # [batch, channels, lat, lon]
    
    print(f"\nCoordinate information:")
    print(f"  Latitude range: {dataset.lat.min():.2f} to {dataset.lat.max():.2f}")
    print(f"  Longitude range: {dataset.lon.min():.2f} to {dataset.lon.max():.2f}")
    print(f"  Spatial dimensions: {len(dataset.lat)} x {len(dataset.lon)}")
    
    print(f"\nAvailable channels:")
    print(f"  Input channels ({len(dataset.input_channels)}): {dataset.input_channels}")
    print(f"  Output channels ({len(dataset.output_channels)}): {dataset.output_channels}")
    
    print(f"\nTime information:")
    print(f"  First timestamp: {sample_batch['timestamp'][0]}")
    print(f"  Years covered: {dataset.years}")
    print("==========================================\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Model ERA5 (2020-23) data using the data loader.")
    parser.add_argument("--yaml_config", default='config.yaml', type=str, help="Path to YAML config file")
    parser.add_argument("--config", default='base', type=str, help="Configuration name to use")
    parser.add_argument("--train", action="store_true", help="Flag to indicate training mode.")
    args = parser.parse_args()

    # Load configuration and create data loader
    print("Loading configuration and creating data loader...")
    params = YParams(args.yaml_config, args.config)

    # Define input channels for the model
    input_channels = [
        "u10", "v10", "skt", "lsm",
        "avg_tprate", "z_1000", "z_600", "z_200",
        "clwc_800", "clwc_600", "clwc_400",
        "ciwc_800", "ciwc_600", "ciwc_400",
        "q_1000", "q_800", "q_600",
        "t_800", "t_600", "t_400",
        "u_800", "u_600", "u_400",
        "v_800", "v_600", "v_400"
    ]
    
    # Set parameters for visualization
    params.local_batch_size = 5
    params.num_data_workers = 8
    params.shuffle = False
    params.train = True
    params.era5_channel_input = input_channels
    params.era5_channel_output = ["t2m"]  # Only predict t2m
    params.region = "us_midwest"  # Set the region to 'us_midwest'
    params.train_years = [2020, 2021, 2022]  # Use years 2020-2023 for training
    params.seq_len = 6  # Number of input timesteps

    start_time = time.time()
    dataloader, dataset = get_data_loader(params, train=True, shuffle=False)
    init_time = time.time() - start_time

    print(f"Data loader initialized in {init_time:.2f} seconds")

    #simple_visualization(dataloader, dataset)

    model = UNetConvLSTM(input_channels=len(input_channels), hidden_channels=[16, 32, 64], output_channels=1, use_attention_gates=True).to(device)
    #criterion = WeightedMSELoss()
    criterion = nn.MSELoss()  # Use standard MSE loss for simplicity
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    train(model, criterion, scheduler, dataloader, optimizer, device, 60)  # Train for 60 epochs

if __name__ == "__main__":
    # For testing purposes, we can call main() directly
    main()