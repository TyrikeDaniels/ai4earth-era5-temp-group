import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os

from utils import YParams
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.convlstmunet import UNetConvLSTM  
from models.weighted_mse_loss import WeightedMSELoss  
from data_loader import get_data_loader  # adjust class name to match yours

def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 50
    ):
    
    for batch in dataloader:
        features = batch["input"].to(device) # (1, seq_len, 27, 53, 97)
        target = batch["output"].to(device)  # (1, 1, 1, 53, 97)

        for step in range(epochs):
            optimizer.zero_grad()
            pred = model(features)
            target = target[:, 0]  # Assuming we want to predict the first timestep of the output
            loss = criterion(pred, target, weights=(target > 0).float() + 1.0)  # weight non-zero targets more
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(f"step {step:02d} | loss {loss.item():.4f}")
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Visualize ERA5 data using the data loader")
    parser.add_argument("--yaml_config", default='config.yaml', type=str, help="Path to YAML config file")
    parser.add_argument("--config", default='base', type=str, help="Configuration name to use")
    parser.add_argument("--sample_idx", default=0, type=int, help="Sample index to visualize")
    parser.add_argument("--compare", action='store_true', help="Compare input and output channels")
    parser.add_argument("--animate_comparison", action='store_true', help="Create animated GIF comparing input vs output")
    parser.add_argument("--num_frames", default=12, type=int, help="Number of frames for animation")
    parser.add_argument("--interval", default=500, type=int, help="Time between frames in milliseconds")
    parser.add_argument("--input_channels", nargs='*', type=str, help="Specific input channel names to visualize (max 3). Example: --input_channels t2m u10 v10")
    args = parser.parse_args()
    
    # Load configuration and create data loader
    print("Loading configuration and creating data loader...")
    params = YParams(args.yaml_config, args.config)
    
    # Set parameters for visualization
    params.local_batch_size = 1
    params.num_data_workers = 0
    params.shuffle = False

    dataloader, dataset = get_data_loader(params, train=True, shuffle=False)


    #model = UNetConvLSTM(input_channels=27, hidden_channels=[16, 32, 64], output_channels=1).to(device)
    #criterion = WeightedMSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #train(model, criterion, optimizer, dataloader, device)

if __name__ == "__main__":
    main()
