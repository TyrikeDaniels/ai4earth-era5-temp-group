import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
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
    num_steps: int = 50):
    
    for batch in dataloader:
        features = batch["input"].to(device)    # (1, seq_len, 27, 53, 97)
        target = batch["output"].to(device)  # (1, 1, 1, 53, 97)

        for step in range(num_steps):
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

    dataloader, dataset = get_data_loader(batch_size=7, num_samples=20, seq_len=6)

    

    #model = UNetConvLSTM(input_channels=27, hidden_channels=[16, 32, 64], output_channels=1).to(device)
    #criterion = WeightedMSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #train(model, criterion, optimizer, dataloader, device)

if __name__ == "__main__":
    main()
