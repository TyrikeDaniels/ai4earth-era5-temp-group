"""
Training Module for the ERA5 U-Net model

This module implements the complete training pipeline for the U-Net model,
including training loop, validation, early stopping, and checkpoint management.
The training process uses Mean Squared Error (MSE) loss with mask-aware
computation to focus learning on valid pixel regions.

Key Components:
    - MaskedMSELoss: Custom loss function that excludes padded/invalid regions
    - EarlyStopping: Callback to prevent overfitting
    - train_one_epoch: Single epoch training logic
    - validate: Validation set evaluation
    - train_model: Complete training orchestration

Training Strategy:
    - Adam optimizer with configurable learning rate
    - Early stopping based on validation loss
    - Best model checkpointing
    - Progress tracking with epoch-level metrics
"""

import os
import time
import argparse

import torch
import torch.nn as nn

from data_loader import get_data_loader
from u_net_model import UNet
from utils.YParams import YParams


class EarlyStopping:
    """
    Early Stopping callback to halt training when validation loss stops improving.

    This class monitors validation loss and stops training if no improvement
    is observed for a specified number of consecutive epochs (patience).
    Early stopping prevents overfitting by stopping before the model starts
    memorizing training data at the expense of generalization.

    The callback also handles saving the best model checkpoint.
    """
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        """
                Initializes the early stopping callback.

                :param patience: Number of epochs with no improvement after which
                                 training will be stopped.
                :param min_delta: Minimum change in validation loss to qualify as
                                  an improvement.
                :param verbose: If True, prints messages when early stopping triggers.
                """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        # Internal state tracking
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """
               Checks if training should stop based on validation loss.

               Called at the end of each epoch to evaluate the current validation loss
               against the best observed loss.

               :param val_loss: Current epoch's validation loss.
               :param model: The model being trained (for state dict saving).
               :return: True if this is a new best loss, False otherwise.
               """

        is_best = False

        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            is_best = True

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            is_best = True

        else:
            self.counter += 1
            if self.verbose:
                print(f"    EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"    Early stopping triggered after {self.patience} epochs without improvement")

        return is_best

    def get_best_model_state(self):
        """
                Returns the state dict of the best model observed during training.

                :return: Dictionary containing model state at best validation loss.
                """
        return self.best_model_state


def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
        Executes one complete training epoch.

        This function iterates through all batches in the training set,
        computing forward passes, calculating losses, and updating model
        weights through backpropagation.

        :param model: The U-Net model to train.
        :param dataloader: DataLoader providing training batches.
        :param criterion: Loss function (MaskedMSELoss).
        :param optimizer: Optimizer (Adam).
        :param device: Device to run computations on (CPU/GPU).
        :return: Average training loss for the epoch.
        """
    # Set model to training mode (enables dropout, batch norm training behavior)
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to the appropriate device (GPU if available)
        x = batch['input'].to(device)
        y = batch['output'].to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    # Compute average loss across all batches
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
       Evaluates the model on the validation set.

       This function computes the validation loss without gradient computation,
       providing an unbiased estimate of model performance on unseen data.

       :param model: The U-Net model to evaluate.
       :param dataloader: DataLoader providing validation batches.
       :param criterion: Loss function (MaskedMSELoss).
       :param device: Device to run computations on.
       :return: Average validation loss.
       """
    # Set model to evaluation mode (disables dropout, uses running stats for batch norm)
    model.eval()

    total_loss = 0.0
    total_rmse = 0.0

    num_batches = 0

    for batch in loader:
        x = batch['input'].to(device)
        y = batch['output'].to(device)

        pred = model(x)
        loss = criterion(pred, y)

        total_loss += loss.item()
        total_rmse += compute_rmse(pred, y)
        num_batches += 1

    num_batches = max(num_batches, 1)
    return total_loss / num_batches, total_rmse / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on ERA5 data with early stopping")
    parser.add_argument("--yaml_config", default='config.yaml', type=str)
    parser.add_argument("--config", default='base', type=str)
    parser.add_argument("--epochs", default=30, type=int, help="Maximum number of epochs")
    parser.add_argument("--patience", default=5, type=int, help="Early stopping patience")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--bilinear", action='store_true', help="Use bilinear upsampling instead of ConvTranspose2d")
    parser.add_argument("--checkpoint_dir", default='checkpoints', type=str)
    parser.add_argument("--resume", action='store_true', help="Resume training from the checkpoint")
    args = parser.parse_args()

    params = YParams(args.yaml_config, args.config)
    params.local_batch_size = getattr(params, 'local_batch_size', 8)
    params.num_data_workers = getattr(params, 'num_data_workers', 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading training data (years: {params.train_years})...")
    train_loader, train_dataset = get_data_loader(params, train=True, shuffle=True)

    print(f"Loading validation data (years: {params.valid_years})...")
    val_loader, val_dataset = get_data_loader(params, train=False, shuffle=False)

    in_channels = len(params.era5_channel_input)
    out_channels = len(params.era5_channel_output)
    print(f"Model: in_channels={in_channels}, out_channels={out_channels}, bilinear={args.bilinear}")

    model = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=args.bilinear).to(device)
    model.print_model_summary()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    start_epoch = 1
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_unet_model.pt')

    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        early_stopping.best_loss = checkpoint['val_loss']
        print(f"Resumed from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_unet_model.pt')

    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'best_epoch': 0}

    print(f"\nStarting training for up to {args.epochs} epochs (patience={args.patience})...")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmse = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)

        elapsed = time.time() - start
        is_best = early_stopping(val_loss, model)
        best_marker = " *BEST*" if is_best else ""

        print(f"Epoch [{epoch}/{args.epochs}] "
              f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"val_rmse={val_rmse:.6f} | {elapsed:.1f}s{best_marker}")

        if is_best:
            history['best_epoch'] = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'bilinear': args.bilinear,
            }, checkpoint_path)
            print(f"    Checkpoint saved: {checkpoint_path}")

        if early_stopping.early_stop:
            print(f"\nTraining stopped early at epoch {epoch}")
            break

    print("=" * 60)
    print(f"Training completed. Best epoch: {history['best_epoch']}")
    print(f"Best validation loss: {early_stopping.best_loss:.6f}")

    # Restore best weights before finishing, so the model in memory is the best one
    model.load_state_dict(early_stopping.get_best_model_state())


if __name__ == '__main__':
    main()
