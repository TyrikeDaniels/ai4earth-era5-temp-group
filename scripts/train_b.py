import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.train_utils import evaluate, compute_loss, load_params
from models.trajgru_unet import UNetTrajGRU  
from data_loader import get_data_loader 

CHECKPOINT_DIR = "./models/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model_b.pt")


def train(
    model: nn.Module,
    dataloader: DataLoader,
    valid_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler = None,
    epochs: int = 50,
    #resume: bool = False,
):
    best_val_loss = float("inf")
    start_epoch = 0

    # if resume and os.path.exists(CHECKPOINT_PATH):
    #     checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     best_val_loss = checkpoint["best_val_loss"]
    #     start_epoch = checkpoint["epoch"] + 1
    #     print(f"Resumed from epoch {checkpoint['epoch']}, best_val_loss={best_val_loss:.4f}")

    losses = []
    for epoch in range(start_epoch, epochs):
            n_batches = 0
            epoch_loss = epoch_bce = epoch_intensity = 0.0
            epoch_tp = epoch_fp = epoch_fn = 0.0
            # epoch_rain_pixels = epoch_total_pixels = 0.0

            for batch in dataloader:
                x = batch["input"].to(device)
                y_log_norm = batch["output_log_norm"].to(device)
                y_raw = batch["output_raw"].to(device)

                optimizer.zero_grad()

                rain_logit, intensity_pred = model(x)
                metrics = compute_loss(
                    rain_logit=rain_logit,
                    intensity_pred=intensity_pred,
                    precip_raw=y_raw,
                    log_precip_norm=y_log_norm,
                    device=device,
                )
                metrics["loss"].backward()
                optimizer.step()

                epoch_loss += metrics["loss"].item()
                epoch_bce += metrics["bce_loss"].item()
                epoch_intensity += metrics["intensity_loss"].item()
                epoch_tp += metrics["tp"].item()
                epoch_fp += metrics["fp"].item()
                epoch_fn += metrics["fn"].item()
                n_batches += 1

            if scheduler: 
                scheduler.step()

            # Average training metrics
            avg_loss = epoch_loss / n_batches
            avg_bce = epoch_bce / n_batches
            avg_intensity = epoch_intensity / n_batches
            train_precision = epoch_tp / (epoch_tp + epoch_fp + 1e-6)
            train_recall = epoch_tp / (epoch_tp + epoch_fn + 1e-6)
            train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-6)

            # Validation
            val_metrics = evaluate(model, valid_dataloader, device)

            # Save checkpoint
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                    },
                    CHECKPOINT_PATH,
                )

            # Output
            print(f"Epoch {epoch:02d}")
            print(f"  Train | loss={avg_loss:.4f} bce={avg_bce:.4f} int={avg_intensity:.4f} | p={train_precision:.3f} r={train_recall:.3f} f1={train_f1:.3f}")
            print(f"  Val   | loss={val_metrics['loss']:.4f} bce={val_metrics['bce_loss']:.4f} int={val_metrics['intensity_loss']:.4f} | p={val_metrics['precision']:.3f} r={val_metrics['recall']:.3f} f1={val_metrics['f1']:.3f}")
            print(f"  LR    | {optimizer.param_groups[0]['lr']:.2e}")



    return best_val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    params = load_params(1)

    train_dataloader, _ = get_data_loader(params, train=True, shuffle=True)
    valid_dataloader, _ = get_data_loader(params, train=False, shuffle=False)

    model = UNetTrajGRU(
        input_channels=len(params.era5_channel_input),
        hidden_channels=[16, 32, 64],
        output_channels=1,
        use_attention_gates=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) 
    warmup_epochs = 2
    total_epochs = 30

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,   # start at 10% of base LR
        end_factor=1.0,     # ramp up to full base LR
        total_iters=warmup_epochs,
    )
    
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

    train(
        model=model,
        dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        epochs=total_epochs,
    )

if __name__ == "__main__":
    main()
