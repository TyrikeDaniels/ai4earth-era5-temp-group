import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
import os
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.train_utils import evaluate, compute_loss, load_params
from utils.mcc import matthews_correlation_coefficient
from models.convlstm_unet import UNetConvLSTM  
from utils.data_loader import get_data_loader 

CHECKPOINT_DIR = "./models/checkpoints"


def train(
    model: nn.Module,
    dataloader: DataLoader,
    valid_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dt: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epochs: int = 50,
    resume: bool = False,
) -> float:

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_a_{dt}.pt")
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, f"latest_model_a_{dt}.pt")

    best_val_loss = float("inf")
    start_epoch = 0

    bce_loss_history = {"val": [], "train": []}
    mse_loss_history = {"val": [], "train": []}

    if resume and os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        bce_loss_history = checkpoint.get("bce_loss_history", bce_loss_history)
        mse_loss_history = checkpoint.get("mse_loss_history", mse_loss_history)
        print(f"Resumed from checkpoint at epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")

    for epoch in range(start_epoch, epochs):
        n_batches = 0
        epoch_loss = epoch_bce = epoch_intensity = 0.0
        epoch_tp = epoch_fp = epoch_fn = epoch_tn = epoch_mcc = 0.0

        for batch in dataloader:
            x = batch["input"].to(device)
            y_log_norm = batch["output_log_norm"].to(device)
            y_raw = batch["output_raw"].to(device)

            optimizer.zero_grad()

            rain_logit, intensity_pred, _ = model(x)
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
            epoch_intensity += metrics["mse_loss"].item()
            epoch_tp += metrics["tp"].item()
            epoch_fp += metrics["fp"].item()
            epoch_fn += metrics["fn"].item()
            epoch_tn += metrics["tn"].item()
            n_batches += 1

        if scheduler:
            scheduler.step()

        # Training metrics
        avg_loss = epoch_loss / n_batches
        avg_bce = epoch_bce / n_batches
        avg_mse = epoch_intensity / n_batches
        train_mcc = matthews_correlation_coefficient(epoch_tp, epoch_tn, epoch_fp, epoch_fn)
        train_precision = epoch_tp / (epoch_tp + epoch_fp + 1e-6)
        train_recall = epoch_tp / (epoch_tp + epoch_fn + 1e-6)
        
        # Validation
        val_metrics = evaluate(model, valid_dataloader, device)

        # Record
        bce_loss_history["train"].append(avg_bce)
        bce_loss_history["val"].append(val_metrics["bce_loss"])
        mse_loss_history["train"].append(avg_mse)
        mse_loss_history["val"].append(val_metrics["mse_loss"])

        # Output
        print(f"Epoch {epoch:02d}")
        print(f"  Train | net loss = {avg_loss:.4f} bce={avg_bce:.4f} mse={avg_mse:.4f} | p={train_precision:.3f} r={train_recall:.3f} mcc={train_mcc:.3f}")
        print(f"  Val   | net loss = {val_metrics['loss']:.4f} bce={val_metrics['bce_loss']:.4f} mse={val_metrics['mse_loss']:.4f} | p={val_metrics['mcc']:.3f} r={val_metrics['recall']:.3f} mcc={val_metrics['mcc']:.3f}")
        print(f"  LR    | {optimizer.param_groups[0]['lr']:.2e}")

        # Benchmark
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_loss": best_val_loss,
            "bce_loss_history": bce_loss_history,
            "mse_loss_history": mse_loss_history,
        }

        # Save latest so resume works even if this run gets cut off
        torch.save(checkpoint, latest_ckpt_path)

        # Save best only when validation loss improves
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_ckpt_path)
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")

    return best_val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for dt in [1, 2, 3, 4]:
        params = load_params(dt)

        train_dataloader, _ = get_data_loader(params, train=True, shuffle=True)
        valid_dataloader, _ = get_data_loader(params, train=False, shuffle=False)

        model = UNetConvLSTM(
            input_channels=len(params.era5_channel_input),
            hidden_channels=[16, 32, 64],
            output_channels=1,
            use_attention_gates=False,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # 1e-4, before
        warmup_epochs = 2
        total_epochs = 30

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer,
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
