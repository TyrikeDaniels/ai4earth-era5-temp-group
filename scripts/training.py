import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.training_utils import compute_loss, load_params

from models.convlstmunet import UNetConvLSTM  
from data_loader import get_data_loader 

CHECKPOINT_DIR = "./models/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    lam: float = 1.0,
) -> dict:
    """Run the model over a validation set with no gradient updates."""
    model.eval()

    n_batches = 0
    total_loss = total_bce = total_mse = 0.0
    total_tp = total_fp = total_fn = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["input"].to(device)
            y = batch["output_log_norm"].to(device)
            y_raw = batch["output_raw"].to(device)

            rain_logit, intensity_pred = model(x)
            loss, bce_loss, mse_loss, tp, fp, fn = compute_loss(
                rain_logit=rain_logit,
                intensity_pred=intensity_pred,
                precip_raw=y_raw,
                log_precip_norm=y,
                device=device,
                lam=lam,
            )

            total_loss += loss.item()
            total_bce += bce_loss.item()
            total_mse += mse_loss.item()
            total_tp += tp.item()
            total_fp += fp.item()
            total_fn += fn.item()
            n_batches += 1

    model.train()  # switch back before returning to the training loop

    avg_loss = total_loss / n_batches
    avg_bce = total_bce / n_batches
    avg_mse = total_mse / n_batches
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "loss": avg_loss, "bce": avg_bce, "mse": avg_mse,
        "precision": precision, "recall": recall, "f1": f1,
    }

def train(
    model: nn.Module,
    dataloader: DataLoader,
    valid_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    epochs: int = 50,
):
    best_val_loss = float("inf")
    start_epoch = 0

    # --- resume from checkpoint if one exists ---
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found checkpoint at {CHECKPOINT_PATH}, resuming...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}, best_val_loss so far: {best_val_loss:.4f}")

    for epoch in range(start_epoch, epochs):
        n_batches = 0
        epoch_loss = epoch_bce = epoch_mse = 0.0
        epoch_tp = epoch_fp = epoch_fn = 0.0

        for batch in dataloader:
            x = batch["input"].to(device)
            y = batch["output_log_norm"].to(device)
            y_raw = batch["output_raw"].to(device)

            optimizer.zero_grad()
            rain_logit, intensity_pred = model(x)
            loss, bce_loss, mse_loss, tp, fp, fn = compute_loss(
                rain_logit=rain_logit,
                intensity_pred=intensity_pred,
                precip_raw=y_raw,
                log_precip_norm=y,
                device=device,
            )
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()

            epoch_loss += loss.item()
            epoch_bce += bce_loss.item()
            epoch_mse += mse_loss.item()
            epoch_tp += tp.item()
            epoch_fp += fp.item()
            epoch_fn += fn.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_bce = epoch_bce / n_batches
        avg_mse = epoch_mse / n_batches
        precision = epoch_tp / (epoch_tp + epoch_fp + 1e-6)
        recall = epoch_tp / (epoch_tp + epoch_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        # --- validation pass ---
        val_metrics = evaluate(model, valid_dataloader, device)

        # --- checkpoint on best VAL loss, not train loss ---
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

        print(
            f"epoch {epoch:02d} | "
            f"train loss {avg_loss:.4f} bce {avg_bce:.4f} mse {avg_mse:.4f} "
            f"p {precision:.4f} r {recall:.4f} f1 {f1:.4f} | "
            f"val loss {val_metrics['loss']:.4f} bce {val_metrics['bce']:.4f} "
            f"mse {val_metrics['mse']:.4f} p {val_metrics['precision']:.4f} "
            f"r {val_metrics['recall']:.4f} f1 {val_metrics['f1']:.4f}"
        )

    return best_val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    params = load_params(device)

    train_dataloader, train_dataset = get_data_loader(params, train=True, shuffle=True)
    valid_dataloader, valid_dataset = get_data_loader(params, train=False, shuffle=False)

    model = UNetConvLSTM(
        input_channels=len(params.era5_channel_input),
        hidden_channels=[64, 128, 256],
        output_channels=1,
        use_attention_gates=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    warmup_epochs = 5
    total_epochs = 100

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
    # For testing purposes, we can call main() directly
    main()
