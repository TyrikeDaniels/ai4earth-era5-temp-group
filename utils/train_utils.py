import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.YParams import YParams


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Run the model over a validation set with no gradient updates."""
    model.eval()

    n_batches = 0
    total_loss = total_bce = total_intensity = 0.0
    total_tp = total_fp = total_fn = 0.0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["input"].to(device)
            y = batch["output_log_norm"].to(device)
            y_raw = batch["output_raw"].to(device)

            rain_logit, intensity_pred = model(x)
            metrics = compute_loss(
                rain_logit=rain_logit,
                intensity_pred=intensity_pred,
                precip_raw=y_raw,
                log_precip_norm=y,
                device=device,
            )

            total_loss += metrics["loss"].item()
            total_bce += metrics["bce_loss"].item()
            total_intensity += metrics["intensity_loss"].item()
            total_tp += metrics["tp"].item()
            total_fp += metrics["fp"].item()
            total_fn += metrics["fn"].item()
            n_batches += 1

    model.train()

    avg_loss = total_loss / n_batches
    avg_bce = total_bce / n_batches
    avg_intensity = total_intensity / n_batches
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "loss": avg_loss,
        "bce_loss": avg_bce,
        "intensity_loss": avg_intensity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def compute_loss(
    rain_logit: torch.Tensor,
    intensity_pred: torch.Tensor,
    precip_raw: torch.Tensor,
    log_precip_norm: torch.Tensor,
    device: torch.device,
    pos_weight: float = 2.0,
    threshold: float = 0.45,
    lambda_intensity: float = 1.0,
    omega_intensity: float = 0.8,
) -> dict:  
    """
    Compute combined binary + intensity loss.
    """
    
    rain_mask = (precip_raw > 0).float()
    n_rain_pixels = rain_mask.sum()
    
    bce_loss = F.binary_cross_entropy_with_logits(
        rain_logit,
        rain_mask,
        pos_weight=torch.tensor(pos_weight, device=device),
        reduction='mean'
    )
    
    intensity_loss = F.mse_loss(
        intensity_pred * rain_mask,
        log_precip_norm * rain_mask,
        reduction='sum'
    ) / (n_rain_pixels + 1e-6) 
    
    total_loss = omega_intensity * bce_loss + lambda_intensity * intensity_loss

    with torch.no_grad():
        rain_pred = (torch.sigmoid(rain_logit) > threshold).float()
        tp = (rain_pred * rain_mask).sum()
        fp = (rain_pred * (1 - rain_mask)).sum()
        fn = ((1 - rain_pred) * rain_mask).sum()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'loss': total_loss,
        'bce_loss': bce_loss,
        'intensity_loss': intensity_loss,
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
    }

def load_params(dt: int = 1):

    # Set parameters 
    parser = argparse.ArgumentParser(description="Model ERA5 (2020-23) data using the data loader.")
    parser.add_argument("--yaml_config", default='config.yaml', type=str, help="Path to YAML config file")
    parser.add_argument("--config", default='base', type=str, help="Configuration name to use")
    parser.add_argument("--train", action="store_true", help="Flag to indicate training mode.")
    args = parser.parse_args()

    # Load configuration and create data loader
    params = YParams(args.yaml_config, args.config)
    
    WIND_SURFACE = ["u10", "v10"]                     # 10m wind components
    TEMPERATURE_SURFACE = ["t2m"]                     # OUTPUT TARGET, not an input
    SURFACE = ["skt", "lsm"]                          # skin temp, land-sea mask
    PRECIP = ["avg_tprate"]                           # OUTPUT TARGET, not an input
    
    GEOPOTENTIAL = ["z_1000", "z_600", "z_200"]        # height of pressure surfaces
    TEMPERATURE_ALTITUDE = ["t_800", "t_600", "t_400"] # air temp at altitude
    HUMIDITY = ["q_1000", "q_800", "q_600"]            # water vapor content
    WIND_U = ["u_800", "u_600", "u_400"]               # zonal wind at altitude
    WIND_V = ["v_800", "v_600", "v_400"]               # meridional wind at altitude
    CLOUD_LIQUID = ["clwc_800", "clwc_600", "clwc_400"]  # cloud liquid water content
    CLOUD_ICE = ["ciwc_800", "ciwc_600", "ciwc_400"]     # cloud ice water content
    
    input_channels = WIND_SURFACE + SURFACE + GEOPOTENTIAL + TEMPERATURE_ALTITUDE + HUMIDITY + WIND_U + WIND_V + CLOUD_LIQUID + CLOUD_ICE + TEMPERATURE_SURFACE 

    params.local_batch_size = 5
    params.num_data_workers = 4
    params.shuffle = True
    params.train = True
    params.era5_channel_input = input_channels
    params.era5_channel_output = PRECIP        # Only predict avg_tprate
    params.region = "us_midwest"               # Set the region to 'us_midwest'
    params.train_years = [2020, 2021, 2022]    # Use years 2020-2022 for training
    params.valid_years = [2023]                # Use 2023 for validating
    params.seq_len = 12                        # Number of input timesteps
    params.dt = dt                             # Time step interval in hours (default=6)

    return params
