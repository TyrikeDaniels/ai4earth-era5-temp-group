import argparse
import torch
import torch.nn.functional as F

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.YParams import YParams
except ImportError:
    class YParams:
        def __init__(self, yaml_config, config):
            self.yaml_config = yaml_config
            self.config = config

def compute_loss(
        rain_logit: torch.Tensor, 
        intensity_pred: torch.Tensor, 
        precip_raw: torch.Tensor, 
        log_precip_norm: torch.Tensor, 
        device: torch.device,
        lam: float = 1.0, 
        eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    rain_mask_true = (precip_raw > 0).float()

    with torch.no_grad():
        pred_rain = (torch.sigmoid(rain_logit) > 0.5).float()
        tp = (pred_rain * rain_mask_true).sum()
        fp = (pred_rain * (1 - rain_mask_true)).sum()
        fn = ((1 - pred_rain) * rain_mask_true).sum()

    pos_weight = torch.tensor(7.0, device=device) 
    bce_loss = F.binary_cross_entropy_with_logits(rain_logit, rain_mask_true, pos_weight=pos_weight) # pos_weight to weigh positivie instances

    mse_loss = F.smooth_l1_loss(
        intensity_pred * rain_mask_true,
        log_precip_norm * rain_mask_true,
        reduction='sum'
    ) / (rain_mask_true.sum() + eps)

    total_loss = bce_loss + lam * mse_loss
    return total_loss, bce_loss, mse_loss, tp, fp, fn

def load_params(device: torch.device):

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
    params.era5_channel_output = PRECIP        # Only predict t2m
    params.region = "us_midwest"               # Set the region to 'us_midwest'
    params.train_years = [2020, 2021, 2022]    # Use years 2020-2022 for training
    params.valid_years = [2023]                # Use 2023 for validating
    params.seq_len = 6                        # Number of input timesteps
    params.dt = 6                              # Time step interval in hours (default=6)

    return params
