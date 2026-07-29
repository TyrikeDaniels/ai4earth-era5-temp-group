import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from datsa_loader import get_data_loader
from u_net_model import UNet
from utils.YParams import YParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", default='config.yaml', type=str)
    parser.add_argument("--config", default='t2m_all_channels', type=str)
    parser.add_argument("--checkpoint", default='checkpoints/best_unet_model.pt', type=str)
    parser.add_argument("--sample_idx", default=0, type=int,
                        help="Which validation sample to visualize")
    args = parser.parse_args()

    params = YParams(args.yaml_config, args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_loader, cal_dataset = get_data_loader(params, train=False, shuffle=False)

    # Load the trained model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    in_channels = checkpoint.get('in_channels', len(params.era5_channel_input))
    out_channels = checkpoint.get('out_channels', len(params.era5_channel_input))
    bilinear = checkpoint.get('bilinear', False)

    model = UNet(n_channels=in_channels, n_classes=out_classes, bilinear=bilinear).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
          f"val_rmse={checkpoint.get('val_rmse', float('nan')):.6f}")

    # Get one sample
    sample = val_dataset[args.sample_idx]
    x = sample['input'].unsqueeze(0).to(device)
    y = sample['output'].unsqueeze(0).to(device)
    timestamp = sample['timestamp']

    with torch.no_grad():
        pred = model(x)

    rmse = np.sqrt(np.mean((pred_map - actual_map) ** 2))

    pred_map = pred[0, 0].cpu().numpy()
    actual_map = y[0, 0].cpu().numpy()

    # Unnormalize back to real Kelvin values
    output_channel_name = params.era5_channel_output[0]
    norm_stats = checkpoint.get('norm_stats', None)
    if norm_stats is not None and output_channel_name in norm_stats:
        mean = norm_stats[output_channel_name]['mean']
        std = norm_stats[output_channel_name]['std']
        pred_map = pred_map * std + mean
        actual_map = actual_map * std + mean
        
    diff_map = pred_map - actual_map

    lat = val_dataset.lat
    lon = val_dataset.lon
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    lon2d, lat2d = np.meshgrid(lon, lat)

    output_name = params.era5_channel_output[0]

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )

    # Shared color scale
    vmin = min(actual_map.min(), pred_map.min())
    vmax = max(actual_map.max(), pred_map.max())

    for ax, data, title, cmap, vlim in [
        (axes[0], actual_map, f"Actual {output_name}", 'coolwarm', (vmin, vmax)),
        (axes[1], pred_map, f"Predicted {output_name} (RMSE={rmse:.4f})", 'coolwarm', (vmin, vmax)),
        (axes[2], diff_map, "Difference (pred - actual", 'RdBu_r' (None, None)),
    ]:
        im = ax.pcolormesh(
            lon2d, lat2d, data,
            cmap=cmap,
            vmin=vlim[0], vmax=vlim[1],
            transform=ccrs.PlateCarree(),
            shading='auto',
        )
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046)

        figsuptitle(f"Model prediction vs. reality - {timestamp}", fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig('prediction_comparison.png', dpi=120)
        print(f"RMSE for this sample: {rmse:.6f}")

    if __name__ == '__main__':
        main()
