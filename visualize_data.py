import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from data_loader import UnifiedERA5Dataset
from utils.YParams import YParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='config.yaml', type=str)
    parser.add_argument("--config", default='base', type=str)
    parser.add_argument("--timestamp_idx", default=0, type=int,
                         help="Which timestamp to visualize (0 = first timestamp in train_years)")
    args = parser.parse_args()
    params = YParams(args.yaml_config, args.config)

    dataset = UnifiedERA5Dataset(
        years=params.train_years,
        input_channels=params.era5_channel_input,
        output_channels=params.era5_channel_output,
        region=getattr(params, 'region', 'us_midwest'),
        dt=getattr(params, 'dt', 6),
    )

    sample = dataset[args.timestamp_idx]
    input_data = sample['input']    # [num_input_channels, H, W]
    output_data = sample['output']  # [num_output_channels, H, W]
    timestamp = sample['timestamp']
    timestamp_readable = pd.to_datetime(timestamp).strftime('%B %d, %Y %H:%M UTC')
    input_names = params.era5_channel_input
    output_names = params.era5_channel_output

    lat = dataset.lat
    lon = dataset.lon
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    lon2d, lat2d = np.meshgrid(lon, lat)

    num_inputs = len(input_names)
    total_panels = num_inputs + len(output_names)
    ncols = 5
    nrows = math.ceil(total_panels / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    axes = axes.flatten()

    # Plot input channels
    for i, name in enumerate(input_names):
        ax = axes[i]
        im = ax.pcolormesh(
            lon2d, lat2d, input_data[i].numpy(),
            cmap='coolwarm',
            transform=ccrs.PlateCarree(),
            shading='auto',
        )
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(f"Input: {name}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Plot Output
    for j, name in enumerate(output_names):
        ax = axes[num_inputs + j]
        im = ax.pcolormesh(
            lon2d, lat2d, output_data[j].numpy(),
            cmap='coolwarm',
            transform=ccrs.PlateCarree(),
            shading='auto',
        )
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(f"Output: {name}", fontsize=11, fontweight='bold', color='darkblue')
        plt.colorbar(im, ax=ax, fraction=0.046)

    for k in range(total_panels, len(axes)):
        axes[k].axis('off')

    fig.suptitle(f"ERA5 Variables  {timestamp_readable}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('all_variables_viz.png', dpi=120)
    print(f"Saved visualization to all_variables_viz.png")
    print(f"Timestamp: {timestamp_readable}")
    print(f"Showing {num_inputs} input channels + {len(output_names)} output channel(s)")


if __name__ == '__main__':
    main()
