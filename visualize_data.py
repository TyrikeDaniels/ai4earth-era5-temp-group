import argparse
import math
import matplotlib.pyplot as plt

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

    #(2020-2022)
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

    input_names = params.era5_channel_input
    output_names = params.era5_channel_output

    num_inputs = len(input_names)
    total_panels = num_inputs + len(output_names)

    #Subplots
    ncols = 5
    nrows = math.ceil(total_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    # Plot input channels
    for i, name in enumerate(input_names):
        ax = axes[i]
        im = ax.imshow(input_data[i].numpy(), cmap='coolwarm')
        ax.set_title(f"Input: {name}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Plot precipitation
    for j, name in enumerate(output_names):
        ax = axes[num_inputs + j]
        im = ax.imshow(output_data[j].numpy(), cmap='Blues')
        ax.set_title(f"Output: {name}", fontsize=11, fontweight='bold', color='black')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    for k in range(total_panels, len(axes)):
        axes[k].axis('off')

    fig.suptitle(f"All variables at timestamp: {timestamp}", fontsize=14)
    plt.tight_layout()
    plt.savefig('all_variables_viz.png', dpi=120)
    print(f"Saved visualization to all_variables_viz.png")
    print(f"Timestamp: {timestamp}")
    print(f"Showing {num_inputs} input channels + {len(output_names)} output channel(s)")


if __name__ == '__main__':
    main()
