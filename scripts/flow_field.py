"""
Visualize TrajGRU learned flows overlaid on animated ERA5 data.
Aggregates across 5 trajectories and overlays on input/output comparison.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import calendar
import os, sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trajgru_unet import UNetTrajGRU
from utils.train_utils import load_params
from utils.data_loader import get_data_loader

CHECKPOINT_DIR = "./models/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./visualization_outputs"


def get_samples_for_timeframe(dataset, time_params):
    """
    Extract samples from dataset within a specific timeframe.
    
    Args:
        dataset: UnifiedERA5Dataset instance
        time_params: Dict with 'month', 'start_day', 'end_day', 'year' (default 2023)
    
    Returns:
        samples: List of (input, output_log_norm) tuples
        indices: List of dataset indices corresponding to each sample
    """
    year = time_params.get('year', 2023)
    month = time_params['month']
    start_day = time_params.get('start_day', 1)
    end_day = time_params.get('end_day', calendar.monthrange(year, month)[1])
    
    samples = []
    indices = []
    dt_hours = dataset.dt
    base_date = datetime(year, 1, 1)
    
    for idx in range(len(dataset)):
        sample_datetime = base_date + timedelta(hours=idx * dt_hours)
        
        if (sample_datetime.year == year and 
            sample_datetime.month == month and
            start_day <= sample_datetime.day <= end_day):
            sample = dataset[idx]
            samples.append((sample['input'], sample['output_log_norm']))
            indices.append(idx)
        
        elif (sample_datetime.year == year and 
              sample_datetime.month == month and
              sample_datetime.day > end_day):
            break
    
    return samples, indices


def aggregate_flows(flows_list):
    """
    Aggregate flows from all 5 trajectories using mean resultant vector.
    
    Args:
        flows_list: List of L flow tensors from model, each (B, 2, H, W)
    
    Returns:
        u_mean: Mean x-displacement (H, W)
        v_mean: Mean y-displacement (H, W)
        magnitude: Magnitude of mean vector (H, W) - indicates consensus
    """
    flow_stacked = torch.stack([f for f in flows_list], dim=0)  # (L, B, 2, H, W)
    
    u_mean = flow_stacked[:, 0, 0, :, :].mean(dim=0).cpu().numpy()  # (H, W)
    v_mean = flow_stacked[:, 0, 1, :, :].mean(dim=0).cpu().numpy()
    
    magnitude = np.sqrt(u_mean**2 + v_mean**2)
    
    return u_mean, v_mean, magnitude


def subsample_vectors(u, v, stride=5):
    """Subsample flow vectors for cleaner visualization."""
    ys = np.arange(0, u.shape[0], stride)
    xs = np.arange(0, u.shape[1], stride)
    return u[::stride, ::stride], v[::stride, ::stride], xs, ys


def create_animated_flow_visualization(
    dataset, samples, flows_per_sample, 
    num_frames=12, stride=8, save_path=None, 
    interval=800, input_channels_to_show=None
):
    """
    Create animated visualization with flows overlaid.
    
    Args:
        dataset: UnifiedERA5Dataset instance
        samples: List of (input, output) tuples
        flows_per_sample: List of flows from model for each sample
        start_idx: Starting sample index
        num_frames: Number of frames to animate
        save_path: Path to save GIF
        interval: Milliseconds between frames
        input_channels_to_show: Which input channels to display
        stride: Number of strides for flow field
    """
    
    # Limit frames
    actual_frames = min(num_frames, len(samples))
    samples = samples[:actual_frames]
    flows_per_sample = flows_per_sample[:actual_frames]
    
    print(f"Creating animation with {actual_frames} frames...")
    
    # Determine channels to show
    n_input = len(dataset.input_channels)
    n_output = len(dataset.output_channels)
    
    if input_channels_to_show is None:
        channels_to_show = min(2, n_input)
        input_indices = list(range(channels_to_show))
    else:
        input_indices = [dataset.input_channels.index(ch) for ch in input_channels_to_show]
    
    total_plots = len(input_indices) + n_output
    cols = min(total_plots, 3)
    rows = (total_plots + cols - 1) // cols
    
    # Get coordinates
    lats = dataset.lat
    lons = dataset.lon
    
    # Pre-compute color ranges
    # Inputs are (B, seq_len, C, H, W), grab last timestep
    all_inputs = np.stack([s[0].numpy()[-1] for s in samples], axis=0)  # (B, C, H, W)
    all_outputs = np.stack([s[1].numpy() for s in samples], axis=0)  # (B, H, W)
    
    input_ranges = []
    for idx in input_indices:
        data = all_inputs[:, idx, :, :]  # (B, H, W)
        vmin, vmax = data.min(), data.max()
        input_ranges.append((vmin, vmax))
    
    output_ranges = []
    for i in range(n_output):
        data = all_outputs[:, i, :, :]
        vmin, vmax = data.min(), data.max()
        output_ranges.append((vmin, vmax))
    
    # Set up figure
    fig = plt.figure(figsize=(6*cols, 5*rows))
    axes = []
    images = []
    quivers = []
    
    plot_idx = 1
    
    # Create subplots for inputs
    for i, ch_idx in enumerate(input_indices):
        ax = plt.subplot(rows, cols, plot_idx, projection=ccrs.PlateCarree())
        axes.append(ax)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
        
        data = samples[0][0][-1, ch_idx].numpy() 
        vmin, vmax = input_ranges[i]
        
        img = ax.pcolormesh(lons, lats, data, cmap='YlOrBr', vmin=vmin, vmax=vmax, 
                           transform=ccrs.PlateCarree())
        images.append(img)
        
        # Add initial flow vectors
        u, v, _ = aggregate_flows(flows_per_sample[0][0]) 
        u_sub, v_sub, xs, ys = subsample_vectors(u, v, stride=stride)
        
        # Create meshgrid for quiver
        xx, yy = np.meshgrid(lons[::stride], lats[::stride])
        quiv = ax.quiver(xx, yy, u_sub, v_sub, transform=ccrs.PlateCarree(), 
                        scale=30, scale_units='inches', width=0.002, alpha=0.7, color='black')
        quivers.append((quiv, u, v))  # Store original u, v for updates
        
        plt.colorbar(img, ax=ax, shrink=0.8, pad=0.05)
        ax.set_title(f"Input: {dataset.input_channels[ch_idx]}")
        
        plot_idx += 1
    
    # Create subplots for outputs
    for i in range(n_output):
        ax = plt.subplot(rows, cols, plot_idx, projection=ccrs.PlateCarree())
        axes.append(ax)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
        
        data = samples[0][1][i].numpy()
        vmin, vmax = output_ranges[i]
        
        img = ax.pcolormesh(lons, lats, data, cmap='Blues', vmin=vmin, vmax=vmax, 
                           transform=ccrs.PlateCarree())
        images.append(img)
        
        plt.colorbar(img, ax=ax, shrink=0.8, pad=0.05)
        ax.set_title(f"Output: {dataset.output_channels[i]}")
        
        plot_idx += 1
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    timestamp_text = fig.text(0.5, 0.01, '', ha='center', fontsize=12, 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def animate(frame):
        """Update all subplots for current frame."""
        updated = []
        
        # Update input images
        for i, ch_idx in enumerate(input_indices):
            data = samples[frame][0][-1, ch_idx].numpy()  # Last timestep, specific channel
            images[i].set_array(data.ravel())
            updated.append(images[i])
        
        # Update output images
        for i in range(n_output):
            data = samples[frame][1][i].numpy()  # Output is (H, W)
            images[len(input_indices) + i].set_array(data.ravel())
            updated.append(images[len(input_indices) + i])
        
        # Update flow vectors on input plots
        u_agg, v_agg, _ = aggregate_flows(flows_per_sample[frame][0])
        u_sub, v_sub, xs, ys = subsample_vectors(u_agg, v_agg, stride=stride)
        
        for i, (quiv, _, _) in enumerate(quivers):
            quiv.set_UVC(u_sub, v_sub)
            updated.append(quiv)
        
        # Update timestamp
        dt_hours = dataset.dt
        base_date = datetime(2023, 1, 1)
        current_time = base_date + timedelta(hours=frame * dt_hours)
        timestamp_text.set_text(f"Time: {current_time.strftime('%Y-%m-%d %H:%M UTC')}")
        updated.append(timestamp_text)
        
        return updated
    
    print(f"Animating {actual_frames} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=actual_frames, 
                                   interval=interval, blit=False, repeat=True)
    
    if save_path:
        print(f"Saving to {save_path}...")
        writer = animation.PillowWriter(fps=1000/interval)
        anim.save(save_path, writer=writer)
        print(f"Saved to {save_path}")
    
    plt.show()
    return anim


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load checkpoint
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model_b_1.pt")
    if not os.path.exists(best_ckpt_path):
        raise Exception(f"Checkpoint not found at {best_ckpt_path}.")
    
    model_dict = torch.load(best_ckpt_path, map_location=DEVICE)
    print("Checkpoint loaded")
    
    # Load params and data
    params = load_params(1)
    _, dataset = get_data_loader(params, train=False, shuffle=False)
    print("Dataset loaded")
    
    # Load model
    model = UNetTrajGRU(
        input_channels=len(params.era5_channel_input),
        hidden_channels=[16, 32, 64],
        output_channels=1,
    ).to(DEVICE)
    model.load_state_dict(model_dict["model_state_dict"])
    model.eval()
    print("Model loaded")
    
    # Get samples for timeframe
    time_params = {'month': 1, 'start_day': 15, 'end_day': 16}
    samples, indices = get_samples_for_timeframe(dataset, time_params)
    if not samples:
        print("ERROR: No samples found for timeframe")
        return
    print(f"Found {len(samples)} samples")
    
    # Run model on all samples to get flows
    print("Running model inference...")
    flows_per_sample = []
    with torch.no_grad():
        for sample_x, sample_y in samples:
            sample_x = sample_x.to(DEVICE).unsqueeze(0) 
            _, _, flows = model(sample_x)  # flows = [flow1, flow2, flow3]
            flows_per_sample.append(flows)
    
    print(f"Computed flows for {len(flows_per_sample)} samples")
    
    # Create animation
    output_path = os.path.join(OUTPUT_DIR, "flows_animated.gif")
    create_animated_flow_visualization(
        dataset=dataset,
        samples=samples,
        flows_per_sample=flows_per_sample,
        stride=3,
        num_frames=len(samples),
        save_path=output_path,
        interval=500,
        input_channels_to_show=["ciwc_600", "t2m", "z_1000"]
    )


if __name__ == "__main__":
    main()