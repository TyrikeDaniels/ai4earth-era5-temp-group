"""
Simplified visualization script for ERA5 data using the data loader.
This script demonstrates how to use the data loader from data_loader.py 
to visualize ERA5 meteorological data with input/output comparison capabilities.

Available functionality:
- Compare input and output channels for specific samples
- Create animated GIFs comparing input vs output channels over time
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import calendar


CHECKPOINT_DIR = "./models/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./visualization_outputs"


def correlate_flows(flows_agg, input_data, channel_names, channel_indices=None):
    """
    Compute alignment score between learned flows and each input variable.
    
    For each variable, check if flows point toward where values are lower/changing.
    High score (+) = flows track that variable
    Low score (-) = flows avoid that variable
    
    Args:
        flows_agg: tuple (u_agg, v_agg), each shape (H, W)
        input_data: (seq_len, C, H, W) tensor or numpy
        channel_names: list of channel names (str)
        channel_indices: list of indices to compute. If None, compute all.
    
    Returns:
        scores: dict mapping channel_name -> alignment_score (float in [-1, 1])
    """
    u_agg, v_agg = flows_agg
    
    # Convert to numpy if needed
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.numpy()
    
    # Normalize flows to unit vectors
    flow_magnitude = np.sqrt(u_agg**2 + v_agg**2)
    flow_magnitude[flow_magnitude == 0] = 1e-6
    u_norm = u_agg / flow_magnitude
    v_norm = v_agg / flow_magnitude
    
    # Determine which channels to check
    if channel_indices is None:
        channel_indices = range(len(channel_names))
    
    scores = {}
    
    for idx in channel_indices:
        channel_name = channel_names[idx]
        
        # Get the LAST timestep of this channel (most recent state)
        channel_data = input_data[-1, idx]  # (H, W)
        
        # Compute gradient (direction of steepest increase)
        dy, dx = np.gradient(channel_data)
        
        # Gradient magnitude (steepness)
        grad_mag = np.sqrt(dx**2 + dy**2)
        grad_mag[grad_mag == 0] = 1e-6
        
        # Normalize gradient to unit vectors
        # Negative sign: we care if flows point toward LOWER values
        dx_norm = -dx / grad_mag  
        dy_norm = -dy / grad_mag
        
        # Dot product: cosine similarity
        # At each pixel, how aligned is (u, v) with the gradient direction?
        alignment = (u_norm * dx_norm + v_norm * dy_norm).mean()
        
        scores[channel_name] = alignment
    
    return scores
 
 
def print_scores(scores, top_n=None):
    """
    Pretty-print alignment scores, ranked by absolute value.
    
    Args:
        scores: dict from correlate_flows_with_all_variables()
        top_n: Print only top N. If None, print all.
    """
    # Sort by absolute alignment (strongest correlation)
    sorted_scores = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
    
    if top_n:
        sorted_scores = sorted_scores[:top_n]
    
    print("\n" + "-"*60)
    print(f"{'Channel':<20} {'Cos simil':<12} {'Interpretation'}")
    print("-"*60)          
    
    for channel, score in sorted_scores:
        if score > 0.3:
            interp = "Strong tracking"
        elif score > 0.1:
            interp = "Weak tracking"
        elif score > -0.1:
            interp = "No correlation"
        elif score > -0.3:
            interp = "Weak avoidance"
        else:
            interp = "Strong avoidance"
        print(f"{channel:<20} {score:>+.4f}       {interp}")
    
    print(""*60)
    print("Interpretation:")
    print("  +1.0 = Flows point toward lower values (perfect tracking)")
    print("   0.0 = No correlation")
    print("  -1.0 = Flows point away from lower values")
    print("-"*60 + "\n")


def get_timeframe(dataset, time_params):
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


def animated_visual(
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

 
if __name__ == "__main__":
    pass
 