"""
Simplified visualization script for ERA5 data using the data loader.
This script demonstrates how to use the data loader from data_loader.py 
to visualize ERA5 meteorological data with input/output comparison capabilities.

Available functionality:
- Compare input and output channels for specific samples
- Create animated GIFs comparing input vs output channels over time
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import torch
from IPython.display import Image, display
import re
from datetime import datetime

# Import our data loader
from data_loader import get_data_loader
from utils.YParams import YParams

# Constants
OUTPUT_IMGS_DIR = './visualization_outputs'

def format_timestamp(timestamp_str):
    """
    Convert timestamp string from format like '20200126000000' to readable format
    
    Args:
        timestamp_str: Timestamp string in format 'YYYYMMDDHHMISS'
        
    Returns:
        str: Formatted datetime string like '2020-01-26 00:00 UTC'
    """
    try:
        # Parse the timestamp string
        dt = datetime.strptime(str(timestamp_str), '%Y%m%d%H%M%S')
        # Format as readable string
        return dt.strftime('%Y-%m-%d %H:%M UTC')
    except (ValueError, TypeError):
        # If parsing fails, return the original string
        return str(timestamp_str)

def print_dataset_info(dataset, dataloader):
    """
    Print information about the dataset and data loader
    
    Args:
        dataset: The ERA5 dataset instance (could be Subset)
        dataloader: PyTorch DataLoader instance
    """
    print("\n========== DATASET INFORMATION ==========")
    print(f"Total samples: {len(dataset)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Get a sample batch to understand data structure
    sample_batch = next(iter(dataloader))
    input_data = sample_batch['input']
    output_data = sample_batch['output']
    
    print(f"\nData shapes:")
    print(f"  Input: {input_data.shape}")  # [batch, channels, lat, lon]
    print(f"  Output: {output_data.shape}") # [batch, channels, lat, lon]
    
    print(f"\nCoordinate information:")
    print(f"  Latitude range: {dataset.lat.min():.2f} to {dataset.lat.max():.2f}")
    print(f"  Longitude range: {dataset.lon.min():.2f} to {dataset.lon.max():.2f}")
    print(f"  Spatial dimensions: {len(dataset.lat)} x {len(dataset.lon)}")
    
    print(f"\nAvailable channels:")
    print(f"  Input channels ({len(dataset.input_channels)}): {dataset.input_channels}")
    print(f"  Output channels ({len(dataset.output_channels)}): {dataset.output_channels}")
    
    print(f"\nTime information:")
    print(f"  First timestamp: {sample_batch['timestamp'][0]}")
    print(f"  Years covered: {dataset.years}")
    print("==========================================\n")

def extract_variable_and_pressure(channel):
    """
    Extract variable and pressure level from a channel name like 't_1000'
    
    Args:
        channel: Name of the channel (e.g., 't_1000')
        
    Returns:
        tuple: (variable, pressure_level)
    """
    match = re.match(r'([a-z_]+)_?(\d+)?', channel)
    if match:
        var_name = match.group(1)
        pressure = int(match.group(2)) if match.group(2) else None
        return var_name, pressure
    return channel, None

def get_cmap_and_range(channel, data_array):
    """
    Get appropriate colormap and value range for a given channel
    
    Args:
        channel: Channel name (e.g., 't2m', 'u10')
        data_array: Numpy array or torch tensor with the data
        
    Returns:
        tuple: (cmap, vmin, vmax, norm)
    """
    variable, pressure = extract_variable_and_pressure(channel)
    
    # Handle both numpy arrays and torch tensors
    if hasattr(data_array, 'numpy'):
        # It's a torch tensor
        data_min = data_array.min().item()
        data_max = data_array.max().item()
    else:
        # It's a numpy array
        data_min = float(np.min(data_array))
        data_max = float(np.max(data_array))
    
    if variable in ['t2m', 'sst', 'skt', 't']:
        cmap = 'RdBu_r'
        vmin, vmax = data_min, data_max
        norm = None
        print(f"Temperature range for {channel}: {vmin:.2f} to {vmax:.2f}")
    
    elif variable in ['u10', 'v10', 'u', 'v', 'w', 'vo', 'd']:
        cmap = 'coolwarm'
        vmin, vmax = data_min, data_max
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        print(f"Wind range for {channel}: {vmin:.2f} to {vmax:.2f}")
    
    elif variable in ['avg_tprate', 'mtpr', 'crr', 'lsrr', 'msr'] or variable in ['clwc', 'ciwc', 'q']:
        cmap = 'Blues'
        vmin = 0
        vmax = data_max
        norm = None
        print(f"Precipitation range for {channel}: {vmin:.5f} to {vmax:.5f}")
    
    elif variable in ['lsm', 'siconc']:
        cmap = 'terrain'
        vmin, vmax = 0, 1
        norm = None
        print(f"Land/sea mask range for {channel}: {vmin:.2f} to {vmax:.2f}")
    
    elif variable in ['z']:
        cmap = 'terrain'
        vmin, vmax = data_min, data_max
        norm = None
        print(f"Geopotential range for {channel}: {vmin:.0f} to {vmax:.0f}")
    
    else:
        cmap = 'viridis'
        vmin, vmax = data_min, data_max
        norm = None
        print(f"Data range for {channel}: {vmin:.5f} to {vmax:.5f}")

    return cmap, vmin, vmax, norm

def compare_input_output(dataset, dataloader, sample_idx=0, save_path=None, input_channels_to_show=None):
    """
    Compare input and output channels for a specific sample
    
    Args:
        dataset: The ERA5 dataset instance (could be Subset)
        dataloader: PyTorch DataLoader instance
        sample_idx: Index of the sample to visualize
        save_path: Path to save the visualization
        input_channels_to_show: List of input channel names to show (max 3). If None, shows first 3 channels.
    """
    
    # Get the sample
    sample = dataset[sample_idx]
    input_data = sample['input']
    output_data = sample['output']
    timestamp = sample['timestamp']
    
    # Determine what to compare
    n_input = len(dataset.input_channels)
    n_output = len(dataset.output_channels)
    
    # Determine which input channels to show
    if input_channels_to_show is not None:
        # Validate requested channels exist
        available_channels = set(dataset.input_channels)
        requested_channels = set(input_channels_to_show)
        invalid_channels = requested_channels - available_channels
        
        if invalid_channels:
            print(f"Warning: Requested channels not found: {invalid_channels}")
            print(f"Available input channels: {dataset.input_channels}")
        
        # Filter to valid channels and limit to 3
        valid_channels = [ch for ch in input_channels_to_show if ch in available_channels]
        channels_to_show = min(3, len(valid_channels))
        input_channel_indices = [dataset.input_channels.index(ch) for ch in valid_channels[:channels_to_show]]
        input_channel_names = valid_channels[:channels_to_show]
        
        print(f"Showing input channels: {input_channel_names}")
    else:
        # Show first few input channels as before
        channels_to_show = min(3, n_input)
        input_channel_indices = list(range(channels_to_show))
        input_channel_names = dataset.input_channels[:channels_to_show]
    
    total_plots = len(input_channel_indices) + n_output
    
    cols = min(total_plots, 4)
    rows = (total_plots + cols - 1) // cols
    
    fig = plt.figure(figsize=(5*cols, 4*rows))
    
    lats = dataset.lat
    lons = dataset.lon
    
    plot_idx = 1
    
    # Plot input channels
    for i, channel_idx in enumerate(input_channel_indices):
        ax = plt.subplot(rows, cols, plot_idx, projection=ccrs.PlateCarree())
        
        channel = input_channel_names[i]
        data = input_data[channel_idx].numpy()
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        
        # Get colormap
        cmap, vmin, vmax, norm = get_cmap_and_range(channel, data)
        
        # Plot
        if norm is not None:
            img = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        else:
            img = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        plt.colorbar(img, ax=ax, shrink=0.8, pad=0.05)
        ax.set_title(f"Input: {channel}")
        
        plot_idx += 1
    
    # Plot output channels
    for i in range(n_output):
        ax = plt.subplot(rows, cols, plot_idx, projection=ccrs.PlateCarree())
        
        channel = dataset.output_channels[i]
        data = output_data[i].numpy()
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        
        # Get colormap
        cmap, vmin, vmax, norm = get_cmap_and_range(channel, data)
        
        # Plot
        if norm is not None:
            img = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        else:
            img = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        plt.colorbar(img, ax=ax, shrink=0.8, pad=0.05)
        ax.set_title(f"Output: {channel}")
        
        plot_idx += 1
    
    plt.suptitle(f"Input vs Output Comparison - {format_timestamp(timestamp)}\nSample Index: {sample_idx}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    plt.show()


def create_input_output_animation_gif(dataset, dataloader, start_idx=0, num_frames=12, save_path=None, interval=800, input_channels_to_show=None):
    """
    Create an animated GIF comparing input and output channels over time
    Shows the evolution of input channels vs output channels in a single animation
    
    Args:
        dataset: The ERA5 dataset instance (could be Subset)
        dataloader: PyTorch DataLoader instance
        start_idx: Starting sample index
        num_frames: Number of frames for the animation
        save_path: Path to save the GIF animation
        interval: Time between frames in milliseconds
        input_channels_to_show: List of input channel names to show (max 3). If None, shows first 3 channels.
    """
    
    # Collect data for the animation
    timestamps = []
    input_data_arrays = []
    output_data_arrays = []
    
    end_idx = min(start_idx + num_frames, len(dataset))
    actual_frames = end_idx - start_idx
    
    print(f"Creating input vs output animation with {actual_frames} frames from index {start_idx} to {end_idx-1}...")
    
    for idx in range(start_idx, end_idx):
        sample = dataset[idx]
        timestamps.append(sample['timestamp'])
        input_data_arrays.append(sample['input'].numpy())  # [channels, lat, lon]
        output_data_arrays.append(sample['output'].numpy())  # [channels, lat, lon]
    
    if not input_data_arrays:
        print("No data collected for animation!")
        return
    
    # Determine which input channels to show
    n_input = len(dataset.input_channels)
    n_output = len(dataset.output_channels)
    
    if input_channels_to_show is not None:
        # Validate requested channels exist
        available_channels = set(dataset.input_channels)
        requested_channels = set(input_channels_to_show)
        invalid_channels = requested_channels - available_channels
        
        if invalid_channels:
            print(f"Warning: Requested channels not found: {invalid_channels}")
            print(f"Available input channels: {dataset.input_channels}")
        
        # Filter to valid channels and limit to 3
        valid_channels = [ch for ch in input_channels_to_show if ch in available_channels]
        channels_to_show = min(3, len(valid_channels))
        input_channel_indices = [dataset.input_channels.index(ch) for ch in valid_channels[:channels_to_show]]
        input_channel_names = valid_channels[:channels_to_show]
        
        print(f"Animating input channels: {input_channel_names}")
    else:
        # Show first few input channels as before
        channels_to_show = min(3, n_input)
        input_channel_indices = list(range(channels_to_show))
        input_channel_names = dataset.input_channels[:channels_to_show]
    
    total_plots = len(input_channel_indices) + n_output
    
    cols = min(total_plots, 4)
    rows = (total_plots + cols - 1) // cols
    
    # Get coordinates
    lats = dataset.lat
    lons = dataset.lon
    
    # Pre-calculate color ranges for consistency across frames
    input_cmaps = []
    output_cmaps = []
    
    # Stack all data to get global ranges
    all_input_data = np.stack(input_data_arrays, axis=0)  # [time, channels, lat, lon]
    all_output_data = np.stack(output_data_arrays, axis=0)  # [time, channels, lat, lon]
    
    for i, channel_idx in enumerate(input_channel_indices):
        channel = input_channel_names[i]
        data_array = all_input_data[:, channel_idx, :, :]
        cmap, vmin, vmax, norm = get_cmap_and_range(channel, data_array)
        input_cmaps.append((cmap, vmin, vmax, norm))
    
    for i in range(n_output):
        channel = dataset.output_channels[i]
        data_array = all_output_data[:, i, :, :]
        cmap, vmin, vmax, norm = get_cmap_and_range(channel, data_array)
        output_cmaps.append((cmap, vmin, vmax, norm))
    
    # Set up the figure
    fig = plt.figure(figsize=(5*cols, 4*rows))
    axes = []
    images = []
    titles = []
    
    plot_idx = 1
    
    # Create input channel subplots
    for i, channel_idx in enumerate(input_channel_indices):
        ax = plt.subplot(rows, cols, plot_idx, projection=ccrs.PlateCarree())
        axes.append(ax)
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
        
        # Initialize with first frame
        channel = input_channel_names[i]
        data = input_data_arrays[0][channel_idx]
        cmap, vmin, vmax, norm = input_cmaps[i]
        
        if norm is not None:
            img = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        else:
            img = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        images.append(img)
        plt.colorbar(img, ax=ax, shrink=0.8, pad=0.05)
        
        title = ax.set_title(f"Input: {channel}")
        titles.append(title)
        
        # Add grid
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        plot_idx += 1
    
    # Create output channel subplots
    for i in range(n_output):
        ax = plt.subplot(rows, cols, plot_idx, projection=ccrs.PlateCarree())
        axes.append(ax)
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)
        
        # Initialize with first frame
        channel = dataset.output_channels[i]
        data = output_data_arrays[0][i]
        cmap, vmin, vmax, norm = output_cmaps[i]
        
        if norm is not None:
            img = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        else:
            img = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        images.append(img)
        plt.colorbar(img, ax=ax, shrink=0.8, pad=0.05)
        
        title = ax.set_title(f"Output: {channel}")
        titles.append(title)
        
        # Add grid
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        plot_idx += 1
    
    # Adjust layout FIRST before creating animation and title
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])  # Leave space for suptitle and bottom text
    
    # # Main title that will be updated - positioned higher and with larger font
    # main_title = fig.suptitle(f"Input vs Output Comparison - {format_timestamp(timestamps[0])}", 
    #                          fontsize=16, y=0.96, fontweight='bold')
    
    # Add timestamp at bottom of figure
    timestamp_text = fig.text(0.5, 0.02, f"Time: {format_timestamp(timestamps[0])}", 
                             ha='center', fontsize=14, fontweight='bold', 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    def animate(frame):
        """Animation function called for each frame"""
        updated_artists = []
        
        # Update input channels
        for i, channel_idx in enumerate(input_channel_indices):
            data = input_data_arrays[frame][channel_idx]
            images[i].set_array(data.ravel())
            updated_artists.append(images[i])
        
        # Update output channels
        for i in range(n_output):
            data = output_data_arrays[frame][i]
            images[len(input_channel_indices) + i].set_array(data.ravel())
            updated_artists.append(images[len(input_channel_indices) + i])
        
        # # Update main title with timestamp
        # main_title.set_text(f"Input vs Output Comparison - {format_timestamp(timestamps[frame])}")
        # updated_artists.append(main_title)
        
        # Update timestamp text at bottom
        timestamp_text.set_text(f"Time: {format_timestamp(timestamps[frame])}")
        updated_artists.append(timestamp_text)
        
        return updated_artists
    
    # Create animation
    print(f"Creating input vs output animation with {actual_frames} frames, interval={interval}ms...")
    
    anim = animation.FuncAnimation(fig, animate, frames=actual_frames, 
                                   interval=interval, blit=False, repeat=True)
    
    # Save as GIF if path provided
    if save_path:
        print(f"Saving input vs output animation to {save_path}...")
        # Use pillow writer for GIF with better settings for text rendering
        writer = animation.PillowWriter(fps=1000/interval, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    
    plt.show()
    
    return anim

if __name__ == '__main__':
    """Main function to run visualization examples
    # Compare specific input channels with output channels
    python using_data_loader.py --config base --compare --input_channels t2m u10 clwc_800

    # Create animation with specific input channels
    python using_data_loader.py --config base --animate_comparison --input_channels t2m u10 clwc_800 --num_frames 8
    """
    parser = argparse.ArgumentParser(description="Visualize ERA5 data using the data loader")
    parser.add_argument("--yaml_config", default='config.yaml', type=str, help="Path to YAML config file")
    parser.add_argument("--config", default='base', type=str, help="Configuration name to use")
    parser.add_argument("--sample_idx", default=0, type=int, help="Sample index to visualize")
    parser.add_argument("--compare", action='store_true', help="Compare input and output channels")
    parser.add_argument("--animate_comparison", action='store_true', help="Create animated GIF comparing input vs output")
    parser.add_argument("--num_frames", default=12, type=int, help="Number of frames for animation")
    parser.add_argument("--interval", default=500, type=int, help="Time between frames in milliseconds")
    parser.add_argument("--input_channels", nargs='*', type=str, help="Specific input channel names to visualize (max 3). Example: --input_channels t2m u10 v10")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(OUTPUT_IMGS_DIR, exist_ok=True)
    output_dir = OUTPUT_IMGS_DIR
    print(f"Output directory: {output_dir}")
    
    # Load configuration and create data loader
    print("Loading configuration and creating data loader...")
    params = YParams(args.yaml_config, args.config)
    
    # Set parameters for visualization
    params.local_batch_size = 1
    params.num_data_workers = 0
    params.shuffle = False
    
    # Create data loader
    start_time = time.time()
    dataloader, dataset = get_data_loader(params, train=True, shuffle=False)
    init_time = time.time() - start_time
    print(f"Data loader initialized in {init_time:.2f} seconds")
    
    # Print dataset information
    print_dataset_info(dataset, dataloader)
    
    # Process input channels argument
    input_channels_to_show = None
    if args.input_channels:
        input_channels_to_show = args.input_channels[:3]  # Limit to 3 channels
        print(f"User requested input channels: {input_channels_to_show}")
    
    if args.compare:
        # Input vs output comparison
        print("Creating input vs output comparison...")
        save_path = os.path.join(output_dir, f"input_output_comparison_{args.sample_idx}.png")
        compare_input_output(dataset, dataloader, sample_idx=args.sample_idx, save_path=save_path, 
                           input_channels_to_show=input_channels_to_show)
    
    if args.animate_comparison:
        # Animated input vs output comparison
        print("Creating animated input vs output comparison...")
        save_path = os.path.join(output_dir, f"animation_input_output_{args.num_frames}frames.gif")
        create_input_output_animation_gif(dataset, dataloader, start_idx=args.sample_idx, 
                                        num_frames=args.num_frames, save_path=save_path, interval=args.interval,
                                        input_channels_to_show=input_channels_to_show)
    
    print(f"\nVisualization complete! Check outputs in: {output_dir}")