"""
Simplified data loader for unified ERA5 datasets.
The new datasets combine surface and pressure level data in single files.
"""

import os
import time
import threading
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import xarray as xr
import zarr
import dask
dask.config.set(scheduler='synchronous')

# Constants
DATA_ROOT = '/users/2/ewappler/era5_data'

def worker_init(wrk_id):
    """Initialize worker with a unique seed for data loading randomization"""
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    dask.config.set(scheduler='synchronous')

def get_data_loader(params, train=True, shuffle=True):
    """
    Creates and returns a data loader for the unified ERA5 dataset.
    
    Args:
        params: Configuration parameters containing:
            - train_years/valid_years: List of years to use
            - era5_channel_input: List of ERA5 input channels to use
            - era5_channel_output: List of ERA5 output channels to use
            - region: Region to load data for (e.g., 'us_midwest')
            - local_batch_size: Batch size
            - num_data_workers: Number of worker processes for data loading
        train: Whether this is a training dataset
        shuffle: Whether to shuffle the data
        
    Returns:
        dataloader: PyTorch DataLoader
        dataset: The dataset instance
    """
    # Select years based on train/valid parameter
    years = params.train_years if train else params.valid_years
    
    # Create dataset
    dataset = UnifiedERA5Dataset(
        years=years,
        input_channels=params.era5_channel_input,
        output_channels=params.era5_channel_output,
        region=getattr(params, 'region', 'us_midwest'),
        dt=getattr(params, 'dt', 6),  # Time interval in hours
    )

    # Create subset if needed
    if getattr(params, 'is_subset', False):
        indices = np.arange(params.step_start, params.step_end)
        dataset = SubDataset(dataset, indices)

    # Prefetch factor helps with loading efficiency
    prefetch_factor = 2 if params.num_data_workers > 0 else None
    
    # Create and return the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=int(params.local_batch_size),
        num_workers=params.num_data_workers,
        shuffle=shuffle,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
        persistent_workers=params.num_data_workers > 0,
    )

    return dataloader, dataset

class SubDataset(Subset):
    """
    A subset of the UnifiedERA5Dataset that only uses a specified set of indices.
    Maintains all the metadata and functionality of the original dataset.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        # Validate indices
        dataset_len = len(self.dataset)
        assert np.max(self.indices) < dataset_len, f"Indices exceed dataset size. Max index: {np.max(self.indices)}, Dataset size: {dataset_len}"
        assert np.min(self.indices) >= 0, "Indices contain negative values."
        assert len(self.indices) > 0, "No indices provided."
        assert len(self.indices) <= dataset_len, "Too many indices for dataset size."
        assert len(self.indices) == len(np.unique(self.indices)), "Indices contain duplicates."
        self._copy_attributes()
        super().__init__(self.dataset, self.indices)
    
    def __len__(self):
        return len(self.indices)
    
    def _copy_attributes(self):
        """Copy necessary metadata from the base dataset"""
        # Core dataset attributes
        self.years = self.dataset.years
        self.input_channels = self.dataset.input_channels
        self.output_channels = self.dataset.output_channels
        self.region = self.dataset.region
        self.dt = self.dataset.dt
        
        # Coordinate information
        self.lat = self.dataset.lat
        self.lon = self.dataset.lon
        self.channels = self.dataset.channels

class UnifiedERA5Dataset(Dataset):
    """
    Dataset for loading unified ERA5 data that combines surface and pressure level variables.
    """
    def __init__(
        self, 
        years: List[int], 
        input_channels: List[str], 
        output_channels: List[str],
        region: str = 'us_midwest',
        dt: int = 6,  # Time interval in hours
    ):
        """
        Initialize the dataset.
        
        Args:
            years: List of years to load data from
            input_channels: List of ERA5 input channels to use
            output_channels: List of ERA5 output channels to use
            region: Region to load data for (e.g., 'us_midwest')
            dt: Time interval in hours
        """
        self.years = sorted(years)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.region = region
        self.dt = dt
        self._load_datasets()

    def __len__(self):
        return len(self.data.time)

    def __getitem__(self, idx):
        # Extract input and output data for selected channels at the given time index
        input_data = self.data.data.sel(channel=self.input_channels).isel(time=idx).values
        output_data = self.data.data.sel(channel=self.output_channels).isel(time=idx).values
        
        # Get timestamp
        timestamp = self.data.time.isel(time=idx).values
        
        result = {
            'input': torch.as_tensor(input_data, dtype=torch.float32), 
            'output': torch.as_tensor(output_data, dtype=torch.float32), 
            'timestamp': str(timestamp),
            'global_idx': idx,
        }
        
        return result
    
    def _load_datasets(self):
        """Load all datasets and concatenate them along time dimension"""
        print(f"Loading unified ERA5 datasets for region '{self.region}' with dt={self.dt}h...")
        
        datasets = []
        
        for year in self.years:
            # Construct file path for unified dataset
            file_path = f'{DATA_ROOT}/{self.region}/{year}_{self.region}_28.zarr'
            print(f"Loading data for year {year} from: {file_path}")
            
            # Open dataset with thread synchronizer
            synchronizer = zarr.ThreadSynchronizer()
            ds = xr.open_zarr(file_path, consolidated=False, synchronizer=synchronizer)
            datasets.append(ds)
        
        # Concatenate all datasets along the time dimension
        print("Concatenating datasets along time dimension...")
        self.data = xr.concat(datasets, dim='time')
        
        # Store coordinate information
        self.lat = self.data.latitude.values.copy()
        self.lon = self.data.longitude.values.copy()
        self.channels = self.data.channel.values.copy()
        
        # Verify that all requested channels are available
        missing_input_channels = set(self.input_channels) - set(self.channels)
        missing_output_channels = set(self.output_channels) - set(self.channels)
        
        if missing_input_channels:
            print(f"Warning: Missing input channels: {missing_input_channels}")
        if missing_output_channels:
            print(f"Warning: Missing output channels: {missing_output_channels}")
        
        print(f"Available channels: {list(self.channels)}")
        print(f"Requested input channels: {self.input_channels}")
        print(f"Requested output channels: {self.output_channels}")
        print(f"Spatial dimensions: lat={len(self.lat)}, lon={len(self.lon)}")
        print(f"Dataset loading complete. Total samples: {len(self.data.time)}")

if __name__ == '__main__':
    """Test script for the simplified data loader
    # Test with custom config file
    python data_loader.py --yaml_config config.yaml --config base --visualize
    """
    import argparse
    from utils.YParams import YParams
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the unified ERA5 data loader")
    parser.add_argument("--yaml_config", default='config.yaml', type=str)
    parser.add_argument("--config", default='base', type=str)
    parser.add_argument("--visualize", action='store_true')
    args = parser.parse_args()
    
    # Load configuration
    params = YParams(args.yaml_config, args.config)
    
    # Set test parameters
    params.local_batch_size = getattr(params, 'local_batch_size', 2)
    params.num_data_workers = 0  # Use 0 for testing
    
    print(f"Configuration:")
    print(f"  Years: {params.train_years}")
    print(f"  Region: {params.region}")
    print(f"  ERA5 input channels: {params.era5_channel_input}")
    print(f"  ERA5 output channels: {params.era5_channel_output}")
    print(f"  Time interval: {params.dt} hours")
    print(f"  Batch size: {params.local_batch_size}")
    
    # Create data loader
    start_time = time.time()
    dataloader, dataset = get_data_loader(params, train=True, shuffle=False)
    init_time = time.time() - start_time
    print(f"Initialization completed in {init_time:.2f} seconds")
    
    # Test loading some batches
    print(f"\nTesting data loading...")
    for i, batch in enumerate(dataloader):
        input_data = batch['input']
        output_data = batch['output']
        timestamp = batch['timestamp']
        
        print(f"Batch {i}: timestamp={timestamp[0]}, "
              f"input={input_data.shape}, output={output_data.shape}")
        
        # Visualize first batch if requested
        if args.visualize and i == 0:
            try:
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(15, 5))
                
                # Plot first input channel
                plt.subplot(1, 3, 1)
                plt.imshow(input_data[0, 0].numpy())
                plt.colorbar()
                plt.title(f"Input: {params.era5_channel_input[0]}")
                
                # Plot output channel
                plt.subplot(1, 3, 2)
                plt.imshow(output_data[0, 0].numpy())
                plt.colorbar()
                plt.title(f"Output: {params.era5_channel_output[0]}")
                
                # Plot second input channel if available
                if len(params.era5_channel_input) > 1:
                    plt.subplot(1, 3, 3)
                    plt.imshow(input_data[0, 1].numpy())
                    plt.colorbar()
                    plt.title(f"Input: {params.era5_channel_input[1]}")
                
                plt.tight_layout()
                viz_path = "unified_era5_test_viz.png"
                # plt.savefig(f"visualization_outputs/{viz_path}")
                plt.savefig(os.path.join('visualization_outputs', viz_path))
                print(f"Visualization saved to: {viz_path}")
                plt.close()
            except Exception as e:
                print(f"Visualization error: {e}")
        
        if i >= 2:  # Test only a few batches
            break
    
    print(f"\nTest completed successfully!")
    print(f"Dataset contains {len(dataset)} samples")
    
    # Test SubDataset functionality
    print(f"\n" + "="*50)
    print("TESTING SUBDATASET FUNCTIONALITY")
    print("="*50)
    
    # Create a SubDataset with a small subset of indices
    print(f"\nOriginal dataset length: {len(dataset)}")
    subset_indices = np.array([0, 5, 6, 8, 32])  # Select specific indices
    subdataset = SubDataset(dataset, subset_indices)
    
    print(f"SubDataset length: {len(subdataset)}")
    print(f"SubDataset indices: {subdataset.indices}")
    
    # Test that all attributes are copied correctly
    print(f"\nTesting attribute copying:")
    print(f"  Original dataset type: {type(dataset).__name__}")
    print(f"  SubDataset type: {type(subdataset).__name__}")
    print(f"  Years: {subdataset.years}")
    print(f"  Region: {subdataset.region}")
    print(f"  Input channels: {subdataset.input_channels}")
    print(f"  Output channels: {subdataset.output_channels}")
    print(f"  Latitude shape: {subdataset.lat.shape}")
    print(f"  Longitude shape: {subdataset.lon.shape}")
    print(f"  Channels: {len(subdataset.channels)} total")
    
    # Test data access
    print(f"\nTesting data access:")
    sample_0 = subdataset[0]  # This should get index 0 from the original dataset
    sample_1 = subdataset[1]  # This should get index 5 from the original dataset
    sample_2 = subdataset[2]  # This should get index 6 from the original dataset
    sample_3 = subdataset[3]  # This should get index 8 from the original dataset
    
    print(f"  SubDataset[0] timestamp: {sample_0['timestamp']}")
    print(f"  SubDataset[1] timestamp: {sample_1['timestamp']}")
    print(f"  SubDataset[2] timestamp: {sample_2['timestamp']}")
    print(f"  SubDataset[3] timestamp: {sample_3['timestamp']}")

    print(f"  SubDataset[0] input shape: {sample_0['input'].shape}")
    print(f"  SubDataset[0] output shape: {sample_0['output'].shape}")
    print(f"  SubDataset[1] input shape: {sample_1['input'].shape}")
    print(f"  SubDataset[1] output shape: {sample_1['output'].shape}")
    
    # Verify that SubDataset[1] corresponds to original dataset[5]
    original_sample_5 = dataset[5]
    print(f"  Original dataset[5] timestamp: {original_sample_5['timestamp']}")
    print(f"  Timestamps match: {sample_1['timestamp'] == original_sample_5['timestamp']}")
    
    # Test with DataLoader
    print(f"\nTesting SubDataset with DataLoader:")
    subdataset_loader = DataLoader(
        subdataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    
    for i, batch in enumerate(subdataset_loader):
        print(f"  SubDataset batch {i}: {batch['input'].shape}, timestamps: {batch['timestamp']}")
        if i >= 1:  # Test only a couple batches
            break
    
    # Test error handling
    print(f"\nTesting error handling:")
    try:
        # Try to create SubDataset with invalid indices
        invalid_indices = np.array([0, 5, len(dataset) + 1])  # Index out of bounds
        SubDataset(dataset, invalid_indices)
        print("  ERROR: Should have failed with out-of-bounds indices!")
    except AssertionError as e:
        print(f"Correctly caught out-of-bounds error: {e}")
    
    try:
        # Try to create SubDataset with duplicate indices
        duplicate_indices = np.array([0, 5, 5, 10])
        SubDataset(dataset, duplicate_indices)
        print("  ERROR: Should have failed with duplicate indices!")
    except AssertionError as e:
        print(f"Correctly caught duplicate indices error: {e}")
    
    try:
        # Try to create SubDataset with negative indices
        negative_indices = np.array([-1, 0, 5])
        SubDataset(dataset, negative_indices)
        print("  ERROR: Should have failed with negative indices!")
    except AssertionError as e:
        print(f"Correctly caught negative indices error: {e}")
    
    print(f"\nSubDataset testing completed successfully!")
    print("="*50)
