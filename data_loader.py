"""
Simplified data loader for unified ERA5 datasets.
The new datasets combine surface and pressure level data in single files.
"""

import os
import time
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


def get_data_loader(params, train=True, shuffle=True, norm_stats=None):
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
        norm_stats: Precomputed mean/std stats (pass train_dataset.norm_stats
                    when building the validation loader, so it reuses the
                    training set's statistics instead of computing its own)

    Returns:
        dataloader: PyTorch DataLoader
        dataset: The dataset instance
    """
    years = params.train_years if train else params.valid_years

    dataset = UnifiedERA5Dataset(
        years=years,
        input_channels=params.era5_channel_input,
        output_channels=params.era5_channel_output,
        region=getattr(params, 'region', 'us_midwest'),
        dt=getattr(params, 'dt', 6),
        normalize=True,
        norm_stats=norm_stats,
    )

    if getattr(params, 'is_subset', False):
        indices = np.arange(params.step_start, params.step_end)
        dataset = SubDataset(dataset, indices)

    prefetch_factor = 2 if params.num_data_workers > 0 else None

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
        self.years = self.dataset.years
        self.input_channels = self.dataset.input_channels
        self.output_channels = self.dataset.output_channels
        self.region = self.dataset.region
        self.dt = self.dataset.dt
        self.lat = self.dataset.lat
        self.lon = self.dataset.lon
        self.channels = self.dataset.channels
        self.normalize = self.dataset.normalize
        self.norm_stats = getattr(self.dataset, 'norm_stats', None)


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
        dt: int = 6,
        normalize: bool = True,
        norm_stats: dict = None,
    ):
        """
        Initialize the dataset.

        Args:
            years: List of years to load data from
            input_channels: List of ERA5 input channels to use
            output_channels: List of ERA5 output channels to use
            region: Region to load data for (e.g., 'us_midwest')
            dt: Time interval in hours
            normalize: Whether to normalize channel values
            norm_stats: Precomputed mean/std stats (pass training stats for validation)
        """
        self.years = sorted(years)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.region = region
        self.dt = dt
        self.normalize = normalize
        self._load_datasets()

        if self.normalize:
            if norm_stats is not None:
                self.norm_stats = norm_stats
            else:
                self.norm_stats = self._compute_norm_stats()
        else:
            self.norm_stats = None

    def __len__(self):
        return len(self.data.time)

    def _compute_norm_stats(self):
        """Compute per-channel mean/std across all timestamps in this dataset."""
        print("Computing normalization statistics from training data...")
        all_channels = list(self.input_channels) + list(self.output_channels)
        stats = {}
        for ch in all_channels:
            values = self.data.data.sel(channel=ch).values
            stats[ch] = {
                'mean': float(np.nanmean(values)),
                'std': float(np.nanstd(values)) + 1e-8,
            }
            print(f"  {ch}: mean={stats[ch]['mean']:.4f}, std={stats[ch]['std']:.4f}")
        return stats

    def __getitem__(self, idx):
        # Extract input and output data for selected channels at the given time index
        input_data = self.data.data.sel(channel=self.input_channels).isel(time=idx).values
        output_data = self.data.data.sel(channel=self.output_channels).isel(time=idx).values

        if self.normalize:
            input_data = input_data.copy()
            for i, ch in enumerate(self.input_channels):
                mean = self.norm_stats[ch]['mean']
                std = self.norm_stats[ch]['std']
                input_data[i] = (input_data[i] - mean) / std

            output_data = output_data.copy()
            for j, ch in enumerate(self.output_channels):
                mean = self.norm_stats[ch]['mean']
                std = self.norm_stats[ch]['std']
                output_data[j] = (output_data[j] - mean) / std

        # Replace any remaining NaN (e.g. from sst over land) with 0 after normalization
        input_data = np.nan_to_num(input_data, nan=0.0)
        output_data = np.nan_to_num(output_data, nan=0.0)

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
            file_path = f'{DATA_ROOT}/{self.region}/{year}_{self.region}_28.zarr'
            print(f"Loading data for year {year} from: {file_path}")

            synchronizer = zarr.ThreadSynchronizer()
            ds = xr.open_zarr(file_path, consolidated=False, synchronizer=synchronizer)
            datasets.append(ds)

        print("Concatenating datasets along time dimension...")
        self.data = xr.concat(datasets, dim='time')

        self.lat = self.data.latitude.values.copy()
        self.lon = self.data.longitude.values.copy()
        self.channels = self.data.channel.values.copy()

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
    python data_loader.py --yaml_config config.yaml --config base --visualize
    """
    import argparse
    from utils.YParams import YParams

    parser = argparse.ArgumentParser(description="Test the unified ERA5 data loader")
    parser.add_argument("--yaml_config", default='config.yaml', type=str)
    parser.add_argument("--config", default='base', type=str)
    parser.add_argument("--visualize", action='store_true')
    args = parser.parse_args()

    params = YParams(args.yaml_config, args.config)

    params.local_batch_size = getattr(params, 'local_batch_size', 2)
    params.num_data_workers = 0

    print(f"Configuration:")
    print(f"  Years: {params.train_years}")
    print(f"  Region: {params.region}")
    print(f"  ERA5 input channels: {params.era5_channel_input}")
    print(f"  ERA5 output channels: {params.era5_channel_output}")
    print(f"  Time interval: {params.dt} hours")
    print(f"  Batch size: {params.local_batch_size}")

    start_time = time.time()
    dataloader, dataset = get_data_loader(params, train=True, shuffle=False)
    init_time = time.time() - start_time
    print(f"Initialization completed in {init_time:.2f} seconds")

    print(f"\nTesting data loading...")
    for i, batch in enumerate(dataloader):
        input_data = batch['input']
        output_data = batch['output']
        timestamp = batch['timestamp']

        print(f"Batch {i}: timestamp={timestamp[0]}, "
              f"input={input_data.shape}, output={output_data.shape}")

        if args.visualize and i == 0:
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(input_data[0, 0].numpy())
                plt.colorbar()
                plt.title(f"Input: {params.era5_channel_input[0]}")

                plt.subplot(1, 3, 2)
                plt.imshow(output_data[0, 0].numpy())
                plt.colorbar()
                plt.title(f"Output: {params.era5_channel_output[0]}")

                if len(params.era5_channel_input) > 1:
                    plt.subplot(1, 3, 3)
                    plt.imshow(input_data[0, 1].numpy())
                    plt.colorbar()
                    plt.title(f"Input: {params.era5_channel_input[1]}")

                plt.tight_layout()
                viz_path = "unified_era5_test_viz.png"
                plt.savefig(os.path.join('visualization_outputs', viz_path))
                print(f"Visualization saved to: {viz_path}")
                plt.close()
            except Exception as e:
                print(f"Visualization error: {e}")

        if i >= 2:
            break

    print(f"\nTest completed successfully!")
    print(f"Dataset contains {len(dataset)} samples")
