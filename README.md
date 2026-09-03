# Precipitation Nowcasting with ERA5 Data

## Overview

This project develops spatio-temporal forecasting models for short-term precipitation prediction using ERA5 weather data. We compare two specialized RNN-based architectures integrated into a UNet framework: **TrajGRU** (trajectory-aware GRU) and **ConvLSTM**. The key finding: TrajGRU better captures learned motion patterns in atmospheric data and achieves superior RMSE performance.

## Problem

Precipitation nowcasting—predicting rainfall in the near future (minutes to hours ahead)—is critical for weather forecasting and hazard mitigation. Deep learning approaches have shown promise, but the choice of spatio-temporal architecture matters significantly. This project investigates which recurrent unit best models the dynamics of atmospheric flow.

## Approach

### Architecture

Both models use a **UNet encoder-decoder structure** with spatio-temporal recurrent units at each level:

**UNet + TrajGRU:**
- Encoder: Downsampling blocks with TrajGRU units (trajectory-aware gating)
- Decoder: Upsampling blocks with TrajGRU units
- Hidden channels: [16, 32, 64]
- Output: Binary rain detection logits + precipitation intensity predictions

**UNet + ConvLSTM:**
- Same structure, but standard ConvLSTM units replace TrajGRU
- Enables direct architectural comparison with controlled variables

### Key Insight: TrajGRU vs ConvLSTM

**TrajGRU** explicitly models spatial transitions through trajectory-aware convolutions:
- Learns directional flow patterns in data (wind direction, precipitation movement)
- More efficient at capturing moving features in atmospheric sequences
- Results in better generalization on weather data

**ConvLSTM** uses traditional convolutional gates:
- Standard baseline for spatio-temporal modeling
- Effective but less specialized for motion-centric patterns

### Training Setup

- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Loss**: Combined BCE (rain detection) + MSE (intensity regression)
- **Scheduler**: Linear warmup (2 epochs) → Cosine annealing decay
- **Evaluation**: k-fold cross-validation with precision, recall, MCC
- **Metrics**: RMSE, BCE loss, MSE loss, Matthews Correlation Coefficient

## Results

TrajGRU outperformed ConvLSTM across all lead times (dt=1,2,3,4 hours):

| Metric | TrajGRU | ConvLSTM | Improvement |
|--------|---------|----------|-------------|
| Validation RMSE | Lower | Higher | TrajGRU captures motion better |
| MCC | Higher | Lower | Better precipitation detection |
| Loss Convergence | Faster | Slower | TrajGRU learns more efficiently |

### Visualizations

Quiver plot visualizations show that TrajGRU learns meaningful directional patterns:

![Learned atmospheric flow patterns](flows_animated.gif)

- **Left**: Input geopotential height at 600mb with learned flow vectors
- **Middle**: Input geopotential height at 1000mb with learned flow vectors  
- **Right**: Model output (predicted precipitation/temperature)

The smooth, coherent flow patterns indicate TrajGRU successfully captures wind-driven weather dynamics.

## Project Structure

```
├── train_convlstm.py           # Training script for ConvLSTM baseline
├── train_trajgru.py            # Training script for TrajGRU (main model)
├── models/
│   ├── convlstm_unet.py        # UNet with ConvLSTM blocks
│   └── trajgru_unet.py         # UNet with TrajGRU blocks
├── utils/
│   ├── train_utils.py          # Loss computation, evaluation
│   ├── data_loader.py          # ERA5 data loading pipeline
│   ├── mcc.py                  # Matthews correlation coefficient
│   └── configs.py              # Configuration for different lead times
└── models/checkpoints/         # Saved model weights
```

## Usage

### Requirements

```
torch
torchvision
numpy
pandas
xarray
netCDF4  # for ERA5 data
```

### Training TrajGRU Model

```bash
python train_trajgru.py
```

This trains TrajGRU-UNet for all lead times (dt=1,2,3,4 hours) and saves:
- Best model (lowest validation loss): `models/checkpoints/best_model_b_{dt}.pt`
- Latest checkpoint (for resuming): `models/checkpoints/latest_model_b_{dt}.pt`

### Training ConvLSTM Baseline

```bash
python train_convlstm.py
```

Trains ConvLSTM-UNet for comparison.

### Resume Training

Set `resume=True` in the training script to continue from the latest checkpoint.

## Data

**Source**: ERA5 (ECMWF Reanalysis v5)

**Input Variables**: 
- Geopotential height (multiple pressure levels)
- Temperature
- Wind components (u, v)
- Relative humidity
- Precipitation

**Output**:
- Binary precipitation occurrence (rain/no rain)
- Precipitation intensity (continuous value)

**Preprocessing**:
- Log normalization of precipitation (handles skewed distribution)
- Standardization of input features
- Train/validation split with temporal consistency

## Training Dynamics

### Loss Convergence
Both models use combined loss: `loss = α·BCE + β·MSE`

Typical training progression:
- **Epoch 0-5**: Rapid loss decrease (learning rate warmup)
- **Epoch 5-30**: Steady improvement via cosine annealing decay
- **Best validation**: Usually epoch 20-28

### Checkpointing
- Saves "latest" checkpoint every epoch (allows resuming)
- Saves "best" checkpoint only when validation improves
- Prevents overfitting by monitoring validation loss

## Key Findings

1. **TrajGRU is specialized for weather data**: The trajectory-aware mechanism explicitly models moving features, making it ideal for atmospheric patterns.

2. **Spatio-temporal architecture matters**: The gap between TrajGRU and ConvLSTM is not marginal—it demonstrates the importance of tailored architectures for domain-specific data.

3. **Motion patterns are learnable**: Quiver visualizations confirm the model learns physically meaningful flow patterns, not random features.

## Future Work

- Extend to longer lead times (6+ hours ahead)
- Add attention mechanisms for selective feature focus
- Incorporate satellite imagery for additional context
- Ensemble methods combining multiple architectures
- Real-time deployment pipeline

## References

- TrajGRU: [Shi et al., NeurIPS 2017] - "Deep Learning for Precipitation Nowcasting"
- ConvLSTM: [Shi et al., NeurIPS 2015] - "Convolutional LSTM Networks"
- ERA5 Data: [Hersbach et al., 2020] - ECMWF Reanalysis

## Citation

If you use this work, please cite:

```
@project{nowcasting2026,
  title={Precipitation Nowcasting with Spatio-Temporal Deep Learning},
  author={Tyrike Daniels},
  year={2026}
}
```
