# WSTATT Crop Classification Notebooks

This repository contains three Jupyter notebooks for exploring satellite and weather data and running crop classification experiments using STATT and WSTATT models.

Preprocessed data is present at: https://drive.google.com/drive/folders/1HSUD74s6N7xoIyRlrflxsV5nZ4mnEFTX?usp=drive_link

## Repository Contents

```text
.
├── Explore_data.ipynb
├── Explore_STATT.ipynb
└── Explore_WSTATT.ipynb
```

## Overview

This repository is organized into three notebooks:

1. `Explore_data.ipynb`  
   Used for exploring the dataset, including satellite images, crop labels, eroded labels, and weather variables.

2. `Explore_STATT.ipynb`  
   Implements and tests the STATT model, which uses multi-temporal satellite imagery for crop classification.

3. `Explore_WSTATT.ipynb`  
   Implements and tests the WSTATT model, which extends STATT by adding weather information along with satellite imagery.

## Notebook Descriptions

### Explore_data.ipynb

This notebook is used for data exploration and visualization.

It includes:

- Loading satellite image arrays
- Checking data shapes and dimensions
- Visualizing Sentinel-2 imagery
- Visualizing crop label maps
- Comparing satellite imagery with crop labels
- Exploring eroded labels
- Exploring weather variables over time

Run this notebook first to understand the dataset format.

### Explore_STATT.ipynb

This notebook implements the STATT model.

STATT stands for Spatio-Temporal Attention Network. It uses satellite image sequences collected over time to perform crop classification.

The notebook includes:

- Loading satellite and label data
- Creating image patches
- Defining the STATT model architecture
- Training the model
- Saving and loading model weights
- Testing the model
- Evaluating classification performance

### Explore_WSTATT.ipynb

This notebook implements the WSTATT model.

WSTATT extends STATT by including weather data along with satellite imagery.

The notebook includes:

- Loading satellite, weather, and label data
- Creating satellite and weather input patches
- Defining the WSTATT model architecture
- Training the model
- Saving and loading model weights
- Testing the model
- Evaluating classification performance

## Dataset

The notebooks expect the dataset to contain satellite imagery, weather data, and crop labels.

The expected dataset folders are:

```text
CalCrop_Data/
├── Satellite/
├── Weather/
├── Label/
├── Label_Eroded/
└── Model/
```

The notebooks are currently written for Google Colab and use Google Drive paths such as:

```python
/content/drive/MyDrive/CalCrop_Data/
```

If running locally or using a different folder structure, update the paths in the notebooks.

## Data Format

Satellite data is stored as NumPy arrays with the format:

```text
[timesteps, channels, height, width]
```

The notebooks use Sentinel-2 satellite bands over multiple time steps.

Label data is stored as 2D NumPy arrays with the format:

```text
[height, width]
```

Weather data is used in the WSTATT notebook and includes variables such as:

```text
dayl, prcp, srad, swe, tmax, tmin, vp
```

## Requirements

The notebooks are designed to run in Google Colab.

Main libraries used:

```text
numpy
matplotlib
torch
scikit-learn
google.colab
```

To install the required Python packages locally, use:

```bash
pip install numpy matplotlib torch scikit-learn
```

If running outside Google Colab, remove or replace the following lines:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then update the dataset paths manually.

## How to Run

Recommended order:

1. Open `Explore_data.ipynb`
2. Mount Google Drive or update the dataset path
3. Run the cells to inspect the dataset
4. Open `Explore_STATT.ipynb`
5. Train and evaluate the STATT model
6. Open `Explore_WSTATT.ipynb`
7. Train and evaluate the WSTATT model

## Model Output

The training notebooks save and load model weights using a file such as:

```text
Model.pt
```

Make sure the model path is correct before loading saved weights.

## Crop Classes

The notebooks use multiple crop and land-cover classes, including crops such as corn, cotton, rice, wheat, tomatoes, grapes, almonds, pistachio, alfalfa, and others.

Unknown or ignored pixels are represented using:

```python
unknown_class = 100
```

These pixels are ignored during training and evaluation.

## Notes

- These notebooks are intended for exploration and experimentation.
- GPU acceleration is recommended for model training.
- STATT uses satellite imagery only.
- WSTATT uses both satellite imagery and weather data.
- Dataset paths may need to be modified before running the notebooks.
- The notebooks use patch-based training for crop classification.

## Acknowledgement

This repository is prepared for experiments with STATT and WSTATT-style spatio-temporal crop classification using satellite imagery and weather data.

## Citation

If you use this repository, please cite the following papers:

```bibtex
@inproceedings{ravirathinam2024wstatt,
  title={Combining Satellite and Weather Data for Crop Type Mapping: An Inverse Modelling Approach},
  author={Ravirathinam, Praveen and Ghosh, Rahul and Khandelwal, Ankush and Jia, Xiaowei and Mulla, David and Kumar, Vipin},
  booktitle={Proceedings of the 2024 SIAM International Conference on Data Mining},
  pages={445--453},
  year={2024},
  publisher={SIAM},
  doi={10.1137/1.9781611978032.52}
}

@inproceedings{ghosh2021statt,
  title={Attention-augmented Spatio-Temporal Segmentation for Land Cover Mapping},
  author={Ghosh, Rahul and Ravirathinam, Praveen and Jia, Xiaowei and Lin, Chenxi and Jin, Zhenong and Kumar, Vipin},
  booktitle={Proceedings of the 2021 IEEE International Conference on Big Data},
  pages={1399--1408},
  year={2021},
  publisher={IEEE},
  doi={10.1109/BigData52589.2021.9671974}
}


```
