# Wet Woodland Research - Machine Learning Pipeline

![Wet Woodlands Research Network Logo](tow/images/wwrn_logo.jpeg)

## Overview

This repository contains a complete machine learning pipeline for wet woodland detection using Google Earth Engine embeddings and LiDAR data. The project focuses on identifying wet woodland areas within the new DEFRA Trees Outsie Woodland map using advanced geospatial analysis and LightGBM classification.

## Project Structure

```
wet-woodland-research/
├── tow/                          # Main project directory
│   ├── src/                      # Source code
│   │   ├── ee_train.js           # Earth Engine training data extraction
│   │   ├── ee_inference.js       # Earth Engine inference data extraction
│   │   ├── lightgbm_trainer.py   # LightGBM model training
│   │   ├── lightgbm_predictor.py # LightGBM model inference
│   │   ├── tidy_ee_tiles_dask.py # Tile preprocessing with Dask
│   │   ├── create_*.py           # Data preparation scripts
│   │   └── tow_gdb_processor.py  # Forestry England data processing
│   ├── models/                   # Trained models
│   ├── cache/                    # Processed data cache
│   └── environment.yml           # Conda environment
├── data/                         # Data directory (not tracked)
└── README.md                     # This file
```

## Quick Start

### 1. Environment Setup
```bash
conda env create -f tow/environment.yml
conda activate wwr
```

### 2. Data Preparation
```bash
# Process Forestry England data
python tow/src/tow_gdb_processor.py --gdb-dir data/forestry/ --peatland-file data/peat.shp

# Create training labels
python tow/src/create_species_mask_raster.py data/forestry_filtered.shp data/labels.tif

# Create peat binary mask
python tow/src/create_peat_binary_mask_raster.py data/peat_prob.tif data/peat_binary.tif
```

### 3. Earth Engine Data Extraction
```javascript
// In Google Earth Engine Code Editor:
// Load and run ee_train.js for training data
// Load and run ee_inference.js for inference data
```

### 4. Model Training
```bash
python tow/src/lightgbm_trainer.py \
  --data-dir data/features \
  --labels-file data/labels.tif \
  --save-model models/wet_woodland_model.txt
```

### 5. Model Inference
```bash
python tow/src/lightgbm_predictor.py \
  --model models/wet_woodland_model.txt \
  --data data/inference_features.tif \
  --output predictions.tif
```

## Key Features

- **64-dimensional temporal embeddings** from Google Earth Engine
- **LiDAR features** (CHM, DTM, DSM) from Environment Agency
- **Mature woodland filtering** (CHM ≥ 3m)
- **Class imbalance handling** with optimized LightGBM parameters
- **Caching system** for efficient data processing
- **Flexible feature handling** (embeddings only, LiDAR only, or combined)

## Script Descriptions

### Data Processing
- `tow_gdb_processor.py`: Process Forestry England geodatabases
- `create_species_mask_raster.py`: Create binary species masks
- `create_peat_binary_mask_raster.py`: Convert peat probability to binary
- `dissolve_peat_vector.py`: Combine peat polygons into single boundary

### Machine Learning
- `lightgbm_trainer.py`: Train LightGBM models with comprehensive evaluation
- `lightgbm_predictor.py`: Apply trained models for inference
- `tidy_ee_tiles_dask.py`: Preprocess Earth Engine tiles efficiently

### Earth Engine
- `ee_train.js`: Extract training data (species labels + mature woodland)
- `ee_inference.js`: Extract inference data (peat areas + mature woodland)

## Performance

- **Training**: ~1M+ pixels processed with caching
- **Inference**: Handles large raster tiles efficiently
- **Memory**: Optimized for large-scale geospatial data
- **Accuracy**: F1 score ~0.91, Balanced Accuracy ~0.9465

## Target Species

- **Alder** (*Alnus* spp.)
- **Birch** (*Betula* spp.) 
- **Willow** (*Salix* spp.)

## License

Part of the DEFRA Trees Outside Woodland (TOW) project and Wet Woodlands Research Network.

## Acknowledgments

- **DEFRA**: Project funding and coordination
- **Wet Woodlands Research Network**: Scientific expertise
- **Forestry England**: Woodland management data
- **Google Earth Engine**: Temporal embeddings and processing platform
