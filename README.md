# Trees Outside Woodland (TOW) - Wet Woodland Component

![Wet Woodlands Research Network Logo](tow/images/wwrn_logo.jpeg)


## About the Project

This repository contains the **Wet Woodland Component** of the DEFRA-funded research project **"Applications of the TOW map: Assessing national carbon stocks, biodiversity benefits, and priority habitat creation of non-woodland trees"**.

**Wet Woodlands Research** is a network of scientists and stakeholders interested in wet woodlands and their potential contribution for nature-based solutions for climate change (https://www.wetwoodlands.com/).

### Work Package 6 (WP6): Peat-Forming Wet Woodland Expansion

This component focuses on identifying current and future opportunities for peat-forming wet woodland expansion. Using advanced geospatial methods, we map wet woodland patches in the Trees Outside Woodland (TOW) dataset, model potential expansion zones, and validate predictions through field surveys.

Our work focuses on:
- **Identification**: Locating existing wet woodland areas within the TOW dataset
- **Restoration Potential**: Identifying areas suitable for wet woodland restoration  
- **Carbon Potential**: Assessing future carbon sequestration opportunities
- **Policy Support**: Providing evidence-based tools for climate change mitigation

This work directly feeds into nature-based solutions policy, informing Net-Zero and Biodiversity Net Gain strategies.

## Project Overview

This repository provides tools for processing and analyzing large-scale geospatial datasets related to wet woodland ecosystems, including:

- **Forestry England Subcompartments**: Processing and filtering of woodland management units
- **Tree of Woodland (TOW) Data**: Large-scale GDB processing for national woodland datasets
- **Peat Extent Analysis**: Intersection analysis with peatland extent data
- **Species Filtering**: Identification and filtering of wet woodland species (alder, birch, willow)

## Repository Structure

```
wet-woodland-research/
├── tow/                          # TOW Wet Woodland Component
│   ├── src/                      # Source code
│   │   ├── map_v1_vector.py      # Vector-based species filtering & intersection
│   │   ├── gdb_processor.py      # Large GDB processing & intersection
│   │   └── utils/                # Utility functions
│   ├── images/                   # Project images and logos
│   ├── meta/                     # Project documentation
│   └── environment.yml           # Conda environment specification
├── data/                         # Data directory (not tracked in git)
├── results/                      # Output directory (not tracked in git)
└── README.md                     # This file
```

## Installation & Setup

### Prerequisites

- Python 3.8+
- Conda (recommended for environment management)

### Environment Setup

1. **Create Conda Environment**:
   ```bash
   conda create -n tow python=3.11
   conda activate tow
   ```

2. **Install Required Packages**:
   ```bash
   conda install -c conda-forge geopandas shapely fiona pyogrio rtree pandas tqdm
   pip install rasterio
   ```

   **Alternative**: Use the provided environment file:
   ```bash
   conda env create -f tow/environment.yml
   conda activate tow
   ```

### Data Requirements

Create a `data/` directory in the project root and place your data files there:

- **Forestry England Subcompartments**: `Forestry_England_Subcompartments.shp`
- **Peat Extent**: `peaty_soil_extent_v1.shp`
- **Tree of Woodland GDB**: `FR_TOW_V1_ALL.gdb/` (directory containing GDB files)

*Note: The `data/` directory is excluded from version control via `.gitignore`*

## Usage

### 1. Vector-Based Species Filtering & Intersection

Process Forestry England subcompartments to identify wet woodland species and intersect with peat extent:

```bash
# Basic usage
python tow/src/map_v1_vector.py --data-dir data

# With custom output
python tow/src/map_v1_vector.py --data-dir data --output results/wet_woodland_intersection

# With polygon consolidation
python tow/src/map_v1_vector.py --data-dir data --consolidate

# Consolidate by specific attribute
python tow/src/map_v1_vector.py --data-dir data --consolidate --consolidate-by SPECIES
```

**Features**:
- Filters for target wet woodland species (alder, birch, willow)
- Performs spatial intersection with peat extent data
- Preserves individual subcompartment granularity
- Optional polygon consolidation
- Handles geometry validation and CRS alignment

### 2. Large GDB Processing & Intersection

Process large Tree of Woodland (TOW) GDB files with peat extent intersection:

```bash
# Fast identification of intersecting features
python tow/src/gdb_processor.py --gdb-dir data/FR_TOW_V1/ --peatland-file data/peaty_soil_extent_v1.shp --method identify

# Exact intersection geometries (clipped shapes)
python tow/src/gdb_processor.py --gdb-dir data/FR_TOW_V1/ --peatland-file data/peaty_soil_extent_v1.shp --method intersect

# With custom output and parallel processing
python tow/src/gdb_processor.py --gdb-dir data/FR_TOW_V1/ --peatland-file data/peaty_soil_extent_v1.shp --method intersect --output results/tow_peat_intersection --processes 16
```

**Features**:
- **Two processing modes**:
  - `identify`: Fast identification of whole features that intersect
  - `intersect`: Exact clipped intersection geometries
- **Highly optimized** for large datasets (millions of features)
- **Parallel processing** across multiple CPU cores
- **Spatial indexing** for efficient queries
- **Memory management** with chunked processing
- **Robust error handling** for complex geometries

## Performance Optimizations

### For Large Datasets

The scripts are optimized for processing very large geospatial datasets:

- **Spatial Indexing**: Uses R-tree spatial indexing for efficient spatial queries
- **Parallel Processing**: Leverages multiple CPU cores for faster processing
- **Chunked Processing**: Processes data in manageable chunks to prevent memory overflow
- **Hybrid Methods**: Combines fast identification with exact intersection for optimal performance

### Memory Management

- Automatic garbage collection between chunks
- Efficient geometry handling with `shapely` operations
- Memory usage monitoring and optimization

## Output Formats

All scripts output to **ESRI Shapefile** format for maximum compatibility with:
- QGIS
- ArcGIS
- Field mapping applications
- Other GIS software

## Target Species

The scripts are configured to identify and filter for key wet woodland species:

- **Alder** (*Alnus* spp.)
- **Birch** (*Betula* spp.)
- **Willow** (*Salix* spp.)

## Contributing

This code is part of the DEFRA Trees Outside Woodland (TOW) project, specifically the Wet Woodland Component. For contributions or questions:

1. Ensure code follows the existing style and structure
2. Add appropriate documentation for new features
3. Test with sample datasets before submitting
4. Consider performance implications for large datasets

## License

This code is developed as part of the Wet Woodlands Research Network for scientific research and policy development related to wet woodland ecosystems and climate change mitigation.

## Acknowledgments

- **DEFRA**: For funding and coordinating the Trees Outside Woodland (TOW) project
- **Wet Woodlands Research Network**: For scientific expertise and stakeholder engagement in the wet woodland component
- **Forestry England**: For providing woodland management data
- **UK Peatland Data**: For peat extent datasets
- **Open Source GIS Community**: For the tools and libraries that make this work possible
