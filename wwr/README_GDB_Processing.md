# GDB to Dask GeoPandas Processor

This script processes multiple GDB (Geodatabase) folders and converts them to Dask GeoPandas DataFrames for distributed processing. It then performs intersection with peatland extent data and exports the results to an efficient vector format (GPKG).

## Features

- **Distributed Processing**: Uses Dask for parallel processing of large spatial datasets
- **GDB Support**: Reads all layers from multiple GDB folders
- **Intersection Analysis**: Performs spatial intersection with peatland extent data
- **Efficient Export**: Outputs results in GPKG format for optimal performance
- **Progress Tracking**: Real-time progress bars and logging
- **Memory Management**: Configurable memory limits and worker counts

## Installation

1. Update your environment with the required dependencies:

```bash
conda env update -f environment.yml
```

2. Activate the environment:

```bash
conda activate wwr
```

## Usage

### Command Line Interface

```bash
python src/gdb_dask_processor.py \
    --gdb-dir /path/to/gdb/directory \
    --peatland-file /path/to/peatland.shp \
    --output intersected_features.gpkg \
    --workers 4 \
    --memory 4GB
```

### Parameters

- `--gdb-dir`: Directory containing GDB folders (required)
- `--peatland-file`: Path to peatland extent shapefile (required)
- `--output`: Output file path (default: intersected_features.gpkg)
- `--workers`: Number of Dask workers (default: 4)
- `--memory`: Memory limit per worker (default: 4GB)

### Example Usage

```bash
# Process GDB files in the data directory
python src/gdb_dask_processor.py \
    --gdb-dir data \
    --peatland-file data/peaty_soil_extent_v1.shp \
    --output data/gdb_peat_intersection.gpkg \
    --workers 2 \
    --memory 2GB
```

### Python Script Usage

You can also use the processor programmatically:

```python
from src.gdb_dask_processor import GDBDaskProcessor

# Create processor
processor = GDBDaskProcessor(
    gdb_directory="data",
    peatland_file="data/peaty_soil_extent_v1.shp",
    output_file="data/intersected_features.gpkg",
    n_workers=4,
    memory_limit="4GB"
)

# Run the processing pipeline
processor.run()
```

## Example Script

A simple example script is provided in `src/run_gdb_processing.py`:

```bash
cd src
python run_gdb_processing.py
```

## How It Works

1. **GDB Discovery**: Scans the specified directory for GDB folders
2. **Layer Reading**: Reads all layers from each GDB file
3. **Dask Conversion**: Converts GeoPandas DataFrames to Dask GeoPandas for distributed processing
4. **Peatland Loading**: Loads peatland extent data and converts to Dask format
5. **Intersection**: Performs spatial intersection between GDB features and peatland extent
6. **Export**: Combines results and exports to GPKG format

## Output

The script produces a GPKG file containing:
- All GDB features that intersect with peatland extent
- Source layer information for each feature
- Preserved attribute data from original GDB layers
- Consistent coordinate reference system

## Performance Tips

- **Memory**: Adjust `--memory` based on your system's available RAM
- **Workers**: Use 2-4 workers for most systems, more for high-end machines
- **Partitioning**: The script automatically determines optimal partitioning based on data size
- **Monitoring**: Access the Dask dashboard URL shown in logs for real-time monitoring

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `--memory` or `--workers` parameters
2. **CRS Mismatch**: The script automatically handles CRS reprojection
3. **Empty Results**: Check that your peatland data covers the GDB extent
4. **Slow Performance**: Increase workers or memory if your system supports it

### Logging

The script provides detailed logging including:
- Progress bars for long operations
- Feature counts for each layer
- Intersection results
- Export summaries

## Dependencies

- `geopandas`: Spatial data processing
- `dask-geopandas`: Distributed spatial processing
- `fiona`: GDB file reading
- `shapely`: Geometric operations
- `tqdm`: Progress tracking

## File Formats

- **Input**: GDB folders, Shapefiles
- **Output**: GPKG (GeoPackage) - efficient, single-file vector format
- **Intermediate**: Dask DataFrames for distributed processing 