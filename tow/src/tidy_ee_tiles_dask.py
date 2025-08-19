#!/usr/bin/env python3
"""
Dask-based preprocessing of Earth Engine tiles to filter areas without labels

DESCRIPTION: High-performance tile filtering using Dask for large-scale raster processing. Checks 
Earth Engine feature tiles for valid label coverage and filters out tiles with insufficient data. 
Uses xarray and Dask for memory-efficient processing of large raster datasets. Optimizes training 
data preparation by removing tiles that would contribute little to model learning.
"""

import dask.array as da
import dask.dataframe as dd
import rasterio
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
from tqdm import tqdm
import xarray as xr

def find_tiff_files(directory):
    """Find all TIFF files recursively"""
    directory = Path(directory)
    tiff_files = []
    
    extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    for ext in extensions:
        files = list(directory.rglob(ext))
        tiff_files.extend(files)
    
    tiff_files = list(set(tiff_files))
    tiff_files.sort()
    return tiff_files

def check_tile_coverage_dask(tile_path, label_path, min_coverage=0.01):
    """
    Check tile coverage using Dask for better performance
    """
    try:
        # Open with xarray (which uses Dask under the hood)
        with xr.open_dataarray(tile_path, engine='rasterio') as tile_ds:
            with xr.open_dataarray(label_path, engine='rasterio') as label_ds:
                # Get tile bounds
                tile_bounds = tile_ds.rio.bounds()
                
                # Clip label to tile bounds
                label_clipped = label_ds.rio.clip_box(*tile_bounds)
                
                # Resize to match tile if needed
                if label_clipped.shape != tile_ds.shape[1:]:  # Skip band dimension
                    label_clipped = label_clipped.interp(
                        y=tile_ds.y, x=tile_ds.x, method='nearest'
                    )
                
                # Sample for speed (every 20th pixel)
                sample_step = 20
                label_sample = label_clipped.values[::sample_step, ::sample_step]
                
                # Check for valid labels
                valid_labels = ~np.isnan(label_sample)
                coverage = np.sum(valid_labels) / valid_labels.size
                
                return coverage >= min_coverage, coverage
                
    except Exception as e:
        print(f"Error with {tile_path.name}: {e}")
        return False, 0.0

def preprocess_tiles_dask(data_dir, labels_dir, output_dir, min_coverage=0.01, force=False):
    """
    Preprocess tiles using Dask for better performance
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Find feature tiles
    print(f"Finding feature tiles in: {data_dir}")
    feature_tiles = find_tiff_files(data_dir)
    print(f"Found {len(feature_tiles)} feature tiles")
    
    # Handle label file path
    if Path(labels_dir).is_file():
        label_path = Path(labels_dir)
    else:
        label_files = list(Path(labels_dir).glob('*.tif'))
        if not label_files:
            raise ValueError(f"No label files found in {labels_dir}")
        label_path = label_files[0]
    
    print(f"Using label file: {label_path}")
    
    if not force:
        response = input(f"Proceed with preprocessing {len(feature_tiles)} tiles? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Process tiles with Dask
    print(f"Processing tiles with minimum coverage: {min_coverage*100}%")
    
    results = {
        'total_tiles': len(feature_tiles),
        'tiles_with_coverage': 0,
        'tiles_processed': 0,
        'tiles_skipped': 0,
        'processing_timestamp': datetime.now().isoformat()
    }
    
    valid_tiles = []
    
    for tile_path in tqdm(feature_tiles, desc="Processing tiles"):
        has_coverage, coverage = check_tile_coverage_dask(
            tile_path, label_path, min_coverage
        )
        
        if has_coverage:
            results['tiles_with_coverage'] += 1
            valid_tiles.append(str(tile_path))
            
            # Copy tile to output directory (or could process it here)
            output_tile = output_dir / tile_path.name
            if not output_tile.exists():
                import shutil
                shutil.copy2(tile_path, output_tile)
            
            results['tiles_processed'] += 1
        else:
            results['tiles_skipped'] += 1
    
    # Save results
    results['valid_tiles'] = valid_tiles
    results_file = output_dir / 'preprocessing_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save valid tiles list
    valid_tiles_file = output_dir / 'valid_tiles.json'
    with open(valid_tiles_file, 'w') as f:
        json.dump(valid_tiles, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total tiles: {results['total_tiles']}")
    print(f"Tiles with coverage: {results['tiles_with_coverage']}")
    print(f"Tiles processed: {results['tiles_processed']}")
    print(f"Tiles skipped: {results['tiles_skipped']}")
    
    if results['tiles_processed'] > 0:
        reduction = (results['tiles_skipped'] / results['total_tiles']) * 100
        print(f"Data reduction: {reduction:.1f}% of tiles removed")
        print(f"Valid tiles saved to: {valid_tiles_file}")
        print(f"Processed tiles saved to: {output_dir}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Dask-based preprocessing of Earth Engine tiles')
    parser.add_argument('data_dir', help='Directory containing feature tiles')
    parser.add_argument('labels_dir', help='Directory or file path for label raster')
    parser.add_argument('--output-dir', '-o', help='Output directory for processed tiles')
    parser.add_argument('--min-coverage', '-c', type=float, default=0.01, 
                       help='Minimum label coverage required (default: 0.01 = 1%%)')
    parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"processed_tiles_dask_{timestamp}"
    
    # Run preprocessing
    results = preprocess_tiles_dask(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        output_dir=output_dir,
        min_coverage=args.min_coverage,
        force=args.force
    )

if __name__ == "__main__":
    main()
