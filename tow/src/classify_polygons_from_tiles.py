#!/usr/bin/env python3
"""
Polygon Classifier using Prediction Tiles

Efficiently classifies polygons based on LightGBM prediction tiles.
For each polygon:
1. Finds intersecting prediction tiles
2. Extracts raster data for polygon area
3. Calculates percentage of wet woodland pixels and mean probability
4. Classifies polygon based on threshold
"""

import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from shapely.geometry import box
import warnings

warnings.filterwarnings('ignore')

def create_tile_spatial_index(tile_dir):
    """Create spatial index for all prediction tiles."""
    from shapely.geometry import box
    from shapely.strtree import STRtree
    
    print("🔍 Creating spatial index for prediction tiles...")
    
    tile_files = glob.glob(str(Path(tile_dir) / "prediction_*.tif"))
    tile_geometries = []
    tile_paths = []
    
    for tile_file in tile_files:
        try:
            with rasterio.open(tile_file) as src:
                # Create bounding box for tile
                tile_bounds = src.bounds
                tile_geom = box(tile_bounds.left, tile_bounds.bottom, 
                              tile_bounds.right, tile_bounds.top)
                tile_geometries.append(tile_geom)
                tile_paths.append(tile_file)
        except:
            continue
    
    # Create spatial index
    spatial_index = STRtree(tile_geometries)
    print(f"   Indexed {len(tile_geometries)} tiles")
    
    return spatial_index, tile_paths

def find_intersecting_tiles_fast(polygon_bounds, spatial_index, tile_paths):
    """Find prediction tiles that intersect with polygon bounds using spatial index."""
    from shapely.geometry import box
    
    # Create bounding box for polygon
    polygon_geom = box(polygon_bounds[0], polygon_bounds[1], 
                      polygon_bounds[2], polygon_bounds[3])
    
    # Query spatial index
    intersecting_indices = list(spatial_index.query(polygon_geom))
    
    # Get corresponding tile paths
    intersecting_tiles = [tile_paths[i] for i in intersecting_indices]
    
    return intersecting_tiles

def extract_polygon_statistics(polygon, tile_files, threshold=0.5):
    """
    Extract statistics for a single polygon from prediction tiles.
    
    Returns:
        dict: Statistics including wet_woodland_percentage, mean_probability, classification
    """
    try:
        # Get polygon bounds
        minx, miny, maxx, maxy = polygon.bounds
        
        all_pixels = []
        all_probabilities = []
        
        # Process each intersecting tile
        for tile_file in tile_files:
            try:
                with rasterio.open(tile_file) as src:
                    # Create window for polygon bounds
                    from rasterio.windows import from_bounds
                    window = from_bounds(minx, miny, maxx, maxy, src.transform)
                    
                    # Read both bands (binary classification and probability)
                    binary_data = src.read(1, window=window)  # Band 1: binary (0/1)
                    prob_data = src.read(2, window=window)    # Band 2: probability (0.0-1.0)
                    
                    # Mask out no-data values
                    valid_mask = (binary_data != 255) & (prob_data != 255)
                    
                    if valid_mask.sum() > 0:
                        # Extract valid pixels
                        valid_binary = binary_data[valid_mask]
                        valid_prob = prob_data[valid_mask]
                        
                        all_pixels.extend(valid_binary)
                        all_probabilities.extend(valid_prob)
                        
            except Exception as e:
                continue
        
        if not all_pixels:
            return {
                'wet_woodland': False,
                'wet_woodland_prob': 0.0
            }
        
        # Calculate statistics
        all_pixels = np.array(all_pixels)
        all_probabilities = np.array(all_probabilities)
        
        wet_woodland_pixels = np.sum(all_pixels == 1)
        total_pixels = len(all_pixels)
        wet_woodland_percentage = wet_woodland_pixels / total_pixels if total_pixels > 0 else 0.0
        mean_probability = np.mean(all_probabilities) if len(all_probabilities) > 0 else 0.0
        
        # Classify based on threshold
        is_wet_woodland = wet_woodland_percentage >= threshold
        
        return {
            'wet_woodland': bool(is_wet_woodland),
            'wet_woodland_prob': float(mean_probability)
        }
        
    except Exception as e:
        return {
            'wet_woodland': False,
            'wet_woodland_prob': 0.0
        }

def process_polygon_chunk(args):
    """Process a chunk of polygons."""
    chunk_gdf, spatial_index, tile_paths, threshold = args
    
    results = []
    
    for idx, row in chunk_gdf.iterrows():
        polygon = row.geometry
        
        # Skip invalid geometries
        if polygon is None or polygon.is_empty:
            results.append({
                'polygon_id': idx,
                'wet_woodland': False,
                'wet_woodland_prob': 0.0
            })
            continue
        
        # Find intersecting tiles using spatial index
        intersecting_tiles = find_intersecting_tiles_fast(polygon.bounds, spatial_index, tile_paths)
        
        if not intersecting_tiles:
            # No intersecting tiles
            results.append({
                'polygon_id': idx,
                'wet_woodland': False,
                'wet_woodland_prob': 0.0
            })
            continue
        
        # Extract statistics
        stats = extract_polygon_statistics(polygon, intersecting_tiles, threshold)
        
        results.append({
            'polygon_id': idx,
            'wet_woodland': stats['wet_woodland'],
            'wet_woodland_prob': stats['wet_woodland_prob']
        })
    
    return results

def classify_polygons_from_tiles(polygon_file, tile_dir, output_file, threshold=0.5, 
                                chunk_size=1000, n_workers=None, keep_existing_attributes=True, 
                                only_wet_woodland=False):
    """
    Classify polygons using prediction tiles.
    
    Args:
        polygon_file: Path to polygon shapefile
        tile_dir: Directory containing prediction tiles
        output_file: Output path for classified shapefile
        threshold: Classification threshold (default 0.5 = 50%)
        chunk_size: Number of polygons per chunk
        n_workers: Number of parallel workers
    """
    print("🚀 Starting polygon classification from prediction tiles")
    print("=" * 60)
    
    start_time = time.time()
    
    # Set number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    print(f"🔧 Using {n_workers} parallel workers")
    
    # Load polygons
    print(f"📂 Loading polygons: {polygon_file}")
    gdf = gpd.read_file(polygon_file)
    print(f"   Loaded {len(gdf):,} polygons")
    print(f"   CRS: {gdf.crs}")
    print(f"   Columns: {list(gdf.columns)}")
    
    # Create spatial index for prediction tiles
    spatial_index, tile_paths = create_tile_spatial_index(tile_dir)
    if not tile_paths:
        print(f"❌ No prediction tiles found in {tile_dir}")
        return None
    
    print(f"📊 Found {len(tile_paths)} prediction tiles")
    
    # Process polygons in chunks
    print(f"\n📊 Processing polygons in chunks of {chunk_size}...")
    
    all_results = []
    total_processed = 0
    
    # Split into chunks
    chunk_indices = range(0, len(gdf), chunk_size)
    
    # Prepare arguments for parallel processing
    chunk_args = []
    for i in chunk_indices:
        chunk_end = min(i + chunk_size, len(gdf))
        chunk_gdf = gdf.iloc[i:chunk_end].copy()
        chunk_args.append((chunk_gdf, spatial_index, tile_paths, threshold))
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all chunks
        future_to_chunk = {executor.submit(process_polygon_chunk, args): i 
                          for i, args in enumerate(chunk_args)}
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_chunk), total=len(chunk_args), desc="Processing"):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_results = future.result()
                all_results.extend(chunk_results)
                total_processed += len(chunk_results)
                
                # Don't print progress updates to avoid interrupting progress bar
                    
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_idx}: {e}")
    
    print(f"\n✅ Completed processing {total_processed:,} polygons")
    
    # Convert results to DataFrame
    print("📋 Converting results to DataFrame...")
    results_df = pd.DataFrame(all_results)
    
    # Calculate summary statistics
    total_polygons = len(results_df)
    wet_woodland_polygons = results_df['wet_woodland'].sum()
    wet_woodland_percentage = (wet_woodland_polygons / total_polygons) * 100
    
    print(f"\n📈 Classification Summary:")
    print(f"   Total polygons: {total_polygons:,}")
    print(f"   Wet woodland polygons: {wet_woodland_polygons:,}")
    print(f"   Wet woodland percentage: {wet_woodland_percentage:.2f}%")
    print(f"   Classification threshold: {threshold * 100}%")
    
    # Merge results with original polygons
    print("🔗 Merging results with original polygons...")
    
    # Create results DataFrame with polygon_id as index
    results_df.set_index('polygon_id', inplace=True)
    
    if keep_existing_attributes:
        # Keep all existing attributes and add classification columns
        print("   Keeping all existing attributes...")
        gdf['wet_woodland'] = results_df['wet_woodland']
        gdf['wet_woodland_prob'] = results_df['wet_woodland_prob']
        
        # Fill NaN values (for polygons that couldn't be processed)
        gdf['wet_woodland'].fillna(False, inplace=True)
        gdf['wet_woodland_prob'].fillna(0.0, inplace=True)
    else:
        # Clear existing attributes and keep only geometry + classification
        print("   Clearing existing attributes, keeping only classification results...")
        
        # Create new GeoDataFrame with only geometry and classification
        new_gdf = gdf[['geometry']].copy()
        
        # Add classification columns
        new_gdf['wet_woodland'] = results_df['wet_woodland']
        new_gdf['wet_woodland_prob'] = results_df['wet_woodland_prob']
        
        # Fill NaN values (for polygons that couldn't be processed)
        new_gdf['wet_woodland'].fillna(False, inplace=True)
        new_gdf['wet_woodland_prob'].fillna(0.0, inplace=True)
        
        # Replace original GeoDataFrame
        gdf = new_gdf
    
    # Filter to only wet woodland polygons if requested
    if only_wet_woodland:
        print("🌲 Filtering to only wet woodland polygons...")
        original_count = len(gdf)
        gdf = gdf[gdf['wet_woodland'] == True].copy()
        filtered_count = len(gdf)
        print(f"   Kept {filtered_count:,} out of {original_count:,} polygons ({filtered_count/original_count*100:.1f}%)")
    
    # Save classified shapefile
    print(f"💾 Saving classified shapefile: {output_file}")
    gdf.to_file(output_file)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n🎉 Classification complete!")
    print(f"   Processing time: {total_time:.1f} seconds")
    print(f"   Processing rate: {total_processed/total_time:.0f} polygons/second")
    print(f"   Output saved to: {output_file}")
    
    return gdf

def main():
    parser = argparse.ArgumentParser(description="Classify polygons using prediction tiles")
    parser.add_argument("--polygons", required=True, help="Input polygon shapefile")
    parser.add_argument("--tile-dir", required=True, help="Directory containing prediction tiles")
    parser.add_argument("--output", required=True, help="Output classified shapefile")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (default: 0.5)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Polygons per chunk (default: 1000)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default: auto)")
    parser.add_argument("--clear-attributes", action="store_true", help="Clear existing attributes, keep only classification results")
    parser.add_argument("--only-wet-woodland", action="store_true", help="Only output polygons classified as wet woodland")
    
    args = parser.parse_args()
    
    # Verify input files
    if not Path(args.polygons).exists():
        print(f"❌ Polygon file not found: {args.polygons}")
        return
    
    if not Path(args.tile_dir).exists():
        print(f"❌ Tile directory not found: {args.tile_dir}")
        return
    
    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run classification
        result_gdf = classify_polygons_from_tiles(
            polygon_file=args.polygons,
            tile_dir=args.tile_dir,
            output_file=args.output,
            threshold=args.threshold,
            chunk_size=args.chunk_size,
            n_workers=args.workers,
            keep_existing_attributes=not args.clear_attributes,
            only_wet_woodland=args.only_wet_woodland
        )
        
        if result_gdf is not None:
            print(f"\n✅ Successfully classified {len(result_gdf):,} polygons!")
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return

if __name__ == "__main__":
    main()
