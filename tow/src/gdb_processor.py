#!/usr/bin/env python3
"""
Clean GDB Processor - Simple and Fast Spatial Intersection

Two main modes:
1. identify: Fast identification of which features intersect (whole features)
2. intersect: Exact intersection geometries (clipped shapes)

Uses spatial indexing for optimal performance.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
import warnings
import gc

import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import fiona
from rtree import index
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def identify_intersections_standalone(gdf, peatland_gdf, layer_name):
    """Standalone version of identify_intersections."""
    logger.info(f"Identifying intersections for {layer_name} ({len(gdf)} features)")
    
    # Build spatial index for peatland
    peat_idx = index.Index()
    for i, geom in enumerate(peatland_gdf.geometry):
        if geom and not geom.is_empty:
            peat_idx.insert(i, geom.bounds)
    
    # Get representative points from features
    if gdf.geometry.geom_type.iloc[0] in ['Polygon', 'MultiPolygon']:
        test_points = gdf.geometry.representative_point()
    else:
        test_points = gdf.geometry
    
    # Find intersecting features
    intersecting_indices = []
    
    for idx, point in tqdm(test_points.items(), desc="Testing features", leave=False):
        if point and not point.is_empty:
            # Use spatial index to find candidate peatland polygons
            candidates = list(peat_idx.intersection(point.bounds))
            
            # Test point against candidate polygons
            for peat_idx_val in candidates:
                peat_geom = peatland_gdf.geometry.iloc[peat_idx_val]
                if peat_geom.contains(point):
                    intersecting_indices.append(idx)
                    break  # Found intersection, no need to test others
    
    if intersecting_indices:
        result = gdf.loc[intersecting_indices].copy()
        result['source_layer'] = layer_name
        logger.info(f"Found {len(result)} intersecting features for {layer_name}")
        return result
    else:
        logger.info(f"No intersections found for {layer_name}")
        return gpd.GeoDataFrame(columns=gdf.columns.tolist() + ['source_layer'], crs=gdf.crs)


def exact_intersections_standalone(gdf, peatland_gdf, layer_name, chunk_size):
    """
    Fast intersection using hybrid approach:
    1. Pre-filter with fast identify method
    2. Exact intersection only on relevant features
    """
    logger.info(f"Computing exact intersections for {layer_name} ({len(gdf)} features)")
    
    # STEP 1: Use fast identify method to pre-filter
    logger.info(f"Pre-filtering with fast identify method...")
    identified_features = identify_intersections_standalone(gdf, peatland_gdf, layer_name + "_temp")
    
    if identified_features.empty:
        logger.info(f"No potential intersections for {layer_name}")
        return gpd.GeoDataFrame(columns=gdf.columns.tolist() + ['source_layer'], crs=gdf.crs)
    
    # Remove the temporary source_layer column
    if 'source_layer' in identified_features.columns:
        identified_features = identified_features.drop(columns=['source_layer'])
    
    # STEP 2: Now do exact intersection only on pre-filtered features
    filtered_gdf = identified_features.copy()
    logger.info(f"Fast pre-filter reduced {len(gdf)} to {len(filtered_gdf)} features ({100*len(filtered_gdf)/len(gdf):.1f}%)")
    
    # Build spatial index for peatland (for chunked processing)
    peat_idx = index.Index()
    for i, geom in enumerate(peatland_gdf.geometry):
        if geom and not geom.is_empty:
            peat_idx.insert(i, geom.bounds)
    
    # Process in chunks
    results = []
    
    for i in tqdm(range(0, len(filtered_gdf), chunk_size), 
                 desc=f"Intersecting {layer_name}"):
        
        chunk = filtered_gdf.iloc[i:i + chunk_size].copy()
        
        try:
            # Perform intersection
            intersected = gpd.overlay(chunk, peatland_gdf, how='intersection', keep_geom_type=False)
            
            if not intersected.empty:
                intersected['source_layer'] = layer_name
                results.append(intersected)
            
            # Memory cleanup
            del chunk, intersected
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Chunk {i//chunk_size + 1} failed: {e}")
            continue
    
    if results:
        final_result = pd.concat(results, ignore_index=True)
        logger.info(f"{layer_name}: {len(final_result)} intersection features")
        return final_result
    else:
        logger.info(f"No intersections found for {layer_name}")
        return gpd.GeoDataFrame(columns=gdf.columns.tolist() + ['source_layer'], crs=gdf.crs)


def process_single_layer_standalone(args):
    """Standalone function for multiprocessing - doesn't use self."""
    gdb_path, layer_name, peatland_file, method, chunk_size = args
    
    try:
        # Read layer
        gdf = gpd.read_file(str(gdb_path), layer=layer_name)
        
        if gdf.empty:
            return None
        
        # Read peatland data
        peatland_gdf = gpd.read_file(peatland_file)
        
        # Ensure same CRS
        if gdf.crs != peatland_gdf.crs:
            peatland_gdf = peatland_gdf.to_crs(gdf.crs)
        
        layer_key = f"{Path(gdb_path).name}_{layer_name}"
        
        # Choose processing method
        if method == "identify":
            result = identify_intersections_standalone(gdf, peatland_gdf, layer_key)
        else:  # intersect - always uses hybrid approach for performance
            result = exact_intersections_standalone(gdf, peatland_gdf, layer_key, chunk_size)
        
        return result if not result.empty else None
        
    except Exception as e:
        logger.error(f"Failed to process {layer_name}: {e}")
        return None


class CleanGDBProcessor:
    """
    Clean, fast GDB processor with two simple modes.
    """
    
    def __init__(self, 
                 gdb_directory: str,
                 peatland_file: str,
                 output_file: str = None,
                 method: str = "identify",
                 chunk_size: int = 5000,
                 n_processes: int = None):
        
        # Auto-generate output filename based on method if not provided
        if output_file is None:
            if method == "identify":
                output_file = "identified.shp"
            else:  # intersect
                output_file = "intersected.shp"
        
        # Ensure output file has .shp extension
        if not output_file.endswith('.shp'):
            output_file = output_file + '.shp'
        
        """
        Initialize the processor.
        
        Parameters:
        -----------
        gdb_directory : str
            Directory containing GDB folders
        peatland_file : str
            Path to peatland extent file
        output_file : str
            Output file path
        method : str
            'identify' (fast, whole features) or 'intersect' (exact clipped shapes)
        chunk_size : int
            Features per chunk for intersect method
        n_processes : int
            Number of processes (default: CPU count - 1)
        """
        self.gdb_directory = Path(gdb_directory)
        self.peatland_file = Path(peatland_file)
        self.output_file = Path(output_file)
        self.method = method.lower()
        self.chunk_size = chunk_size
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        
        if self.method not in ['identify', 'intersect']:
            raise ValueError("Method must be 'identify' or 'intersect'")
        
        logger.info(f"Initialized processor: {self.method} mode")
        logger.info(f"Processes: {self.n_processes}")
        logger.info(f"Chunk size: {self.chunk_size}")
    
    def run(self):
        """Run the processing pipeline."""
        try:
            logger.info(f"Starting {self.method} processing pipeline")
            
            # Find all GDB folders
            gdb_folders = list(self.gdb_directory.glob("*.gdb"))
            
            if not gdb_folders:
                logger.error(f"No GDB folders found in {self.gdb_directory}")
                return
            
            # Collect all layer processing tasks
            processing_tasks = []
            
            for gdb_folder in gdb_folders:
                try:
                    layers = fiona.listlayers(str(gdb_folder))
                    for layer in layers:
                        task = (gdb_folder, layer, self.peatland_file, 
                               self.method, self.chunk_size)
                        processing_tasks.append(task)
                except Exception as e:
                    logger.warning(f"Could not list layers in {gdb_folder}: {e}")
                    continue
            
            logger.info(f"Found {len(processing_tasks)} layers to process")
            
            # Process layers in parallel
            results = []
            
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(process_single_layer_standalone, task): task 
                    for task in processing_tasks
                }
                
                # Collect results
                for future in tqdm(as_completed(future_to_task), 
                                 total=len(processing_tasks),
                                 desc="Processing layers"):
                    try:
                        result = future.result()
                        if result is not None and not result.empty:
                            results.append(result)
                    except Exception as e:
                        task = future_to_task[future]
                        logger.error(f"Task failed {task[1]}: {e}")
            
            # Combine and export results
            if results:
                logger.info(f"Combining {len(results)} result datasets...")
                combined_result = pd.concat(results, ignore_index=True)
                
                # Export to file
                logger.info(f"Exporting {len(combined_result)} features to {self.output_file}")
                
                # Ensure output directory exists
                output_dir = Path(self.output_file).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Write to shapefile
                combined_result.to_file(self.output_file, driver='ESRI Shapefile')
                
                logger.info("Processing completed successfully!")
                logger.info(f"Total features: {len(combined_result)}")
                logger.info(f"Source layers: {combined_result['source_layer'].nunique()}")
                
            else:
                logger.warning("No intersecting features found")
                
        except Exception as e:
            logger.error(f"Processing pipeline failed: {e}")
            raise


def main():
    """Main function with clean argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean GDB intersection processor")
    parser.add_argument("--gdb-dir", required=True, help="Directory containing GDB folders")
    parser.add_argument("--peatland-file", required=True, help="Path to peatland extent file")
    parser.add_argument("--output", help="Output file path (auto-generated if not provided)")
    parser.add_argument("--method", choices=['identify', 'intersect'], default='identify',
                       help="identify: fast whole features, intersect: exact clipped shapes")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Features per chunk for intersect method")
    parser.add_argument("--processes", type=int, help="Number of processes")
    
    args = parser.parse_args()
    
    processor = CleanGDBProcessor(
        gdb_directory=args.gdb_dir,
        peatland_file=args.peatland_file,
        output_file=args.output,
        method=args.method,
        chunk_size=args.chunk_size,
        n_processes=args.processes
    )
    
    processor.run()


if __name__ == "__main__":
    main() 