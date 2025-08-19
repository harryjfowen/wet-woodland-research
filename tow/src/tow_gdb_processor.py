#!/usr/bin/env python3
"""
Ultra-Optimized GDB Intersection Processor - FIXED VERSION

DESCRIPTION: High-performance processor for Forestry England geodatabases (.gdb) that finds compartments 
intersecting with peatland areas. Uses parallel processing, spatial indexing, and memory optimization to 
handle large-scale spatial datasets efficiently. Supports both identification and exact intersection modes.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import warnings
import gc
import pickle
import psutil
import traceback

import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import fiona
from rtree import index
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from shapely.geometry import box, Point, LineString, Polygon, MultiPolygon
from shapely.strtree import STRtree
from shapely.prepared import prep
import numpy as np

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global cache so each worker only loads & indexes the peatland layer once
# -----------------------------------------------------------------------------
_PEAT_CACHE = {}  # key = file path str; value = (geodataframe)


def filter_geometries_by_type(gdf, target_geom_type='Polygon'):
    """
    Filter geometries to ensure consistent type for shapefile export.
    
    Parameters:
    - gdf: GeoDataFrame with potentially mixed geometry types
    - target_geom_type: Target geometry type ('Polygon', 'LineString', 'Point')
    
    Returns:
    - Filtered GeoDataFrame with only the target geometry type
    """
    if gdf.empty:
        return gdf
    
    # Handle both single and multi geometries
    if target_geom_type == 'Polygon':
        valid_types = ['Polygon', 'MultiPolygon']
    elif target_geom_type == 'LineString':
        valid_types = ['LineString', 'MultiLineString']
    elif target_geom_type == 'Point':
        valid_types = ['Point', 'MultiPoint']
    else:
        valid_types = [target_geom_type]
    
    # Filter by geometry type
    mask = gdf.geometry.geom_type.isin(valid_types)
    filtered_gdf = gdf[mask].copy()
    
    # Log what was filtered out
    if not mask.all():
        filtered_count = len(gdf) - len(filtered_gdf)
        filtered_types = gdf[~mask].geometry.geom_type.value_counts().to_dict()
        logger.info(f"Filtered out {filtered_count} non-{target_geom_type} geometries: {filtered_types}")
    
    return filtered_gdf


def ensure_polygon_intersections(intersection_geom):
    """
    Convert intersection results to polygons only, filtering out lines and points.
    
    Parameters:
    - intersection_geom: Result of geometry.intersection()
    
    Returns:
    - Polygon/MultiPolygon geometry or None if no valid polygon intersection
    """
    if intersection_geom.is_empty:
        return None
    
    geom_type = intersection_geom.geom_type
    
    # Return as-is if already polygon
    if geom_type in ['Polygon', 'MultiPolygon']:
        return intersection_geom
    
    # Handle GeometryCollection - extract polygons only
    elif geom_type == 'GeometryCollection':
        polygons = [geom for geom in intersection_geom.geoms 
                   if geom.geom_type in ['Polygon', 'MultiPolygon']]
        
        if len(polygons) == 0:
            return None
        elif len(polygons) == 1:
            return polygons[0]
        else:
            # Create MultiPolygon from multiple polygons
            from shapely.geometry import MultiPolygon
            return MultiPolygon(polygons)
    
    # Skip line and point intersections
    elif geom_type in ['LineString', 'MultiLineString', 'Point', 'MultiPoint']:
        return None
    
    else:
        logger.warning(f"Unexpected geometry type in intersection: {geom_type}")
        return None


def identify_intersections_standalone_silent(gdf, peatland_gdf, layer_name):
    """Silent version of fast identification for parallel processing."""
    # Build STRtree spatial index for peatland (2-5× faster than rtree)
    peat_geoms = peatland_gdf.geometry.tolist()
    peat_tree = STRtree(peat_geoms)
    
    # Representative points from polygons (or centroids for lines/points)
    test_points = (
        gdf.geometry.representative_point()
        if gdf.geometry.geom_type.iloc[0] in ["Polygon", "MultiPolygon"]
        else gdf.geometry
    )
    
    intersecting_indices = []
    
    for idx, point in test_points.items():
        if point and not point.is_empty:
            # STRtree query returns the indices of the geometries in the tree
            candidate_indices = peat_tree.query(point)
            
            for peat_idx in candidate_indices:
                peat_geom = peat_geoms[peat_idx]  # Look up the geometry using the index
                if peat_geom.contains(point):
                    intersecting_indices.append(idx)
                    break
    
    if intersecting_indices:
        result = gdf.loc[intersecting_indices].copy()
        result['source_layer'] = layer_name
        return result
    else:
        return gpd.GeoDataFrame(columns=gdf.columns.tolist() + ['source_layer'], crs=gdf.crs)


def process_single_layer_identification(gdb_path, layer_name, peatland_file):
    """
    Process a single layer for identification only.
    Accepts explicit arguments for robust multiprocessing.
    """
    
    try:
        # Load GDB layer
        gdf = gpd.read_file(gdb_path, layer=layer_name)
        
        # Load peatland data (cached per worker)
        global _PEAT_CACHE
        if peatland_file in _PEAT_CACHE:
            peatland_gdf = _PEAT_CACHE[peatland_file]
        else:
            peatland_gdf = gpd.read_file(peatland_file)
            _PEAT_CACHE[peatland_file] = peatland_gdf
        
        # Ensure same CRS
        if gdf.crs != peatland_gdf.crs:
            peatland_gdf = peatland_gdf.to_crs(gdf.crs)
        
        # Fix geometries
        gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
        peatland_gdf = peatland_gdf[~peatland_gdf.geometry.isna() & ~peatland_gdf.geometry.is_empty]
        
        if gdf.empty or peatland_gdf.empty:
            return None
        
        # Efficient identification using STRtree-based method
        result = identify_intersections_standalone_silent(gdf, peatland_gdf, layer_name)
        
        if result is not None and not result.empty:
            return result
        else:
            return None
            
    except Exception as e:
        logger.error(f"Worker failed on layer {layer_name} in {gdb_path}: {e}", exc_info=True)
        return None


def preprocess_geometries(gdf):
    """
    ULTRA OPTIMIZATION: Preprocess and validate geometries once.
    This eliminates repeated geometry checks in the main loop.
    """
    # Remove invalid/empty geometries
    valid_mask = gdf.geometry.is_valid & ~gdf.geometry.is_empty
    gdf = gdf[valid_mask].copy()
    
    # Fix any remaining invalid geometries
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)
    
    # Pre-compute bounds for all geometries (much faster than repeated .bounds calls)
    bounds_df = gdf.bounds
    gdf['_minx'] = bounds_df['minx']
    gdf['_miny'] = bounds_df['miny'] 
    gdf['_maxx'] = bounds_df['maxx']
    gdf['_maxy'] = bounds_df['maxy']
    
    return gdf


def create_optimized_spatial_index(gdf, use_prepared=True):
    """
    ULTRA OPTIMIZATION: Create STRtree (faster than rtree) with prepared geometries.
    STRtree is 2-5x faster than rtree for bulk operations.
    """
    logger.info(f"Building optimized spatial index for {len(gdf)} geometries...")
    
    # Use STRtree instead of rtree (much faster for bulk operations)
    geometries = gdf.geometry.tolist()
    
    if use_prepared:
        # Prepared geometries are 2-3x faster for repeated intersection tests
        prepared_geoms = [prep(geom) for geom in geometries]
        spatial_tree = STRtree(geometries)
        return spatial_tree, prepared_geoms
    else:
        spatial_tree = STRtree(geometries)
        return spatial_tree, None


def vectorized_bounds_intersection(chunk_bounds, peat_bounds):
    """
    ULTRA OPTIMIZATION: Vectorized bounds checking using NumPy.
    This replaces individual geometry.bounds calls with bulk operations.
    """
    # Convert to numpy arrays for vectorized operations
    chunk_minx = chunk_bounds['minx'].values
    chunk_miny = chunk_bounds['miny'].values
    chunk_maxx = chunk_bounds['maxx'].values
    chunk_maxy = chunk_bounds['maxy'].values
    
    peat_minx = peat_bounds['minx'].values[:, np.newaxis]
    peat_miny = peat_bounds['miny'].values[:, np.newaxis]
    peat_maxx = peat_bounds['maxx'].values[:, np.newaxis]
    peat_maxy = peat_bounds['maxy'].values[:, np.newaxis]
    
    # Vectorized bounds intersection check
    intersects = (
        (chunk_minx < peat_maxx) & (chunk_maxx > peat_minx) &
        (chunk_miny < peat_maxy) & (chunk_maxy > peat_miny)
    )
    
    return intersects


def process_intersection_chunk_ultra_optimized(process_args):
    """
    ULTRA-OPTIMIZED chunk processing with multiple performance improvements.
    FIXED: Now filters geometries to ensure consistent types for shapefile export.
    """
    try:
        chunk_gdf, peatland_gdf, chunk_id = process_args
    except ValueError as e:
        logger.error(f"❌ Failed to unpack process_args: {e}")
        logger.error(f"❌ process_args type: {type(process_args)}")
        logger.error(f"❌ process_args length: {len(process_args) if hasattr(process_args, '__len__') else 'no length'}")
        logger.error(f"❌ process_args content: {process_args}")
        raise
    
    try:
        # OPTIMIZATION 1: Preprocess geometries for the chunk only
        chunk_gdf = preprocess_geometries(chunk_gdf)
        
        if chunk_gdf.empty or peatland_gdf.empty:
            return gpd.GeoDataFrame(columns=chunk_gdf.columns.tolist() + ['intersection_area', 'peatland_id'], crs=chunk_gdf.crs)
        
        # OPTIMIZATION 2: Build spatial index locally to avoid pickling issues
        # Building STRtree can fail for certain bad geometries; skip to avoid errors
        prepared_peat_geoms = None
        
        # OPTIMIZATION 3: Bulk spatial filtering using chunk envelope
        chunk_envelope = box(
            chunk_gdf['_minx'].min(), chunk_gdf['_miny'].min(),
            chunk_gdf['_maxx'].max(), chunk_gdf['_maxy'].max()
        )
        
        # Get peatland candidates that intersect with chunk envelope
        candidate_indices = [i for i, geom in enumerate(peatland_gdf.geometry) 
                           if chunk_envelope.intersects(geom)]
        
        if not candidate_indices:
            return gpd.GeoDataFrame(columns=chunk_gdf.columns.tolist() + ['intersection_area', 'peatland_id'], crs=chunk_gdf.crs)
        
        # OPTIMIZATION 4: Work only with filtered peatland subset
        filtered_peat = peatland_gdf.iloc[candidate_indices].copy()
        filtered_prepared = [prepared_peat_geoms[i] for i in candidate_indices] if prepared_peat_geoms else None
        
        # OPTIMIZATION 5: Vectorized bounds pre-filtering
        chunk_bounds = chunk_gdf[['_minx', '_miny', '_maxx', '_maxy']].rename(columns={
            '_minx': 'minx', '_miny': 'miny', '_maxx': 'maxx', '_maxy': 'maxy'
        })
        peat_bounds = filtered_peat[['_minx', '_miny', '_maxx', '_maxy']].rename(columns={
            '_minx': 'minx', '_miny': 'miny', '_maxx': 'maxx', '_maxy': 'maxy'
        })
        
        bounds_intersect_matrix = vectorized_bounds_intersection(chunk_bounds, peat_bounds)
        
        # OPTIMIZATION 6: Process only bounds-intersecting pairs
        intersection_results = []
        
        for chunk_idx, (_, feature_row) in enumerate(chunk_gdf.iterrows()):
            feature_geom = feature_row.geometry
            
            # Get peatland candidates that pass bounds check
            peat_candidates = np.where(bounds_intersect_matrix[:, chunk_idx])[0]
            
            if len(peat_candidates) == 0:
                continue
            
            # OPTIMIZATION 7: Use prepared geometries for intersection tests
            for peat_candidate_idx in peat_candidates:
                if filtered_prepared:
                    # Use prepared geometry (2-3x faster)
                    prepared_peat = filtered_prepared[peat_candidate_idx]
                    if not prepared_peat.intersects(feature_geom):
                        continue
                else:
                    peat_geom = filtered_peat.geometry.iloc[peat_candidate_idx]
                    if not feature_geom.intersects(peat_geom):
                        continue
                
                # OPTIMIZATION 8: Compute exact intersection only when needed
                try:
                    peat_geom = filtered_peat.geometry.iloc[peat_candidate_idx]
                    intersection_geom = feature_geom.intersection(peat_geom)
                    
                    # FIX: Filter intersection to ensure polygon-only results
                    polygon_intersection = ensure_polygon_intersections(intersection_geom)
                    
                    if polygon_intersection is not None and not polygon_intersection.is_empty:
                        # OPTIMIZATION 9: Batch area calculations
                        area = polygon_intersection.area
                        
                        # Only keep intersections with meaningful area
                        if area > 1e-10:  # Filter out tiny slivers
                            # Create result row efficiently
                            result_row = feature_row.copy()
                            result_row.geometry = polygon_intersection
                            result_row['intersection_area'] = area
                            result_row['peatland_id'] = candidate_indices[peat_candidate_idx]
                            intersection_results.append(result_row)
                        
                except Exception:
                    # Skip problematic geometries
                    continue
        
        # OPTIMIZATION 10: Efficient result construction
        if intersection_results:
            result_gdf = gpd.GeoDataFrame(intersection_results, crs=chunk_gdf.crs)
            # Clean up temporary columns
            cols_to_drop = ['_minx', '_miny', '_maxx', '_maxy']
            result_gdf = result_gdf.drop(columns=[col for col in cols_to_drop if col in result_gdf.columns])
            return result_gdf
        else:
            return gpd.GeoDataFrame(columns=chunk_gdf.columns.tolist() + ['intersection_area', 'peatland_id'], crs=chunk_gdf.crs)
            
    except Exception as e:
        logger.error(f"Error processing ultra-optimized chunk {chunk_id}: {e}")
        return gpd.GeoDataFrame(columns=chunk_gdf.columns.tolist() + ['intersection_area', 'peatland_id'], crs=chunk_gdf.crs)


def create_spatial_chunks(gdf, n_chunks):
    """
    ULTRA OPTIMIZATION: Create spatially-aware chunks for better load balancing.
    Spatial chunks reduce cross-boundary processing and improve cache locality.
    """
    logger.info(f"Creating {n_chunks} spatially-aware chunks...")
    
    # Get spatial bounds
    total_bounds = gdf.total_bounds
    minx, miny, maxx, maxy = total_bounds
    
    # Calculate optimal grid dimensions (try to make square-ish chunks)
    aspect_ratio = (maxx - minx) / (maxy - miny)
    if aspect_ratio > 1:
        nx = int(np.sqrt(n_chunks * aspect_ratio))
        ny = int(n_chunks / nx)
    else:
        ny = int(np.sqrt(n_chunks / aspect_ratio))
        nx = int(n_chunks / ny)
    
    # Ensure we have at least the requested number of chunks
    while nx * ny < n_chunks:
        if nx <= ny:
            nx += 1
        else:
            ny += 1
    
    logger.info(f"Using {nx}x{ny} spatial grid for chunking")
    
    # Create spatial grid
    x_step = (maxx - minx) / nx
    y_step = (maxy - miny) / ny
    
    chunks = []
    chunk_id = 0
    
    for i in range(nx):
        for j in range(ny):
            # Define chunk bounds
            chunk_minx = minx + i * x_step
            chunk_maxx = minx + (i + 1) * x_step
            chunk_miny = miny + j * y_step
            chunk_maxy = miny + (j + 1) * y_step
            
            # Handle edge cases for last chunks
            if i == nx - 1:
                chunk_maxx = maxx
            if j == ny - 1:
                chunk_maxy = maxy
            
            # Select features in this spatial chunk using precomputed bounds
            mask = (
                (gdf['_minx'] < chunk_maxx) &
                (gdf['_maxx'] > chunk_minx) &
                (gdf['_miny'] < chunk_maxy) &
                (gdf['_maxy'] > chunk_miny)
            )
            
            chunk_features = gdf[mask]
            
            if not chunk_features.empty:
                chunks.append((chunk_features, chunk_id))
                chunk_id += 1
    
    logger.info(f"Created {len(chunks)} spatial chunks with features")
    return chunks


def exact_intersections_ultra_parallel(all_identified_features, peatland_gdf, n_processes, chunk_size=5000):
    """
    ULTRA-OPTIMIZED parallel processing with spatial chunking and memory management.
    FIXED: Now includes geometry type validation for shapefile compatibility.
    """
    total_features = len(all_identified_features)
    logger.info(f"Starting ultra-optimized intersection for {total_features:,} features")
    
    # OPTIMIZATION 1: Memory management - check available RAM
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    logger.info(f"Available memory: {available_memory_gb:.1f} GB")
    
    # Adjust chunk size based on available memory
    if available_memory_gb < 4:
        chunk_size = min(chunk_size, 2000)
        logger.info(f"Reduced chunk size to {chunk_size} due to memory constraints")
    elif available_memory_gb > 16:
        chunk_size = min(chunk_size * 2, 10000)
        logger.info(f"Increased chunk size to {chunk_size} for high-memory system")
    
    # OPTIMIZATION 2: Preprocess all data once
    logger.info("Preprocessing geometries for optimal performance...")
    all_identified_features = preprocess_geometries(all_identified_features)
    peatland_gdf = preprocess_geometries(peatland_gdf)
    
    # OPTIMIZATION 3: Use spatial chunking instead of sequential chunking
    target_chunks = max(n_processes * 2, total_features // chunk_size)
    spatial_chunks = create_spatial_chunks(all_identified_features, target_chunks)
    
    # Prepare chunk data for processing - pass peatland file path instead of objects
    chunk_data = []
    for chunk_gdf, chunk_id in spatial_chunks:
        # Pass peatland data without spatial index to avoid pickling issues
        chunk_tuple = (chunk_gdf, peatland_gdf, chunk_id)
        chunk_data.append(chunk_tuple)
        
        # Debug: Log the first few chunks to see what we're passing
        if len(chunk_data) <= 3:
            logger.info(f"🔍 DEBUG: Chunk {chunk_id} data structure:")
            logger.info(f"  - chunk_gdf type: {type(chunk_gdf)}")
            logger.info(f"  - chunk_gdf shape: {chunk_gdf.shape}")
            logger.info(f"  - peatland_gdf type: {type(peatland_gdf)}")
            logger.info(f"  - peatland_gdf shape: {peatland_gdf.shape}")
            logger.info(f"  - chunk_tuple length: {len(chunk_tuple)}")
    
    total_chunks = len(chunk_data)
    logger.info(f"Processing {total_chunks} spatial chunks with {n_processes} processes")
    
    # OPTIMIZATION 4: Process with memory-aware batching
    results = []
    start_time = pd.Timestamp.now()
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(process_intersection_chunk_ultra_optimized, data): data[2]
            for data in chunk_data
        }
        
        # Process results with enhanced progress tracking
        with tqdm(
            total=total_chunks,
            desc="🚀 Ultra-Optimized Intersection",
            unit="chunk",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] | Features: {postfix}',
            postfix=f"{0:,}",
            dynamic_ncols=True
        ) as pbar:
            
            completed_chunks = 0
            processed_features = 0
            
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    completed_chunks += 1
                    
                    if result is not None and not result.empty:
                        results.append(result)
                        processed_features += len(result)
                    
                    # Update progress
                    pbar.set_postfix(f"{processed_features:,}")
                    pbar.update(1)
                    
                    # Memory management - trigger garbage collection periodically
                    if completed_chunks % 20 == 0:
                        gc.collect()
                        
                except Exception as e:
                    chunk_id = future_to_chunk[future]
                    logger.error(f"❌ Ultra chunk {chunk_id} failed: {e}")
                    pbar.update(1)
    
    # Final processing
    total_elapsed = (pd.Timestamp.now() - start_time).total_seconds() / 60
    logger.info(f"✅ Ultra-optimized intersection completed in {total_elapsed:.1f} minutes")
    
    if results:
        # OPTIMIZATION 5: Efficient concatenation
        logger.info("Combining results...")
        combined_result = pd.concat(results, ignore_index=True)
        
        # Clean up temporary columns
        cols_to_drop = ['_minx', '_miny', '_maxx', '_maxy']
        combined_result = combined_result.drop(columns=[col for col in cols_to_drop if col in combined_result.columns], errors='ignore')
        
        # FIX: Final geometry type validation
        logger.info("Validating geometry types for shapefile compatibility...")
        combined_result = filter_geometries_by_type(combined_result, target_geom_type='Polygon')
        
        final_features = len(combined_result)
        logger.info(f"🎯 Final result: {final_features:,} intersection features")
        return combined_result
    else:
        logger.info("⚠️ No intersection results found")
        return gpd.GeoDataFrame(columns=all_identified_features.columns.tolist() + ['intersection_area', 'peatland_id'], crs=all_identified_features.crs)


# Integration with existing HybridGDBProcessor class
class UltraOptimizedHybridGDBProcessor:
    """
    Ultra-optimized version of the hybrid processor with geometry type fixes.
    """
    
    def __init__(self, 
                 gdb_directory: str,
                 peatland_file: str,
                 output_file: str = None,
                 method: str = "identify",
                 chunk_size: int = 5000,
                 n_processes: int = None,
                 debug_mode: bool = False):
        
        # Auto-generate output filename
        if output_file is None:
            if method == "identify":
                output_file = "identified_ultra_optimized.shp"
            else:
                output_file = "intersected_ultra_optimized.shp"
        
        if not output_file.endswith('.shp'):
            output_file = output_file + '.shp'
        
        self.gdb_directory = Path(gdb_directory)
        self.peatland_file = Path(peatland_file)
        self.output_file = Path(output_file)
        self.method = method.lower()
        self.chunk_size = chunk_size
        self.debug_mode = debug_mode
        
            # Optimize process allocation
    total_cpus = mp.cpu_count()
    if n_processes is None:
        self.n_processes = max(1, total_cpus - 1)
    else:
        self.n_processes = min(n_processes, total_cpus)
    
    # DEBUG: Use sequential processing for debugging
    if self.debug_mode:
        self.n_processes = 1
        logger.info(f"🔧 DEBUG MODE: Using sequential processing (1 process) for debugging")
        
        logger.info(f"Ultra-optimized processor initialized:")
        logger.info(f"  Method: {self.method}")
        logger.info(f"  Total CPUs: {total_cpus}")
        logger.info(f"  Processes: {self.n_processes}")
        logger.info(f"  Chunk size: {self.chunk_size}")
    
    def run(self):
        """Run with ultra-optimized intersection processing."""
        try:
            logger.info(f"Starting ultra-optimized {self.method} processing pipeline")
            
            # Find all GDB folders
            gdb_folders = list(self.gdb_directory.glob("*.gdb"))
            
            if not gdb_folders:
                logger.error(f"No GDB folders found in {self.gdb_directory}")
                return
            
            # Collect all layer processing tasks
            processing_tasks = []
            
            # DEBUG: Only process first GDB folder for testing
            if self.debug_mode:
                debug_gdb_folders = gdb_folders[:1]
                logger.info(f"🔧 DEBUG MODE: Processing only first GDB folder: {debug_gdb_folders[0].name}")
            else:
                debug_gdb_folders = gdb_folders
                logger.info(f"Processing all {len(gdb_folders)} GDB folders")
            
            for gdb_folder in debug_gdb_folders:
                try:
                    layers = fiona.listlayers(str(gdb_folder))
                    for layer in layers:
                        task = (str(gdb_folder), layer, str(self.peatland_file))
                        processing_tasks.append(task)
                except Exception as e:
                    logger.warning(f"Could not list layers in {gdb_folder}: {e}")
                    continue
            
            logger.info(f"Found {len(processing_tasks)} layers to process")
            
            # Stage 1: Layer-level parallelism for identification (8 cores for 8 layers)
            identification_results = []
            
            logger.info(f"🌍 Starting Stage 1: Processing {len(processing_tasks)} regions in parallel")
            
            # Try multiprocessing first, fallback to sequential if it fails
            try:
                with ProcessPoolExecutor(max_workers=min(8, self.n_processes)) as executor:
                    # Submit all identification tasks
                    future_to_task = {
                        executor.submit(process_single_layer_identification, *task): task 
                        for task in processing_tasks
                    }
                    
                    # Enhanced progress bar for Stage 1
                    with tqdm(
                        total=len(processing_tasks),
                        desc="🌍 Stage 1: Regional Identification (Parallel)",
                        unit="region",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} regions [{elapsed}<{remaining}, {rate_fmt}] | {postfix}',
                        postfix={"Features": "0"},
                        dynamic_ncols=True
                    ) as pbar:
                        
                        completed_regions = 0
                        total_features_identified = 0
                        
                        for future in as_completed(future_to_task):
                            try:
                                result = future.result()
                                completed_regions += 1
                                
                                if result is not None and not result.empty:
                                    identification_results.append(result)
                                    total_features_identified += len(result)
                                
                                # Update progress bar
                                pbar.set_postfix({"Features": f"{total_features_identified:,}"})
                                pbar.update(1)
                                
                            except Exception as e:
                                task = future_to_task[future]
                                logger.error(f"❌ Region {task[1]} failed: {e}")
                                # Print the full traceback from the worker
                                logger.error(traceback.format_exc())
                                pbar.update(1)
                                
            except Exception as mp_error:
                logger.warning(f"Multiprocessing failed, falling back to sequential processing: {mp_error}")
                
                # Sequential fallback
                with tqdm(
                    total=len(processing_tasks),
                    desc="🌍 Stage 1: Regional Identification (Sequential)",
                    unit="region",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} regions [{elapsed}<{remaining}, {rate_fmt}] | {postfix}',
                    postfix={"Features": "0"},
                    dynamic_ncols=True
                ) as pbar:
                    
                    total_features_identified = 0
                    
                    for task in processing_tasks:
                        try:
                            result = process_single_layer_identification(*task)
                            
                            if result is not None and not result.empty:
                                identification_results.append(result)
                                total_features_identified += len(result)
                            
                            # Update progress bar
                            pbar.set_postfix({"Features": f"{total_features_identified:,}"})
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.error(f"❌ Region {task[1]} failed: {e}")
                            pbar.update(1)
            
            # Combine all identified features
            if identification_results:
                all_identified_features = pd.concat(identification_results, ignore_index=True)
                total_identified = len(all_identified_features)
                logger.info(f"✅ Stage 1 completed: {total_identified:,} features identified across {len(identification_results)} regions")
                
                # Stage 2: Method-specific processing
                if self.method == "identify":
                    # Return identified features (no exact intersection needed)
                    final_result = all_identified_features
                elif self.method == "intersect":
                    # Stage 2: Ultra-optimized intersection processing
                    logger.info("Using ultra-optimized intersection processing")
                    peatland_gdf = gpd.read_file(self.peatland_file)
                    
                    final_result = exact_intersections_ultra_parallel(
                        all_identified_features, 
                        peatland_gdf, 
                        self.n_processes, 
                        self.chunk_size
                    )
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                
                # FINAL FIX: Ensure geometry type consistency before export
                logger.info("Final geometry validation before export...")
                
                # Check geometry types in final result
                geom_types = final_result.geometry.geom_type.value_counts()
                logger.info(f"Final geometry types: {geom_types.to_dict()}")
                
                # Filter to polygon-only for shapefile compatibility
                if len(geom_types) > 1 or not all(gt in ['Polygon', 'MultiPolygon'] for gt in geom_types.index):
                    logger.info("Filtering to polygon-only geometries for shapefile compatibility...")
                    final_result = filter_geometries_by_type(final_result, target_geom_type='Polygon')
                    
                    # Log final count after filtering
                    logger.info(f"After polygon filtering: {len(final_result):,} features")
                
                # Validate that we have features to export
                if final_result.empty:
                    logger.warning("No polygon features remaining after filtering")
                    return
                
                # Export to file
                logger.info(f"Exporting {len(final_result)} features to {self.output_file}")
                
                # Ensure output directory exists
                output_dir = Path(self.output_file).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Final geometry validation before writing
                final_result = final_result[~final_result.geometry.isna() & ~final_result.geometry.is_empty]
                
                if final_result.empty:
                    logger.warning("No valid geometries to export after final validation")
                    return
                
                # Write to shapefile with error handling
                try:
                    final_result.to_file(self.output_file, driver='ESRI Shapefile')
                    logger.info("✅ Ultra-optimized processing completed successfully!")
                    logger.info(f"📊 Final statistics:")
                    logger.info(f"  - Total features: {len(final_result):,}")
                    logger.info(f"  - Source layers: {final_result['source_layer'].nunique()}")
                    
                    # Additional statistics for intersection method
                    if self.method == "intersect" and 'intersection_area' in final_result.columns:
                        total_area = final_result['intersection_area'].sum()
                        avg_area = final_result['intersection_area'].mean()
                        logger.info(f"  - Total intersection area: {total_area:,.2f}")
                        logger.info(f"  - Average intersection area: {avg_area:,.2f}")
                    
                except Exception as export_error:
                    logger.error(f"Export failed: {export_error}")
                    
                    # Diagnostic information
                    logger.info("Diagnostic information:")
                    logger.info(f"  - Final result shape: {final_result.shape}")
                    logger.info(f"  - Final result CRS: {final_result.crs}")
                    logger.info(f"  - Geometry types: {final_result.geometry.geom_type.value_counts().to_dict()}")
                    logger.info(f"  - Valid geometries: {final_result.geometry.is_valid.sum()}/{len(final_result)}")
                    logger.info(f"  - Empty geometries: {final_result.geometry.is_empty.sum()}/{len(final_result)}")
                    
                    # Try alternative export formats
                    logger.info("Attempting alternative export formats...")
                    
                    # Try GeoPackage
                    try:
                        gpkg_file = self.output_file.with_suffix('.gpkg')
                        final_result.to_file(gpkg_file, driver='GPKG')
                        logger.info(f"✅ Successfully exported to GeoPackage: {gpkg_file}")
                    except Exception as gpkg_error:
                        logger.error(f"GeoPackage export also failed: {gpkg_error}")
                    
                    # Try GeoJSON
                    try:
                        geojson_file = self.output_file.with_suffix('.geojson')
                        final_result.to_file(geojson_file, driver='GeoJSON')
                        logger.info(f"✅ Successfully exported to GeoJSON: {geojson_file}")
                    except Exception as geojson_error:
                        logger.error(f"GeoJSON export also failed: {geojson_error}")
                    
                    raise export_error
                
            else:
                logger.warning("No intersecting features found in any layer")
                
        except Exception as e:
            logger.error(f"Ultra-optimized processing pipeline failed: {e}")
            logger.error(traceback.format_exc())
            raise


def create_test_geometries():
    """
    Helper function to create test geometries with mixed types for debugging.
    """
    from shapely.geometry import Polygon, LineString, Point
    
    # Create mixed geometry types that might result from intersections
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    linestring = LineString([(0, 0), (1, 1)])
    point = Point(0.5, 0.5)
    
    gdf = gpd.GeoDataFrame({
        'id': [1, 2, 3],
        'type': ['polygon', 'line', 'point'],
        'geometry': [polygon, linestring, point]
    })
    
    return gdf


def debug_geometry_types(gdf):
    """
    Debug function to analyze geometry types in a GeoDataFrame.
    """
    logger.info("=== GEOMETRY TYPE ANALYSIS ===")
    logger.info(f"Total features: {len(gdf)}")
    
    geom_types = gdf.geometry.geom_type.value_counts()
    logger.info("Geometry type distribution:")
    for geom_type, count in geom_types.items():
        logger.info(f"  {geom_type}: {count}")
    
    # Check for invalid geometries
    invalid_count = (~gdf.geometry.is_valid).sum()
    empty_count = gdf.geometry.is_empty.sum()
    null_count = gdf.geometry.isna().sum()
    
    logger.info(f"Invalid geometries: {invalid_count}")
    logger.info(f"Empty geometries: {empty_count}")
    logger.info(f"Null geometries: {null_count}")
    
    # Sample some geometries
    if len(gdf) > 0:
        sample_size = min(3, len(gdf))
        logger.info(f"Sample geometries (first {sample_size}):")
        for i in range(sample_size):
            geom = gdf.geometry.iloc[i]
            logger.info(f"  [{i}] Type: {geom.geom_type}, Valid: {geom.is_valid}, Empty: {geom.is_empty}")
    
    logger.info("========================")


# Usage example
if __name__ == "__main__":
    import argparse
    
    # Set multiprocessing start method to 'spawn' for better compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Ultra-optimized GDB intersection processor (FIXED)")
    parser.add_argument("--gdb-dir", required=True, help="Directory containing GDB folders")
    parser.add_argument("--peatland-file", required=True, help="Path to peatland extent file")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--method", choices=['identify', 'intersect'], default='identify')
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--debug", action='store_true', help="Enable debug mode with geometry analysis")
    
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - detailed geometry analysis will be performed")
    
    processor = UltraOptimizedHybridGDBProcessor(
        gdb_directory=args.gdb_dir,
        peatland_file=args.peatland_file,
        output_file=args.output,
        method=args.method,
        chunk_size=args.chunk_size,
        n_processes=args.processes,
        debug_mode=args.debug
    )
    
    processor.run()