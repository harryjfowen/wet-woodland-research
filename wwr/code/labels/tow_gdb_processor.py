#!/usr/bin/env python3
"""
Enhanced GDB Intersection Processor - V2 with Woodland Type Encoding (4 classes)

DESCRIPTION: Enhanced version of the GDB processor that adds woodland type encoding and polygon
simplification. Extracts 'Type of Woodland' column and converts to integer codes:
- 0: Other/Unknown
- 1: Lone Tree
- 2: Group of Trees
- 3: Small Woodland
- 4: Woodland

Also includes polygon simplification to reduce file sizes and improve performance.
"""

import os
import logging
from datetime import datetime
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
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global cache so each worker only loads & indexes the peatland layer once
# -----------------------------------------------------------------------------
_PEAT_CACHE = {}  # key = file path str; value = (geodataframe)


def encode_woodland_type(woodland_type_series):
    """
    Encode woodland type strings to integer codes.

    Parameters:
    - woodland_type_series: Pandas Series with woodland type strings

    Returns:
    - Series with integer codes:
      0=Other/Unknown, 1=Lone Tree, 2=Group of Trees, 3=Small Woodland,
      4=NFI OHC (Fringe Woodland), 5=NFI Woodland Shapefile
    """
    # Create mapping dictionary
    woodland_mapping = {
        'Lone Tree': 1,
        'Group of Trees': 2,
        'Small Woodland': 3,
        'Woodland': 4,
        'NFI OHC': 4
    }

    # Convert to lowercase for case-insensitive matching
    normalized_series = woodland_type_series.astype(str).str.lower().str.strip()

    # Create mapping for case-insensitive lookup
    normalized_mapping = {k.lower(): v for k, v in woodland_mapping.items()}

    # Map values, defaulting to 0 for unknown/other types
    encoded = normalized_series.map(normalized_mapping).fillna(0).astype(int)

    # Log the mapping statistics
    unique_values = woodland_type_series.value_counts()
    encoded_values = encoded.value_counts().sort_index()

    logger.info(f"Woodland type encoding statistics:")
    logger.info(f"  Original unique values: {len(unique_values)}")
    for original, count in unique_values.head(10).items():
        logger.info(f"    '{original}': {count}")

    logger.info(f"  Encoded distribution:")
    code_names = {
        0: 'Unknown',
        1: 'Lone Tree',
        2: 'Group of Trees',
        3: 'Small Woodland',
        4: 'OHC (Fringe Woodland)',
        5: 'NFI Woodland'
    }
    for code, count in encoded_values.items():
        logger.info(f"    {code} ({code_names.get(code, 'Unknown')}): {count}")

    return encoded


def simplify_geometries(gdf, tolerance=1.0):
    """
    Simplify polygon geometries to reduce complexity.

    Parameters:
    - gdf: GeoDataFrame with geometries to simplify
    - tolerance: Simplification tolerance in map units (default: 1.0 meter)

    Returns:
    - GeoDataFrame with simplified geometries
    """
    logger.info(f"Simplifying geometries with tolerance {tolerance}m...")

    # Count original vertices
    original_vertices = 0
    for geom in gdf.geometry:
        if hasattr(geom, 'exterior'):
            original_vertices += len(geom.exterior.coords)
        elif hasattr(geom, 'geoms'):  # MultiPolygon
            for sub_geom in geom.geoms:
                if hasattr(sub_geom, 'exterior'):
                    original_vertices += len(sub_geom.exterior.coords)

    # Simplify geometries
    simplified_geoms = gdf.geometry.simplify(tolerance, preserve_topology=True)

    # Count simplified vertices
    simplified_vertices = 0
    for geom in simplified_geoms:
        if hasattr(geom, 'exterior'):
            simplified_vertices += len(geom.exterior.coords)
        elif hasattr(geom, 'geoms'):  # MultiPolygon
            for sub_geom in geom.geoms:
                if hasattr(sub_geom, 'exterior'):
                    simplified_vertices += len(sub_geom.exterior.coords)

    # Update GeoDataFrame
    gdf_simplified = gdf.copy()
    gdf_simplified.geometry = simplified_geoms

    # Remove any invalid geometries that might result from simplification
    gdf_simplified = gdf_simplified[gdf_simplified.geometry.is_valid & ~gdf_simplified.geometry.is_empty]

    reduction_percent = ((original_vertices - simplified_vertices) / original_vertices * 100) if original_vertices > 0 else 0

    logger.info(f"Simplification results:")
    logger.info(f"  Original vertices: {original_vertices:,}")
    logger.info(f"  Simplified vertices: {simplified_vertices:,}")
    logger.info(f"  Reduction: {reduction_percent:.1f}%")
    logger.info(f"  Features after simplification: {len(gdf_simplified)}/{len(gdf)}")

    return gdf_simplified


def process_woodland_features(gdf, simplify_tolerance=1.0):
    """
    Process woodland features by encoding types and simplifying geometries.
    Keeps only essential columns plus the new woodland type code.

    Parameters:
    - gdf: GeoDataFrame with woodland features
    - simplify_tolerance: Tolerance for geometry simplification

    Returns:
    - Simplified GeoDataFrame with only woodland_type_code column and geometry
    """
    # Check for woodland type column (case-insensitive)
    woodland_type_col = None
    for col in gdf.columns:
        if 'type' in col.lower() and 'woodland' in col.lower():
            woodland_type_col = col
            break

    if woodland_type_col:
        logger.info(f"Found woodland type column: '{woodland_type_col}'")
        # Encode woodland types
        woodland_type_code = encode_woodland_type(gdf[woodland_type_col])
    else:
        logger.warning("No 'Type of Woodland' column found. Setting all features to code 4 (Woodland)")
        available_cols = list(gdf.columns)
        logger.info(f"Available columns: {available_cols}")
        woodland_type_code = pd.Series([4] * len(gdf), index=gdf.index)

    # Create minimal GeoDataFrame with only essential columns
    gdf_minimal = gpd.GeoDataFrame({
        'woodland_type_code': woodland_type_code,
        'geometry': gdf.geometry
    }, crs=gdf.crs)

    # Simplify geometries if tolerance > 0
    if simplify_tolerance > 0:
        gdf_minimal = simplify_geometries(gdf_minimal, simplify_tolerance)

    logger.info(f"Reduced from {len(gdf.columns)} columns to {len(gdf_minimal.columns)} columns")
    logger.info(f"Kept columns: {list(gdf_minimal.columns)}")

    return gdf_minimal


def _dissolve_chunk_group(chunk_list):
    """
    Dissolve all chunks in a group and return their union.
    This function must be at module level to be picklable for multiprocessing.
    """
    from shapely.ops import unary_union

    group_geoms = []
    for chunk_gdf in chunk_list:
        if not chunk_gdf.empty:
            try:
                chunk_union = unary_union(chunk_gdf.geometry.tolist())
                if not chunk_union.is_empty:
                    group_geoms.append(chunk_union)
            except Exception as e:
                logger.warning(f"Failed to dissolve chunk with {len(chunk_gdf)} features: {e}")
                # Try individual geometries if batch fails
                for geom in chunk_gdf.geometry:
                    if not geom.is_empty:
                        group_geoms.append(geom)

    # Union all geometries in this group
    if group_geoms:
        try:
            return unary_union(group_geoms)
        except Exception as e:
            logger.warning(f"Failed to union group geometries: {e}")
            return None
    return None


def _parallel_spatial_dissolve(gdf):
    """
    Parallel spatial dissolve for very large datasets.
    Splits data into spatial chunks and dissolves each chunk in parallel.
    """
    from shapely.ops import unary_union
    from multiprocessing import Pool, cpu_count

    # Determine number of processes (use available cores but not more than 8 for memory reasons)
    n_processes = min(cpu_count() - 2, len(gdf) // 50000)  # Use all cores - 2, at least 50K features per process
    n_processes = max(2, n_processes)  # At least 2 processes

    logger.info(f"Using {n_processes} processes for parallel spatial dissolve")

    # Create spatial chunks using the same logic as rasterization
    spatial_chunks = _create_spatial_chunks_for_dissolve(gdf, n_processes * 2)  # More chunks than processes

    if len(spatial_chunks) <= n_processes:
        # If we don't have enough chunks, process sequentially but still in parallel groups
        chunk_groups = [spatial_chunks[i::n_processes] for i in range(n_processes)]
        chunk_groups = [group for group in chunk_groups if group]  # Remove empty groups
    else:
        # Group chunks for processing
        chunk_groups = [spatial_chunks[i::n_processes] for i in range(n_processes)]
        chunk_groups = [group for group in chunk_groups if group]

    logger.info(f"Created {len(spatial_chunks)} spatial chunks grouped into {len(chunk_groups)} process groups")

    # Process groups in parallel with progress tracking
    group_results = [None] * len(chunk_groups)
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submit all dissolve tasks
        future_to_index = {
            executor.submit(_dissolve_chunk_group, chunk_group): i
            for i, chunk_group in enumerate(chunk_groups)
        }

        # Progress bar for dissolve operations
        with tqdm(
            total=len(chunk_groups),
            desc="🔄 NFI Dissolve Progress",
            unit="group",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} groups [{elapsed}<{remaining}, {rate_fmt}]',
            dynamic_ncols=True
        ) as pbar:

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    group_results[index] = result
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"❌ Dissolve group {index} failed: {e}")
                    group_results[index] = None
                    pbar.update(1)

    # Filter out None results and union all group results
    valid_results = [result for result in group_results if result is not None and not result.is_empty]

    if not valid_results:
        logger.error("All parallel dissolve operations failed")
        return None

    try:
        final_union = unary_union(valid_results)
        logger.info(f"Parallel dissolve completed: {len(valid_results)} group unions combined")
        return final_union
    except Exception as e:
        logger.error(f"Failed to union group results: {e}")
        # Fallback: return the first valid result
        return valid_results[0] if valid_results else None


def _create_spatial_chunks_for_dissolve(gdf, n_chunks):
    """
    Create spatial chunks optimized for parallel dissolve operations.
    """
    # Get spatial bounds
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    # Create a grid for chunking (simpler than the rasterization version)
    n_cols = int(np.sqrt(n_chunks))
    n_rows = int(np.ceil(n_chunks / n_cols))

    x_step = (maxx - minx) / n_cols
    y_step = (maxy - miny) / n_rows

    chunks = []

    for i in range(n_rows):
        for j in range(n_cols):
            chunk_minx = minx + j * x_step
            chunk_maxx = minx + (j + 1) * x_step
            chunk_miny = miny + i * y_step
            chunk_maxy = miny + (i + 1) * y_step

            # Select features in this spatial chunk
            mask = (
                (gdf.geometry.bounds.minx < chunk_maxx) &
                (gdf.geometry.bounds.maxx > chunk_minx) &
                (gdf.geometry.bounds.miny < chunk_maxy) &
                (gdf.geometry.bounds.maxy > chunk_miny)
            )

            chunk_gdf = gdf[mask].copy()
            if not chunk_gdf.empty:
                chunks.append(chunk_gdf)

    logger.info(f"Created {len(chunks)} spatial chunks for dissolve")
    return chunks


def fast_dissolve_nfi(gdf, dissolve_field=None, simplify_tolerance=1.0):
    """
    Fast optimized dissolve function for NFI shapefile to remove overlaps.

    Parameters:
    - gdf: GeoDataFrame to dissolve
    - dissolve_field: Field to dissolve on (None = dissolve all into single geometry)
    - simplify_tolerance: Simplification tolerance in map units

    Returns:
    - Dissolved GeoDataFrame
    """
    logger.info(f"Starting fast dissolve of {len(gdf)} NFI features...")

    if gdf.empty:
        return gdf

    # Pre-process geometries for better dissolve performance
    gdf = gdf.copy()
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]

    if gdf.empty:
        logger.warning("No valid geometries to dissolve")
        return gdf

    # Fix invalid geometries
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        logger.info(f"Fixing {invalid_mask.sum()} invalid geometries...")
        gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)

    # CRITICAL OPTIMIZATION: Simplify individual geometries BEFORE dissolving
    # This prevents dissolve from creating massive complex geometries that are slow to simplify
    if simplify_tolerance > 0 and not gdf.empty:
        logger.info(f"Pre-simplifying {len(gdf)} individual geometries with tolerance {simplify_tolerance}m...")

        # For very large datasets, use batch processing to avoid memory issues
        if len(gdf) > 50000:
            logger.info(f"Large dataset ({len(gdf):,} features) - using memory-efficient batch processing...")

            # For extremely large datasets, consider skipping simplification if tolerance is small
            if len(gdf) > 500000 and simplify_tolerance < 10.0:
                logger.warning(f"Extremely large dataset ({len(gdf):,} features) with fine tolerance ({simplify_tolerance}m)")
                logger.warning("Skipping pre-simplification to prioritize speed - dissolve will handle overlaps")
                # Skip simplification entirely for massive datasets with fine tolerances
            else:
                # Process in larger chunks to reduce overhead
                chunk_size = 100000  # Increased from 50,000
                simplified_geoms = []

                # Progress bar for batch processing
                with tqdm(
                    total=len(gdf),
                    desc="🔧 NFI Pre-simplification",
                    unit="geoms",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    dynamic_ncols=True
                ) as pbar:

                    for start_idx in range(0, len(gdf), chunk_size):
                        end_idx = min(start_idx + chunk_size, len(gdf))
                        chunk = gdf.iloc[start_idx:end_idx]

                        # Simplify this chunk
                        chunk_simplified = []
                        for geom in chunk.geometry:
                            try:
                                simplified = geom.simplify(simplify_tolerance, preserve_topology=True)
                                # Only keep if simplification didn't make it empty or invalid
                                if not simplified.is_empty and simplified.is_valid:
                                    chunk_simplified.append(simplified)
                                else:
                                    chunk_simplified.append(geom)  # Keep original
                            except Exception:
                                chunk_simplified.append(geom)  # Keep original

                        simplified_geoms.extend(chunk_simplified)
                        pbar.update(len(chunk))

                gdf['geometry'] = simplified_geoms
        else:
            # Standard processing for smaller datasets with progress bar
            simplified_geoms = []
            with tqdm(
                total=len(gdf),
                desc="🔧 NFI Pre-simplification",
                unit="geoms",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                dynamic_ncols=True
            ) as pbar:
                for geom in gdf.geometry:
                    try:
                        simplified = geom.simplify(simplify_tolerance, preserve_topology=True)
                        # Only keep if simplification didn't make it empty or invalid
                        if not simplified.is_empty and simplified.is_valid:
                            simplified_geoms.append(simplified)
                        else:
                            simplified_geoms.append(geom)  # Keep original
                    except Exception:
                        simplified_geoms.append(geom)  # Keep original

                    pbar.update(1)

            gdf['geometry'] = simplified_geoms

        logger.info(f"Pre-simplification completed for {len(gdf)} features")

    # Fast dissolve using unary_union for all geometries (most efficient for full dissolve)
    if dissolve_field is None or dissolve_field not in gdf.columns:
        logger.info("Performing full dissolve (no dissolve field specified)...")

        # For very large datasets, use parallel spatial chunking for dissolve
        if len(gdf) > 100000:
            logger.info(f"Large dataset ({len(gdf):,} features) - using parallel spatial dissolve...")
            dissolved_geom = _parallel_spatial_dissolve(gdf)
        else:
            # Standard dissolve for smaller datasets
            try:
                from shapely.ops import unary_union
                dissolved_geom = unary_union(gdf.geometry.tolist())
            except Exception as e:
                logger.warning(f"Unary union failed, falling back to dissolve: {e}")
                dissolved_gdf = gdf.dissolve()
                dissolved_geom = dissolved_gdf.geometry.iloc[0] if not dissolved_gdf.empty else None

        # Convert back to GeoDataFrame
        if dissolved_geom is None or dissolved_geom.is_empty:
            logger.warning("Dissolved geometry is empty")
            return gpd.GeoDataFrame(columns=['woodland_type_code', 'source_layer', 'geometry'], crs=gdf.crs)

        # Handle both single geometry and MultiGeometry results
        if dissolved_geom.geom_type in ['Polygon', 'MultiPolygon']:
            dissolved_gdf = gpd.GeoDataFrame({
                'geometry': [dissolved_geom],
                'woodland_type_code': [5],  # NFI woodland (distinct from OHC class 4)
                'source_layer': ['NFI_dissolved']
            }, crs=gdf.crs)
        else:
            logger.warning(f"Unexpected dissolved geometry type: {dissolved_geom.geom_type}")
            return gpd.GeoDataFrame(columns=['woodland_type_code', 'source_layer', 'geometry'], crs=gdf.crs)

    else:
        logger.info(f"Performing dissolve by field '{dissolve_field}'...")
        dissolved_gdf = gdf.dissolve(by=dissolve_field)

        # Reset index to make dissolve field a column again
        dissolved_gdf = dissolved_gdf.reset_index()

    # Light post-dissolve simplification (since we pre-simplified)
    if simplify_tolerance > 0 and not dissolved_gdf.empty:
        logger.info(f"Post-dissolve light simplification with tolerance {simplify_tolerance}m...")
        dissolved_gdf['geometry'] = dissolved_gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)

        # Remove any resulting empty geometries
        dissolved_gdf = dissolved_gdf[~dissolved_gdf.geometry.is_empty]

    final_count = len(dissolved_gdf)

    # Skip expensive area calculations for very large datasets to improve performance
    if len(gdf) > 100000:
        logger.info(f"Dissolve completed: {len(gdf):,} → {final_count} features (area calculation skipped for performance)")
    else:
        logger.info(f"Dissolve completed: {len(gdf)} → {final_count} features")
        logger.info(f"Area reduction: {gdf.geometry.area.sum():,.0f} → {dissolved_gdf.geometry.area.sum():,.0f} m²")

    return dissolved_gdf


def process_nfi_shapefile(nfi_path, simplify_tolerance=1.0):
    """
    Process NFI shapefile with dissolve and woodland type encoding.
    Optimized for large NFI datasets with memory-efficient processing.

    Parameters:
    - nfi_path: Path to NFI shapefile
    - simplify_tolerance: Simplification tolerance for dissolved geometry

    Returns:
    - Processed GeoDataFrame with woodland_type_code = 5
    """
    logger.info(f"Processing NFI shapefile: {nfi_path}")

    try:
        # Load NFI shapefile with memory optimization
        logger.info("Loading NFI shapefile...")
        gdf = gpd.read_file(nfi_path)

        # Early exit for empty files
        if gdf.empty:
            logger.warning("NFI shapefile is empty")
            return gpd.GeoDataFrame(columns=['woodland_type_code', 'source_layer', 'geometry'], crs=None)

        logger.info(f"Loaded NFI shapefile with {len(gdf):,} features")

        # OPTIMIZATION: For very large NFI files, we can skip some preprocessing
        if len(gdf) > 1000000:
            logger.info("Large NFI dataset detected - using ultra-fast processing mode")
            # Skip detailed geometry validation for massive datasets
            # Just ensure basic validity
            gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
        else:
            # Standard processing with full validation
            gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]

            # Fix invalid geometries only for smaller datasets
            invalid_mask = ~gdf.geometry.is_valid
            if invalid_mask.any():
                logger.info(f"Fixing {invalid_mask.sum()} invalid geometries...")
                gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)

        # Fast dissolve to remove overlaps
        # Use same tolerance as GDB layers for consistency, unless user explicitly wants NFI optimization
        logger.info(f"Using tolerance {simplify_tolerance}m for NFI processing (same as GDB layers)")
        dissolved_gdf = fast_dissolve_nfi(gdf, simplify_tolerance=simplify_tolerance)

        if dissolved_gdf.empty:
            logger.warning("Dissolved NFI shapefile is empty")
            return gpd.GeoDataFrame(columns=['woodland_type_code', 'source_layer', 'geometry'], crs=gdf.crs)

        # Add woodland type code for NFI (woodland = 5, distinct from OHC class 4)
        dissolved_gdf['woodland_type_code'] = 5
        dissolved_gdf['source_layer'] = 'NFI_dissolved'

        logger.info(f"NFI processing complete: {len(dissolved_gdf)} dissolved features")
        return dissolved_gdf

    except Exception as e:
        logger.error(f"Failed to process NFI shapefile: {e}")
        return gpd.GeoDataFrame(columns=['woodland_type_code', 'source_layer', 'geometry'], crs=None)


def export_to_raster(gdf, output_path, resolution=10.0, woodland_type_col='woodland_type_code', n_cores=None):
    """
    Optimized export of GeoDataFrame to raster format with memory-efficient chunked processing.

    Parameters:
    - gdf: GeoDataFrame with woodland features
    - output_path: Path for output raster file
    - resolution: Pixel size in map units (default: 10.0 meters)
    - woodland_type_col: Column containing woodland type codes

    Returns:
    - Success boolean
    """
    logger.info(f"Rasterizing {len(gdf)} features to {resolution}m resolution raster...")

    if gdf.empty:
        logger.warning("No features to rasterize")
        return False

    # Get bounds and calculate raster dimensions
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    # Calculate raster dimensions
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    logger.info(f"Raster dimensions: {width} x {height} pixels")
    logger.info(f"Coverage area: {(maxx-minx)/1000:.1f} x {(maxy-miny)/1000:.1f} km")

    # Set default cores if not specified
    if n_cores is None:
        import multiprocessing as mp
        n_cores = max(1, mp.cpu_count() - 2)

    # Estimate memory usage and determine if chunking is needed
    estimated_memory_gb = (width * height * 1) / (1024**3)  # uint8 = 1 byte per pixel
    logger.info(f"Estimated memory usage: {estimated_memory_gb:.2f} GB")

    # Create transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Memory-efficient processing based on raster size
    if estimated_memory_gb > 8.0 or len(gdf) > 10_000_000:
        logger.info(f"Using parallel chunked rasterization with {n_cores} cores...")
        return _export_to_raster_chunked_parallel(gdf, output_path, width, height, transform, woodland_type_col, n_cores)
    else:
        logger.info("Using standard rasterization...")
        return _export_to_raster_standard(gdf, output_path, width, height, transform, woodland_type_col)


def _export_to_raster_standard(gdf, output_path, width, height, transform, woodland_type_col):
    """Standard rasterization for smaller datasets with enhanced polygon coverage."""
    logger.info("Preparing geometries for robust rasterization...")

    # Validate and fix geometries
    gdf = gdf.copy()
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]

    if gdf.empty:
        logger.warning("No valid geometries to rasterize")
        # Create empty raster
        raster = np.full((height, width), 255, dtype=np.uint8)
        return _write_raster_and_stats(raster, output_path, width, height, transform, gdf.crs)

    # Fix invalid geometries
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        logger.info(f"Fixing {invalid_mask.sum()} invalid geometries...")
        gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)

    # For very thin polygons, add a small buffer to ensure coverage
    # This helps with linear features and very narrow polygons
    buffer_distance = 0.1  # 10cm buffer to ensure minimum coverage
    logger.info(f"Applying {buffer_distance}m buffer to ensure polygon coverage...")

    buffered_geoms = []
    for geom in gdf.geometry:
        if geom is not None and not geom.is_empty:
            # Buffer to ensure minimum width, but don't change area significantly
            buffered = geom.buffer(buffer_distance, cap_style=2, join_style=2)  # Square caps, miter joins
            buffered_geoms.append(buffered)
        else:
            buffered_geoms.append(geom)

    gdf['geometry'] = buffered_geoms

    # Prepare geometries and values for rasterization
    if woodland_type_col in gdf.columns:
        shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[woodland_type_col]) if geom is not None and not geom.is_empty]
    else:
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]

    logger.info(f"Rasterizing {len(shapes)} geometries with enhanced coverage...")

    # Use multiple rasterization passes for better coverage
    raster = np.full((height, width), 255, dtype=np.uint8)  # Initialize with nodata

    if shapes:
        # First pass: standard rasterization with all_touched=True
        temp_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
            fill=255,
            dtype=np.uint8,
            all_touched=True
        )

        # Second pass: use higher resolution intermediate rasterization for thin features
        # This helps catch features that might be missed at the target resolution
        fine_resolution = 2.0  # 2m intermediate resolution
        fine_width = int(np.ceil(width * 10.0 / fine_resolution))
        fine_height = int(np.ceil(height * 10.0 / fine_resolution))

        if fine_width > 0 and fine_height > 0:
            logger.info(f"Using intermediate {fine_resolution}m rasterization for thin features...")
            fine_transform = from_bounds(
                transform.c, transform.f - fine_height * fine_resolution,
                transform.c + fine_width * fine_resolution, transform.f,
                fine_width, fine_height
            )

            fine_raster = rasterize(
                shapes,
                out_shape=(fine_height, fine_width),
                transform=fine_transform,
                fill=255,
                dtype=np.uint8,
                all_touched=True
            )

            # Resample fine raster back to target resolution
            from rasterio.enums import Resampling
            from rasterio.warp import reproject

            fine_raster = fine_raster.astype(np.uint8)
            reproject(
                source=fine_raster,
                destination=temp_raster,
                src_transform=fine_transform,
                src_crs=gdf.crs,
                dst_transform=transform,
                dst_crs=gdf.crs,
                resampling=Resampling.max,  # Use max to preserve woodland types
                src_nodata=255,
                dst_nodata=255
            )

        # Combine results - use temp_raster where it has data, otherwise keep original
        mask = (temp_raster != 255)
        raster[mask] = temp_raster[mask]

    return _write_raster_and_stats(raster, output_path, width, height, transform, gdf.crs)


def _export_to_raster_chunked_parallel(gdf, output_path, width, height, transform, woodland_type_col, n_cores):
    """Parallel memory-efficient chunked rasterization for large datasets."""
    logger.info(f"Initializing parallel chunked rasterization with {n_cores} cores...")

    # Initialize output raster array in memory
    logger.info("Creating output raster array...")
    output_raster = np.zeros((height, width), dtype=np.uint8)

    # Process each woodland type separately
    woodland_types = gdf[woodland_type_col].unique() if woodland_type_col in gdf.columns else [1]

    for woodland_type in sorted(woodland_types):
        if woodland_type_col in gdf.columns:
            subset = gdf[gdf[woodland_type_col] == woodland_type]
        else:
            subset = gdf
            woodland_type = 1

        if subset.empty:
            continue

        logger.info(f"Processing woodland type {woodland_type}: {len(subset):,} features with {n_cores} cores")

        # Create spatial chunks for parallel processing
        bounds = subset.total_bounds
        minx, miny, maxx, maxy = bounds

        # Create more chunks for better parallelization (ensure at least n_cores chunks)
        grid_size = max(4, int(np.ceil(np.sqrt(n_cores * 2))))  # e.g., 6x6 for 8 cores
        logger.info(f"Using {grid_size}x{grid_size} spatial grid for parallel processing")

        # Prepare chunks for parallel processing
        chunk_tasks = []
        chunk_id = 0

        for i in range(grid_size):
            for j in range(grid_size):
                chunk_minx = minx + i * (maxx - minx) / grid_size
                chunk_maxx = minx + (i + 1) * (maxx - minx) / grid_size
                chunk_miny = miny + j * (maxy - miny) / grid_size
                chunk_maxy = miny + (j + 1) * (maxy - miny) / grid_size

                # Select features in this spatial chunk
                try:
                    spatial_subset = subset.cx[chunk_minx:chunk_maxx, chunk_miny:chunk_maxy]
                    if not spatial_subset.empty:
                        chunk_tasks.append((spatial_subset, woodland_type, width, height, transform, chunk_id))
                        chunk_id += 1
                except Exception as e:
                    logger.warning(f"Skipping spatial chunk {i+1},{j+1}: {e}")
                    continue

        if not chunk_tasks:
            continue

        logger.info(f"Processing {len(chunk_tasks)} spatial chunks in parallel...")

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunk_results = list(executor.map(_rasterize_chunk_parallel, chunk_tasks))

        # Merge all chunk results into output raster
        logger.info("Merging parallel chunk results...")
        for chunk_result in chunk_results:
            if chunk_result is not None:
                np.maximum(output_raster, chunk_result, out=output_raster)

    # Write final raster to file
    logger.info("Writing final raster to file...")
    # Convert background zeros to 255 (NoData) before writing for consistent nodata handling
    output_raster[output_raster == 0] = 255
    return _write_raster_and_stats(output_raster, output_path, width, height, transform, gdf.crs)


def _export_to_raster_chunked(gdf, output_path, width, height, transform, woodland_type_col):
    """Memory-efficient chunked rasterization for large datasets."""
    logger.info("Initializing chunked rasterization...")

    # Initialize output raster array in memory
    logger.info("Creating output raster array...")
    output_raster = np.zeros((height, width), dtype=np.uint8)

    # Process each woodland type separately to reduce memory usage
    woodland_types = gdf[woodland_type_col].unique() if woodland_type_col in gdf.columns else [1]

    for woodland_type in sorted(woodland_types):
        if woodland_type_col in gdf.columns:
            subset = gdf[gdf[woodland_type_col] == woodland_type]
        else:
            subset = gdf
            woodland_type = 1

        if subset.empty:
            continue

        logger.info(f"Processing woodland type {woodland_type}: {len(subset):,} features")

        # Process in spatial chunks if dataset is very large
        if len(subset) > 5_000_000:
            logger.info(f"Large dataset - processing in spatial chunks...")
            bounds = subset.total_bounds
            minx, miny, maxx, maxy = bounds

            # Create 4x4 spatial grid for manageable chunks
            for i in range(4):
                for j in range(4):
                    chunk_minx = minx + i * (maxx - minx) / 4
                    chunk_maxx = minx + (i + 1) * (maxx - minx) / 4
                    chunk_miny = miny + j * (maxy - miny) / 4
                    chunk_maxy = miny + (j + 1) * (maxy - miny) / 4

                    # Select features in this spatial chunk
                    try:
                        spatial_subset = subset.cx[chunk_minx:chunk_maxx, chunk_miny:chunk_maxy]
                        if not spatial_subset.empty:
                            logger.info(f"  Processing spatial chunk {i+1},{j+1}: {len(spatial_subset):,} features")
                            _rasterize_chunk_to_array(spatial_subset, woodland_type, output_raster, width, height, transform)
                    except Exception as e:
                        logger.warning(f"Skipping spatial chunk {i+1},{j+1}: {e}")
                        continue
        else:
            # Process all features of this type at once
            _rasterize_chunk_to_array(subset, woodland_type, output_raster, width, height, transform)

    # Write final raster to file
    logger.info("Writing final raster to file...")
    # Convert background zeros to 255 (NoData) before writing for consistent nodata handling
    output_raster[output_raster == 0] = 255
    return _write_raster_and_stats(output_raster, output_path, width, height, transform, gdf.crs)


def _rasterize_chunk_parallel(chunk_task):
    """Parallel worker function to rasterize a spatial chunk with enhanced coverage."""
    try:
        subset, woodland_type, width, height, transform, chunk_id = chunk_task

        if subset.empty:
            return None

        # Prepare and validate geometries for this chunk
        subset = subset.copy()
        subset = subset[~subset.geometry.isna() & ~subset.geometry.is_empty]

        if subset.empty:
            return None

        # Fix invalid geometries
        invalid_mask = ~subset.geometry.is_valid
        if invalid_mask.any():
            subset.loc[invalid_mask, 'geometry'] = subset.loc[invalid_mask, 'geometry'].buffer(0)

        # Apply small buffer to ensure coverage of thin features
        buffer_distance = 0.1  # 10cm buffer
        buffered_geoms = []
        for geom in subset.geometry:
            if geom is not None and not geom.is_empty:
                buffered = geom.buffer(buffer_distance, cap_style=2, join_style=2)
                buffered_geoms.append(buffered)
            else:
                buffered_geoms.append(geom)

        subset['geometry'] = buffered_geoms

        # Prepare shapes for rasterization
        shapes = [(geom, woodland_type) for geom in subset.geometry if geom is not None and not geom.is_empty]

        if not shapes:
            return None

        # Use dual-resolution rasterization for better coverage
        chunk_raster = np.zeros((height, width), dtype=np.uint8)

        # First pass: standard resolution
        temp_raster = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )

        # Second pass: higher resolution for thin features
        try:
            fine_resolution = 2.0  # 2m intermediate
            fine_width = int(np.ceil(width * 10.0 / fine_resolution))
            fine_height = int(np.ceil(height * 10.0 / fine_resolution))

            if fine_width > 0 and fine_height > 0:
                from rasterio.transform import from_bounds
                from rasterio.enums import Resampling
                from rasterio.warp import reproject

                fine_transform = from_bounds(
                    transform.c, transform.f - fine_height * fine_resolution,
                    transform.c + fine_width * fine_resolution, transform.f,
                    fine_width, fine_height
                )

                fine_raster = rasterize(
                    shapes,
                    out_shape=(fine_height, fine_width),
                    transform=fine_transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True
                )

                # Resample back to target resolution
                reproject(
                    source=fine_raster,
                    destination=temp_raster,
                    src_transform=fine_transform,
                    src_crs=None,  # Same CRS
                    dst_transform=transform,
                    dst_crs=None,
                    resampling=Resampling.max,
                    src_nodata=0,
                    dst_nodata=0
                )
        except Exception:
            # If fine resolution fails, just use the standard rasterization
            pass

        # Combine results
        mask = (temp_raster > 0)
        chunk_raster[mask] = temp_raster[mask]

        return chunk_raster

    except Exception as e:
        logger.error(f"Failed to process chunk {chunk_id}: {e}")
        return None


def _rasterize_chunk_to_array(subset, woodland_type, output_raster, width, height, transform):
    """Rasterize a chunk of features into the output array with enhanced coverage."""
    if subset.empty:
        return

    # Prepare and validate geometries
    subset = subset.copy()
    subset = subset[~subset.geometry.isna() & ~subset.geometry.is_empty]

    if subset.empty:
        return

    # Fix invalid geometries
    invalid_mask = ~subset.geometry.is_valid
    if invalid_mask.any():
        subset.loc[invalid_mask, 'geometry'] = subset.loc[invalid_mask, 'geometry'].buffer(0)

    # Apply small buffer for coverage
    buffer_distance = 0.1  # 10cm buffer
    buffered_geoms = []
    for geom in subset.geometry:
        if geom is not None and not geom.is_empty:
            buffered = geom.buffer(buffer_distance, cap_style=2, join_style=2)
            buffered_geoms.append(buffered)
        else:
            buffered_geoms.append(geom)

    subset['geometry'] = buffered_geoms

    # Prepare shapes for rasterization
    shapes = [(geom, woodland_type) for geom in subset.geometry if geom is not None and not geom.is_empty]

    if not shapes:
        return

    logger.info(f"    Rasterizing {len(shapes)} geometries with enhanced coverage...")

    # Use enhanced rasterization approach
    chunk_raster = np.zeros((height, width), dtype=np.uint8)

    # First pass: standard rasterization
    temp_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )

    # Second pass: higher resolution for thin features
    try:
        fine_resolution = 2.0  # 2m intermediate
        fine_width = int(np.ceil(width * 10.0 / fine_resolution))
        fine_height = int(np.ceil(height * 10.0 / fine_resolution))

        if fine_width > 0 and fine_height > 0:
            from rasterio.transform import from_bounds
            from rasterio.enums import Resampling
            from rasterio.warp import reproject

            fine_transform = from_bounds(
                transform.c, transform.f - fine_height * fine_resolution,
                transform.c + fine_width * fine_resolution, transform.f,
                fine_width, fine_height
            )

            fine_raster = rasterize(
                shapes,
                out_shape=(fine_height, fine_width),
                transform=fine_transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )

            # Resample back to target resolution
            reproject(
                source=fine_raster,
                destination=temp_raster,
                src_transform=fine_transform,
                src_crs=None,
                dst_transform=transform,
                dst_crs=None,
                resampling=Resampling.max,
                src_nodata=0,
                dst_nodata=0
            )
    except Exception:
        # If fine resolution fails, use standard result
        pass

    # Combine results
    mask = (temp_raster > 0)
    chunk_raster[mask] = temp_raster[mask]

    # Merge with existing output (take max value to preserve higher priority types)
    # This ensures that higher woodland type codes (3 > 2 > 1) take precedence
    np.maximum(output_raster, chunk_raster, out=output_raster)


def _write_raster_and_stats(raster, output_path, width, height, transform, crs):
    """Write raster to file and calculate statistics."""
    logger.info(f"Writing raster to {output_path}")

    profile = {
        'driver': 'GTiff',
        'dtype': raster.dtype,
        'nodata': 255,
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw',
        'tiled': True
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(raster, 1)

    return _calculate_raster_stats_from_array(raster, width, height)


def _calculate_raster_stats_from_array(raster, width, height):
    """Calculate statistics from raster array."""
    total_pixels = width * height
    woodland_pixels = np.count_nonzero(raster)

    logger.info(f"Raster export complete:")
    logger.info(f"  Total pixels: {total_pixels:,}")
    logger.info(f"  Non-nodata pixels: {woodland_pixels:,} ({woodland_pixels/total_pixels*100:.1f}%)")

    # Log pixel value distribution
    unique_values, counts = np.unique(raster, return_counts=True)
    code_names = {0: 'Unknown', 1: 'Lone Tree', 2: 'Group of Trees', 3: 'Small Woodland', 4: 'OHC (Fringe)', 5: 'NFI Woodland', 255: 'NoData'}
    logger.info("Raster value distribution:")
    for value, count in zip(unique_values, counts):
        percentage = count / total_pixels * 100
        logger.info(f"  {value} ({code_names.get(value, 'Unknown')}): {count:,} pixels ({percentage:.1f}%)")

    return True


def _calculate_raster_stats(output_path, width, height):
    """Calculate statistics by reading raster file in chunks."""
    total_pixels = width * height
    woodland_pixels = 0
    value_counts = {}

    # Read in chunks to avoid memory issues
    with rasterio.open(output_path, 'r') as src:
        # Read in 1000x1000 chunks
        chunk_size = 1000
        for row in range(0, height, chunk_size):
            for col in range(0, width, chunk_size):
                window = rasterio.windows.Window(
                    col, row,
                    min(chunk_size, width - col),
                    min(chunk_size, height - row)
                )
                chunk = src.read(1, window=window)

                # Count non-zero pixels
                woodland_pixels += np.count_nonzero(chunk)

                # Count unique values
                unique_values, counts = np.unique(chunk, return_counts=True)
                for value, count in zip(unique_values, counts):
                    value_counts[value] = value_counts.get(value, 0) + count

    logger.info(f"Raster export complete:")
    logger.info(f"  Total pixels: {total_pixels:,}")
    logger.info(f"  Non-nodata pixels: {woodland_pixels:,} ({woodland_pixels/total_pixels*100:.1f}%)")

    # Log pixel value distribution
    code_names = {0: 'Unknown', 1: 'Lone Tree', 2: 'Group of Trees', 3: 'Small Woodland', 4: 'OHC (Fringe)', 5: 'NFI Woodland', 255: 'NoData'}
    logger.info("Raster value distribution:")
    for value in sorted(value_counts.keys()):
        count = value_counts[value]
        percentage = count / total_pixels * 100
        logger.info(f"  {value} ({code_names.get(value, 'Unknown')}): {count:,} pixels ({percentage:.1f}%)")

    return True


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
        # Return empty result with minimal column structure
        return gpd.GeoDataFrame(columns=['woodland_type_code', 'source_layer', 'geometry'], crs=gdf.crs)


def process_single_layer_identification(gdb_path, layer_name, peatland_file, simplify_tolerance=1.0):
    """
    Process a single layer for identification with woodland type encoding.
    Enhanced version that processes woodland features.
    If peatland_file is None, processes all features without spatial filtering.
    """

    try:
        # Load GDB layer
        gdf = gpd.read_file(gdb_path, layer=layer_name)

        # Fix geometries
        gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]

        if gdf.empty:
            return None

        # Process woodland features (encoding and simplification)
        gdf = process_woodland_features(gdf, simplify_tolerance)

        # Add source layer info
        gdf['source_layer'] = layer_name

        # If no peatland file provided, return all processed features
        if peatland_file is None:
            logger.info(f"No peatland filtering - returning all {len(gdf)} features from layer {layer_name}")
            return gdf

        # Load peatland data (cached per worker) only if needed
        global _PEAT_CACHE
        if peatland_file in _PEAT_CACHE:
            peatland_gdf = _PEAT_CACHE[peatland_file]
        else:
            peatland_gdf = gpd.read_file(peatland_file)
            _PEAT_CACHE[peatland_file] = peatland_gdf

        # Ensure same CRS
        if gdf.crs != peatland_gdf.crs:
            peatland_gdf = peatland_gdf.to_crs(gdf.crs)

        # Fix peatland geometries
        peatland_gdf = peatland_gdf[~peatland_gdf.geometry.isna() & ~peatland_gdf.geometry.is_empty]

        if peatland_gdf.empty:
            logger.warning(f"Empty peatland data for layer {layer_name}")
            return None

        # Remove source_layer temporarily for intersection processing
        gdf_for_intersection = gdf.drop(columns=['source_layer'])

        # Efficient identification using STRtree-based method
        result = identify_intersections_standalone_silent(gdf_for_intersection, peatland_gdf, layer_name)

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
    Enhanced with woodland type processing.
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
            # Return empty result with minimal columns
            minimal_columns = ['woodland_type_code', 'source_layer', 'intersection_area', 'peatland_id', 'geometry']
            return gpd.GeoDataFrame(columns=minimal_columns, crs=chunk_gdf.crs)

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
            minimal_columns = ['woodland_type_code', 'source_layer', 'intersection_area', 'peatland_id', 'geometry']
            return gpd.GeoDataFrame(columns=minimal_columns, crs=chunk_gdf.crs)

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
            minimal_columns = ['woodland_type_code', 'source_layer', 'intersection_area', 'peatland_id', 'geometry']
            return gpd.GeoDataFrame(columns=minimal_columns, crs=chunk_gdf.crs)

    except Exception as e:
        logger.error(f"Error processing ultra-optimized chunk {chunk_id}: {e}")
        minimal_columns = ['woodland_type_code', 'source_layer', 'intersection_area', 'peatland_id', 'geometry']
        return gpd.GeoDataFrame(columns=minimal_columns, crs=chunk_gdf.crs)


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
    Enhanced with woodland type processing.
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
        minimal_columns = ['woodland_type_code', 'source_layer', 'intersection_area', 'peatland_id', 'geometry']
        return gpd.GeoDataFrame(columns=minimal_columns, crs=all_identified_features.crs)


# Integration with existing HybridGDBProcessor class
class EnhancedGDBProcessor:
    """
    Enhanced GDB processor with woodland type encoding and polygon simplification.
    """

    def __init__(self,
                 gdb_directory: str,
                 peatland_file: str = None,
                 nfi_shapefile: str = None,
                 output_file: str = None,
                 report_file: str = None,
                 method: str = "identify",
                 chunk_size: int = 5000,
                 n_processes: int = None,
                 debug_mode: bool = False,
                 simplify_tolerance: float = 1.0,
                 exclude_unknown: bool = False,
                 export_raster: bool = False,
                 raster_resolution: float = 10.0,
                 raster_cores: int = None):

        # Auto-generate output filename under canonical data/output tree.
        default_output_dir = Path(__file__).resolve().parents[2] / "data" / "output" / "labels"

        if output_file is None:
            if export_raster:
                if peatland_file is None:
                    output_file = str(default_output_dir / "woodland_all_v2.tif")
                elif method == "identify":
                    output_file = str(default_output_dir / "woodland_identified_v2.tif")
                else:
                    output_file = str(default_output_dir / "woodland_intersected_v2.tif")
            else:
                if peatland_file is None:
                    output_file = str(default_output_dir / "woodland_all_v2.gpkg")
                elif method == "identify":
                    output_file = str(default_output_dir / "woodland_identified_v2.gpkg")
                else:
                    output_file = str(default_output_dir / "woodland_intersected_v2.gpkg")

        # Set default extension based on export type - fix for proper extensions
        if export_raster:
            # If exporting raster but filename has vector extension, replace it
            if output_file.endswith('.shp') or output_file.endswith('.gpkg') or output_file.endswith('.geojson'):
                output_file = output_file.rsplit('.', 1)[0] + '.tif'
            elif not (output_file.endswith('.tif') or output_file.endswith('.tiff')):
                output_file = output_file + '.tif'
        else:
            if not (output_file.endswith('.shp') or output_file.endswith('.gpkg') or output_file.endswith('.geojson')):
                output_file = output_file + '.gpkg'

        self.gdb_directory = Path(gdb_directory)
        self.peatland_file = Path(peatland_file) if peatland_file else None
        self.nfi_shapefile = Path(nfi_shapefile) if nfi_shapefile else None
        self.output_file = Path(output_file)
        default_report_dir = Path(__file__).resolve().parents[2] / "data" / "output" / "reports"
        self.report_file = Path(report_file) if report_file else default_report_dir / f"{self.output_file.stem}.report.txt"
        self.method = method.lower()
        self.chunk_size = chunk_size
        self.debug_mode = debug_mode
        self.simplify_tolerance = simplify_tolerance
        self.exclude_unknown = exclude_unknown
        self.export_raster = export_raster
        self.raster_resolution = raster_resolution

        # Optimize process allocation - use all cores - 2
        total_cpus = mp.cpu_count()
        available_cores = max(1, total_cpus - 2)
        self.raster_cores = raster_cores if raster_cores else available_cores
        if n_processes is None:
            self.n_processes = available_cores
        else:
            self.n_processes = min(n_processes, total_cpus)

        # DEBUG: Use sequential processing for debugging
        if self.debug_mode:
            self.n_processes = 1
            logger.info(f"🔧 DEBUG MODE: Using sequential processing (1 process) for debugging")

        logger.info(f"Enhanced GDB processor initialized:")
        logger.info(f"  Method: {self.method}")
        logger.info(f"  Peatland file: {self.peatland_file if self.peatland_file else 'None (process all features)'}")
        logger.info(f"  NFI shapefile: {self.nfi_shapefile if self.nfi_shapefile else 'None'}")
        logger.info(f"  Total CPUs: {total_cpus}")
        logger.info(f"  Available cores (total - 2): {available_cores}")
        logger.info(f"  Processes: {self.n_processes}")
        logger.info(f"  Chunk size: {self.chunk_size}")
        logger.info(f"  Simplify tolerance: {self.simplify_tolerance}m")
        logger.info(f"  Exclude unknown types: {self.exclude_unknown}")
        logger.info(f"  Export as raster: {self.export_raster}")
        if self.export_raster:
            logger.info(f"  Raster resolution: {self.raster_resolution}m")
            logger.info(f"  Raster cores: {self.raster_cores}")

        self.report_data = {
            "gdb_directory": str(self.gdb_directory),
            "peatland_file": str(self.peatland_file) if self.peatland_file else None,
            "nfi_shapefile": str(self.nfi_shapefile) if self.nfi_shapefile else None,
            "output_file": str(self.output_file),
            "report_file": str(self.report_file),
            "method": self.method,
            "debug_mode": bool(self.debug_mode),
            "simplify_tolerance": float(self.simplify_tolerance),
            "exclude_forests": bool(self.exclude_unknown),
            "export_raster": bool(self.export_raster),
            "raster_resolution": float(self.raster_resolution),
            "n_processes": int(self.n_processes),
            "raster_cores": int(self.raster_cores),
        }

    def _set_report_counts(self, key: str, value) -> None:
        self.report_data[key] = value

    def _write_run_report(self, status: str, error_message: Optional[str] = None) -> None:
        self.report_file.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "=" * 70,
            "TOW GDB PROCESSING REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "1. INPUTS",
            "-" * 70,
            f"GDB directory: {self.report_data.get('gdb_directory')}",
            f"Peatland file: {self.report_data.get('peatland_file') or 'Not used'}",
            f"NFI shapefile: {self.report_data.get('nfi_shapefile') or 'Not used'}",
            f"Output file: {self.report_data.get('output_file')}",
            f"Report file: {self.report_data.get('report_file')}",
            "",
            "2. SETTINGS",
            "-" * 70,
            f"Method: {self.report_data.get('method')}",
            f"Debug mode: {self.report_data.get('debug_mode')}",
            f"Simplify tolerance (m): {self.report_data.get('simplify_tolerance')}",
            f"Exclude forests: {self.report_data.get('exclude_forests')}",
            f"Export raster: {self.report_data.get('export_raster')}",
            f"Raster resolution (m): {self.report_data.get('raster_resolution')}",
            f"Processes: {self.report_data.get('n_processes')}",
            f"Raster cores: {self.report_data.get('raster_cores')}",
            "",
            "3. RUN STATUS",
            "-" * 70,
            f"Status: {status}",
        ]
        if error_message:
            lines.append(f"Error: {error_message}")

        lines.extend(
            [
                "",
                "4. COUNTS",
                "-" * 70,
                f"GDB folders found: {int(self.report_data.get('gdb_folders', 0)):,}",
                f"GDB folders processed: {int(self.report_data.get('gdb_folders_processed', 0)):,}",
                f"Layer tasks discovered: {int(self.report_data.get('layer_tasks', 0)):,}",
                f"Layer results returned: {int(self.report_data.get('layer_results', 0)):,}",
                f"Features identified from GDB: {int(self.report_data.get('gdb_features', 0)):,}",
                f"Features added from NFI: {int(self.report_data.get('nfi_features', 0)):,}",
                f"Combined features before filtering: {int(self.report_data.get('combined_features', 0)):,}",
                f"Features excluded by forest filter: {int(self.report_data.get('excluded_features', 0)):,}",
                f"Final exported features: {int(self.report_data.get('final_features', 0)):,}",
                f"Distinct source layers: {int(self.report_data.get('source_layers', 0)):,}",
            ]
        )

        woodland_counts = self.report_data.get("woodland_type_counts", {})
        if woodland_counts:
            code_names = {
                0: "Unknown",
                1: "Lone Tree",
                2: "Group of Trees",
                3: "Small Woodland",
                4: "NFI OHC (Fringe)",
                5: "NFI Woodland",
            }
            lines.extend(["", "5. WOODLAND TYPE COUNTS", "-" * 70])
            for code in sorted(woodland_counts):
                lines.append(f"{code} ({code_names.get(code, 'Unknown')}): {int(woodland_counts[code]):,}")

        if self.output_file.exists() and self.export_raster:
            with rasterio.open(self.output_file) as src:
                value_counts = {}
                for _, window in src.block_windows(1):
                    block = src.read(1, window=window, masked=False)
                    values, counts = np.unique(block, return_counts=True)
                    for value, count in zip(values, counts):
                        value_counts[int(value)] = value_counts.get(int(value), 0) + int(count)
                lines.extend(
                    [
                        "",
                        "6. RASTER OUTPUT",
                        "-" * 70,
                        f"Dimensions: {src.width} x {src.height} pixels",
                        f"CRS: {src.crs}",
                        f"Pixel size: {abs(src.transform.a):.3f} x {abs(src.transform.e):.3f}",
                    ]
                )
                for value in sorted(value_counts):
                    lines.append(f"Value {int(value)}: {int(value_counts[value]):,} pixels")

        self.report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self):
        """Run with enhanced processing including woodland type encoding."""
        run_status = "completed"
        error_message = None
        try:
            logger.info(f"Starting enhanced {self.method} processing pipeline")

            # Find all GDB folders
            gdb_folders = list(self.gdb_directory.glob("*.gdb"))
            self._set_report_counts("gdb_folders", len(gdb_folders))

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
            self._set_report_counts("gdb_folders_processed", len(debug_gdb_folders))

            for gdb_folder in debug_gdb_folders:
                try:
                    layers = fiona.listlayers(str(gdb_folder))
                    for layer in layers:
                        peatland_file_str = str(self.peatland_file) if self.peatland_file else None
                        task = (str(gdb_folder), layer, peatland_file_str, self.simplify_tolerance)
                        processing_tasks.append(task)
                except Exception as e:
                    logger.warning(f"Could not list layers in {gdb_folder}: {e}")
                    continue

            logger.info(f"Found {len(processing_tasks)} layers to process")
            self._set_report_counts("layer_tasks", len(processing_tasks))

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

            # Combine all identified features from GDB
            all_identified_features = pd.DataFrame()
            if identification_results:
                all_identified_features = pd.concat(identification_results, ignore_index=True)
                total_identified = len(all_identified_features)
                logger.info(f"✅ GDB processing completed: {total_identified:,} features identified across {len(identification_results)} regions")
            self._set_report_counts("layer_results", len(identification_results))
            self._set_report_counts("gdb_features", len(all_identified_features))

            # Process NFI shapefile if provided
            nfi_features = pd.DataFrame()
            if self.nfi_shapefile and self.nfi_shapefile.exists():
                logger.info(f"🌲 Processing NFI shapefile: {self.nfi_shapefile}")
                nfi_features = process_nfi_shapefile(self.nfi_shapefile, self.simplify_tolerance)

                if not nfi_features.empty:
                    logger.info(f"✅ NFI shapefile processed: {len(nfi_features)} dissolved features")

                    # Combine GDB and NFI features
                    if not all_identified_features.empty:
                        all_identified_features = pd.concat([all_identified_features, nfi_features], ignore_index=True)
                        logger.info(f"✅ Combined GDB + NFI: {len(all_identified_features)} total features")
                    else:
                        all_identified_features = nfi_features
                        logger.info(f"✅ Using NFI features only: {len(all_identified_features)} features")
                else:
                    logger.warning("NFI shapefile processing returned no features")
            elif self.nfi_shapefile:
                logger.warning(f"NFI shapefile not found: {self.nfi_shapefile}")
            self._set_report_counts("nfi_features", len(nfi_features))

            total_identified = len(all_identified_features)
            logger.info(f"✅ Stage 1 completed: {total_identified:,} total features (GDB + NFI)")
            self._set_report_counts("combined_features", total_identified)

            code_names = {
                0: 'Unknown',
                1: 'Lone Tree',
                2: 'Group of Trees',
                3: 'Small Woodland',
                4: 'NFI OHC (Fringe)',
                5: 'NFI Woodland',
            }

            # Log woodland type statistics
            if not all_identified_features.empty and 'woodland_type_code' in all_identified_features.columns:
                woodland_stats = all_identified_features['woodland_type_code'].value_counts().sort_index()
                logger.info("Initial woodland type distribution:")
                for code, count in woodland_stats.items():
                    logger.info(f"  {code} ({code_names.get(code, 'Unknown')}): {count:,}")

            # Filter out forests/woodland (woodland type 5) if requested
            if self.exclude_unknown and not all_identified_features.empty and 'woodland_type_code' in all_identified_features.columns:
                initial_count = len(all_identified_features)
                all_identified_features = all_identified_features[all_identified_features['woodland_type_code'] != 5]
                filtered_count = len(all_identified_features)
                excluded_count = initial_count - filtered_count
                logger.info(f"Excluded {excluded_count:,} NFI woodland features (woodland type code 5)")
                logger.info(f"Remaining features: {filtered_count:,}")
                self._set_report_counts("excluded_features", excluded_count)

                # Log filtered woodland type statistics
                if not all_identified_features.empty:
                    filtered_woodland_stats = all_identified_features['woodland_type_code'].value_counts().sort_index()
                    logger.info("Filtered woodland type distribution:")
                    for code, count in filtered_woodland_stats.items():
                        logger.info(f"  {code} ({code_names.get(code, 'Unknown')}): {count:,}")

                # Check if we still have features after filtering
                if all_identified_features.empty:
                    logger.warning("No features remaining after filtering. Exiting.")
                    return

            # Early raster export optimization - skip unnecessary processing for raster export
            if self.export_raster:
                logger.info("🎯 Raster export mode - skipping vector-specific processing steps")

                # For raster export, use the identified features directly.
                final_result = all_identified_features

                # Ensure output directory exists
                output_dir = Path(self.output_file).parent
                output_dir.mkdir(parents=True, exist_ok=True)

                # Export directly to raster
                logger.info(f"Exporting {len(final_result)} features directly to raster: {self.output_file}")

                try:
                    success = export_to_raster(
                        final_result,
                        self.output_file,
                        resolution=self.raster_resolution,
                        woodland_type_col='woodland_type_code'
                    )

                    if success:
                        logger.info("✅ Raster export completed successfully!")
                        logger.info("📊 Final statistics:")
                        logger.info(f"  - Total features rasterized: {len(final_result):,}")
                        logger.info(f"  - Source layers: {final_result['source_layer'].nunique()}")
                        logger.info(f"  - Raster resolution: {self.raster_resolution}m")
                        self._set_report_counts("final_features", len(final_result))
                        self._set_report_counts("source_layers", int(final_result['source_layer'].nunique()))

                        # Woodland type statistics
                        if 'woodland_type_code' in final_result.columns:
                            final_woodland_stats = final_result['woodland_type_code'].value_counts().sort_index()
                            self._set_report_counts("woodland_type_counts", {int(code): int(count) for code, count in final_woodland_stats.items()})
                            logger.info("Features rasterized by woodland type:")
                            for code, count in final_woodland_stats.items():
                                logger.info(f"    {code} ({code_names.get(code, 'Unknown')}): {count:,}")

                        return  # Exit early for raster export

                    logger.error("Raster export failed")
                    return

                except Exception as raster_error:
                    logger.error(f"Raster export failed: {raster_error}")
                    raise raster_error

            # Stage 2: Method-specific processing (vector export only)
                if self.method == "identify" or self.peatland_file is None:
                    # Return identified features (no exact intersection needed)
                    # If no peatland file, we've already processed all features
                    final_result = all_identified_features
                elif self.method == "intersect":
                    # Stage 2: Ultra-optimized intersection processing
                    if self.peatland_file is None:
                        logger.warning("Cannot perform intersection without peatland file. Returning identified features.")
                        final_result = all_identified_features
                    else:
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

                # Choose export method based on format
                if self.export_raster:
                    # Export as raster
                    try:
                        success = export_to_raster(
                            final_result,
                            self.output_file,
                            resolution=self.raster_resolution,
                            woodland_type_col='woodland_type_code'
                        )

                        if success:
                            logger.info(f"✅ Enhanced processing completed successfully! Exported as raster")
                            logger.info(f"📊 Final statistics:")
                            logger.info(f"  - Total features: {len(final_result):,}")
                            logger.info(f"  - Source layers: {final_result['source_layer'].nunique()}")
                            logger.info(f"  - Raster resolution: {self.raster_resolution}m")

                            # Woodland type statistics in final output
                            if 'woodland_type_code' in final_result.columns:
                                final_woodland_stats = final_result['woodland_type_code'].value_counts().sort_index()
                                logger.info("Final exported woodland type distribution:")
                                code_names = {0: 'Unknown', 1: 'Lone Tree', 2: 'Group of Trees', 3: 'Small Woodland', 4: 'NFI OHC (Fringe)', 5: 'NFI Woodland'}
                                for code, count in final_woodland_stats.items():
                                    logger.info(f"    {code} ({code_names.get(code, 'Unknown')}): {count:,}")
                        else:
                            logger.error("Raster export failed")

                    except Exception as raster_error:
                        logger.error(f"Raster export failed: {raster_error}")
                        raise raster_error
                else:
                    # Export as vector format
                    try:
                        # Determine driver based on file extension
                        if self.output_file.suffix.lower() == '.shp':
                            driver = 'ESRI Shapefile'
                        elif self.output_file.suffix.lower() == '.gpkg':
                            driver = 'GPKG'
                        elif self.output_file.suffix.lower() == '.geojson':
                            driver = 'GeoJSON'
                        else:
                            # Default to GeoPackage for unknown extensions
                            driver = 'GPKG'

                        final_result.to_file(self.output_file, driver=driver)
                        logger.info(f"✅ Enhanced processing completed successfully! Exported as {driver}")
                        logger.info(f"📊 Final statistics:")
                        logger.info(f"  - Total features: {len(final_result):,}")
                        logger.info(f"  - Source layers: {final_result['source_layer'].nunique()}")
                        self._set_report_counts("final_features", len(final_result))
                        self._set_report_counts("source_layers", int(final_result['source_layer'].nunique()))

                        # Woodland type statistics in final output
                        if 'woodland_type_code' in final_result.columns:
                            final_woodland_stats = final_result['woodland_type_code'].value_counts().sort_index()
                            self._set_report_counts("woodland_type_counts", {int(code): int(count) for code, count in final_woodland_stats.items()})
                            logger.info("Final exported woodland type distribution:")
                            code_names = {0: 'Unknown', 1: 'Lone Tree', 2: 'Group of Trees', 3: 'Small Woodland', 4: 'NFI OHC (Fringe)', 5: 'NFI Woodland'}
                            for code, count in final_woodland_stats.items():
                                logger.info(f"    {code} ({code_names.get(code, 'Unknown')}): {count:,}")

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
            run_status = "failed"
            error_message = str(e)
            logger.error(f"Enhanced processing pipeline failed: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self._write_run_report(run_status, error_message)


# Usage example
if __name__ == "__main__":
    import argparse

    # Set multiprocessing start method to 'spawn' for better compatibility
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Enhanced GDB processor with woodland type encoding and simplification")
    parser.add_argument("--gdb-dir", required=True, help="Directory containing GDB folders")
    parser.add_argument("--peatland-file", help="Path to peatland extent file (optional - if not provided, processes all woodland features)")
    parser.add_argument("--nfi-shapefile", help="Path to NFI shapefile to include as woodland type 5")
    parser.add_argument(
        "--output",
        help="Output file path (default: tow/data/output/labels/<auto_name>.gpkg or .tif)",
    )
    parser.add_argument("--report-file", help="Optional report path. Default: data/output/reports/<output>.report.txt")
    parser.add_argument("--method", choices=['identify', 'intersect'], default='identify',
                       help="Processing method (intersect requires peatland-file)")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--debug", action='store_true', help="Enable debug mode with geometry analysis")
    parser.add_argument("--simplify-tolerance", type=float, default=1.0,
                       help="Polygon simplification tolerance in meters (default: 1.0, use 0 to disable)")
    parser.add_argument("--exclude-forests", action='store_true',
                       help="Exclude NFI woodland (woodland type code 5) from output")
    parser.add_argument("--export-raster", action='store_true',
                       help="Export as raster (GeoTIFF) instead of vector format")
    parser.add_argument("--raster-resolution", type=float, default=10.0,
                       help="Raster pixel size in meters (default: 10.0)")
    parser.add_argument("--raster-cores", type=int,
                       help="Number of cores for parallel rasterization (default: min(8, available_cores))")

    args = parser.parse_args()

    # Validate arguments
    if args.method == 'intersect' and args.peatland_file is None:
        logger.error("❌ Error: --method intersect requires --peatland-file to be specified")
        parser.print_help()
        exit(1)

    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - detailed geometry analysis will be performed")

    # Log the processing mode
    if args.peatland_file is None:
        logger.info("🌳 Mode: Extract ALL woodland features from GDB (no peatland filtering)")
    else:
        logger.info(f"🌿 Mode: Extract woodland features intersecting with peatland ({args.method})")

    processor = EnhancedGDBProcessor(
        gdb_directory=args.gdb_dir,
        peatland_file=args.peatland_file,
        nfi_shapefile=args.nfi_shapefile,
        output_file=args.output,
        report_file=args.report_file,
        method=args.method,
        chunk_size=args.chunk_size,
        n_processes=args.processes,
        debug_mode=args.debug,
        simplify_tolerance=args.simplify_tolerance,
        exclude_unknown=args.exclude_forests,
        export_raster=args.export_raster,
        raster_resolution=args.raster_resolution,
        raster_cores=args.raster_cores
    )
    logger.info(f"📁 Output file: {processor.output_file}")
    logger.info(f"📝 Report file: {processor.report_file}")

    processor.run()
