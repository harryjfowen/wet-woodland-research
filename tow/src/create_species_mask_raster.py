#!/usr/bin/env python3
"""
Create binary raster mask from shapefile where target species polygons = 1, others = 0

DESCRIPTION: Creates binary raster masks from Forestry England shapefiles for wet woodland detection. 
Filters compartments for target species (alder, birch, willow), optionally applies peatland masking, 
and rasterizes to binary classification (1 = target species, 0 = other species, 255 = no data). 
Used to generate training labels for machine learning models.
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
import numpy as np
import pandas as pd
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Define target species for filtering
TARGET_SPECIES = ['alder', 'birch', 'willow']

def check_for_species(gdf, species_list):
    """
    Check if any of the target species are mentioned in the primary species column.
    Returns a dictionary with 'found' boolean and 'columns' list.
    """
    species_columns = []
    species_found = False
    
    # Look specifically for primary species column
    primary_species_cols = ['PRISPECIES', 'Primary_Species', 'Species', 'SPECIES']
    
    for col in primary_species_cols:
        if col in gdf.columns:
            # Convert to string and check for any species (case insensitive)
            string_values = gdf[col].astype(str).str.lower()
            for species in species_list:
                if string_values.str.contains(species, na=False).any():
                    species_columns.append(col)
                    species_found = True
                    break  # Found at least one species in this column
    
    # If no primary species column found, fall back to checking all string columns
    if not species_found:
        print("⚠️  No primary species column found, checking all string columns...")
        for col in gdf.columns:
            if gdf[col].dtype == 'object':  # String columns
                # Convert to string and check for any species (case insensitive)
                string_values = gdf[col].astype(str).str.lower()
                for species in species_list:
                    if string_values.str.contains(species, na=False).any():
                        species_columns.append(col)
                        species_found = True
                        break  # Found at least one species in this column
    
    return {
        'found': species_found,
        'columns': list(set(species_columns))  # Remove duplicates
    }

def filter_by_species(gdf, species_list, species_columns):
    """
    Filter the GeoDataFrame to only include rows where ALL species columns contain target species or are empty/NA.
    Returns filtered dataframe and found species info.
    """
    # Use only the first (primary) species column
    primary_col = species_columns[0]
    print(f"🔍 Filtering by primary species column: {primary_col}")
    
    # Create a mask for rows containing any of the target species in primary column
    primary_mask = gdf[primary_col].astype(str).str.lower().str.contains('|'.join(species_list), na=False)
    
    # Also check secondary and tertiary species columns if they exist
    secondary_cols = ['SECSPECIES', 'Secondary_Species', 'SEC_SPECIES']
    tertiary_cols = ['TERSPECIES', 'Tertiary_Species', 'TER_SPECIES']
    
    # Find which secondary and tertiary columns exist in the data
    existing_secondary_cols = [col for col in secondary_cols if col in gdf.columns]
    existing_tertiary_cols = [col for col in tertiary_cols if col in gdf.columns]
    
    print(f"🔍 Checking secondary species columns: {existing_secondary_cols}")
    print(f"🔍 Checking tertiary species columns: {existing_tertiary_cols}")
    
    # Create masks for secondary and tertiary columns
    secondary_mask = pd.Series([True] * len(gdf), index=gdf.index)  # Default to True
    tertiary_mask = pd.Series([True] * len(gdf), index=gdf.index)   # Default to True
    
    # Check secondary columns - must contain target species OR be empty/NA
    for col in existing_secondary_cols:
        # Check if column contains target species OR is empty/NA
        col_mask = (
            gdf[col].astype(str).str.lower().str.contains('|'.join(species_list), na=False) |  # Contains target species
            gdf[col].isna() |  # Is NA
            (gdf[col].astype(str).str.strip() == '') |  # Is empty string
            (gdf[col].astype(str).str.lower() == 'nan') |  # Is 'nan' string
            (gdf[col].astype(str).str.lower() == 'none') |  # Is 'none' string
            (gdf[col].astype(str).str.lower() == 'n/a')  # Is 'n/a' string
        )
        secondary_mask = secondary_mask & col_mask
    
    # Check tertiary columns - must contain target species OR be empty/NA
    for col in existing_tertiary_cols:
        # Check if column contains target species OR be empty/NA
        col_mask = (
            gdf[col].astype(str).str.lower().str.contains('|'.join(species_list), na=False) |  # Contains target species
            gdf[col].isna() |  # Is NA
            (gdf[col].astype(str).str.strip() == '') |  # Is empty string
            (gdf[col].astype(str).str.lower() == 'nan') |  # Is 'nan' string
            (gdf[col].astype(str).str.lower() == 'none') |  # Is 'none' string
            (gdf[col].astype(str).str.lower() == 'n/a')  # Is 'n/a' string
        )
        tertiary_mask = tertiary_mask & col_mask
    
    # Combine all masks
    final_mask = primary_mask & secondary_mask & tertiary_mask
    
    filtered_gdf = gdf[final_mask].copy()
    
    # Check which species were actually found
    found_species = []
    for species in species_list:
        if filtered_gdf[primary_col].astype(str).str.lower().str.contains(species, na=False).any():
            found_species.append(species)
    
    print(f"📊 Filtering results:")
    print(f"  • Primary species filter: {primary_mask.sum():,} features")
    print(f"  • Secondary species filter: {secondary_mask.sum():,} features")
    print(f"  • Tertiary species filter: {tertiary_mask.sum():,} features")
    print(f"  • Combined filter: {final_mask.sum():,} features")
    
    return filtered_gdf, found_species

def create_species_raster_mask(shapefile_path, output_path, peatland_mask_path=None, reference_raster_path=None, 
                              resolution=5, crs=None, bounds=None):
    """
    Create binary raster mask where target species polygons = 1, others = 0
    
    Args:
        shapefile_path: Path to shapefile with species data
        output_path: Path for output raster mask
        peatland_mask_path: Optional peatland mask shapefile for additional filtering
        reference_raster_path: Optional reference raster for extent/resolution/CRS
        resolution: Resolution in meters (if no reference raster)
        crs: Coordinate reference system (if no reference raster)
        bounds: Bounds (left, bottom, right, top) (if no reference raster)
    """
    print(f"Loading shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    
    print(f"Shapefile info:")
    print(f"  CRS: {gdf.crs}")
    print(f"  Number of features: {len(gdf)}")
    print(f"  Bounds: {gdf.total_bounds}")
    
    # Check for target species mentions
    species_info = check_for_species(gdf, TARGET_SPECIES)
    if not species_info['found']:
        print(f"❌ No target species found in shapefile")
        print(f"Target species: {TARGET_SPECIES}")
        return None
    
    print(f"🌳 TARGET SPECIES FOUND in shapefile!")
    print(f"Species columns: {species_info['columns']}")
    print(f"Target species: {TARGET_SPECIES}")
    
    # Filter the data to only include rows with target species
    target_species_gdf, found_species = filter_by_species(gdf, TARGET_SPECIES, species_info['columns'])
    print(f"Filtered to {len(target_species_gdf)} features containing target species")
    print(f"✅ Species found: {found_species}")
    
    # Load and process peatland mask if provided
    if peatland_mask_path:
        print(f"Loading peatland mask: {peatland_mask_path}")
        
        # Check if it's a raster or shapefile
        peatland_path = Path(peatland_mask_path)
        if peatland_path.suffix.lower() in ['.tif', '.tiff']:
            # Handle raster mask - store for later use in rasterization
            print("Peatland mask is a raster file")
            peatland_raster_path = peatland_mask_path
            with rasterio.open(peatland_mask_path) as peat_src:
                peat_crs = peat_src.crs
                peat_bounds = peat_src.bounds
                peat_data = peat_src.read(1)
                
            print(f"Peatland raster info:")
            print(f"  CRS: {peat_crs}")
            print(f"  Bounds: {peat_bounds}")
            print(f"  Shape: {peat_data.shape}")
            print(f"  Values: {np.unique(peat_data)}")
            
            # Store peatland raster info for later use
            peatland_raster_info = {
                'path': peatland_raster_path,
                'crs': peat_crs,
                'bounds': peat_bounds,
                'transform': peat_src.transform,
                'shape': peat_data.shape,
                'type': 'file'
            }
            
            # Skip polygon conversion - we'll use raster directly later
            print("Will use peatland raster directly during rasterization")
            peatland_gdf = None
                
        else:
            # Handle shapefile
            print("Peatland mask is a shapefile")
            peatland_gdf = gpd.read_file(peatland_mask_path)
            print(f"Peatland shapefile info:")
            print(f"  CRS: {peatland_gdf.crs}")
            print(f"  Number of features: {len(peatland_gdf)}")
            print(f"  Bounds: {peatland_gdf.total_bounds}")
            
            # Fix invalid geometries in peatland mask
            peatland_gdf["geometry"] = peatland_gdf["geometry"].apply(lambda geom: make_valid(geom))
            peatland_gdf = peatland_gdf[~peatland_gdf["geometry"].is_empty & peatland_gdf["geometry"].notna()]
            
            # Reproject peatland mask to match species data CRS if needed
            if peatland_gdf.crs != gdf.crs:
                print(f"Reprojecting peatland mask from {peatland_gdf.crs} to {gdf.crs}")
                peatland_gdf = peatland_gdf.to_crs(gdf.crs)
        
        # Handle peatland filtering - optimized approach
        if peatland_gdf is not None:
            # Shapefile approach - convert to raster for efficient masking
            if len(peatland_gdf) > 0:
                print("Converting peatland shapefile to raster for efficient masking...")
                
                # Create peatland raster mask
                peatland_raster = rasterize(
                    [(geom, 1) for geom in peatland_gdf.geometry],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,  # NoData value
                    dtype=np.uint8
                )
                
                # Store for later use during rasterization
                peatland_raster_info = {
                    'raster': peatland_raster,
                    'type': 'array'
                }
                
                print("Peatland shapefile converted to raster mask")
            else:
                print("⚠️  No peatland polygons available")
                peatland_raster_info = None
        else:
            # Raster approach - will handle during rasterization
            print("Peatland raster will be used during rasterization")
    
    # Fix invalid geometries
    print("Fixing invalid geometries...")
    target_species_gdf["geometry"] = target_species_gdf["geometry"].apply(lambda geom: make_valid(geom))
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: make_valid(geom))
    
    # Drop empty geometries
    target_species_gdf = target_species_gdf[~target_species_gdf["geometry"].is_empty & target_species_gdf["geometry"].notna()]
    gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()]
    
    # Determine raster parameters
    if reference_raster_path:
        print(f"Using reference raster: {reference_raster_path}")
        with rasterio.open(reference_raster_path) as src:
            transform = src.transform
            crs = src.crs
            width = src.width
            height = src.height
            bounds = src.bounds
    else:
        # Use shapefile's CRS by default
        if crs is None:
            crs = gdf.crs
            print(f"Using shapefile CRS: {crs}")
        else:
            print(f"Using specified CRS: {crs}")
            
        print(f"Using custom parameters: resolution={resolution}m")
        if bounds is None:
            bounds = gdf.total_bounds
        
        # Calculate transform and dimensions
        left, bottom, right, top = bounds
        width = int((right - left) / resolution)
        height = int((top - bottom) / resolution)
        transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
    
    print(f"Raster parameters:")
    print(f"  CRS: {crs}")
    print(f"  Bounds: {bounds}")
    print(f"  Size: {width} x {height}")
    print(f"  Resolution: {resolution}m")
    
    # Reproject geometries to match raster CRS
    if gdf.crs != crs:
        print(f"Reprojecting geometries to {crs}")
        gdf = gdf.to_crs(crs)
        target_species_gdf = target_species_gdf.to_crs(crs)
    
    # Create binary mask - optimized single rasterization
    print("Creating binary raster mask...")
    
    # More efficient: rasterize target species directly as 1, others as 0
    # This avoids the intermediate value 2 and the extra numpy operation
    
    # Create polygons list with proper values
    polygons = []
    
    # Add target species polygons as value 1
    for geom in target_species_gdf.geometry:
        polygons.append((geom, 1))
    
    # Add other species polygons as value 0 (we'll handle this differently)
    # Instead of rasterizing all as 2 then converting, we'll use a different approach
    
    # Rasterize target species first
    target_mask = rasterize(
        polygons,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # NoData value (outside target species polygons)
        dtype=np.uint8
    )
    
    # Create a mask for all valid areas (inside any polygon)
    all_polygons = [(geom, 1) for geom in gdf.geometry]
    valid_areas = rasterize(
        all_polygons,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # NoData value (outside all polygons)
        dtype=np.uint8
    )
    
    # Final mask: 1 = target species, 0 = other species, 255 = NoData (outside all polygons)
    mask_raster = np.where(valid_areas == 1, target_mask, 255)
    
    # Apply peatland mask if provided
    if 'peatland_raster_info' in locals():
        print("Applying peatland mask...")
        
        if peatland_raster_info['type'] == 'array':
            # Direct array masking (from shapefile conversion)
            peat_mask = peatland_raster_info['raster']
            mask_raster = np.where((mask_raster == 1) & (peat_mask == 1), 1, 0)
            print("Peatland array mask applied")
            
        elif peatland_raster_info['type'] == 'file':
            # File-based masking (from raster file)
            import tempfile
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                tmp_peat_path = tmp_file.name
            
            try:
                # Use gdalwarp to clip peatland raster to species bounds
                # bounds is a numpy array [left, bottom, right, top]
                gdalwarp_cmd = [
                    'gdalwarp',
                    '-te', str(bounds[0]), str(bounds[1]), str(bounds[2]), str(bounds[3]),
                    '-tr', '5', '5',  # 5m resolution
                    '-r', 'near',  # nearest neighbor resampling
                    '-t_srs', str(crs),
                    peatland_raster_info['path'],
                    tmp_peat_path
                ]
                
                print(f"Running: {' '.join(gdalwarp_cmd)}")
                result = subprocess.run(gdalwarp_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"⚠️  gdalwarp failed: {result.stderr}")
                    print("Continuing without peatland mask...")
                else:
                    # Load the clipped peatland raster
                    with rasterio.open(tmp_peat_path) as peat_src:
                        peat_mask = peat_src.read(1)
                        
                        # Ensure same shape
                        if peat_mask.shape != mask_raster.shape:
                            print(f"⚠️  Shape mismatch: peat {peat_mask.shape} vs mask {mask_raster.shape}")
                            print("Resampling peat mask...")
                            from rasterio.warp import reproject, Resampling
                            resampled_peat = np.empty_like(mask_raster)
                            reproject(
                                peat_mask,
                                resampled_peat,
                                src_transform=peat_src.transform,
                                dst_transform=transform,
                                src_crs=peat_src.crs,
                                dst_crs=crs,
                                resampling=Resampling.nearest
                            )
                            peat_mask = resampled_peat
                        
                        # Apply peatland mask: 
                        # 1 = target species on peat soil
                        # 0 = other species on peat soil
                        # 255 = NoData (non-peat areas, outside compartments)
                        mask_raster = np.where(peat_mask == 1, mask_raster, 255)
                        print("Peatland raster mask applied")
            
            finally:
                # Clean up temporary file
                import os
                if os.path.exists(tmp_peat_path):
                    os.unlink(tmp_peat_path)
    
    # Save raster
    print(f"Writing raster mask to: {output_path}")
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=255,  # Set NoData value
    ) as dst:
        dst.write(mask_raster, 1)
    
    # Calculate statistics - corrected for peatland masking
    total_pixels = mask_raster.size
    target_species_pixels = np.sum(mask_raster == 1)
    other_species_pixels = np.sum(mask_raster == 0)
    nodata_pixels = np.sum(mask_raster == 255)
    
    print(f"\n✅ Species raster mask created successfully!")
    print(f"  Output file: {output_path}")
    print(f"  Shape: {mask_raster.shape}")
    print(f"  Values: {np.unique(mask_raster)}")
    
    print(f"\nStatistics:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Target species pixels (1): {target_species_pixels:,} ({target_species_pixels/total_pixels*100:.1f}%)")
    print(f"  Other species pixels (0): {other_species_pixels:,} ({other_species_pixels/total_pixels*100:.1f}%)")
    print(f"  NoData pixels (outside polygons): {nodata_pixels:,} ({nodata_pixels/total_pixels*100:.1f}%)")
    
    return mask_raster

def main():
    parser = argparse.ArgumentParser(description='Create binary raster mask from shapefile with target species')
    parser.add_argument('shapefile', help='Path to shapefile with species data')
    parser.add_argument('--output', '-o', help='Output path for raster mask')
    parser.add_argument('--peatland-mask', '-p', help='Peatland mask (raster .tif or shapefile .shp) for additional filtering')
    parser.add_argument('--reference-raster', '-r', help='Reference raster for extent/resolution/CRS')
    parser.add_argument('--resolution', type=float, default=5, help='Resolution in meters (if no reference raster)')
    parser.add_argument('--crs', help='Coordinate reference system (if no reference raster)')
    parser.add_argument('--bounds', nargs=4, type=float, metavar=('LEFT', 'BOTTOM', 'RIGHT', 'TOP'), 
                       help='Bounds: left bottom right top (if no reference raster)')
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing output')
    
    args = parser.parse_args()
    
    # Validate inputs
    shapefile_path = Path(args.shapefile)
    if not shapefile_path.exists():
        print(f"❌ Shapefile not found: {shapefile_path}")
        return
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"species_mask_{timestamp}.tif")
    
    # Check if output exists
    if output_path.exists() and not args.force:
        response = input(f"Output file {output_path} exists. Overwrite? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Create species raster mask
    try:
        mask_raster = create_species_raster_mask(
            shapefile_path=shapefile_path,
            output_path=output_path,
            peatland_mask_path=args.peatland_mask,
            reference_raster_path=args.reference_raster,
            resolution=args.resolution,
            crs=args.crs,
            bounds=args.bounds
        )
        
        if mask_raster is not None:
            print(f"\n✅ Species raster mask created successfully!")
        else:
            print(f"\n❌ Failed to create species raster mask")
            
    except Exception as e:
        print(f"❌ Error creating species raster mask: {e}")
        return

if __name__ == "__main__":
    main()
