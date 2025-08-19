#!/usr/bin/env python3
"""
Dissolve and Rasterize - Single Shapefile to Binary Raster

DESCRIPTION: Simple utility to dissolve multiple polygons into a single feature and rasterize to binary mask. 
Used for creating simplified masks from complex shapefiles. Outputs compressed GeoTIFF with 0/1 values.
"""

import sys
import argparse
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

def dissolve_and_rasterize(input_path: str, output_path: str, resolution: float = 10.0):
    """Dissolve polygons and rasterize to binary mask."""
    
    print(f"📥 Loading: {input_path}")
    gdf = gpd.read_file(input_path)
    print(f"✅ Loaded: {len(gdf):,} features")
    
    # Dissolve all polygons into one
    unified = unary_union(list(gdf.geometry))
    dissolved = gpd.GeoDataFrame({"geometry": [unified]}, crs=gdf.crs)
    print(f"🧮 Dissolved: 1 feature")
    
    # Rasterize
    bounds = dissolved.total_bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = from_bounds(*bounds, width, height)
    
    mask = rasterize(
        shapes=dissolved.geometry,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )
    
    # Write compressed raster
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height, width=width, count=1,
        dtype=np.uint8, crs=dissolved.crs, transform=transform,
        nodata=0, compress='deflate', predictor=2, zlevel=9
    ) as dst:
        dst.write(mask, 1)
    
    coverage = np.sum(mask)
    print(f"🎉 Done: {width}x{height} pixels, {coverage:,} masked ({100*coverage/(width*height):.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Dissolve shapefile to binary raster")
    parser.add_argument("input", help="Input shapefile")
    parser.add_argument("output", help="Output raster (.tif)")
    parser.add_argument("--resolution", type=float, default=10.0, help="Resolution (default: 10)")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ Input not found: {args.input}")
        sys.exit(1)
    
    try:
        dissolve_and_rasterize(args.input, args.output, args.resolution)
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
