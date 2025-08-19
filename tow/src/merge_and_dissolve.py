#!/usr/bin/env python3
"""
Merge and Dissolve

DESCRIPTION: Vector processing utility that merges multiple shapefiles and dissolves geometries. 
Supports attribute-based dissolving, geometry simplification, and raster output. Optimized with pyogrio 
for fast I/O operations. Used for combining and simplifying complex vector datasets.
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.validation import make_valid
from shapely.ops import unary_union
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds


warnings.filterwarnings("ignore")


def read_vector(path: str) -> gpd.GeoDataFrame:
    """Read a vector dataset using the fastest available backend.

    Tries pyogrio first (fast), falls back to GeoPandas/Fiona.
    """
    try:
        import pyogrio  # noqa: F401
        # geopandas automatically uses pyogrio if installed in recent versions
        return gpd.read_file(path)
    except Exception:
        # Fallback to default engine
        return gpd.read_file(path)


def write_vector(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    """Write a vector dataset, inferring driver from extension.

    Uses pyogrio when available for performance.
    """
    out_path = str(out_path)
    suffix = Path(out_path).suffix.lower()

    # Infer driver
    if suffix == ".shp":
        driver = "ESRI Shapefile"
    elif suffix == ".gpkg":
        driver = "GPKG"
    elif suffix in {".geojson", ".json"}:
        driver = "GeoJSON"
    else:
        # Default to GeoPackage if unknown and add extension
        driver = "GPKG"
        out_path = out_path + ".gpkg" if suffix == "" else out_path

    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyogrio  # noqa: F401
        # Use GeoPandas API; it will leverage pyogrio if available
        gdf.to_file(out_path, driver=driver)
    except Exception:
        # Fallback
        gdf.to_file(out_path, driver=driver)


def fix_and_filter_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make geometries valid and drop empties/nulls."""
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: make_valid(geom) if geom is not None else None)
    # Drop empty or null geometries
    gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()].copy()
    return gdf


def ensure_polygon_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter to polygonal geometries for Shapefile compatibility."""
    if gdf.empty:
        return gdf
    valid_types = {"Polygon", "MultiPolygon"}
    mask = gdf.geometry.geom_type.isin(valid_types)
    filtered = gdf[mask].copy()
    return filtered


def merge_and_dissolve(
    input_a: str,
    input_b: str,
    output_path: str,
    dissolve_by: str | None = None,
    simplify_tolerance: float | None = None,
    fast: bool = False,
    raster: bool = False,
    resolution: float = 10.0,
) -> None:
    if fast:
        # Favor speed over logs/warnings
        os.environ["OGR_SKIP_FAILURES"] = "YES"
        os.environ["CPL_LOG_ERRORS"] = "OFF"
        os.environ["GDAL_PAM_ENABLED"] = "NO"

    print("🧩 Merge + Dissolve")
    print(f"📥 A: {input_a}")
    print(f"📥 B: {input_b}")
    print(f"💾 Out: {output_path}")
    if dissolve_by:
        print(f"🧮 Dissolve by: {dissolve_by}")
    if simplify_tolerance is not None:
        print(f"📏 Simplify tolerance: {simplify_tolerance}")
    if raster:
        print(f"📊 Raster output: {resolution} unit resolution")

    # Read inputs
    gdf_a = read_vector(input_a)
    gdf_b = read_vector(input_b)

    if gdf_a.empty and gdf_b.empty:
        raise ValueError("Both inputs are empty; nothing to merge.")

    print(f"✅ Loaded: A={len(gdf_a):,} features, B={len(gdf_b):,} features")

    # Align CRS (reproject B to A)
    if gdf_a.crs != gdf_b.crs and gdf_a.crs is not None:
        gdf_b = gdf_b.to_crs(gdf_a.crs)
        print(f"📐 Reprojected B to CRS {gdf_a.crs}")

    # Fix invalids and drop empties
    gdf_a = fix_and_filter_geometries(gdf_a)
    gdf_b = fix_and_filter_geometries(gdf_b)

    # Merge
    merged = gpd.GeoDataFrame(pd.concat([gdf_a, gdf_b], ignore_index=True), crs=gdf_a.crs or gdf_b.crs)
    print(f"🔗 Merged features: {len(merged):,}")

    # Dissolve
    if dissolve_by:
        if dissolve_by not in merged.columns:
            raise ValueError(f"Column '{dissolve_by}' not found in merged dataset")
        dissolved = merged.dissolve(by=dissolve_by, as_index=False)
    else:
        # Dissolve everything into one
        unified = unary_union(list(merged.geometry))
        dissolved = gpd.GeoDataFrame({"geometry": [unified]}, crs=merged.crs)

    # Simplify (optional)
    if simplify_tolerance is not None and not dissolved.empty:
        dissolved = dissolved.copy()
        dissolved["geometry"] = dissolved["geometry"].simplify(tolerance=simplify_tolerance, preserve_topology=True)

    # Ensure polygonal output for Shapefile
    if Path(output_path).suffix.lower() == ".shp":
        dissolved = ensure_polygon_only(dissolved)
        if dissolved.empty:
            raise ValueError("No polygon geometries remain after filtering; cannot write Shapefile.")

    # Minimal attributes if dissolved-all
    if dissolve_by is None:
        dissolved = gpd.GeoDataFrame({"geometry": dissolved.geometry}, crs=dissolved.crs)

    # Write
    if raster:
        # Output as binary raster mask
        print(f"📊 Rasterizing to binary mask (resolution: {resolution})...")
        
        # Get bounds of dissolved geometry
        bounds = dissolved.total_bounds
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)
        
        # Create transform
        transform = from_bounds(*bounds, width, height)
        
        # Rasterize geometries to binary mask
        mask = rasterize(
            shapes=dissolved.geometry,
            out_shape=(height, width),
            transform=transform,
            fill=0,  # background
            default_value=1,  # foreground (mask)
            dtype=np.uint8,
        )
        
        # Write raster
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            crs=dissolved.crs,
            transform=transform,
            nodata=0,
            compress='deflate',
            predictor=2,
            zlevel=9,
        ) as dst:
            dst.write(mask, 1)
        
        print(f"🎉 Wrote binary raster mask → {output_path}")
        print(f"   Size: {width}x{height} pixels")
        print(f"   Resolution: {resolution} units")
        print(f"   Coverage: {np.sum(mask):,} pixels ({100*np.sum(mask)/(width*height):.1f}%)")
    else:
        # Output as vector
        write_vector(dissolved, output_path)
        print(f"🎉 Wrote {len(dissolved):,} feature(s) → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two vector datasets, dissolve, and export")
    parser.add_argument("input_a", help="Path to first input (e.g., .shp, .gpkg)")
    parser.add_argument("input_b", help="Path to second input (e.g., .shp, .gpkg)")
    parser.add_argument("output", help="Path to output (e.g., .shp, .gpkg, .geojson, .tif)")
    parser.add_argument("--by", dest="dissolve_by", help="Attribute column to dissolve by (default: dissolve all)")
    parser.add_argument("--simplify", type=float, help="Simplification tolerance (units of CRS)")
    parser.add_argument("--fast", action="store_true", help="Enable fast-mode GDAL settings")
    parser.add_argument("--raster", action="store_true", help="Output as binary raster mask (.tif)")
    parser.add_argument("--resolution", type=float, default=10.0, help="Raster resolution in CRS units (default: 10)")

    args = parser.parse_args()

    for p in [args.input_a, args.input_b]:
        if not Path(p).exists():
            print(f"❌ Input not found: {p}")
            sys.exit(1)

    try:
        merge_and_dissolve(
            input_a=args.input_a,
            input_b=args.input_b,
            output_path=args.output,
            dissolve_by=args.dissolve_by,
            simplify_tolerance=args.simplify,
            fast=args.fast,
            raster=args.raster,
            resolution=args.resolution,
        )
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


