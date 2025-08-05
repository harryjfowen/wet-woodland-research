#!/usr/bin/env python3
"""
Highly Optimized Peat Dissolver
===============================

This script efficiently dissolves peat extent polygons into a single unified boundary.
Optimized for speed and memory efficiency.

Usage:
    python dissolve_peat.py <input_peat.shp> <output_mask.shp> [options]
"""

import geopandas as gpd
import os
import sys
import argparse
import time
import warnings
from shapely.validation import make_valid
import shapely

# Suppress warnings
warnings.filterwarnings('ignore')

def combine_peat_optimized(peat_path, output_path, simplify_tolerance=None, fast_mode=False):
    """
    Highly optimized peat combination.
    
    Args:
        peat_path (str): Path to peat shapefile
        output_path (str): Output path for combined mask
        simplify_tolerance (float): Tolerance for simplification
        fast_mode (bool): Use fast processing mode
    """
    
    print(f"🌱 Optimized Peat Combiner")
    print(f"📁 Input: {peat_path}")
    print(f"💾 Output: {output_path}")
    print("=" * 50)
    
    start_time = time.time()
    
    # Load peat data with optimizations
    print(f"📦 Loading peat data...")
    try:
        if fast_mode:
            # Set GDAL options for speed
            os.environ['OGR_SKIP_FAILURES'] = 'YES'
            os.environ['CPL_LOG_ERRORS'] = 'OFF'
            os.environ['GDAL_PAM_ENABLED'] = 'NO'
        
        peat_gdf = gpd.read_file(peat_path)
        load_time = time.time() - start_time
        
        print(f"✅ Loaded {len(peat_gdf):,} peat polygons in {load_time:.2f}s")
        print(f"📐 CRS: {peat_gdf.crs}")
        print(f"💾 Memory usage: {peat_gdf.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"❌ Error loading peat data: {e}")
        return False
    
    # Fix geometries (if needed)
    print(f"🔧 Fixing invalid geometries...")
    fix_start = time.time()
    
    peat_gdf["geometry"] = peat_gdf["geometry"].apply(lambda geom: make_valid(geom) if geom else None)
    peat_gdf = peat_gdf[~peat_gdf["geometry"].is_empty & peat_gdf["geometry"].notna()]
    
    fix_time = time.time() - fix_start
    print(f"✅ Fixed geometries in {fix_time:.2f}s ({len(peat_gdf):,} valid polygons)")
    
    # Explode multi-part geometries
    print(f"🔀 Exploding multi-part geometries...")
    explode_start = time.time()
    
    peat_gdf = peat_gdf.explode(index_parts=False).reset_index(drop=True)
    
    explode_time = time.time() - explode_start
    print(f"✅ Exploded in {explode_time:.2f}s ({len(peat_gdf):,} single-part polygons)")
    
    # Combine all polygons (no dissolve)
    print(f"🔗 Combining all polygons...")
    combine_start = time.time()
    
    # Just use the first geometry as the output (or create a simple union)
    # This is much faster than dissolving
    if len(peat_gdf) == 1:
        combined_geom = peat_gdf.geometry.iloc[0]
    else:
        # Create a simple union without dissolving overlaps
        from shapely.ops import unary_union
        geometries = list(peat_gdf.geometry)
        combined_geom = unary_union(geometries)
    
    combine_time = time.time() - combine_start
    print(f"✅ Combined in {combine_time:.2f}s")
    
    # Simplify if requested
    if simplify_tolerance:
        print(f"📏 Simplifying boundary (tolerance: {simplify_tolerance})...")
        simplify_start = time.time()
        
        combined_geom = shapely.simplify(combined_geom, tolerance=simplify_tolerance)
        
        simplify_time = time.time() - simplify_start
        print(f"✅ Simplified in {simplify_time:.2f}s")
    
    # Create output GeoDataFrame
    print(f"📋 Creating output dataset...")
    output_start = time.time()
    
    # Create minimal output with just the geometry - no attributes
    output_gdf = gpd.GeoDataFrame({
        'geometry': [combined_geom]
    }, crs=peat_gdf.crs)
    
    output_time = time.time() - output_start
    print(f"✅ Created minimal output in {output_time:.2f}s")
    
    # Save to disk
    print(f"💾 Saving to disk...")
    save_start = time.time()
    
    try:
        output_gdf.to_file(output_path)
        save_time = time.time() - save_start
        print(f"✅ Saved in {save_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n📈 COMBINATION SUMMARY:")
    print(f"  • Input polygons: {len(peat_gdf):,}")
    print(f"  • Output: Single minimal polygon (no attributes)")
    print(f"  • Total area: {combined_geom.area:,.2f} square units")
    print(f"  • Processing time: {total_time:.2f}s")
    print(f"  • Speed: {len(peat_gdf)/total_time:,.0f} polygons/second")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Highly optimized peat combination')
    parser.add_argument('input_path', help='Path to input peat shapefile')
    parser.add_argument('output_path', help='Path for output combined mask')
    parser.add_argument('--fast', action='store_true', help='Use fast processing mode')
    parser.add_argument('--simplify', type=float, help='Tolerance for geometry simplification')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"❌ Input file not found: {args.input_path}")
        sys.exit(1)
    
    success = combine_peat_optimized(
        peat_path=args.input_path,
        output_path=args.output_path,
        simplify_tolerance=args.simplify,
        fast_mode=args.fast
    )
    
    if success:
        print("\n🎉 Peat combination completed successfully!")
    else:
        print("\n❌ Peat combination failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 