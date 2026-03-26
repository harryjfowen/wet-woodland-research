#!/usr/bin/env python3
"""
Convert peat soil probability raster to binary mask based on threshold

DESCRIPTION: Converts peat soil probability rasters to binary masks for spatial analysis. Applies 
configurable probability thresholds to create binary classification (1 = peat soil, 0 = non-peat, 255 = no data). 
Supports resampling to target resolution and outputs compressed GeoTIFF files. Used to create 
peatland masks for wet woodland detection and training data preparation.
"""

import rasterio
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

def create_peat_binary_mask(input_raster_path, output_path, threshold=0.5, resolution=5):
    """
    Create binary peat mask from probability raster
    
    Args:
        input_raster_path: Path to peat probability raster
        output_path: Path for output binary mask
        threshold: Probability threshold (default: 0.5)
        resolution: Target resolution in meters (default: 5)
    """
    print(f"Loading peat probability raster: {input_raster_path}")
    
    with rasterio.open(input_raster_path) as src:
        # Read the probability data
        prob_data = src.read(1)  # Read first band
        
        print(f"Input raster info:")
        print(f"  Shape: {prob_data.shape}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Current resolution: {src.res}")
        print(f"  Data range: {np.nanmin(prob_data):.3f} to {np.nanmax(prob_data):.3f}")
        print(f"  NoData values: {np.sum(np.isnan(prob_data)):,}")
        
        # Create binary mask based on threshold
        print(f"Creating binary mask with threshold: {threshold}")
        binary_mask = np.where(prob_data >= threshold, 1, 0)
        
        # Preserve NoData areas - use a special value that will become NoData
        binary_mask = np.where(np.isnan(prob_data), 255, binary_mask)  # Use 255 as NoData placeholder
        
        # Resample to 5m resolution if needed
        current_res = src.res[0]  # Assuming square pixels
        if abs(current_res - resolution) > 0.1:  # If resolution differs by more than 0.1m
            print(f"Resampling from {current_res}m to {resolution}m resolution...")
            
            # Calculate new dimensions
            scale_factor = current_res / resolution
            new_height = int(src.height * scale_factor)
            new_width = int(src.width * scale_factor)
            
            # Update transform
            new_transform = rasterio.transform.from_origin(
                src.bounds.left, src.bounds.top, resolution, resolution
            )
            
            # Resample using nearest neighbor (preserves binary values)
            from rasterio.warp import reproject, Resampling
            
            resampled_mask = np.empty((new_height, new_width), dtype=np.uint8)
            reproject(
                binary_mask,
                resampled_mask,
                src_transform=src.transform,
                dst_transform=new_transform,
                src_crs=src.crs,
                dst_crs=src.crs,
                resampling=Resampling.nearest
            )
            
            binary_mask = resampled_mask
            transform = new_transform
            height, width = new_height, new_width
        else:
            transform = src.transform
            height, width = src.height, src.width
        
        # Save binary mask
        print(f"Writing binary mask to: {output_path}")
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            crs=src.crs,
            transform=transform,
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256,
            nodata=255,  # Set NoData value
        ) as dst:
            dst.write(binary_mask, 1)
    
    # Calculate statistics
    total_pixels = binary_mask.size
    peat_pixels = np.sum(binary_mask == 1)
    non_peat_pixels = np.sum(binary_mask == 0)
    nodata_pixels = np.sum(binary_mask == 255)
    
    # Calculate valid pixels (excluding NoData)
    valid_pixels = total_pixels - nodata_pixels
    
    print(f"\n✅ Peat binary mask created successfully!")
    print(f"  Output file: {output_path}")
    print(f"  Shape: {binary_mask.shape}")
    print(f"  Values: {np.unique(binary_mask)}")
    print(f"  Resolution: {resolution}m")
    
    print(f"\nStatistics:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
    print(f"  Peat pixels (1): {peat_pixels:,} ({peat_pixels/valid_pixels*100:.1f}% of valid pixels)")
    print(f"  Non-peat pixels (0): {non_peat_pixels:,} ({non_peat_pixels/valid_pixels*100:.1f}% of valid pixels)")
    print(f"  NoData pixels: {nodata_pixels:,} ({nodata_pixels/total_pixels*100:.1f}%)")
    
    return binary_mask

def main():
    parser = argparse.ArgumentParser(description='Convert peat probability raster to binary mask')
    parser.add_argument('input_raster', help='Path to peat probability raster')
    parser.add_argument(
        '--output',
        '-o',
        help='Output path for binary mask (default: tow/data/output/labels/peat_binary_mask_<timestamp>.tif)',
    )
    parser.add_argument('--threshold', '-t', type=float, default=0.5, 
                       help='Probability threshold (default: 0.5)')
    parser.add_argument('--resolution', '-r', type=float, default=5, 
                       help='Target resolution in meters (default: 5)')
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing output')
    
    args = parser.parse_args()
    
    # Validate inputs
    input_raster_path = Path(args.input_raster)
    if not input_raster_path.exists():
        print(f"❌ Input raster not found: {input_raster_path}")
        return
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "output"
            / "labels"
            / f"peat_binary_mask_{timestamp}.tif"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if output exists
    if output_path.exists() and not args.force:
        response = input(f"Output file {output_path} exists. Overwrite? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Create binary mask
    try:
        binary_mask = create_peat_binary_mask(
            input_raster_path=input_raster_path,
            output_path=output_path,
            threshold=args.threshold,
            resolution=args.resolution
        )
        print(f"\n✅ Peat binary mask created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating peat binary mask: {e}")
        return

if __name__ == "__main__":
    main()
