#!/usr/bin/env python3
"""
Fast Parallel LightGBM Prediction - Memory Efficient

Uses the same approach as lightgbm_trainer.py but with memory optimization:
1. Load data from tiles in parallel (multiprocessing for I/O)
2. Use LightGBM's built-in parallelization for prediction (n_jobs=-1)
3. Process tiles immediately as they're loaded to prevent memory buildup
4. Clear memory after each tile
"""

import numpy as np
import rasterio
from rasterio.merge import merge
from pathlib import Path
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import lightgbm as lgb
import glob
import gc


def load_and_predict_single_tile_worker(args):
    """Worker function to load data and predict for a single tile - memory efficient."""
    model_path, data_file, output_file, threshold = args
    
    try:
        # Load model in each worker
        model = lgb.Booster(model_file=model_path)
        
        with rasterio.open(data_file) as src:
            # Read all bands (same as trainer)
            data = src.read()  # Shape: (bands, height, width)
            
            if data.shape[0] != 67:
                return None
            
            # Find valid pixels (same logic as trainer)
            data_valid = ~np.isnan(data).any(axis=0)
            
            if data_valid.sum() == 0:
                return None
            
            # Extract valid pixels (same as trainer) - this reduces memory usage significantly
            features = data[:, data_valid].T  # Shape: (n_pixels, n_bands)
            
            # Clear the full data array to save memory
            del data
            
            # Predict immediately
            predictions_proba = model.predict(features)
            binary_predictions = (predictions_proba > threshold).astype(np.uint8)
            
            # Create output arrays
            binary_output = np.full((src.height, src.width), 255, dtype=np.uint8)
            prob_output = np.full((src.height, src.width), np.nan, dtype=np.float32)
            
            binary_output[data_valid] = binary_predictions
            prob_output[data_valid] = predictions_proba
            
            # Save 2-band output
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=src.height,
                width=src.width,
                count=2,
                dtype='float32',
                crs=src.crs,
                transform=src.transform,
                nodata=255
            ) as dst:
                dst.write(binary_output.astype(np.float32), 1)
                dst.write(prob_output, 2)
                dst.set_band_description(1, "Binary Classification (0=No Wet Woodland, 1=Wet Woodland)")
                dst.set_band_description(2, "Probability (0.0-1.0)")
            
            # Calculate statistics
            n_wet_woodland = binary_predictions.sum()
            n_total = len(features)
            percentage = (n_wet_woodland / n_total * 100) if n_total > 0 else 0
            
            # Clear memory
            del features, predictions_proba, binary_predictions, binary_output, prob_output
            
            return {
                'file': Path(data_file).name,
                'n_wet_woodland': n_wet_woodland,
                'n_total': n_total,
                'percentage': percentage,
                'output_file': output_file
            }
            
    except Exception as e:
        return None


def find_tiles(data_dir):
    """Find all GeoTIFF files in directory."""
    data_files = glob.glob(str(Path(data_dir) / "*.tif"))
    return data_files


def predict_directory_parallel(model_path, data_dir, output_dir, threshold=0.550, n_workers=None, create_mosaic=False):
    """Predict for all tiles using parallel processing with immediate memory clearing."""
    print("🚀 Fast Parallel LightGBM Prediction (Memory Optimized)")
    print("=" * 50)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"✅ Using model: {model_path}")
    print(f"🎯 Using threshold: {threshold}")
    
    # Find all tiles
    data_files = find_tiles(data_dir)
    if not data_files:
        print(f"❌ No TIFF files found in {data_dir}")
        return []
    
    print(f"🔍 Found {len(data_files)} tiles to process")
    
    # Auto-detect number of workers (unless specified)
    if n_workers is None:
        # Use fewer workers to avoid memory issues
        n_workers = min(mp.cpu_count() // 2, len(data_files), 16)
        # Ensure at least 1 worker
        n_workers = max(1, n_workers)
    print(f"🔧 Using {n_workers} workers (memory optimized)")
    
    # Prepare arguments for parallel processing
    process_args = [(model_path, f, str(output_dir / f"prediction_{Path(f).name}"), threshold) 
                   for f in data_files]
    
    # Process tiles in parallel with immediate memory clearing
    results = []
    total_wet_woodland = 0
    total_pixels = 0
    
    print("🚀 Processing tiles in parallel...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(load_and_predict_single_tile_worker, args): args[1] 
                         for args in process_args}
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_file), total=len(data_files), desc="Processing"):
            data_file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    total_wet_woodland += result['n_wet_woodland']
                    total_pixels += result['n_total']
                    # Don't print during processing to avoid interrupting progress bar
                # Don't print failures to avoid interrupting progress bar
            except Exception as e:
                # Only print critical errors
                pass
    
    # Summary
    if results:
        avg_percentage = (total_wet_woodland / total_pixels * 100) if total_pixels > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"   Successful tiles: {len(results)}/{len(data_files)}")
        print(f"   Total wet woodland pixels: {total_wet_woodland:,}")
        print(f"   Total valid pixels: {total_pixels:,}")
        print(f"   Average wet woodland percentage: {avg_percentage:.2f}%")
    
    # Create mosaic if requested
    if create_mosaic and results:
        print(f"\n🧩 Creating mosaic from {len(results)} tiles...")
        mosaic_file = create_mosaic_from_results(results, output_dir)
        if mosaic_file:
            print(f"✅ Mosaic created: {mosaic_file}")
    
    return results


def create_mosaic_from_results(results, output_dir):
    """Create a mosaic from all prediction tiles."""
    try:
        # Get all output files
        raster_files = [result['output_file'] for result in results if result['output_file']]
        
        if not raster_files:
            return None
        
        # Read all rasters and keep them open
        raster_list = []
        open_files = []
        for raster_file in raster_files:
            src = rasterio.open(raster_file)
            raster_list.append(src)
            open_files.append(src)
        
        # Merge rasters
        mosaic, out_transform = merge(raster_list)
        
        # Get metadata from first file (while it's still open)
        out_meta = open_files[0].meta.copy()
        
        # Close all files
        for src in open_files:
            src.close()
        
        # Update metadata for mosaic
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "count": 2,
            "dtype": "float32"
        })
        
        # Save mosaic
        mosaic_file = output_dir / "prediction_mosaic.tif"
        with rasterio.open(mosaic_file, "w", **out_meta) as dest:
            dest.write(mosaic)
            dest.set_band_description(1, "Binary Classification (0=No Wet Woodland, 1=Wet Woodland)")
            dest.set_band_description(2, "Probability (0.0-1.0)")
        
        return str(mosaic_file)
        
    except Exception as e:
        print(f"❌ Error creating mosaic: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fast Parallel LightGBM Prediction")
    parser.add_argument("--model", required=True, help="Path to trained LightGBM model")
    parser.add_argument("--data", required=True, help="Directory containing GeoTIFF files")
    parser.add_argument("--output", required=True, help="Output directory for predictions")
    parser.add_argument("--threshold", type=float, default=0.550, help="Classification threshold")
    parser.add_argument("--workers", type=int, help="Number of parallel workers for data loading")
    parser.add_argument("--create-mosaic", action="store_true", help="Create mosaic from all tiles")
    
    args = parser.parse_args()
    
    # Run prediction
    results = predict_directory_parallel(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        threshold=args.threshold,
        n_workers=args.workers,
        create_mosaic=args.create_mosaic
    )
    
    if results:
        print(f"\n✅ Prediction complete! Results saved to {args.output}")
    else:
        print(f"\n❌ No predictions generated")


if __name__ == "__main__":
    main()
