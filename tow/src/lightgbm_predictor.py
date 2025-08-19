#!/usr/bin/env python3
"""
LightGBM Predictor for Wet Woodland Detection

DESCRIPTION: Inference script for trained LightGBM models. Takes feature tiles (64 embeddings + 3 LiDAR bands) 
and predicts wet woodland probability for each pixel. Outputs binary classification maps (0/1/255) 
with configurable thresholds for precision vs recall trade-offs.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import rasterio
from pathlib import Path
import argparse
from tqdm import tqdm
import gc


class WetWoodlandPredictor:
    def __init__(self, model_path):
        """Initialize predictor with trained model."""
        self.model = lgb.Booster(model_file=model_path)
        # Get feature names from the model
        self.feature_names = self.model.feature_name()
        print(f"📊 Loaded model with {len(self.feature_names)} features")
        
        # Default threshold (can be adjusted)
        self.threshold = 0.690  # F1-optimized threshold from training
        
        print(f"✅ Loaded model from {model_path}")
        print(f"📊 Model has {len(self.feature_names)} features")
        print(f"🎯 Using threshold: {self.threshold}")
    
    def predict_tile(self, data_file, output_file=None, threshold=None):
        """Predict wet woodland probability for a single tile."""
        if threshold is None:
            threshold = self.threshold
            
        print(f"🔍 Processing {Path(data_file).name}...")
        
        try:
            with rasterio.open(data_file) as src:
                # Read data (67 bands: 64 embeddings + 3 LiDAR)
                data = src.read()  # (67, H, W)
                
                if data.shape[0] != 67:
                    print(f"❌ Expected 67 bands, got {data.shape[0]}")
                    return None
                
                # Reshape for prediction: (H*W, 67)
                original_shape = data.shape[1:]
                data_flat = data.reshape(67, -1).T  # (n_pixels, 67)
                
                # Find valid pixels (no NaN)
                valid_mask = ~np.isnan(data_flat).any(axis=1)
                valid_pixels = data_flat[valid_mask]
                
                if len(valid_pixels) == 0:
                    print("❌ No valid pixels found")
                    return None
                
                print(f"📊 Predicting for {len(valid_pixels):,} valid pixels...")
                
                # Predict probabilities
                probabilities = self.model.predict(valid_pixels)
                
                # Convert to binary predictions
                predictions = (probabilities > threshold).astype(np.uint8)
                
                # Create output array
                output = np.full(original_shape, 255, dtype=np.uint8)  # 255 = no data
                output_flat = output.flatten()
                output_flat[valid_mask] = predictions
                output = output_flat.reshape(original_shape)
                
                # Save output
                if output_file:
                    self._save_prediction(output, src, output_file)
                    print(f"✅ Saved prediction to {output_file}")
                
                # Calculate statistics
                n_wet_woodland = predictions.sum()
                n_total = len(valid_pixels)
                percentage = (n_wet_woodland / n_total * 100) if n_total > 0 else 0
                
                print(f"📈 Results: {n_wet_woodland:,} wet woodland pixels ({percentage:.2f}%)")
                
                return {
                    'probabilities': probabilities,
                    'predictions': predictions,
                    'n_wet_woodland': n_wet_woodland,
                    'n_total': n_total,
                    'percentage': percentage
                }
                
        except Exception as e:
            print(f"❌ Error processing {data_file}: {e}")
            return None
    
    def _save_prediction(self, prediction, src, output_file):
        """Save prediction as GeoTIFF."""
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=prediction.shape[0],
            width=prediction.shape[1],
            count=1,
            dtype=prediction.dtype,
            crs=src.crs,
            transform=src.transform,
            nodata=255
        ) as dst:
            dst.write(prediction, 1)
    
    def predict_directory(self, data_dir, output_dir, threshold=None):
        """Predict for all tiles in a directory."""
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Find all feature tiles
        data_files = list(data_dir.glob("*.tif"))
        print(f"🔍 Found {len(data_files)} tiles to process")
        
        results = []
        
        for data_file in tqdm(data_files, desc="Predicting tiles"):
            output_file = output_dir / f"prediction_{data_file.name}"
            
            result = self.predict_tile(str(data_file), str(output_file), threshold)
            if result:
                results.append({
                    'file': data_file.name,
                    'n_wet_woodland': result['n_wet_woodland'],
                    'n_total': result['n_total'],
                    'percentage': result['percentage']
                })
        
        # Summary
        if results:
            total_wet_woodland = sum(r['n_wet_woodland'] for r in results)
            total_pixels = sum(r['n_total'] for r in results)
            avg_percentage = (total_wet_woodland / total_pixels * 100) if total_pixels > 0 else 0
            
            print(f"\n📊 Summary:")
            print(f"   Total tiles processed: {len(results)}")
            print(f"   Total wet woodland pixels: {total_wet_woodland:,}")
            print(f"   Total valid pixels: {total_pixels:,}")
            print(f"   Average wet woodland percentage: {avg_percentage:.2f}%")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Wet Woodland Prediction")
    parser.add_argument("--model", required=True, help="Path to trained LightGBM model")
    parser.add_argument("--data", required=True, help="Input data file or directory")
    parser.add_argument("--output", required=True, help="Output file or directory")
    parser.add_argument("--threshold", type=float, default=0.690, help="Prediction threshold")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = WetWoodlandPredictor(args.model)
    
    # Check if input is file or directory
    data_path = Path(args.data)
    
    if data_path.is_file():
        # Single file prediction
        predictor.predict_tile(args.data, args.output, args.threshold)
    elif data_path.is_dir():
        # Directory prediction
        predictor.predict_directory(args.data, args.output, args.threshold)
    else:
        print(f"❌ Input path not found: {args.data}")


if __name__ == "__main__":
    main()
