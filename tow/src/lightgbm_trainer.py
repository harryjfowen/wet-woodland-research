#!/usr/bin/env python3
"""
Simple Parallel GeoTIFF Processing for LightGBM Training

Much simpler approach than Dask:
1. Use multiprocessing to process each GeoTIFF in parallel
2. Extract valid pixels from each tile independently 
3. Combine results into a single training dataset
4. Train one LightGBM model (LightGBM handles internal parallelization)

Key advantages:
- No Dask cluster setup needed
- Simpler memory management
- Better handling of sparse data (skips empty areas)
- LightGBM's built-in parallelization is very efficient
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import rasterio
import rasterio.windows
from rasterio.warp import reproject, Resampling
from pathlib import Path
import argparse
import glob
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import psutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score


def process_single_tile(args):
    """Process a single GeoTIFF tile to extract valid pixels."""
    data_file, labels_file, max_pixels_per_tile = args
    
    try:
        with rasterio.open(data_file) as data_src:
            # Read all bands
            data = data_src.read()  # Shape: (bands, height, width)
            
            # Create labels array matching data dimensions
            labels = np.zeros((data_src.height, data_src.width), dtype=np.float32)
            
            # Reproject labels to match data tile
            with rasterio.open(labels_file) as labels_src:
                reproject(
                    source=rasterio.band(labels_src, 1),
                    destination=labels,
                    src_transform=labels_src.transform,
                    src_crs=labels_src.crs,
                    dst_transform=data_src.transform,
                    dst_crs=data_src.crs,
                    resampling=Resampling.nearest
                )
            
            # Find valid pixels (no NaN in data, valid labels)
            data_valid = ~np.isnan(data).any(axis=0)
            
            # Handle different label encodings
            if 255 in np.unique(labels):
                labels_valid = (labels == 0) | (labels == 1)
            else:
                labels_valid = (labels >= 0) & (labels <= 1)
            
            valid_mask = data_valid & labels_valid
            
            if valid_mask.sum() == 0:
                return None, None, Path(data_file).name
            
            # Extract valid pixels
            features = data[:, valid_mask].T  # Shape: (n_pixels, n_bands)
            labels_valid = labels[valid_mask].astype(np.int32)
            
            # Subsample if too many pixels (stratified)
            if len(features) > max_pixels_per_tile:
                pos_indices = np.where(labels_valid == 1)[0]
                neg_indices = np.where(labels_valid == 0)[0]
                
                # Keep class balance while subsampling
                pos_ratio = len(pos_indices) / len(labels_valid) if len(labels_valid) > 0 else 0
                n_pos = min(len(pos_indices), int(max_pixels_per_tile * pos_ratio))
                n_neg = min(len(neg_indices), max_pixels_per_tile - n_pos)
                
                sample_indices = []
                if n_pos > 0:
                    sample_indices.extend(np.random.choice(pos_indices, n_pos, replace=False))
                if n_neg > 0:
                    sample_indices.extend(np.random.choice(neg_indices, n_neg, replace=False))
                
                sample_indices = np.array(sample_indices)
                features = features[sample_indices]
                labels_valid = labels_valid[sample_indices]
            
            return features, labels_valid, Path(data_file).name
            
    except Exception as e:
        return None, None, f"{Path(data_file).name}: {str(e)}"


def find_overlapping_tiles(data_dir, labels_file):
    """Find GeoTIFF files that overlap with labels."""
    print("🔍 Finding overlapping tiles...")
    
    data_files = glob.glob(str(Path(data_dir) / "*.tif"))
    overlapping_files = []
    
    with rasterio.open(labels_file) as labels_src:
        labels_bounds = labels_src.bounds
    
    for data_file in tqdm(data_files, desc="Checking overlaps"):
        try:
            with rasterio.open(data_file) as data_src:
                data_bounds = data_src.bounds
                
                # Simple bounding box overlap check
                if (data_bounds.left < labels_bounds.right and 
                    data_bounds.right > labels_bounds.left and
                    data_bounds.bottom < labels_bounds.top and 
                    data_bounds.top > labels_bounds.bottom):
                    overlapping_files.append(data_file)
                    
        except Exception:
            continue
    
    print(f"✅ Found {len(overlapping_files)} overlapping tiles")
    return overlapping_files


def load_data_parallel(data_dir, labels_file, max_tiles=None, max_pixels_per_tile=50000, n_workers=None):
    """Load data using multiprocessing - much simpler than Dask!"""
    print("🔄 Loading data with parallel processing...")
    
    # Find overlapping files
    data_files = find_overlapping_tiles(data_dir, labels_file)
    
    if max_tiles:
        data_files = data_files[:max_tiles]
        print(f"Using first {max_tiles} tiles for testing")
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(data_files), 8)  # Don't use too many
    
    print(f"Processing {len(data_files)} tiles with {n_workers} workers...")
    
    # Prepare arguments for parallel processing
    process_args = [(f, labels_file, max_pixels_per_tile) for f in data_files]
    
    # Process files in parallel
    all_features = []
    all_labels = []
    successful_tiles = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_tile, args): args[0] 
                         for args in process_args}
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_file), total=len(data_files), desc="Processing"):
            features, labels, info = future.result()
            
            if features is not None:
                all_features.append(features)
                all_labels.append(labels)
                successful_tiles += 1
            else:
                print(f"❌ Failed: {info}")
    
    if not all_features:
        print("❌ No valid data found!")
        return None, None
    
    # Combine all data
    print("🔗 Combining data from all tiles...")
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print(f"✅ Final dataset:")
    print(f"   Successful tiles: {successful_tiles}/{len(data_files)}")
    print(f"   Total samples: {len(X):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Positive samples: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
    print(f"   Memory usage: ~{X.nbytes / 1024**2:.1f} MB")
    
    return X, y


def train_lightgbm_simple(X, y, test_size=0.2, **lgb_params):
    """Train LightGBM with its built-in parallelization."""
    print("🚀 Training LightGBM (simple approach)...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    print(f"Train: {len(X_train):,} samples ({y_train.sum():,} positive)")
    print(f"Test:  {len(X_test):,} samples ({y_test.sum():,} positive)")
    
    # Default parameters optimized for your use case
    default_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'random_state': 42,
        'n_jobs': -1,  # Use all CPU cores
        'verbose': -1,
        'scale_pos_weight': 10.0,  # Handle class imbalance
    }
    
    # Update with user parameters
    default_params.update(lgb_params)
    
    # Create and train model
    model = lgb.LGBMClassifier(**default_params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with multiple metrics."""
    print("\n📊 Model Evaluation:")
    print("=" * 50)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.02)
    f1_scores = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba > thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_thresh))
    
    # Use optimal threshold
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
    
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Best F1 score: {max(f1_scores):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, y_pred_optimal):.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_optimal, target_names=['Other', 'Wet Woodland']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_optimal)
    print(f"\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"Actual    Other  Wet Woodland")
    print(f"Other     {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"Wet Wood  {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Feature importance (top 10)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        n_features = len(importance)
        
        if n_features == 67:
            feature_names = [f'embed_{i+1}' for i in range(64)] + ['chm', 'dtm', 'dsm']
        elif n_features == 64:
            feature_names = [f'embed_{i+1}' for i in range(64)]
        elif n_features == 3:
            feature_names = ['chm', 'dtm', 'dsm']
        else:
            feature_names = [f'feature_{i+1}' for i in range(n_features)]
        
        # Sort by importance
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, (name, imp) in enumerate(feature_importance[:10]):
            print(f"{i+1:2d}. {name:12s}: {imp:8.0f}")


def main():
    parser = argparse.ArgumentParser(description="Simple Parallel GeoTIFF Processing for LightGBM")
    parser.add_argument("--data-dir", required=True, help="Directory with GeoTIFF tiles")
    parser.add_argument("--labels-file", required=True, help="Labels GeoTIFF file")
    parser.add_argument("--max-tiles", type=int, help="Max tiles to process")
    parser.add_argument("--max-pixels-per-tile", type=int, default=50000, help="Max pixels per tile")
    parser.add_argument("--n-workers", type=int, help="Number of parallel workers")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--save-model", default="wet_woodland_model.pkl", help="Model save path")
    
    # LightGBM parameters
    parser.add_argument("--num-leaves", type=int, default=31, help="Number of leaves")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--n-estimators", type=int, default=1000, help="Number of estimators")
    
    args = parser.parse_args()
    
    print("🌲 Simple Parallel GeoTIFF Processing for LightGBM")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Labels file: {args.labels_file}")
    print(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total/1024**3:.0f}GB RAM")
    
    # Verify files exist
    if not Path(args.labels_file).exists():
        print(f"❌ Labels file not found: {args.labels_file}")
        return
    
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Load data using multiprocessing
    X, y = load_data_parallel(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        max_tiles=args.max_tiles,
        max_pixels_per_tile=args.max_pixels_per_tile,
        n_workers=args.n_workers
    )
    
    if X is None:
        print("❌ Failed to load data")
        return
    
    # Train model
    lgb_params = {
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
    }
    
    model = train_lightgbm_simple(X, y, test_size=args.test_size, **lgb_params)
    
    # Save model
    model_path = Path(args.save_model)
    model_path.parent.mkdir(exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✅ Training complete! Model saved to {args.save_model}")


if __name__ == "__main__":
    main()