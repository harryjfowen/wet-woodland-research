#!/usr/bin/env python3
"""
LightGBM Trainer for Wet Woodland Detection

DESCRIPTION: Training script for LightGBM models using 64-dimensional temporal embeddings + 3 LiDAR bands. 
Handles large-scale raster data with caching, supports class imbalance, and provides comprehensive evaluation 
with F1/balanced accuracy optimization. Includes early stopping and feature importance analysis.
Flexible feature handling - can work with embeddings only, LiDAR only, or any combination of features.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score, precision_score, recall_score
import rasterio
from pathlib import Path
import argparse
import glob
from tqdm import tqdm
import gc
import pickle
import hashlib


def find_overlapping_tiles(data_dir, labels_file, debug=False):
    """Find data tiles that overlap with the labels file."""
    print("🔍 Finding overlapping tiles...")
    
    data_files = glob.glob(str(Path(data_dir) / "*.tif"))
    print(f"Found {len(data_files)} total data files")
    
    if debug and len(data_files) > 0:
        print(f"First few data files: {data_files[:3]}")
    
    overlapping_files = []
    
    # Get labels bounds
    with rasterio.open(labels_file) as labels_src:
        labels_bounds = labels_src.bounds
        labels_crs = labels_src.crs
        print(f"Labels bounds: {labels_bounds}")
        print(f"Labels CRS: {labels_crs}")
    
    for data_file in tqdm(data_files, desc="Checking overlaps"):
        try:
            with rasterio.open(data_file) as data_src:
                data_bounds = data_src.bounds
                data_crs = data_src.crs
                
                if debug:
                    print(f"Checking {Path(data_file).name}: bounds={data_bounds}, crs={data_crs}")
                
                # Check if bounds overlap (simple bbox check)
                if (data_bounds.left < labels_bounds.right and 
                    data_bounds.right > labels_bounds.left and
                    data_bounds.bottom < labels_bounds.top and 
                    data_bounds.top > labels_bounds.bottom):
                    overlapping_files.append(data_file)
                    if debug:
                        print(f"✅ Overlap found: {Path(data_file).name}")
        except Exception as e:
            print(f"⚠️ Skipping {data_file}: {e}")
            continue
    
    print(f"✅ Found {len(overlapping_files)} overlapping tiles out of {len(data_files)} total")
    if debug and overlapping_files:
        print(f"Overlapping files: {[Path(f).name for f in overlapping_files[:5]]}")
    return overlapping_files


def extract_valid_pixels(data_file, labels_file, max_pixels_per_tile=50000, debug=False, use_cache=True):
    """Extract valid pixels from a single data tile."""
    
    # Try to load from cache first
    if use_cache:
        cache_key = get_cache_key(data_file, labels_file, max_pixels_per_tile)
        cached_data = load_from_cache(cache_key)
        if cached_data is not None:
            if debug:
                print(f"✅ Loaded from cache: {Path(data_file).name}")
            return cached_data
    
    try:
        with rasterio.open(data_file) as data_src, rasterio.open(labels_file) as labels_src:
            if debug:
                print(f"Processing {Path(data_file).name}")
                print(f"Data shape: {data_src.shape}, bands: {data_src.count}")
                print(f"Labels shape: {labels_src.shape}, bands: {labels_src.count}")
            
            # Read data (64 embeddings + 3 lidar bands)
            data = data_src.read()  # (67, H, W)
            
            if debug:
                print(f"Data array shape: {data.shape}")
                print(f"Data value range: {data.min():.3f} to {data.max():.3f}")
                print(f"Data has NaN: {np.isnan(data).any()}")
            
            # Get matching window in labels file
            # The labels file is full-scale, so we need to extract the window that corresponds to this data tile
            try:
                # Calculate the window in the labels file that corresponds to this data tile
                window = rasterio.windows.from_bounds(
                    *data_src.bounds, labels_src.transform
                )
                
                if debug:
                    print(f"Data tile bounds: {data_src.bounds}")
                    print(f"Labels transform: {labels_src.transform}")
                    print(f"Calculated window: {window}")
                
                # Read the labels for this window
                labels = labels_src.read(1, window=window)
                
                if debug:
                    print(f"Labels read with window: {labels.shape}")
                
                # The labels should now be the same size as the data tile
                # If not, we need to handle the resolution difference
                if labels.shape != (data_src.height, data_src.width):
                    if debug:
                        print(f"Shape mismatch: labels {labels.shape} vs data {(data_src.height, data_src.width)}")
                        print("This might be due to different resolutions - resizing labels to match data")
                    
                    from skimage.transform import resize
                    labels = resize(labels, (data_src.height, data_src.width), order=0, preserve_range=True)
                
            except Exception as e:
                if debug:
                    print(f"Window reading failed: {e}")
                    print("Trying alternative approach with reprojection...")
                
                # Alternative approach: use rasterio's reproject function
                try:
                    from rasterio.warp import reproject, Resampling
                    
                    # Create a destination array with the data tile dimensions
                    labels = np.zeros((data_src.height, data_src.width), dtype=np.float32)
                    
                    # Reproject the labels to match the data tile's coordinate system and resolution
                    reproject(
                        source=rasterio.band(labels_src, 1),
                        destination=labels,
                        src_transform=labels_src.transform,
                        src_crs=labels_src.crs,
                        dst_transform=data_src.transform,
                        dst_crs=data_src.crs,
                        resampling=Resampling.nearest
                    )
                    
                    if debug:
                        print(f"Reprojection successful: {labels.shape}")
                        
                except Exception as e2:
                    if debug:
                        print(f"Reprojection also failed: {e2}")
                        print("This tile might be outside the labels extent or have CRS issues")
                    return None, None
            
            if debug:
                print(f"Labels shape after processing: {labels.shape}")
                print(f"Labels unique values: {np.unique(labels)}")
                print(f"Labels value range: {labels.min():.3f} to {labels.max():.3f}")
            
            # Find valid pixels
            # Valid = has data in embeddings AND valid label (0 or 1, not 255/nodata)
            data_valid = ~np.isnan(data).any(axis=0)  # No NaN in any band
            
            # Handle different label encodings
            # Some files use 0/1, others might use 255 for no-data
            if debug:
                print(f"Labels unique values: {np.unique(labels)}")
                print(f"Labels value range: {labels.min():.3f} to {labels.max():.3f}")
            
            # Check if labels are binary (0,1) or have no-data values (255)
            unique_labels = np.unique(labels)
            if debug:
                print(f"Labels unique values: {unique_labels}")
                print(f"Labels value counts: {np.bincount(labels.astype(int)) if len(unique_labels) <= 10 else 'Too many unique values'}")
            
            if 255 in unique_labels:
                # Labels use 255 for no-data, 0 and 1 for classes
                labels_valid = (labels == 0) | (labels == 1)
                if debug:
                    print("Using 255 as no-data value")
                    print(f"Valid labels (0 or 1): {(labels == 0).sum() + (labels == 1).sum()}")
                    print(f"No-data labels (255): {(labels == 255).sum()}")
            elif set(unique_labels).issubset({0, 1}):
                # Labels are already binary
                labels_valid = (labels == 0) | (labels == 1)
                if debug:
                    print("Using binary labels (0,1)")
            else:
                # Try to interpret other values
                # Assume any non-zero value is positive class
                labels_valid = (labels >= 0) & (labels <= 1)
                if debug:
                    print(f"Interpreting labels with values: {unique_labels}")
            
            # Ensure labels are properly typed for indexing
            labels = labels.astype(np.float32)  # Convert to float for comparison operations
            
            if debug:
                print(f"Data valid pixels: {data_valid.sum()}")
                print(f"Labels valid pixels: {labels_valid.sum()}")
            
            valid_mask = data_valid & labels_valid
            
            # Ensure valid_mask is boolean and properly shaped
            valid_mask = valid_mask.astype(bool)
            
            if debug:
                print(f"Final valid pixels: {valid_mask.sum()}")
                print(f"valid_mask shape: {valid_mask.shape}, dtype: {valid_mask.dtype}")
            
            if valid_mask.sum() == 0:
                if debug:
                    print("❌ No valid pixels found!")
                return None, None
            
            # Extract valid pixels
            try:
                features = data[:, valid_mask].T  # (n_pixels, 67)
                labels_valid_pixels = labels[valid_mask].astype(np.int32)  # (n_pixels,) - ensure integer type
            except Exception as e:
                if debug:
                    print(f"Error extracting valid pixels: {e}")
                    print(f"valid_mask shape: {valid_mask.shape}, dtype: {valid_mask.dtype}")
                    print(f"data shape: {data.shape}, labels shape: {labels.shape}")
                return None, None
            
            if debug:
                print(f"Extracted {len(features)} valid pixels")
                print(f"Class distribution: {np.bincount(labels_valid_pixels.astype(int))}")
            
            # Subsample if too many pixels
            if len(features) > max_pixels_per_tile:
                try:
                    # Stratified sampling to preserve class balance
                    pos_indices = np.where(labels_valid_pixels == 1)[0]
                    neg_indices = np.where(labels_valid_pixels == 0)[0]
                    
                    # Sample proportionally
                    n_pos = min(len(pos_indices), max_pixels_per_tile // 10)  # Max 10% positive
                    n_neg = min(len(neg_indices), max_pixels_per_tile - n_pos)
                    
                    if n_pos > 0:
                        pos_sample = np.random.choice(pos_indices, n_pos, replace=False)
                    else:
                        pos_sample = []
                    
                    neg_sample = np.random.choice(neg_indices, n_neg, replace=False)
                    
                    sample_indices = np.concatenate([pos_sample, neg_sample])
                    features = features[sample_indices]
                    labels_valid_pixels = labels_valid_pixels[sample_indices]
                    
                    if debug:
                        print(f"After subsampling: {len(features)} pixels")
                        
                except Exception as e:
                    if debug:
                        print(f"Error during subsampling: {e}")
                    # If subsampling fails, just use all the data
                    pass
                
                if debug:
                    print(f"After subsampling: {len(features)} pixels")
                
                if debug:
                    print(f"After subsampling: {len(features)} pixels")
            
            # Save to cache
            if use_cache:
                cache_key = get_cache_key(data_file, labels_file, max_pixels_per_tile)
                save_to_cache(cache_key, features, labels_valid_pixels)
            
            return features, labels_valid_pixels
            
    except Exception as e:
        print(f"❌ Error processing {Path(data_file).name}: {e}")
        return None, None


def load_data_optimized(data_dir, labels_file, max_tiles=None, max_pixels_per_tile=50000, debug=False, use_cache=True):
    """Load data with optimized sampling strategy."""
    print("🔄 Loading data with optimized sampling...")
    
    # Find overlapping tiles
    overlapping_files = find_overlapping_tiles(data_dir, labels_file, debug=debug)
    
    if not overlapping_files:
        print("❌ No overlapping tiles found!")
        return None, None
    
    if max_tiles:
        overlapping_files = overlapping_files[:max_tiles]
        print(f"🔧 Using only first {max_tiles} tiles for testing")
    
    features_list = []
    labels_list = []
    total_positive = 0
    
    # Process each tile
    for i, data_file in enumerate(tqdm(overlapping_files, desc="Processing tiles")):
        features, labels_data = extract_valid_pixels(data_file, labels_file, max_pixels_per_tile, debug=debug, use_cache=use_cache)
        
        if features is not None:
            features_list.append(features)
            labels_list.append(labels_data)
            
            n_positive = labels_data.sum()
            total_positive += n_positive
            
            # Only print if debug mode is enabled
            if debug:
                print(f"Tile {i+1}: {len(features)} pixels, {n_positive} positive ({n_positive/len(features)*100:.2f}%)")
        else:
            if debug:
                print(f"Tile {i+1}: No valid data")
        
        # Memory management
        if (i + 1) % 10 == 0:
            gc.collect()
    
    if not features_list:
        print("❌ No valid data found!")
        return None, None
    
    # Combine all data
    print("🔗 Combining data...")
    X = np.vstack(features_list)
    y = np.hstack(labels_list)
    
    print(f"✅ Final dataset:")
    print(f"   Total samples: {X.shape[0]:,} pixels")
    print(f"   Features: {X.shape[1]} bands")
    print(f"   Positive samples: {y.sum():,} ({y.sum()/len(y)*100:.3f}%)")
    
    return X, y


class OptimizedLightGBMTrainer:
    def __init__(self, num_leaves=31, learning_rate=0.05, num_iterations=1000):
        self.model = None
        # Feature names will be set dynamically based on actual data
        self.feature_names = None
        
        # Calculate class weights based on your rare positive class
        # For wet woodland detection, we want to focus more on positive class
        pos_weight = 10.0  # Slightly lower weight to allow more exploration
        
        self.params = {
            'objective': 'binary',
            'metric': ['binary_f1', 'auc', 'binary_logloss'],  # Put F1 first to track it during training
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': 0.9,  # Use more features
            'bagging_fraction': 0.9,  # Use more data
            'bagging_freq': 3,  # More frequent bagging
            'min_data_in_leaf': 30,  # Allow smaller leaves for more exploration
            'num_iterations': num_iterations,
            'verbose': -1,
            'scale_pos_weight': pos_weight,  # Handle class imbalance
            'early_stopping_rounds': 500,  # More patience for early stopping
            'seed': 42
        }
    
    def train(self, data_dir, labels_file, test_size=0.2, max_tiles=None, max_pixels_per_tile=50000, debug=False, use_cache=True):
        """Train LightGBM model with optimized data loading."""
        print("🚀 Training optimized LightGBM model...")
        
        # Load data
        X, y = load_data_optimized(data_dir, labels_file, max_tiles, max_pixels_per_tile, debug=debug, use_cache=use_cache)
        
        if X is None:
            print("❌ Failed to load data")
            return None
        
        # Split with stratification to preserve class balance
        print(f"\n📊 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training: {len(X_train):,} samples ({y_train.sum():,} positive, {y_train.sum()/len(y_train)*100:.3f}%)")
        print(f"Test: {len(X_test):,} samples ({y_test.sum():,} positive, {y_test.sum()/len(y_test)*100:.3f}%)")
        
        # Set feature names dynamically based on actual data
        if self.feature_names is None:
            n_features = X_train.shape[1]
            if n_features == 67:
                # Full dataset: 64 embeddings + 3 LiDAR
                self.feature_names = [f'embed_{i+1}' for i in range(64)] + ['chm', 'dtm', 'dsm']
            elif n_features == 64:
                # Embeddings only
                self.feature_names = [f'embed_{i+1}' for i in range(64)]
            elif n_features == 3:
                # LiDAR only
                self.feature_names = ['chm', 'dtm', 'dsm']
            else:
                # Generic naming for other combinations
                self.feature_names = [f'feature_{i+1}' for i in range(n_features)]
            
            print(f"📊 Detected {n_features} features: {self.feature_names[:5]}{'...' if n_features > 5 else ''}")
        
        # Create LightGBM datasets with feature names
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train without early stopping to allow full exploration
        print("\n🏃 Training LightGBM (full exploration mode)...")
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'eval'],
            callbacks=[lgb.log_evaluation(period=50)]
        )
        
        # Evaluate
        self.evaluate(X_test, y_test)
        
        return self.model
    
    def evaluate(self, X, y):
        """Comprehensive model evaluation."""
        if self.model is None:
            print("❌ No trained model available")
            return
        
        # Predictions with probability
        y_pred_proba = self.model.predict(X)
        
        # Find optimal threshold using multiple metrics
        thresholds = np.arange(0.05, 0.95, 0.02)  # More granular thresholds
        f1_scores = []
        balanced_acc_scores = []
        precision_scores = []
        recall_scores = []
        
        for thresh in thresholds:
            y_pred_thresh = (y_pred_proba > thresh).astype(int)
            f1_scores.append(f1_score(y, y_pred_thresh))
            balanced_acc_scores.append(balanced_accuracy_score(y, y_pred_thresh))
            precision_scores.append(precision_score(y, y_pred_thresh))
            recall_scores.append(recall_score(y, y_pred_thresh))
        
        # Find optimal threshold for F1 score
        optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
        y_pred_optimal_f1 = (y_pred_proba > optimal_threshold_f1).astype(int)
        
        # Find optimal threshold for balanced accuracy
        optimal_threshold_ba = thresholds[np.argmax(balanced_acc_scores)]
        y_pred_optimal_ba = (y_pred_proba > optimal_threshold_ba).astype(int)
        
        print("\n📊 LightGBM Model Performance:")
        print("=" * 50)
        
        # F1-optimized results
        print(f"🎯 F1-Optimized Results:")
        print(f"   Optimal threshold: {optimal_threshold_f1:.3f}")
        print(f"   Best F1 score: {max(f1_scores):.4f}")
        print(f"   Precision: {precision_scores[np.argmax(f1_scores)]:.4f}")
        print(f"   Recall: {recall_scores[np.argmax(f1_scores)]:.4f}")
        print(f"   Balanced Accuracy: {balanced_acc_scores[np.argmax(f1_scores)]:.4f}")
        
        # Balanced Accuracy-optimized results
        print(f"\n⚖️  Balanced Accuracy-Optimized Results:")
        print(f"   Optimal threshold: {optimal_threshold_ba:.3f}")
        print(f"   Best Balanced Accuracy: {max(balanced_acc_scores):.4f}")
        print(f"   F1 score: {f1_scores[np.argmax(balanced_acc_scores)]:.4f}")
        print(f"   Precision: {precision_scores[np.argmax(balanced_acc_scores)]:.4f}")
        print(f"   Recall: {recall_scores[np.argmax(balanced_acc_scores)]:.4f}")
        
        print(f"\n📈 Classification Report (F1-Optimized Threshold):")
        print(classification_report(y, y_pred_optimal_f1, target_names=['Other', 'Wet Woodland']))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred_optimal_f1)
        print(f"\nConfusion Matrix:")
        print(f"               Predicted")
        print(f"Actual    Other  Wet Woodland")
        print(f"Other     {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"Wet Wood  {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = list(zip(self.feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔍 Top 15 Most Important Features:")
        for i, (name, imp) in enumerate(feature_importance[:15]):
            print(f"{i+1:2d}. {name:12s}: {imp:8.1f}")
        
        # Feature group importance analysis
        if self.feature_names and len(self.feature_names) > 0:
            embed_importance = sum(imp for name, imp in feature_importance if name.startswith('embed'))
            lidar_importance = sum(imp for name, imp in feature_importance if name in ['chm', 'dtm', 'dsm'])
            other_importance = sum(imp for name, imp in feature_importance if not name.startswith('embed') and name not in ['chm', 'dtm', 'dsm'])
            total_importance = embed_importance + lidar_importance + other_importance
            
            if total_importance > 0:
                print(f"\n📈 Feature Group Importance:")
                if embed_importance > 0:
                    print(f"Embeddings: {embed_importance/total_importance*100:.1f}%")
                if lidar_importance > 0:
                    print(f"LiDAR:      {lidar_importance/total_importance*100:.1f}%")
                if other_importance > 0:
                    print(f"Other:      {other_importance/total_importance*100:.1f}%")
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model:
            # Create models directory if it doesn't exist
            model_path = Path(filepath)
            model_dir = model_path.parent
            model_dir.mkdir(exist_ok=True)
            
            self.model.save_model(filepath)
            print(f"✅ Model saved to {filepath}")


def get_cache_key(data_file, labels_file, max_pixels_per_tile):
    """Generate a cache key based on file paths and parameters."""
    # Create a hash of the file paths and parameters
    key_string = f"{data_file}_{labels_file}_{max_pixels_per_tile}"
    return hashlib.md5(key_string.encode()).hexdigest()


def load_from_cache(cache_key):
    """Load processed data from cache if available."""
    # Create cache directory if it doesn't exist
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    cache_file = cache_dir / f"cache_{cache_key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None


def save_to_cache(cache_key, features, labels):
    """Save processed data to cache."""
    # Create cache directory if it doesn't exist
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    cache_file = cache_dir / f"cache_{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump((features, labels), f)
    except Exception as e:
        print(f"Warning: Could not save to cache: {e}")


def cleanup_existing_files():
    """Move existing cache files and models to organized folders."""
    print("🧹 Organizing existing files...")
    
    # Create directories
    Path("cache").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Move cache files
    cache_files = list(Path(".").glob("cache_*.pkl"))
    if cache_files:
        print(f"Moving {len(cache_files)} cache files to cache/ folder...")
        for cache_file in cache_files:
            try:
                cache_file.rename(Path("cache") / cache_file.name)
            except Exception as e:
                print(f"Warning: Could not move {cache_file}: {e}")
    
    # Move model files
    model_files = list(Path(".").glob("*_model.txt")) + list(Path(".").glob("*_lgb.txt"))
    if model_files:
        print(f"Moving {len(model_files)} model files to models/ folder...")
        for model_file in model_files:
            try:
                model_file.rename(Path("models") / model_file.name)
            except Exception as e:
                print(f"Warning: Could not move {model_file}: {e}")
    
    print("✅ File organization complete!")


def main():
    parser = argparse.ArgumentParser(description="Optimized LightGBM Training")
    parser.add_argument("--data-dir", help="Directory with GEE feature tiles")
    parser.add_argument("--labels-file", help="Single labels.tiff file")
    parser.add_argument("--max-tiles", type=int, help="Max tiles to process (all if not specified)")
    parser.add_argument("--max-pixels-per-tile", type=int, default=50000, help="Max pixels per tile")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--num-leaves", type=int, default=31, help="Number of leaves")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--num-iterations", type=int, default=1000, help="Max iterations")
    parser.add_argument("--save-model", default="models/wet_woodland_lgb.txt", help="Model save path")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching (reprocess all tiles)")
    parser.add_argument("--cleanup", action="store_true", help="Organize existing cache and model files")
    
    args = parser.parse_args()
    
    # Clean up existing files if requested
    if args.cleanup:
        cleanup_existing_files()
        return
    
    # Check if required arguments are provided for training
    if not args.data_dir or not args.labels_file:
        print("❌ --data-dir and --labels-file are required for training")
        print("Use --cleanup to organize existing files without training")
        return
    
    print("🌲 Optimized LightGBM Training for Wet Woodland Detection")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Labels file: {args.labels_file}")
    print(f"Max tiles: {args.max_tiles if args.max_tiles else 'All'}")
    print(f"Max pixels per tile: {args.max_pixels_per_tile:,}")
    print(f"Debug mode: {args.debug}")
    
    # Verify files exist
    if not Path(args.labels_file).exists():
        print(f"❌ Labels file not found: {args.labels_file}")
        return
    
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Create trainer
    trainer = OptimizedLightGBMTrainer(
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations
    )
    
    # Train
    model = trainer.train(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        test_size=args.test_size,
        max_tiles=args.max_tiles,
        max_pixels_per_tile=args.max_pixels_per_tile,
        debug=args.debug,
        use_cache=not args.no_cache
    )
    
    if model:
        trainer.save_model(args.save_model)
        print(f"\n✅ Training complete! Model saved to {args.save_model}")
    else:
        print("\n❌ Training failed!")


if __name__ == "__main__":
    main()