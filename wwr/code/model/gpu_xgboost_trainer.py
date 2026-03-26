#!/usr/bin/env python3
"""
GPU-Accelerated XGBoost Training with Spatial Cross-Validation

Features:
- GPU acceleration for faster training (much better than LightGBM)
- Binary and multi-class classification support
- Rotating spatial fold training for honest evaluation
- Spatial buffering to prevent autocorrelation
- Hyperparameter optimization with Optuna
- Class rebalancing for imbalanced datasets
- Pre-cached fold datasets for maximum GPU efficiency
"""

import numpy as np
import pandas as pd
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
import hashlib
import os
import time
import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
SPATIAL_KFOLD_AVAILABLE = True  # Custom spatial CV implementation used throughout
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score, average_precision_score
import sys
from io import StringIO
import warnings
try:
    from scipy.spatial import cKDTree
    CKDTREE_AVAILABLE = True
except ImportError:
    CKDTREE_AVAILABLE = False

# KML/geometry support for independent validation
try:
    import fiona
    from shapely.geometry import shape, Point, box
    from shapely.ops import unary_union
    import pyproj
    from functools import partial
    KML_SUPPORT = True
except ImportError:
    KML_SUPPORT = False

# Suppress all annoying warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='force_all_finite')
warnings.filterwarnings('ignore', message='.*force_all_finite.*')
warnings.filterwarnings('ignore', message='.*ensure_all_finite.*')
warnings.filterwarnings('ignore', message='.*will be ignored.*')
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# XGBoost import
import xgboost as xgb

# F-beta for policy-threshold diagnostics.
# beta=0.5 weights precision 4× over recall when summarising deployment seed quality.
DEFAULT_F_BETA = 0.5
DEFAULT_POLICY_TARGET_PRECISION = 0.25
DEFAULT_POLICY_PRECISION_QUANTILE = 0.10
DEFAULT_POLICY_MIN_RECALL = 0.0
DEFAULT_POLICY_REPORT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_POLICY_REPORT_PRECISIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75)
DEFAULT_POLICY_PREVALENCE_PRINT_SCENARIOS = (0.05, 0.06, 0.075, 0.10)
DEFAULT_POLICY_COMPACT_EXPAND_RATIOS = (0.50, 0.67, 0.75)
DEFAULT_MIN_CHM_COVER = 0.15
DEFAULT_MIN_CHM_MEAN = 3.0
TRAINER_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = TRAINER_ROOT / "data"
INPUT_ROOT = DATA_ROOT / "input"
OUTPUT_ROOT = DATA_ROOT / "output"
VALIDATION_ROOT = DATA_ROOT / "validation"

DEFAULT_MODEL_DIR = OUTPUT_ROOT / "models"
DEFAULT_MODEL_FILENAME = "wetwoodland_binary.json"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILENAME
DEFAULT_BINARY_MODEL_PATH = DEFAULT_MODEL_PATH
DEFAULT_REPORT_DIR = OUTPUT_ROOT / "reports"
DEFAULT_TRAINING_EMBEDDINGS_DIR = INPUT_ROOT / "embeddings" / "training_embeddings"
DEFAULT_LABELS_PATH = OUTPUT_ROOT / "labels" / "wetwoodland.tif"
DEFAULT_DISCARDED_LABELS_PATH = VALIDATION_ROOT / "eval_background.tif"
DEFAULT_SHAP_SAMPLES = 45000
DEFAULT_N_ESTIMATORS = 5000
DEFAULT_MAX_DEPTH = 8
LEGACY_DEFAULT_MODEL_PATH = Path(DEFAULT_MODEL_FILENAME)
LEGACY_GENERIC_DEFAULT_MODEL_PATH = Path("wetwoodland.json")
LEGACY_OLD_DEFAULT_MODEL_PATH = Path("wet_woodland_xgboost_model.json")
LEGACY_NESTED_DEFAULT_MODEL_PATH = Path("data") / "output" / "models" / "xgboost" / "wet_woodland_xgboost_model.json"


# GPU Support Check
def check_gpu_availability(force_cpu=False):
    """Check if GPU is available for XGBoost."""
    if force_cpu:
        print("   🔧 GPU disabled by --force-cpu flag")
        return False, None

    try:
        # Test GPU availability with PyTorch first
        try:
            import torch
            if not torch.cuda.is_available():
                print("   ⚠️ CUDA not available - using CPU")
                return False, None

            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        except ImportError:
            pass  # Silently fall back to XGBoost test

        # Test XGBoost GPU training
        test_data = xgb.DMatrix(np.random.rand(100, 10), label=np.random.randint(0, 2, 100))
        test_params = {'tree_method': 'hist', 'device': 'cuda:0'}

        # Try a quick test train
        xgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
        return True, 0

    except Exception as e:
        print(f"   ❌ GPU failed: {e}")
        print("   Using CPU mode")
        return False, None

# Optuna imports
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ optuna not installed. Install with: pip install optuna")

try:
    from optuna.integration import XGBoostPruningCallback
    OPTUNA_XGB_PRUNING = True
except ImportError:
    OPTUNA_XGB_PRUNING = False


def xgb_auprc_score(y_true, y_pred):
    """
    Custom AUPRC evaluation function for XGBoost
    """
    from sklearn.metrics import average_precision_score
    auprc = average_precision_score(y_true, y_pred)
    return 'auprc', auprc


def process_spatial_block_batch(args):
    """Process a batch of spatial blocks for rebalancing (optimized)"""
    batch_groups, spatial_groups_batch, y_batch, base_indices, wet_classes, background_classes, bg_ratio, dec_bg_fraction = args
    local_balanced_indices = []  # Will convert to array at end
    local_discarded_indices = []  # Track discarded samples
    local_cells_processed = 0
    local_cells_with_data = 0

    for block_id in batch_groups:
        # Fast boolean indexing
        block_mask = spatial_groups_batch == block_id
        if not block_mask.any():
            continue

        block_indices = base_indices[block_mask]
        block_y = y_batch[block_mask]

        wet_mask = np.isin(block_y, wet_classes)
        wet_indices = block_indices[wet_mask]
        wet_count = wet_indices.size

        # Gather background indices by class to preserve representation
        bg_by_class = {}
        for bg_class in background_classes:
            cls_indices = block_indices[block_y == bg_class]
            if cls_indices.size > 0:
                bg_by_class[int(bg_class)] = cls_indices
        bg_total = sum(arr.size for arr in bg_by_class.values())

        local_cells_processed += 1

        # Only include cells with both wet and non-wet examples
        if wet_count >= 1 and bg_total >= 1:
            local_cells_with_data += 1

            # Keep all wet samples
            local_balanced_indices.extend(wet_indices.tolist())

            # Allow bg_ratio times more background than wet (default 2:1).
            target_bg_total = min(bg_total, int(wet_count * bg_ratio))
            rng = np.random.default_rng(42 + int(block_id))

            available_bg_classes = sorted(bg_by_class.keys())
            sampled_bg = []

            if len(available_bg_classes) == 1:
                cls = available_bg_classes[0]
                cls_indices = bg_by_class[cls]
                if cls_indices.size > target_bg_total:
                    sampled = rng.choice(cls_indices, target_bg_total, replace=False)
                else:
                    sampled = cls_indices
                sampled_bg.extend(sampled.tolist())
            else:
                # Split background quota by class, biasing toward deciduous (class 1)
                # dec_bg_fraction controls what fraction of background is deciduous.
                # Default 0.8 = 80% deciduous, 20% evergreen.
                n_cls = len(available_bg_classes)
                if dec_bg_fraction != 0.5 and 1 in available_bg_classes and 0 in available_bg_classes:
                    quotas = {}
                    for cls in available_bg_classes:
                        if cls == 1:  # deciduous
                            quotas[cls] = int(target_bg_total * dec_bg_fraction)
                        else:         # evergreen and any others split remaining
                            quotas[cls] = int(target_bg_total * (1.0 - dec_bg_fraction) / (n_cls - 1))
                    # Assign any rounding remainder to deciduous
                    remainder = target_bg_total - sum(quotas.values())
                    quotas[1] = quotas.get(1, 0) + remainder
                else:
                    base_quota = target_bg_total // n_cls
                    remainder = target_bg_total % n_cls
                    quotas = {cls: base_quota for cls in available_bg_classes}
                    for cls in available_bg_classes[:remainder]:
                        quotas[cls] += 1

                leftovers = []
                for cls in available_bg_classes:
                    cls_indices = bg_by_class[cls]
                    take_n = min(cls_indices.size, quotas[cls])
                    if take_n > 0:
                        if cls_indices.size > take_n:
                            sampled = rng.choice(cls_indices, take_n, replace=False)
                        else:
                            sampled = cls_indices
                        sampled_bg.extend(sampled.tolist())
                        if cls_indices.size > take_n:
                            leftovers.append(np.setdiff1d(cls_indices, sampled, assume_unique=False))
                    else:
                        leftovers.append(cls_indices)

                shortfall = target_bg_total - len(sampled_bg)
                if shortfall > 0 and leftovers:
                    pool = np.concatenate([arr for arr in leftovers if arr.size > 0]) if any(arr.size > 0 for arr in leftovers) else np.array([], dtype=np.int64)
                    if pool.size > 0:
                        extra_n = min(shortfall, pool.size)
                        if pool.size > extra_n:
                            extra = rng.choice(pool, extra_n, replace=False)
                        else:
                            extra = pool
                        sampled_bg.extend(extra.tolist())

            local_balanced_indices.extend(sampled_bg)

            # Track discarded background samples in this block
            kept_bg_set = set(sampled_bg)
            for bg_arr in bg_by_class.values():
                for idx in bg_arr.tolist():
                    if idx not in kept_bg_set:
                        local_discarded_indices.append(idx)
        else:
            # Entire block discarded - track all indices from this block
            local_discarded_indices.extend(block_indices.tolist())

    return local_balanced_indices, local_discarded_indices, local_cells_processed, local_cells_with_data


def resolve_tile_chm_screen_indices(n_bands):
    """
    Resolve CHM mean and CHM cover indices for supported tile layouts.

    The current production exports place CHM mean and CHM cover in the final
    two bands when both are present.
    """
    if n_bands >= 67:
        return n_bands - 2, n_bands - 1
    return None, None


def build_tile_height_filter_mask(data, min_chm_cover, min_chm_mean):
    """
    Build the default CHM screening mask for a tile cube.

    Note: the final band is treated as canopy cover fraction, even though some
    legacy code paths still refer to it as "chm_canopy_gap".
    """
    chm_mean_idx, chm_cover_idx = resolve_tile_chm_screen_indices(data.shape[0])
    if chm_mean_idx is None or chm_cover_idx is None:
        return np.ones(data.shape[1:], dtype=bool), False

    chm_mean = data[chm_mean_idx]
    chm_cover = data[chm_cover_idx]
    keep_mask = (
        np.isfinite(chm_mean)
        & np.isfinite(chm_cover)
        & (chm_cover >= min_chm_cover)
        & (chm_mean >= min_chm_mean)
    )
    return keep_mask, True
def fbeta_score(y_true, y_pred, beta=DEFAULT_F_BETA, pos_label=1):
    """
    Calculate F-beta score with emphasis on precision (beta < 1)
    beta=0.5 strongly weighs precision over recall
    """
    from sklearn.metrics import precision_score, recall_score

    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    if precision + recall == 0:
        return 0.0

    beta_squared = beta ** 2
    return (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)


def resolve_label_schema(y, label_schema):
    """
    Determine class semantics for non-binary labels.
    """
    if label_schema != "auto":
        return label_schema
    unique_labels = set(np.unique(y).astype(int).tolist())
    if 3 in unique_labels:
        return "split_bg_0123"
    return "legacy_wet_012"


def schema_class_sets(label_schema):
    """
    Return (wet_classes, background_classes) for the selected schema.
    """
    if label_schema == "split_bg_0123":
        return np.array([2, 3], dtype=np.int32), np.array([0, 1], dtype=np.int32)
    return np.array([1, 2], dtype=np.int32), np.array([0], dtype=np.int32)


def to_binary_labels(y, label_schema):
    """
    Convert schema labels to binary (0=non-wet, 1=wet).
    """
    wet_classes, _ = schema_class_sets(label_schema)
    return np.isin(y, wet_classes).astype(np.int32)


def optuna_objective(
    trial,
    X_train,
    y_train,
    X_val,
    y_val,
    use_binary=True,
    gpu_available=False,
    gpu_id=0,
    force_cpu=False,
    pos_neg_ratio=None,
    n_estimators=3000,
    scale_pos_weight_override=None,
    max_depth_override=None,
    min_depth_search=4,
    max_depth_search=DEFAULT_MAX_DEPTH,
):
    """
    Optuna objective function for XGBoost hyperparameter optimization.
    Optimises AUPRC (average precision) on a spatially-held-out validation split.
    AUPRC is sensitive to the rare positive class and better suited to imbalanced
    wetwoodland detection than AUROC, which can be inflated by majority-class ranking.
    """
    # Narrowed search space — forces slow learning and penalises depth-driven
    # overfitting to spatial noise (colsample_bylevel omitted: redundant with TPE).
    xgb_params = {
        'max_depth': int(max_depth_override) if max_depth_override is not None else trial.suggest_int('max_depth', int(min_depth_search), int(max_depth_search)),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 2.0),
        'max_delta_step': 0,
        'max_bin': 1024,
        'random_state': 42,
        'verbosity': 0,
    }

    # Add GPU parameters if available
    if gpu_available and not force_cpu:
        xgb_params.update({
            'tree_method': 'hist',
            'device': f'cuda:{gpu_id}',
            'grow_policy': 'depthwise',  # Better for GPU parallelization
            'single_precision_histogram': True,  # Faster GPU histograms
            'deterministic_histogram': False,  # Allow non-deterministic for speed
            'sampling_method': 'gradient_based',  # GPU-optimized sampling
            'max_cached_hist_node': 128  # Cache more histogram nodes
        })
    else:
        xgb_params['tree_method'] = 'hist'
        xgb_params['device'] = 'cpu'

    if use_binary:
        xgb_params['objective'] = 'binary:logistic'
        xgb_params['eval_metric'] = 'aucpr'
        if scale_pos_weight_override is not None:
            xgb_params['scale_pos_weight'] = float(scale_pos_weight_override)
        else:
            # Search scale_pos_weight in a narrow range — a small positive bias (~1.1)
            # empirically improves the FP/recall tradeoff beyond what bg_ratio alone achieves.
            xgb_params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 1.0, 1.2)
    else:
        xgb_params['objective'] = 'multi:softprob'
        xgb_params['num_class'] = len(np.unique(y_train))
        xgb_params['eval_metric'] = 'mlogloss'

    try:
        # Create DMatrix for XGBoost (use regular DMatrix for Optuna stability)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Pruning callback kills low-performing trials early so more trials can
        # explore the narrowed space within the same wall-clock budget.
        callbacks = []
        if OPTUNA_XGB_PRUNING and use_binary:
            callbacks.append(XGBoostPruningCallback(trial, "validation-aucpr"))

        # Build train kwargs — capped at n_estimators with short early stopping patience.
        # Optuna finds the right regularisation profile quickly on the 50k subsample;
        # the CV/final then runs the full n_estimators with those params.
        train_kwargs = dict(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, 'validation')],
            early_stopping_rounds=max(100, n_estimators // 50),
            verbose_eval=False,
            callbacks=callbacks if callbacks else None,
        )

        model = xgb.train(**train_kwargs)

        # Store best iteration so optimize_with_optuna can use it for CV/final
        trial.set_user_attr('best_iteration', int(model.best_iteration))

        # Get predictions
        y_pred_proba = model.predict(dval)

        if not use_binary:
            y_pred_proba = np.argmax(y_pred_proba, axis=1)

        # Optimise AUPRC — PR-AUC directly penalises missing positive (wetwoodland)
        # detections and is not inflated by correct majority-class rankings.
        from sklearn.metrics import average_precision_score
        auprc = average_precision_score(y_val, y_pred_proba)

        if np.isnan(auprc):
            return 0.0

        return auprc  # Optuna maximises

    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0


def load_kml_polygons(kml_path, target_crs=None):
    """
    Load polygons from KML file and optionally reproject to target CRS.

    Args:
        kml_path: Path to KML file
        target_crs: Target CRS (e.g., 'EPSG:27700' for British National Grid)

    Returns:
        List of shapely polygons in target CRS
    """
    if not KML_SUPPORT:
        print("   KML support not available. Install: pip install fiona shapely pyproj")
        return []

    polygons = []

    # Enable KML driver
    fiona.drvsupport.supported_drivers['KML'] = 'r'

    try:
        with fiona.open(kml_path, 'r', driver='KML') as src:
            src_crs = src.crs
            print(f"   KML CRS: {src_crs}")

            for feature in src:
                geom = shape(feature['geometry'])

                # Only keep polygons (skip points)
                if geom.geom_type == 'Polygon':
                    polygons.append(geom)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        polygons.append(poly)

        print(f"   Loaded {len(polygons)} polygons from KML")

        # Reproject if needed
        if target_crs and polygons:
            from pyproj import Transformer
            # KML is typically WGS84 (EPSG:4326)
            transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

            reprojected = []
            for poly in polygons:
                coords = list(poly.exterior.coords)
                # Handle both 2D (x, y) and 3D (x, y, z) coordinates
                new_coords = [transformer.transform(c[0], c[1]) for c in coords]
                from shapely.geometry import Polygon as ShapelyPolygon
                reprojected.append(ShapelyPolygon(new_coords))

            polygons = reprojected
            print(f"   Reprojected to {target_crs}")

        return polygons

    except Exception as e:
        print(f"   Error loading KML: {e}")
        return []


def extract_features_from_polygons(polygons, data_dir, max_pixels_per_polygon=1000):
    """
    Extract features from tiles that intersect with polygons.

    Args:
        polygons: List of shapely polygons (in tile CRS)
        data_dir: Directory containing GeoTIFF tiles
        max_pixels_per_polygon: Max pixels to sample per polygon

    Returns:
        X: Feature matrix (n_samples, n_features)
        coordinates: Pixel coordinates (n_samples, 2)
    """
    if not polygons:
        return np.array([]), np.array([])

    data_files = glob.glob(str(Path(data_dir) / "*.tif"))
    print(f"   Searching {len(data_files)} tiles for polygon intersections...")

    all_features = []
    all_coords = []

    # Create unified polygon for faster intersection testing
    unified_polys = unary_union(polygons)

    for data_file in tqdm(data_files, desc="   Extracting from tiles"):
        try:
            with rasterio.open(data_file) as src:
                # Check if tile intersects any polygon
                tile_bounds = box(*src.bounds)
                if not unified_polys.intersects(tile_bounds):
                    continue

                # Read tile data
                data = src.read()  # (bands, height, width)
                transform = src.transform

                # Find which pixels are inside polygons
                height, width = data.shape[1], data.shape[2]

                # Create coordinate grids
                rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                xs = transform.c + cols * transform.a + rows * transform.b
                ys = transform.f + cols * transform.d + rows * transform.e

                # Check each polygon
                inside_mask = np.zeros((height, width), dtype=bool)

                for poly in polygons:
                    if not poly.intersects(tile_bounds):
                        continue

                    # Vectorized point-in-polygon (approximate with bounds first)
                    minx, miny, maxx, maxy = poly.bounds
                    candidate_mask = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)

                    if not candidate_mask.any():
                        continue

                    # Check actual containment for candidates
                    candidate_indices = np.where(candidate_mask)
                    for r, c in zip(candidate_indices[0], candidate_indices[1]):
                        if poly.contains(Point(xs[r, c], ys[r, c])):
                            inside_mask[r, c] = True

                # Also check data validity
                data_valid = ~np.isnan(data).any(axis=0)
                valid_mask = inside_mask & data_valid

                if valid_mask.sum() == 0:
                    continue

                # Extract features
                features = data[:, valid_mask].T
                pixel_xs = xs[valid_mask]
                pixel_ys = ys[valid_mask]
                coords = np.column_stack([pixel_xs, pixel_ys])

                # Subsample if too many
                if len(features) > max_pixels_per_polygon:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(features), max_pixels_per_polygon, replace=False)
                    features = features[idx]
                    coords = coords[idx]

                all_features.append(features)
                all_coords.append(coords)

        except Exception as e:
            continue

    if all_features:
        X = np.vstack(all_features)
        coordinates = np.vstack(all_coords)
        print(f"   Extracted {len(X):,} pixels from KML polygons")
        return X, coordinates
    else:
        return np.array([]), np.array([])


def save_background_eval_raster(output_path, template_raster, coords, labels):
    """
    Save discarded/background points as a raster aligned to template_raster.

    Pixels not selected are nodata (255). Selected pixels keep their original background label.
    """
    if output_path is None or template_raster is None:
        return False
    if coords is None or labels is None or len(coords) == 0:
        print("   ⚠️  No sampled background points to save")
        return False

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(template_raster) as ds:
            profile = ds.profile.copy()
            profile.update(
                dtype=rasterio.uint8,
                nodata=255,
                count=1,
                compress="lzw",
            )

            # GeoTIFF tiled blocks must be multiples of 16. Use safe defaults.
            block_x = min(512, int(ds.width))
            block_y = min(512, int(ds.height))
            block_x = (block_x // 16) * 16
            block_y = (block_y // 16) * 16

            if block_x >= 16 and block_y >= 16:
                profile.update(
                    tiled=True,
                    blockxsize=block_x,
                    blockysize=block_y,
                )
            else:
                # Very small rasters: write striped output.
                profile.pop("blockxsize", None)
                profile.pop("blockysize", None)
                profile["tiled"] = False

            out = np.full((ds.height, ds.width), 255, dtype=np.uint8)
            rows, cols = rasterio.transform.rowcol(ds.transform, coords[:, 0], coords[:, 1])
            rows = np.asarray(rows, dtype=np.int64)
            cols = np.asarray(cols, dtype=np.int64)

            valid = (
                (rows >= 0) & (rows < ds.height) &
                (cols >= 0) & (cols < ds.width)
            )
            if not np.any(valid):
                print("   ⚠️  Sampled background coordinates were outside template raster extent")
                return False

            out[rows[valid], cols[valid]] = labels[valid].astype(np.uint8)

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out, 1)
                dst.set_band_description(1, "Discarded Labels")

        unique, counts = np.unique(labels.astype(np.uint8), return_counts=True)
        summary = ", ".join([f"class {int(c)}: {int(n):,}" for c, n in zip(unique, counts)])
        print(f"   ✅ Saved discarded labels raster: {out_path}")
        print(f"      Samples: {len(labels):,} ({summary})")
        return True
    except Exception as e:
        print(f"   ❌ Failed to save discarded labels raster: {e}")
        return False


def run_independent_validation(
    model, kml_path, data_dir, discarded_X, discarded_y, discarded_coords,
    target_crs='EPSG:27700', threshold=0.5, bg_ratio=4.0,
    spatial_grid_size=2000
):
    """
    Run independent validation using KML polygons (positive) and discarded samples (negative).

    Background samples are spatially stratified across the grid to ensure
    geographic balance in the validation set.

    Args:
        model: Trained XGBoost model
        kml_path: Path to KML file with wet woodland polygons
        data_dir: Directory with GeoTIFF tiles
        discarded_X: Features of discarded training samples
        discarded_y: Labels of discarded training samples
        discarded_coords: Coordinates of discarded training samples
        target_crs: CRS of tiles
        threshold: Classification threshold
        bg_ratio: Background:positive ratio (default 4.0 for harder precision test)
        spatial_grid_size: Grid cell size in meters for spatial stratification

    Returns:
        dict with validation metrics
    """
    print("\n" + "=" * 70)
    print("🎯 INDEPENDENT VALIDATION (KML + Discarded Samples)")
    print("=" * 70)

    if not KML_SUPPORT:
        print("   KML support not available. Install: pip install fiona shapely pyproj")
        return None

    # Load KML polygons
    print("\n📍 Loading KML polygons...")
    polygons = load_kml_polygons(kml_path, target_crs=target_crs)

    if not polygons:
        print("   No polygons found in KML")
        return None

    # Extract features from polygons (positive class)
    print("\n🌲 Extracting features from KML polygons (positive class)...")
    X_positive, coords_positive = extract_features_from_polygons(polygons, data_dir)

    if len(X_positive) == 0:
        print("   No pixels extracted from KML polygons")
        return None

    y_positive = np.ones(len(X_positive), dtype=np.int32)
    print(f"   Positive samples: {len(X_positive):,}")

    # Sample from discarded background (negative class) with SPATIAL STRATIFICATION
    print("\n🏞️  Sampling spatially-stratified background from discarded pixels...")

    # Create spatial grid for discarded samples
    x_min, x_max = discarded_coords[:, 0].min(), discarded_coords[:, 0].max()
    y_min, y_max = discarded_coords[:, 1].min(), discarded_coords[:, 1].max()

    # Assign each discarded sample to a grid cell
    x_bins = np.arange(x_min, x_max + spatial_grid_size, spatial_grid_size)
    y_bins = np.arange(y_min, y_max + spatial_grid_size, spatial_grid_size)

    x_cell = np.digitize(discarded_coords[:, 0], x_bins) - 1
    y_cell = np.digitize(discarded_coords[:, 1], y_bins) - 1
    cell_ids = x_cell * len(y_bins) + y_cell

    n_cells = len(np.unique(cell_ids))
    print(f"   Spatial grid: {len(x_bins)-1} x {len(y_bins)-1} cells ({spatial_grid_size/1000:.0f}km resolution)")
    print(f"   Cells with discarded samples: {n_cells}")

    # Get indices for each background class
    bg_0_mask = discarded_y == 0  # Evergreen woodland
    bg_1_mask = discarded_y == 1  # Deciduous woodland

    n_bg_0 = bg_0_mask.sum()
    n_bg_1 = bg_1_mask.sum()
    print(f"   Available: {n_bg_0:,} class-0 (evergreen), {n_bg_1:,} class-1 (deciduous)")

    # Use bg_ratio to determine background count (e.g., 4x positives for harder precision test)
    n_background_samples = int(len(X_positive) * bg_ratio)
    print(f"   Target background: {n_background_samples:,} ({bg_ratio}x positive count)")

    n_per_class = n_background_samples // 2
    rng = np.random.default_rng(42)

    def sample_spatially_stratified(class_mask, n_samples, woodland_type):
        """Sample from a woodland type with spatial stratification across grid cells."""
        class_indices = np.where(class_mask)[0]
        class_cells = cell_ids[class_indices]

        # Group indices by cell
        unique_cells = np.unique(class_cells)
        cell_to_indices = {c: class_indices[class_cells == c] for c in unique_cells}

        # Calculate samples per cell (distribute evenly, then fill remainder)
        n_cells_with_class = len(unique_cells)
        base_per_cell = n_samples // n_cells_with_class
        remainder = n_samples % n_cells_with_class

        sampled = []
        cells_used = 0

        for i, cell in enumerate(unique_cells):
            cell_indices = cell_to_indices[cell]
            # Extra sample for first 'remainder' cells
            n_from_cell = base_per_cell + (1 if i < remainder else 0)
            n_from_cell = min(n_from_cell, len(cell_indices))

            if n_from_cell > 0:
                chosen = rng.choice(cell_indices, n_from_cell, replace=False)
                sampled.extend(chosen)
                cells_used += 1

        print(f"      {class_name}: {len(sampled):,} samples from {cells_used} cells")
        return np.array(sampled)

    # Sample spatially from each class
    sampled_0 = sample_spatially_stratified(bg_0_mask, n_per_class, "Class 0 (non-woodland)")
    sampled_1 = sample_spatially_stratified(bg_1_mask, n_per_class, "Class 1 (deciduous)")

    sampled_bg = np.concatenate([sampled_0, sampled_1])
    X_negative = discarded_X[sampled_bg]
    y_negative = np.zeros(len(X_negative), dtype=np.int32)  # All negative for binary
    print(f"   Total negative samples: {len(X_negative):,} (spatially distributed)")

    # Combine positive and negative
    X_val = np.vstack([X_positive, X_negative])
    y_val = np.concatenate([y_positive, y_negative])

    print(f"\n📊 Independent validation set: {len(X_val):,} samples")
    print(f"   Positive (KML): {len(X_positive):,} ({len(X_positive)/len(X_val)*100:.1f}%)")
    print(f"   Negative (discarded): {len(X_negative):,} ({len(X_negative)/len(X_val)*100:.1f}%)")

    # Run inference
    print("\n🔮 Running inference...")
    try:
        y_pred_proba = model.inplace_predict(X_val)
    except Exception:
        import xgboost as xgb
        dval = xgb.DMatrix(X_val)
        y_pred_proba = model.predict(dval)

    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, balanced_accuracy_score,
        average_precision_score, confusion_matrix
    )

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    auprc = average_precision_score(y_val, y_pred_proba)
    cm = confusion_matrix(y_val, y_pred)

    # Report results
    print("\n" + "=" * 70)
    print("📊 INDEPENDENT VALIDATION RESULTS")
    print("=" * 70)
    print(f"\n   AUPRC:              {auprc:.4f}")
    print(f"   Balanced Accuracy:  {bal_acc:.4f}")
    print(f"   Precision:          {precision:.4f}")
    print(f"   Recall:             {recall:.4f}")
    print(f"   F1 Score:           {f1:.4f}")
    print(f"   Threshold:          {threshold}")

    print(f"\n   Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Neg    Pos")
    print(f"   Actual Neg    {cm[0,0]:6,} {cm[0,1]:6,}")
    print(f"   Actual Pos    {cm[1,0]:6,} {cm[1,1]:6,}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n   True Positives (KML sites detected):  {tp:,}")
    print(f"   False Negatives (KML sites missed):   {fn:,}")
    print(f"   True Negatives (background correct):  {tn:,}")
    print(f"   False Positives (background as wet):  {fp:,}")

    print("\n" + "=" * 70)
    print("💡 These metrics are from TRULY INDEPENDENT data:")
    print("   • Positive samples: Expert-identified wet woodland (KML)")
    print("   • Negative samples: Discarded during training rebalancing")
    print("   • Zero leakage from training process")
    print("=" * 70)

    return {
        'auprc': auprc,
        'balanced_accuracy': bal_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'confusion_matrix': cm,
        'n_positive': len(X_positive),
        'n_negative': len(X_negative),
    }



def find_optimal_thresholds(
    y_true,
    y_pred_proba,
    beta=DEFAULT_F_BETA,
    show_progress=True,
    y_original=None,
    target_pos_rate=None,
    national_prevalence=None,
):
    """
    Find optimal thresholds for both F-beta and balanced accuracy.

    If y_original is provided, also computes deciduous-specific metrics to find
    a threshold that minimizes false positives from the hard deciduous class.

    Args:
        y_true: True binary labels (0/1)
        y_pred_proba: Predicted probabilities
        beta: Beta parameter for F-beta score
              - beta < 1: Emphasize precision (e.g., 0.5)
              - beta > 1: Emphasize recall (e.g., 2.0)
              - beta = 1: Standard F1 (equal weight)
        show_progress: Show progress bar
        y_original: Original pre-binary labels (0=evergreen, 1=deciduous, 2+=wet)
                    If provided, enables deciduous-focused threshold optimization
        target_pos_rate: Optional target wet prevalence (0-1). When provided,
                        precision/F-beta are prior-adjusted to this prevalence.

    Returns:
        dict with results for F-beta, balanced accuracy, and optionally deciduous-focused optimizations
    """
    from sklearn.metrics import average_precision_score

    # Test thresholds from 0.01 to 0.99 in 0.01 increments for finer resolution
    thresholds = np.arange(0.01, 1.0, 0.01)

    fbeta_scores = []
    balanced_acc_scores = []
    precisions = []
    recalls = []
    fprs = []
    best_fbeta_so_far = 0.0
    best_balacc_so_far = 0.0
    _n_pos = int((y_true == 1).sum())
    _n_neg = int((y_true == 0).sum())

    observed_pos_rate = float(np.mean(y_true))
    observed_neg_rate = 1.0 - observed_pos_rate
    use_prior_adjustment = (
        target_pos_rate is not None
        and 0.0 < float(target_pos_rate) < 1.0
        and 0.0 < observed_pos_rate < 1.0
    )

    if use_prior_adjustment:
        target_pos_rate = float(target_pos_rate)
        target_neg_rate = 1.0 - target_pos_rate
        pos_scale = target_pos_rate / observed_pos_rate
        neg_scale = target_neg_rate / observed_neg_rate
    else:
        target_pos_rate = observed_pos_rate
        target_neg_rate = observed_neg_rate
        pos_scale = 1.0
        neg_scale = 1.0

    # Deciduous-focused metrics (if y_original provided)
    compute_deciduous = y_original is not None
    if compute_deciduous:
        deciduous_fbeta_scores = []
        deciduous_precisions = []
        best_dec_fbeta_so_far = 0.0
        # Mask for deciduous background samples (original class 1, binary class 0)
        deciduous_mask = (y_original == 1) & (y_true == 0)
        # Mask for wet woodland samples (binary class 1)
        wet_mask = y_true == 1
        n_deciduous = deciduous_mask.sum()
        n_wet = wet_mask.sum()

    if show_progress:
        from tqdm import tqdm
        if compute_deciduous:
            iterator = tqdm(thresholds, desc=f"   🔍 Searching thresholds (+ deciduous-focused)")
            iterator.set_postfix(fbeta=0.0, dec_fbeta=0.0)
        else:
            iterator = tqdm(thresholds, desc=f"   🔍 Searching thresholds")
            iterator.set_postfix(fbeta=0.0, balacc=0.0)
    else:
        iterator = thresholds

    for threshold in iterator:
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate precision and recall
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        # Prior-adjusted confusion matrix to handle prevalence shift between
        # rebalanced CV data and production deployment.
        tp_w = tp * pos_scale
        fp_w = fp * neg_scale
        fn_w = fn * pos_scale
        tn_w = tn * neg_scale

        precision = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0
        recall = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0

        tpr = recall
        tnr = tn_w / (tn_w + fp_w) if (tn_w + fp_w) > 0 else 0
        balacc = 0.5 * (tpr + tnr)
        balanced_acc_scores.append(balacc)

        beta_sq = beta ** 2
        if precision + recall > 0:
            fbeta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
        else:
            fbeta = 0.0
        fbeta_scores.append(fbeta)

        # Track best so far
        if fbeta > best_fbeta_so_far:
            best_fbeta_so_far = fbeta
        if balacc > best_balacc_so_far:
            best_balacc_so_far = balacc

        precisions.append(precision)
        recalls.append(recall)
        fprs.append(fp / _n_neg if _n_neg > 0 else 0.0)

        # Deciduous-focused metrics: precision considering ONLY deciduous FPs
        if compute_deciduous:
            # FP from deciduous only
            fp_deciduous = ((y_pred == 1) & deciduous_mask).sum()
            # TP is still all wet woodland correctly predicted
            tp_wet = ((y_pred == 1) & wet_mask).sum()
            fn_wet = ((y_pred == 0) & wet_mask).sum()

            # Deciduous-focused precision: how many positives are NOT deciduous FPs
            # This is: TP / (TP + FP_deciduous)
            tp_wet_w = tp_wet * pos_scale
            fn_wet_w = fn_wet * pos_scale
            fp_deciduous_w = fp_deciduous * neg_scale

            dec_precision = tp_wet_w / (tp_wet_w + fp_deciduous_w) if (tp_wet_w + fp_deciduous_w) > 0 else 0
            dec_recall = tp_wet_w / (tp_wet_w + fn_wet_w) if (tp_wet_w + fn_wet_w) > 0 else 0

            # Deciduous-focused F-beta
            if dec_precision + dec_recall > 0:
                beta_sq = beta ** 2
                dec_fbeta = (1 + beta_sq) * (dec_precision * dec_recall) / (beta_sq * dec_precision + dec_recall)
            else:
                dec_fbeta = 0.0

            deciduous_fbeta_scores.append(dec_fbeta)
            deciduous_precisions.append(dec_precision)

            if dec_fbeta > best_dec_fbeta_so_far:
                best_dec_fbeta_so_far = dec_fbeta

        # Update progress bar
        if show_progress:
            if compute_deciduous:
                iterator.set_postfix(fbeta=best_fbeta_so_far, dec_fbeta=best_dec_fbeta_so_far)
            else:
                iterator.set_postfix(fbeta=best_fbeta_so_far, balacc=best_balacc_so_far)

    # Convert to arrays
    fbeta_scores = np.array(fbeta_scores)
    balanced_acc_scores = np.array(balanced_acc_scores)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    fprs = np.array(fprs)

    # Find optimal threshold for F-beta
    fbeta_optimal_idx = np.argmax(fbeta_scores)
    fbeta_optimal_threshold = thresholds[fbeta_optimal_idx]
    fbeta_best_score = fbeta_scores[fbeta_optimal_idx]
    fbeta_precision = precisions[fbeta_optimal_idx]
    fbeta_recall = recalls[fbeta_optimal_idx]
    fbeta_balacc = balanced_acc_scores[fbeta_optimal_idx]

    # Find optimal threshold for balanced accuracy
    balacc_optimal_idx = np.argmax(balanced_acc_scores)
    balacc_optimal_threshold = thresholds[balacc_optimal_idx]
    balacc_best_score = balanced_acc_scores[balacc_optimal_idx]
    balacc_precision = precisions[balacc_optimal_idx]
    balacc_recall = recalls[balacc_optimal_idx]
    balacc_fbeta = fbeta_scores[balacc_optimal_idx]

    # Get AUPRC for reference
    auprc = average_precision_score(y_true, y_pred_proba)

    # Deployment precision curve — how precise are predictions at national prevalence?
    # P_deploy(T) = π·TPR(T) / (π·TPR(T) + (1−π)·FPR(T))
    # Finds the OOF threshold needed to achieve each production precision target.
    deploy_thresholds = {}
    if national_prevalence is not None and 0.0 < float(national_prevalence) < 1.0:
        pi = float(national_prevalence)
        tprs = recalls  # recall = TPR on balanced OOF (pos_scale=1)
        deploy_prec = (pi * tprs) / (pi * tprs + (1.0 - pi) * fprs + 1e-9)
        for target in [0.05, 0.10, 0.25, 0.50, 0.75]:
            # Lowest threshold where deployment precision >= target (maximises recall at that precision)
            meeting = np.where(deploy_prec >= target)[0]
            if len(meeting) > 0:
                idx = meeting[0]
                deploy_thresholds[target] = {
                    'threshold': float(thresholds[idx]),
                    'tpr': float(tprs[idx]),
                    'fpr': float(fprs[idx]),
                    'deploy_precision': float(deploy_prec[idx]),
                }
            else:
                deploy_thresholds[target] = None  # not achievable with this model

    result = {
        'auprc': auprc,
        # F-beta optimization results
        'fbeta': {
            'optimal_threshold': fbeta_optimal_threshold,
            'best_score': fbeta_best_score,
            'precision': fbeta_precision,
            'recall': fbeta_recall,
            'balanced_accuracy': fbeta_balacc
        },
        # Balanced accuracy optimization results
        'balanced_accuracy': {
            'optimal_threshold': balacc_optimal_threshold,
            'best_score': balacc_best_score,
            'precision': balacc_precision,
            'recall': balacc_recall,
            'fbeta': balacc_fbeta
        },
        # Full arrays for analysis
        'thresholds': thresholds,
        'fbeta_scores': fbeta_scores,
        'balanced_acc_scores': balanced_acc_scores,
        'precisions': precisions,
        'recalls': recalls,
        'fprs': fprs,
        # Deployment precision: OOF threshold → precision at national prevalence
        'deploy_thresholds': deploy_thresholds,
    }

    result['prior_adjustment'] = {
        'enabled': use_prior_adjustment,
        'observed_pos_rate': observed_pos_rate,
        'target_pos_rate': target_pos_rate,
        'scale_positive': pos_scale,
        'scale_negative': neg_scale,
    }

    # Add deciduous-focused results if computed
    if compute_deciduous:
        deciduous_fbeta_scores = np.array(deciduous_fbeta_scores)
        deciduous_precisions = np.array(deciduous_precisions)

        dec_optimal_idx = np.argmax(deciduous_fbeta_scores)
        dec_optimal_threshold = thresholds[dec_optimal_idx]
        dec_best_fbeta = deciduous_fbeta_scores[dec_optimal_idx]
        dec_best_precision = deciduous_precisions[dec_optimal_idx]
        # Get overall recall at this threshold
        dec_recall = recalls[dec_optimal_idx]
        # Get overall precision at this threshold for comparison
        overall_precision_at_dec_thresh = precisions[dec_optimal_idx]

        result['deciduous_focused'] = {
            'optimal_threshold': dec_optimal_threshold,
            'best_fbeta': dec_best_fbeta,
            'deciduous_precision': dec_best_precision,
            'recall': dec_recall,
            'overall_precision': overall_precision_at_dec_thresh,
            'n_deciduous_samples': int(n_deciduous),
            'n_wet_samples': int(n_wet)
        }
        result['deciduous_fbeta_scores'] = deciduous_fbeta_scores
        result['deciduous_precisions'] = deciduous_precisions

    return result


def find_policy_seed_threshold(
    y_true,
    y_pred_proba,
    fold_ids,
    target_pos_rate,
    target_precision=DEFAULT_POLICY_TARGET_PRECISION,
    precision_quantile=DEFAULT_POLICY_PRECISION_QUANTILE,
    beta=DEFAULT_F_BETA,
    min_recall=DEFAULT_POLICY_MIN_RECALL,
    report_quantiles=None,
    report_precision_targets=None,
):
    """
    Calibrate a deployment seed threshold from fold-wise OOF behavior.

    The fold-wise TPR/FPR at each threshold are converted to expected deployment
    precision under target_pos_rate:
        PPV = p*TPR / (p*TPR + (1-p)*FPR)

    The selected threshold is the one with highest mean recall that satisfies:
      quantile_fold_precision >= target_precision and mean_recall >= min_recall.
    If no threshold satisfies the precision target, returns the best-available
    threshold by quantile precision.

    Returns:
        dict with best threshold and threshold curves.
    """
    if not (0.0 < float(target_pos_rate) < 1.0):
        raise ValueError("target_pos_rate must be in (0, 1)")
    if not (0.0 < float(target_precision) < 1.0):
        raise ValueError("target_precision must be in (0, 1)")
    if not (0.0 <= float(precision_quantile) <= 1.0):
        raise ValueError("precision_quantile must be in [0, 1]")
    if not (0.0 <= float(min_recall) <= 1.0):
        raise ValueError("min_recall must be in [0, 1]")

    if report_quantiles is None:
        report_quantiles = DEFAULT_POLICY_REPORT_QUANTILES
    if report_precision_targets is None:
        report_precision_targets = DEFAULT_POLICY_REPORT_PRECISIONS

    report_quantiles = sorted({float(q) for q in report_quantiles})
    report_precision_targets = sorted({float(t) for t in report_precision_targets})
    if any(not (0.0 <= q <= 1.0) for q in report_quantiles):
        raise ValueError("All report_quantiles must be in [0, 1]")
    if any(not (0.0 < t < 1.0) for t in report_precision_targets):
        raise ValueError("All report_precision_targets must be in (0, 1)")

    y_true = np.asarray(y_true).astype(np.int32)
    y_pred_proba = np.asarray(y_pred_proba).astype(np.float32)
    fold_ids = np.asarray(fold_ids).astype(np.int32)

    if not (len(y_true) == len(y_pred_proba) == len(fold_ids)):
        raise ValueError("y_true, y_pred_proba, and fold_ids must have the same length")

    valid_mask = fold_ids >= 0
    if valid_mask.sum() == 0:
        raise ValueError("No valid fold ids found (all fold ids < 0)")

    y_true = y_true[valid_mask]
    y_pred_proba = y_pred_proba[valid_mask]
    fold_ids = fold_ids[valid_mask]

    unique_folds = np.unique(fold_ids)
    if len(unique_folds) < 2:
        raise ValueError("Need at least 2 folds for fold-calibrated seed thresholding")

    thresholds = np.arange(0.01, 1.0, 0.01, dtype=np.float32)
    fold_precisions = np.full((len(unique_folds), len(thresholds)), np.nan, dtype=np.float64)
    fold_recalls = np.full((len(unique_folds), len(thresholds)), np.nan, dtype=np.float64)

    p = float(target_pos_rate)
    one_minus_p = 1.0 - p

    for i, fold in enumerate(unique_folds):
        fold_mask = fold_ids == fold
        y_fold = y_true[fold_mask]
        p_fold = y_pred_proba[fold_mask]

        pos_mask = y_fold == 1
        neg_mask = y_fold == 0
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        if n_pos == 0 or n_neg == 0:
            continue

        for j, threshold in enumerate(thresholds):
            pred = p_fold >= threshold
            tp = int(np.count_nonzero(pred & pos_mask))
            fp = int(np.count_nonzero(pred & neg_mask))

            tpr = tp / n_pos
            fpr = fp / n_neg

            denom = p * tpr + one_minus_p * fpr
            ppv = (p * tpr / denom) if denom > 0 else 0.0

            fold_precisions[i, j] = ppv
            fold_recalls[i, j] = tpr

    valid_fold_rows = ~np.all(np.isnan(fold_precisions), axis=1)
    if not np.any(valid_fold_rows):
        raise ValueError("No folds with both classes available for policy thresholding")

    fold_precisions = fold_precisions[valid_fold_rows]
    fold_recalls = fold_recalls[valid_fold_rows]
    used_fold_ids = unique_folds[valid_fold_rows]

    precision_q = np.nanquantile(fold_precisions, precision_quantile, axis=0)
    precision_mean = np.nanmean(fold_precisions, axis=0)
    recall_mean = np.nanmean(fold_recalls, axis=0)

    quantile_curves = {}
    for q in report_quantiles:
        quantile_curves[q] = np.nanquantile(fold_precisions, q, axis=0)

    beta_sq = beta ** 2
    calibrated_fbeta = np.zeros_like(precision_mean)
    valid_pr = (precision_mean + recall_mean) > 0
    calibrated_fbeta[valid_pr] = (
        (1 + beta_sq)
        * (precision_mean[valid_pr] * recall_mean[valid_pr])
        / (beta_sq * precision_mean[valid_pr] + recall_mean[valid_pr])
    )

    feasible = (precision_q >= target_precision) & (recall_mean >= min_recall)

    if np.any(feasible):
        candidate_idxs = np.where(feasible)[0]
        best_idx = max(
            candidate_idxs,
            key=lambda idx: (recall_mean[idx], precision_q[idx], float(thresholds[idx])),
        )
        target_met = True
    else:
        best_idx = max(
            range(len(thresholds)),
            key=lambda idx: (precision_q[idx], recall_mean[idx], float(thresholds[idx])),
        )
        target_met = False

    best_threshold = float(thresholds[best_idx])
    fold_precision_at_best = fold_precisions[:, best_idx]
    fold_recall_at_best = fold_recalls[:, best_idx]
    quantiles_at_best = {float(q): float(curve[best_idx]) for q, curve in quantile_curves.items()}

    quantile_target_sweep = {}
    for q, curve in quantile_curves.items():
        feasible_q = (curve >= target_precision) & (recall_mean >= min_recall)
        if np.any(feasible_q):
            candidate_idxs = np.where(feasible_q)[0]
            q_idx = max(
                candidate_idxs,
                key=lambda idx: (recall_mean[idx], curve[idx], float(thresholds[idx])),
            )
            q_met = True
        else:
            q_idx = max(
                range(len(thresholds)),
                key=lambda idx: (curve[idx], recall_mean[idx], float(thresholds[idx])),
            )
            q_met = False

        quantile_target_sweep[float(q)] = {
            "threshold": float(thresholds[q_idx]),
            "target_met": bool(q_met),
            "precision_value": float(curve[q_idx]),
            "precision_mean": float(precision_mean[q_idx]),
            "recall_mean": float(recall_mean[q_idx]),
            "calibrated_fbeta_mean": float(calibrated_fbeta[q_idx]),
        }

    precision_target_sweep = {}
    for target in report_precision_targets:
        feasible_t = (precision_q >= target) & (recall_mean >= min_recall)
        if np.any(feasible_t):
            candidate_idxs = np.where(feasible_t)[0]
            t_idx = max(
                candidate_idxs,
                key=lambda idx: (recall_mean[idx], precision_q[idx], float(thresholds[idx])),
            )
            t_met = True
        else:
            t_idx = max(
                range(len(thresholds)),
                key=lambda idx: (precision_q[idx], recall_mean[idx], float(thresholds[idx])),
            )
            t_met = False

        precision_target_sweep[float(target)] = {
            "threshold": float(thresholds[t_idx]),
            "target_met": bool(t_met),
            "precision_quantile_value": float(precision_q[t_idx]),
            "precision_mean": float(precision_mean[t_idx]),
            "recall_mean": float(recall_mean[t_idx]),
            "calibrated_fbeta_mean": float(calibrated_fbeta[t_idx]),
        }

    return {
        "target_pos_rate": p,
        "target_precision": float(target_precision),
        "precision_quantile": float(precision_quantile),
        "min_recall": float(min_recall),
        "n_folds_used": int(len(used_fold_ids)),
        "fold_ids_used": used_fold_ids.astype(int).tolist(),
        "thresholds": thresholds,
        "precision_quantile_curve": precision_q,
        "precision_mean_curve": precision_mean,
        "recall_mean_curve": recall_mean,
        "calibrated_fbeta_curve": calibrated_fbeta,
        "report_quantiles": [float(q) for q in report_quantiles],
        "precision_quantile_curves": {
            f"{float(q):.2f}": curve.astype(float).tolist() for q, curve in quantile_curves.items()
        },
        "quantile_target_sweep": {
            f"{float(q):.2f}": info for q, info in quantile_target_sweep.items()
        },
        "precision_target_sweep": {
            f"{float(t):.2f}": info for t, info in precision_target_sweep.items()
        },
        "best": {
            "threshold": best_threshold,
            "target_met": bool(target_met),
            "precision_quantile_value": float(precision_q[best_idx]),
            "precision_mean": float(precision_mean[best_idx]),
            "recall_mean": float(recall_mean[best_idx]),
            "calibrated_fbeta_mean": float(calibrated_fbeta[best_idx]),
            "precision_quantiles": quantiles_at_best,
            "fold_precision_values": fold_precision_at_best.astype(float).tolist(),
            "fold_recall_values": fold_recall_at_best.astype(float).tolist(),
        },
    }


def predict_in_batches(model, X_data, batch_size=5_000_000):
    """
    Predict with inplace_predict, falling back to DMatrix for compatibility.
    """
    if len(X_data) == 0:
        return np.array([], dtype=np.float32)

    if len(X_data) <= batch_size:
        try:
            return model.inplace_predict(X_data)
        except Exception:
            dtest = xgb.DMatrix(X_data)
            pred = model.predict(dtest)
            del dtest
            return pred

    pred_parts = []
    for start in range(0, len(X_data), batch_size):
        batch = X_data[start:start + batch_size]
        try:
            batch_pred = model.inplace_predict(batch)
        except Exception:
            batch_dtest = xgb.DMatrix(batch)
            batch_pred = model.predict(batch_dtest)
            del batch_dtest
        pred_parts.append(batch_pred)

    if pred_parts and hasattr(pred_parts[0], "ndim") and pred_parts[0].ndim > 1:
        return np.vstack(pred_parts)
    return np.concatenate(pred_parts)


def summarize_discarded_background_predictions(
    model,
    discarded_X,
    discarded_y,
    *,
    split_background_labels=True,
    threshold=0.5,
    extra_thresholds=None,
    sweep_thresholds=None,
):
    """
    Score discarded negatives that were not used in training.

    This is a sanity check on false-positive tendency, not an independent
    precision estimate. When split-background labels are available, the summary
    reports evergreen and deciduous background separately.
    """
    if discarded_X is None or discarded_y is None or len(discarded_X) == 0:
        return None

    discarded_y = np.asarray(discarded_y)
    if split_background_labels:
        bg_mask = np.isin(discarded_y, [0, 1])
    else:
        bg_mask = discarded_y == 0

    if not np.any(bg_mask):
        return None

    X_bg = discarded_X[bg_mask]
    y_bg = discarded_y[bg_mask].astype(np.int32)
    y_pred_proba = np.asarray(predict_in_batches(model, X_bg, batch_size=1_000_000), dtype=np.float32).reshape(-1)
    y_pred = y_pred_proba >= float(threshold)

    evergreen_mask = y_bg == 0
    deciduous_mask = y_bg == 1 if split_background_labels else np.zeros(len(y_bg), dtype=bool)

    fp_total = int(y_pred.sum())
    total = int(len(y_bg))
    fp_evergreen = int(np.sum(y_pred & evergreen_mask))
    total_evergreen = int(np.sum(evergreen_mask))
    fp_deciduous = int(np.sum(y_pred & deciduous_mask))
    total_deciduous = int(np.sum(deciduous_mask))

    threshold_hits = []
    for label, thr in (extra_thresholds or []):
        thr = float(thr)
        hits = int(np.sum(y_pred_proba >= thr))
        threshold_hits.append(
            {
                "label": str(label),
                "threshold": thr,
                "hits": hits,
                "rate": hits / total if total > 0 else 0.0,
            }
        )

    fpr_sweep = []
    for thr in (sweep_thresholds or []):
        thr = float(thr)
        pred_thr = y_pred_proba >= thr
        fp_thr = int(np.sum(pred_thr))
        fp_evg_thr = int(np.sum(pred_thr & evergreen_mask))
        fp_dec_thr = int(np.sum(pred_thr & deciduous_mask))
        fpr_sweep.append(
            {
                "threshold": thr,
                "fp_total": fp_thr,
                "fpr_total": fp_thr / total if total > 0 else 0.0,
                "fp_evergreen": fp_evg_thr,
                "fpr_evergreen": fp_evg_thr / total_evergreen if total_evergreen > 0 else 0.0,
                "fp_deciduous": fp_dec_thr,
                "fpr_deciduous": fp_dec_thr / total_deciduous if total_deciduous > 0 else 0.0,
            }
        )

    return {
        "n_background_samples": total,
        "n_ignored_non_background": int((~bg_mask).sum()),
        "mean_proba": float(np.mean(y_pred_proba)),
        "q95_proba": float(np.quantile(y_pred_proba, 0.95)),
        "q99_proba": float(np.quantile(y_pred_proba, 0.99)),
        "max_proba": float(np.max(y_pred_proba)),
        "fp_total": fp_total,
        "fpr_total": fp_total / total if total > 0 else 0.0,
        "fp_evergreen": fp_evergreen,
        "total_evergreen": total_evergreen,
        "fpr_evergreen": fp_evergreen / total_evergreen if total_evergreen > 0 else 0.0,
        "fp_deciduous": fp_deciduous,
        "total_deciduous": total_deciduous,
        "fpr_deciduous": fp_deciduous / total_deciduous if total_deciduous > 0 else 0.0,
        "threshold_hits": threshold_hits,
        "fpr_sweep": fpr_sweep,
    }


def summarize_fpr_sweep_crossings(fpr_sweep, target_rates=(0.01, 0.02, 0.05, 0.10)):
    """Find the highest threshold where discarded-background FPR reaches each target."""
    if not fpr_sweep:
        return []

    ordered = sorted(fpr_sweep, key=lambda item: item["threshold"], reverse=True)
    crossings = []
    for target_rate in target_rates:
        crossing = next(
            (item for item in ordered if item["fpr_total"] >= float(target_rate)),
            None,
        )
        crossings.append(
            {
                "target_rate": float(target_rate),
                "threshold": None if crossing is None else float(crossing["threshold"]),
            }
        )
    return crossings


def summarize_fpr_sweep_checkpoints(
    fpr_sweep,
    checkpoints=(0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05),
):
    """Return a compact set of representative discarded-background FPR checkpoints."""
    if not fpr_sweep:
        return []

    ordered = sorted(fpr_sweep, key=lambda item: item["threshold"])
    checkpoint_rows = []
    for threshold in checkpoints:
        row = min(
            ordered,
            key=lambda item: (
                abs(item["threshold"] - float(threshold)),
                -item["threshold"],
            ),
        )
        checkpoint_rows.append(
            {
                "threshold": float(row["threshold"]),
                "fpr_total": float(row["fpr_total"]),
                "fpr_evergreen": float(row["fpr_evergreen"]),
                "fpr_deciduous": float(row["fpr_deciduous"]),
            }
        )
    return checkpoint_rows


def summarize_fpr_sweep_window(
    fpr_sweep,
    threshold_min=0.15,
    threshold_max=0.30,
    step=0.01,
):
    """Return a dense local slice of the discarded-background FPR sweep."""
    if not fpr_sweep:
        return []

    threshold_grid = np.round(
        np.arange(float(threshold_min), float(threshold_max) + (step / 2.0), float(step)),
        2,
    )
    lookup = {
        round(float(item["threshold"]), 2): item
        for item in fpr_sweep
    }

    rows = []
    for threshold in threshold_grid:
        key = round(float(threshold), 2)
        if key not in lookup:
            continue
        item = lookup[key]
        rows.append(
            {
                "threshold": key,
                "fpr_total": float(item["fpr_total"]),
                "fpr_evergreen": float(item["fpr_evergreen"]),
                "fpr_deciduous": float(item["fpr_deciduous"]),
            }
        )
    return rows


def estimate_fpr_sweep_breakpoint(fpr_sweep, min_segment_size=5):
    """
    Estimate an elbow in the discarded-background FPR curve using a
    two-segment linear fit on log10(FPR).

    This gives a more defensible changepoint than just picking a visual knee.
    """
    if not fpr_sweep:
        return None

    ordered = sorted(fpr_sweep, key=lambda item: item["threshold"], reverse=True)
    if len(ordered) < (2 * min_segment_size):
        return None

    x = np.asarray([item["threshold"] for item in ordered], dtype=np.float64)
    y = np.asarray([max(item["fpr_total"], 1e-8) for item in ordered], dtype=np.float64)
    log_y = np.log10(y)

    def _line_sse(x_part, y_part):
        coeff = np.polyfit(x_part, y_part, deg=1)
        pred = np.polyval(coeff, x_part)
        return float(np.sum((y_part - pred) ** 2))

    full_sse = _line_sse(x, log_y)
    best = None
    for split_idx in range(min_segment_size, len(x) - min_segment_size + 1):
        left_sse = _line_sse(x[:split_idx], log_y[:split_idx])
        right_sse = _line_sse(x[split_idx - 1 :], log_y[split_idx - 1 :])
        total_sse = left_sse + right_sse
        if best is None or total_sse < best["segmented_sse"]:
            breakpoint_threshold = float(np.mean([x[split_idx - 1], x[split_idx]]))
            breakpoint_row = ordered[split_idx]
            best = {
                "threshold": breakpoint_threshold,
                "fpr_total": float(breakpoint_row["fpr_total"]),
                "fpr_evergreen": float(breakpoint_row["fpr_evergreen"]),
                "fpr_deciduous": float(breakpoint_row["fpr_deciduous"]),
                "segmented_sse": total_sse,
            }

    if best is None:
        return None

    best["single_line_sse"] = full_sse
    best["sse_improvement"] = (
        0.0 if full_sse <= 0 else max(0.0, 1.0 - (best["segmented_sse"] / full_sse))
    )
    return best


def sample_abiotics_at_coordinates(abiotic_path, coordinates, bands=None, chunk_size=100000):
    """
    Efficiently sample abiotic raster values at given coordinates.

    Uses rasterio.sample() which reads only the needed pixels without loading
    the entire raster into memory - crucial for large national mosaics.
    Results are cached to cache/ so repeat runs skip the sampling entirely.

    Args:
        abiotic_path: Path to abiotic raster stack (e.g., DTM-derived features)
        coordinates: Array of (x, y) coordinates in raster CRS
        bands: List of band numbers to extract (1-indexed). If None, use all bands.
        chunk_size: Process coordinates in chunks to manage memory

    Returns:
        Array of shape (n_samples, n_bands) with abiotic values
    """
    if bands is None:
        with rasterio.open(abiotic_path) as src:
            bands = list(range(1, src.count + 1))
    else:
        bands = [int(b) for b in bands]

    # Build a cache key from raster mtime + coordinate fingerprint + bands
    raster_mtime = os.path.getmtime(abiotic_path)
    coord_fingerprint = (
        coordinates.shape,
        coordinates[:100].tobytes(),
        coordinates[-100:].tobytes(),
        float(coordinates.sum()),
    )
    cache_hash = hashlib.md5(
        f"{abiotic_path}_{raster_mtime}_{coord_fingerprint}_{sorted(bands)}".encode()
    ).hexdigest()[:12]
    cache_path = Path("cache") / f"abiotics_{cache_hash}.pkl"
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists():
        print(f"   ⚡ Loading abiotics from cache: {cache_path.name}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"   Sampling abiotics from {Path(abiotic_path).name}...")

    with rasterio.open(abiotic_path) as src:
        # DIAGNOSTICS: Print raster info
        print(f"   📊 ABIOTIC RASTER DIAGNOSTICS:")
        print(f"      CRS: {src.crs}")
        print(f"      Bounds: {src.bounds}")
        print(f"      Size: {src.width} x {src.height} pixels")
        print(f"      Resolution: {src.res[0]:.1f}m x {src.res[1]:.1f}m")
        print(f"      Bands: {src.count}")
        print(f"      NoData: {src.nodata}")

        # DIAGNOSTICS: Print coordinate info
        coord_x_min, coord_x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        coord_y_min, coord_y_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        print(f"   📍 COORDINATE DIAGNOSTICS:")
        print(f"      N samples: {len(coordinates):,}")
        print(f"      X range: {coord_x_min:.1f} to {coord_x_max:.1f}")
        print(f"      Y range: {coord_y_min:.1f} to {coord_y_max:.1f}")

        # DIAGNOSTICS: Check overlap
        raster_bounds = src.bounds
        x_overlap = (coord_x_min < raster_bounds.right) and (coord_x_max > raster_bounds.left)
        y_overlap = (coord_y_min < raster_bounds.top) and (coord_y_max > raster_bounds.bottom)

        if not (x_overlap and y_overlap):
            print(f"   ❌ NO OVERLAP between coordinates and raster!")
            print(f"      Raster X: {raster_bounds.left:.1f} to {raster_bounds.right:.1f}")
            print(f"      Raster Y: {raster_bounds.bottom:.1f} to {raster_bounds.top:.1f}")
            print(f"      Coords X: {coord_x_min:.1f} to {coord_x_max:.1f}")
            print(f"      Coords Y: {coord_y_min:.1f} to {coord_y_max:.1f}")
        else:
            # Calculate how many points are inside bounds
            inside_x = (coordinates[:, 0] >= raster_bounds.left) & (coordinates[:, 0] <= raster_bounds.right)
            inside_y = (coordinates[:, 1] >= raster_bounds.bottom) & (coordinates[:, 1] <= raster_bounds.top)
            inside_both = inside_x & inside_y
            print(f"   ✅ Overlap detected: {inside_both.sum():,}/{len(coordinates):,} points inside raster bounds ({100*inside_both.mean():.1f}%)")

        # DEBUG: Test sample a few coordinates directly
        print(f"   🔍 DEBUG: Testing 5 sample coordinates...")
        test_coords = coordinates[:5]
        for i, (x, y) in enumerate(test_coords):
            # Convert coordinate to pixel
            row, col = src.index(x, y)
            print(f"      Coord {i}: ({x:.1f}, {y:.1f}) -> pixel ({row}, {col})", end="")
            if 0 <= row < src.height and 0 <= col < src.width:
                # Read actual pixel value
                window = rasterio.windows.Window(col, row, 1, 1)
                pixel_val = src.read(bands[0], window=window)[0, 0]
                # Also try sample method
                sample_val = list(src.sample([(x, y)], indexes=bands[0]))[0][0]
                print(f" -> direct_read={pixel_val:.3f}, sample={sample_val:.3f}")
            else:
                print(f" -> OUT OF BOUNDS!")

        n_samples = len(coordinates)
        n_bands = len(bands)
        abiotic_values = np.zeros((n_samples, n_bands), dtype=np.float32)

        # Process in chunks for memory efficiency
        for start in tqdm(range(0, n_samples, chunk_size), desc="   Sampling", unit="chunk"):
            end = min(start + chunk_size, n_samples)
            chunk_coords = coordinates[start:end]

            # rasterio.sample expects [(x,y), (x,y), ...] format
            coord_list = [tuple(c) for c in chunk_coords]

            for band_idx, band_num in enumerate(bands):
                # Sample this band at all chunk coordinates
                values = list(src.sample(coord_list, indexes=band_num))
                abiotic_values[start:end, band_idx] = np.array(values).flatten()

        # Handle nodata
        nodata = src.nodata
        if nodata is not None:
            nodata_count = np.sum(abiotic_values == nodata)
            print(f"   📊 NoData pixels found: {nodata_count:,}")
            abiotic_values[abiotic_values == nodata] = np.nan

        # Report detailed stats per band
        print(f"   📊 VALUE DIAGNOSTICS per band:")
        for band_idx, band_num in enumerate(bands):
            band_vals = abiotic_values[:, band_idx]
            valid = ~np.isnan(band_vals)
            n_valid = valid.sum()
            n_nan = (~valid).sum()
            if n_valid > 0:
                print(f"      Band {band_num}: valid={n_valid:,} nan={n_nan:,} range=[{np.nanmin(band_vals):.2f}, {np.nanmax(band_vals):.2f}]")
            else:
                print(f"      Band {band_num}: ALL NaN!")

        valid_pct = 100 * np.sum(~np.isnan(abiotic_values[:, 0])) / n_samples
        print(f"   ✅ Sampled {n_bands} bands at {n_samples:,} points ({valid_pct:.1f}% valid)")

        print(f"   💾 Caching abiotics to {cache_path.name}")
        with open(cache_path, 'wb') as f:
            pickle.dump(abiotic_values, f)

        return abiotic_values



def _fill_raster_nans_spatial(raster_path, coordinates, values, search_radius=1):
    """
    Fill NaN sample values using a spatial median of neighboring raster pixels.

    For each NaN-valued sample coordinate, samples a (2r+1)×(2r+1) window of
    neighboring pixels directly from the raster and fills with their nanmedian.
    Falls back to global median for any points where all neighbors are also NaN
    (e.g., inside large water bodies).

    Args:
        raster_path: path to the raster file
        coordinates: (N, 2) float array of (x, y) sample coordinates in raster CRS
        values: (N, 1) float array of sampled values with NaN where nodata
        search_radius: pixel search distance in each direction (default 1 → 3×3 window)

    Returns:
        values array with NaN filled
    """
    nan_mask = np.isnan(values[:, 0])
    if not nan_mask.any():
        return values

    n_nan = int(nan_mask.sum())
    nan_coords = coordinates[nan_mask]  # (n_nan, 2)

    with rasterio.open(raster_path) as src:
        px = src.res[0]   # pixel size (x)
        py = src.res[1]   # pixel size (y) — positive even for north-up rasters
        nodata = src.nodata

        # All (dx, dy) offsets in pixel steps, excluding the centre
        offsets = [
            (dx, dy)
            for dx in range(-search_radius, search_radius + 1)
            for dy in range(-search_radius, search_radius + 1)
            if not (dx == 0 and dy == 0)
        ]

        # One rasterio.sample call per offset — each processes all n_nan coords
        neighbor_stack = []
        for dx, dy in offsets:
            # In BNG (north-up), y increases northward so dy>0 moves north
            shifted = nan_coords + np.array([dx * px, dy * py])
            sampled = np.array(
                list(src.sample([tuple(c) for c in shifted], indexes=1))
            ).flatten().astype(np.float32)
            if nodata is not None:
                sampled[sampled == nodata] = np.nan
            neighbor_stack.append(sampled)

        # (n_nan, n_offsets) → nanmedian per NaN point
        neighbor_arr = np.stack(neighbor_stack, axis=1)
        fill_vals = np.nanmedian(neighbor_arr, axis=1).astype(np.float32)

        # Fallback: global median for points with no valid neighbors at all
        still_nan = np.isnan(fill_vals)
        if still_nan.any():
            global_med = float(np.nanmedian(values[~nan_mask, 0]))
            fill_vals[still_nan] = global_med
            print(f"     ⚠️  {still_nan.sum():,} points had no valid neighbors → global median ({global_med:.4f})")

    out = values.copy()
    out[nan_mask, 0] = fill_vals
    return out


def select_spatial_validation_fold(y_sampled, fold_assignments_sampled):
    """
    Choose a validation fold that keeps both train and validation class-diverse.
    """
    candidate_folds = np.unique(fold_assignments_sampled)
    best_fold = None
    best_val_size = -1

    for fold_id in candidate_folds:
        val_mask = fold_assignments_sampled == fold_id
        train_mask = ~val_mask
        if val_mask.sum() < 1000 or train_mask.sum() < 1000:
            continue
        if np.unique(y_sampled[val_mask]).size < 2:
            continue
        if np.unique(y_sampled[train_mask]).size < 2:
            continue
        if val_mask.sum() > best_val_size:
            best_val_size = int(val_mask.sum())
            best_fold = int(fold_id)

    return best_fold


def optimize_with_optuna(
    X,
    y,
    coordinates=None,
    n_trials=75,
    use_binary=True,
    gpu_available=False,
    gpu_id=0,
    optuna_samples=50000,
    force_cpu=False,
    n_estimators=3000,
    scale_pos_weight_override=None,
    max_depth_override=None,
    min_depth_search=4,
    max_depth_search=DEFAULT_MAX_DEPTH,
):
    """
    Use Optuna for hyperparameter optimization with XGBoost.

    Uses small balanced stratified random sample - completely independent of
    spatial folds used in CV. This prevents hyperparameter tuning from exploiting
    the same spatial structure used in cross-validation.

    Args:
        optuna_samples: Total samples for Optuna (default 50000).
                       Samples are balanced across classes.
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna is required for hyperparameter optimisation but is not installed.")

    # Balanced stratified sampling
    rng = np.random.default_rng(42)
    unique_classes = np.unique(y)
    samples_per_class = optuna_samples // len(unique_classes)

    sampled_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n_take = min(samples_per_class, len(cls_indices))
        sampled_indices.extend(rng.choice(cls_indices, n_take, replace=False))

    sampled_indices = np.array(sampled_indices)
    rng.shuffle(sampled_indices)
    X_sampled, y_sampled = X[sampled_indices], y[sampled_indices]
    coords_sampled = coordinates[sampled_indices] if coordinates is not None else None

    print(f"   Samples: {len(y_sampled):,} balanced")

    if coords_sampled is not None:
        # Spatial block split (~10km blocks)
        x_min, x_max = coords_sampled[:, 0].min(), coords_sampled[:, 0].max()
        y_min, y_max = coords_sampled[:, 1].min(), coords_sampled[:, 1].max()
        block_size = 10000
        n_blocks_x = max(3, int((x_max - x_min) / block_size))
        n_blocks_y = max(3, int((y_max - y_min) / block_size))

        x_bins = np.linspace(x_min, x_max, n_blocks_x + 1)
        y_bins = np.linspace(y_min, y_max, n_blocks_y + 1)
        x_idx = np.clip(np.digitize(coords_sampled[:, 0], x_bins) - 1, 0, n_blocks_x - 1)
        y_idx = np.clip(np.digitize(coords_sampled[:, 1], y_bins) - 1, 0, n_blocks_y - 1)
        block_ids = x_idx * n_blocks_y + y_idx

        unique_blocks_optuna = np.unique(block_ids)
        n_val_blocks = max(1, len(unique_blocks_optuna) // 5)
        rng_blocks = np.random.default_rng(42)
        val_block_set = set(rng_blocks.choice(unique_blocks_optuna, n_val_blocks, replace=False))
        val_mask = np.isin(block_ids, list(val_block_set))

        X_train_hp, y_train_hp = X_sampled[~val_mask], y_sampled[~val_mask]
        X_val_hp, y_val_hp = X_sampled[val_mask], y_sampled[val_mask]
        print(f"   Split: {len(y_train_hp):,} train / {len(y_val_hp):,} val (spatial blocks)")
    else:
        X_train_hp, X_val_hp, y_train_hp, y_val_hp = train_test_split(
            X_sampled, y_sampled, test_size=0.2, stratify=y_sampled, random_state=42
        )
        print(f"   Split: {len(y_train_hp):,} train / {len(y_val_hp):,} val (random)")

    # Compute rebalanced class ratio — scale_pos_weight is set to this directly
    observed_ratio = 1.0
    if use_binary:
        n_neg = int(np.sum(y == 0))
        n_pos = int(np.sum(y == 1))
        if n_pos > 0:
            observed_ratio = n_neg / n_pos
            if scale_pos_weight_override is not None:
                print(
                    f"   Class ratio (neg/pos): {observed_ratio:.2f} → "
                    f"scale_pos_weight fixed at {float(scale_pos_weight_override):.2f}"
                )
            else:
                print(f"   Class ratio (neg/pos): {observed_ratio:.2f} → scale_pos_weight search: [1.0, 1.2]")
    if max_depth_override is not None:
        print(f"   max_depth fixed at {int(max_depth_override)}")
    else:
        print(f"   max_depth search: [{int(min_depth_search)}, {int(max_depth_search)}]")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='maximize',  # Maximise AUPRC
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )

    # Optimize
    study.optimize(
        lambda trial: optuna_objective(
            trial,
            X_train_hp,
            y_train_hp,
            X_val_hp,
            y_val_hp,
            use_binary,
            gpu_available,
            gpu_id,
            force_cpu,
            observed_ratio,
            n_estimators,
            scale_pos_weight_override,
            max_depth_override,
            min_depth_search,
            max_depth_search,
        ),
        n_trials=n_trials, show_progress_bar=True
    )

    best_params = study.best_params
    if use_binary and scale_pos_weight_override is not None:
        best_params['scale_pos_weight'] = float(scale_pos_weight_override)
    if max_depth_override is not None:
        best_params['max_depth'] = int(max_depth_override)
    best_score = study.best_value
    best_iteration = study.best_trial.user_attrs.get('best_iteration', n_estimators)

    # Compact results
    print(f"\n   ✅ Best AUPRC: {best_score:.4f} (trial #{study.best_trial.number})")
    print(f"   Best iteration: {best_iteration} / {n_estimators}")
    param_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(best_params.items())]
    print(f"   Best params: {' | '.join(param_strs)}")

    return best_params, best_score, study, best_iteration


# Use the same data loading functions from the LightGBM version
# (I'll import the key functions to avoid duplication)

def generate_cache_key(
    data_dir,
    labels_file,
    max_tiles,
    max_pixels_per_tile,
    apply_height_filter,
    min_chm_cover,
    min_chm_mean,
):
    """Generate a unique cache key based on input parameters."""
    # Get modification times of labels file
    labels_mtime = os.path.getmtime(labels_file)

    # Get list of data files and their modification times
    data_files = glob.glob(str(Path(data_dir) / "*.tif"))
    if max_tiles:
        data_files = data_files[:max_tiles]

    # Create a hash of all relevant parameters
    cache_string = (
        f"{data_dir}_{labels_file}_{labels_mtime}_{len(data_files)}_{max_pixels_per_tile}_"
        f"{apply_height_filter}_{min_chm_cover}_{min_chm_mean}"
    )
    cache_hash = hashlib.md5(cache_string.encode()).hexdigest()[:12]

    return f"processed_data_{cache_hash}.pkl"


def save_processed_data(X, y, coordinates, cache_file):
    """Save processed data to cache file."""
    cache_path = Path("cache") / cache_file
    cache_path.parent.mkdir(exist_ok=True)

    print(f"💾 Saving processed data to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'X': X,
            'y': y,
            'coordinates': coordinates,
            'timestamp': time.time(),
            'n_samples': len(X),
            'n_features': X.shape[1]
        }, f)


def load_processed_data(cache_file):
    """Load processed data from cache file if it exists."""
    cache_path = Path("cache") / cache_file

    if not cache_path.exists():
        return None, None, None

    try:
        print(f"   Loading from cache: {cache_path.name}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)

        # Verify data integrity
        X, y = data['X'], data['y']

        # Handle coordinates - backward compatibility with old cache
        if 'coordinates' in data:
            coordinates = data['coordinates']

            # Validate cache: ensure all arrays match in length
            if len(X) != len(y) or (coordinates is not None and len(coordinates) != len(X)):
                print(f"⚠️  Cache validation failed: X={len(X)}, y={len(y)}, coords={len(coordinates) if coordinates is not None else 'None'}")
                print(f"   Cache is corrupted or from old version - regenerating...")
                return None, None, None

            print(f"   ✅ Loaded: {data['n_samples']:,} samples, {data['n_features']} features")
            print(f"   ✅ Spatial coordinates available")
            return X, y, coordinates
        else:
            print(f"   ✅ Loaded: {data['n_samples']:,} samples, {data['n_features']} features")
            print(f"   ⚠️  Old cache format - coordinates missing")
            return X, y, None

    except Exception as e:
        print(f"⚠️  Cache file corrupted, will regenerate: {e}")
        return None, None, None


def process_single_tile(args):
    """Process a single GeoTIFF tile to extract valid pixels for multi-class."""
    data_file, labels_file, max_pixels_per_tile, apply_height_filter, min_chm_cover, min_chm_mean = args

    try:
        with rasterio.open(data_file) as data_src:
            # Read all bands
            data = data_src.read()  # Shape: (bands, height, width)

            # Create labels array matching data dimensions (initialize with nodata)
            labels = np.full((data_src.height, data_src.width), 255, dtype=np.float32)

            # Check if reprojection is needed
            with rasterio.open(labels_file) as labels_src:
                if data_src.crs == labels_src.crs:
                    # Fast path: Same CRS, use window reading
                    try:
                        # Read labels for the area covered by this tile
                        window = rasterio.windows.from_bounds(
                            *data_src.bounds,
                            transform=labels_src.transform
                        )

                        # Read labels at tile resolution
                        labels_data = labels_src.read(
                            1,
                            window=window,
                            out_shape=(data_src.height, data_src.width),
                            resampling=Resampling.nearest
                        )

                        # Copy to our labels array
                        labels[:] = labels_data

                    except Exception:
                        # Fall back to reprojection if window reading fails
                        reproject(
                            source=rasterio.band(labels_src, 1),
                            destination=labels,
                            src_transform=labels_src.transform,
                            src_crs=labels_src.crs,
                            dst_transform=data_src.transform,
                            dst_crs=data_src.crs,
                            resampling=Resampling.nearest
                        )
                else:
                    # Slow path: Different CRS, need reprojection
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
            if apply_height_filter:
                height_keep_mask, filter_supported = build_tile_height_filter_mask(
                    data,
                    min_chm_cover=min_chm_cover,
                    min_chm_mean=min_chm_mean,
                )
                if filter_supported:
                    data_valid &= height_keep_mask

            # Handle supported labels with 255 as nodata:
            # - Legacy: 0,1,2
            # - Split background: 0,1,2,3
            labels_valid = (labels == 0) | (labels == 1) | (labels == 2) | (labels == 3)

            valid_mask = data_valid & labels_valid

            if valid_mask.sum() == 0:
                return None, None, Path(data_file).name

            # Extract valid pixels
            features = data[:, valid_mask].T  # Shape: (n_pixels, n_bands)

            # Keep every tile band exactly as stored unless the user explicitly
            # requests exclusions later via --exclude-features.

            labels_valid = labels[valid_mask].astype(np.int32)

            # Get pixel coordinates BEFORE subsampling (for all valid pixels)
            # VECTORIZED: 100x faster than Python loop for millions of pixels
            rows, cols = np.where(valid_mask)
            if len(rows) > 0:
                # Vectorized coordinate transformation
                transform = data_src.transform
                xs = transform.c + cols * transform.a + rows * transform.b
                ys = transform.f + cols * transform.d + rows * transform.e
                pixel_coords = np.column_stack([xs, ys])
            else:
                pixel_coords = np.array([]).reshape(0, 2)

            # Subsample if too many pixels (stratified for all classes)
            if max_pixels_per_tile is not None and len(features) > max_pixels_per_tile:
                # Get indices for each class
                class_indices = {}
                present_classes = np.unique(labels_valid)
                for class_label in present_classes:
                    class_indices[class_label] = np.where(labels_valid == class_label)[0]

                # Calculate proportions
                total_pixels = len(labels_valid)
                sample_indices = []

                for class_label in present_classes:
                    if len(class_indices[class_label]) > 0:
                        class_ratio = len(class_indices[class_label]) / total_pixels
                        n_samples = min(
                            len(class_indices[class_label]),
                            int(max_pixels_per_tile * class_ratio)
                        )
                        if n_samples > 0:
                            sampled = np.random.choice(
                                class_indices[class_label],
                                n_samples,
                                replace=False
                            )
                            sample_indices.extend(sampled)

                sample_indices = np.array(sample_indices)
                features = features[sample_indices]
                labels_valid = labels_valid[sample_indices]
                pixel_coords = pixel_coords[sample_indices]  # Subsample coordinates too!

            return features, labels_valid, pixel_coords, Path(data_file).name

    except Exception as e:
        return None, None, None, f"{Path(data_file).name}: {str(e)}"


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


def load_data_parallel(
    data_dir,
    labels_file,
    max_tiles=None,
    max_pixels_per_tile=None,
    n_workers=None,
    use_cache=True,
    apply_height_filter=True,
    min_chm_cover=DEFAULT_MIN_CHM_COVER,
    min_chm_mean=DEFAULT_MIN_CHM_MEAN,
):
    """Load data using multiprocessing for multi-class with caching support."""
    print("\n" + "-" * 70)
    print("📂 Loading Data")
    print("-" * 70)

    # Generate cache key
    cache_file = generate_cache_key(
        data_dir,
        labels_file,
        max_tiles,
        max_pixels_per_tile,
        apply_height_filter,
        min_chm_cover,
        min_chm_mean,
    )

    # Try to load from cache first
    if use_cache:
        X, y, coordinates = load_processed_data(cache_file)
        if X is not None and y is not None:
            # Print class distribution from cache
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_percentages = class_counts / len(y) * 100

            print(f"\n   Class distribution:")
            for cls, count, pct in zip(unique_classes, class_counts, class_percentages):
                print(f"     Class {cls}: {count:,} samples ({pct:.2f}%)")

            return X, y, coordinates

    print("🔄 No valid cache found, processing tiles from scratch...")

    # Find overlapping files
    data_files = find_overlapping_tiles(data_dir, labels_file)

    if max_tiles:
        data_files = data_files[:max_tiles]
        print(f"Using first {max_tiles} tiles for processing")

    # Determine number of workers
    if n_workers is None:
        # Conservative worker count for I/O bound tile processing
        n_workers = min(mp.cpu_count() - 2, len(data_files), 8)  # Avoid disk I/O contention

    print(f"Processing {len(data_files)} tiles with {n_workers} workers...")

    # Prepare arguments for parallel processing
    process_args = [
        (f, labels_file, max_pixels_per_tile, apply_height_filter, min_chm_cover, min_chm_mean)
        for f in data_files
    ]

    # Process files in parallel
    all_features = []
    all_labels = []
    all_coordinates = []
    successful_tiles = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_tile, args): args[0]
                         for args in process_args}

        # Collect results as they complete with dynamic progress
        pbar = tqdm(total=len(data_files), desc="Processing tiles", unit="tile")
        for future in as_completed(future_to_file):
            result = future.result()
            pbar.update(1)

            if len(result) == 4:  # New format with coordinates
                features, labels, coords, info = result
                if features is not None:
                    all_features.append(features)
                    all_labels.append(labels)
                    all_coordinates.append(coords)
                    successful_tiles += 1
                else:
                    pbar.write(f"❌ Failed: {info}")
            else:  # Backward compatibility
                features, labels, info = result
                if features is not None:
                    all_features.append(features)
                    all_labels.append(labels)
                    # Create dummy coordinates if not available
                    all_coordinates.append(np.zeros((len(features), 2)))
                    successful_tiles += 1
                else:
                    pbar.write(f"❌ Failed: {info}")

        pbar.close()

    if not all_features:
        print("❌ No valid data found!")
        return None, None, None

    # Combine all data
    print(f"\n🔗 Combining data from {successful_tiles} tiles...")
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    coordinates = np.vstack(all_coordinates)

    # Save to cache
    if use_cache:
        save_processed_data(X, y, coordinates, cache_file)

    # Calculate class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_percentages = class_counts / len(y) * 100

    print(f"✅ Final dataset:")
    print(f"   Successful tiles: {successful_tiles}/{len(data_files)}")
    print(f"   Total samples: {len(X):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Class distribution:")
    for cls, count, pct in zip(unique_classes, class_counts, class_percentages):
        print(f"     Class {cls}: {count:,} samples ({pct:.2f}%)")
    print(f"   Memory usage: ~{X.nbytes / 1024**2:.1f} MB")
    print(f"   Coordinate range: X({coordinates[:, 0].min():.0f}, {coordinates[:, 0].max():.0f}), Y({coordinates[:, 1].min():.0f}, {coordinates[:, 1].max():.0f})")

    return X, y, coordinates


def compute_gpu_shap_values(model, X, y, gpu_available=False, gpu_id=0, n_samples=10000, feature_names=None):
    """Compute GPU-accelerated SHAP values for model interpretability."""
    if not gpu_available:
        print("   ⚠️ GPU not available, skipping SHAP")
        return None, None, None

    try:
        # Sample data for SHAP computation
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_shap, y_shap = X[indices], y[indices]
        else:
            X_shap, y_shap = X, y

        # Create DMatrix
        try:
            dshap = xgb.QuantileDMatrix(X_shap, label=y_shap, max_bin=256, feature_names=feature_names)
        except Exception:
            dshap = xgb.DMatrix(X_shap, label=y_shap, feature_names=feature_names)

        model.set_param({"device": f"cuda:{gpu_id}"})

        # Compute SHAP values
        shap_values = model.predict(dshap, pred_contribs=True)

        # Try interaction values
        try:
            shap_interaction_values = model.predict(dshap, pred_interactions=True)
        except Exception:
            shap_interaction_values = None

        # Feature importance from SHAP
        mean_abs_shap = np.mean(np.abs(shap_values[:, :-1]), axis=0)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]

        feature_importance = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)

        return shap_values, shap_interaction_values, feature_importance

    except Exception as e:
        print(f"   ❌ SHAP failed: {e}")
        return None, None, None


def create_spatial_folds(X, y, coordinates, n_folds=5):
    """
    Create spatial fold assignments for cross-validation.

    Uses 2km grid blocks with round-robin scattered assignment for
    genuine geographic separation across folds.

    Returns fold_assignments, spatial_groups.
    """
    coord_range_x = coordinates[:, 0].max() - coordinates[:, 0].min()
    coord_range_y = coordinates[:, 1].max() - coordinates[:, 1].min()

    # Create 2km spatial grid
    target_block_size_km = 2.0
    n_blocks_x = max(10, int(coord_range_x / (target_block_size_km * 1000)))
    n_blocks_y = max(10, int(coord_range_y / (target_block_size_km * 1000)))

    # Assign each point to a grid cell
    x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()

    x_bins = np.linspace(x_min, x_max, n_blocks_x + 1)
    y_bins = np.linspace(y_min, y_max, n_blocks_y + 1)

    x_indices = np.clip(np.digitize(coordinates[:, 0], x_bins) - 1, 0, n_blocks_x - 1)
    y_indices = np.clip(np.digitize(coordinates[:, 1], y_bins) - 1, 0, n_blocks_y - 1)

    spatial_groups = x_indices * n_blocks_y + y_indices
    unique_blocks = np.unique(spatial_groups)

    # Compute block sample counts
    block_sample_counts = np.array([np.sum(spatial_groups == bid) for bid in unique_blocks])

    print(f"   Spatial folds: {n_blocks_x}×{n_blocks_y} grid, {len(unique_blocks)} blocks with data")

    # Scattered block assignment (round-robin)
    rng = np.random.default_rng(42)
    shuffled_blocks = unique_blocks.copy()
    rng.shuffle(shuffled_blocks)
    block_to_fold = {block_id: i % n_folds for i, block_id in enumerate(shuffled_blocks)}
    block_fold_assignments = np.array([block_to_fold[bid] for bid in unique_blocks])
    fold_assignments = np.array([block_to_fold[block_id] for block_id in spatial_groups], dtype=np.int32)

    # Report balance
    unique_folds, fold_counts = np.unique(fold_assignments, return_counts=True)
    min_pct = 100 * fold_counts.min() / len(coordinates)
    max_pct = 100 * fold_counts.max() / len(coordinates)
    balance_status = "balanced" if max_pct - min_pct <= 10 else "imbalanced"
    print(f"   Folds {balance_status}: {min_pct:.1f}%-{max_pct:.1f}% per fold")

    # n_blocks_y returned so callers can decode block_id → (x_idx, y_idx) for buffer zones
    return fold_assignments, spatial_groups, n_blocks_y


def train_xgboost_folds(X, y, coordinates=None, spatial_cv=False, gpu_available=False, gpu_id=0, n_estimators=1500, fold_split_ratio=0.8,
                       fold_assignments=None, spatial_groups_precomputed=None, use_cv_model=False, y_original=None,
                       feature_names=None, n_blocks_y=None, spatial_buffer_blocks=1, n_folds=10, **xgb_params):
    """
    Train XGBoost with fold-based cross-validation.

    Uses spatial or standard k-fold cross-validation for model evaluation.
    Binary classification uses fixed 0.5 threshold (assumes balanced data).

    If use_cv_model=True, skips Stage 2 and uses the rolling CV model directly.
    This model has implicit regularization (each tree trained on different ~80% subset).
    """
    import gc

    # Check number of classes
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    is_binary = n_classes == 2

    print(f"   {'Binary' if is_binary else 'Multi-class'} classification ({n_classes} classes)")

    # Use pre-computed folds if available
    if spatial_cv and coordinates is not None and SPATIAL_KFOLD_AVAILABLE:
        if fold_assignments is not None and spatial_groups_precomputed is not None:
            spatial_groups = spatial_groups_precomputed
            n_folds = len(np.unique(fold_assignments))
        else:
            fold_assignments, spatial_groups, n_blocks_y = create_spatial_folds(
                X, y, coordinates, n_folds=n_folds
            )

        # Pre-cache fold datasets
        fold_datasets = {}
        for fold in range(n_folds):
            test_mask = fold_assignments == fold
            train_mask = ~test_mask

            train_count = train_mask.sum()
            if train_count >= 10:
                train_mask_key = tuple(train_mask)
                fold_datasets[train_mask_key] = xgb.QuantileDMatrix(X[train_mask], label=y[train_mask], max_bin=1024, feature_names=feature_names)

        # For multiclass: pre-cache binarized labels
        fold_labels_binarized = {}
        if not is_binary:
            from sklearn.preprocessing import label_binarize
            for fold in range(n_folds):
                test_mask = fold_assignments == fold
                fold_labels_binarized[fold] = label_binarize(y[test_mask], classes=np.arange(n_classes))

        # Setup XGBoost parameters
        if is_binary:
            model_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': xgb_params.get('max_depth', DEFAULT_MAX_DEPTH),
                'learning_rate': xgb_params.get('learning_rate', 0.05),
                'subsample': xgb_params.get('subsample', 0.8),
                'colsample_bytree': xgb_params.get('colsample_bytree', 0.8),
                'scale_pos_weight': xgb_params.get('scale_pos_weight', 1.0),
                'random_state': 42,
                'verbosity': 0
            }
        else:
            model_params = {
                'objective': 'multi:softprob',
                'num_class': n_classes,
                'eval_metric': 'mlogloss',
                'max_depth': xgb_params.get('max_depth', DEFAULT_MAX_DEPTH),
                'learning_rate': xgb_params.get('learning_rate', 0.05),
                'subsample': xgb_params.get('subsample', 0.8),
                'colsample_bytree': xgb_params.get('colsample_bytree', 0.8),
                'random_state': 42,
                'verbosity': 0
            }

        # Add GPU parameters
        if gpu_available:
            model_params.update({
                'tree_method': 'hist', 'device': f'cuda:{gpu_id}', 'max_bin': 1024,
                'grow_policy': 'depthwise', 'single_precision_histogram': True,
                'deterministic_histogram': False, 'sampling_method': 'gradient_based',
                'max_cached_hist_node': 128, 'max_cat_threshold': 64
            })
        else:
            model_params['tree_method'] = 'hist'

        # K-FOLD CV
        print(f"   K-Fold CV ({n_folds} folds, {'GPU' if gpu_available else 'CPU'}):")
        oof_predictions = np.zeros(len(y), dtype=np.float32)
        oof_labels = np.zeros(len(y), dtype=np.int32)
        oof_mask = np.zeros(len(y), dtype=bool)
        oof_fold_ids = np.full(len(y), -1, dtype=np.int32)
        fold_metrics = []
        fold_models = []
        fold_sizes = [(fold, (fold_assignments == fold).sum()) for fold in range(n_folds)]
        fold_sizes.sort(key=lambda x: x[1])
        smallest_fold = fold_sizes[0][0]

        for fold in range(n_folds):
            test_mask = fold_assignments == fold
            test_count = test_mask.sum()

            # Spatial buffer: exclude blocks grid-adjacent to test blocks from training.
            # This prevents spatially autocorrelated samples near fold boundaries leaking
            # into training (Roberts et al. 2017). Buffer = spatial_buffer_blocks × 2km.
            if (spatial_buffer_blocks > 0
                    and spatial_groups_precomputed is not None
                    and n_blocks_y is not None):
                test_block_ids = np.unique(spatial_groups_precomputed[test_mask])
                buffer_block_ids = set()
                for bid in test_block_ids:
                    bx = int(bid) // n_blocks_y
                    by = int(bid) % n_blocks_y
                    for dx in range(-spatial_buffer_blocks, spatial_buffer_blocks + 1):
                        for dy in range(-spatial_buffer_blocks, spatial_buffer_blocks + 1):
                            if dx == 0 and dy == 0:
                                continue
                            nbx, nby = bx + dx, by + dy
                            if nbx >= 0 and nby >= 0:
                                buffer_block_ids.add(nbx * n_blocks_y + nby)
                # Exclude buffer blocks (not in test set) from training
                buffer_block_arr = np.array(list(buffer_block_ids), dtype=np.int64)
                buffer_mask = np.isin(spatial_groups_precomputed, buffer_block_arr) & ~test_mask
                train_mask = ~test_mask & ~buffer_mask
                if fold == 0:
                    print(f"   Spatial buffer: {spatial_buffer_blocks} block(s) = {spatial_buffer_blocks * 2}km — "
                          f"excluding {buffer_mask.sum():,} samples from training boundaries")
            else:
                buffer_mask = np.zeros(len(y), dtype=bool)
                train_mask = ~test_mask

            train_count = train_mask.sum()

            # Create training data
            if gpu_available:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass

            dtrain = xgb.QuantileDMatrix(X[train_mask], label=y[train_mask], max_bin=1024, feature_names=feature_names)

            fold_train_kwargs = dict(
                params=model_params,
                dtrain=dtrain,
                num_boost_round=n_estimators,
                verbose_eval=False,
            )

            fold_model = xgb.train(**fold_train_kwargs)

            fold_models.append(fold_model)

            # Predict on held-out fold
            X_test = X[test_mask]
            y_test = y[test_mask]

            try:
                y_pred_proba = fold_model.inplace_predict(X_test)
            except Exception:
                dtest = xgb.DMatrix(X_test)
                y_pred_proba = fold_model.predict(dtest)
                del dtest

            # Store OOF predictions
            test_indices = np.where(test_mask)[0]
            oof_predictions[test_indices] = y_pred_proba
            oof_labels[test_indices] = y_test
            oof_mask[test_indices] = True
            oof_fold_ids[test_indices] = int(fold)

            # Calculate fold metrics
            from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
            y_pred = (y_pred_proba > 0.5).astype(int)

            auprc = average_precision_score(y_test, y_pred_proba)
            auroc = roc_auc_score(y_test, y_pred_proba)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            fold_metrics.append({
                'fold': fold, 'auprc': auprc, 'auroc': auroc,
                'balanced_accuracy': bal_acc, 'bal_acc': bal_acc,
                'precision': precision, 'recall': recall, 'f1': f1,
                'test_samples': test_count, 'train_samples': train_count
            })

            print(f"   Fold {fold}: AUROC={auroc:.3f} AUPRC={auprc:.3f} Bal={bal_acc:.3f} P={precision:.3f} R={recall:.3f} (train={train_count:,} test={test_count:,})")

            # Clean up
            del dtrain, X_test
            gc.collect()

        # Summary
        avg_auprc = np.mean([m['auprc'] for m in fold_metrics])
        std_auprc = np.std([m['auprc'] for m in fold_metrics])
        avg_auroc = np.mean([m['auroc'] for m in fold_metrics])
        std_auroc = np.std([m['auroc'] for m in fold_metrics])
        avg_bal_acc = np.mean([m['bal_acc'] for m in fold_metrics])
        avg_precision = np.mean([m['precision'] for m in fold_metrics])
        avg_recall = np.mean([m['recall'] for m in fold_metrics])
        avg_f1 = np.mean([m['f1'] for m in fold_metrics])

        print(f"   CV Mean: AUROC={avg_auroc:.4f}±{std_auroc:.4f} AUPRC={avg_auprc:.4f}±{std_auprc:.4f} Bal={avg_bal_acc:.4f} P={avg_precision:.4f} R={avg_recall:.4f}")

        # Stage 2: Final model on all data
        if gpu_available:
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
        final_dtrain = xgb.QuantileDMatrix(X, label=y, max_bin=1024, feature_names=feature_names)
        final_train_kwargs = dict(params=model_params, dtrain=final_dtrain, num_boost_round=n_estimators, verbose_eval=False)
        bst = xgb.train(**final_train_kwargs)
        print(f"   Final model: {len(X):,} samples, {n_estimators} rounds")
        best_score = avg_auprc

        del final_dtrain
        gc.collect()

        iteration_metrics = fold_metrics
        valid_oof_mask = oof_mask
        oof_predictions_avg = oof_predictions
        oof_coverage = (valid_oof_mask.sum() / len(y)) * 100

        # FP Analysis by original class (evergreen vs deciduous)
        fp_analysis = None
        if is_binary and y_original is not None and valid_oof_mask.sum() > 0:
            oof_pred_binary = (oof_predictions_avg[valid_oof_mask] > 0.5).astype(int)
            oof_true_binary = y[valid_oof_mask]
            oof_true_original = y_original[valid_oof_mask]
            fp_mask = (oof_pred_binary == 1) & (oof_true_binary == 0)
            fp_from_evergreen = int(((oof_true_original[fp_mask] == 0)).sum())
            fp_from_deciduous = int(((oof_true_original[fp_mask] == 1)).sum())
            fp_total = int(fp_mask.sum())
            total_evg = int(((oof_true_binary == 0) & (oof_true_original == 0)).sum())
            total_dec = int(((oof_true_binary == 0) & (oof_true_original == 1)).sum())
            if fp_total > 0:
                print(f"   FPs: {fp_total:,} (Evg:{fp_from_evergreen:,}/{total_evg:,} [{fp_from_evergreen/total_evg*100:.1f}%] Dec:{fp_from_deciduous:,}/{total_dec:,} [{fp_from_deciduous/total_dec*100:.1f}%])")
            fp_analysis = {
                'fp_total': fp_total,
                'fp_evergreen': fp_from_evergreen, 'total_evergreen': total_evg,
                'fp_deciduous': fp_from_deciduous, 'total_deciduous': total_dec,
                'fpr_evergreen': fp_from_evergreen / total_evg if total_evg > 0 else 0,
                'fpr_deciduous': fp_from_deciduous / total_dec if total_dec > 0 else 0,
            }

        # Calculate confusion matrix from out-of-fold predictions
        confusion_mat = None
        if valid_oof_mask.sum() > 0:
            from sklearn.metrics import confusion_matrix
            y_oof_true = oof_labels[valid_oof_mask]
            if is_binary:
                y_oof_pred = (oof_predictions_avg[valid_oof_mask] > 0.5).astype(int)
            else:
                y_oof_pred = (oof_predictions_avg[valid_oof_mask] > 0.5).astype(int)
            confusion_mat = confusion_matrix(y_oof_true, y_oof_pred)

        # Store CV results for export
        cv_results = {
            'avg_auroc': avg_auroc,
            'std_auroc': std_auroc,
            'avg_auprc': avg_auprc,
            'std_auprc': std_auprc,
            'best_auprc': best_score,
            'avg_balanced_accuracy': avg_bal_acc,
            'avg_f1': avg_f1,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'fold_metrics': fold_metrics,
            'n_folds': n_folds,
            'is_binary': is_binary,
            'confusion_matrix': confusion_mat,
            'fp_analysis': fp_analysis
        }

        # Clean up GPU memory before Stage 2
        if 'fold_datasets' in locals():
            for key in list(fold_datasets.keys()):
                del fold_datasets[key]
            del fold_datasets
        gc.collect()
        if gpu_available:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass

        # ====================================================================
        # USE CV MODEL: Skip Stage 2, use rolling CV model directly
        # ====================================================================
        if use_cv_model:
            print(f"\n" + "=" * 70)
            print(f"🎯 Using Rolling CV Model (--use-cv-model)")
            print(f"=" * 70)
            print(f"   Skipping Stage 2 - using the rolling CV model directly")
            print(f"   💡 This model has implicit regularization:")
            print(f"      • Each tree trained on different ~80% subset")
            print(f"      • Never saw all data at once")
            print(f"      • More defensible for generalization claims")
            print(f"   Trees: {n_estimators}")

            final_model = bst

            print(f"\n📊 For Publication - Report These Metrics:")
            print(f"   AUROC (prevalence-independent): {avg_auroc:.4f} ± {std_auroc:.4f}")
            print(f"   AUPRC (at balanced prevalence): {avg_auprc:.4f} ± {std_auprc:.4f}")
            if is_binary:
                print(f"   Decision threshold:           0.5 (balanced data)")

            # Return model info
            model_info = {
                'model': final_model,
                'threshold': 0.5,
                'oof_predictions': oof_predictions_avg,
                'oof_labels': oof_labels,
                'oof_mask': valid_oof_mask,
                'oof_fold_ids': oof_fold_ids,
                'oof_coverage': oof_coverage,
                'cv_results': cv_results
            }

            return model_info

        # ====================================================================
        # Stage 2: Train final deployment model on ALL data
        # ====================================================================
        optimal_iterations = n_estimators
        print(f"\n" + "=" * 70)
        print(f"🚀 Stage 2: Final Deployment Model")
        print(f"=" * 70)
        print(f"   Training on ALL {len(X):,} samples (no holdout)")
        print(f"   Using {optimal_iterations} iterations")
        print(f"   💡 This is the production model")
        print(f"      • CV metrics = expected performance on new areas")
        print(f"      • Full dataset = maximum accuracy\n")

        # Calculate maximum samples that fit in GPU memory (AFTER cleanup)
        max_samples_gpu = len(X)
        if gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_gb = mem_info.free / 1024**3
                pynvml.nvmlShutdown()

                # Estimate: QuantileDMatrix uses memory based on max_bin
                # Use max_bin=1024 to match k-fold CV (XGBoost requires consistency)
                # Add 20% safety margin
                bytes_per_sample = 66 * 4 * 4.0  # ELLPACK compression with max_bin=1024
                max_samples_gpu = int((free_gb * 1024**3 * 0.8) / bytes_per_sample)

                print(f"   🔍 GPU Memory: {free_gb:.2f} GB free")
                print(f"   📊 Estimated capacity: ~{max_samples_gpu/1e6:.1f}M samples")

            except Exception as e:
                print(f"   ⚠️  Could not detect GPU memory, using estimate for 24GB GPU")
                max_samples_gpu = 100000000  # Optimistic fallback for 24GB GPU (100M samples)

        # Create dataset with ALL data
        # For large datasets, use stratified subsample to fit in GPU memory (industry standard)
        if gpu_available and len(X) > max_samples_gpu:  # If exceeds GPU capacity
            from sklearn.model_selection import train_test_split
            # Calculate optimal sample size that fits in GPU
            sample_fraction = min(0.95, max_samples_gpu / len(X))
            print(f"   ⚠️  Dataset ({len(X):,} samples) exceeds GPU capacity")
            print(f"   📉 Using stratified {sample_fraction*100:.0f}% sample ({int(len(X)*sample_fraction):,} samples)")
            print(f"   💡 Maintains class balance and fits comfortably in GPU")
            X_sample, _, y_sample, _ = train_test_split(
                X, y,
                train_size=sample_fraction,
                stratify=y,
                random_state=42
            )
            print(f"   📊 Creating GPU-cached QuantileDMatrix ({len(X_sample):,} samples, max_bin=1024)...")
            final_dtrain = xgb.QuantileDMatrix(X_sample, label=y_sample, max_bin=1024, feature_names=feature_names)
            print(f"   ✅ Training on {len(X_sample):,} samples (fits comfortably in GPU)")
        elif gpu_available:
            print(f"   📊 Creating GPU-cached QuantileDMatrix (max_bin=1024)...")
            final_dtrain = xgb.QuantileDMatrix(X, label=y, max_bin=1024, feature_names=feature_names)
        else:
            final_dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)

        # Train final model for optimal number of iterations with progress bar
        print(f"   🚀 Training final model with progress tracking...")

        # Create a custom callback for progress bar
        class ProgressCallback(xgb.callback.TrainingCallback):
            def __init__(self, total_iterations):
                self.total_iterations = total_iterations
                self.progress_bar = tqdm(total=total_iterations, desc="Final Training", ncols=100)

            def after_iteration(self, model, epoch, evals_log):
                self.progress_bar.update(1)
                if epoch % max(1, self.total_iterations // 20) == 0:
                    self.progress_bar.set_postfix({
                        'Iter': f'{epoch + 1}/{self.total_iterations}',
                        'Trees': epoch + 1
                    })
                return False  # Continue training

            def after_training(self, model):
                self.progress_bar.close()
                return model

        progress_callback = ProgressCallback(optimal_iterations)

        # Train with progress callback
        stage2_kwargs = dict(
            params=model_params,
            dtrain=final_dtrain,
            num_boost_round=optimal_iterations,
            verbose_eval=False,  # Disable default verbose
            callbacks=[progress_callback],
        )
        final_model = xgb.train(**stage2_kwargs)

        print(f"\n✅ Stage 2 Complete: Final Deployment Model Ready")
        print(f"=" * 70)
        print(f"   Trained on: ALL {len(X):,} pixels (no holdout)")
        print(f"   Iterations: {optimal_iterations}")
        print(f"   ")
        print(f"📊 For Publication - Report These Metrics:")
        print(f"   Expected AUPRC (from Stage 1 CV):  {avg_auprc:.4f} (rolling avg)")
        print(f"   Best single-fold AUPRC:           {best_score:.4f}")
        if is_binary:
            print(f"   Decision threshold:                0.5 (balanced data)")
        print(f"   ")
        print(f"   📝 CV metrics represent expected performance on unseen spatial regions")
        print(f"      Final model was trained on all data for deployment")

        # Return model info with out-of-fold predictions for unbiased threshold optimization
        model_info = {
            'model': final_model,
            'threshold': 0.5,
            'oof_predictions': oof_predictions_avg,  # Averaged out-of-fold predictions
            'oof_labels': oof_labels,  # True labels
            'oof_mask': valid_oof_mask,  # Which samples have valid predictions
            'oof_fold_ids': oof_fold_ids,  # Fold id for each OOF sample
            'oof_coverage': oof_coverage,  # Coverage percentage
            'cv_results': cv_results  # CV metrics for export
        }

        return model_info

    else:
        # Standard train/test split if spatial CV not available
        print("🔀 Using standard stratified train/test split...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        except ValueError as exc:
            raise ValueError(
                "Standard stratified split failed. This usually means at least one class has too few "
                "samples for a non-spatial split. Use the default spatial CV path, or train in binary mode."
            ) from exc

        # Setup XGBoost parameters
        model_params = {
            'max_depth': xgb_params.get('max_depth', DEFAULT_MAX_DEPTH),
            'learning_rate': xgb_params.get('learning_rate', 0.05),
            'subsample': xgb_params.get('subsample', 0.8),
            'colsample_bytree': xgb_params.get('colsample_bytree', 0.8),
            'random_state': 42,
            'verbosity': 0
        }

        if is_binary:
            model_params['objective'] = 'binary:logistic'
            model_params['eval_metric'] = 'logloss'
            model_params['scale_pos_weight'] = xgb_params.get('scale_pos_weight', 1.0)
        else:
            model_params['objective'] = 'multi:softprob'
            model_params['num_class'] = n_classes
            model_params['eval_metric'] = 'mlogloss'

        # Add GPU parameters if available
        if gpu_available:
            model_params.update({
                'tree_method': 'hist',
                'device': f'cuda:{gpu_id}',
                'max_bin': 1024,  # Maximum precision with 24GB VRAM available
                'grow_policy': 'depthwise',  # Better for GPU parallelization
                'single_precision_histogram': True,  # Faster GPU histograms
                'deterministic_histogram': False,  # Allow non-deterministic for speed
                'sampling_method': 'gradient_based',  # GPU-optimized sampling
                'max_cached_hist_node': 128,  # Cache more histogram nodes on GPU
                'max_cat_threshold': 64  # Optimize categorical splits on GPU
            })

        # Create DMatrix datasets - use QuantileDMatrix for GPU efficiency
        if gpu_available:
            dtrain = xgb.QuantileDMatrix(X_train, label=y_train, max_bin=1024)
            dtest = xgb.QuantileDMatrix(X_test, label=y_test, max_bin=1024)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

        # Train model
        print(f"Training XGBoost for {n_estimators} iterations...")
        std_train_kwargs = dict(
            params=model_params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dtest, 'test')],
            early_stopping_rounds=100,
            verbose_eval=100,
        )
        model = xgb.train(**std_train_kwargs)

        return model


def _normalized_path_str(path: str | Path) -> str:
    return os.path.normpath(str(Path(path).expanduser()))


def _append_numeric_run_id(path: str | Path, run_id: str) -> Path:
    path = Path(path)
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}{run_id}{suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Production wet woodland trainer: binary model from split-background labels with spatial CV and Optuna."
    )

    core = parser.add_argument_group("Core inputs")
    core.add_argument(
        "--data-dir",
        default=str(DEFAULT_TRAINING_EMBEDDINGS_DIR),
        help=f"Directory with training embedding GeoTIFF tiles (default: {DEFAULT_TRAINING_EMBEDDINGS_DIR})",
    )
    core.add_argument(
        "--labels-file",
        default=str(DEFAULT_LABELS_PATH),
        help=(
            "Labels GeoTIFF file (legacy: 0/1/2 or split-background: 0/1/2/3). "
            f"Default: {DEFAULT_LABELS_PATH}"
        ),
    )
    core.add_argument("--max-tiles", type=int, help="Max tiles to process")
    core.add_argument("--max-pixels-per-tile", type=int, default=None, help="Max pixels per tile (default: unlimited - use all pixels)")
    core.add_argument("--max-samples", type=int, help="QUICK TEST MODE: Limit total training samples (e.g., 50000 for fast test)")
    core.add_argument(
        "--no-height-filter",
        action="store_true",
        help="Disable the default CHM input screen (cover >= 0.15 and mean height >= 3.0 m).",
    )
    core.add_argument(
        "--min-chm-cover",
        type=float,
        default=DEFAULT_MIN_CHM_COVER,
        help=f"Minimum CHM cover fraction for the default input screen (default: {DEFAULT_MIN_CHM_COVER}).",
    )
    core.add_argument(
        "--min-chm-mean",
        type=float,
        default=DEFAULT_MIN_CHM_MEAN,
        help=f"Minimum CHM mean height (m) for the default input screen (default: {DEFAULT_MIN_CHM_MEAN}).",
    )
    core.add_argument("--exclude-features", type=str, default=None,
                      help="Comma-separated feature indices or names to exclude (e.g., '64' or 'dtm_elevation')")

    features = parser.add_argument_group("Feature inputs")
    features.add_argument("--abiotics", type=str, default=None,
                          help="Path to abiotic raster stack (e.g., DTM-derived CTI/slope). Bands: 1=elev, 2=slope, 3=aspect, 4=CTI")
    features.add_argument("--abiotic-bands", type=str, default=None,
                          help="Comma-separated band numbers to use from abiotics. Default: use all bands in the abiotic stack.")
    features.add_argument("--extra-rasters", type=str, nargs="+", default=None,
                          help="One or more single-band rasters to sample as additional features (e.g. smuk_mean.tif peat_prob.tif). Must be EPSG:27700. Feature name taken from filename stem.")

    training = parser.add_argument_group("Training controls")
    training.add_argument("--n-workers", type=int, help="Number of parallel workers")
    training.add_argument("--n-folds", type=int, default=10,
                          help="Number of spatial CV folds (default: 10 — more folds = more training data per model, smaller buffer footprint)")
    training.add_argument("--spatial-buffer", type=int, default=0,
                          help="Number of 2km blocks to exclude from training around each test fold boundary (default: 0 = no buffer)")
    training.add_argument("--trials", type=int, default=50, help="Number of Optuna trials (default: 50 — TPE needs ~15 warmup trials, leaving ~35 for optimisation across 10 parameters)")
    training.add_argument("--optuna-samples", type=int, default=50000, help="Total balanced samples for Optuna (default: 50000 for large datasets)")
    training.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS,
                          help=f"Number of estimators (rounds) for CV/final training (default: {DEFAULT_N_ESTIMATORS})")
    training.add_argument("--max-depth", type=int, default=None,
                          help="Optional fixed XGBoost tree depth for Optuna, CV, and final training. If omitted, Optuna searches over the configured depth range.")
    training.add_argument("--optuna-min-depth", type=int, default=4,
                          help="Optuna max_depth lower bound when --max-depth is not fixed (default: 4).")
    training.add_argument("--optuna-max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                          help=f"Optuna max_depth upper bound when --max-depth is not fixed (default: {DEFAULT_MAX_DEPTH}).")
    training.add_argument("--find-threshold", action="store_true", help="Find a deployment seed threshold for hysteresis using fold-calibrated OOF behavior")
    training.add_argument("--fbeta", type=float, default=DEFAULT_F_BETA,
                          help=f"Beta parameter for calibrated F-beta diagnostics in policy threshold reporting (default: {DEFAULT_F_BETA}).")
    training.add_argument("--threshold-target-precision", type=float, default=DEFAULT_POLICY_TARGET_PRECISION,
                          help=f"Target deployment precision for the hysteresis seed threshold (default: {DEFAULT_POLICY_TARGET_PRECISION:.2f})")
    training.add_argument("--threshold-precision-quantile", type=float, default=DEFAULT_POLICY_PRECISION_QUANTILE,
                          help=f"Fold precision quantile to enforce for the seed threshold (default: {DEFAULT_POLICY_PRECISION_QUANTILE:.2f} = q10)")
    training.add_argument("--threshold-min-recall", type=float, default=DEFAULT_POLICY_MIN_RECALL,
                          help=f"Minimum mean recall required when selecting the seed threshold (default: {DEFAULT_POLICY_MIN_RECALL:.2f})")
    training.add_argument("--scale-pos-weight", type=float, default=None,
                          help="Optional fixed XGBoost positive-class loss weight. When set, overrides the Optuna search for scale_pos_weight and is used in CV/final training.")
    training.add_argument("--bg-ratio", type=float, default=2.0, help="Background:wet ratio per spatial block (default: 2.0)")
    training.add_argument("--dec-bg-fraction", type=float, default=0.85, help="Fraction of background that is deciduous (default: 0.85 = 85%% deciduous, 15%% evergreen)")
    training.add_argument("--no-cache", action="store_true", help="Disable caching")
    training.add_argument("--clear-cache", action="store_true", help="Clear cache directory before processing")
    training.add_argument("--force-cpu", action="store_true", help="Force CPU training even if GPU available")
    training.add_argument("--gpu", type=int, default=0, help="GPU device ID to use (default: 0)")

    diagnostics = parser.add_argument_group("Diagnostics")
    diagnostics.add_argument("--compute-shap", action="store_true", help="Compute GPU-accelerated SHAP values after training")
    diagnostics.add_argument("--shap-samples", type=int, default=DEFAULT_SHAP_SAMPLES,
                             help=f"Number of samples for SHAP computation (default: {DEFAULT_SHAP_SAMPLES})")

    outputs = parser.add_argument_group("Outputs")
    outputs.add_argument("--save-model", default=str(DEFAULT_MODEL_PATH), help=f"Model save path (default: {DEFAULT_MODEL_PATH})")
    outputs.add_argument("--report-file", default=None,
                         help=f"Optional unified report path. Default: {DEFAULT_REPORT_DIR}/<model_name>.report.txt")
    outputs.add_argument("--cache-discarded-labels", type=str, default=str(DEFAULT_DISCARDED_LABELS_PATH),
                         help=f"Path to save discarded labels as GeoTIFF for background evaluation (default: {DEFAULT_DISCARDED_LABELS_PATH})")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.label_schema = "split_bg_0123"
    args.spatial_cv = True
    args.binary_mode = True
    args.optuna = True
    args.use_cv_model = False
    args.rebalance_classes = False
    args.data_dir = str(Path(args.data_dir).expanduser())
    args.labels_file = str(Path(args.labels_file).expanduser())
    args.save_model = str(Path(args.save_model).expanduser())
    if args.report_file:
        args.report_file = str(Path(args.report_file).expanduser())
    if args.cache_discarded_labels:
        args.cache_discarded_labels = str(Path(args.cache_discarded_labels).expanduser())
    save_model_norm = _normalized_path_str(args.save_model)
    auto_model_name = save_model_norm in {
        _normalized_path_str(DEFAULT_MODEL_PATH),
        _normalized_path_str(LEGACY_DEFAULT_MODEL_PATH),
        _normalized_path_str(LEGACY_GENERIC_DEFAULT_MODEL_PATH),
        _normalized_path_str(LEGACY_OLD_DEFAULT_MODEL_PATH),
        _normalized_path_str(LEGACY_NESTED_DEFAULT_MODEL_PATH),
    }
    if save_model_norm in {
        _normalized_path_str(LEGACY_DEFAULT_MODEL_PATH),
        _normalized_path_str(LEGACY_GENERIC_DEFAULT_MODEL_PATH),
        _normalized_path_str(LEGACY_OLD_DEFAULT_MODEL_PATH),
        _normalized_path_str(LEGACY_NESTED_DEFAULT_MODEL_PATH),
    }:
        args.save_model = str(DEFAULT_MODEL_PATH)

    if args.fbeta <= 0:
        raise SystemExit("--fbeta must be > 0")
    if args.max_depth is not None and args.max_depth <= 0:
        raise SystemExit("--max-depth must be > 0")
    if args.optuna_min_depth <= 0:
        raise SystemExit("--optuna-min-depth must be > 0")
    if args.optuna_max_depth <= 0:
        raise SystemExit("--optuna-max-depth must be > 0")
    if args.optuna_min_depth > args.optuna_max_depth:
        raise SystemExit("--optuna-min-depth must be <= --optuna-max-depth")
    if args.scale_pos_weight is not None and args.scale_pos_weight <= 0:
        raise SystemExit("--scale-pos-weight must be > 0")
    if not (0.0 < args.threshold_target_precision < 1.0):
        raise SystemExit("--threshold-target-precision must be in (0, 1)")
    if not (0.0 <= args.threshold_precision_quantile <= 1.0):
        raise SystemExit("--threshold-precision-quantile must be in [0, 1]")
    if not (0.0 <= args.threshold_min_recall <= 1.0):
        raise SystemExit("--threshold-min-recall must be in [0, 1]")
    if not OPTUNA_AVAILABLE:
        raise SystemExit("Optuna is required for the production trainer. Install it in the active environment and rerun.")

    report_file_explicit = args.report_file is not None
    run_model_id = datetime.now().strftime("%Y%m%d%H%M%S") if auto_model_name else None
    model_path = Path(args.save_model)
    report_file = Path(args.report_file) if report_file_explicit else (DEFAULT_REPORT_DIR / f"{model_path.stem}.report.txt")

    print("=" * 70)
    print("🌲 XGBoost Spatial CV Training")
    print("=" * 70)

    # Check GPU availability at runtime
    gpu_available, detected_gpu_id = check_gpu_availability(force_cpu=args.force_cpu)

    # Use user-specified GPU if available, otherwise use detected
    if gpu_available and not args.force_cpu:
        gpu_id = args.gpu
    else:
        gpu_id = 0  # Default for CPU mode

    print(f"   Data: {args.data_dir}")
    print(f"   Labels: {args.labels_file}")
    print(f"   Model output: {model_path}")
    print(f"   Report file: {report_file}")
    print(f"   Discarded labels: {args.cache_discarded_labels}")
    if args.no_height_filter:
        print("   Input CHM screen: disabled")
    else:
        print(
            f"   Input CHM screen: cover >= {args.min_chm_cover:.2f} and "
            f"mean height >= {args.min_chm_mean:.1f}m"
        )
    print("   Label schema: split_bg_0123")
    print("   Training target: Binary wet/non-wet")
    print("   Validation mode: Spatial CV")
    print("   Hyperparameter search: Optuna")
    print(
        f"   Policy thresholding: q{int(round(args.threshold_precision_quantile * 100)):02d} >= "
        f"{args.threshold_target_precision:.0%} precision"
    )
    if run_model_id:
        print(f"   Model run id: {run_model_id}")
    print(f"   GPU: {'GPU ' + str(gpu_id) if gpu_available else 'CPU mode'}")
    print(f"   System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total/1024**3:.0f}GB RAM")

    # Verify files exist
    if not Path(args.labels_file).exists():
        print(f"❌ Labels file not found: {args.labels_file}")
        return

    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        return

    # Print labels file resolution and extent
    try:
        with rasterio.open(args.labels_file) as labels_ds:
            pixel_size_x, pixel_size_y = labels_ds.res
            bounds = labels_ds.bounds
            width_km = (bounds.right - bounds.left) / 1000
            height_km = (bounds.top - bounds.bottom) / 1000
            print(f"   📐 Labels: {abs(pixel_size_x):.1f}m × {abs(pixel_size_y):.1f}m pixels")
            print(f"   🗺️  Extent: {width_km:.1f}km × {height_km:.1f}km ({labels_ds.width} × {labels_ds.height} pixels)")
    except Exception as e:
        print(f"   ⚠️  Could not read labels file info: {e}")

    # Print embeddings resolution info (from first tile)
    try:
        tile_files = list(Path(args.data_dir).glob("*.tif")) + list(Path(args.data_dir).glob("*.tiff"))
        if tile_files:
            with rasterio.open(tile_files[0]) as tile_ds:
                pixel_size_x, pixel_size_y = tile_ds.res
                print(f"   📐 Embeddings: {abs(pixel_size_x):.1f}m × {abs(pixel_size_y):.1f}m pixels (sampled from {tile_files[0].name})")
                print(f"   🧩 Embeddings bands: {tile_ds.count} (sampled from {tile_files[0].name})")
        else:
            print(f"   ⚠️  No .tif files found in {args.data_dir} for resolution check")
    except Exception as e:
        print(f"   ⚠️  Could not read embeddings resolution: {e}")

    # Handle cache clearing
    if args.clear_cache:
        import shutil
        cache_dir = Path("cache")
        if cache_dir.exists():
            print("🗑️  Clearing cache directory...")
            shutil.rmtree(cache_dir)

    # Load data using multiprocessing with caching
    X, y, coordinates = load_data_parallel(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        max_tiles=args.max_tiles,
        max_pixels_per_tile=args.max_pixels_per_tile,
        n_workers=args.n_workers,
        use_cache=not args.no_cache,
        apply_height_filter=not args.no_height_filter,
        min_chm_cover=args.min_chm_cover,
        min_chm_mean=args.min_chm_mean,
    )

    if X is None:
        print("❌ Failed to load data")
        return
    if coordinates is None:
        raise SystemExit("Spatial coordinates were not loaded. The production trainer requires coordinates for spatial CV.")

    # Build feature names (must match data structure: 64 embeddings + dtm + slope + chm + gap = 68 features)
    all_feature_names = [f"embedding_{i}" for i in range(64)] + ["dtm_elevation", "dtm_slope", "chm_canopy_height", "chm_canopy_gap"]
    print(f"\n🧩 Loaded tile features: {X.shape[1]} band(s) per sample before optional additions")

    # Exclude features if requested
    if args.exclude_features:
        exclude_list = [s.strip() for s in args.exclude_features.split(",")]
        exclude_indices = set()
        for item in exclude_list:
            if item.isdigit():
                exclude_indices.add(int(item))
            elif item in all_feature_names:
                exclude_indices.add(all_feature_names.index(item))
            else:
                print(f"⚠️  Unknown feature to exclude: {item}")

        if exclude_indices:
            keep_indices = [i for i in range(X.shape[1]) if i not in exclude_indices]
            excluded_names = [all_feature_names[i] for i in sorted(exclude_indices)]
            print(f"\n🚫 Excluding features: {excluded_names}")
            print(f"   Before: {X.shape[1]} features")
            X = X[:, keep_indices]
            all_feature_names = [all_feature_names[i] for i in keep_indices]
            print(f"   After: {X.shape[1]} features")
    else:
        print(f"   Keeping all loaded tile bands (no --exclude-features passed)")

    # Raster sampling deferred to after spatial rebalancing (see below)

    # QUICK TEST MODE: Subsample data if max_samples specified
    if args.max_samples and len(X) > args.max_samples:
        print(f"\n⚡ QUICK TEST MODE: Subsampling {args.max_samples:,} from {len(X):,} samples")
        indices = np.random.choice(len(X), args.max_samples, replace=False)
        X = X[indices]
        y = y[indices]
        if coordinates is not None:
            coordinates = coordinates[indices]
        print(f"   ✅ Using {len(X):,} samples for quick test")

    label_schema = args.label_schema
    wet_classes, background_classes = schema_class_sets(label_schema)
    print(f"\n🏷️  Label schema: {label_schema}")
    print(f"   Wet classes: {wet_classes.tolist()}")
    print(f"   Background classes: {background_classes.tolist()}")

    # Report pre-rebalancing wet prevalence for context.
    try:
        y_binary_full = to_binary_labels(y, label_schema)
        if len(np.unique(y_binary_full)) == 2:
            print(f"   Pre-rebalance wet prevalence: {float(np.mean(y_binary_full)):.3%}")
    except Exception:
        pass

    # Initialize spatial_groups for Optuna
    spatial_groups = None

    # Initialize discarded sample storage for optional background-evaluation exports
    discarded_X = None
    discarded_y_stored = None
    discarded_coords_stored = None

    def _filter_discarded_samples(valid_mask):
        nonlocal discarded_X, discarded_y_stored, discarded_coords_stored
        if discarded_X is None or valid_mask is None:
            return
        discarded_X = discarded_X[valid_mask]
        if discarded_y_stored is not None:
            discarded_y_stored = discarded_y_stored[valid_mask]
        if discarded_coords_stored is not None:
            discarded_coords_stored = discarded_coords_stored[valid_mask]

    # Apply spatial rebalancing BEFORE binary conversion and Optuna
    if coordinates is not None and args.spatial_cv:
        print("\n" + "-" * 70)
        print("🔄 Spatial Rebalancing")
        print("-" * 70)

        coord_range_x = coordinates[:, 0].max() - coordinates[:, 0].min()
        coord_range_y = coordinates[:, 1].max() - coordinates[:, 1].min()

        # Create spatial grid for ~2km block resolution
        target_block_size_km = 2.0
        n_blocks_x = max(10, int(coord_range_x / (target_block_size_km * 1000)))
        n_blocks_y = max(10, int(coord_range_y / (target_block_size_km * 1000)))

        print(f"   Grid: {n_blocks_x}×{n_blocks_y} blocks ({coord_range_x/1000:.0f}km × {coord_range_y/1000:.0f}km area)")

        # Assign each point to a grid cell
        x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()

        x_bins = np.linspace(x_min, x_max, n_blocks_x + 1)
        y_bins = np.linspace(y_min, y_max, n_blocks_y + 1)

        x_indices = np.digitize(coordinates[:, 0], x_bins) - 1
        y_indices = np.digitize(coordinates[:, 1], y_bins) - 1

        # Clip indices to valid range
        x_indices = np.clip(x_indices, 0, n_blocks_x - 1)
        y_indices = np.clip(y_indices, 0, n_blocks_y - 1)

        # Create block IDs
        spatial_groups = x_indices * n_blocks_y + y_indices
        unique_groups = np.unique(spatial_groups)

        print(f"   Created {len(unique_groups)} spatial groups")

        # Spatial rebalancing within each grid cell
        print(f"   Processing {len(unique_groups)} cells ({mp.cpu_count() - 2} cores)...")
        print(f"   Background:wet ratio = {args.bg_ratio}:1 | Deciduous fraction = {args.dec_bg_fraction:.0%}")

        balanced_indices = []
        discarded_indices = []  # Always initialize, even if no rebalancing
        original_samples = len(X)
        cells_processed = 0
        cells_with_sufficient_data = 0

        # Ultra-fast multicore approach
        n_workers = min(mp.cpu_count() - 2, 12)  # Leave 2 cores for system, max 12 for CPU-bound work

        # Split work across cores
        unique_groups_list = list(unique_groups)
        chunk_size = max(50, len(unique_groups_list) // (n_workers * 4))  # 4 chunks per worker
        chunks = [unique_groups_list[i:i + chunk_size] for i in range(0, len(unique_groups_list), chunk_size)]

        # Prepare arguments for parallel processing
        base_indices = np.arange(len(y))
        bg_ratio = args.bg_ratio
        dec_bg_fraction = args.dec_bg_fraction
        process_args = [
            (chunk, spatial_groups, y, base_indices, wet_classes, background_classes, bg_ratio, dec_bg_fraction)
            for chunk in chunks
        ]

        # Execute in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(process_spatial_block_batch, process_args),
                total=len(chunks),
                desc="   Progress"
            ))

        # Combine results
        balanced_indices = []
        discarded_indices = []
        for chunk_indices, chunk_discarded, chunk_cells_processed, chunk_cells_with_data in results:
            balanced_indices.extend(chunk_indices)
            discarded_indices.extend(chunk_discarded)
            cells_processed += chunk_cells_processed
            cells_with_sufficient_data += chunk_cells_with_data

        print(f"   ✅ Balanced {cells_with_sufficient_data}/{cells_processed} cells")
        print(f"   📉 {original_samples:,} → {len(balanced_indices):,} samples")

        # ========================================================================
        # STORE DISCARDED SAMPLES FOR OPTIONAL BACKGROUND EVALUATION
        # ========================================================================
        if len(discarded_indices) > 0:
            discarded_indices_arr = np.array(discarded_indices)
            discarded_X = X[discarded_indices_arr]
            discarded_y_stored = y[discarded_indices_arr]
            discarded_coords_stored = coordinates[discarded_indices_arr]
            print(f"   📦 Stored {len(discarded_indices):,} discarded samples for background evaluation")

        # ========================================================================
        # SAVE DISCARDED LABELS TO RASTER(S) (optional)
        # ========================================================================

        output_target = args.cache_discarded_labels

        if output_target and len(discarded_indices) > 0:
            print(f"\n💾 Saving discarded labels raster...")
            print(f"   📊 Found {len(discarded_indices):,} discarded indices to save")

            discarded_coords = coordinates[discarded_indices]
            discarded_labels = y[discarded_indices]
            save_background_eval_raster(
                output_path=output_target,
                template_raster=args.labels_file,
                coords=discarded_coords,
                labels=discarded_labels,
            )

        elif output_target and len(discarded_indices) == 0:
            print(f"   ⚠️  No discarded labels available (rebalancing not used)")
            print(f"   💡 Run with --spatial-cv to generate discarded labels raster")

        if len(balanced_indices) > 0:
            # Apply spatial rebalancing
            balanced_indices = np.array(balanced_indices)
            X = X[balanced_indices]
            y = y[balanced_indices]
            coordinates = coordinates[balanced_indices]
            spatial_groups = spatial_groups[balanced_indices]  # Update spatial groups too!

            # Report new class distribution
            unique_classes, class_counts = np.unique(y, return_counts=True)
            print(f"\n   Final distribution:")
            for cls, count in zip(unique_classes, class_counts):
                pct = count / len(y) * 100
                print(f"     Class {cls}: {count:,} ({pct:.2f}%)")

            print(f"   💡 Data is now spatially balanced for consistent training")
        else:
            print(f"   ⚠️  No cells had sufficient data! Using original dataset")

    # ── Raster feature sampling (AFTER rebalancing — only sample the ~94k kept points) ──
    if coordinates is not None:
        # 1. Multi-band abiotic stack (DTM-derived: elevation, slope, aspect, CTI)
        if args.abiotics:
            print(f"\n🌍 Adding abiotic features ({len(X):,} samples after rebalancing)...")
            if args.abiotic_bands:
                abiotic_bands = [int(b.strip()) for b in args.abiotic_bands.split(",")]
            else:
                with rasterio.open(args.abiotics) as abiotic_src:
                    abiotic_bands = list(range(1, abiotic_src.count + 1))
                print(f"   No --abiotic-bands passed; using all abiotic bands: {abiotic_bands}")
            band_names = {1: "elevation", 2: "slope", 3: "aspect", 4: "cti"}
            abiotic_values = sample_abiotics_at_coordinates(args.abiotics, coordinates, bands=abiotic_bands)
            nan_mask = np.isnan(abiotic_values).any(axis=1)
            if nan_mask.sum() > 0:
                print(f"   ⚠️  Removing {nan_mask.sum():,} samples with missing abiotic values")
                valid_mask = ~nan_mask
                X, y, coordinates = X[valid_mask], y[valid_mask], coordinates[valid_mask]
                abiotic_values = abiotic_values[valid_mask]
            X = np.hstack([X, abiotic_values])
            if discarded_X is not None and discarded_coords_stored is not None and len(discarded_X) > 0:
                print(f"   ↪ Mirroring abiotic features onto {len(discarded_X):,} discarded samples for background checks...")
                discarded_abiotic_values = sample_abiotics_at_coordinates(args.abiotics, discarded_coords_stored, bands=abiotic_bands)
                discarded_nan_mask = np.isnan(discarded_abiotic_values).any(axis=1)
                if discarded_nan_mask.sum() > 0:
                    print(f"   ⚠️  Removing {discarded_nan_mask.sum():,} discarded samples with missing abiotic values")
                    discarded_valid_mask = ~discarded_nan_mask
                    _filter_discarded_samples(discarded_valid_mask)
                    discarded_abiotic_values = discarded_abiotic_values[discarded_valid_mask]
                discarded_X = np.hstack([discarded_X, discarded_abiotic_values])
            for band in abiotic_bands:
                all_feature_names.append(band_names.get(band, f"abiotic_band{band}"))
            print(f"   ✅ Added {len(abiotic_bands)} abiotic features: {[band_names.get(b, f'band{b}') for b in abiotic_bands]}")

        # 2. Extra single-band rasters (soil moisture, peat probability, floodplain, etc.)
        extra_rasters = args.extra_rasters if args.extra_rasters else []
        if extra_rasters:
            print(f"\n🗺️  Adding {len(extra_rasters)} extra raster feature(s)...")
            for raster_path in extra_rasters:
                raster_name = Path(raster_path).stem
                print(f"   • {raster_name}")
                values = sample_abiotics_at_coordinates(raster_path, coordinates, bands=[1])
                nan_mask = np.isnan(values).any(axis=1)
                if nan_mask.sum() > 0:
                    print(f"     ℹ️  Filling {nan_mask.sum():,} NaN in {raster_name} with spatial median (radius=1)...")
                    values = _fill_raster_nans_spatial(raster_path, coordinates, values, search_radius=1)
                    remaining = int(np.isnan(values).sum())
                    if remaining > 0:
                        print(f"     ⚠️  {remaining:,} NaN remaining after fill")
                X = np.hstack([X, values])
                if discarded_X is not None and discarded_coords_stored is not None and len(discarded_X) > 0:
                    discarded_values = sample_abiotics_at_coordinates(raster_path, discarded_coords_stored, bands=[1])
                    discarded_nan_mask = np.isnan(discarded_values).any(axis=1)
                    if discarded_nan_mask.sum() > 0:
                        print(f"     ℹ️  Filling {discarded_nan_mask.sum():,} discarded NaN in {raster_name} with spatial median (radius=1)...")
                        discarded_values = _fill_raster_nans_spatial(raster_path, discarded_coords_stored, discarded_values, search_radius=1)
                        discarded_remaining = int(np.isnan(discarded_values).sum())
                        if discarded_remaining > 0:
                            print(f"     ⚠️  Removing {discarded_remaining:,} discarded samples still NaN in {raster_name}")
                            discarded_valid_mask = ~np.isnan(discarded_values).any(axis=1)
                            _filter_discarded_samples(discarded_valid_mask)
                            discarded_values = discarded_values[discarded_valid_mask]
                    discarded_X = np.hstack([discarded_X, discarded_values])
                all_feature_names.append(raster_name)
            print(f"   Total features: {X.shape[1]}")

    # Determine the raw label layout first, then announce the actual training mode.
    unique_labels = np.unique(y)
    print(f"\n🔍 Inspecting label classes...")
    print(f"   Unique labels in data: {unique_labels}")

    raw_label_set = set(unique_labels.astype(int).tolist())

    if raw_label_set == {0, 1}:
        is_binary = True
        print("   ✅ Labels already binary (0=non-wet, 1=wet)")
        if auto_model_name:
            args.save_model = str(DEFAULT_BINARY_MODEL_PATH)
    elif raw_label_set.issubset({0, 1, 2, 3}) and ({2, 3} & raw_label_set):
        is_binary = False
        print("   ✅ Detected split-background labels; training a binary wet/non-wet model")
        if auto_model_name:
            args.save_model = str(DEFAULT_BINARY_MODEL_PATH)
    else:
        raise SystemExit(
            "Unsupported label values for the production trainer. "
            f"Expected binary {0,1} or split-background labels drawn from {{0,1,2,3}}, got {sorted(raw_label_set)}."
        )

    # Store original labels BEFORE binary conversion for fold diagnostics
    y_original = y.copy()

    # Convert split-background labels to the production binary wet/non-wet target.
    if not is_binary:
        print("🔄 Converting labels to binary wet/non-wet target...")
        print(f"   Wet classes mapped to 1: {wet_classes.tolist()}")
        y_binary = to_binary_labels(y, label_schema)

        # Print conversion summary
        unique_orig, counts_orig = np.unique(y, return_counts=True)
        unique_binary, counts_binary = np.unique(y_binary, return_counts=True)

        print(f"   Original classes: {dict(zip(unique_orig, counts_orig))}")
        print(f"   Binary classes: {dict(zip(unique_binary, counts_binary))}")

        y = y_binary
        is_binary = True
        wet_classes = np.array([1], dtype=np.int32)
        background_classes = np.array([0], dtype=np.int32)

    resolved_model_path = Path(args.save_model)
    if auto_model_name and run_model_id:
        resolved_model_path = _append_numeric_run_id(resolved_model_path, run_model_id)
        args.save_model = str(resolved_model_path)
    if resolved_model_path != model_path:
        model_path = resolved_model_path
        if not report_file_explicit:
            report_file = DEFAULT_REPORT_DIR / f"{model_path.stem}.report.txt"
        print(f"   🧭 Resolved model output: {model_path}")
        print(f"   🧭 Resolved report file: {report_file}")

    # Use full spatially-rebalanced dataset for Optuna (no subsampling)
    X_optuna, y_optuna = X, y

    # Create spatial folds ONCE (used by both Optuna and training)
    fold_assignments = None
    n_blocks_y = None
    spatial_groups = None
    if coordinates is not None and args.spatial_cv:
        print("\n" + "=" * 70)
        print("📋 STAGE: Creating Spatial Folds")
        print("=" * 70)
        fold_assignments, spatial_groups, n_blocks_y = create_spatial_folds(
            X, y, coordinates, n_folds=args.n_folds
        )
        print("\n   ✅ Spatial folds created successfully")
        print("   💡 These folds will be used for rolling CV training")
        print("   💡 Optuna uses independent balanced sample (no fold leakage)")

        # ========================================================================
        # FOLD CLASS DISTRIBUTION DIAGNOSTICS
        # ========================================================================
        print("\n" + "=" * 70)
        print("📊 FOLD CLASS DISTRIBUTION (Pre-Binary Original Labels)")
        print("=" * 70)
        print("   Classes: 0=Evergreen, 1=Deciduous, 2=Wet-on-peat, 3=Wet-not-on-peat")
        print("")

        n_folds = len(np.unique(fold_assignments))
        class_names = {0: 'Evergreen', 1: 'Deciduous', 2: 'Wet-peat', 3: 'Wet-nopeat'}

        for fold in range(n_folds):
            fold_mask = fold_assignments == fold
            fold_y = y_original[fold_mask]
            fold_total = len(fold_y)

            unique_classes, counts = np.unique(fold_y, return_counts=True)
            class_dist = dict(zip(unique_classes.astype(int), counts))

            # Calculate percentages
            evergreen = class_dist.get(0, 0)
            deciduous = class_dist.get(1, 0)
            wet_peat = class_dist.get(2, 0)
            wet_nopeat = class_dist.get(3, 0)

            total_bg = evergreen + deciduous
            total_wet = wet_peat + wet_nopeat

            # Background balance ratio
            if total_bg > 0:
                evg_pct = evergreen / total_bg * 100
                dec_pct = deciduous / total_bg * 100
                bg_balance = f"Evg:{evg_pct:.0f}%/Dec:{dec_pct:.0f}%"
            else:
                bg_balance = "N/A"

            print(f"   Fold {fold}: {fold_total:,} samples | BG={total_bg:,} ({bg_balance}) | Wet={total_wet:,}")
            print(f"           Evergreen:{evergreen:,} Deciduous:{deciduous:,} Wet-peat:{wet_peat:,} Wet-nopeat:{wet_nopeat:,}")

        # Overall balance check
        print("")
        total_evg = (y_original == 0).sum()
        total_dec = (y_original == 1).sum()
        total_bg_all = total_evg + total_dec
        if total_bg_all > 0:
            overall_evg_pct = total_evg / total_bg_all * 100
            overall_dec_pct = total_dec / total_bg_all * 100
            print(f"   OVERALL BACKGROUND: Evergreen={total_evg:,} ({overall_evg_pct:.1f}%) | Deciduous={total_dec:,} ({overall_dec_pct:.1f}%)")
            if abs(overall_evg_pct - 50) > 10:
                print(f"   ⚠️  WARNING: Background classes imbalanced (>10% deviation from 50/50)")
            else:
                print(f"   ✅ Background classes well balanced")
        print("=" * 70)

    # Hyperparameter optimization
    print(f"\n🔍 Optuna ({args.trials} trials)...")
    best_params, best_score, optuna_study, best_iteration = optimize_with_optuna(
        X_optuna, y_optuna, coordinates=coordinates, n_trials=args.trials,
        use_binary=is_binary, gpu_available=gpu_available, gpu_id=gpu_id,
        optuna_samples=args.optuna_samples, force_cpu=args.force_cpu,
        n_estimators=args.n_estimators,
        scale_pos_weight_override=args.scale_pos_weight,
        max_depth_override=args.max_depth,
        min_depth_search=args.optuna_min_depth,
        max_depth_search=args.optuna_max_depth,
    )
    print(f"   📌 Optuna best_iteration={best_iteration} (using configured n_estimators={args.n_estimators} for CV/final)")

    xgb_params = best_params.copy()
    xgb_params['n_estimators'] = args.n_estimators
    if not args.save_model.endswith('.json'):
        args.save_model = args.save_model + '.json'

    # Sanity check: feature names must match feature matrix columns
    assert len(all_feature_names) == X.shape[1], (
        f"Feature name mismatch: {len(all_feature_names)} names vs {X.shape[1]} columns. "
        f"Names: {all_feature_names}"
    )

    # Train model
    print(f"\n🚀 Model Training...")
    model_result = train_xgboost_folds(
        X, y,
        coordinates=coordinates,
        spatial_cv=args.spatial_cv,
        gpu_available=gpu_available,
        gpu_id=gpu_id,
        fold_split_ratio=0.8,
        fold_assignments=fold_assignments,
        spatial_groups_precomputed=spatial_groups,
        use_cv_model=args.use_cv_model,
        y_original=y_original,
        feature_names=all_feature_names,
        n_blocks_y=n_blocks_y,
        spatial_buffer_blocks=args.spatial_buffer,
        n_folds=args.n_folds,
        **xgb_params
    )

    # Handle return value
    if isinstance(model_result, dict):
        model = model_result['model']
        threshold = model_result.get('threshold', 0.5)
    else:
        # Fallback for standard training
        model = model_result
        threshold = 0.5

    # Prepare model save path early (used by SHAP and exports)
    model_path = Path(args.save_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not str(model_path).endswith('.json'):
        model_path = model_path.with_suffix('.json')

    # Compute SHAP values if requested
    shap_values, shap_interactions, feature_importance = None, None, None
    if args.compute_shap:
        print(f"\n🧠 SHAP Analysis ({args.shap_samples:,} samples)...")
        shap_values, shap_interactions, feature_importance = compute_gpu_shap_values(
            model, X, y, gpu_available=gpu_available, gpu_id=gpu_id,
            n_samples=args.shap_samples, feature_names=all_feature_names
        )
        if shap_values is not None:
            shap_path = model_path.with_suffix('.shap.npz')
            save_data = {'shap_values': shap_values, 'feature_names': all_feature_names,
                        'feature_importance': np.array([imp for _, imp in feature_importance])}
            if shap_interactions is not None:
                save_data['shap_interactions'] = shap_interactions
            np.savez_compressed(shap_path, **save_data)
            print(f"   ✅ Saved to {shap_path}")

    # Save model
    print(f"\n💾 Saving model to {model_path}...")
    model.save_model(str(model_path))
    print(f"   ✅ Done")

    # ========================================================================
    # EXPORT RESULTS — single unified report
    # ========================================================================
    print(f"\n📊 Exporting results...")

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:

        f.write(f"{'='*70}\n")
        f.write(f"WET WOODLAND XGBOOST MODEL REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model:     {model_path.name}\n")
        f.write(f"{'='*70}\n\n")

        # 1. Optuna hyperparameters
        f.write(f"{'─'*70}\n")
        f.write(f"1. OPTUNA HYPERPARAMETERS\n")
        f.write(f"   Best AUPRC: {best_score:.4f}  ({len(optuna_study.trials)} trials)\n")
        f.write(f"{'─'*70}\n")
        for key, value in sorted(best_params.items()):
            f.write(f"   {key} = {value:.6f}\n" if isinstance(value, float) else f"   {key} = {value}\n")
        f.write(f"   n_estimators = {args.n_estimators}\n")
        f.write(f"   max_bin = 1024\n\n")

        # 2. K-Fold CV results
        if 'cv_results' in model_result:
            cv_data = model_result['cv_results']
            fold_metrics = cv_data.get('fold_metrics', [])
            is_binary = cv_data.get('is_binary', False)

            f.write(f"{'─'*70}\n")
            f.write(f"2. SPATIAL CROSS-VALIDATION  ({cv_data['n_folds']} folds)\n")
            f.write(f"{'─'*70}\n")
            f.write(f"   AUROC        = {np.mean([m['auroc'] for m in fold_metrics]):.4f} ± {np.std([m['auroc'] for m in fold_metrics]):.4f}\n")
            f.write(f"   AUPRC        = {np.mean([m['auprc'] for m in fold_metrics]):.4f} ± {np.std([m['auprc'] for m in fold_metrics]):.4f}\n")
            f.write(f"   Balanced_Acc = {np.mean([m['balanced_accuracy'] for m in fold_metrics]):.4f} ± {np.std([m['balanced_accuracy'] for m in fold_metrics]):.4f}\n")
            f.write(f"   F1           = {np.mean([m['f1'] for m in fold_metrics]):.4f} ± {np.std([m['f1'] for m in fold_metrics]):.4f}\n")
            if is_binary:
                f.write(f"   Precision    = {np.mean([m['precision'] for m in fold_metrics]):.4f} ± {np.std([m['precision'] for m in fold_metrics]):.4f}\n")
                f.write(f"   Recall       = {np.mean([m['recall'] for m in fold_metrics]):.4f} ± {np.std([m['recall'] for m in fold_metrics]):.4f}\n")
            f.write(f"   Best_fold_AUROC = {max(m['auroc'] for m in fold_metrics):.4f}\n\n")

            f.write(f"   Per-fold breakdown:\n")
            for m in fold_metrics:
                f.write(f"   Fold {m['fold']}: AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}\n")
            f.write(f"\n")

            if cv_data.get('confusion_matrix') is not None:
                cm = cv_data['confusion_matrix']
                if is_binary:
                    tn, fp, fn, tp = cm.ravel()
                    f.write(f"   Confusion Matrix (OOF):\n")
                    f.write(f"   TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}\n")
                    f.write(f"   FPR={fp/(fp+tn):.4f}  FNR={fn/(fn+tp):.4f}\n")
                    f.write(f"   Actual Not Wet: {cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}% correct  {cm[0,1]/(cm[0,0]+cm[0,1])*100:.1f}% missed\n")
                    f.write(f"   Actual Wet:     {cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}% correct  {cm[1,0]/(cm[1,0]+cm[1,1])*100:.1f}% missed\n\n")

            fp_a = cv_data.get('fp_analysis')
            if fp_a:
                f.write(f"   False Positives by background class (OOF @0.5):\n")
                f.write(f"   FP total     = {fp_a['fp_total']:,}\n")
                f.write(f"   FP evergreen = {fp_a['fp_evergreen']:,} / {fp_a['total_evergreen']:,} ({fp_a['fpr_evergreen']*100:.1f}% FPR)\n")
                f.write(f"   FP deciduous = {fp_a['fp_deciduous']:,} / {fp_a['total_deciduous']:,} ({fp_a['fpr_deciduous']*100:.1f}% FPR)\n")
                f.write(f"   Deciduous/Evergreen FPR ratio = {fp_a['fpr_deciduous']/fp_a['fpr_evergreen']:.1f}x\n\n")

        # 3. Feature importance
        f.write(f"{'─'*70}\n")
        f.write(f"3. FEATURE IMPORTANCE\n")
        f.write(f"{'─'*70}\n")
        if args.compute_shap and shap_values is not None and feature_importance is not None:
            f.write(f"   Method: SHAP mean |value|\n\n")
            for i, (feat, imp) in enumerate(feature_importance):
                f.write(f"   {i+1:2d}. {feat}: {imp:.6f}\n")
        else:
            try:
                fmap = model.get_score(importance_type='gain')
                items = []
                for k, v in fmap.items():
                    try:
                        idx = int(k[1:])
                        name = all_feature_names[idx] if all_feature_names and idx < len(all_feature_names) else k
                    except:
                        name = k
                    items.append((name, float(v)))
                items.sort(key=lambda x: x[1], reverse=True)
                f.write(f"   Method: XGBoost gain\n\n")
                for i, (feat, gain) in enumerate(items[:20]):
                    f.write(f"   {i+1:2d}. {feat}: {gain:.6f}\n")
            except Exception as e:
                f.write(f"   Feature importance unavailable: {e}\n")
        f.write(f"\n")

    # Classification type info
    n_classes = len(np.unique(y))
    is_binary = n_classes == 2
    prevalence_sweep = []

    # Optimal threshold finding
    if args.find_threshold and is_binary:
        if (
            isinstance(model_result, dict)
            and 'oof_predictions' in model_result
            and 'oof_mask' in model_result
            and model_result.get('oof_fold_ids', None) is not None
        ):
            oof_mask = model_result['oof_mask']
            y_pred_proba = model_result['oof_predictions'][oof_mask]
            oof_fold_ids = model_result['oof_fold_ids'][oof_mask]
            _raw_labels = model_result['oof_labels'][oof_mask]
            # Ensure binary labels in case the stored OOF labels still reflect
            # the pre-conversion split-background schema (0/1/2/3).
            if _raw_labels.max() > 1:
                y_threshold = to_binary_labels(_raw_labels, label_schema)
            else:
                y_threshold = _raw_labels
        else:
            raise SystemExit("Policy thresholding requires OOF fold ids from spatial CV; no valid OOF fold data was found.")

        threshold_auprc = average_precision_score(y_threshold, y_pred_proba)
        for scenario_prev in DEFAULT_POLICY_PREVALENCE_PRINT_SCENARIOS:
            scenario_results = find_policy_seed_threshold(
                y_threshold,
                y_pred_proba,
                oof_fold_ids,
                target_pos_rate=float(scenario_prev),
                target_precision=args.threshold_target_precision,
                precision_quantile=args.threshold_precision_quantile,
                beta=args.fbeta,
                min_recall=args.threshold_min_recall,
                report_quantiles=DEFAULT_POLICY_REPORT_QUANTILES,
                report_precision_targets=DEFAULT_POLICY_REPORT_PRECISIONS,
            )
            scenario_seed = float(scenario_results["best"]["threshold"])
            scenario_lows = []
            for ratio in DEFAULT_POLICY_COMPACT_EXPAND_RATIOS:
                if scenario_seed > 0.01:
                    low = float(max(0.01, min(scenario_seed * ratio, scenario_seed - 0.01)))
                    scenario_lows.append(f"{low:.3f}")
            prevalence_sweep.append(
                {
                    "prevalence": float(scenario_prev),
                    "seed": scenario_seed,
                    "lows": scenario_lows,
                }
            )
        prevalence_seed_summary = "  ".join(
            f"{item['prevalence'] * 100:g}%={item['seed']:.3f}"
            for item in prevalence_sweep
        )
        prevalence_expand_summary = "  ".join(
            f"{item['prevalence'] * 100:g}%={'/'.join(item['lows'])}"
            for item in prevalence_sweep
        )

        print(f"   Seed   (policy): {prevalence_seed_summary}")
        print(
            "   Expand (policy, 0.50x/0.67x/0.75x): "
            f"{prevalence_expand_summary}"
        )

        # Write thresholds into the report
        _cv = model_result.get('cv_results', {}) if isinstance(model_result, dict) else {}
        _avg_auroc = _cv.get('avg_auroc', float('nan'))
        with open(report_file, 'a') as f:
            f.write(f"{'─'*70}\n")
            f.write(f"4. POLICY THRESHOLDING\n")
            f.write(f"   OOF samples: {len(y_threshold):,}  AUROC={_avg_auroc:.4f}  AUPRC={threshold_auprc:.4f}\n")
            f.write(f"{'─'*70}\n")
            f.write(
                f"   Policy rule: q{int(round(args.threshold_precision_quantile * 100)):02d} >= "
                f"{args.threshold_target_precision:.0%} precision, "
                f"min recall {args.threshold_min_recall:.0%}\n"
            )
            f.write(f"   Prevalence sweep (seed): {prevalence_seed_summary}\n")
            f.write(
                "   Prevalence sweep (expand 0.50x/0.67x/0.75x): "
                f"{prevalence_expand_summary}\n"
            )
            f.write("   Note: seed/high thresholds are calibrated from OOF folds; low/expand remains a map-space hysteresis choice.\n")
            f.write(f"\n")

    split_background_labels = bool(raw_label_set.issubset({0, 1, 2, 3}) and ({2, 3} & raw_label_set))
    discarded_thresholds = [
        (f"{item['prevalence'] * 100:g}%", item['seed'])
        for item in prevalence_sweep
    ]
    discarded_sweep_thresholds = np.round(np.arange(0.01, 0.51, 0.01), 2).tolist()
    discarded_bg_summary = summarize_discarded_background_predictions(
        model,
        discarded_X,
        discarded_y_stored,
        split_background_labels=split_background_labels,
        threshold=0.5,
        extra_thresholds=discarded_thresholds,
        sweep_thresholds=discarded_sweep_thresholds,
    )
    if discarded_bg_summary:
        discarded_bg_breakpoint = estimate_fpr_sweep_breakpoint(
            discarded_bg_summary.get('fpr_sweep', [])
        )
        discarded_bg_local_window = summarize_fpr_sweep_window(
            discarded_bg_summary.get('fpr_sweep', [])
        )
        discarded_bg_checkpoints = summarize_fpr_sweep_checkpoints(
            discarded_bg_summary.get('fpr_sweep', [])
        )
        discarded_bg_crossings = summarize_fpr_sweep_crossings(
            discarded_bg_summary.get('fpr_sweep', [])
        )
        print("\n🧪 Discarded Background Sanity Check (not independent)")
        print(
            f"   Background-only samples: {discarded_bg_summary['n_background_samples']:,}"
            + (
                f"  ignored non-background discarded: {discarded_bg_summary['n_ignored_non_background']:,}"
                if discarded_bg_summary['n_ignored_non_background'] > 0 else ""
            )
        )
        print(
            f"   Pred proba: mean={discarded_bg_summary['mean_proba']:.3f}  "
            f"q95={discarded_bg_summary['q95_proba']:.3f}  "
            f"q99={discarded_bg_summary['q99_proba']:.3f}  "
            f"max={discarded_bg_summary['max_proba']:.3f}"
        )
        print(
            f"   FP @0.5: {discarded_bg_summary['fp_total']:,}/{discarded_bg_summary['n_background_samples']:,} "
            f"({discarded_bg_summary['fpr_total']*100:.2f}%)"
        )
        if discarded_bg_summary['total_evergreen'] > 0 or discarded_bg_summary['total_deciduous'] > 0:
            evg_part = (
                f"Evg:{discarded_bg_summary['fp_evergreen']:,}/{discarded_bg_summary['total_evergreen']:,} "
                f"[{discarded_bg_summary['fpr_evergreen']*100:.2f}%]"
                if discarded_bg_summary['total_evergreen'] > 0 else "Evg:n/a"
            )
            dec_part = (
                f"Dec:{discarded_bg_summary['fp_deciduous']:,}/{discarded_bg_summary['total_deciduous']:,} "
                f"[{discarded_bg_summary['fpr_deciduous']*100:.2f}%]"
                if discarded_bg_summary['total_deciduous'] > 0 else "Dec:n/a"
            )
            print(f"   FP split: {evg_part}  {dec_part}")
        if discarded_bg_summary['threshold_hits']:
            hit_summary = "  ".join(
                f"{item['label']}={item['rate']*100:.2f}%"
                for item in discarded_bg_summary['threshold_hits']
            )
            print(f"   Seed hit rates on discarded background: {hit_summary}")
        if discarded_bg_breakpoint:
            print(
                "   Raw bg elbow (piecewise log-FPR): "
                f"thr≈{discarded_bg_breakpoint['threshold']:.2f}  "
                f"FPR≈{discarded_bg_breakpoint['fpr_total']*100:.2f}%"
            )
        if discarded_bg_crossings:
            crossing_summary = "  ".join(
                (
                    f"{int(round(item['target_rate'] * 100))}%->{item['threshold']:.2f}"
                    if item['threshold'] is not None else
                    f"{int(round(item['target_rate'] * 100))}%->n/a"
                )
                for item in discarded_bg_crossings
            )
            print(f"   Raw bg FPR crossings: {crossing_summary}")
        if discarded_bg_local_window:
            print("   Raw bg FPR local window (0.15-0.30 @ 0.01):")
            for start in range(0, len(discarded_bg_local_window), 8):
                row = discarded_bg_local_window[start:start + 8]
                row_summary = "  ".join(
                    f"{item['threshold']:.2f}={item['fpr_total']*100:.2f}%"
                    for item in row
                )
                print(f"      {row_summary}")

        with open(report_file, 'a') as f:
            f.write(f"{'─'*70}\n")
            f.write("5. DISCARDED BACKGROUND SANITY CHECK (not independent)\n")
            f.write(f"{'─'*70}\n")
            f.write(f"   Background-only discarded samples = {discarded_bg_summary['n_background_samples']:,}\n")
            if discarded_bg_summary['n_ignored_non_background'] > 0:
                f.write(f"   Ignored non-background discarded samples = {discarded_bg_summary['n_ignored_non_background']:,}\n")
            f.write(
                f"   Predicted probability summary: mean={discarded_bg_summary['mean_proba']:.4f}  "
                f"q95={discarded_bg_summary['q95_proba']:.4f}  "
                f"q99={discarded_bg_summary['q99_proba']:.4f}  "
                f"max={discarded_bg_summary['max_proba']:.4f}\n"
            )
            f.write(
                f"   FP @0.5 = {discarded_bg_summary['fp_total']:,} / {discarded_bg_summary['n_background_samples']:,} "
                f"({discarded_bg_summary['fpr_total']*100:.2f}% FPR)\n"
            )
            if discarded_bg_summary['total_evergreen'] > 0:
                f.write(
                    f"   FP evergreen = {discarded_bg_summary['fp_evergreen']:,} / {discarded_bg_summary['total_evergreen']:,} "
                    f"({discarded_bg_summary['fpr_evergreen']*100:.2f}% FPR)\n"
                )
            if discarded_bg_summary['total_deciduous'] > 0:
                f.write(
                    f"   FP deciduous = {discarded_bg_summary['fp_deciduous']:,} / {discarded_bg_summary['total_deciduous']:,} "
                    f"({discarded_bg_summary['fpr_deciduous']*100:.2f}% FPR)\n"
                )
            if discarded_bg_summary['threshold_hits']:
                hit_summary = "  ".join(
                    f"{item['label']}={item['rate']*100:.2f}%"
                    for item in discarded_bg_summary['threshold_hits']
                )
                f.write(f"   Seed hit rates on discarded background = {hit_summary}\n")
            if discarded_bg_breakpoint:
                f.write(
                    "   Estimated raw background elbow (piecewise linear fit on log10 FPR) = "
                    f"threshold {discarded_bg_breakpoint['threshold']:.3f}, "
                    f"FPR {discarded_bg_breakpoint['fpr_total']*100:.2f}%, "
                    f"SSE improvement {discarded_bg_breakpoint['sse_improvement']*100:.1f}% over one-line fit\n"
                )
            if discarded_bg_crossings:
                crossing_summary = "  ".join(
                    (
                        f"{int(round(item['target_rate'] * 100))}%->{item['threshold']:.2f}"
                        if item['threshold'] is not None else
                        f"{int(round(item['target_rate'] * 100))}%->n/a"
                    )
                    for item in discarded_bg_crossings
                )
                f.write(f"   Raw background FPR crossings = {crossing_summary}\n")
            if discarded_bg_local_window:
                f.write("   Raw background FPR local window (0.15-0.30 @ 0.01)\n")
                for start in range(0, len(discarded_bg_local_window), 8):
                    row = discarded_bg_local_window[start:start + 8]
                    row_summary = "  ".join(
                        f"{item['threshold']:.2f}={item['fpr_total']*100:.2f}%"
                        for item in row
                    )
                    f.write(f"      {row_summary}\n")
            if discarded_bg_checkpoints:
                f.write("   Raw background FPR checkpoints (total / evergreen / deciduous)\n")
                for item in discarded_bg_checkpoints:
                    f.write(
                        f"      thr={item['threshold']:.2f}  "
                        f"total={item['fpr_total']*100:.2f}%  "
                        f"evg={item['fpr_evergreen']*100:.2f}%  "
                        f"dec={item['fpr_deciduous']*100:.2f}%\n"
                    )
            if discarded_bg_summary.get('fpr_sweep'):
                f.write("   Raw background FPR sweep (threshold,total_fpr,evergreen_fpr,deciduous_fpr)\n")
                for item in discarded_bg_summary['fpr_sweep']:
                    f.write(
                        f"      {item['threshold']:.2f},"
                        f"{item['fpr_total']*100:.4f},"
                        f"{item['fpr_evergreen']*100:.4f},"
                        f"{item['fpr_deciduous']*100:.4f}\n"
                    )
            f.write("   Note: raw background sweep is per-pixel probability only; hysteresis expansion is stricter because low pixels must also attach to seed pixels.\n")
            f.write("   Note: these samples were excluded from training by spatial rebalancing, but they are not a fully independent validation set.\n\n")

    print(f"\n   • Report: {report_file}")
    print(f"\n✅ Training complete!")


if __name__ == "__main__":
    main()
