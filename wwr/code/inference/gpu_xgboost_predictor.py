#!/usr/bin/env python3
"""
GPU-Accelerated XGBoost Predictor
Uses GPU for prediction to dramatically speed up large-scale inference.
"""

import numpy as np
import rasterio
from pathlib import Path
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob
import gc
import os
import json
import re
import traceback
import warnings

# Thread control for CPU components
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Suppress XGBoost device mismatch warnings (predictions still use GPU)
warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')

# Global model cache (loaded once per worker process)
_worker_model = None
_worker_gpu_id = None
_worker_has_cupy = None
_worker_num_features = None


def resolve_default_model_path(models_dir):
    """Return the newest model in the canonical models dir."""
    models_dir = Path(models_dir)
    candidates = [
        p for p in models_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".model"}
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No .json or .model files found in {models_dir}. "
            "Pass --model explicitly or add a model to the canonical models directory."
        )

    def sort_key(path: Path):
        match = re.search(r"(\d+)$", path.stem)
        numeric_id = int(match.group(1)) if match else -1
        stat = path.stat()
        return (numeric_id, stat.st_mtime_ns, path.name)

    latest = max(candidates, key=sort_key)
    if len(candidates) > 1:
        print(f"🧭 Multiple models found in {models_dir}; using newest: {latest.name}")
    return latest


def inspect_model_schema(model_path):
    """Best-effort inspection of model feature count/names for startup diagnostics."""
    model_info = {"num_feature": None, "feature_names": None}
    model_path = Path(model_path)

    if model_path.suffix.lower() == ".json":
        try:
            model_data = json.loads(model_path.read_text())
            learner = model_data.get("learner", {})
            learner_model_param = learner.get("learner_model_param", {})
            num_feature = learner_model_param.get("num_feature")
            if num_feature is not None:
                model_info["num_feature"] = int(num_feature)

            feature_names = learner.get("feature_names")
            if isinstance(feature_names, list) and feature_names:
                model_info["feature_names"] = feature_names

            return model_info
        except Exception:
            pass

    try:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        model_info["num_feature"] = booster.num_features()
        if getattr(booster, "feature_names", None):
            model_info["feature_names"] = list(booster.feature_names)
    except Exception:
        pass

    return model_info


def feature_names_for_tile(n_tile_bands):
    """Feature names for supported tile layouts."""
    embeddings = [f"embedding_{i}" for i in range(64)]
    if n_tile_bands == 66:
        return embeddings + ["dtm_elevation", "chm_canopy_height"]
    if n_tile_bands == 67:
        return embeddings + ["dtm_elevation", "chm_canopy_height", "chm_canopy_gap"]
    if n_tile_bands == 68:
        return embeddings + ["dtm_elevation", "dtm_slope", "chm_canopy_height", "chm_canopy_gap"]
    return embeddings + [f"feature_{i}" for i in range(64, n_tile_bands)]


def resolve_exclude_indices(exclude_specs, n_tile_bands):
    """Resolve numeric/name exclusions against the current tile layout."""
    if not exclude_specs:
        return set()

    resolved = set()
    tile_feature_names = feature_names_for_tile(n_tile_bands)
    for spec in exclude_specs:
        if isinstance(spec, int):
            if 0 <= spec < n_tile_bands:
                resolved.add(spec)
        elif spec in tile_feature_names:
            resolved.add(tile_feature_names.index(spec))
    return resolved


def _init_worker(model_path, gpu_id):
    """Initialize worker with pre-loaded model (called once per worker process)."""
    global _worker_model, _worker_gpu_id, _worker_has_cupy, _worker_num_features
    import xgboost as xgb

    _worker_model = xgb.Booster()
    _worker_model.load_model(model_path)
    _worker_model.set_param({"device": f"cuda:{gpu_id}"})
    _worker_gpu_id = gpu_id
    _worker_num_features = _worker_model.num_features()

    # Check CuPy availability once
    try:
        import cupy as cp
        _worker_has_cupy = True
    except ImportError:
        _worker_has_cupy = False


def predict_tile_gpu_worker(args):
    """GPU-accelerated worker for single tile prediction."""
    data_file, output_file, threshold, exclude_specs, expected_bands = args
    global _worker_model, _worker_has_cupy, _worker_num_features
    tile_name = Path(data_file).name

    try:
        import xgboost as xgb
        model = _worker_model

        with rasterio.open(data_file) as src:
            # Read all bands
            data = src.read()  # Shape: (bands, height, width)

            n_tile_bands = data.shape[0]
            valid_exclude = resolve_exclude_indices(exclude_specs, n_tile_bands)
            effective_feature_count = n_tile_bands - len(valid_exclude)
            target_feature_count = _worker_num_features if _worker_num_features is not None else expected_bands

            if target_feature_count is not None and effective_feature_count != target_feature_count:
                exclude_suffix = ""
                if valid_exclude:
                    exclude_suffix = f" ({effective_feature_count} after exclusions)"
                return {
                    "success": False,
                    "file": tile_name,
                    "reason": (
                        f"Feature count mismatch: tile has {n_tile_bands} band(s){exclude_suffix}, "
                        f"model expects {target_feature_count} feature(s)"
                    ),
                    "traceback": None,
                }

            # Exclude features — only exclude indices that exist in this tile
            if valid_exclude:
                keep_indices = [i for i in range(n_tile_bands) if i not in valid_exclude]
                data = data[keep_indices, :, :]

            _expected = target_feature_count if target_feature_count is not None else effective_feature_count
            if data.shape[0] != _expected:
                return {
                    "success": False,
                    "file": tile_name,
                    "reason": (
                        f"Feature count mismatch after exclusions: got {data.shape[0]}, "
                        f"expected {_expected}"
                    ),
                    "traceback": None,
                }

            # Find valid pixels
            data_valid = np.isfinite(data).all(axis=0)

            if data_valid.sum() == 0:
                return {
                    "success": False,
                    "file": tile_name,
                    "reason": "No valid pixels found in tile",
                    "traceback": None,
                }

            # Extract valid pixels
            features = data[:, data_valid].T  # Shape: (n_pixels, n_bands)

            # Clear data array to save memory
            del data
            gc.collect()

            # GPU-accelerated prediction with XGBoost optimizations
            try:
                if _worker_has_cupy:
                    import cupy as cp
                    # Move data to GPU using CuPy
                    gpu_features = cp.asarray(features)
                    # Use inplace_predict with GPU data
                    predictions_proba = model.inplace_predict(gpu_features)
                    # Convert back to numpy if needed
                    if hasattr(predictions_proba, 'get'):
                        predictions_proba = predictions_proba.get()
                    del gpu_features
                else:
                    # CuPy not available, use inplace_predict with numpy (avoids feature name check)
                    predictions_proba = model.inplace_predict(features)

                gc.collect()

            except Exception as pred_error:
                print(f"⚠️ GPU prediction error, falling back to CPU: {pred_error}")
                # CPU fallback — inplace_predict accepts numpy directly
                predictions_proba = model.inplace_predict(features)

            # Always create probability output band
            prob_output = np.full((src.height, src.width), 255.0, dtype=np.float32)
            prob_output[data_valid] = predictions_proba

            # Save with compression
            profile = src.profile.copy()
            profile.update({
                'dtype': 'float32',
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'nodata': 255.0
            })

            # Default: probability-only output (single band).
            # If threshold is provided, write binary + probability (2 bands).
            if threshold is None:
                profile.update({'count': 1})
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(prob_output, 1)
                    dst.set_band_description(1, 'Probability')
            else:
                binary_predictions = (predictions_proba > threshold).astype(np.uint8)
                binary_output = np.full((src.height, src.width), 255.0, dtype=np.float32)
                binary_output[data_valid] = binary_predictions.astype(np.float32)

                profile.update({'count': 2})
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(binary_output, 1)
                    dst.write(prob_output, 2)
                    dst.set_band_description(1, 'Binary Prediction')
                    dst.set_band_description(2, 'Probability')

            return {
                "success": True,
                "file": tile_name,
                "output_file": output_file,
            }

    except Exception as e:
        return {
            "success": False,
            "file": tile_name,
            "reason": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def main():
    tow_root = Path(__file__).resolve().parents[2]
    default_models_dir = tow_root / "data" / "output" / "models"
    default_data_dir = tow_root / "data" / "input" / "embeddings" / "inference_embeddings"
    default_output_dir = tow_root / "data" / "output" / "rasters" / "prediction_tiles"

    parser = argparse.ArgumentParser(description="GPU-Accelerated XGBoost Predictor")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to XGBoost model (.json or .model). Default: auto-detect the newest model "
            f"in {default_models_dir}"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=str(default_data_dir),
        help=f"Directory with GeoTIFF tiles to predict (default: {default_data_dir})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help=f"Output directory for predictions (default: {default_output_dir})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold. If set, output is 2-band (binary + probability). If omitted, output is probability-only single band.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel CPU workers (all use GPU 0)")
    parser.add_argument("--pattern", default="*.tif", help="File pattern to match")
    parser.add_argument("--gpu-start", type=int, default=0, help="Starting GPU ID")
    parser.add_argument("--exclude-features", type=str, default=None,
                        help="Comma-separated feature indices or names to exclude (e.g., '64' or 'dtm_elevation')")
    parser.add_argument("--tile-bands", type=int, default=None,
                        help="Number of bands in input tiles (66, 67, or 68). Auto-detected if omitted.")

    args = parser.parse_args()
    try:
        model_path = Path(args.model) if args.model else resolve_default_model_path(default_models_dir)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    args.model = str(model_path)

    # Parse exclusions as numeric indices or feature names; names are resolved
    # per tile layout at prediction time.
    exclude_specs = []
    if args.exclude_features:
        exclude_list = [s.strip() for s in args.exclude_features.split(",")]
        for item in exclude_list:
            if item.isdigit():
                exclude_specs.append(int(item))
            else:
                exclude_specs.append(item)
        print(f"🚫 Requested feature exclusions: {exclude_list}")

    # Fallback only if model metadata is unavailable.
    _tile_bands_hint = args.tile_bands  # 66, 67, or None (auto)
    expected_bands = (_tile_bands_hint - len(exclude_specs)) if _tile_bands_hint is not None else None

    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be between 0 and 1")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find input files
    data_dir = Path(args.data_dir)
    input_files = list(data_dir.glob(args.pattern))

    if not input_files:
        print(f"❌ No files found matching {args.pattern} in {data_dir}")
        return

    model_schema = inspect_model_schema(args.model)
    sample_tile_bands = None
    try:
        with rasterio.open(input_files[0]) as sample_src:
            sample_tile_bands = sample_src.count
    except Exception:
        sample_tile_bands = None
    sample_exclude_indices = resolve_exclude_indices(exclude_specs, sample_tile_bands) if sample_tile_bands is not None else set()
    sample_effective_features = (
        sample_tile_bands - len(sample_exclude_indices) if sample_tile_bands is not None else None
    )

    print(f"🌲 XGBoost GPU Predictor")
    print(f"Model: {args.model}")
    print(f"Input files: {len(input_files)} tiles")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    if model_schema["num_feature"] is not None:
        print(f"Model expects: {model_schema['num_feature']} feature(s)")
    if model_schema["feature_names"]:
        feature_names = model_schema["feature_names"]
        preview = ", ".join(feature_names[:3])
        trailer = ", ".join(feature_names[-3:])
        print(f"Model features: {preview} ... {trailer}")
    if sample_tile_bands is not None:
        print(f"Sample tile bands: {sample_tile_bands} ({input_files[0].name})")
    if sample_effective_features is not None and sample_effective_features != sample_tile_bands:
        print(f"Sample tile features after exclusions: {sample_effective_features}")
    if (
        model_schema["num_feature"] is not None
        and sample_effective_features is not None
        and model_schema["num_feature"] != sample_effective_features
    ):
        print(
            f"⚠️  Model/tile feature mismatch: model expects {model_schema['num_feature']} "
            f"feature(s), sample tile yields {sample_effective_features}"
        )
    if args.threshold is None:
        print("Threshold: None (probability-only output)")
    else:
        print(f"Threshold: {args.threshold}")

    # Prepare worker arguments (model loaded once per worker via initializer)
    worker_args = []
    for input_file in input_files:
        output_file = output_dir / f"{input_file.stem}_prediction.tif"
        worker_args.append((str(input_file), str(output_file), args.threshold, exclude_specs, expected_bands))

    # Process files in parallel
    successful = 0
    failed = 0
    gpu_id = args.gpu_start
    failure_counts = {}
    failure_examples = {}

    print(f"\n🚀 Starting prediction with {args.workers} workers (model loaded once per worker)...")

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.model, gpu_id)
    ) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(predict_tile_gpu_worker, arg): arg[0]
                         for arg in worker_args}

        # Process results with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(worker_args),
                          desc="Predicting tiles"):
            input_file = future_to_file[future]
            try:
                result = future.result()
                if result and result.get("success"):
                    successful += 1
                else:
                    failed += 1
                    tile_name = result.get("file", Path(input_file).name) if isinstance(result, dict) else Path(input_file).name
                    reason = result.get("reason", "Unknown worker failure") if isinstance(result, dict) else "Unknown worker failure"
                    signature = (reason, result.get("traceback")) if isinstance(result, dict) else (reason, None)
                    failure_counts[signature] = failure_counts.get(signature, 0) + 1
                    if signature not in failure_examples:
                        failure_examples[signature] = tile_name
                        tqdm.write(f"❌ {tile_name}: {reason}")
                        worker_traceback = result.get("traceback") if isinstance(result, dict) else None
                        if worker_traceback:
                            tqdm.write(worker_traceback.rstrip())
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                signature = (reason, None)
                failure_counts[signature] = failure_counts.get(signature, 0) + 1
                if signature not in failure_examples:
                    failure_examples[signature] = Path(input_file).name
                    tqdm.write(f"❌ Worker failed for {Path(input_file).name}: {reason}")
                failed += 1

    print(f"\n✅ Prediction complete!")
    print(f"   Successful: {successful}/{len(input_files)} tiles")
    print(f"   Failed: {failed}/{len(input_files)} tiles")
    print(f"   Output directory: {output_dir}")

    if failure_counts:
        print(f"\n🪵 Failure summary ({len(failure_counts)} unique issue(s)):")
        for signature, count in sorted(failure_counts.items(), key=lambda item: item[1], reverse=True):
            reason, _traceback = signature
            example_tile = failure_examples[signature]
            print(f"   {count}x {reason} [example: {example_tile}]")

    if successful > 0:
        if args.threshold is None:
            print(f"\n💡 Output format: 1-band GeoTIFF")
            print(f"   Band 1: Probability values (0.0-1.0, 255.0=nodata)")
        else:
            print(f"\n💡 Output format: 2-band GeoTIFF")
            print(f"   Band 1: Binary predictions (0.0/1.0, 255.0=nodata)")
            print(f"   Band 2: Probability values (0.0-1.0, 255.0=nodata)")
        print(f"   Consistent nodata: 255.0")


if __name__ == "__main__":
    main()
