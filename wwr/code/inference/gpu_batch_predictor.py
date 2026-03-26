#!/usr/bin/env python3
"""
GPU Batch Predictor - Optimal for Sparse Data
==============================================
Strategy: Load all valid pixels into RAM, predict in one big GPU batch, write back.

This is MUCH faster than chunking when data is sparse (<5% valid pixels).
"""

import numpy as np
import rasterio
from pathlib import Path
import argparse
from tqdm import tqdm
import xgboost as xgb
import json
import time
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Optional CRF support
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
    HAS_CRF = True
except ImportError:
    HAS_CRF = False

def scan_tile_worker(tile_path):
    """Worker function to scan a single tile for valid pixels (supports NPZ and TIFF)."""
    try:
        # Check if NPZ or TIFF
        if tile_path.suffix == '.npz':
            # Fast NPZ loading (already preprocessed!)
            data = np.load(tile_path, allow_pickle=True)
            features = data['features']
            valid_mask = data['valid_mask']

            # Handle profile - extract from numpy object array
            profile_data = data['profile']
            if isinstance(profile_data, dict):
                profile = profile_data
            elif hasattr(profile_data, 'item') and profile_data.size == 1:
                profile = profile_data.item()
            elif hasattr(profile_data, '__len__') and len(profile_data) > 0:
                # Multi-element array, take first element
                profile = profile_data[0] if isinstance(profile_data[0], dict) else {}
            else:
                profile = {}

            # Fix: If profile is corrupted (only keys, no CRS/transform), reconstruct from filename
            if not isinstance(profile, dict) or 'crs' not in profile or 'transform' not in profile:
                # Extract coordinates from filename pattern: *-XXXXXXXX-YYYYYYYY.npz
                import re
                match = re.search(r'-(\d+)-(\d+)\.npz$', str(tile_path))
                if match:
                    x_origin = int(match.group(1))
                    y_origin = int(match.group(2))
                    h, w = valid_mask.shape
                    pixel_size = 10  # 10m pixels

                    # Reconstruct profile with proper georeferencing
                    from rasterio.transform import from_origin
                    profile = {
                        'driver': 'GTiff',
                        'dtype': 'float32',
                        'width': w,
                        'height': h,
                        'count': 2,  # Will be updated later
                        'crs': 'EPSG:27700',  # British National Grid
                        'transform': from_origin(x_origin, y_origin + h * pixel_size, pixel_size, pixel_size),
                        'nodata': None
                    }

            # Convert transform tuple back to Affine object if needed
            if isinstance(profile, dict) and 'transform' in profile:
                if isinstance(profile['transform'], (tuple, list)):
                    from rasterio.transform import Affine
                    profile['transform'] = Affine(*profile['transform'])

            # Ensure width/height are Python ints (rasterio requirement)
            if isinstance(profile, dict):
                if 'width' in profile:
                    profile['width'] = int(profile['width'])
                if 'height' in profile:
                    profile['height'] = int(profile['height'])

            n_bands = int(data['n_bands'])
            n_valid = len(features)

            return {
                'path': tile_path,
                'features': features,
                'valid_mask': valid_mask,
                'profile': profile,
                'n_valid': n_valid,
                'n_bands': n_bands
            }

        else:
            # TIFF loading (legacy path)
            with rasterio.open(tile_path) as src:
                data = src.read(out_dtype=np.float32)
                n_bands = data.shape[0]
                profile = src.profile.copy()

            # Vectorized validity check
            valid_mask = np.all(np.isfinite(data), axis=0)
            n_valid = valid_mask.sum()

            if n_valid == 0:
                return None

            # Extract features
            n_bands, height, width = data.shape
            data_flat = data.reshape(n_bands, -1)
            valid_flat = valid_mask.ravel()
            features = data_flat[:, valid_flat].T.copy()

            del data, data_flat, valid_flat

            return {
                'path': tile_path,
                'features': features,
                'valid_mask': valid_mask,
                'profile': profile,
                'n_valid': n_valid,
                'n_bands': n_bands
            }

    except Exception as e:
        return None

def crf_worker(tile_data_tuple):
    """Worker function for parallel CRF processing."""
    (tile_predictions, tile_info, crf_params) = tile_data_tuple

    try:
        valid_mask = tile_info['valid_mask']
        h, w = valid_mask.shape

        # Reconstruct probability map
        prob_output = np.full((h, w), np.nan, dtype=np.float32)
        prob_output[valid_mask] = tile_predictions

        # Reconstruct features
        n_bands = tile_info['n_bands']
        features_2d = np.full((h, w, n_bands), np.nan, dtype=np.float32)
        features_2d[valid_mask] = tile_info['features']

        # Apply CRF
        prob_output_refined = apply_dense_crf_binary(
            probabilities=prob_output,
            image_features=features_2d,
            valid_mask=valid_mask,
            **crf_params
        )

        return {
            'success': True,
            'tile_info': tile_info,
            'prob_output': prob_output_refined,
            'valid_mask': valid_mask
        }

    except Exception as e:
        return {
            'success': False,
            'tile_info': tile_info,
            'error': str(e)
        }

def apply_dense_crf_binary(probabilities, image_features, valid_mask,
                           spatial_std=3.0, spatial_weight=1.0,
                           bilateral_spatial_std=80.0, bilateral_color_std=13.0, bilateral_weight=5.0,
                           iterations=5):
    """
    Apply Dense CRF refinement for binary classification.

    Uses both spatial smoothness and appearance-based (bilateral) terms to refine predictions.
    Similar pixels (in feature space) are encouraged to have similar labels.

    Args:
        probabilities: 2D array (h, w) with probability predictions (0-1)
        image_features: 3D array (h, w, n_bands) with original satellite features
        valid_mask: 2D boolean array indicating valid pixels
        spatial_std: Standard deviation for spatial Gaussian kernel (pixels)
        spatial_weight: Weight for spatial smoothness term
        bilateral_spatial_std: Spatial std for bilateral term (pixels)
        bilateral_color_std: Color/feature std for bilateral term
        bilateral_weight: Weight for bilateral (appearance) term
        iterations: Number of mean-field inference iterations

    Returns:
        Refined probability map (2D array)
    """
    h, w = probabilities.shape
    n_bands = image_features.shape[2] if len(image_features.shape) == 3 else 1

    # Prepare full probability map (including invalid pixels as class 0)
    prob_map = probabilities.copy()
    prob_map[~valid_mask] = 0.0  # Invalid pixels get 0 probability

    # Binary: P(class=0) and P(class=1) for ALL pixels
    prob_class1 = np.clip(prob_map, 1e-5, 1.0 - 1e-5)
    prob_class0 = 1.0 - prob_class1

    # Stack probabilities: shape (2, h, w)
    softmax_probs = np.stack([prob_class0, prob_class1], axis=0)

    # Create unary potentials (negative log likelihood)
    # Reshape to (n_labels, h * w)
    unary = unary_from_softmax(softmax_probs.reshape(2, -1))

    # Initialize DenseCRF2D (designed for 2D images)
    d = dcrf.DenseCRF2D(w, h, 2)  # width, height, n_labels
    d.setUnaryEnergy(unary)

    # Add Gaussian pairwise potential (spatial smoothness)
    d.addPairwiseGaussian(sxy=(spatial_std, spatial_std), compat=spatial_weight,
                         kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Add Bilateral pairwise potential (appearance + spatial)
    # Only process first 3 bands for efficiency (pydensecrf expects RGB)
    if bilateral_weight > 0 and image_features is not None:
        # Extract first 3 bands only (much faster than processing all 67!)
        n_bands_use = min(3, n_bands)
        features_rgb = image_features[:, :, :n_bands_use]

        # Create RGB image (h, w, 3) - C-contiguous from the start
        img_rgb = np.zeros((h, w, 3), dtype=np.uint8, order='C')

        # Normalize valid pixels to 0-255
        features_valid = features_rgb[valid_mask]  # (n_valid, 3)
        features_min = np.nanmin(features_valid, axis=0, keepdims=True)
        features_max = np.nanmax(features_valid, axis=0, keepdims=True)
        features_range = features_max - features_min
        features_range[features_range == 0] = 1
        features_normalized = ((features_valid - features_min) / features_range * 255).astype(np.uint8)

        # Fill RGB image (only valid pixels)
        if n_bands_use == 3:
            img_rgb[valid_mask] = features_normalized
        elif n_bands_use == 1:
            # Repeat single band to RGB
            img_rgb[valid_mask, 0] = features_normalized[:, 0]
            img_rgb[valid_mask, 1] = features_normalized[:, 0]
            img_rgb[valid_mask, 2] = features_normalized[:, 0]
        else:  # n_bands_use == 2
            # Use 2 bands + repeat last
            img_rgb[valid_mask, 0] = features_normalized[:, 0]
            img_rgb[valid_mask, 1] = features_normalized[:, 1]
            img_rgb[valid_mask, 2] = features_normalized[:, 1]

        # Verify C-contiguous (should be by construction)
        if not img_rgb.flags['C_CONTIGUOUS']:
            img_rgb = np.ascontiguousarray(img_rgb)

        d.addPairwiseBilateral(sxy=(bilateral_spatial_std, bilateral_spatial_std),
                              srgb=(bilateral_color_std, bilateral_color_std, bilateral_color_std),
                              rgbim=img_rgb,
                              compat=bilateral_weight,
                              kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run inference
    Q = d.inference(iterations)
    Q = np.array(Q).reshape((2, h, w))

    # Extract refined probabilities for class 1
    refined_probabilities = Q[1, :, :]

    # Zero out invalid pixels
    refined_probabilities[~valid_mask] = np.nan

    return refined_probabilities

def main():
    parser = argparse.ArgumentParser(description='GPU Batch Predictor for Sparse Data')
    parser.add_argument('--data-dir', required=True, help='Directory with tiles')
    parser.add_argument('--model', required=True, help='XGBoost model path')
    parser.add_argument(
        '--output-dir',
        default='data/output/predictions/tiles',
        help='Output directory (default: data/output/predictions/tiles)',
    )
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binary classification threshold')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch-size', type=int, default=40_000_000,
                       help='Max pixels per GPU batch (default: 40M)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of workers for tile scanning (default: CPU count - 2)')

    # CRF post-processing arguments
    parser.add_argument('--crf', action='store_true',
                       help='Apply Dense CRF post-processing for spatial refinement')
    parser.add_argument('--crf-spatial-std', type=float, default=1.5,
                       help='CRF spatial smoothness std deviation (default: 1.5 pixels = 15m, conservative for rare habitat)')
    parser.add_argument('--crf-spatial-weight', type=float, default=1.0,
                       help='CRF spatial smoothness weight (default: 1.0)')
    parser.add_argument('--crf-bilateral-spatial', type=float, default=3.0,
                       help='CRF bilateral spatial std (default: 3.0 pixels = 30m, local refinement for rare habitat)')
    parser.add_argument('--crf-bilateral-color', type=float, default=10.0,
                       help='CRF bilateral color/feature std (default: 10.0, strict spectral matching)')
    parser.add_argument('--crf-bilateral-weight', type=float, default=5.0,
                       help='CRF bilateral weight (default: 5.0, set to 0 for spatial-only fast mode)')
    parser.add_argument('--crf-iterations', type=int, default=5,
                       help='Number of CRF mean-field iterations (default: 5)')
    parser.add_argument('--crf-fast', action='store_true',
                       help='Fast CRF mode: spatial smoothing only, no bilateral term (~10x faster)')

    args = parser.parse_args()

    # Check CRF availability
    if args.crf and not HAS_CRF:
        print("❌ Error: --crf flag requires pydensecrf library")
        print("   Install with: pip install pydensecrf")
        print("   Or install from source: https://github.com/lucasb-eyer/pydensecrf")
        return

    # Fast mode disables bilateral term
    if args.crf_fast:
        args.crf_bilateral_weight = 0.0

    # Determine number of workers
    n_workers = args.workers if args.workers else max(1, mp.cpu_count() - 2)

    print("🚀 GPU Batch Predictor - Optimized for Sparse Data")
    print("=" * 70)
    if args.crf:
        mode = "FAST (spatial only)" if args.crf_bilateral_weight == 0 else "FULL (spatial + bilateral)"
        print(f"   🌊 Dense CRF post-processing ENABLED - {mode}")
        print(f"      Spatial: σ={args.crf_spatial_std}, w={args.crf_spatial_weight}")
        if args.crf_bilateral_weight > 0:
            print(f"      Bilateral: σ_xy={args.crf_bilateral_spatial}, σ_rgb={args.crf_bilateral_color}, w={args.crf_bilateral_weight}")
        print(f"      Iterations: {args.crf_iterations}")
        print("=" * 70)

    # Find tiles
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tiles = sorted(list(data_dir.glob('*.tif')) + list(data_dir.glob('*.npz')))
    if not tiles:
        print(f"❌ No tiles found in {data_dir}")
        return

    print(f"\n📂 Found {len(tiles)} tiles (.tif and .npz)")

    # Load model
    print(f"\n🤖 Loading model...")
    model = xgb.Booster()
    model.load_model(args.model)

    # Detect model type BEFORE setting device
    config = json.loads(model.save_config())
    objective = config.get('learner', {}).get('objective', {}).get('name', '')
    is_binary = 'binary' in objective
    num_class = int(config.get('learner', {}).get('learner_model_param', {}).get('num_class', 2)) if not is_binary else 2

    print(f"   Model type: {'Binary' if is_binary else f'Multi-class ({num_class} classes)'}")

    # Set GPU device
    model.set_param({'device': f'cuda:{args.gpu}'})
    print(f"   ✅ GPU {args.gpu} enabled")

    # Phase 1: Scan all tiles and collect valid pixels (PARALLEL)
    print(f"\n1️⃣  Scanning tiles for valid pixels...")
    print(f"   Using {n_workers} workers")

    tile_data = []
    total_valid_pixels = 0

    # Process tiles in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_tile = {executor.submit(scan_tile_worker, tile): tile for tile in tiles}

        # Collect results as they complete
        for future in tqdm(as_completed(future_to_tile), total=len(tiles), desc="   Scanning"):
            result = future.result()
            if result is not None:
                tile_data.append(result)
                total_valid_pixels += result['n_valid']

    # Validate band count consistency
    if len(tile_data) > 0:
        n_bands = tile_data[0]['n_bands']
        inconsistent_tiles = [td['path'].name for td in tile_data if td['n_bands'] != n_bands]
        if inconsistent_tiles:
            print(f"\n⚠️  Warning: Inconsistent band counts found!")
            print(f"   Expected: {n_bands} bands")
            print(f"   Inconsistent tiles: {len(inconsistent_tiles)}")
            print(f"   First few: {inconsistent_tiles[:5]}")
    else:
        n_bands = 67  # Default

    print(f"\n   ✅ Scanned {len(tile_data)} tiles with valid data")
    print(f"   📊 Total valid pixels: {total_valid_pixels:,}")
    print(f"   📊 Bands per pixel: {n_bands}")
    print(f"   💾 Estimated RAM: ~{total_valid_pixels * n_bands * 4 / 1024**3:.1f} GB")

    if len(tile_data) == 0:
        print("❌ No valid pixels found!")
        return

    # Phase 2: Batch prediction on GPU
    print(f"\n2️⃣  Running GPU batch prediction...")

    # Process in batches if needed (to avoid GPU OOM)
    batch_size = args.batch_size
    n_batches = (total_valid_pixels + batch_size - 1) // batch_size

    print(f"   Batches: {n_batches} × {batch_size:,} pixels")

    start_time = time.time()

    # Collect all features into one array (optimized)
    print(f"   Merging features...")
    # Pre-allocate array for efficiency
    all_features = np.empty((total_valid_pixels, n_bands), dtype=np.float32)
    idx = 0
    for td in tile_data:
        n = len(td['features'])
        all_features[idx:idx + n] = td['features']
        idx += n
    print(f"   ✅ Merged {len(all_features):,} pixels")

    # Predict in batches (optimized)
    print(f"   Predicting...")

    # Determine prediction shape
    if is_binary:
        pred_shape = (total_valid_pixels,)
    else:
        pred_shape = (total_valid_pixels, num_class)

    # Pre-allocate prediction array
    all_predictions = np.empty(pred_shape, dtype=np.float32)

    for i in tqdm(range(0, len(all_features), batch_size), desc="   GPU batches"):
        batch = all_features[i:i + batch_size]
        dtest = xgb.DMatrix(batch, device=f'cuda:{args.gpu}')  # Specify GPU device
        preds = model.predict(dtest)

        # Write directly to pre-allocated array
        batch_end = min(i + batch_size, total_valid_pixels)
        all_predictions[i:batch_end] = preds

        del dtest
        gc.collect()

    elapsed = time.time() - start_time
    speed = len(all_features) / elapsed / 1e6

    print(f"\n   ✅ Prediction complete!")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   Speed: {speed:.2f}M pixels/sec")

    # Phase 3: CRF Refinement (parallel) + Write results (sequential)
    print(f"\n3️⃣  Post-processing and writing results...")

    idx = 0
    successful = 0
    crf_time_total = 0.0

    # Prepare CRF parameters if enabled
    if args.crf:
        crf_params = {
            'spatial_std': args.crf_spatial_std,
            'spatial_weight': args.crf_spatial_weight,
            'bilateral_spatial_std': args.crf_bilateral_spatial,
            'bilateral_color_std': args.crf_bilateral_color,
            'bilateral_weight': args.crf_bilateral_weight,
            'iterations': args.crf_iterations
        }
        print(f"   🌊 Running parallel CRF refinement on {n_workers} workers...")
        crf_start_total = time.time()

        # Prepare all CRF tasks
        crf_tasks = []
        for i, tile_info in enumerate(tile_data):
            n_valid = tile_info['n_valid']
            tile_predictions = all_predictions[idx:idx + n_valid]
            idx += n_valid
            crf_tasks.append((tile_predictions, tile_info, crf_params))

        # Process CRF in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            crf_results = list(tqdm(
                executor.map(crf_worker, crf_tasks),
                total=len(crf_tasks),
                desc="   CRF refinement"
            ))

        crf_time_total = time.time() - crf_start_total
        print(f"   ✅ CRF complete: {crf_time_total:.1f}s ({crf_time_total/len(tile_data):.2f}s per tile)")

        # Now write results sequentially
        print(f"   💾 Writing {len(crf_results)} tiles to disk...")
        for result in tqdm(crf_results, desc="   Writing"):
            if not result['success']:
                print(f"\n⚠️  CRF failed for {result['tile_info']['path'].name}: {result['error']}")
                continue

            try:
                tile_info = result['tile_info']
                prob_output = result['prob_output']
                valid_mask = result['valid_mask']
                h, w = valid_mask.shape

                profile = tile_info['profile'].copy()

                # Ensure profile has Python types
                if isinstance(profile, dict):
                    for key, value in profile.items():
                        if hasattr(value, 'item'):
                            profile[key] = value.item()
                    if 'width' in profile:
                        profile['width'] = int(profile['width'])
                    else:
                        profile['width'] = int(w)
                    if 'height' in profile:
                        profile['height'] = int(profile['height'])
                    else:
                        profile['height'] = int(h)

                # Threshold refined probabilities
                binary_preds = (prob_output[valid_mask] > args.threshold).astype(np.uint8)
                class_output = np.full((h, w), 255, dtype=np.float32)
                class_output[valid_mask] = binary_preds

                # Update profile
                profile.update({
                    'count': 2,
                    'dtype': 'float32',
                    'compress': 'ZSTD',
                    'predictor': 3,
                    'tiled': True,
                    'blockxsize': 512,
                    'blockysize': 512
                })

                # Write output
                output_path = output_dir / f"{tile_info['path'].stem}_prediction.tif"
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(class_output, 1)
                    dst.write(prob_output, 2)

                successful += 1

            except Exception as e:
                print(f"\n⚠️  Error writing {tile_info['path'].name}: {e}")
                continue

    else:
        # No CRF - write directly (sequential)
        print(f"   💾 Writing {len(tile_data)} tiles...")
        for tile_info in tqdm(tile_data, desc="   Writing"):
            try:
                n_valid = tile_info['n_valid']
                tile_predictions = all_predictions[idx:idx + n_valid]
                idx += n_valid

                # Create output arrays
                profile = tile_info['profile'].copy()
                valid_mask = tile_info['valid_mask']
                h, w = valid_mask.shape

                # Ensure profile has Python types
                if isinstance(profile, dict):
                    for key, value in profile.items():
                        if hasattr(value, 'item'):
                            profile[key] = value.item()
                    if 'width' in profile:
                        profile['width'] = int(profile['width'])
                    else:
                        profile['width'] = int(w)
                    if 'height' in profile:
                        profile['height'] = int(profile['height'])
                    else:
                        profile['height'] = int(h)

                if is_binary:
                    # Binary classification - no CRF
                    prob_output = np.full((h, w), np.nan, dtype=np.float32)
                    prob_output[valid_mask] = tile_predictions

                    binary_preds = (tile_predictions > args.threshold).astype(np.uint8)
                    class_output = np.full((h, w), 255, dtype=np.float32)
                    class_output[valid_mask] = binary_preds

                    # Update profile
                    profile.update({
                        'count': 2,
                        'dtype': 'float32',
                        'compress': 'ZSTD',
                        'predictor': 3,
                        'tiled': True,
                        'blockxsize': 512,
                        'blockysize': 512
                    })

                    # Write output
                    output_path = output_dir / f"{tile_info['path'].stem}_prediction.tif"
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(class_output, 1)
                        dst.write(prob_output, 2)

                else:
                    # Multi-class classification
                    class_preds = np.argmax(tile_predictions, axis=1).astype(np.uint8)
                    class_output = np.full((h, w), 255, dtype=np.float32)
                    class_output[valid_mask] = class_preds

                    # Update profile
                    profile.update({
                        'count': 1 + num_class,
                        'dtype': 'float32',
                        'compress': 'ZSTD',
                        'predictor': 3,
                        'tiled': True,
                        'blockxsize': 512,
                        'blockysize': 512
                    })

                    # Write output
                    output_path = output_dir / f"{tile_info['path'].stem}_prediction.tif"
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        # Band 1: Classes
                        dst.write(class_output, 1)

                        # Bands 2-N+1: Probabilities
                        for c in range(num_class):
                            prob_output = np.full((h, w), np.nan, dtype=np.float32)
                            prob_output[valid_mask] = tile_predictions[:, c]
                            dst.write(prob_output, c + 2)

                successful += 1

            except Exception as e:
                print(f"\n⚠️  Error writing {tile_info['path'].name}: {e}")
                continue

    total_time = time.time() - start_time
    print(f"\n✅ Complete!")
    print(f"   Tiles processed: {successful}/{len(tile_data)}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    if args.crf and successful > 0:
        print(f"   CRF time: {crf_time_total:.1f}s ({crf_time_total/successful:.2f}s per tile)")
        print(f"   CRF overhead: {crf_time_total/total_time*100:.1f}% of total time")
    if successful > 0:
        print(f"   Average speed: {total_valid_pixels/total_time/1e6:.2f}M pixels/sec")
    print(f"   Output: {output_dir}")

    if successful == 0:
        print("\n❌ WARNING: No tiles were successfully processed!")
        print("   Check the error messages above for details.")


if __name__ == "__main__":
    main()
