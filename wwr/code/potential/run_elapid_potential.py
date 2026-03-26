#!/usr/bin/env python3
"""
Run Elapid MaxEnt potential model for wet woodland extent.

Compatibility entrypoint retained for older commands. The preferred script name
is ``maxent.py`` in the same directory.

Uses:
  1) Predictor stack (from build_abiotic_stack.py), e.g. potential_predictors.tif
  2) Wet woodland presence raster: align to stack, apply threshold to probability
     (band 2) to get presence, then optionally union with rasterized KML polygons
  3) Elapid: sample presence/background points, annotate with stack, fit MaxentModel,
     apply to stack → potential map

Publication-ready evaluation: presence balanced by peat (on/off), distance-based sample
weights for spatial representativeness, and geographic K-fold CV to report spatially
held-out test AUC (mean ± std) before fitting the final model on full data for mapping.

Wet woodland raster format: band 1 = binary 0/1, band 2 = probability 0–1, 255 = nodata.
If --threshold is set, band 2 is used (pixel is presence if prob >= threshold).
Otherwise band 1 is used as binary presence. The raster is reprojected to match the
stack's CRS, resolution, and extent; use a single merged raster (e.g. from
wet_woodland_stats or your tile mosaic) that overlaps the stack.

Output suitability: existing wet woodland (presence) pixels are always set to nodata
in the potential map, since suitability there is uninformative. Use --mask-forest to
additionally mask the full mapped/forest domain.

Usage:
  pip install elapid geopandas   # or conda install -c conda-forge elapid

  python maxent.py

  # Optional: union observed KML polygons with the wet woodland raster before sampling
  python maxent.py --wet-woodland-kml data/validation/wetwoodlands.kml

  # Optional: cap presence samples (default 50k) and set background count (default 50k)
  python maxent.py --threshold 0.5 --max-presence 30000 --n-background 50000
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress elapid hinge-feature NumPy 2 "where without out" warnings (cosmetic)
warnings.filterwarnings("ignore", message=".*'where' used without 'out'.*", category=UserWarning)

import numpy as np
import rasterio
from sklearn.metrics import roc_auc_score
from rasterio.features import rasterize
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.transform import from_origin

# Cap so product of two bands can't overflow float32 (e.g. 1e10 * 1e10 = 1e20 < 3.4e38)
SAFE_PREDICTOR_MIN = -1.0e10
SAFE_PREDICTOR_MAX = 1.0e10

try:
    import elapid as ela
except ImportError:
    ela = None
try:
    from elapid.train_test_split import GeographicKFold
except ImportError:
    GeographicKFold = None
try:
    import geopandas as gpd
except ImportError:
    gpd = None
try:
    import fiona
except ImportError:
    fiona = None
try:
    import shap
except ImportError:
    shap = None


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def write_sanitized_stack(
    stack_path: str,
    band_names: List[str],
    bounds: List[Tuple[float, float]],
    out_path: str,
    nodata_val: float = np.nan,
    known_categorical: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    """
    Write a copy of the stack with each band clipped to [lo, hi] and non-finite set to nodata,
    so apply_model_to_rasters never sees inf or extreme values.
    For categorical bands, values not in known_categorical[j] are replaced with the first
    known category so the encoder never sees "unknown" during predict.
    """
    known_cat = known_categorical or {}
    with rasterio.open(stack_path) as src:
        profile = src.profile.copy()
        nbands = src.count
        profile.update(dtype=np.float32, nodata=nodata_val)
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                block = src.read(window=window, masked=False)
                out_block = np.full((nbands, window.height, window.width), nodata_val, dtype=np.float32)
                for j in range(nbands):
                    b = block[j].astype(np.float64)
                    valid = np.isfinite(b)
                    if j < len(bounds):
                        lo, hi = bounds[j]
                        clipped = np.clip(b, lo, hi)
                    else:
                        clipped = b
                    # Also clip to safe range so product features never overflow float32
                    clipped = np.clip(clipped, SAFE_PREDICTOR_MIN, SAFE_PREDICTOR_MAX)
                    # Keep categorical bands as integers; map unknown categories to a known one
                    if j < len(band_names) and "categorical" in band_names[j].lower():
                        clipped = np.round(clipped)
                        if j in known_cat:
                            allowed_arr = np.asarray(known_cat[j]).ravel().astype(np.int64)
                            allowed_list = allowed_arr.tolist()
                            fallback = float(allowed_arr[0]) if len(allowed_arr) > 0 else 0.0
                            # Cast only valid pixels to int to avoid "invalid value in cast" on NaN
                            clipped_int = np.full(clipped.shape, int(fallback), dtype=np.int64)
                            clipped_int[valid] = np.round(clipped[valid]).astype(np.int64)
                            # Any valid pixel not in training set -> fallback (use list for np.isin)
                            mask_unknown = valid & ~np.isin(clipped_int, allowed_list)
                            clipped = np.where(mask_unknown, fallback, clipped)
                    out_block[j] = np.where(valid, clipped.astype(np.float32), nodata_val)
                dst.write(out_block, window=window)


def apply_model_to_raster_safe(
    model: "ela.MaxentModel",
    stack_path: str,
    output_path: str,
    nodata: float = -9999.0,
    quiet: bool = False,
) -> None:
    """
    Apply a fitted Maxent model to a predictor stack in windows. Skips windows
    with zero valid pixels so we never call model.predict() with 0 samples
    (avoids sklearn "0 sample(s)" error when a block is entirely nodata).
    """
    with rasterio.open(stack_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype=np.float32, nodata=nodata)
        windows_list = list(src.block_windows(1))
    try:
        from tqdm import tqdm
        windows_list = tqdm(windows_list, desc="Window", disable=quiet)
    except Exception:
        pass
    with rasterio.open(stack_path) as src:
        with rasterio.open(output_path, "w", **profile) as dst:
            for _, window in windows_list:
                block = src.read(window=window, masked=True)
                # valid: (height, width), True where all bands are valid (not masked)
                valid = ~np.any(block.mask, axis=0)
                if not np.any(valid):
                    out_block = np.full((1, window.height, window.width), nodata, dtype=np.float32)
                    dst.write(out_block, window=window)
                    continue
                covariates = block.data[:, valid].T.astype(np.float64)
                pred = model.predict(covariates)
                out_block = np.full((1, window.height, window.width), nodata, dtype=np.float32)
                out_block[0][valid] = pred.astype(np.float32)
                dst.write(out_block, window=window)


def get_stack_profile_and_labels(stack_path: str) -> Tuple[dict, List[str]]:
    """Return (profile, band_names) for the predictor stack."""
    with rasterio.open(stack_path) as src:
        profile = src.profile.copy()
        names = []
        for i in range(1, src.count + 1):
            desc = src.descriptions[i - 1]
            names.append(desc if desc else f"band_{i}")
    return profile, names


def parse_band_name_args(values: Optional[List[List[str]]]) -> List[str]:
    """Flatten repeated/nested CLI band-name args, supporting comma-separated values."""
    if not values:
        return []
    names: List[str] = []
    seen = set()
    for group in values:
        for item in group:
            for name in item.split(","):
                band_name = name.strip()
                if band_name and band_name not in seen:
                    names.append(band_name)
                    seen.add(band_name)
    return names


def write_stack_band_subset(
    stack_path: str,
    keep_indices: List[int],
    keep_names: List[str],
    out_path: str,
) -> None:
    """Write a reduced copy of the predictor stack containing only selected bands."""
    if len(keep_indices) != len(keep_names):
        raise ValueError("keep_indices and keep_names must have the same length.")
    if not keep_indices:
        raise ValueError("At least one predictor band must be retained.")
    band_numbers = [idx + 1 for idx in keep_indices]
    with rasterio.open(stack_path) as src:
        profile = src.profile.copy()
        profile.update(count=len(band_numbers))
        with rasterio.open(out_path, "w", **profile) as dst:
            for out_idx, name in enumerate(keep_names, 1):
                dst.set_band_description(out_idx, name)
            for _, window in src.block_windows(1):
                block = src.read(indexes=band_numbers, window=window, masked=False)
                dst.write(block, window=window)


def _is_nan_number(value: object) -> bool:
    return isinstance(value, (float, np.floating)) and np.isnan(value)


def _valid_data_mask(arr: np.ndarray, nodata_value: Optional[float]) -> np.ndarray:
    """Return finite mask, excluding nodata where defined."""
    valid = np.isfinite(arr)
    if nodata_value is None or _is_nan_number(nodata_value):
        return valid
    return valid & (arr != nodata_value)


PEAT_PROB_THRESHOLD = 0.5


def _find_peat_band_info(names: List[Optional[str]]) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Return (band_index_1_based, mode, description) for peat predictor bands."""
    normalized = [(i, (name or "").lower(), name or "") for i, name in enumerate(names, start=1)]
    for i, low, raw in normalized:
        if "peat_depth" in low:
            return i, "depth", raw
    for i, low, raw in normalized:
        if "peat_prob" in low or "peat_probability" in low:
            return i, "probability", raw
    for i, low, raw in normalized:
        if "peat" in low:
            return i, "probability", raw
    return None, None, None


def _peat_extent_mask(values: np.ndarray, mode: Optional[str]) -> np.ndarray:
    """Classify peat extent from either peat probability or peat depth values."""
    valid = np.isfinite(values)
    if mode == "depth":
        return valid & (values > 0.0)
    return valid & (values >= PEAT_PROB_THRESHOLD)


def _peat_definition_label(mode: Optional[str], description: Optional[str] = None) -> str:
    if mode == "depth":
        return f"{description or 'peat_depth_m'} > 0"
    return f"{description or 'peat_prob'} >= {PEAT_PROB_THRESHOLD:g}"


def load_polygon_geodataframe(
    vector_path: str,
    *,
    target_crs=None,
    source_label: str = "Vector",
) -> "gpd.GeoDataFrame":
    """
    Load a polygon vector dataset, handling KML CRS defaults and filtering to
    polygon geometries only.
    """
    if gpd is None:
        raise ImportError("geopandas is required for vector polygon inputs.")
    suffix = Path(vector_path).suffix.lower()
    if suffix == ".kml" and fiona is not None:
        try:
            fiona.drvsupport.supported_drivers["KML"] = "r"
        except Exception:
            pass
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        return gdf
    poly_mask = gdf.geometry.geom_type.isin(("Polygon", "MultiPolygon"))
    gdf = gdf.loc[poly_mask].copy()
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        if suffix == ".kml":
            gdf.set_crs("EPSG:4326", inplace=True)
        else:
            raise ValueError(f"{source_label} has no CRS: {vector_path}")
    if target_crs is not None and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    return gdf.reset_index(drop=True)


def align_wet_valid_mask_to_stack(
    wet_path: str,
    stack_profile: dict,
) -> np.ndarray:
    """
    Reproject wet woodland valid-data mask (all non-nodata pixels) to stack grid.
    Used with --mask-forest to mask the entire mapped domain from the output.
    """
    with rasterio.open(wet_path) as src:
        nodata_src = getattr(src, "nodata", None)
        arr = src.read(1, masked=False)
        valid_src = _valid_data_mask(arr, nodata_src).astype(np.uint8)
        src_transform = src.transform
        src_crs = src.crs

    dst_transform = stack_profile["transform"]
    dst_crs = stack_profile["crs"]
    dst_height = stack_profile["height"]
    dst_width = stack_profile["width"]
    out = np.zeros((dst_height, dst_width), dtype=np.uint8)
    reproject(
        source=valid_src,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.max,  # Use max so ANY valid pixel in cell → masked
    )
    return out.astype(bool)


def rasterize_vector_mask(
    mask_path: str,
    stack_profile: dict,
    *,
    buffer_m: float = 0.0,
) -> np.ndarray:
    """Rasterize a polygon mask shapefile to the target grid, with optional buffering."""
    gdf = load_polygon_geodataframe(
        mask_path,
        target_crs=stack_profile["crs"],
        source_label="Mask vector",
    )
    if gdf.empty:
        return np.zeros((stack_profile["height"], stack_profile["width"]), dtype=bool)
    if buffer_m != 0.0:
        gdf = gdf.copy()
        gdf.geometry = gdf.geometry.buffer(buffer_m)
    geometries = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        return np.zeros((stack_profile["height"], stack_profile["width"]), dtype=bool)
    mask = rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=(stack_profile["height"], stack_profile["width"]),
        transform=stack_profile["transform"],
        fill=0,
        dtype=np.uint8,
    )
    return mask.astype(bool)


def apply_vector_clip_to_output(
    output_path: str,
    mask_path: Optional[str],
    *,
    buffer_m: float = 0.0,
    label: str = "output",
) -> int:
    """Set pixels outside a polygon mask to nodata in-place."""
    if not mask_path or not os.path.isfile(mask_path):
        return 0
    with rasterio.open(output_path, "r+") as dst:
        profile = {
            "transform": dst.transform,
            "crs": dst.crs,
            "height": dst.height,
            "width": dst.width,
        }
        inside_mask = rasterize_vector_mask(mask_path, profile, buffer_m=buffer_m)
        arr = dst.read(1, masked=False)
        nodata_value = dst.nodata if dst.nodata is not None else -9999.0
        valid = _valid_data_mask(arr, dst.nodata)
        to_mask = valid & ~inside_mask
        if np.any(to_mask):
            arr[to_mask] = nodata_value
            dst.write(arr, 1)
        masked_n = int(to_mask.sum())
    if masked_n > 0:
        print(
            f"    Clipped {label} to {mask_path}"
            f"{f' (buffer {buffer_m:g} m)' if buffer_m != 0.0 else ''}: "
            f"masked {masked_n:,} pixels"
        )
    else:
        print(
            f"    Clip mask applied to {label}"
            f"{f' (buffer {buffer_m:g} m)' if buffer_m != 0.0 else ''}: no additional pixels masked"
        )
    return masked_n


def rasterize_presence_vector_to_stack(
    vector_path: str,
    stack_profile: dict,
    *,
    all_touched: bool = True,
) -> np.ndarray:
    """
    Rasterize polygon wet woodland observations to the stack grid.
    `all_touched=True` approximates the existing raster-path behaviour where any
    contributing fine-resolution wet pixel makes the coarser stack cell present.
    """
    gdf = load_polygon_geodataframe(
        vector_path,
        target_crs=stack_profile["crs"],
        source_label="Wet woodland polygon vector",
    )
    if gdf.empty:
        return np.zeros((stack_profile["height"], stack_profile["width"]), dtype=bool)
    geometries = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        return np.zeros((stack_profile["height"], stack_profile["width"]), dtype=bool)
    presence = rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=(stack_profile["height"], stack_profile["width"]),
        transform=stack_profile["transform"],
        fill=0,
        all_touched=all_touched,
        dtype=np.uint8,
    )
    return presence.astype(bool)


def compute_and_report_vif(
    X: np.ndarray,
    band_names: List[str],
    outdir: str,
) -> None:
    """
    Compute Variance Inflation Factor (VIF) for all continuous predictors and
    write a plain-text report to outdir/vif_report.txt.

    VIF = 1 / (1 - R²_j) where R²_j is from regressing predictor j on all others.
    VIF > 10: strong multicollinearity (consider dropping one).
    VIF 5–10: moderate (worth noting in methods).
    VIF < 5: acceptable.

    Categorical bands (name contains 'categorical') are skipped.
    """
    continuous_idx = [j for j, n in enumerate(band_names) if "categorical" not in n.lower()]
    if len(continuous_idx) < 2:
        return
    continuous_names = [band_names[j] for j in continuous_idx]
    X_cont = X[:, continuous_idx].astype(np.float64)

    # Standardise to improve numerical stability
    means = X_cont.mean(axis=0)
    stds = X_cont.std(axis=0)
    stds[stds < 1e-12] = 1.0
    X_std = (X_cont - means) / stds

    vifs = []
    for j in range(X_std.shape[1]):
        others = np.delete(X_std, j, axis=1)
        design = np.hstack([np.ones((len(others), 1)), others])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, X_std[:, j], rcond=None)
            resid = X_std[:, j] - design @ coeffs
            ss_res = float(np.dot(resid, resid))
            ss_tot = float(np.sum((X_std[:, j] - X_std[:, j].mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
            vif = 1.0 / (1.0 - r2) if r2 < 0.9999 else np.inf
        except Exception:
            vif = np.nan
        vifs.append(vif)

    # Sort descending by VIF
    order = sorted(range(len(vifs)), key=lambda i: -vifs[i] if np.isfinite(vifs[i]) else 1e18)

    lines = [
        "# Variance Inflation Factor (VIF) report",
        "# VIF > 10 = strong multicollinearity | 5-10 = moderate | < 5 = acceptable",
        "# Categorical predictors excluded.",
        "",
        f"{'Predictor':<30}  {'VIF':>8}  Flag",
        "-" * 55,
    ]
    print("    VIF (continuous predictors):")
    for i in order:
        name = continuous_names[i]
        v = vifs[i]
        if not np.isfinite(v):
            flag = "  ! PERFECT COLLINEARITY"
        elif v > 10:
            flag = "  ! HIGH"
        elif v > 5:
            flag = "  ~ moderate"
        else:
            flag = ""
        v_str = f"{v:8.2f}" if np.isfinite(v) else "     inf"
        lines.append(f"{name:<30}  {v_str}  {flag.strip()}")
        print(f"      {name:<30}  VIF = {v_str.strip()}{flag}")

    vif_path = os.path.join(outdir, "vif_report.txt")
    with open(vif_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"    VIF report written: {vif_path}")


def _compute_shap_importance(
    model: "ela.MaxentModel",
    X: np.ndarray,
    band_names: List[str],
    outdir: str,
    n_background: int = 150,
    n_eval: int = 500,
    seed: Optional[int] = None,
) -> None:
    """
    Compute SHAP-based variable importance using KernelExplainer (model-agnostic).
    Writes shap_importance.csv and shap_importance.png to outdir.
    """
    if shap is None:
        print("    SHAP skipped: install with pip install shap", file=sys.stderr)
        return
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    n_bg = min(n_background, n)
    n_ev = min(n_eval, n)
    X_bg = X[rng.choice(n, size=n_bg, replace=False)]
    X_ev = X[rng.choice(n, size=n_ev, replace=False)]
    print(f"    SHAP: background={n_bg}, eval={n_ev} (KernelExplainer may take a minute)...")
    explainer = shap.KernelExplainer(model.predict, X_bg)
    shap_vals = explainer.shap_values(X_ev)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    csv_path = os.path.join(outdir, "shap_importance.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("predictor,mean_abs_shap\n")
        for i in order:
            f.write(f"{band_names[i]},{mean_abs[i]:.6f}\n")
    print(f"    Wrote {csv_path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(band_names))))
        ax.barh(range(len(band_names)), mean_abs[order], color="steelblue", alpha=0.8)
        ax.set_yticks(range(len(band_names)))
        ax.set_yticklabels([band_names[j] for j in order], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("MaxEnt variable importance (SHAP)")
        fig.tight_layout()
        plot_path = os.path.join(outdir, "shap_importance.png")
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"    Wrote {plot_path}")
    except Exception as e:
        print(f"    SHAP plot skipped: {e}", file=sys.stderr)


def write_elapid_report(
    report_path: str,
    *,
    args: argparse.Namespace,
    stack_profile: dict,
    band_names: List[str],
    excluded_band_names: List[str],
    presence_sampling_mode: str,
    n_presence_pixels_raster: int,
    n_presence_pixels_kml: Optional[int],
    n_presence_pixels: int,
    n_presence_points: int,
    n_background_points: int,
    n_dropped_nonfinite: int,
    n_training_samples: int,
    n_training_presence: int,
    n_training_background: int,
    categorical_indices: List[int],
    forest_mask_pixels: Optional[int],
    urban_mask_pixels: Optional[int],
    mean_auc: Optional[float],
    std_auc: Optional[float],
    cv_aucs: List[float],
    thresholds: Dict[str, float],
    thresh_path: str,
    unmasked_output_path: str,
    output_path: str,
    output_10m_path: Optional[str],
    output_10m_info: Optional[Dict[str, object]],
    eval_background_raster: Optional[str],
    vif_path: Optional[str],
    shap_csv_path: Optional[str],
    shap_plot_path: Optional[str],
    landvalue_report_path: Optional[str],
    merged_path: str,
) -> None:
    _ensure_dir(str(Path(report_path).parent))

    lines = [
        "=" * 70,
        "WET WOODLAND ELAPID MODEL REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "1. INPUTS",
        "-" * 70,
        f"Stack: {args.stack}",
        f"Wet woodland raster: {args.wet_woodland}",
        f"Wet woodland KML: {args.wet_woodland_kml if args.wet_woodland_kml else 'None'}",
        f"Excluded predictors: {', '.join(excluded_band_names) if excluded_band_names else 'None'}",
        f"Outdir: {args.outdir}",
        f"Threshold source: {'probability band >= ' + str(args.threshold) if args.threshold is not None else 'band 1 (binary or collapsed split labels)'}",
        f"Predictors: {len(band_names)} band(s)",
        f"Predictor names: {', '.join(band_names)}",
        f"Stack size: {stack_profile['width']} x {stack_profile['height']} pixels",
        f"Stack CRS: {stack_profile['crs']}",
        f"Mask forest: {bool(args.mask_forest)}",
        f"Urban mask: {args.urban_shp if args.urban_shp else 'None'}",
        f"Clip mask: {args.clip_mask_shp if args.clip_mask_shp else 'None'}",
        f"Clip mask buffer (m): {args.clip_mask_buffer_m}",
    ]
    if forest_mask_pixels is not None:
        lines.append(f"Forest mask pixels: {forest_mask_pixels:,}")
    if urban_mask_pixels is not None:
        lines.append(f"Urban mask pixels: {urban_mask_pixels:,}")

    lines.extend(
        [
            "",
            "2. TRAINING SAMPLES",
            "-" * 70,
            f"Presence pixels from wet woodland raster: {n_presence_pixels_raster:,}",
            f"Presence sampling mode: {presence_sampling_mode}",
            f"Sampled presence points: {n_presence_points:,}",
            f"Sampled background points: {n_background_points:,}",
        ]
    )
    if n_presence_pixels_kml is not None:
        lines.append(f"Presence pixels from wet woodland KML: {n_presence_pixels_kml:,}")
    lines.append(f"Presence pixels after merge/alignment: {n_presence_pixels:,}")
    if n_dropped_nonfinite > 0:
        lines.append(f"Dropped non-finite annotated samples: {n_dropped_nonfinite:,}")
    lines.extend(
        [
            f"Training samples used: {n_training_samples:,}",
            f"  Presence: {n_training_presence:,}",
            f"  Background: {n_training_background:,}",
            f"Categorical predictors: {', '.join([band_names[j] for j in categorical_indices]) if categorical_indices else 'None'}",
            "",
            "3. DIAGNOSTICS",
            "-" * 70,
            f"VIF report: {vif_path if vif_path else 'Not generated'}",
        ]
    )

    lines.extend(
        [
            "",
            "4. SPATIAL CROSS-VALIDATION",
            "-" * 70,
        ]
    )
    if mean_auc is not None and std_auc is not None and cv_aucs:
        lines.append(f"Spatial CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        lines.append("Per-fold AUC: " + ", ".join(f"{auc:.3f}" for auc in cv_aucs))
    else:
        lines.append("Spatial CV: not available")

    lines.extend(
        [
            "",
            "5. THRESHOLDS",
            "-" * 70,
            f"Threshold file: {thresh_path}",
            f"5th percentile training presence: {thresholds['p5']:.6f}",
            f"10th percentile training presence: {thresholds['p10']:.6f}",
            f"20th percentile training presence: {thresholds['p20']:.6f}",
            f"Recommended policy-safe suitability threshold: {thresholds['p10']:.6f}",
            "",
            "6. OUTPUTS",
            "-" * 70,
            f"Suitability raster (100m, unmasked): {unmasked_output_path}",
            f"Suitability raster (100m): {output_path}",
            f"Suitability raster (10m): {output_10m_path if output_10m_path else 'Not generated'}",
            f"Eval background raster: {eval_background_raster if eval_background_raster else 'Not generated'}",
            f"Annotated samples: {merged_path}",
            f"Landvalue/LNRS report: {landvalue_report_path if landvalue_report_path else 'Not generated'}",
        ]
    )
    if output_10m_info is not None:
        lines.append(f"10m valid pixels: {int(output_10m_info['valid_pixels']):,}")
        lines.append(f"10m valid area: {float(output_10m_info['valid_area_ha']):,.2f} ha")

    summary_raster = output_10m_path if output_10m_path and os.path.isfile(output_10m_path) else output_path
    peat_summary = summarize_suitability_peat_breakdown(
        summary_raster,
        args.stack,
        suitable_threshold=thresholds.get("p10"),
    )
    if peat_summary is not None:
        lines.extend(
            [
                "",
                "6A. PEAT BREAKDOWN",
                "-" * 70,
                f"Summary raster: {summary_raster}",
                f"Peat definition: {peat_summary['peat_definition']}",
                f"Peat-defined valid pixels: {int(peat_summary['peat_defined_pixels']):,} "
                f"({peat_summary['peat_defined_pct_of_valid']:.2f}% of valid suitability pixels)",
                f"Valid pixels on peat: {int(peat_summary['on_peat_pixels']):,} "
                f"({peat_summary['on_peat_pct_of_peat_defined']:.2f}% of peat-defined valid pixels)",
                f"Valid pixels off peat: {int(peat_summary['off_peat_pixels']):,} "
                f"({peat_summary['off_peat_pct_of_peat_defined']:.2f}% of peat-defined valid pixels)",
            ]
        )
        if "suitable_pixels" in peat_summary:
            lines.extend(
                [
                    f"Suitable threshold used: {peat_summary['suitable_threshold']:.6f}",
                    f"Suitable pixels on peat: {int(peat_summary['suitable_on_peat_pixels']):,} "
                    f"({peat_summary['suitable_on_peat_pct_of_suitable']:.2f}% of suitable pixels)",
                    f"Suitable pixels off peat: {int(peat_summary['suitable_off_peat_pixels']):,} "
                    f"({peat_summary['suitable_off_peat_pct_of_suitable']:.2f}% of suitable pixels)",
                    f"Share of peat pixels that are suitable: {peat_summary['suitable_pct_within_on_peat']:.2f}%",
                    f"Share of off-peat pixels that are suitable: {peat_summary['suitable_pct_within_off_peat']:.2f}%",
                ]
            )

    lines.extend(
        [
            "",
            "7. SHAP IMPORTANCE",
            "-" * 70,
            f"SHAP CSV: {shap_csv_path if shap_csv_path else 'Not generated'}",
            f"SHAP plot: {shap_plot_path if shap_plot_path else 'Not generated'}",
        ]
    )

    if landvalue_report_path and os.path.isfile(landvalue_report_path):
        try:
            zonal_text = Path(landvalue_report_path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            zonal_text = f"[Could not read zonal stats report: {exc}]"
        lines.extend(
            [
                "",
                "8. LANDVALUE AND LNRS ZONAL STATS",
                "-" * 70,
                f"Source file: {landvalue_report_path}",
                "",
                zonal_text,
            ]
        )

    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_masks_to_output(
    output_path: str,
    mask_arrays: List[np.ndarray],
    nodata_value: float = -9999.0,
) -> int:
    """Apply one or more boolean masks to output raster (set masked pixels to nodata)."""
    if not mask_arrays:
        return 0
    with rasterio.open(output_path, "r+") as dst:
        data = dst.read(1, masked=False).astype(np.float32)
        combined = np.zeros((dst.height, dst.width), dtype=bool)
        for m in mask_arrays:
            if m.shape != combined.shape:
                raise ValueError(
                    f"Mask shape {m.shape} does not match output shape {combined.shape}"
                )
            combined |= m
        valid_pred = np.isfinite(data) & (data != nodata_value)
        to_mask = combined & valid_pred
        data[to_mask] = nodata_value
        dst.write(data, 1)
        return int(to_mask.sum())


def annotate_output_raster(
    raster_path: str,
    *,
    band_description: str,
    extra_tags: Optional[Dict[str, object]] = None,
) -> None:
    """Set a band description and lightweight metadata tags on a single-band output raster."""
    tags = {k: str(v) for k, v in (extra_tags or {}).items() if v is not None}
    with rasterio.open(raster_path, "r+") as dst:
        dst.set_band_description(1, band_description)
        if tags:
            dst.update_tags(**tags)


def write_background_eval_raster(
    output_path: str,
    template_raster: str,
    discarded_background_mask: np.ndarray,
    sampled_background_mask: np.ndarray,
    *,
    extra_tags: Optional[Dict[str, object]] = None,
) -> Dict[str, int]:
    """
    Write an evaluation background raster aligned to template_raster.

    Codes:
      0 = discarded / held-out background candidate
      1 = sampled training background
      255 = not a valid background candidate
    """
    with rasterio.open(template_raster) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=255,
            compress="lzw",
        )
        block_x = min(512, int(src.width))
        block_y = min(512, int(src.height))
        block_x = (block_x // 16) * 16
        block_y = (block_y // 16) * 16
        if block_x >= 16 and block_y >= 16:
            profile.update(tiled=True, blockxsize=block_x, blockysize=block_y)
        else:
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile["tiled"] = False

        out = np.full((src.height, src.width), 255, dtype=np.uint8)
        if discarded_background_mask.shape != out.shape:
            raise ValueError(
                f"discarded background mask shape {discarded_background_mask.shape} "
                f"does not match raster shape {out.shape}"
            )
        if sampled_background_mask.shape != out.shape:
            raise ValueError(
                f"sampled background mask shape {sampled_background_mask.shape} "
                f"does not match raster shape {out.shape}"
            )
        out[discarded_background_mask] = 0
        out[sampled_background_mask] = 1

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out, 1)
            dst.set_band_description(1, "eval_background")
            tags = {
                "class_codes": "0=discarded_eval_background,1=sampled_training_background,255=non_background_or_nodata",
                "n_discarded_eval_background": int(discarded_background_mask.sum()),
                "n_sampled_training_background": int(sampled_background_mask.sum()),
            }
            if extra_tags:
                tags.update({k: str(v) for k, v in extra_tags.items() if v is not None})
            dst.update_tags(**tags)

    return {
        "discarded_eval_background": int(discarded_background_mask.sum()),
        "sampled_training_background": int(sampled_background_mask.sum()),
    }


def _alc_group(value: object) -> str:
    # Robust parsing for common formats: 1, 2, 3, 3a, 3b, "Grade 3", 1.0, etc.
    if value is None or _is_nan_number(value):
        return "unknown_grade"
    if isinstance(value, (int, np.integer)):
        grade_num = int(value)
    elif isinstance(value, (float, np.floating)):
        grade_num = int(np.floor(value))
    else:
        s = str(value).strip().lower()
        if not s or s == "nan":
            return "unknown_grade"
        s_norm = re.sub(r"[_\-\s]+", " ", s)
        # Domain labels seen in ALC layers
        if "non agricultural" in s_norm:
            # Treat as effective "grade 6" (lower-priority bucket)
            return "grade_4_to_6"
        if "urban" in s_norm or "exclusion" in s_norm:
            return "unknown_grade"
        m = re.search(r"[1-5]", s)
        if not m:
            return "unknown_grade"
        grade_num = int(m.group(0))
    if grade_num in {1, 2}:
        return "grade_1_to_2"
    if grade_num == 3:
        return "grade_3"
    if grade_num in {4, 5}:
        return "grade_4_to_6"
    return "unknown_grade"


def _resolve_column_name(columns: List[str], requested: str) -> Optional[str]:
    """Find a column by exact or case-insensitive name."""
    if requested in columns:
        return requested
    requested_l = requested.strip().lower()
    for col in columns:
        if col.strip().lower() == requested_l:
            return col
    return None


def _resolve_lnrs_label_field(columns: List[str], requested: Optional[str]) -> Optional[str]:
    """
    Resolve LNRS name/label field.
    If not explicitly provided, try common name-like columns, then first non-geometry column.
    """
    if requested:
        return _resolve_column_name(columns, requested)

    for candidate in (
        "lnrs_name",
        "lnrs",
        "name",
        "region_name",
        "region",
        "area_name",
        "lpa19nm",
        "id",
        "fid",
        "objectid",
    ):
        resolved = _resolve_column_name(columns, candidate)
        if resolved is not None:
            return resolved

    non_geom = [c for c in columns if c.strip().lower() != "geometry"]
    return non_geom[0] if non_geom else None


def _presence_seed_mask_from_array(
    arr1: np.ndarray,
    arr2: Optional[np.ndarray],
    nodata_src: Optional[float],
    threshold: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (seed_mask, valid_mask) from wet woodland arrays."""
    valid = _valid_data_mask(arr1, nodata_src)
    if arr2 is not None and threshold is not None:
        seed = valid & np.isfinite(arr2) & (arr2.astype(np.float32) >= threshold)
    elif threshold is not None:
        if np.issubdtype(arr1.dtype, np.floating):
            seed = valid & (arr1.astype(np.float32) >= threshold)
        else:
            seed = valid & (arr1 == 1)
    else:
        collapsed = _collapse_multiclass_to_binary(arr1, nodata_src)
        if collapsed is not None:
            seed = valid & (collapsed == 1)
        elif np.issubdtype(arr1.dtype, np.integer):
            seed = valid & (arr1 == 1)
        else:
            seed = valid & (arr1 > 0.5)
    return seed, valid


def _adaptive_seed_grid(
    wet_path: str,
    threshold: Optional[float],
    max_cells_for_native: int = 120_000_000,
) -> Tuple[np.ndarray, np.ndarray, rasterio.Affine, object, float, float, str]:
    """
    Build seed/valid masks on a grid suitable for distance transform.
    Uses native raster if feasible; otherwise auto-coarsens to control memory.
    Returns (seed_mask, valid_mask, transform, crs, res_x, res_y, mode).
    """
    with rasterio.open(wet_path) as src:
        nbands = src.count
        nodata_src = getattr(src, "nodata", None)
        src_crs = src.crs
        src_transform = src.transform
        src_cells = int(src.width) * int(src.height)
        res_x_native = abs(src_transform.a)
        res_y_native = abs(src_transform.e)

        if src_cells <= max_cells_for_native:
            arr1 = src.read(1, masked=False)
            arr2 = src.read(2, masked=False) if nbands >= 2 else None
            seed, valid = _presence_seed_mask_from_array(arr1, arr2, nodata_src, threshold)
            return (
                seed.astype(bool),
                valid.astype(bool),
                src_transform,
                src_crs,
                float(res_x_native),
                float(res_y_native),
                "native",
            )

        # Auto-coarsen very large rasters to keep distance transform feasible.
        factor = int(np.ceil(np.sqrt(src_cells / float(max_cells_for_native))))
        factor = max(2, factor)
        res_x = float(res_x_native * factor)
        res_y = float(res_y_native * factor)
        bounds = src.bounds
        width = int(np.ceil((bounds.right - bounds.left) / res_x))
        height = int(np.ceil((bounds.top - bounds.bottom) / res_y))
        dst_transform = from_origin(bounds.left, bounds.top, res_x, res_y)

        # valid-any from band 1 min/max against nodata
        if nodata_src is None:
            valid_any = np.ones((height, width), dtype=bool)
        else:
            b1_min = np.full((height, width), nodata_src, dtype=np.float32)
            b1_max = np.full((height, width), nodata_src, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=b1_min,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=src_crs,
                resampling=Resampling.min,
            )
            reproject(
                source=rasterio.band(src, 1),
                destination=b1_max,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=src_crs,
                resampling=Resampling.max,
            )
            valid_any = (b1_min != nodata_src) | (b1_max != nodata_src)

        if nbands >= 2 and threshold is not None:
            arr2_max = np.full((height, width), -np.inf, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 2),
                destination=arr2_max,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=src_crs,
                resampling=Resampling.max,
            )
            seed = valid_any & (arr2_max >= threshold)
        else:
            arr1_max = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=arr1_max,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=src_crs,
                resampling=Resampling.max,
            )
            # keep existing behaviour: int single-band -> binary 1; float -> threshold or >0.5
            if threshold is not None and np.issubdtype(src.dtypes[0], np.floating):
                seed = valid_any & (arr1_max >= threshold)
            elif threshold is not None:
                seed = valid_any & (arr1_max == 1)
            elif np.issubdtype(src.dtypes[0], np.integer):
                seed = valid_any & (arr1_max == 1)
            else:
                seed = valid_any & (arr1_max > 0.5)

    return seed.astype(bool), valid_any.astype(bool), dst_transform, src_crs, res_x, res_y, "coarsened"


def compute_seed_distance_to_suitability_grid(
    wet_source_path: str,
    suitability_path: str,
    wet_threshold: Optional[float],
    max_cells_for_native: int = 120_000_000,
) -> Tuple[np.ndarray, str]:
    """
    Compute distance (meters) from each suitability pixel to nearest wet woodland seed pixel.
    Uses adaptive source resolution to stay memory-safe on very large rasters.
    Returns (distance_on_suit_grid, mode_used).
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as e:
        raise ImportError(
            "scipy is required for seed-distance stats. Install with: pip install scipy"
        ) from e

    seed, valid_src, src_transform, src_crs, res_x, res_y, mode = _adaptive_seed_grid(
        wet_source_path,
        threshold=wet_threshold,
        max_cells_for_native=max_cells_for_native,
    )
    if not np.any(seed):
        raise ValueError("No wet woodland seed pixels found in source raster for distance stats.")

    not_seed = ~seed
    dist_src = distance_transform_edt(not_seed, sampling=(res_y, res_x)).astype(np.float32)
    # Keep only source-valid domain
    dist_src[~valid_src] = np.nan

    with rasterio.open(suitability_path) as suit_src:
        out_shape = (suit_src.height, suit_src.width)
        dst_transform = suit_src.transform
        dst_crs = suit_src.crs

    dist_dst = np.full(out_shape, np.nan, dtype=np.float32)
    reproject(
        source=dist_src,
        destination=dist_dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.min,
    )
    return dist_dst, mode


def summarize_suitability_peat_breakdown(
    suitability_path: str,
    stack_path: str,
    *,
    suitable_threshold: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """Summarize valid and suitable pixels on peat vs off-peat for a suitability raster."""
    if not os.path.isfile(suitability_path) or not os.path.isfile(stack_path):
        return None

    with rasterio.open(suitability_path) as suit_src:
        suit = suit_src.read(1, masked=False).astype(np.float64)
        valid_suit = _valid_data_mask(suit, suit_src.nodata)
        transform = suit_src.transform
        crs = suit_src.crs
        out_shape = (suit_src.height, suit_src.width)

    if not np.any(valid_suit):
        return None

    with rasterio.open(stack_path) as stack_src:
        peat_band_idx, peat_mode, peat_desc = _find_peat_band_info(list(stack_src.descriptions))
        if peat_band_idx is None:
            return None

        peat_src = stack_src.read(peat_band_idx, masked=False).astype(np.float32)
        peat_src = np.where(
            _valid_data_mask(peat_src, stack_src.nodata),
            peat_src,
            np.nan,
        )

        peat_aligned = np.full(out_shape, np.nan, dtype=np.float32)
        reproject(
            source=peat_src,
            destination=peat_aligned,
            src_transform=stack_src.transform,
            src_crs=stack_src.crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    peat_valid = np.isfinite(peat_aligned) & valid_suit
    if not np.any(peat_valid):
        return None

    on_peat = peat_valid & _peat_extent_mask(peat_aligned, peat_mode)
    off_peat = peat_valid & ~on_peat

    on_peat_n = int(on_peat.sum())
    off_peat_n = int(off_peat.sum())
    peat_valid_n = int(peat_valid.sum())
    valid_suit_n = int(valid_suit.sum())

    summary: Dict[str, float] = {
        "peat_definition": _peat_definition_label(peat_mode, peat_desc),
        "valid_pixels": float(valid_suit_n),
        "peat_defined_pixels": float(peat_valid_n),
        "on_peat_pixels": float(on_peat_n),
        "off_peat_pixels": float(off_peat_n),
        "peat_defined_pct_of_valid": (100.0 * peat_valid_n / valid_suit_n) if valid_suit_n > 0 else 0.0,
        "on_peat_pct_of_peat_defined": (100.0 * on_peat_n / peat_valid_n) if peat_valid_n > 0 else 0.0,
        "off_peat_pct_of_peat_defined": (100.0 * off_peat_n / peat_valid_n) if peat_valid_n > 0 else 0.0,
    }

    if suitable_threshold is not None:
        suitable = valid_suit & (suit >= suitable_threshold)
        suitable_n = int(suitable.sum())
        suitable_on_peat_n = int((suitable & on_peat).sum())
        suitable_off_peat_n = int((suitable & off_peat).sum())
        summary.update(
            {
                "suitable_threshold": float(suitable_threshold),
                "suitable_pixels": float(suitable_n),
                "suitable_on_peat_pixels": float(suitable_on_peat_n),
                "suitable_off_peat_pixels": float(suitable_off_peat_n),
                "suitable_on_peat_pct_of_suitable": (100.0 * suitable_on_peat_n / suitable_n) if suitable_n > 0 else 0.0,
                "suitable_off_peat_pct_of_suitable": (100.0 * suitable_off_peat_n / suitable_n) if suitable_n > 0 else 0.0,
                "suitable_pct_within_on_peat": (100.0 * suitable_on_peat_n / on_peat_n) if on_peat_n > 0 else 0.0,
                "suitable_pct_within_off_peat": (100.0 * suitable_off_peat_n / off_peat_n) if off_peat_n > 0 else 0.0,
            }
        )

    return summary


def write_landvalue_group_stats(
    suitability_path: str,
    landvalue_shp: str,
    report_path: str,
    grade_field: str = "ALC_grade",
    high_potential_threshold: Optional[float] = None,
    seed_source_path: Optional[str] = None,
    seed_threshold: Optional[float] = None,
    seed_distance_m: float = 100.0,
    stack_path: Optional[str] = None,
    wet_woodland_path: Optional[str] = None,
    suitable_threshold: Optional[float] = None,
    lnrs_shp: Optional[str] = None,
    lnrs_label_field: Optional[str] = None,
) -> None:
    """
    Group ALC polygons into grade_1_to_2, grade_3, and grade_4_to_6,
    dissolve each group, and
    compute zonal stats on suitability map, and write a plain-text report.

    Additional breakdowns:
    - Peat vs off-peat (using peat depth extent from the stack where available)
    - Forest vs bare land (using wet woodland raster valid mask)
    - Per-LNRS summary with peat and forest splits (if LNRS polygons provided)
    """
    with rasterio.open(suitability_path) as src:
        suit = src.read(1, masked=False).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        out_shape = (src.height, src.width)

    valid_suit = _valid_data_mask(suit, nodata)
    pixel_area_m2 = abs(transform.a * transform.e)
    candidate_total_pixels = int(valid_suit.sum())
    candidate_total_area_km2 = (candidate_total_pixels * pixel_area_m2) / 1_000_000.0
    candidate_total_area_ha = (candidate_total_pixels * pixel_area_m2) / 10_000.0

    alc_gdf = gpd.read_file(landvalue_shp)
    resolved_grade_field = _resolve_column_name(list(alc_gdf.columns), grade_field)
    if resolved_grade_field is None:
        # Common fallback aliases so users don't need exact casing.
        for alias in ("alc_grade", "ALC_grade", "alcgrade", "ALCGrade"):
            resolved_grade_field = _resolve_column_name(list(alc_gdf.columns), alias)
            if resolved_grade_field is not None:
                break
    if resolved_grade_field is None:
        raise ValueError(
            f"Field '{grade_field}' not found in {landvalue_shp}. "
            f"Available: {list(alc_gdf.columns)}"
        )
    alc_gdf = alc_gdf.loc[alc_gdf.geometry.notna()].copy()
    alc_gdf = alc_gdf.loc[~alc_gdf.geometry.is_empty].copy()
    if alc_gdf.empty:
        raise ValueError(f"No valid geometries found in {landvalue_shp}")
    if alc_gdf.crs is None:
        raise ValueError(f"Landvalue shapefile has no CRS: {landvalue_shp}")
    if str(alc_gdf.crs) != str(crs):
        alc_gdf = alc_gdf.to_crs(crs)

    alc_gdf["group"] = alc_gdf[resolved_grade_field].apply(_alc_group)
    group_counts = alc_gdf["group"].value_counts(dropna=False).to_dict()
    n12 = int(group_counts.get("grade_1_to_2", 0))
    n3 = int(group_counts.get("grade_3", 0))
    n46 = int(group_counts.get("grade_4_to_6", 0))
    n_unknown = int(group_counts.get("unknown_grade", 0))
    if (n12 + n3 + n46) == 0:
        sample_vals = (
            alc_gdf[resolved_grade_field]
            .dropna()
            .astype(str)
            .head(20)
            .tolist()
        )
        raise ValueError(
            "Could not map any landvalue grades into 1-2, 3, or 4-6 groups. "
            f"Field '{resolved_grade_field}' sample values: {sample_vals}"
        )
    dissolved = alc_gdf.dissolve(by="group", as_index=False)
    dissolved_map = {str(row["group"]): row.geometry for _, row in dissolved.iterrows()}

    lines: List[str] = []
    lines.append("Wet woodland potential zonal statistics by land grade group")
    lines.append(f"Suitability raster: {suitability_path}")
    lines.append(f"Land value polygons: {landvalue_shp}")
    lines.append(f"Grouping field: {resolved_grade_field}")
    lines.append(
        "Grouping rule: grades 1-2 -> grade_1_to_2, grade 3 -> grade_3, "
        "grades 4-5 plus Non_Agricultural(as grade 6) -> grade_4_to_6; "
        "Urban/Exclusion -> unknown"
    )
    if high_potential_threshold is not None:
        lines.append(f"High potential threshold: {high_potential_threshold}")
    if seed_source_path:
        lines.append(f"Seed source raster: {seed_source_path}")
        lines.append(f"Seed distance threshold (m): {seed_distance_m}")
    if lnrs_shp:
        lines.append(f"LNRS polygons: {lnrs_shp}")
    lines.append("")
    lines.append("[candidate_overall]")
    lines.append(f"  candidate_valid_pixels: {candidate_total_pixels}")
    lines.append(f"  candidate_valid_area_km2: {candidate_total_area_km2:.3f}")
    lines.append(f"  candidate_valid_area_ha: {candidate_total_area_ha:.3f}")
    lines.append("")
    lines.append("[landvalue_grouping_qc]")
    lines.append(f"  source_feature_count: {len(alc_gdf)}")
    lines.append(f"  mapped_grade_1_to_2_features: {n12}")
    lines.append(f"  mapped_grade_3_features: {n3}")
    lines.append(f"  mapped_grade_4_to_6_features: {n46}")
    lines.append(f"  unknown_grade_features: {n_unknown}")
    lines.append("")

    seed_distance = None
    on_peat: Optional[np.ndarray] = None
    off_peat: Optional[np.ndarray] = None
    under_forest: Optional[np.ndarray] = None
    bare_land: Optional[np.ndarray] = None

    if seed_source_path:
        seed_distance, mode = compute_seed_distance_to_suitability_grid(
            seed_source_path,
            suitability_path,
            wet_threshold=seed_threshold,
        )
        cand_valid = valid_suit & np.isfinite(seed_distance)
        cand_n = int(cand_valid.sum())
        within = cand_valid & (seed_distance <= float(seed_distance_m))
        within_n = int(within.sum())
        within_pct = (100.0 * within_n / cand_n) if cand_n > 0 else 0.0
        within_area_km2 = (within_n * pixel_area_m2) / 1_000_000.0
        within_area_ha = (within_n * pixel_area_m2) / 10_000.0
        lines.append("[seed_proximity_overall]")
        lines.append(f"  source_grid_mode: {mode}")
        lines.append(f"  candidate_valid_pixels: {cand_n}")
        lines.append(f"  within_{int(seed_distance_m)}m_pixels: {within_n}")
        lines.append(f"  within_{int(seed_distance_m)}m_pct_of_candidate: {within_pct:.3f}")
        lines.append(f"  within_{int(seed_distance_m)}m_area_km2: {within_area_km2:.3f}")
        lines.append(f"  within_{int(seed_distance_m)}m_area_ha: {within_area_ha:.3f}")
        lines.append("")

    # Peat vs off-peat breakdown
    if stack_path and os.path.isfile(stack_path):
        try:
            with rasterio.open(stack_path) as stack_src:
                peat_band_idx, peat_mode, peat_desc = _find_peat_band_info(list(stack_src.descriptions))
                if peat_band_idx is not None:
                    peat_src = stack_src.read(peat_band_idx, masked=False).astype(np.float32)

                    # Align peat predictor to suitability grid before pixel-wise math.
                    stack_nodata = getattr(stack_src, "nodata", None)
                    if stack_nodata is not None:
                        stack_valid = stack_src.read(1, masked=False) != stack_nodata
                        peat_src = np.where(stack_valid, peat_src, np.nan)
                    else:
                        peat_src = np.where(np.isfinite(peat_src), peat_src, np.nan)

                    peat_aligned = np.full(out_shape, np.nan, dtype=np.float32)
                    reproject(
                        source=peat_src,
                        destination=peat_aligned,
                        src_transform=stack_src.transform,
                        src_crs=stack_src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        src_nodata=np.nan,
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )

                    peat_valid = np.isfinite(peat_aligned) & valid_suit
                    on_peat = peat_valid & _peat_extent_mask(peat_aligned, peat_mode)
                    off_peat = peat_valid & ~on_peat

                    on_peat_n = int(on_peat.sum())
                    off_peat_n = int(off_peat.sum())
                    peat_definition = _peat_definition_label(peat_mode, peat_desc)

                    lines.append("[peat_breakdown]")
                    lines.append(f"  peat_definition: {peat_definition}")
                    lines.append(f"  on_peat_pixels: {on_peat_n}")
                    lines.append(f"  on_peat_area_ha: {(on_peat_n * pixel_area_m2) / 10_000.0:.3f}")
                    lines.append(f"  on_peat_pct_of_candidate: {(100.0 * on_peat_n / candidate_total_pixels) if candidate_total_pixels > 0 else 0.0:.3f}")
                    lines.append(f"  off_peat_pixels: {off_peat_n}")
                    lines.append(f"  off_peat_area_ha: {(off_peat_n * pixel_area_m2) / 10_000.0:.3f}")
                    lines.append(f"  off_peat_pct_of_candidate: {(100.0 * off_peat_n / candidate_total_pixels) if candidate_total_pixels > 0 else 0.0:.3f}")

                    # Suitable area breakdown by peat (using suitable_threshold)
                    if suitable_threshold is not None:
                        suit_on_peat = on_peat & (suit >= suitable_threshold)
                        suit_off_peat = off_peat & (suit >= suitable_threshold)
                        suit_on_peat_n = int(suit_on_peat.sum())
                        suit_off_peat_n = int(suit_off_peat.sum())
                        lines.append(f"  suitable_on_peat_pixels (>= {suitable_threshold:.2f}): {suit_on_peat_n}")
                        lines.append(f"  suitable_on_peat_area_ha: {(suit_on_peat_n * pixel_area_m2) / 10_000.0:.3f}")
                        lines.append(f"  suitable_off_peat_pixels (>= {suitable_threshold:.2f}): {suit_off_peat_n}")
                        lines.append(f"  suitable_off_peat_area_ha: {(suit_off_peat_n * pixel_area_m2) / 10_000.0:.3f}")

                    if high_potential_threshold is not None:
                        high_on_peat = on_peat & (suit >= high_potential_threshold)
                        high_off_peat = off_peat & (suit >= high_potential_threshold)
                        high_on_peat_n = int(high_on_peat.sum())
                        high_off_peat_n = int(high_off_peat.sum())
                        lines.append(f"  high_potential_on_peat_pixels (>= {high_potential_threshold}): {high_on_peat_n}")
                        lines.append(f"  high_potential_on_peat_area_ha: {(high_on_peat_n * pixel_area_m2) / 10_000.0:.3f}")
                        lines.append(f"  high_potential_off_peat_pixels (>= {high_potential_threshold}): {high_off_peat_n}")
                        lines.append(f"  high_potential_off_peat_area_ha: {(high_off_peat_n * pixel_area_m2) / 10_000.0:.3f}")

                    lines.append("")
        except Exception as e:
            lines.append(f"[peat_breakdown] Error: {e}")
            lines.append("")

    # Forest vs bare land breakdown
    if wet_woodland_path and os.path.isfile(wet_woodland_path):
        try:
            # Align wet woodland valid mask to suitability grid
            forest_mask = align_wet_valid_mask_to_stack(
                wet_woodland_path,
                {
                    "transform": transform,
                    "crs": crs,
                    "height": out_shape[0],
                    "width": out_shape[1],
                },
            )
            # Forest = valid pixels in wet woodland raster (entire forest domain)
            # Bare land = suitability pixels NOT in forest domain
            under_forest = valid_suit & forest_mask
            bare_land = valid_suit & ~forest_mask

            under_forest_n = int(under_forest.sum())
            bare_land_n = int(bare_land.sum())

            lines.append("[land_cover_breakdown]")
            lines.append(f"  under_existing_forest_pixels: {under_forest_n}")
            lines.append(f"  under_existing_forest_area_ha: {(under_forest_n * pixel_area_m2) / 10_000.0:.3f}")
            lines.append(f"  under_existing_forest_pct_of_candidate: {(100.0 * under_forest_n / candidate_total_pixels) if candidate_total_pixels > 0 else 0.0:.3f}")
            lines.append(f"  bare_open_land_pixels: {bare_land_n}")
            lines.append(f"  bare_open_land_area_ha: {(bare_land_n * pixel_area_m2) / 10_000.0:.3f}")
            lines.append(f"  bare_open_land_pct_of_candidate: {(100.0 * bare_land_n / candidate_total_pixels) if candidate_total_pixels > 0 else 0.0:.3f}")

            # Suitable area breakdown by land cover
            if suitable_threshold is not None:
                suit_forest = under_forest & (suit >= suitable_threshold)
                suit_bare = bare_land & (suit >= suitable_threshold)
                suit_forest_n = int(suit_forest.sum())
                suit_bare_n = int(suit_bare.sum())
                lines.append(f"  suitable_under_forest_pixels (>= {suitable_threshold:.2f}): {suit_forest_n}")
                lines.append(f"  suitable_under_forest_area_ha: {(suit_forest_n * pixel_area_m2) / 10_000.0:.3f}")
                lines.append(f"  suitable_bare_land_pixels (>= {suitable_threshold:.2f}): {suit_bare_n}")
                lines.append(f"  suitable_bare_land_area_ha: {(suit_bare_n * pixel_area_m2) / 10_000.0:.3f}")

            if high_potential_threshold is not None:
                high_forest = under_forest & (suit >= high_potential_threshold)
                high_bare = bare_land & (suit >= high_potential_threshold)
                high_forest_n = int(high_forest.sum())
                high_bare_n = int(high_bare.sum())
                lines.append(f"  high_potential_under_forest_pixels (>= {high_potential_threshold}): {high_forest_n}")
                lines.append(f"  high_potential_under_forest_area_ha: {(high_forest_n * pixel_area_m2) / 10_000.0:.3f}")
                lines.append(f"  high_potential_bare_land_pixels (>= {high_potential_threshold}): {high_bare_n}")
                lines.append(f"  high_potential_bare_land_area_ha: {(high_bare_n * pixel_area_m2) / 10_000.0:.3f}")

            lines.append("")
        except Exception as e:
            lines.append(f"[land_cover_breakdown] Error: {e}")
            lines.append("")

    if lnrs_shp and os.path.isfile(lnrs_shp):
        try:
            lnrs_gdf = gpd.read_file(lnrs_shp)
            lnrs_gdf = lnrs_gdf.loc[lnrs_gdf.geometry.notna()].copy()
            lnrs_gdf = lnrs_gdf.loc[~lnrs_gdf.geometry.is_empty].copy()
            if lnrs_gdf.empty:
                raise ValueError(f"No valid geometries found in {lnrs_shp}")
            if lnrs_gdf.crs is None:
                raise ValueError(f"LNRS shapefile has no CRS: {lnrs_shp}")
            if str(lnrs_gdf.crs) != str(crs):
                lnrs_gdf = lnrs_gdf.to_crs(crs)

            resolved_lnrs_label = _resolve_lnrs_label_field(
                list(lnrs_gdf.columns), lnrs_label_field
            )
            if resolved_lnrs_label is None:
                raise ValueError(
                    f"Could not resolve LNRS label field. Available: {list(lnrs_gdf.columns)}"
                )

            lines.append("[lnrs_summary]")
            lines.append(f"  lnrs_source: {lnrs_shp}")
            lines.append(f"  lnrs_label_field: {resolved_lnrs_label}")
            lines.append(f"  lnrs_feature_count: {len(lnrs_gdf)}")
            if suitable_threshold is not None:
                lines.append(f"  suitable_threshold: {suitable_threshold:.6f}")
            lines.append("")

            seen_labels = set()
            for row_idx, row in lnrs_gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                label_val = row.get(resolved_lnrs_label)
                if label_val is None or (
                    isinstance(label_val, (float, np.floating)) and np.isnan(label_val)
                ):
                    label_raw = f"feature_{row_idx + 1}"
                else:
                    label_raw = str(label_val).strip()
                    if not label_raw or label_raw.lower() == "nan":
                        label_raw = f"feature_{row_idx + 1}"

                safe_base = re.sub(r"[^0-9A-Za-z_]+", "_", label_raw).strip("_")
                if not safe_base:
                    safe_base = f"feature_{row_idx + 1}"
                safe_label = safe_base
                suffix = 2
                while safe_label in seen_labels:
                    safe_label = f"{safe_base}_{suffix}"
                    suffix += 1
                seen_labels.add(safe_label)

                zone = rasterize(
                    [(geom, 1)],
                    out_shape=out_shape,
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                ).astype(bool)
                zone_valid = zone & valid_suit

                total_pix = int(zone.sum())
                valid_pix = int(zone_valid.sum())
                total_area_ha = (total_pix * pixel_area_m2) / 10_000.0
                valid_area_ha = (valid_pix * pixel_area_m2) / 10_000.0
                valid_pct_of_zone = (100.0 * valid_pix / total_pix) if total_pix > 0 else 0.0

                lines.append(f"[lnrs:{safe_label}]")
                lines.append(f"  name: {label_raw}")
                lines.append(f"  total_zone_pixels: {total_pix}")
                lines.append(f"  total_zone_area_ha: {total_area_ha:.3f}")
                lines.append(f"  valid_suitability_pixels: {valid_pix}")
                lines.append(f"  valid_suitability_area_ha: {valid_area_ha:.3f}")
                lines.append(f"  valid_suitability_pct_of_zone: {valid_pct_of_zone:.3f}")

                if valid_pix > 0:
                    vals = suit[zone_valid]
                    lines.append(f"  mean: {float(np.mean(vals)):.6f}")
                    lines.append(f"  median: {float(np.median(vals)):.6f}")
                    lines.append(f"  min: {float(np.min(vals)):.6f}")
                    lines.append(f"  max: {float(np.max(vals)):.6f}")
                else:
                    lines.append("  mean: n/a")
                    lines.append("  median: n/a")
                    lines.append("  min: n/a")
                    lines.append("  max: n/a")

                if suitable_threshold is not None:
                    suitable_zone = zone_valid & (suit >= suitable_threshold)
                    suitable_n = int(suitable_zone.sum())
                    suitable_area_ha = (suitable_n * pixel_area_m2) / 10_000.0
                    suitable_pct_of_valid = (100.0 * suitable_n / valid_pix) if valid_pix > 0 else 0.0
                    lines.append(
                        f"  suitable_pixels (>= {suitable_threshold:.2f}): {suitable_n}"
                    )
                    lines.append(f"  suitable_area_ha: {suitable_area_ha:.3f}")
                    lines.append(f"  suitable_pct_of_valid: {suitable_pct_of_valid:.3f}")

                if on_peat is not None and off_peat is not None:
                    lnrs_on_peat = zone_valid & on_peat
                    lnrs_off_peat = zone_valid & off_peat
                    lnrs_on_peat_n = int(lnrs_on_peat.sum())
                    lnrs_off_peat_n = int(lnrs_off_peat.sum())
                    lines.append(f"  on_peat_pixels: {lnrs_on_peat_n}")
                    lines.append(
                        f"  on_peat_area_ha: {(lnrs_on_peat_n * pixel_area_m2) / 10_000.0:.3f}"
                    )
                    lines.append(
                        f"  on_peat_pct_of_valid: {(100.0 * lnrs_on_peat_n / valid_pix) if valid_pix > 0 else 0.0:.3f}"
                    )
                    lines.append(f"  off_peat_pixels: {lnrs_off_peat_n}")
                    lines.append(
                        f"  off_peat_area_ha: {(lnrs_off_peat_n * pixel_area_m2) / 10_000.0:.3f}"
                    )
                    lines.append(
                        f"  off_peat_pct_of_valid: {(100.0 * lnrs_off_peat_n / valid_pix) if valid_pix > 0 else 0.0:.3f}"
                    )
                    if suitable_threshold is not None:
                        lnrs_suitable_on_peat = lnrs_on_peat & (suit >= suitable_threshold)
                        lnrs_suitable_off_peat = lnrs_off_peat & (suit >= suitable_threshold)
                        lnrs_suitable_on_peat_n = int(lnrs_suitable_on_peat.sum())
                        lnrs_suitable_off_peat_n = int(lnrs_suitable_off_peat.sum())
                        lines.append(
                            f"  suitable_on_peat_pixels (>= {suitable_threshold:.2f}): {lnrs_suitable_on_peat_n}"
                        )
                        lines.append(
                            f"  suitable_on_peat_area_ha: {(lnrs_suitable_on_peat_n * pixel_area_m2) / 10_000.0:.3f}"
                        )
                        lines.append(
                            f"  suitable_off_peat_pixels (>= {suitable_threshold:.2f}): {lnrs_suitable_off_peat_n}"
                        )
                        lines.append(
                            f"  suitable_off_peat_area_ha: {(lnrs_suitable_off_peat_n * pixel_area_m2) / 10_000.0:.3f}"
                        )
                else:
                    lines.append("  peat_breakdown: unavailable")

                if under_forest is not None and bare_land is not None:
                    lnrs_under_forest = zone_valid & under_forest
                    lnrs_bare_open = zone_valid & bare_land
                    lnrs_under_forest_n = int(lnrs_under_forest.sum())
                    lnrs_bare_open_n = int(lnrs_bare_open.sum())
                    lines.append(f"  under_existing_forest_pixels: {lnrs_under_forest_n}")
                    lines.append(
                        f"  under_existing_forest_area_ha: {(lnrs_under_forest_n * pixel_area_m2) / 10_000.0:.3f}"
                    )
                    lines.append(
                        f"  under_existing_forest_pct_of_valid: {(100.0 * lnrs_under_forest_n / valid_pix) if valid_pix > 0 else 0.0:.3f}"
                    )
                    lines.append(f"  bare_open_land_pixels: {lnrs_bare_open_n}")
                    lines.append(
                        f"  bare_open_land_area_ha: {(lnrs_bare_open_n * pixel_area_m2) / 10_000.0:.3f}"
                    )
                    lines.append(
                        f"  bare_open_land_pct_of_valid: {(100.0 * lnrs_bare_open_n / valid_pix) if valid_pix > 0 else 0.0:.3f}"
                    )
                    if suitable_threshold is not None:
                        lnrs_suitable_forest = lnrs_under_forest & (suit >= suitable_threshold)
                        lnrs_suitable_bare = lnrs_bare_open & (suit >= suitable_threshold)
                        lnrs_suitable_forest_n = int(lnrs_suitable_forest.sum())
                        lnrs_suitable_bare_n = int(lnrs_suitable_bare.sum())
                        lines.append(
                            f"  suitable_under_forest_pixels (>= {suitable_threshold:.2f}): {lnrs_suitable_forest_n}"
                        )
                        lines.append(
                            f"  suitable_under_forest_area_ha: {(lnrs_suitable_forest_n * pixel_area_m2) / 10_000.0:.3f}"
                        )
                        lines.append(
                            f"  suitable_bare_land_pixels (>= {suitable_threshold:.2f}): {lnrs_suitable_bare_n}"
                        )
                        lines.append(
                            f"  suitable_bare_land_area_ha: {(lnrs_suitable_bare_n * pixel_area_m2) / 10_000.0:.3f}"
                        )
                else:
                    lines.append("  land_cover_breakdown: unavailable")

                if seed_distance is not None:
                    lnrs_candidate = zone_valid & np.isfinite(seed_distance)
                    lnrs_n = int(lnrs_candidate.sum())
                    lnrs_within = lnrs_candidate & (seed_distance <= float(seed_distance_m))
                    lnrs_within_n = int(lnrs_within.sum())
                    lnrs_within_pct = (100.0 * lnrs_within_n / lnrs_n) if lnrs_n > 0 else 0.0
                    lines.append(f"  within_{int(seed_distance_m)}m_pixels: {lnrs_within_n}")
                    lines.append(
                        f"  within_{int(seed_distance_m)}m_pct_of_valid: {lnrs_within_pct:.3f}"
                    )

                lines.append("")
        except Exception as e:
            lines.append(f"[lnrs_summary] Error: {e}")
            lines.append("")

    for group in ("grade_1_to_2", "grade_3", "grade_4_to_6"):
        geom = dissolved_map.get(group)
        if geom is None or geom.is_empty:
            zone = np.zeros(out_shape, dtype=bool)
        else:
            zone = rasterize(
                [(geom, 1)],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
        zone_valid = zone & valid_suit
        total_pix = int(zone.sum())
        valid_pix = int(zone_valid.sum())
        total_area_km2 = (total_pix * pixel_area_m2) / 1_000_000.0
        valid_area_km2 = (valid_pix * pixel_area_m2) / 1_000_000.0
        total_area_ha = (total_pix * pixel_area_m2) / 10_000.0
        valid_area_ha = (valid_pix * pixel_area_m2) / 10_000.0
        valid_pct_of_zone = (100.0 * valid_pix / total_pix) if total_pix > 0 else 0.0
        valid_pct_of_candidate = (
            (100.0 * valid_pix / candidate_total_pixels) if candidate_total_pixels > 0 else 0.0
        )

        lines.append(f"[{group}]")
        lines.append(f"  total_zone_pixels: {total_pix}")
        lines.append(f"  valid_suitability_pixels: {valid_pix}")
        lines.append(f"  valid_suitability_pct_of_zone: {valid_pct_of_zone:.3f}")
        lines.append(f"  valid_suitability_pct_of_candidate: {valid_pct_of_candidate:.3f}")
        lines.append(f"  total_zone_area_km2: {total_area_km2:.3f}")
        lines.append(f"  total_zone_area_ha: {total_area_ha:.3f}")
        lines.append(f"  valid_suitability_area_km2: {valid_area_km2:.3f}")
        lines.append(f"  valid_suitability_area_ha: {valid_area_ha:.3f}")

        if valid_pix == 0:
            lines.append("  mean: n/a")
            lines.append("  median: n/a")
            lines.append("  min: n/a")
            lines.append("  max: n/a")
            lines.append("")
            continue

        vals = suit[zone_valid]
        lines.append(f"  mean: {float(np.mean(vals)):.6f}")
        lines.append(f"  median: {float(np.median(vals)):.6f}")
        lines.append(f"  min: {float(np.min(vals)):.6f}")
        lines.append(f"  max: {float(np.max(vals)):.6f}")

        if seed_distance is not None:
            grp_candidate = zone_valid & np.isfinite(seed_distance)
            grp_n = int(grp_candidate.sum())
            grp_within = grp_candidate & (seed_distance <= float(seed_distance_m))
            grp_within_n = int(grp_within.sum())
            grp_within_pct = (100.0 * grp_within_n / grp_n) if grp_n > 0 else 0.0
            grp_within_pct_zone = (100.0 * grp_within_n / total_pix) if total_pix > 0 else 0.0
            grp_within_km2 = (grp_within_n * pixel_area_m2) / 1_000_000.0
            grp_within_ha = (grp_within_n * pixel_area_m2) / 10_000.0
            lines.append(f"  within_{int(seed_distance_m)}m_pixels: {grp_within_n}")
            lines.append(f"  within_{int(seed_distance_m)}m_pct_of_valid: {grp_within_pct:.3f}")
            lines.append(f"  within_{int(seed_distance_m)}m_pct_of_zone: {grp_within_pct_zone:.3f}")
            lines.append(f"  within_{int(seed_distance_m)}m_area_km2: {grp_within_km2:.3f}")
            lines.append(f"  within_{int(seed_distance_m)}m_area_ha: {grp_within_ha:.3f}")

        if high_potential_threshold is not None:
            high_count = int(np.sum(vals >= high_potential_threshold))
            high_pct = (100.0 * high_count / valid_pix) if valid_pix > 0 else 0.0
            high_pct_zone = (100.0 * high_count / total_pix) if total_pix > 0 else 0.0
            high_area_km2 = (high_count * pixel_area_m2) / 1_000_000.0
            high_area_ha = (high_count * pixel_area_m2) / 10_000.0
            lines.append(f"  high_potential_pixels: {high_count}")
            lines.append(f"  high_potential_pct_of_valid: {high_pct:.3f}")
            lines.append(f"  high_potential_pct_of_zone: {high_pct_zone:.3f}")
            lines.append(f"  high_potential_area_km2: {high_area_km2:.3f}")
            lines.append(f"  high_potential_area_ha: {high_area_ha:.3f}")
        lines.append("")

    _ensure_dir(str(Path(report_path).parent))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _collapse_multiclass_to_binary(arr: np.ndarray, nodata_src: Optional[float]) -> Optional[np.ndarray]:
    """
    If arr contains values > 1 (multi-class label schema 0/1/2/3), collapse to binary:
      0,1 → 0 (background)  |  2,3 → 1 (wet woodland presence)
    Returns collapsed array, or None if arr is already binary/float (no collapse needed).
    """
    if not np.issubdtype(arr.dtype, np.integer):
        return None
    unique_vals = np.unique(arr[arr != nodata_src] if nodata_src is not None else arr)
    if unique_vals.max() <= 1:
        return None  # already binary
    binary = np.where(arr >= 2, 1, 0).astype(np.uint8)
    if nodata_src is not None:
        binary[arr == int(nodata_src)] = 0
    print(f"    Multi-class labels detected (max={unique_vals.max()}): collapsing 0/1→0, 2/3→1")
    return binary


def align_wet_woodland_to_stack(
    wet_path: str,
    stack_profile: dict,
    threshold: Optional[float],
) -> np.ndarray:
    """
    Load wet woodland raster, apply threshold if given (band 2 prob, or band 1 if
    only one band), reproject to stack grid. Returns binary array (1 = presence,
    0 = absence).

    Multi-class label rasters (0/1/2/3 schema) are automatically collapsed:
    classes 0/1 → background, classes 2/3 → presence.
    """
    with rasterio.open(wet_path) as src:
        nbands = src.count
        nodata_src = getattr(src, "nodata", 255)
        arr1 = src.read(1, masked=False)
        arr2 = src.read(2, masked=False) if nbands >= 2 else None
        presence_mask, valid = _presence_seed_mask_from_array(
            arr1,
            arr2,
            nodata_src,
            threshold,
        )
        presence_src = np.where(valid, presence_mask.astype(np.uint8), 0)
        src_transform = src.transform
        src_crs = src.crs

    dst_transform = stack_profile["transform"]
    dst_crs = stack_profile["crs"]
    dst_height = stack_profile["height"]
    dst_width = stack_profile["width"]
    out = np.zeros((dst_height, dst_width), dtype=np.uint8)
    reproject(
        source=presence_src,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.max,  # Use max so ANY wet woodland pixel in cell → masked
    )
    return out


def create_10m_suitability(
    suitability_100m_path: str,
    wet_woodland_path: str,
    output_10m_path: str,
    threshold: Optional[float] = None,
    mask_forest: bool = False,
    urban_shp_path: Optional[str] = None,
    clip_mask_shp: Optional[str] = None,
    clip_mask_buffer_m: float = 0.0,
    nodata: float = -9999.0,
) -> dict:
    """
    Resample 100m suitability to 10m, then apply the same wet-threshold and
    optional forest/urban masking logic on the native wet woodland grid.
    Returns dict with stats for the 10m output.
    """
    print("    Resampling suitability to 10m...")

    # Get 10m grid from wet woodland raster
    with rasterio.open(wet_woodland_path) as wet_src:
        wet_profile = wet_src.profile.copy()
        wet_transform = wet_src.transform
        wet_crs = wet_src.crs
        wet_height = wet_src.height
        wet_width = wet_src.width
        wet_nodata = getattr(wet_src, "nodata", 255)
        wet_band1 = wet_src.read(1, masked=False)
        wet_band2 = wet_src.read(2, masked=False) if wet_src.count >= 2 else None
        wet_presence, wet_valid = _presence_seed_mask_from_array(
            wet_band1,
            wet_band2,
            wet_nodata,
            threshold,
        )

    # Resample 100m suitability to 10m grid
    with rasterio.open(suitability_100m_path) as suit_src:
        suit_nodata = suit_src.nodata

        # Create output array
        suit_10m = np.full((wet_height, wet_width), nodata, dtype=np.float32)

        reproject(
            source=rasterio.band(suit_src, 1),
            destination=suit_10m,
            src_transform=suit_src.transform,
            src_crs=suit_src.crs,
            dst_transform=wet_transform,
            dst_crs=wet_crs,
            resampling=Resampling.bilinear,
            src_nodata=suit_nodata,
            dst_nodata=nodata,
        )

    # Mask existing wet woodland
    suit_valid = (suit_10m != nodata) & np.isfinite(suit_10m)
    n_wet_masked = int((suit_valid & wet_presence).sum())
    suit_10m[wet_presence] = nodata
    print(f"    Masked {n_wet_masked:,} existing wet woodland pixels")

    n_forest_masked = 0
    forest_mask = wet_valid
    if mask_forest:
        suit_valid = (suit_10m != nodata) & np.isfinite(suit_10m)
        forest_only_mask = forest_mask & ~wet_presence
        n_forest_masked = int((suit_valid & forest_only_mask).sum())
        suit_10m[forest_only_mask] = nodata
        print(f"    Masked {n_forest_masked:,} additional forest-domain pixels")

    # Optional urban mask
    n_urban_masked = 0
    if urban_shp_path and os.path.isfile(urban_shp_path):
        print(f"    Applying urban mask at 10m...")
        urban_mask = rasterize_vector_mask(urban_shp_path, {
            "transform": wet_transform,
            "crs": wet_crs,
            "height": wet_height,
            "width": wet_width,
        })
        suit_valid = (suit_10m != nodata) & np.isfinite(suit_10m)
        n_urban_masked = int((suit_valid & urban_mask).sum())
        suit_10m[urban_mask] = nodata
        print(f"    Masked {n_urban_masked:,} urban pixels")

    n_clip_masked = 0
    if clip_mask_shp and os.path.isfile(clip_mask_shp):
        print("    Applying clip mask at 10m...")
        clip_mask = rasterize_vector_mask(
            clip_mask_shp,
            {
                "transform": wet_transform,
                "crs": wet_crs,
                "height": wet_height,
                "width": wet_width,
            },
            buffer_m=clip_mask_buffer_m,
        )
        suit_valid = (suit_10m != nodata) & np.isfinite(suit_10m)
        n_clip_masked = int((suit_valid & ~clip_mask).sum())
        suit_10m[~clip_mask] = nodata
        if n_clip_masked > 0:
            print(
                f"    Clipped 10m suitability to {clip_mask_shp}"
                f"{f' (buffer {clip_mask_buffer_m:g} m)' if clip_mask_buffer_m != 0.0 else ''}: "
                f"masked {n_clip_masked:,} pixels"
            )
        else:
            print(
                f"    Clip mask applied to 10m suitability"
                f"{f' (buffer {clip_mask_buffer_m:g} m)' if clip_mask_buffer_m != 0.0 else ''}: "
                "no additional pixels masked"
            )

    # Write 10m output
    out_profile = wet_profile.copy()
    out_profile.update(
        dtype=np.float32,
        count=1,
        nodata=nodata,
        compress="lzw",
        BIGTIFF="YES",  # Required for 10m national coverage
    )

    with rasterio.open(output_10m_path, "w", **out_profile) as dst:
        dst.write(suit_10m, 1)
        dst.set_band_description(1, "suitability_10m")

    print(f"    Wrote {output_10m_path}")

    # Return basic stats
    final_valid = (suit_10m != nodata) & np.isfinite(suit_10m)
    pixel_area_m2 = abs(wet_transform.a * wet_transform.e)

    return {
        "path": output_10m_path,
        "valid_pixels": int(final_valid.sum()),
        "valid_area_ha": (final_valid.sum() * pixel_area_m2) / 10000.0,
        "wet_masked": n_wet_masked,
        "forest_masked": n_forest_masked,
        "urban_masked": n_urban_masked,
        "clip_masked": n_clip_masked,
        "transform": wet_transform,
        "crs": wet_crs,
        "height": wet_height,
        "width": wet_width,
        "pixel_area_m2": pixel_area_m2,
    }


def _rows_cols_to_geoseries(
    rows: np.ndarray,
    cols: np.ndarray,
    stack_transform,
    stack_crs,
) -> gpd.GeoSeries:
    """Convert (rows, cols) pixel indices to point geometries (pixel centroids)."""
    xs, ys = [], []
    for r, c in zip(rows, cols):
        x, y = rasterio.transform.xy(stack_transform, r, c, offset="center")
        xs.append(x)
        ys.append(y)
    return ela.xy_to_geoseries(xs, ys, crs=stack_crs)


def compute_stack_valid_mask(stack_path: str) -> np.ndarray:
    """Return a boolean mask of stack pixels where every predictor band is valid."""
    with rasterio.open(stack_path) as src:
        valid_mask = np.zeros((src.height, src.width), dtype=bool)
        nodata = src.nodata
        for _, window in src.block_windows(1):
            block = src.read(window=window, masked=False).astype(np.float64)
            block_valid = np.isfinite(block).all(axis=0)
            if nodata is not None and not _is_nan_number(nodata):
                block_valid &= ~(block == nodata).any(axis=0)
            row_off = int(window.row_off)
            col_off = int(window.col_off)
            valid_mask[
                row_off:row_off + int(window.height),
                col_off:col_off + int(window.width),
            ] = block_valid
    return valid_mask


def sample_points_from_mask(
    candidate_mask: np.ndarray,
    stack_transform,
    stack_crs,
    max_points: Optional[int],
    rng: np.random.Generator,
) -> gpd.GeoSeries:
    """Sample point centroids uniformly from True pixels in a candidate mask."""
    rows, cols = sample_indices_from_mask(candidate_mask, max_points, rng)
    return _rows_cols_to_geoseries(rows, cols, stack_transform, stack_crs)


def sample_indices_from_mask(
    candidate_mask: np.ndarray,
    max_points: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (row, col) indices uniformly from True pixels in a candidate mask."""
    flat_indices = np.flatnonzero(candidate_mask.ravel())
    if flat_indices.size == 0:
        raise ValueError("No candidate pixels available for sampling.")
    if max_points is not None and flat_indices.size > max_points:
        flat_indices = rng.choice(flat_indices, size=max_points, replace=False)
    width = candidate_mask.shape[1]
    rows = flat_indices // width
    cols = flat_indices % width
    return rows.astype(np.int64), cols.astype(np.int64)


def presence_mask_to_points(
    presence_aligned: np.ndarray,
    stack_transform,
    stack_crs,
    max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> gpd.GeoSeries:
    """Convert presence raster (1 = presence) to point geometries (pixel centroids).
    If max_points is set and there are more presence pixels, subsamples uniformly at random."""
    rows, cols = np.where(presence_aligned == 1)
    if rng is None:
        rng = np.random.default_rng()
    n = len(rows)
    if max_points is not None and n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        rows, cols = rows[idx], cols[idx]
    return _rows_cols_to_geoseries(rows, cols, stack_transform, stack_crs)


def sample_presence_balanced(
    presence_aligned: np.ndarray,
    stack_path: str,
    band_names: List[str],
    balance_by_band: str,
    max_points: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample presence pixels so that roughly half are on peat and half off.
    For peat probability this means band >= 0.5; for peat depth it means depth > 0.
    Returns (rows, cols).
    """
    rows, cols = np.where(presence_aligned == 1)
    n = len(rows)
    if n == 0:
        return rows, cols
    try:
        band_idx = band_names.index(balance_by_band)
    except ValueError:
        raise ValueError(
            f"Balance band '{balance_by_band}' not in stack bands: {band_names}"
        ) from None
    with rasterio.open(stack_path) as src:
        band_data = src.read(band_idx + 1, masked=False)
    vals = band_data[rows, cols].astype(np.float64)
    mode = "depth" if "peat_depth" in balance_by_band.lower() else "probability"
    in_stratum_a = _peat_extent_mask(vals, mode)
    idx_a = np.where(in_stratum_a)[0]
    idx_b = np.where(~in_stratum_a)[0]
    n_a, n_b = len(idx_a), len(idx_b)
    if n_a == 0 or n_b == 0:
        # No variation; fall back to random subsample
        if max_points is not None and n > max_points:
            sel = rng.choice(n, size=max_points, replace=False)
            return rows[sel], cols[sel]
        return rows, cols
    target_n = min(n, max_points) if max_points is not None else n
    target_a = target_n // 2
    target_b = target_n - target_a

    idx_a = rng.permutation(idx_a)
    idx_b = rng.permutation(idx_b)
    take_a = min(target_a, n_a)
    take_b = min(target_b, n_b)

    selected_parts = []
    if take_a:
        selected_parts.append(idx_a[:take_a])
    if take_b:
        selected_parts.append(idx_b[:take_b])

    remaining = target_n - take_a - take_b
    if remaining > 0:
        leftovers = np.concatenate([idx_a[take_a:], idx_b[take_b:]])
        if leftovers.size > 0:
            take_extra = min(remaining, leftovers.size)
            selected_parts.append(rng.choice(leftovers, size=take_extra, replace=False))

    combined = np.concatenate(selected_parts)
    combined = rng.permutation(combined)
    return rows[combined], cols[combined]


def main(argv: Optional[List[str]] = None) -> int:
    if ela is None or gpd is None:
        print("Install elapid and geopandas: pip install elapid geopandas", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    default_boundary_dir = repo_root / "data" / "input" / "boundaries"
    default_stack = str(repo_root / "data" / "output" / "potential" / "potential_predictors_100m.tif")
    default_wet = str(repo_root / "data" / "output" / "labels" / "wetwoodland.tif")
    default_outdir = str(repo_root / "data" / "output" / "potential" / "maxent")
    default_clip_mask = default_boundary_dir / "england.shp"
    default_urban_shp = default_boundary_dir / "england_urban.shp"
    default_landvalue_shp = default_boundary_dir / "agricultural_land_classification.shp"
    default_lnrs_shp = default_boundary_dir / "lnrs_areas.shp"

    p = argparse.ArgumentParser(
        description="Run Elapid MaxEnt potential model using predictor stack and wet woodland presence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--stack",
        default=default_stack,
        help=f"Path to predictor stack (multi-band GeoTIFF). Default: {default_stack}",
    )
    p.add_argument(
        "--exclude-bands",
        nargs="+",
        action="append",
        default=None,
        help=(
            "One or more predictor band names to exclude from the MaxEnt run. "
            "Accepts repeated use or comma-separated values, e.g. "
            "--exclude-bands smuk_mean smuk_std"
        ),
    )
    p.add_argument(
        "--wet-woodland",
        default=default_wet,
        help=(
            "Path to wet woodland raster. Supports binary presence rasters, "
            "probability rasters (band 2 with --threshold), or split-background "
            "label rasters where classes 2/3 are collapsed to presence. "
            f"Default: {default_wet}"
        ),
    )
    p.add_argument(
        "--wet-woodland-kml",
        default=None,
        help=(
            "Optional KML or other polygon vector of observed wet woodland sites. "
            "Rasterized to the stack grid and unioned with --wet-woodland before "
            "presence sampling."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Threshold on band 2 probability for presence (0–1). "
            "If not set, uses band 1 as binary (1=presence). "
            "Set explicitly for probability-only rasters."
        ),
    )
    p.add_argument(
        "--max-presence",
        type=int,
        default=50_000,
        help="Max number of presence points to use (subsample if more). Default: 50000",
    )
    p.add_argument(
        "--n-background",
        type=int,
        default=50_000,
        help="Number of background (pseudoabsence) points. Default: 50000",
    )
    p.add_argument(
        "--outdir",
        default=default_outdir,
        help=f"Output directory for potential map and intermediates. Default: {default_outdir}",
    )
    p.add_argument(
        "--unmasked-output",
        default=None,
        help=(
            "Optional path for the unmasked 100m suitability surface written before "
            "existing-wet/forest/urban masking. Default: <outdir>/wet_woodland_potential_unmasked.tif"
        ),
    )
    p.add_argument(
        "--eval-background-raster",
        default=None,
        help=(
            "Optional path for a held-out background raster used for conformal calibration. "
            "Default: <outdir>/suitability_eval_background.tif"
        ),
    )
    p.add_argument(
        "--report-file",
        default=None,
        help="Optional unified report path. Default: data/output/reports/<outdir_name>.report.txt",
    )
    p.add_argument(
        "--mask-forest",
        action="store_true",
        help=(
            "Also mask the full forest/mapped domain (all valid pixels in --wet-woodland) from "
            "the output. By default only existing wet woodland (presence) pixels are masked."
        ),
    )
    p.add_argument(
        "--urban-shp",
        default=str(default_urban_shp),
        help=(
            "Optional urban-area polygon shapefile to mask from final suitability output. "
            f"Default: {default_urban_shp} (skipped if missing)"
        ),
    )
    p.add_argument(
        "--clip-mask-shp",
        default=str(default_clip_mask),
        help=(
            "Optional polygon shapefile used to clip final suitability outputs to the target land area. "
            f"Default: {default_clip_mask} (skipped if missing)"
        ),
    )
    p.add_argument(
        "--clip-mask-buffer-m",
        type=float,
        default=-100.0,
        help=(
            "Buffer in metres applied when clipping final suitability outputs. "
            "Negative shrinks inland to suppress coastal edge artifacts. Default: -100."
        ),
    )
    p.add_argument(
        "--landvalue-shp",
        default=str(default_landvalue_shp),
        help=(
            "Optional land value / ALC polygon shapefile. If provided, grouped zonal stats "
            "report is written for grade_1_to_2, grade_3, and grade_4_to_6. "
            f"Default: {default_landvalue_shp} (skipped if missing)"
        ),
    )
    p.add_argument(
        "--alc-grade-field",
        default="ALC_grade",
        help="Field name in --landvalue-shp containing ALC grade values. Default: ALC_grade",
    )
    p.add_argument(
        "--high-potential-threshold",
        type=float,
        default=None,
        help=(
            "Optional threshold for high potential in landvalue report (e.g. 0.7). "
            "If set, high-potential percent/area is reported per group."
        ),
    )
    p.add_argument(
        "--landvalue-report",
        default=None,
        help=(
            "Optional output text path for landvalue zonal stats report. "
            "Default: <outdir>/landvalue_potential_stats.txt"
        ),
    )
    p.add_argument(
        "--lnrs-shp",
        default=str(default_lnrs_shp),
        help=(
            "Optional LNRS polygon shapefile. If provided, per-LNRS suitability, "
            "peat (on/off), and land cover (under forest/bare-open) stats are added "
            f"to the landvalue report. Default: {default_lnrs_shp} (skipped if missing)"
        ),
    )
    p.add_argument(
        "--lnrs-label-field",
        default=None,
        help=(
            "Optional field in --lnrs-shp used for LNRS names. "
            "If omitted, a name-like field is auto-detected."
        ),
    )
    p.add_argument(
        "--seed-source-raster",
        default=None,
        help=(
            "Optional source wet woodland raster for seed-distance stats in report. "
            "Default: use --wet-woodland."
        ),
    )
    p.add_argument(
        "--seed-distance-threshold",
        type=float,
        default=100.0,
        help="Distance threshold in meters for seed-proximity stats. Default: 100",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling presence/background.",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of geographic K-fold CV folds for robustness (spatial train/test AUC). Default: 5",
    )
    p.add_argument(
        "--compute-10m",
        action="store_true",
        help=(
            "Resample suitability to 10m (wet woodland resolution), mask at native resolution, "
            "and compute stats at 10m for accuracy. Outputs both 100m and 10m TIF files. "
            "Slower but eliminates resolution mismatch issues."
        ),
    )
    p.add_argument(
        "--shap",
        action="store_true",
        help="Compute SHAP-based variable importance (requires pip install shap). Writes CSV and bar plot to outdir.",
    )
    p.add_argument(
        "--shap-n-background",
        type=int,
        default=150,
        help="Number of background samples for SHAP KernelExplainer. Default: 150",
    )
    p.add_argument(
        "--shap-n-eval",
        type=int,
        default=500,
        help="Number of samples to explain for mean |SHAP| importance. Default: 500",
    )
    args = p.parse_args(argv)
    excluded_band_names = parse_band_name_args(args.exclude_bands)

    stack_path = args.stack
    wet_path = args.wet_woodland
    if not os.path.isfile(stack_path):
        print(f"Stack not found: {stack_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(wet_path):
        print(f"Wet woodland raster not found: {wet_path}", file=sys.stderr)
        return 1
    if args.wet_woodland_kml and not os.path.isfile(args.wet_woodland_kml):
        print(f"Wet woodland KML not found: {args.wet_woodland_kml}", file=sys.stderr)
        return 1
    default_optional_vectors = {
        "clip_mask_shp": (default_clip_mask, "Clip mask"),
        "urban_shp": (default_urban_shp, "Urban shapefile"),
        "landvalue_shp": (default_landvalue_shp, "Landvalue shapefile"),
        "lnrs_shp": (default_lnrs_shp, "LNRS shapefile"),
    }
    for attr, (default_path, label) in default_optional_vectors.items():
        value = getattr(args, attr)
        if not value:
            continue
        if os.path.isfile(value):
            continue
        try:
            value_norm = os.path.normpath(os.path.abspath(value))
            default_norm = os.path.normpath(str(default_path.resolve()))
        except OSError:
            value_norm = os.path.normpath(os.path.abspath(value))
            default_norm = os.path.normpath(str(default_path))
        if value_norm == default_norm:
            print(f"{label}: skipped (not found at {value})")
            setattr(args, attr, None)
            continue
        print(f"{label} not found: {value}", file=sys.stderr)
        return 1
    if args.seed_source_raster and not os.path.isfile(args.seed_source_raster):
        print(f"Seed source raster not found: {args.seed_source_raster}", file=sys.stderr)
        return 1

    _ensure_dir(args.outdir)
    output_path = os.path.join(args.outdir, "wet_woodland_potential.tif")
    unmasked_output_path = (
        args.unmasked_output
        if args.unmasked_output
        else os.path.join(args.outdir, "wet_woodland_potential_unmasked.tif")
    )
    eval_background_raster = (
        args.eval_background_raster
        if args.eval_background_raster
        else os.path.join(args.outdir, "suitability_eval_background.tif")
    )
    output_10m_path = os.path.join(args.outdir, "wet_woodland_potential_10m.tif") if args.compute_10m else None
    default_report_dir = Path(__file__).resolve().parents[2] / "data" / "output" / "reports"
    report_path = (
        args.report_file
        if args.report_file
        else str(default_report_dir / f"{Path(args.outdir).name}.report.txt")
    )
    planned_landvalue_report = None
    if args.landvalue_shp:
        stats_suffix = "_10m" if args.compute_10m else ""
        planned_landvalue_report = (
            args.landvalue_report
            if args.landvalue_report
            else os.path.join(args.outdir, f"landvalue_potential_stats{stats_suffix}.txt")
        )
    print("Elapid potential")
    print("=" * 60)
    print(f"Stack:            {stack_path}")
    print(f"Excluded bands:   {', '.join(excluded_band_names) if excluded_band_names else 'None'}")
    print(f"Wet woodland:     {wet_path}")
    print(f"Wet woodland KML: {args.wet_woodland_kml if args.wet_woodland_kml else 'None'}")
    print(f"Output dir:       {args.outdir}")
    print(f"Output raw:       {unmasked_output_path}")
    print(f"Output raster:    {output_path}")
    print(f"Eval background:  {eval_background_raster}")
    print(f"Clip mask:        {args.clip_mask_shp if args.clip_mask_shp else 'None'}")
    print(f"Clip buffer:      {args.clip_mask_buffer_m:g} m")
    if output_10m_path is not None:
        print(f"Output 10m:       {output_10m_path}")
    print(f"Report file:      {report_path}")
    if planned_landvalue_report is not None:
        print(f"Landvalue report: {planned_landvalue_report}")
    rng = np.random.default_rng(args.seed)

    # 1) Stack profile and band names
    print("[1] Reading stack profile and band names...")
    stack_profile, band_names = get_stack_profile_and_labels(stack_path)
    if excluded_band_names:
        missing = [name for name in excluded_band_names if name not in band_names]
        if missing:
            print(
                "Unknown predictor band(s) in --exclude-bands: "
                + ", ".join(missing)
                + f". Available bands: {band_names}",
                file=sys.stderr,
            )
            return 1
        keep_indices = [i for i, name in enumerate(band_names) if name not in excluded_band_names]
        keep_names = [band_names[i] for i in keep_indices]
        if not keep_names:
            print("All predictor bands were excluded; nothing left to fit.", file=sys.stderr)
            return 1
        filtered_stack_path = os.path.join(args.outdir, "predictor_stack_filtered.tif")
        print(f"    Writing filtered stack without excluded predictors → {filtered_stack_path}")
        write_stack_band_subset(stack_path, keep_indices, keep_names, filtered_stack_path)
        stack_path = filtered_stack_path
        stack_profile, band_names = get_stack_profile_and_labels(stack_path)
    stack_crs = stack_profile["crs"]
    stack_transform = stack_profile["transform"]
    print(f"    Active stack: {stack_path}, bands: {band_names}")

    # Sanity check: per-band stats to catch nodata bleed, extreme values, empty bands
    print("    Band sanity check:")
    with rasterio.open(stack_path) as _src:
        _nodata = _src.nodata
        for _i, _name in enumerate(band_names, 1):
            _band = _src.read(_i, masked=False).astype(np.float64)
            _valid = np.isfinite(_band)
            if _nodata is not None:
                _valid &= (_band != _nodata)
            _n = int(_valid.sum())
            if _n == 0:
                print(f"      [{_i}] {_name}: *** ALL NODATA — band is empty ***")
                continue
            _vals = _band[_valid]
            _pct = [float(np.percentile(_vals, p)) for p in (1, 50, 99)]
            _inf = int((~np.isfinite(_band)).sum())
            _flag = " *** HAS INF/NAN ***" if _inf > 0 else ""
            print(f"      [{_i}] {_name}: min={_vals.min():.3g}  p1={_pct[0]:.3g}  median={_pct[1]:.3g}  p99={_pct[2]:.3g}  max={_vals.max():.3g}  valid={_n:,}{_flag}")

    forest_mask: Optional[np.ndarray] = None
    urban_mask: Optional[np.ndarray] = None
    forest_mask_pixels: Optional[int] = None
    urban_mask_pixels: Optional[int] = None
    if args.mask_forest:
        print("    Preparing forest mask from wet woodland valid pixels...")
        forest_mask = align_wet_valid_mask_to_stack(wet_path, stack_profile)
        forest_mask_pixels = int(forest_mask.sum())
        print(f"    Forest mask pixels (will be excluded): {forest_mask_pixels:,}")
    if args.urban_shp:
        print(f"    Preparing urban mask from {args.urban_shp}...")
        urban_mask = rasterize_vector_mask(args.urban_shp, stack_profile)
        urban_mask_pixels = int(urban_mask.sum())
        print(f"    Urban mask pixels (will be excluded): {urban_mask_pixels:,}")

    # 2) Align wet woodland to stack and apply threshold
    if args.threshold is not None:
        print(f"[2] Aligning wet woodland raster to stack (threshold >= {args.threshold})...")
    else:
        print("[2] Aligning wet woodland raster to stack (using band 1 binary)...")
    presence_raster_aligned = align_wet_woodland_to_stack(
        wet_path, stack_profile, threshold=args.threshold
    )
    n_presence_pixels_raster = int((presence_raster_aligned == 1).sum())
    n_presence_pixels_kml: Optional[int] = None
    presence_aligned = presence_raster_aligned.copy()
    print(f"    Presence pixels from wet woodland raster: {n_presence_pixels_raster:,}")
    if args.wet_woodland_kml:
        print(f"    Rasterizing wet woodland polygons to stack: {args.wet_woodland_kml}")
        kml_presence = rasterize_presence_vector_to_stack(args.wet_woodland_kml, stack_profile)
        n_presence_pixels_kml = int(kml_presence.sum())
        if n_presence_pixels_kml == 0:
            print(
                "Wet woodland KML produced 0 presence pixels on the stack grid. "
                "Check CRS, extent, and polygon geometry.",
                file=sys.stderr,
            )
            return 1
        presence_aligned = np.where((presence_raster_aligned == 1) | kml_presence, 1, 0).astype(np.uint8)
        print(f"    Presence pixels from wet woodland KML: {n_presence_pixels_kml:,}")
    n_presence_pixels = int((presence_aligned == 1).sum())
    if args.wet_woodland_kml:
        print(f"    Combined presence pixels (raster ∪ KML): {n_presence_pixels:,}")
    else:
        print(f"    Presence pixels (after threshold): {n_presence_pixels:,}")

    if n_presence_pixels == 0:
        print("No presence pixels; cannot fit model. Check --wet-woodland and --threshold.", file=sys.stderr)
        return 1

    # 3) Sample presence points (balanced by peat extent: ~50% on peat, ~50% off peat)
    print("[3] Sampling presence points...")
    presence_sampling_mode = "random"
    balance_band = "peat_depth_m" if "peat_depth_m" in band_names else "peat_prob"
    try:
        rows, cols = sample_presence_balanced(
            presence_aligned,
            stack_path,
            band_names,
            balance_by_band=balance_band,
            max_points=args.max_presence,
            rng=rng,
        )
        presence_geom = _rows_cols_to_geoseries(rows, cols, stack_transform, stack_crs)
        if balance_band == "peat_depth_m":
            presence_sampling_mode = "balanced by peat_depth_m (> 0 m = peat extent, 0 m = off peat)"
            print("    Balanced by peat_depth_m (> 0 m = peat extent, 0 m = off peat)")
        else:
            presence_sampling_mode = "balanced by peat_prob (>= 0.5 = on peat)"
            print("    Balanced by peat_prob (>= 0.5 = on peat)")
    except ValueError:
        presence_geom = presence_mask_to_points(
            presence_aligned,
            stack_transform,
            stack_crs,
            max_points=args.max_presence,
            rng=rng,
        )
        presence_sampling_mode = "random (peat predictor not in stack)"
        print("    Random (peat predictor not in stack)")
    n_presence_pts = len(presence_geom)
    print(f"    Presence points: {n_presence_pts:,}")

    # 4) Sample background points from stack extent (valid pixels only)
    print("[4] Sampling background points...")
    stack_valid_mask = compute_stack_valid_mask(stack_path)
    candidate_background_mask = stack_valid_mask & (presence_aligned != 1)
    if forest_mask is not None:
        candidate_background_mask &= ~forest_mask
    if urban_mask is not None:
        candidate_background_mask &= ~urban_mask
    n_candidate_background = int(candidate_background_mask.sum())
    print(f"    Candidate background pixels: {n_candidate_background:,}")
    if n_candidate_background == 0:
        print("No valid background pixels remain after exclusions.", file=sys.stderr)
        return 1
    bg_rows, bg_cols = sample_indices_from_mask(
        candidate_background_mask,
        args.n_background,
        rng,
    )
    sampled_background_mask = np.zeros_like(candidate_background_mask, dtype=bool)
    sampled_background_mask[bg_rows, bg_cols] = True
    discarded_background_mask = candidate_background_mask & ~sampled_background_mask
    background_geom = _rows_cols_to_geoseries(bg_rows, bg_cols, stack_transform, stack_crs)
    n_background = len(background_geom)
    print(f"    Background points: {n_background:,}")
    bg_eval_counts = write_background_eval_raster(
        eval_background_raster,
        stack_path,
        discarded_background_mask,
        sampled_background_mask,
        extra_tags={
            "stack_path": stack_path,
            "wet_woodland_raster": args.wet_woodland,
            "wet_woodland_kml": args.wet_woodland_kml or "",
        },
    )
    print(
        "    Eval background raster: "
        f"discarded={bg_eval_counts['discarded_eval_background']:,}  "
        f"sampled={bg_eval_counts['sampled_training_background']:,}"
    )

    # 5) Annotate presence and background with stack covariates
    print("[5] Annotating points with predictor stack...")
    presence_annotated = ela.annotate(
        presence_geom,
        [stack_path],
        labels=band_names,
        drop_na=True,
        quiet=False,
    )
    background_annotated = ela.annotate(
        background_geom,
        [stack_path],
        labels=band_names,
        drop_na=True,
        quiet=False,
    )
    # Distance-based sample weights: downweight points in dense clusters for spatial representativeness
    presence_annotated["SampleWeight"] = ela.distance_weights(presence_annotated, n_neighbors=-1)
    background_annotated["SampleWeight"] = ela.distance_weights(background_annotated, n_neighbors=-1)
    # Merge with class label
    merged = ela.stack_geodataframes(
        presence_annotated,
        background_annotated,
        add_class_label=True,
        target_crs="presence",
    )
    # Drop rows with non-finite predictors so CV indices align with X
    X_full = merged[band_names].values.astype(np.float64)
    valid = np.isfinite(X_full).all(axis=1)
    n_dropped_nonfinite = 0
    if not valid.all():
        n_drop = int((~valid).sum())
        n_dropped_nonfinite = n_drop
        merged = merged.loc[valid].reset_index(drop=True)
        print(f"    Dropped {n_drop} samples with non-finite predictor values.")
    X = merged[band_names].values.astype(np.float64)
    y = merged["class"].values
    sample_weight = merged["SampleWeight"].values.astype(np.float64)
    # Winsorize each predictor to 0.5–99.5% to tame extreme tails (e.g. distance_to_river at grid edges)
    # Save bounds so we can sanitize the stack the same way when applying the model.
    # Categorical bands (e.g. soil_code_categorical) are not percentile-winsorized; use integer min/max.
    predictor_bounds: List[Tuple[float, float]] = []
    for j in range(X.shape[1]):
        col = X[:, j]
        is_cat = "categorical" in band_names[j].lower()
        if is_cat:
            finite = col[np.isfinite(col)]
            lo = float(np.floor(np.nanmin(finite))) if len(finite) else 0.0
            hi = float(np.ceil(np.nanmax(finite))) if len(finite) else 1.0
        else:
            lo, hi = np.nanpercentile(col, [0.5, 99.5])
            if not (np.isfinite(lo) and np.isfinite(hi)):
                lo, hi = np.nanmin(col), np.nanmax(col)
        # Never use full float32 range: product features would overflow (col_i * col_j)
        if not np.isfinite(lo):
            lo = SAFE_PREDICTOR_MIN
        if not np.isfinite(hi):
            hi = SAFE_PREDICTOR_MAX
        lo = max(lo, SAFE_PREDICTOR_MIN)
        hi = min(hi, SAFE_PREDICTOR_MAX)
        predictor_bounds.append((float(lo), float(hi)))
        if np.isfinite(lo) and np.isfinite(hi):
            clipped = np.clip(col, lo, hi)
            X[:, j] = np.round(clipped).astype(np.float64) if is_cat else clipped
    # Clip to safe range so product/quadratic features don't overflow during fit or predict
    X = np.clip(X, SAFE_PREDICTOR_MIN, SAFE_PREDICTOR_MAX)
    print(f"    Training samples: {len(y)} (presence: {(y == 1).sum()}, background: {(y == 0).sum()})")

    # Categorical predictor indices (0-based): treat as factors, not continuous (e.g. soil_code_categorical)
    categorical_indices = [j for j, name in enumerate(band_names) if "categorical" in name.lower()]
    known_categorical: Dict[int, np.ndarray] = {}
    for j in categorical_indices:
        known_categorical[j] = np.unique(X[:, j].astype(np.int64))
    if categorical_indices:
        print(f"    Categorical predictors (indices {categorical_indices}): {[band_names[j] for j in categorical_indices]}")

    # 5b) VIF check for multicollinearity
    print("[5b] Computing VIF for multicollinearity inspection...")
    compute_and_report_vif(X, band_names, args.outdir)
    vif_path = os.path.join(args.outdir, "vif_report.txt")
    if not os.path.exists(vif_path):
        vif_path = None

    # 6) Geographic K-fold CV: report spatially held-out AUC (publication-ready robustness)
    n_folds = max(2, min(args.cv_folds, 10))
    cv_aucs: List[float] = []
    mean_auc: Optional[float] = None
    std_auc: Optional[float] = None
    if GeographicKFold is not None and n_folds > 1:
        print(f"[6] Geographic {n_folds}-fold CV (spatial train/test)...")
        gkf = GeographicKFold(n_splits=n_folds, random_state=args.seed)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(merged), 1):
            cv_model = ela.MaxentModel()
            X_train_fold = X[train_idx]
            cv_model.fit(
                X_train_fold,
                y[train_idx],
                sample_weight=sample_weight[train_idx],
                categorical=categorical_indices if categorical_indices else None,
                labels=band_names,
            )
            # Test fold can contain categorical values not seen in train fold -> map to train fallback
            X_test_fold = X[test_idx].copy()
            for j in categorical_indices:
                allowed = np.unique(X_train_fold[:, j].astype(np.int64))
                fallback = float(allowed[0]) if len(allowed) > 0 else 0.0
                test_vals = np.round(X_test_fold[:, j]).astype(np.int64)
                mask_unknown = ~np.isin(test_vals, allowed)
                X_test_fold[mask_unknown, j] = fallback
            y_pred = cv_model.predict(X_test_fold)
            auc = roc_auc_score(y[test_idx], y_pred)
            cv_aucs.append(auc)
        mean_auc = float(np.mean(cv_aucs))
        std_auc = float(np.std(cv_aucs))
        print(f"    Spatial CV AUC (test): {mean_auc:.3f} ± {std_auc:.3f} (folds: {[f'{a:.3f}' for a in cv_aucs]})")

    # 7) Fit final MaxEnt on full data (with distance weights) for mapping
    print("[7] Fitting final MaxentModel on full data...")
    model = ela.MaxentModel()
    model.fit(
        X,
        y,
        sample_weight=sample_weight,
        categorical=categorical_indices if categorical_indices else None,
        labels=band_names,
    )
    print("    Model fitted.")

    # Policy-safe thresholds: percentiles of predicted suitability at training presence points
    pred_presence = model.predict(X[y == 1])
    p5 = float(np.percentile(pred_presence, 5))
    p10 = float(np.percentile(pred_presence, 10))
    p20 = float(np.percentile(pred_presence, 20))
    thresh_path = os.path.join(args.outdir, "suitability_thresholds.txt")
    with open(thresh_path, "w", encoding="utf-8") as f:
        f.write("# Suitability thresholds from training presence predictions (policy-safe)\n")
        f.write("# Use e.g. 'suitable = raster >= value' for binary mapping.\n")
        f.write("5th_percentile_training_presence\t{}\n".format(p5))
        f.write("10th_percentile_training_presence\t{}\n".format(p10))
        f.write("20th_percentile_training_presence\t{}\n".format(p20))
    print(f"    Suitability thresholds (training presence): 5th={p5:.4f}, 10th={p10:.4f}, 20th={p20:.4f}")
    print(f"    Wrote {thresh_path} (recommended policy-safe: 10th = {p10:.4f})")

    shap_csv_path = None
    shap_plot_path = None
    if args.shap:
        print("[7b] Computing SHAP variable importance...")
        _compute_shap_importance(
            model,
            X,
            band_names,
            args.outdir,
            n_background=args.shap_n_background,
            n_eval=args.shap_n_eval,
            seed=args.seed,
        )
        _candidate_csv = os.path.join(args.outdir, "shap_importance.csv")
        _candidate_plot = os.path.join(args.outdir, "shap_importance.png")
        shap_csv_path = _candidate_csv if os.path.exists(_candidate_csv) else None
        shap_plot_path = _candidate_plot if os.path.exists(_candidate_plot) else None

    # 8) Apply model to stack → potential map (via sanitized stack so no inf/extremes)
    print(f"[8] Applying model to stack → {unmasked_output_path}")
    sanitized_path = os.path.join(args.outdir, "temp_sanitized_stack.tif")
    masks_to_apply_100m: List[np.ndarray] = []
    existing_wet_woodland_mask = (presence_aligned == 1)
    masks_to_apply_100m.append(existing_wet_woodland_mask)
    if forest_mask is not None:
        masks_to_apply_100m.append(forest_mask)
    if urban_mask is not None:
        masks_to_apply_100m.append(urban_mask)
    try:
        print("    Writing sanitized stack (clip to training bounds, map unknown categories)...")
        write_sanitized_stack(
            stack_path,
            band_names,
            predictor_bounds,
            sanitized_path,
            nodata_val=np.nan,
            known_categorical=known_categorical,
        )
        apply_model_to_raster_safe(
            model,
            sanitized_path,
            unmasked_output_path,
            nodata=-9999.0,
            quiet=False,
        )
        annotate_output_raster(
            unmasked_output_path,
            band_description="suitability_100m_unmasked",
            extra_tags={
                "wet_woodland_raster": args.wet_woodland,
                "wet_woodland_kml": args.wet_woodland_kml or "",
                "masked_output": False,
            },
        )
        print(f"    Wrote unmasked suitability: {unmasked_output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(unmasked_output_path, output_path)
        annotate_output_raster(
            output_path,
            band_description="suitability_100m",
            extra_tags={
                "wet_woodland_raster": args.wet_woodland,
                "wet_woodland_kml": args.wet_woodland_kml or "",
                "masked_output": True,
                "unmasked_source": unmasked_output_path,
            },
        )
        print(f"    Wrote masked suitability base copy: {output_path}")
        if getattr(args, "compute_10m", False):
            print("    Deferring 100m masks until after native-grid 10m masking.")
        else:
            n_existing = int(existing_wet_woodland_mask.sum())
            print(f"    Masking existing wet woodland (presence) pixels: {n_existing:,}")
            n_masked = apply_masks_to_output(
                output_path,
                masks_to_apply_100m,
                nodata_value=-9999.0,
            )
            print(f"    Applied output mask(s); set {n_masked:,} predicted pixels to nodata.")
            apply_vector_clip_to_output(
                output_path,
                args.clip_mask_shp,
                buffer_m=args.clip_mask_buffer_m,
                label="100m suitability",
            )
    finally:
        if os.path.isfile(sanitized_path):
            try:
                os.remove(sanitized_path)
            except OSError:
                pass

    # 9) Optional: create 10m output with native-resolution masking
    output_10m_path = None
    info_10m = None
    if getattr(args, "compute_10m", False):
        output_10m_path = os.path.join(args.outdir, "wet_woodland_potential_10m.tif")
        print(f"[9] Creating 10m suitability output → {output_10m_path}")
        info_10m = create_10m_suitability(
            output_path,
            wet_path,
            output_10m_path,
            threshold=args.threshold,
            mask_forest=bool(args.mask_forest),
            urban_shp_path=args.urban_shp,
            clip_mask_shp=args.clip_mask_shp,
            clip_mask_buffer_m=args.clip_mask_buffer_m,
            nodata=-9999.0,
        )
        print(f"    10m output: {info_10m['valid_pixels']:,} valid pixels ({info_10m['valid_area_ha']:.1f} ha)")
        n_existing = int(existing_wet_woodland_mask.sum())
        print(f"    Masking aligned 100m output for consistency (existing wet woodland: {n_existing:,})")
        n_masked = apply_masks_to_output(
            output_path,
            masks_to_apply_100m,
            nodata_value=-9999.0,
        )
        print(f"    Applied 100m output mask(s); set {n_masked:,} predicted pixels to nodata.")
        apply_vector_clip_to_output(
            output_path,
            args.clip_mask_shp,
            buffer_m=args.clip_mask_buffer_m,
            label="100m suitability",
        )

    landvalue_report_path = None
    if args.landvalue_shp:
        # Use 10m output for stats if available, otherwise 100m
        stats_raster = output_10m_path if output_10m_path else output_path
        stats_suffix = "_10m" if output_10m_path else ""

        report_path = (
            args.landvalue_report
            if args.landvalue_report
            else os.path.join(args.outdir, f"landvalue_potential_stats{stats_suffix}.txt")
        )
        landvalue_report_path = report_path
        print(f"[10] Writing landvalue zonal stats report → {report_path}")
        print(f"    Using {'10m' if output_10m_path else '100m'} suitability for stats")
        write_landvalue_group_stats(
            stats_raster,
            args.landvalue_shp,
            report_path,
            grade_field=args.alc_grade_field,
            high_potential_threshold=args.high_potential_threshold,
            seed_source_path=(args.seed_source_raster if args.seed_source_raster else wet_path),
            seed_threshold=args.threshold,
            seed_distance_m=args.seed_distance_threshold,
            stack_path=stack_path,
            wet_woodland_path=wet_path,
            suitable_threshold=p10,  # 10th percentile threshold
            lnrs_shp=args.lnrs_shp,
            lnrs_label_field=args.lnrs_label_field,
        )
        print(f"    Wrote {report_path}")

    # Optional: save presence/background points and merged table for inspection
    merged_path = os.path.join(args.outdir, "presence_background_annotated.gpkg")
    merged.to_file(merged_path, driver="GPKG")
    print(f"    Saved annotated points to {merged_path}")

    default_report_dir = Path(__file__).resolve().parents[2] / "data" / "output" / "reports"
    report_path = (
        args.report_file
        if args.report_file
        else str(default_report_dir / f"{Path(args.outdir).name}.report.txt")
    )
    write_elapid_report(
        report_path,
        args=args,
        stack_profile=stack_profile,
        band_names=band_names,
        excluded_band_names=excluded_band_names,
        presence_sampling_mode=presence_sampling_mode,
        n_presence_pixels_raster=n_presence_pixels_raster,
        n_presence_pixels_kml=n_presence_pixels_kml,
        n_presence_pixels=n_presence_pixels,
        n_presence_points=n_presence_pts,
        n_background_points=n_background,
        n_dropped_nonfinite=n_dropped_nonfinite,
        n_training_samples=len(y),
        n_training_presence=int((y == 1).sum()),
        n_training_background=int((y == 0).sum()),
        categorical_indices=categorical_indices,
        forest_mask_pixels=forest_mask_pixels,
        urban_mask_pixels=urban_mask_pixels,
        mean_auc=mean_auc,
        std_auc=std_auc,
        cv_aucs=cv_aucs,
        thresholds={"p5": p5, "p10": p10, "p20": p20},
        thresh_path=thresh_path,
        unmasked_output_path=unmasked_output_path,
        output_path=output_path,
        output_10m_path=output_10m_path,
        output_10m_info=info_10m,
        eval_background_raster=eval_background_raster,
        vif_path=vif_path,
        shap_csv_path=shap_csv_path,
        shap_plot_path=shap_plot_path,
        landvalue_report_path=landvalue_report_path,
        merged_path=merged_path,
    )
    print(f"    Wrote {report_path}")

    if output_10m_path:
        print(f"\nDone. Outputs:")
        print(f"  Raw 100m: {unmasked_output_path}")
        print(f"  100m: {output_path}")
        print(f"  10m:  {output_10m_path} (recommended for visualization)")
    else:
        print(f"\nDone. Use {output_path} as the MaxEnt suitability surface.")
        print(f"  Raw 100m: {unmasked_output_path}")
        print("  Tip: Use --compute-10m for native-resolution masking and accurate stats.")
    print(f"  Eval background: {eval_background_raster}")
    print(f"  Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
