#!/usr/bin/env python3
"""
Build the abiotic predictor stack for wet woodland potential mapping.

This script prepares a set of aligned predictor rasters ready for Elapid
potential modelling. It is deliberately ordered from
easiest to hardest processing stages so you can get early pieces working and
debugged before adding more complexity.

Stages (in order):
  1) Load peat depth raster and align it to the modelling grid.
  2) Optionally resample SMUK soil moisture (multi-band) to the template and
     compute per-pixel mean and standard deviation over time.
  3) Derive terrain attributes from the DTM at its native resolution using
     richdem: elevation, slope, aspect, CTI/TWI.
  4) Resample DTM-derived rasters to the modelling template grid (100 m).

Intended predictors (all aligned to the modelling grid):
  - elev_100m.tif        : elevation (m)
  - slope_100m.tif       : slope (degrees)
  - aspect_100m.tif      : aspect (degrees 0-360)
  - cti_100m.tif         : compound topographic index / TWI
  - smuk_mean_100m.tif   : mean SMUK soil moisture over time
  - smuk_std_100m.tif    : std. dev. of SMUK over time
  - peat_depth_m_100m.tif : peat depth converted to metres; off-peat filled as 0 m
  - soil_code_100m.tif   : optional; unique integer code per soil class (from soils shapefile,
                           only added to the predictor stack when --include-soil is passed)
  - distance_to_water_100m.tif : optional; Euclidean distance to nearest water body (rivers + lakes, m)

Soils shapefile (optional):
  If --include-soil and --soils-shp are provided, polygons are rasterized using
  SOIL_GROUP and SOIL_TEX (or custom field names) to form unique integer codes
  (1, 2, 3, ...). A lookup CSV (soil_lookup.csv) is written for interpretation.
  For downstream MaxEnt: treat soil_code as a *categorical* (factor) variable so
  the model uses binary indicators per class rather than treating codes as continuous.

Distance to water (optional):
  If --rivers and/or --lakes are provided, the shapefiles are merged and rasterized
  to the template grid; Euclidean distance to the nearest water body (m) is computed
  per pixel. Output: distance_to_water_100m.tif (band: distance_to_water_m).

Mask (optional):
  If --mask-shp (e.g. england.shp) is provided, all output rasters on the template grid
  are masked so pixels outside the polygon(s) are set to nodata. This keeps the stack
  clean and avoids edge-of-grid artefacts for downstream use (e.g. MaxEnt).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import reproject
from scipy import ndimage
from tqdm import tqdm

try:
    import geopandas as gpd  # type: ignore
except ImportError:
    gpd = None

try:
    import richdem as rd  # type: ignore
except ImportError:
    rd = None  # richdem is optional but recommended

# Clip float bands to this range so downstream (e.g. MaxEnt product features) never overflow
SAFE_FLOAT_MIN = -1.0e10
SAFE_FLOAT_MAX = 1.0e10


def _discover_default_peat_depth_source(peat_dir: Path) -> Path:
    """Prefer a peat-depth raster if present; otherwise fall back to a conventional path."""
    for pattern in ("*depth*.tif", "*depth*.tiff", "*.tif", "*.tiff"):
        matches = sorted(peat_dir.glob(pattern))
        if matches:
            return matches[0]
    return peat_dir / "peat_depth.tif"


def _default_preprocess_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    input_root = repo_root / "data" / "input"
    output_root = repo_root / "data" / "output"
    preprocess_root = output_root / "preprocess"
    potential_root = output_root / "potential"
    peat_dir = input_root / "peat"
    return {
        "template": preprocess_root / "dtm_metrics.tif",
        "dtm_metrics": preprocess_root / "dtm_metrics.tif",
        "peat_depth": _discover_default_peat_depth_source(peat_dir),
        "smuk": input_root / "hydro" / "smuk",
        "soils_shp": input_root / "soils" / "soils_parent_material.shp",
        "rivers": input_root / "hydro" / "rivers.shp",
        "lakes": input_root / "hydro" / "lakes.gpkg",
        "mask_shp": input_root / "boundaries" / "england.shp",
        "outdir": potential_root,
        "output_stack": potential_root / "potential_predictors_100m.tif",
    }


def _norm_path(path: str | Path) -> str:
    return os.path.normpath(str(Path(path).expanduser().resolve(strict=False)))


def _require_stage2_dtm_inputs(template_path: str, dtm_metrics_path: str, defaults: dict[str, Path]) -> None:
    template_default = _norm_path(defaults["template"])
    dtm_default = _norm_path(defaults["dtm_metrics"])
    template_norm = _norm_path(template_path)
    dtm_norm = _norm_path(dtm_metrics_path)

    if not os.path.isfile(template_path):
        if template_norm == template_default:
            raise SystemExit(
                "Template raster not found at "
                f"{template_path}\n"
                "Run build_dtm_metrics.py first to create "
                "data/output/preprocess/dtm_metrics.tif, then rerun this abiotic stage."
            )
        raise SystemExit(f"Template raster not found: {template_path}")

    if dtm_norm == template_norm:
        return

    if not os.path.isfile(dtm_metrics_path):
        if dtm_norm == dtm_default:
            raise SystemExit(
                "DTM metrics raster not found at "
                f"{dtm_metrics_path}\n"
                "Run build_dtm_metrics.py first to create "
                "data/output/preprocess/dtm_metrics.tif, then rerun this abiotic stage."
            )
        raise SystemExit(f"DTM metrics raster not found: {dtm_metrics_path}")


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _default_nodata_for_dtype(dtype: np.dtype, src_nodata: Optional[float] = None):
    if src_nodata is not None:
        return src_nodata
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.unsignedinteger):
        return 0
    return -9999


def _gtiff_block_size(length: int, default: int = 512) -> int:
    if length <= 0:
        return 16
    size = min(default, int(length))
    if size < 16:
        return 16
    return max(16, (size // 16) * 16)


def _prepare_gtiff_profile(
    base_profile: dict,
    *,
    count: int,
    dtype,
    nodata,
) -> dict:
    profile = base_profile.copy()
    width = int(profile.get("width", 0))
    height = int(profile.get("height", 0))
    profile.update(
        driver="GTiff",
        count=count,
        dtype=dtype,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        blockxsize=_gtiff_block_size(width),
        blockysize=_gtiff_block_size(height),
        BIGTIFF="YES",
    )
    if np.issubdtype(np.dtype(dtype), np.floating):
        profile["predictor"] = 3
    return profile


def _remove_existing_output(path: str) -> None:
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
    except FileNotFoundError:
        pass


def build_mask_from_shapefile(
    mask_shp_path: str,
    template_profile: rasterio.profiles.Profile,
) -> Optional[np.ndarray]:
    """
    Rasterize a polygon shapefile to the template grid. Returns a boolean array
    (height, width) with True = inside the polygon(s), False = outside.
    Returns None if the shapefile is missing, empty, or geopandas is not available.
    """
    if gpd is None:
        return None
    path = Path(mask_shp_path)
    if not path.exists():
        return None
    try:
        gdf = gpd.read_file(path)
    except Exception:
        return None
    if gdf.empty or gdf.geometry is None:
        return None
    template_crs = template_profile.get("crs")
    if gdf.crs is None or str(gdf.crs) != str(template_crs):
        gdf = gdf.to_crs(template_crs or "EPSG:27700")
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return None
    transform = template_profile["transform"]
    height = template_profile["height"]
    width = template_profile["width"]
    # geometry_mask(..., invert=True) -> True for pixels that overlap shapes (inside)
    inside = geometry_mask(geoms, transform=transform, out_shape=(height, width), invert=True, all_touched=True)
    return inside


def apply_mask_to_raster(
    raster_path: str,
    mask_inside: np.ndarray,
) -> None:
    """
    Set pixels outside the mask to nodata in-place. mask_inside is (height, width),
    True = keep value, False = set to nodata. Uses the raster's existing nodata value.
    """
    if not os.path.isfile(raster_path):
        return
    with rasterio.open(raster_path, "r+") as src:
        if src.height != mask_inside.shape[0] or src.width != mask_inside.shape[1]:
            return
        nd = src.nodata
        if nd is None:
            nd = np.nan if np.issubdtype(src.dtypes[0], np.floating) else (0 if src.dtypes[0] in (np.uint8, np.int16, np.uint16) else -9999)
        for b in range(1, src.count + 1):
            arr = src.read(b, masked=False)
            arr[~mask_inside] = nd
            src.write(arr, b)


def sanitize_raster(
    raster_path: str,
    safe_min: float = SAFE_FLOAT_MIN,
    safe_max: float = SAFE_FLOAT_MAX,
) -> None:
    """
    In-place: replace inf/-inf/nan with nodata and clip float bands to [safe_min, safe_max]
    so downstream (e.g. MaxEnt) never sees non-finite or extreme values. Int bands are
    only cleaned of non-finite (if any); categorical bands are left unchanged.
    """
    if not os.path.isfile(raster_path):
        return
    with rasterio.open(raster_path, "r+") as src:
        nd = src.nodata
        for b in range(1, src.count + 1):
            arr = src.read(b, masked=False)
            dtype = arr.dtype
            if nd is None:
                band_nd = np.nan if np.issubdtype(dtype, np.floating) else (0 if dtype in (np.uint8, np.int16, np.uint16) else -9999)
            else:
                band_nd = nd
            # Replace non-finite with nodata
            if np.issubdtype(dtype, np.floating):
                bad = ~np.isfinite(arr)
                arr = np.where(bad, band_nd, arr)
                # Clip to safe range (avoid overflow in product features downstream)
                arr = np.clip(arr, safe_min, safe_max)
            else:
                bad = ~np.isfinite(arr.astype(np.float64))
                arr = np.where(bad, band_nd, arr)
            src.write(arr.astype(dtype), b)


def _sanitize_outdir(outdir: str, output_stack_path: Optional[str] = None) -> None:
    """Run sanitize_raster on all known template-grid outputs in outdir (and optional stack path)."""
    known = [
        "smuk_mean.tif",
        "smuk_std.tif",
        "peat_depth_m.tif",
        "soil_code_100m.tif",
        "distance_to_water_100m.tif",
        "elev_100m.tif",
        "slope_100m.tif",
        "aspect_100m.tif",
        "cti_100m.tif",
    ]
    n = 0
    for name in known:
        p = os.path.join(outdir, name)
        if os.path.isfile(p):
            sanitize_raster(p)
            n += 1
    if output_stack_path and os.path.isfile(output_stack_path):
        sanitize_raster(output_stack_path)
        n += 1
    if n:
        print(f"[sanitize] Cleaned inf/nan/extremes in {n} raster(s).")


def _resolve_saved_intermediates_dir(args: argparse.Namespace) -> str:
    """Choose a stable output directory for optional saved intermediates."""
    output_stack = getattr(args, "output_stack", None)
    if output_stack:
        stack_dir = os.path.dirname(os.path.abspath(output_stack))
        if stack_dir:
            return os.path.join(stack_dir, "intermediates")
    outdir = getattr(args, "outdir", None)
    if outdir:
        return os.path.join(os.path.abspath(outdir), "intermediates")
    return os.path.abspath("preprocess_potential_intermediates")


def _copy_intermediate_files(workdir: str, dest_dir: str) -> int:
    """Copy all stage-2 intermediate files out of the temp workdir before cleanup."""
    _ensure_dir(dest_dir)
    copied = 0
    for name in sorted(os.listdir(workdir)):
        src_path = os.path.join(workdir, name)
        if not os.path.isfile(src_path):
            continue
        shutil.copy2(src_path, os.path.join(dest_dir, name))
        copied += 1
    return copied


def prepare_peat_depth_to_template(
    src_path: str,
    dst_path: str,
    template_profile: rasterio.profiles.Profile,
    *,
    unit: str = "cm",
) -> None:
    """
    Reproject peat depth to the template grid, convert to metres, and fill off-peat as 0 m.

    The input depth raster is assumed to be valid only over mapped peat extent. Pixels with
    no valid source depth are set to 0 m so the national predictor stack remains finite
    outside peat and the model can still predict off peat.
    """
    if unit not in {"cm", "m"}:
        raise ValueError(f"Unsupported peat depth unit: {unit}")

    out_profile = _prepare_gtiff_profile(template_profile, count=1, dtype="float32", nodata=np.nan)

    with rasterio.open(src_path) as src:
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=template_profile["transform"],
                dst_crs=template_profile["crs"],
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
            dst.set_band_description(1, "peat_depth_m")

    max_depth = 0.0
    with rasterio.open(dst_path, "r+") as dst:
        for _, window in dst.block_windows(1):
            arr = dst.read(1, window=window, masked=False).astype(np.float32)
            valid = np.isfinite(arr)
            if unit == "cm":
                arr = np.where(valid, arr / 100.0, np.nan)
            arr = np.where(valid, arr, 0.0).astype(np.float32)
            if np.any(valid):
                max_depth = max(max_depth, float(np.nanmax(arr)))
            dst.write(arr, 1, window=window)
    print(f"[peat] Prepared peat depth in metres → {dst_path} (max={max_depth:.3f} m)")


def _resolve_raster_path(path_or_dir: str, pattern: str = "*.tif") -> str:
    """
    Resolve a user/path that might be either:
      - a direct raster path, or
      - a directory containing one or more rasters.

    For directories, the first matching GeoTIFF (sorted) is used.
    """
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        candidates = sorted(glob.glob(os.path.join(path_or_dir, pattern)))
        if not candidates:
            raise FileNotFoundError(f"No rasters matching {pattern} found in {path_or_dir}")
        return candidates[0]
    raise FileNotFoundError(f"Raster path not found: {path_or_dir}")


def load_template_from_raster(
    template_path: str,
    resolution_m: Optional[float] = None,
    crop_mask_shp: Optional[str] = None,
) -> Tuple[dict, rasterio.profiles.Profile]:
    """
    Load a reference raster and use its grid (CRS, transform, width, height)
    as the template grid for all derived predictors.

    If resolution_m is set, the grid is rebuilt from the reference's bounds and CRS
    at that pixel size (meters); otherwise the reference raster's native resolution is used.
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template raster not found: {template_path}")

    with rasterio.open(template_path) as ds:
        profile = ds.profile.copy()
        bounds = ds.bounds
        crs = ds.crs
        crop_bounds = None
        crop_note = ""
        if crop_mask_shp and gpd is not None and os.path.exists(crop_mask_shp):
            try:
                gdf = gpd.read_file(crop_mask_shp)
                if not gdf.empty and gdf.geometry is not None:
                    if gdf.crs is None:
                        raise ValueError("mask has no CRS")
                    if str(gdf.crs) != str(crs):
                        gdf = gdf.to_crs(crs or "EPSG:27700")
                    minx, miny, maxx, maxy = [float(v) for v in gdf.total_bounds]
                    minx = max(minx, bounds.left)
                    miny = max(miny, bounds.bottom)
                    maxx = min(maxx, bounds.right)
                    maxy = min(maxy, bounds.top)
                    if minx < maxx and miny < maxy:
                        crop_bounds = (minx, miny, maxx, maxy)
            except Exception:
                crop_bounds = None

        effective_bounds = crop_bounds if crop_bounds is not None else (
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
        )
        if resolution_m is not None and resolution_m > 0:
            left, bottom, right, top = effective_bounds
            origin_x = float(bounds.left)
            origin_y = float(bounds.top)
            left = origin_x + np.floor((left - origin_x) / resolution_m) * resolution_m
            right = origin_x + np.ceil((right - origin_x) / resolution_m) * resolution_m
            top = origin_y - np.floor((origin_y - top) / resolution_m) * resolution_m
            bottom = origin_y - np.ceil((origin_y - bottom) / resolution_m) * resolution_m
            width = int(np.ceil((right - left) / resolution_m))
            height = int(np.ceil((top - bottom) / resolution_m))
            transform = rasterio.transform.from_origin(
                left, top, resolution_m, resolution_m
            )
            profile.update(height=height, width=width, transform=transform)
            if crop_bounds is not None:
                crop_note = (
                    f"\n  Cropped to mask bounds from {os.path.basename(crop_mask_shp)}"
                    f" → Bounds: ({left:.1f}, {bottom:.1f}, {right:.1f}, {top:.1f})"
                )
            print(
                f"[template] Reference raster: {os.path.basename(template_path)}\n"
                f"  CRS: {crs}\n"
                f"  Bounds: {bounds}\n"
                f"  Output resolution: {resolution_m} m → Shape: {height} x {width}"
                f"{crop_note}"
            )
        else:
            if crop_bounds is not None:
                window = rasterio.windows.from_bounds(*crop_bounds, transform=ds.transform)
                col_off = max(0, int(np.floor(window.col_off)))
                row_off = max(0, int(np.floor(window.row_off)))
                col_end = min(ds.width, int(np.ceil(window.col_off + window.width)))
                row_end = min(ds.height, int(np.ceil(window.row_off + window.height)))
                width = max(1, col_end - col_off)
                height = max(1, row_end - row_off)
                window = rasterio.windows.Window(col_off, row_off, width, height)
                transform = rasterio.windows.transform(window, ds.transform)
                profile.update(height=height, width=width, transform=transform)
                crop_note = (
                    f"\n  Cropped to mask bounds from {os.path.basename(crop_mask_shp)}"
                    f" → Shape: {height} x {width}"
                )
            print(
                f"[template] Reference raster: {os.path.basename(template_path)}\n"
                f"  CRS: {ds.crs}\n"
                f"  Shape: {profile['height']} x {profile['width']}\n"
                f"  Pixel size: {ds.transform.a:.2f} x {abs(ds.transform.e):.2f}"
                f"{crop_note}"
            )
    return profile, profile


def reproject_to_template(
    src_path: str,
    dst_path: str,
    template_profile: rasterio.profiles.Profile,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """
    Reproject & resample a single-band raster to match the peat template grid.
    """
    with rasterio.open(src_path) as src:
        src_dtype = np.dtype(src.dtypes[0])
        dst_nodata = _default_nodata_for_dtype(src_dtype, src.nodata)
        dst_profile = _prepare_gtiff_profile(
            template_profile,
            count=1,
            dtype=src.dtypes[0],
            nodata=dst_nodata,
        )

        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_profile["transform"],
                dst_crs=dst_profile["crs"],
                resampling=resampling,
                src_nodata=src.nodata,
                dst_nodata=dst_nodata,
            )
    print(f"[resample] {os.path.basename(src_path)} → {os.path.basename(dst_path)}")


def _read_band_to_template(
    src_path: str,
    band: int,
    template_profile: dict,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Read one band from a raster and resample to template grid. Returns 2D float32 array."""
    H = template_profile["height"]
    W = template_profile["width"]
    out = np.full((H, W), np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
    )
    return out


def _transform_matches(a, b, tol: float = 1e-6) -> bool:
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(tuple(a), tuple(b)))


def _raster_matches_template(src_path: str, template_profile: dict) -> bool:
    with rasterio.open(src_path) as src:
        return (
            src.width == int(template_profile["width"])
            and src.height == int(template_profile["height"])
            and str(src.crs) == str(template_profile["crs"])
            and _transform_matches(src.transform, template_profile["transform"])
        )


def _copy_aligned_band(
    src: rasterio.io.DatasetReader,
    src_band: int,
    dst: rasterio.io.DatasetWriter,
    dst_band: int,
) -> None:
    for _, window in dst.block_windows(dst_band):
        arr = src.read(src_band, window=window, masked=False).astype(np.float32)
        if src.nodata is not None:
            if np.isnan(src.nodata):
                arr[~np.isfinite(arr)] = np.nan
            else:
                arr[arr == src.nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        dst.write(arr, dst_band, window=window)


def _copy_aligned_aspect_to_sin_cos(
    src: rasterio.io.DatasetReader,
    src_band: int,
    dst: rasterio.io.DatasetWriter,
    sin_band: int,
    cos_band: int,
) -> None:
    for _, window in dst.block_windows(sin_band):
        arr = src.read(src_band, window=window, masked=False).astype(np.float32)
        if src.nodata is not None:
            if np.isnan(src.nodata):
                arr[~np.isfinite(arr)] = np.nan
            else:
                arr[arr == src.nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        sin_arr = np.where(np.isfinite(arr), np.sin(np.deg2rad(arr)), np.nan).astype(np.float32)
        cos_arr = np.where(np.isfinite(arr), np.cos(np.deg2rad(arr)), np.nan).astype(np.float32)
        dst.write(sin_arr, sin_band, window=window)
        dst.write(cos_arr, cos_band, window=window)


def _read_aspect_components_to_template(
    src_path: str,
    band: int,
    template_profile: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample aspect circularly by projecting sin/cos components rather than raw
    0-360 degree values to avoid wrap-around artefacts near north.
    """
    H = template_profile["height"]
    W = template_profile["width"]
    sin_out = np.full((H, W), np.nan, dtype=np.float32)
    cos_out = np.full((H, W), np.nan, dtype=np.float32)

    with rasterio.open(src_path) as src:
        arr = src.read(band, masked=False).astype(np.float32)
        valid = np.isfinite(arr)
        if src.nodata is not None and not np.isnan(src.nodata):
            valid &= (arr != src.nodata)
        aspect_rad = np.deg2rad(arr.astype(np.float64))
        sin_src = np.where(valid, np.sin(aspect_rad), np.nan).astype(np.float32)
        cos_src = np.where(valid, np.cos(aspect_rad), np.nan).astype(np.float32)

        reproject(
            source=sin_src,
            destination=sin_out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        reproject(
            source=cos_src,
            destination=cos_out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    magnitude = np.hypot(sin_out, cos_out).astype(np.float32)
    valid_out = np.isfinite(magnitude) & (magnitude > 1e-6)
    sin_out = np.where(valid_out, sin_out / magnitude, np.nan).astype(np.float32)
    cos_out = np.where(valid_out, cos_out / magnitude, np.nan).astype(np.float32)
    return sin_out, cos_out


def write_predictor_stack(
    template_profile: dict,
    dtm_metrics_path: str,
    smuk_mean_path: str,
    smuk_std_path: str,
    peat_depth_path: str,
    output_stack_path: str,
    soil_code_path: Optional[str] = None,
    distance_water_path: Optional[str] = None,
) -> None:
    """
    Stack DTM (4 bands) + smuk_mean + smuk_std + peat_depth_m [+ optional soil_code] [+ optional distance_to_water] into one multi-band GeoTIFF.
    All layers are resampled to the template grid if needed.
    Band order: elev, slope, aspect, cti, smuk_mean, smuk_std, peat_depth_m
    [, soil_code_categorical if --include-soil] [, distance_to_water_m].
    Soil code is stored as float32 (integer values) so the stack stays float32; treat as categorical in MaxEnt.
    """
    band_names = [
        "elevation_m", "slope_deg", "sin_aspect", "cos_aspect", "cti_twi",
        "smuk_mean", "smuk_std", "peat_depth_m",
    ]
    input_paths = [dtm_metrics_path, smuk_mean_path, smuk_std_path, peat_depth_path]
    if soil_code_path and os.path.isfile(soil_code_path):
        input_paths.append(soil_code_path)
        band_names.append("soil_code_categorical")
    if distance_water_path and os.path.isfile(distance_water_path):
        input_paths.append(distance_water_path)
        band_names.append("distance_to_water_m")

    aligned_fast_path = all(_raster_matches_template(path, template_profile) for path in input_paths)

    profile = _prepare_gtiff_profile(
        template_profile,
        count=len(band_names),
        dtype="float32",
        nodata=np.nan,
    )
    _remove_existing_output(output_stack_path)

    if aligned_fast_path:
        with rasterio.open(output_stack_path, "w", **profile) as dst:
            with rasterio.open(dtm_metrics_path) as dtm_src:
                _copy_aligned_band(dtm_src, 1, dst, 1)
                _copy_aligned_band(dtm_src, 2, dst, 2)
                _copy_aligned_aspect_to_sin_cos(dtm_src, 3, dst, 3, 4)
                _copy_aligned_band(dtm_src, 4, dst, 5)
            with rasterio.open(smuk_mean_path) as src:
                _copy_aligned_band(src, 1, dst, 6)
            with rasterio.open(smuk_std_path) as src:
                _copy_aligned_band(src, 1, dst, 7)
            with rasterio.open(peat_depth_path) as src:
                _copy_aligned_band(src, 1, dst, 8)

            next_band = 9
            if soil_code_path and os.path.isfile(soil_code_path):
                with rasterio.open(soil_code_path) as src:
                    _copy_aligned_band(src, 1, dst, next_band)
                next_band += 1
            if distance_water_path and os.path.isfile(distance_water_path):
                with rasterio.open(distance_water_path) as src:
                    _copy_aligned_band(src, 1, dst, next_band)

            for i, name in enumerate(band_names, 1):
                dst.set_band_description(i, name)
        print(f"[stack] Wrote {len(band_names)} bands to {output_stack_path} (streaming aligned path)")
        return

    H = template_profile["height"]
    W = template_profile["width"]
    stack: List[np.ndarray] = []

    with rasterio.open(dtm_metrics_path) as ds:
        n_dtm = min(4, ds.count)
    for b in range(1, n_dtm + 1):
        if b == 3 and n_dtm >= 3:
            sin_aspect, cos_aspect = _read_aspect_components_to_template(
                dtm_metrics_path, b, template_profile
            )
            stack.append(sin_aspect)
            stack.append(cos_aspect)
        else:
            arr = _read_band_to_template(dtm_metrics_path, b, template_profile, Resampling.bilinear)
            stack.append(arr)
    stack.append(_read_band_to_template(smuk_mean_path, 1, template_profile, Resampling.bilinear))
    stack.append(_read_band_to_template(smuk_std_path, 1, template_profile, Resampling.bilinear))
    stack.append(_read_band_to_template(peat_depth_path, 1, template_profile, Resampling.bilinear))

    if soil_code_path and os.path.isfile(soil_code_path):
        soil_arr = _read_band_to_template(
            soil_code_path, 1, template_profile, Resampling.nearest
        )
        stack.append(soil_arr.astype(np.float32))

    if distance_water_path and os.path.isfile(distance_water_path):
        dist_arr = _read_band_to_template(
            distance_water_path, 1, template_profile, Resampling.bilinear
        )
        stack.append(dist_arr.astype(np.float32))

    with rasterio.open(output_stack_path, "w", **profile) as dst:
        for i, arr in enumerate(stack):
            dst.write(arr, i + 1)
            dst.set_band_description(i + 1, band_names[i])
    print(f"[stack] Wrote {len(stack)} bands to {output_stack_path}")


def resample_smuk_to_template(
    smuk_path_or_dir: str,
    template_profile: rasterio.profiles.Profile,
    out_mean_path: str,
    out_std_path: str,
) -> None:
    """
    Stage 2 (easy-medium):
      - Reproject/resample SMUK time series to the 100 m template grid.
      - Compute per-pixel mean and standard deviation over time.

    Memory-efficient: processes one time slice at a time and updates running
    mean / variance, instead of stacking all slices in memory.

    SMUK can be stored either:
      - as multiple single-band GeoTIFFs in a directory (one time step per file), OR
      - as multiple bands in a single GeoTIFF (one time step per band).
    """
    def _valid_mask(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
        valid = np.isfinite(arr)
        if nodata is None:
            return valid
        if np.isnan(nodata):
            return valid
        return valid & (arr != nodata)

    def _metadata_signature(src: rasterio.io.DatasetReader) -> Tuple[int, int, str, Tuple[float, ...], Optional[str]]:
        if src.nodata is None:
            nodata_sig = None
        elif np.isnan(src.nodata):
            nodata_sig = "nan"
        else:
            nodata_sig = f"{float(src.nodata):.12g}"
        return (
            src.height,
            src.width,
            str(src.crs),
            tuple(float(v) for v in src.transform),
            nodata_sig,
        )

    def _write_native_summary(path: str, arr: np.ndarray, ref_profile: dict, desc: str) -> None:
        profile = _prepare_gtiff_profile(ref_profile, count=1, dtype="float32", nodata=np.nan)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr.astype(np.float32), 1)
            dst.set_band_description(1, desc)

    native_mean_path = os.path.join(os.path.dirname(out_mean_path), "_smuk_mean_native_tmp.tif")
    native_std_path = os.path.join(os.path.dirname(out_std_path), "_smuk_std_native_tmp.tif")

    try:
        if os.path.isdir(smuk_path_or_dir):
            file_list = sorted(glob.glob(os.path.join(smuk_path_or_dir, "*.tif")))
            if not file_list:
                raise FileNotFoundError(f"No SMUK *.tif files found in {smuk_path_or_dir}")
            with rasterio.open(file_list[0]) as ref:
                ref_profile = ref.profile.copy()
                signature = _metadata_signature(ref)
                mean = np.zeros((ref.height, ref.width), dtype=np.float32)
                M2 = np.zeros((ref.height, ref.width), dtype=np.float64)
                count = np.zeros((ref.height, ref.width), dtype=np.uint16)
            consistent = True
            for fpath in file_list[1:]:
                with rasterio.open(fpath) as src:
                    if _metadata_signature(src) != signature:
                        consistent = False
                        break
            if consistent:
                print(
                    f"[smuk] Fast path: summarising {len(file_list)} slices on native SMUK grid, "
                    "then warping mean/std once"
                )
                for fpath in tqdm(file_list, desc="[smuk] Time slices", unit="slice"):
                    with rasterio.open(fpath) as src:
                        arr = src.read(1, masked=False).astype(np.float32)
                        mask = _valid_mask(arr, src.nodata)
                        if not np.any(mask):
                            continue
                        c_prev = count[mask].astype(np.float64)
                        c_new = c_prev + 1.0
                        delta = arr[mask].astype(np.float64) - mean[mask].astype(np.float64)
                        mean[mask] = (mean[mask].astype(np.float64) + delta / c_new).astype(np.float32)
                        delta2 = arr[mask].astype(np.float64) - mean[mask].astype(np.float64)
                        M2[mask] += delta * delta2
                        count[mask] = c_new.astype(count.dtype)
                std = np.full_like(mean, np.nan, dtype=np.float32)
                valid = count > 1
                std[valid] = np.sqrt(M2[valid] / (count[valid].astype(np.float64) - 1.0)).astype(np.float32)
                mean[count == 0] = np.nan
                _write_native_summary(native_mean_path, mean, ref_profile, "smuk_mean_native")
                _write_native_summary(native_std_path, std, ref_profile, "smuk_std_native")
                reproject_to_template(native_mean_path, out_mean_path, template_profile, resampling=Resampling.bilinear)
                reproject_to_template(native_std_path, out_std_path, template_profile, resampling=Resampling.bilinear)
                print(f"[smuk] Wrote mean to {out_mean_path}")
                print(f"[smuk] Wrote std  to {out_std_path}")
                return
            print("[smuk] Falling back to slow path (inconsistent slice grids detected)")
        else:
            smuk_path = smuk_path_or_dir
            if not os.path.exists(smuk_path):
                raise FileNotFoundError(f"SMUK raster not found: {smuk_path}")
            with rasterio.open(smuk_path) as src:
                ref_profile = src.profile.copy()
                mean = np.zeros((src.height, src.width), dtype=np.float32)
                M2 = np.zeros((src.height, src.width), dtype=np.float64)
                count = np.zeros((src.height, src.width), dtype=np.uint16)
                print(
                    f"[smuk] Fast path: summarising {src.count} bands on native SMUK grid, "
                    "then warping mean/std once"
                )
                for b in tqdm(range(1, src.count + 1), desc="[smuk] Time slices", unit="slice"):
                    arr = src.read(b, masked=False).astype(np.float32)
                    mask = _valid_mask(arr, src.nodata)
                    if not np.any(mask):
                        continue
                    c_prev = count[mask].astype(np.float64)
                    c_new = c_prev + 1.0
                    delta = arr[mask].astype(np.float64) - mean[mask].astype(np.float64)
                    mean[mask] = (mean[mask].astype(np.float64) + delta / c_new).astype(np.float32)
                    delta2 = arr[mask].astype(np.float64) - mean[mask].astype(np.float64)
                    M2[mask] += delta * delta2
                    count[mask] = c_new.astype(count.dtype)
            std = np.full_like(mean, np.nan, dtype=np.float32)
            valid = count > 1
            std[valid] = np.sqrt(M2[valid] / (count[valid].astype(np.float64) - 1.0)).astype(np.float32)
            mean[count == 0] = np.nan
            _write_native_summary(native_mean_path, mean, ref_profile, "smuk_mean_native")
            _write_native_summary(native_std_path, std, ref_profile, "smuk_std_native")
            reproject_to_template(native_mean_path, out_mean_path, template_profile, resampling=Resampling.bilinear)
            reproject_to_template(native_std_path, out_std_path, template_profile, resampling=Resampling.bilinear)
            print(f"[smuk] Wrote mean to {out_mean_path}")
            print(f"[smuk] Wrote std  to {out_std_path}")
            return
    finally:
        for tmp in (native_mean_path, native_std_path):
            if os.path.exists(tmp):
                os.remove(tmp)

    H = template_profile["height"]
    W = template_profile["width"]
    mean = np.zeros((H, W), dtype=np.float32)
    M2 = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.uint16)

    def _update_from_slice(src_ds: rasterio.io.DatasetReader, band: int) -> None:
        layer = np.full((H, W), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src_ds, band),
            destination=layer,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=Resampling.bilinear,
            src_nodata=src_ds.nodata,
            dst_nodata=np.nan,
        )
        mask = np.isfinite(layer)
        if not np.any(mask):
            return
        c_prev = count[mask].astype(np.float64)
        c_new = c_prev + 1.0
        delta = layer[mask].astype(np.float64) - mean[mask].astype(np.float64)
        mean[mask] = (mean[mask].astype(np.float64) + delta / c_new).astype(np.float32)
        delta2 = layer[mask].astype(np.float64) - mean[mask].astype(np.float64)
        M2[mask] += delta * delta2
        count[mask] = c_new.astype(count.dtype)

    if os.path.isdir(smuk_path_or_dir):
        file_list = sorted(glob.glob(os.path.join(smuk_path_or_dir, "*.tif")))
        if not file_list:
            raise FileNotFoundError(f"No SMUK *.tif files found in {smuk_path_or_dir}")
        n_slices = len(file_list)
        print(f"[smuk] Slow path: reprojecting {n_slices} SMUK time slices directly to template grid")
        for fpath in tqdm(file_list, desc="[smuk] Time slices", unit="slice"):
            with rasterio.open(fpath) as src:
                _update_from_slice(src, band=1)
    else:
        smuk_path = smuk_path_or_dir
        print(f"[smuk] Slow path: reprojecting SMUK time series from {smuk_path}")
        with rasterio.open(smuk_path) as src:
            n_bands = src.count
            for b in tqdm(range(1, n_bands + 1), desc="[smuk] Time slices", unit="slice"):
                _update_from_slice(src, band=b)

    smuk_mean = mean.astype("float32")
    smuk_std = np.full((H, W), np.nan, dtype=np.float32)
    valid = count > 1
    smuk_std[valid] = np.sqrt((M2[valid] / (count[valid].astype(np.float64) - 1.0))).astype(np.float32)

    out_profile = _prepare_gtiff_profile(template_profile, count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(out_mean_path, "w", **out_profile) as dst:
        dst.write(smuk_mean, 1)
    with rasterio.open(out_std_path, "w", **out_profile) as dst:
        dst.write(smuk_std, 1)
    print(f"[smuk] Wrote mean to {out_mean_path}")
    print(f"[smuk] Wrote std  to {out_std_path}")


def derive_dtm_terrain_native(
    dtm_path_or_dir: str,
    out_dir: str,
) -> Tuple[str, str, str, str]:
    """
    Stage 3 (harder):
      - Compute terrain derivatives from the DTM at its native grid using richdem.

    Returns paths to the native-resolution rasters:
      (elev_native, slope_native, aspect_native, cti_native)
    """
    if rd is None:
        raise RuntimeError(
            "richdem is not installed, but is required for DTM derivatives.\n"
            "Install it via conda (recommended), e.g.: conda install -c conda-forge richdem"
        )
    dtm_path = _resolve_raster_path(dtm_path_or_dir)

    print(f"[dtm] Loading DTM from {dtm_path}")
    with rasterio.open(dtm_path) as src:
        dem = src.read(1, masked=True).astype("float32")
        profile = src.profile.copy()

    rd_dem = rd.rdarray(dem.filled(np.nan), no_data=np.nan)

    print("[dtm] Computing slope (degrees)...")
    slope = rd.TerrainAttribute(rd_dem, attrib="slope_degrees")
    print("[dtm] Computing aspect (degrees)...")
    aspect = rd.TerrainAttribute(rd_dem, attrib="aspect")
    print("[dtm] Computing CTI/TWI...")
    cti = rd.TerrainAttribute(rd_dem, attrib="twi")

    elev_native_path = os.path.join(out_dir, "elev_native.tif")
    slope_native_path = os.path.join(out_dir, "slope_native.tif")
    aspect_native_path = os.path.join(out_dir, "aspect_native.tif")
    cti_native_path = os.path.join(out_dir, "cti_native.tif")

    profile.update(count=1, dtype="float32", nodata=np.nan)

    for arr, path in [
        (dem.filled(np.nan).astype("float32"), elev_native_path),
        (slope.astype("float32"), slope_native_path),
        (aspect.astype("float32"), aspect_native_path),
        (cti.astype("float32"), cti_native_path),
    ]:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
        print(f"[dtm] Wrote native terrain raster: {path}")

    return elev_native_path, slope_native_path, aspect_native_path, cti_native_path


def resample_dtm_products_to_template(
    elev_native: str,
    slope_native: str,
    aspect_native: str,
    cti_native: str,
    template_profile: rasterio.profiles.Profile,
    out_dir: str,
) -> None:
    """
    Stage 4 (harder):
      - Resample DTM terrain products to the modelling grid (100 m by default).
    """
    elev_100m = os.path.join(out_dir, "elev_100m.tif")
    slope_100m = os.path.join(out_dir, "slope_100m.tif")
    aspect_100m = os.path.join(out_dir, "aspect_100m.tif")
    cti_100m = os.path.join(out_dir, "cti_100m.tif")

    reproject_to_template(elev_native, elev_100m, template_profile, resampling=Resampling.bilinear)
    reproject_to_template(slope_native, slope_100m, template_profile, resampling=Resampling.bilinear)
    reproject_to_template(aspect_native, aspect_100m, template_profile, resampling=Resampling.bilinear)
    reproject_to_template(cti_native, cti_100m, template_profile, resampling=Resampling.bilinear)

    print("[dtm] Finished resampling terrain products to 100 m grid.")


def rasterize_soils_shapefile(
    soils_shp_path: str,
    template_profile: rasterio.profiles.Profile,
    out_dir: str,
    soil_group_field: str = "SOIL_GROUP",
    soil_tex_field: str = "SOIL_TEX",
) -> Optional[str]:
    """
    Rasterize a soils polygon shapefile to the template grid using unique integer
    codes derived from SOIL_GROUP and SOIL_TEX (or custom field names).

    Each (SOIL_GROUP, SOIL_TEX) combination gets a stable integer code (1, 2, 3, ...).
    Outputs:
      - soil_code_100m.tif  : int16 raster (0 = no data / no polygon)
      - soil_lookup.csv     : code, SOIL_GROUP, SOIL_TEX for downstream use

    For MaxEnt: use soil_code as a *categorical* (factor) variable so the model
    creates binary features per class instead of treating codes as continuous.

    Returns path to soil_code raster, or None if skipped/failed.
    """
    if gpd is None:
        print("[soils] Skipped: geopandas is required for soils shapefile (pip install geopandas)")
        return None

    path = Path(soils_shp_path)
    if not path.exists():
        print(f"[soils] Skipped: shapefile not found: {path}")
        return None

    print(f"[soils] Loading soils shapefile: {path.name}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        print("[soils] Skipped: shapefile is empty")
        return None

    # Normalize field names (case-insensitive match)
    cols = {c.upper(): c for c in gdf.columns}
    group_col = cols.get(soil_group_field.upper()) or cols.get("SOIL_GROUP")
    tex_col = cols.get(soil_tex_field.upper()) or cols.get("SOIL_TEX")

    if group_col is None and tex_col is None:
        print("[soils] Skipped: no SOIL_GROUP or SOIL_TEX (or given) columns found")
        return None

    # Build composite key: (group, tex) with safe string for missing
    def _safe(s) -> str:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return "NA"
        return str(s).strip() or "NA"

    if group_col is not None and tex_col is not None:
        gdf["_key"] = gdf.apply(lambda r: (_safe(r[group_col]), _safe(r[tex_col])), axis=1)
    elif group_col is not None:
        gdf["_key"] = gdf[group_col].map(_safe)
    else:
        gdf["_key"] = gdf[tex_col].map(_safe)

    # Stable unique integer code per key (sorted for reproducibility)
    unique_keys = sorted(gdf["_key"].unique())
    key_to_code = {k: i + 1 for i, k in enumerate(unique_keys)}
    gdf["_code"] = gdf["_key"].map(key_to_code)

    # Reproject to template CRS
    template_crs = template_profile.get("crs")
    if gdf.crs is None or (template_crs and str(gdf.crs) != str(template_crs)):
        gdf = gdf.to_crs(template_crs or "EPSG:27700")

    # Rasterize: (geometry, code); use 0 for no data
    transform = template_profile["transform"]
    height = template_profile["height"]
    width = template_profile["width"]
    shapes = [(geom, code) for geom, code in zip(gdf.geometry, gdf["_code"])]

    out = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int16,
        all_touched=True,
    )

    soil_code_path = os.path.join(out_dir, "soil_code_100m.tif")
    profile = _prepare_gtiff_profile(template_profile, count=1, dtype=np.int16, nodata=0)
    with rasterio.open(soil_code_path, "w", **profile) as dst:
        dst.write(out.astype(np.int16), 1)
        dst.set_band_description(1, "soil_code_categorical")
    print(f"[soils] Wrote {soil_code_path} ({len(unique_keys)} soil classes)")

    # Lookup CSV: code, SOIL_GROUP, SOIL_TEX (expand _key for two-column case)
    lookup_path = os.path.join(out_dir, "soil_lookup.csv")
    with open(lookup_path, "w", newline="") as f:
        w = csv.writer(f)
        if group_col and tex_col:
            w.writerow(["code", "SOIL_GROUP", "SOIL_TEX"])
            for k in unique_keys:
                group_val = k[0] if isinstance(k, tuple) else "NA"
                tex_val = k[1] if isinstance(k, tuple) else "NA"
                w.writerow([key_to_code[k], group_val, tex_val])
        else:
            w.writerow(["code", "SOIL_GROUP" if group_col else "SOIL_TEX"])
            for k in unique_keys:
                w.writerow([key_to_code[k], k])
    print(f"[soils] Wrote {lookup_path}")
    return soil_code_path


def compute_distance_to_water(
    rivers_shp_path: Optional[str],
    lakes_shp_path: Optional[str],
    template_profile: rasterio.profiles.Profile,
    out_dir: str,
) -> Optional[str]:
    """
    Rasterize rivers and/or lakes shapefiles to the template grid (merged into one
    water layer) and compute Euclidean distance to the nearest water body (m) per pixel.
    Water = 0 m. If both rivers and lakes are provided, they are concatenated then
    rasterized so distance is to nearest river or lake.

    Returns path to distance_to_water_100m.tif, or None if skipped/failed.
    """
    if gpd is None:
        print("[water] Skipped: geopandas is required (pip install geopandas)")
        return None

    template_crs = template_profile.get("crs")
    all_geoms: List = []

    if rivers_shp_path and Path(rivers_shp_path).exists():
        print(f"[water] Loading rivers: {Path(rivers_shp_path).name}")
        try:
            rivers_gdf = gpd.read_file(rivers_shp_path)
            if not rivers_gdf.empty and rivers_gdf.geometry is not None:
                if str(rivers_gdf.crs) != str(template_crs):
                    rivers_gdf = rivers_gdf.to_crs(template_crs or "EPSG:27700")
                all_geoms.extend([g for g in rivers_gdf.geometry if g is not None and not g.is_empty])
        except Exception as e:
            print(f"[water] Warning: could not load rivers: {e}")

    if lakes_shp_path and Path(lakes_shp_path).exists():
        print(f"[water] Loading lakes: {Path(lakes_shp_path).name}")
        try:
            lakes_gdf = gpd.read_file(lakes_shp_path)
            if not lakes_gdf.empty and lakes_gdf.geometry is not None:
                if str(lakes_gdf.crs) != str(template_crs):
                    lakes_gdf = lakes_gdf.to_crs(template_crs or "EPSG:27700")
                all_geoms.extend([g for g in lakes_gdf.geometry if g is not None and not g.is_empty])
        except Exception as e:
            print(f"[water] Warning: could not load lakes: {e}")

    if not all_geoms:
        print("[water] Skipped: no rivers or lakes shapefiles found, or both empty")
        return None

    transform = template_profile["transform"]
    height = template_profile["height"]
    width = template_profile["width"]
    shapes = [(geom, 1) for geom in all_geoms]
    water_mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    water_pixels = (water_mask > 0)
    if not water_pixels.any():
        print("[water] Skipped: no water pixels in template extent")
        return None

    # Euclidean distance transform (pixel units), then convert to meters
    pixel_size = abs(transform[0])
    pixel_distance = ndimage.distance_transform_edt(~water_pixels)
    distance_m = np.where(water_pixels, 0.0, pixel_distance * pixel_size).astype(np.float32)
    max_distance_m = 1_000.0
    distance_m = np.minimum(distance_m, max_distance_m)
    out_path = os.path.join(out_dir, "distance_to_water_100m.tif")
    profile = _prepare_gtiff_profile(template_profile, count=1, dtype=np.float32, nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(distance_m, 1)
        dst.set_band_description(1, "distance_to_water_m")
    print(f"[water] Wrote {out_path} (rivers + lakes)")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    defaults = _default_preprocess_paths()
    p = argparse.ArgumentParser(
        description="Build the abiotic predictor stack for wet woodland Elapid modelling."
    )

    p.add_argument(
        "--template",
        default=str(defaults["template"]),
        help=(
            "Reference raster defining the template grid (e.g., DTM metrics mosaic). "
            f"Default: {defaults['template']}"
        ),
    )
    p.add_argument(
        "--resolution",
        type=float,
        default=None,
        metavar="METRES",
        help=(
            "Output pixel size in metres. If set, the grid is built from the template's "
            "bounds and CRS at this resolution (e.g. 250 for 250 m). If not set, the "
            "template raster's native resolution is used."
        ),
    )
    p.add_argument(
        "--dtm-metrics",
        default=str(defaults["dtm_metrics"]),
        help=(
            "Path to 4-band DTM metrics raster (elev, slope, aspect, cti). "
            f"Default: same as --template ({defaults['dtm_metrics']})"
        ),
    )
    p.add_argument(
        "--output-stack",
        default=str(defaults["output_stack"]),
        help=(
            "Output path for stacked multi-band predictor raster (DTM 4 bands + smuk_mean + smuk_std + peat_depth_m). "
            f"Default: {defaults['output_stack']}. Set to empty to skip stacking."
        ),
    )
    p.add_argument(
        "--peat-depth",
        "--peat-prob",
        dest="peat_depth",
        default=str(defaults["peat_depth"]),
        help=(
            "Path to peat depth raster, or a directory containing it. "
            "Depth is converted to metres and off-peat is filled as 0 m on the template grid. "
            f"Default: {defaults['peat_depth']}"
        ),
    )
    p.add_argument(
        "--peat-depth-unit",
        choices=("cm", "m"),
        default="cm",
        help="Units of the peat depth raster. Default: cm",
    )
    p.add_argument(
        "--smuk",
        default=str(defaults["smuk"]),
        help=(
            "Path to SMUK multi-band soil moisture raster, or directory containing it. "
            "If provided, mean and std over bands will be computed on the template grid. "
            f"Default: {defaults['smuk']}"
        ),
    )
    p.add_argument(
        "--dtm",
        default=None,
        help=(
            "Optional path to DTM raster (no longer required if DTM metrics were computed "
            "via the tiled script). Kept for backwards compatibility."
        ),
    )
    p.add_argument(
        "--outdir",
        default=str(defaults["outdir"]),
        help=f"Used only as parent dir for default --output-stack path. No other files written here. Default: {defaults['outdir']}",
    )
    p.add_argument(
        "--save-intermediates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Copy stage-2 intermediate rasters and lookup tables into "
            "<output-stack-dir>/intermediates before the temp workdir is removed "
            "(default: enabled)."
        ),
    )
    p.add_argument(
        "--soils-shp",
        default=str(defaults["soils_shp"]),
        help=(
            "Path to soils polygon shapefile. Rasterized to template grid using "
            "unique integer codes from SOIL_GROUP and SOIL_TEX when --include-soil is set. "
            "For MaxEnt, declare soil_code as categorical. "
            f"Default: {defaults['soils_shp']} (skipped if missing or if soil is not included)."
        ),
    )
    p.add_argument(
        "--include-soil",
        action="store_true",
        help=(
            "Include soil_code_categorical in the predictor stack. Default: off, because "
            "parent-material class boundaries can create hard 1 km edges."
        ),
    )
    p.add_argument(
        "--soil-group-field",
        default="SOIL_GROUP",
        help="Attribute name for soil group in soils shapefile. Default: SOIL_GROUP",
    )
    p.add_argument(
        "--soil-tex-field",
        default="SOIL_TEX",
        help="Attribute name for soil texture in soils shapefile. Default: SOIL_TEX",
    )
    p.add_argument(
        "--rivers",
        default=str(defaults["rivers"]),
        help=(
            "Path to rivers/streams shapefile for distance-to-water (m) per pixel. "
            f"Default: {defaults['rivers']} (skipped if missing)."
        ),
    )
    p.add_argument(
        "--lakes",
        default=str(defaults["lakes"]),
        help=(
            "Path to lakes shapefile. Merged with rivers for distance-to-water (m). "
            f"Default: {defaults['lakes']} (skipped if missing)."
        ),
    )
    p.add_argument(
        "--mask-shp",
        default=str(defaults["mask_shp"]),
        help=(
            "Path to polygon shapefile (e.g. england.shp). All template-grid outputs are "
            "masked so pixels outside the polygon(s) are set to nodata. "
            f"Default: {defaults['mask_shp']} (skipped if missing)."
        ),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    defaults = _default_preprocess_paths()
    args = build_parser().parse_args(argv)
    args.template = str(Path(args.template).expanduser())
    args.dtm_metrics = str(Path(args.dtm_metrics).expanduser())
    if getattr(args, "output_stack", None):
        args.output_stack = str(Path(args.output_stack).expanduser())
    if getattr(args, "outdir", None):
        args.outdir = str(Path(args.outdir).expanduser())
    _require_stage2_dtm_inputs(args.template, args.dtm_metrics, defaults)

    print("🧭 Abiotic preprocessing for Elapid")
    print("=" * 60)
    print(f"Template raster:  {args.template}")
    print(f"DTM metrics:      {args.dtm_metrics}")
    print(f"SMUK source:      {args.smuk}")
    print(f"Peat depth:       {args.peat_depth}")
    print(f"Peat depth unit:  {args.peat_depth_unit}")
    print(f"Output stack:     {args.output_stack if getattr(args, 'output_stack', None) else 'Off'}")
    if getattr(args, "include_soil", False):
        print(f"Soil predictor:   Included from {args.soils_shp}")
    else:
        print("Soil predictor:   Excluded by default")
        print("WARNING: soil_code_categorical is not included in the predictor stack. Pass --include-soil to add it.")

    # Only the stack is written to disk; all intermediates go to a temp dir and are removed
    workdir = tempfile.mkdtemp(prefix="preprocess_potential_")
    try:
        # Ensure output stack path has a parent directory
        if getattr(args, "output_stack", None) and args.output_stack:
            out_stack_dir = os.path.dirname(os.path.abspath(args.output_stack))
            if out_stack_dir:
                os.makedirs(out_stack_dir, exist_ok=True)

        # 1) Template grid from reference raster (optionally at --resolution)
        template_profile, _ = load_template_from_raster(
            args.template,
            resolution_m=getattr(args, "resolution", None),
            crop_mask_shp=getattr(args, "mask_shp", None),
        )

        # 1b) Optional mask (e.g. england.shp): True = inside, pixels outside set to nodata on all outputs
        mask_inside = None
        if getattr(args, "mask_shp", None):
            mask_inside = build_mask_from_shapefile(args.mask_shp, template_profile)
            if mask_inside is not None:
                print(f"[mask] Using mask from {args.mask_shp} ({mask_inside.sum():,} pixels inside)")
            else:
                print(f"[mask] Skipped (missing or empty: {args.mask_shp})")

        # 2) SMUK mean & std on template grid (easy-medium)
        if args.smuk:
            smuk_mean_path = os.path.join(workdir, "smuk_mean.tif")
            smuk_std_path = os.path.join(workdir, "smuk_std.tif")
            resample_smuk_to_template(args.smuk, template_profile, smuk_mean_path, smuk_std_path)
            if mask_inside is not None:
                apply_mask_to_raster(smuk_mean_path, mask_inside)
                apply_mask_to_raster(smuk_std_path, mask_inside)
        else:
            print("[smuk] Skipped (no --smuk provided)")

        # 3) Peat depth resampled to template grid and converted to metres
        if args.peat_depth:
            peat_src = _resolve_raster_path(args.peat_depth)
            peat_out = os.path.join(workdir, "peat_depth_m.tif")
            print(f"[peat] Resampling peat depth from {peat_src} to template grid...")
            prepare_peat_depth_to_template(
                peat_src,
                peat_out,
                template_profile,
                unit=args.peat_depth_unit,
            )
            if mask_inside is not None:
                apply_mask_to_raster(peat_out, mask_inside)
        else:
            print("[peat] Skipped (no --peat-depth provided)")

        # 3b) Soils shapefile -> soil_code_100m.tif + soil_lookup.csv (optional)
        soil_code_path = None
        if getattr(args, "include_soil", False) and getattr(args, "soils_shp", None):
            soil_code_path = rasterize_soils_shapefile(
                args.soils_shp,
                template_profile,
                workdir,
                soil_group_field=getattr(args, "soil_group_field", "SOIL_GROUP"),
                soil_tex_field=getattr(args, "soil_tex_field", "SOIL_TEX"),
            )
        elif getattr(args, "include_soil", False):
            print("[soils] Skipped (no --soils-shp provided)")
        else:
            print("[soils] Not included in predictor stack (default). Use --include-soil to enable.")
        if mask_inside is not None and soil_code_path and os.path.isfile(soil_code_path):
            apply_mask_to_raster(soil_code_path, mask_inside)

        # 3c) Rivers + lakes -> distance_to_water_100m.tif
        distance_water_path = None
        if getattr(args, "rivers", None) or getattr(args, "lakes", None):
            distance_water_path = compute_distance_to_water(
                getattr(args, "rivers", None),
                getattr(args, "lakes", None),
                template_profile,
                workdir,
            )
        else:
            print("[water] Skipped (no --rivers or --lakes provided)")
        if mask_inside is not None and distance_water_path and os.path.isfile(distance_water_path):
            apply_mask_to_raster(distance_water_path, mask_inside)

        # 4) DTM terrain derivatives (normally already done via tiled script)
        if args.dtm:
            print("[dtm] DTM path provided, deriving terrain attributes at native resolution...")
            elev_native, slope_native, aspect_native, cti_native = derive_dtm_terrain_native(
                args.dtm,
                workdir,
            )

            print("[dtm] Resampling terrain products to template grid...")
            resample_dtm_products_to_template(
                elev_native=elev_native,
                slope_native=slope_native,
                aspect_native=aspect_native,
                cti_native=cti_native,
                template_profile=template_profile,
                out_dir=workdir,
            )
            if mask_inside is not None:
                for name in ("elev_100m", "slope_100m", "aspect_100m", "cti_100m"):
                    p = os.path.join(workdir, f"{name}.tif")
                    if os.path.isfile(p):
                        apply_mask_to_raster(p, mask_inside)
        else:
            print("[dtm] Skipped (no --dtm provided; assuming DTM metrics supplied separately)")

        # 5) Write only output: multi-band stack
        if getattr(args, "output_stack", None) and args.output_stack:
            dtm_path = getattr(args, "dtm_metrics", None) or args.template
            smuk_mean_path = os.path.join(workdir, "smuk_mean.tif")
            smuk_std_path = os.path.join(workdir, "smuk_std.tif")
            peat_path = os.path.join(workdir, "peat_depth_m.tif")
            if all(os.path.isfile(p) for p in (dtm_path, smuk_mean_path, smuk_std_path, peat_path)):
                write_predictor_stack(
                    template_profile,
                    dtm_path,
                    smuk_mean_path,
                    smuk_std_path,
                    peat_path,
                    args.output_stack,
                    soil_code_path=soil_code_path,
                    distance_water_path=distance_water_path,
                )
                if mask_inside is not None and os.path.isfile(args.output_stack):
                    apply_mask_to_raster(args.output_stack, mask_inside)
                # Sanitize the stack (inf/nan/extremes)
                sanitize_raster(args.output_stack)
                print(f"\nOutput written: {args.output_stack}")
            else:
                print("[stack] Skipped (missing one of: dtm_metrics, smuk_mean, smuk_std, peat_depth_m)")
        else:
            print("[stack] Skipped (no --output-stack path)")
    finally:
        if getattr(args, "save_intermediates", False):
            try:
                saved_dir = _resolve_saved_intermediates_dir(args)
                copied = _copy_intermediate_files(workdir, saved_dir)
                print(f"[save] Copied {copied} intermediate file(s) to {saved_dir}")
            except Exception as exc:
                print(f"[save] WARNING: Failed to copy intermediates: {exc}")
        shutil.rmtree(workdir, ignore_errors=True)

    if getattr(args, "save_intermediates", False):
        print("Done. The stack GeoTIFF and saved intermediate files were written to disk.")
    else:
        print("Done. Only the stack GeoTIFF was written to disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
