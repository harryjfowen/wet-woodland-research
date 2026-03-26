#!/usr/bin/env python3
"""
Tiled preprocessing of DTM-based abiotic metrics for wet woodland potential mapping.

This script reuses the robust tiling/VRT pattern from `wet_areas_tiled.py`
to compute terrain metrics from DTM tiles in a memory-safe, parallel way.

Per-tile outputs (multiband, at the requested output resolution):
  Band 1: elevation (m)
  Band 2: slope (degrees)
  Band 3: aspect (degrees)
  Band 4: CTI / TWI (dimensionless)

Workflow:
  1) Build a VRT over a directory of DTM tiles (EPSG:27700).
  2) Define a grid of processing windows (tiles) with optional buffer.
  3) For each tile, load DTM, compute metrics with richdem, optionally downsample
     to a coarser grid (e.g. 100 m), and write a multiband GeoTIFF.
  4) Optionally mosaic all tile outputs into a single national raster.

Example:
  python build_dtm_metrics.py \
      --dtm-dir data/input/dtm \
      --outdir data/output/preprocess \
      --pixel-size 10 \
      --output-resolution 100 \
      --tile-size 8192 \
      --buffer 512 \
      --workers 8 \
      --mosaic
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.warp import reproject
from tqdm import tqdm

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    from shapely.geometry import box as shapely_box
    from shapely.ops import unary_union
    from shapely.prepared import prep as shapely_prep
except ImportError:
    shapely_box = None
    unary_union = None
    shapely_prep = None


def _default_dtm_stage_paths() -> Dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    input_root = repo_root / "data" / "input"
    output_root = repo_root / "data" / "output"
    return {
        "dtm_dir": input_root / "dtm",
        "england_shp": input_root / "boundaries" / "england.shp",
        "outdir": output_root / "preprocess",
    }


def suppress_output():
    """
    Context manager to silence noisy C/C++ libraries (like richdem/GDAL)
    that write directly to stdout/stderr.
    """
    import contextlib
    import os as _os
    import sys as _sys

    @contextlib.contextmanager
    def _cm():
        with open(_os.devnull, "w") as devnull:
            old_stdout_fd, old_stderr_fd = _os.dup(1), _os.dup(2)
            old_stdout, old_stderr = _sys.stdout, _sys.stderr
            try:
                _sys.stdout = _sys.stderr = devnull
                _os.dup2(devnull.fileno(), 1)
                _os.dup2(devnull.fileno(), 2)
                yield
            finally:
                _os.dup2(old_stdout_fd, 1)
                _os.dup2(old_stderr_fd, 2)
                _os.close(old_stdout_fd)
                _os.close(old_stderr_fd)
                _sys.stdout, _sys.stderr = old_stdout, old_stderr

    return _cm()


try:
    with suppress_output():
        import richdem as rd  # type: ignore
    try:
        rd.rdOptions["VERBOSE"] = False
    except Exception:
        pass
except ImportError:
    rd = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


_WORKER_VRT = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _init_worker_vrt(vrt_path: str) -> None:
    """Open the shared VRT once per worker process."""
    global _WORKER_VRT
    _WORKER_VRT = rasterio.open(vrt_path, sharing=False)


def _get_worker_vrt(vrt_path: str):
    """Return the worker-local VRT handle, opening lazily if needed."""
    global _WORKER_VRT
    if _WORKER_VRT is None:
        _WORKER_VRT = rasterio.open(vrt_path, sharing=False)
    return _WORKER_VRT


def find_dtm_tiles(dtm_dir: Path, pattern: str = "*.tif") -> List[Path]:
    """Find all candidate DTM tiles in a directory."""
    tiles = list(dtm_dir.glob(pattern))
    tiles.sort()
    return tiles


def validate_dtm_inputs(dtm_files: List[Path]) -> Tuple[List[Path], List[Dict[str, str]]]:
    """
    Validate that all source rasters are readable and compatible with the
    single-band elevation VRT expected by this stage.
    """
    valid_files: List[Path] = []
    rejected: List[Dict[str, str]] = []
    reference_crs = None

    for dtm_file in dtm_files:
        try:
            with rasterio.open(dtm_file) as src:
                if src.count != 1:
                    rejected.append(
                        {
                            "path": str(dtm_file),
                            "reason": f"expected 1 band, found {src.count}",
                        }
                    )
                    continue
                if src.crs is None:
                    rejected.append(
                        {
                            "path": str(dtm_file),
                            "reason": "missing CRS",
                        }
                    )
                    continue
                if reference_crs is None:
                    reference_crs = src.crs
                elif src.crs != reference_crs:
                    rejected.append(
                        {
                            "path": str(dtm_file),
                            "reason": f"CRS mismatch: expected {reference_crs}, found {src.crs}",
                        }
                    )
                    continue
                valid_files.append(dtm_file)
        except Exception as exc:
            rejected.append(
                {
                    "path": str(dtm_file),
                    "reason": f"unreadable: {exc}",
                }
            )

    return valid_files, rejected


def create_virtual_raster(
    dtm_files: List[Path],
    vrt_path: Path,
    mask_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """Create a VRT from source rasters, cropped to the union or mask extent."""
    if not dtm_files:
        raise ValueError("No input rasters provided for VRT creation.")

    print(f"📄 Creating virtual raster from {len(dtm_files)} tiles...")

    if mask_bounds is not None:
        min_x, min_y, max_x, max_y = mask_bounds
    else:
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        readable = 0
        for dtm_file in dtm_files:
            try:
                with rasterio.open(dtm_file) as src:
                    bounds = src.bounds
                    min_x = min(min_x, bounds.left)
                    min_y = min(min_y, bounds.bottom)
                    max_x = max(max_x, bounds.right)
                    max_y = max(max_y, bounds.top)
                    readable += 1
            except Exception as e:
                print(f"⚠️  Could not read bounds from {dtm_file}: {e}")
        if readable == 0:
            raise RuntimeError("No readable rasters found for VRT creation.")

    cmd = [
        "gdalbuildvrt",
        "-overwrite",
        "-te",
        str(min_x),
        str(min_y),
        str(max_x),
        str(max_y),
        str(vrt_path),
    ] + [str(p) for p in dtm_files]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "gdalbuildvrt not found. Install GDAL and ensure gdalbuildvrt is on PATH."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gdalbuildvrt failed: {e.stderr or e}") from e
    stderr_lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    skipped_lines = [line for line in stderr_lines if "Skipping " in line]
    if skipped_lines:
        joined = "\n".join(skipped_lines)
        raise RuntimeError(
            "gdalbuildvrt skipped one or more source rasters:\n"
            f"{joined}"
        )
    if stderr_lines:
        print("⚠️  gdalbuildvrt reported warnings:")
        for line in stderr_lines:
            print(f"   {line}")


def get_tile_bounds_with_buffer(
    vrt_dataset,
    tile_size: int,
    buffer: int,
    tile_row: int,
    tile_col: int,
) -> Tuple[Window, rasterio.coords.BoundingBox, Window, rasterio.coords.BoundingBox]:
    """Calculate core and buffered tile windows plus their bounds."""
    height, width = vrt_dataset.height, vrt_dataset.width

    col_start = tile_col * tile_size
    row_start = tile_row * tile_size
    col_end = min(col_start + tile_size, width)
    row_end = min(row_start + tile_size, height)

    core_window = Window(
        col_off=col_start,
        row_off=row_start,
        width=col_end - col_start,
        height=row_end - row_start,
    )

    buffered_col_start = max(0, col_start - buffer)
    buffered_row_start = max(0, row_start - buffer)
    buffered_col_end = min(width, col_end + buffer)
    buffered_row_end = min(height, row_end + buffer)

    buffered_window = Window(
        col_off=buffered_col_start,
        row_off=buffered_row_start,
        width=buffered_col_end - buffered_col_start,
        height=buffered_row_end - buffered_row_start,
    )

    core_bounds = rasterio.coords.BoundingBox(
        *rasterio.windows.bounds(core_window, vrt_dataset.transform)
    )
    buffered_bounds = rasterio.coords.BoundingBox(
        *rasterio.windows.bounds(buffered_window, vrt_dataset.transform)
    )
    return core_window, core_bounds, buffered_window, buffered_bounds


def _load_polygon_geometries(mask_path: Path, target_crs) -> Tuple[List[object], rasterio.coords.BoundingBox]:
    """Load polygon geometries from a vector layer and project them to target_crs."""
    if gpd is None:
        raise RuntimeError("geopandas is required for coverage-mask diagnostics.")
    gdf = gpd.read_file(mask_path)
    if gdf.empty:
        raise RuntimeError(f"Coverage mask is empty: {mask_path}")
    if gdf.crs is None:
        raise RuntimeError(f"Coverage mask has no CRS: {mask_path}")
    if str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    geometries = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        raise RuntimeError(f"Coverage mask contains no polygon geometries: {mask_path}")
    return geometries, rasterio.coords.BoundingBox(*gdf.total_bounds)


def _intersect_bounds(
    a: rasterio.coords.BoundingBox,
    b: rasterio.coords.BoundingBox,
) -> Optional[rasterio.coords.BoundingBox]:
    left = max(a.left, b.left)
    bottom = max(a.bottom, b.bottom)
    right = min(a.right, b.right)
    top = min(a.top, b.top)
    if left >= right or bottom >= top:
        return None
    return rasterio.coords.BoundingBox(left, bottom, right, top)


def _geometry_bounds(geometries: List[object]) -> rasterio.coords.BoundingBox:
    """Compute overall bounds for an iterable of shapely geometries."""
    left = min(geom.bounds[0] for geom in geometries)
    bottom = min(geom.bounds[1] for geom in geometries)
    right = max(geom.bounds[2] for geom in geometries)
    top = max(geom.bounds[3] for geom in geometries)
    return rasterio.coords.BoundingBox(left, bottom, right, top)


def _expand_bounds(
    bounds: rasterio.coords.BoundingBox,
    margin_m: float,
) -> rasterio.coords.BoundingBox:
    """Expand bounds uniformly in all directions by the given margin in metres."""
    return rasterio.coords.BoundingBox(
        bounds.left - margin_m,
        bounds.bottom - margin_m,
        bounds.right + margin_m,
        bounds.top + margin_m,
    )


def _prepare_mask_filter(
    mask_path: Path,
    target_crs,
) -> Tuple[rasterio.coords.BoundingBox, Optional[object]]:
    """
    Load a polygon mask for window pruning.
    Returns the mask bounds and an optional prepared geometry for exact intersects.
    """
    geometries, geom_bounds = _load_polygon_geometries(mask_path, target_crs)
    prepared = None
    if unary_union is not None and shapely_prep is not None:
        prepared = shapely_prep(unary_union(geometries))
    elif geometries:
        geom_bounds = _geometry_bounds(geometries)
    return geom_bounds, prepared


def _tile_intersects_mask(
    tile_bounds: rasterio.coords.BoundingBox,
    mask_bounds: rasterio.coords.BoundingBox,
    prepared_mask: Optional[object],
) -> bool:
    """Fast bbox prune with optional exact polygon intersection."""
    if _intersect_bounds(tile_bounds, mask_bounds) is None:
        return False
    if prepared_mask is None or shapely_box is None:
        return True
    tile_geom = shapely_box(
        tile_bounds.left,
        tile_bounds.bottom,
        tile_bounds.right,
        tile_bounds.top,
    )
    return bool(prepared_mask.intersects(tile_geom))


def _build_profile_from_bounds(
    bounds: rasterio.coords.BoundingBox,
    crs,
    resolution_m: float,
    dtype: str,
    nodata,
) -> Dict[str, object]:
    width = max(1, int(math.ceil((bounds.right - bounds.left) / resolution_m)))
    height = max(1, int(math.ceil((bounds.top - bounds.bottom) / resolution_m)))
    transform = Affine(resolution_m, 0.0, bounds.left, 0.0, -resolution_m, bounds.top)
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }


def check_vrt_coverage(
    vrt_path: Path,
    mask_path: Path,
    outdir: Path,
    *,
    resolution_m: float,
    min_hole_area_ha: float,
    max_report_holes: int = 8,
) -> Optional[Dict[str, object]]:
    """
    Sample the VRT onto a coarse grid inside a polygon mask and report missing-data holes.
    Writes a small diagnostic raster where 1 = covered, 2 = missing, 0 = outside mask.
    """
    if resolution_m <= 0:
        return None

    with rasterio.open(vrt_path) as vrt:
        geometries, geom_bounds = _load_polygon_geometries(mask_path, vrt.crs)
        check_bounds = _intersect_bounds(vrt.bounds, geom_bounds)
        if check_bounds is None:
            raise RuntimeError(
                f"Coverage mask {mask_path} does not overlap the DTM VRT extent."
            )
        profile = _build_profile_from_bounds(
            check_bounds,
            vrt.crs,
            resolution_m,
            dtype="uint8",
            nodata=0,
        )
        mask = rasterize(
            [(geom, 1) for geom in geometries],
            out_shape=(profile["height"], profile["width"]),
            transform=profile["transform"],
            fill=0,
            dtype=np.uint8,
        ).astype(bool)
        if not np.any(mask):
            raise RuntimeError(
                f"Coverage mask {mask_path} rasterized to 0 cells at {resolution_m:g} m."
            )
        sample = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(vrt, 1),
            destination=sample,
            src_transform=vrt.transform,
            src_crs=vrt.crs,
            dst_transform=profile["transform"],
            dst_crs=vrt.crs,
            resampling=Resampling.nearest,
            src_nodata=vrt.nodata,
            dst_nodata=np.nan,
        )

    covered = mask & np.isfinite(sample)
    missing = mask & ~covered
    inside_cells = int(mask.sum())
    covered_cells = int(covered.sum())
    missing_cells = int(missing.sum())
    cell_area_ha = (resolution_m * resolution_m) / 10_000.0
    coverage_pct = (100.0 * covered_cells / inside_cells) if inside_cells > 0 else 0.0
    missing_area_ha = missing_cells * cell_area_ha

    coverage_raster = np.zeros(mask.shape, dtype=np.uint8)
    coverage_raster[covered] = 1
    coverage_raster[missing] = 2
    coverage_path = outdir / f"dtm_vrt_coverage_{int(round(resolution_m))}m.tif"
    with rasterio.open(coverage_path, "w", **profile) as dst:
        dst.write(coverage_raster, 1)
        dst.set_band_description(1, "vrt_coverage")
        dst.update_tags(
            coverage_codes="0=outside_mask,1=covered,2=missing",
            coverage_mask=str(mask_path),
            coverage_resolution_m=str(resolution_m),
        )

    holes: List[Dict[str, object]] = []
    if missing_cells > 0 and ndimage is not None:
        labels, n_labels = ndimage.label(missing, structure=np.ones((3, 3), dtype=np.uint8))
        counts = np.bincount(labels.ravel())[1:]
        order = np.argsort(counts)[::-1]
        for idx in order:
            cell_count = int(counts[idx])
            area_ha = cell_count * cell_area_ha
            if area_ha < min_hole_area_ha:
                break
            label_id = idx + 1
            rows, cols = np.where(labels == label_id)
            if rows.size == 0:
                continue
            center_row = float(rows.mean())
            center_col = float(cols.mean())
            center_x = profile["transform"].c + (center_col + 0.5) * profile["transform"].a
            center_y = profile["transform"].f + (center_row + 0.5) * profile["transform"].e
            holes.append(
                {
                    "area_ha": area_ha,
                    "cells": cell_count,
                    "center_x": center_x,
                    "center_y": center_y,
                }
            )
            if len(holes) >= max_report_holes:
                break

    print("\n🔎 VRT coverage preflight")
    print(f"   Mask: {mask_path}")
    print(f"   Check resolution: {resolution_m:.1f} m")
    print(f"   Cells inside mask: {inside_cells:,}")
    print(f"   Covered cells: {covered_cells:,} ({coverage_pct:.2f}%)")
    print(f"   Missing cells: {missing_cells:,} ({missing_area_ha:,.1f} ha)")
    print(f"   Coverage raster: {coverage_path}")
    if missing_cells == 0:
        print("   No missing-data holes detected inside the mask.")
    elif holes:
        print(f"   Large missing regions (>= {min_hole_area_ha:,.1f} ha):")
        for hole in holes:
            print(
                "     "
                f"{hole['area_ha']:,.1f} ha around "
                f"({hole['center_x']:.0f}, {hole['center_y']:.0f})"
            )
    elif ndimage is None:
        print("   Missing cells found, but scipy.ndimage is unavailable for hole grouping.")
    else:
        print(
            "   Missing cells found, but no individual region exceeded "
            f"{min_hole_area_ha:,.1f} ha."
        )

    return {
        "mask_path": str(mask_path),
        "resolution_m": float(resolution_m),
        "cells_inside_mask": inside_cells,
        "covered_cells": covered_cells,
        "missing_cells": missing_cells,
        "coverage_pct": coverage_pct,
        "missing_area_ha": missing_area_ha,
        "coverage_raster": str(coverage_path),
        "large_holes": holes,
    }


def write_vrt_elevation_preview(
    vrt_path: Path,
    outdir: Path,
    *,
    resolution_m: float,
    mask_path: Optional[Path] = None,
) -> Optional[Dict[str, object]]:
    """
    Write a coarse elevation-only preview from the VRT for quick visual QA.
    If a polygon mask is provided, clip the preview extent to that mask and set
    pixels outside the mask to nodata.
    """
    if resolution_m <= 0:
        return None

    with rasterio.open(vrt_path) as vrt:
        preview_bounds = vrt.bounds
        mask = None
        mask_used: Optional[str] = None
        if mask_path is not None and mask_path.is_file():
            geometries, geom_bounds = _load_polygon_geometries(mask_path, vrt.crs)
            intersect_bounds = _intersect_bounds(vrt.bounds, geom_bounds)
            if intersect_bounds is None:
                raise RuntimeError(
                    f"Preview mask {mask_path} does not overlap the DTM VRT extent."
                )
            preview_bounds = intersect_bounds
            profile = _build_profile_from_bounds(
                preview_bounds,
                vrt.crs,
                resolution_m,
                dtype="float32",
                nodata=np.nan,
            )
            mask = rasterize(
                [(geom, 1) for geom in geometries],
                out_shape=(profile["height"], profile["width"]),
                transform=profile["transform"],
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
            mask_used = str(mask_path)
        else:
            profile = _build_profile_from_bounds(
                preview_bounds,
                vrt.crs,
                resolution_m,
                dtype="float32",
                nodata=np.nan,
            )

        preview = np.full((profile["height"], profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(vrt, 1),
            destination=preview,
            src_transform=vrt.transform,
            src_crs=vrt.crs,
            dst_transform=profile["transform"],
            dst_crs=vrt.crs,
            resampling=Resampling.bilinear,
            src_nodata=vrt.nodata,
            dst_nodata=np.nan,
        )
        if mask is not None:
            preview = np.where(mask, preview, np.nan).astype(np.float32)

    preview_path = outdir / f"dtm_vrt_elevation_preview_{int(round(resolution_m))}m.tif"
    with rasterio.open(preview_path, "w", **profile) as dst:
        dst.write(preview, 1)
        dst.set_band_description(1, "elevation_m_preview")
        dst.update_tags(
            title="DTM VRT elevation preview",
            preview_resolution_m=str(resolution_m),
            preview_mask=mask_used or "None",
        )

    valid = np.isfinite(preview)
    print("\n🖼️  VRT elevation preview")
    print(f"   Resolution: {resolution_m:.1f} m")
    print(f"   Valid cells: {int(valid.sum()):,}")
    print(f"   Output: {preview_path}")
    if mask_used:
        print(f"   Mask: {mask_used}")

    return {
        "resolution_m": float(resolution_m),
        "preview_path": str(preview_path),
        "mask_path": mask_used,
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "valid_cells": int(valid.sum()),
    }


def apply_polygon_mask_to_raster(
    raster_path: Path,
    mask_path: Path,
    *,
    buffer_m: float = 0.0,
) -> bool:
    """
    Mask a raster in-place so pixels outside the polygon mask are set to nodata.
    Returns True if the mask was applied, False if skipped.
    """
    if not raster_path.is_file() or not mask_path.is_file():
        return False

    with rasterio.open(raster_path, "r+") as src:
        if gpd is None:
            raise RuntimeError("geopandas is required for polygon clipping.")

        gdf = gpd.read_file(mask_path)
        if gdf.empty:
            raise RuntimeError(f"Clip mask is empty: {mask_path}")
        if gdf.crs is None:
            raise RuntimeError(f"Clip mask has no CRS: {mask_path}")
        if str(gdf.crs) != str(src.crs):
            gdf = gdf.to_crs(src.crs)
        if buffer_m != 0.0:
            gdf = gdf.copy()
            gdf.geometry = gdf.geometry.buffer(buffer_m)
        geometries = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
        if not geometries:
            raise RuntimeError(
                f"Clip mask has no polygon area after buffering by {buffer_m:g} m: {mask_path}"
            )
        mask = rasterize(
            [(geom, 1) for geom in geometries],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)

        nodata = src.nodata
        if nodata is None:
            nodata = np.nan if np.issubdtype(np.dtype(src.dtypes[0]), np.floating) else 0

        for band in range(1, src.count + 1):
            arr = src.read(band, masked=False)
            arr[~mask] = nodata
            src.write(arr, band)

    return True


def extract_channels_flow_and_depressions(
    dem: np.ndarray,
    accumulation_threshold: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract channels, flow accumulation, and depression depth from a DEM."""
    valid_mask = np.isfinite(dem)
    if not np.any(valid_mask):
        channels = np.zeros_like(dem, dtype=bool)
        flow_accum = np.full_like(dem, np.nan, dtype=np.float32)
        depression_depth = np.full_like(dem, np.nan, dtype=np.float32)
        return channels, flow_accum, depression_depth

    if rd is None:
        raise RuntimeError(
            "richdem is required for CTI/flow metrics. Install via conda-forge."
        )

    with suppress_output():
        original_dem = rd.rdarray(dem.copy(), no_data=np.nan)
        filled_dem = rd.rdarray(dem.copy(), no_data=np.nan)
        rd.FillDepressions(filled_dem, in_place=True)
        flow_accum_rd = rd.FlowAccumulation(filled_dem, method="D8")

    flow_accum = np.array(flow_accum_rd, dtype=np.float32)
    flow_accum[~valid_mask] = np.nan

    depression_depth = np.array(filled_dem) - np.array(original_dem)
    depression_depth = depression_depth.astype(np.float32)
    depression_depth[depression_depth < 0] = 0
    depression_depth[~valid_mask] = np.nan

    channels = flow_accum >= accumulation_threshold
    channels = np.where(valid_mask, channels, False)
    if ndimage is not None:
        channels = ndimage.binary_opening(channels, structure=np.ones((3, 3)))

    return channels.astype(bool), flow_accum, depression_depth


def calculate_topographic_wetness_index(
    dem: np.ndarray,
    flow_accum: np.ndarray,
    pixel_size: float = 4.0,
) -> np.ndarray:
    """Calculate CTI/TWI robustly while preserving DEM nodata."""
    valid_mask = np.isfinite(dem)
    twi = np.full_like(dem, np.nan, dtype=np.float32)
    if not np.any(valid_mask):
        return twi

    dem_fill_value = float(np.nanmedian(dem[valid_mask]))
    dem_filled = np.where(valid_mask, dem, dem_fill_value).astype(np.float32)

    dy, dx = np.gradient(dem_filled, pixel_size)
    slope = np.sqrt(dx**2 + dy**2).astype(np.float32)
    slope[~valid_mask] = np.nan
    slope = np.where(np.isfinite(slope) & (slope > 1e-6), slope, 1e-3).astype(np.float32)

    flow = np.asarray(flow_accum, dtype=np.float32)
    flow = np.where(np.isfinite(flow), flow, np.nan)
    flow = np.where(np.isnan(flow), pixel_size, flow)
    flow = np.maximum(flow, pixel_size).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        twi = np.log(flow / slope).astype(np.float32)

    twi[~valid_mask] = np.nan
    twi[~np.isfinite(twi)] = np.nan
    return twi


def _resample_aspect_circular(
    aspect: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    out_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Resample aspect by projecting sin/cos components instead of raw degrees so
    north-facing values do not smear across the 0/360 wrap boundary.
    """
    sin_src = np.where(np.isfinite(aspect), np.sin(np.deg2rad(aspect)), np.nan).astype(np.float32)
    cos_src = np.where(np.isfinite(aspect), np.cos(np.deg2rad(aspect)), np.nan).astype(np.float32)
    sin_dst = np.full(out_shape, np.nan, dtype=np.float32)
    cos_dst = np.full(out_shape, np.nan, dtype=np.float32)

    reproject(
        sin_src,
        sin_dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    reproject(
        cos_src,
        cos_dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    magnitude = np.hypot(sin_dst, cos_dst).astype(np.float32)
    valid = np.isfinite(magnitude) & (magnitude > 1e-6)
    aspect_dst = np.full(out_shape, np.nan, dtype=np.float32)
    if np.any(valid):
        sin_norm = sin_dst[valid] / magnitude[valid]
        cos_norm = cos_dst[valid] / magnitude[valid]
        aspect_dst[valid] = np.mod(np.degrees(np.arctan2(sin_norm, cos_norm)), 360.0).astype(np.float32)
    return aspect_dst


def process_single_tile_potential(args_tuple) -> Optional[Dict]:
    """
    Worker to process a single DTM tile window for abiotic metrics.
    """
    (
        vrt_path,
        window,
        tile_id,
        output_dir,
        pixel_size,
        output_resolution,
    ) = args_tuple

    try:
        if rd is None:
            raise RuntimeError(
                "richdem is required for DTM terrain metrics. "
                "Install via conda-forge: conda install -c conda-forge richdem"
            )

        vrt = _get_worker_vrt(vrt_path)
        dem = vrt.read(1, window=window).astype(np.float32)
        if vrt.nodata is not None:
            if np.isnan(vrt.nodata):
                dem[~np.isfinite(dem)] = np.nan
            else:
                dem[dem == vrt.nodata] = np.nan
        dem[~np.isfinite(dem)] = np.nan
        tile_transform = rasterio.windows.transform(window, vrt.transform)
        crs = vrt.crs

        if not np.any(~np.isnan(dem)):
            return {
                "tile_id": tile_id,
                "skipped": True,
                "reason": "empty_or_nodata_window",
            }

        # Elevation
        elev = dem

        # Provide geotransform-like metadata to richdem so it knows cell size
        px = float(abs(tile_transform.a))
        py = float(abs(tile_transform.e))
        with suppress_output():
            rd_dem = rd.rdarray(
                dem.copy(),
                no_data=np.nan,
                xres=px,
                yres=py,
                xllcorner=0.0,
                yllcorner=0.0,
            )

            # Slope / aspect via richdem
            slope = rd.TerrainAttribute(rd_dem, attrib="slope_degrees").astype(np.float32)
            aspect = rd.TerrainAttribute(rd_dem, attrib="aspect").astype(np.float32)

        # CTI/TWI from local hydrology helpers
        channels, flow_accum, _ = extract_channels_flow_and_depressions(dem)
        cti = calculate_topographic_wetness_index(dem, flow_accum, px)
        cti = cti.astype(np.float32)
        if not np.any(np.isfinite(cti)):
            valid_dem = int(np.isfinite(dem).sum())
            valid_flow = int(np.isfinite(flow_accum).sum())
            raise RuntimeError(
                f"CTI is all NaN for tile {tile_id} "
                f"(valid_dem={valid_dem}, valid_flow={valid_flow}). "
                "Check that --dtm-dir contains elevation DTM tiles only."
            )

        # Resample to exact target resolution (e.g. 250 m) using reproject
        actual_resolution = float(abs(tile_transform.a))
        if output_resolution > actual_resolution:
            H_in, W_in = elev.shape
            minx = tile_transform.c
            maxy = tile_transform.f
            maxx = minx + W_in * tile_transform.a
            miny = maxy + H_in * tile_transform.e
            out_w = max(1, int(math.ceil((maxx - minx) / output_resolution)))
            out_h = max(1, int(math.ceil((maxy - miny) / output_resolution)))
            out_transform = Affine(
                output_resolution, 0.0, minx,
                0.0, -output_resolution, maxy,
            )
            resampling = Resampling.bilinear
            elev_out = np.full((out_h, out_w), np.nan, dtype=np.float32)
            slope_out = np.full((out_h, out_w), np.nan, dtype=np.float32)
            cti_out = np.full((out_h, out_w), np.nan, dtype=np.float32)
            reproject(elev, elev_out, src_transform=tile_transform, src_crs=crs, dst_transform=out_transform, dst_crs=crs, resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
            reproject(slope, slope_out, src_transform=tile_transform, src_crs=crs, dst_transform=out_transform, dst_crs=crs, resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
            aspect_out = _resample_aspect_circular(
                aspect,
                tile_transform,
                crs,
                out_transform,
                crs,
                (out_h, out_w),
            )
            reproject(cti, cti_out, src_transform=tile_transform, src_crs=crs, dst_transform=out_transform, dst_crs=crs, resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
            elev, slope, aspect, cti = elev_out, slope_out, aspect_out, cti_out
            tile_transform = out_transform

        # Build profile
        H, W = elev.shape
        profile = {
            "driver": "GTiff",
            "height": H,
            "width": W,
            "count": 4,  # elev, slope, aspect, cti
            "dtype": "float32",
            "crs": crs,
            "transform": tile_transform,
            "nodata": np.nan,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        output_dir = Path(output_dir)
        _ensure_dir(output_dir)
        out_path = output_dir / f"dtm_metrics_tile_{tile_id}.tif"

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(elev, 1)
            dst.set_band_description(1, "elevation_m")
            dst.write(slope, 2)
            dst.set_band_description(2, "slope_deg")
            dst.write(aspect, 3)
            dst.set_band_description(3, "aspect_deg")
            dst.write(cti, 4)
            dst.set_band_description(4, "cti_twi")

            dst.update_tags(
                title="DTM-based abiotic metrics for potential mapping",
                processing_date=datetime.now().isoformat(),
                output_resolution=f"{output_resolution}m",
                tile_id=tile_id,
            )

        return {
            "tile_id": tile_id,
            "output_file": str(out_path),
            "height": H,
            "width": W,
            "output_pixel_size_m": float(abs(tile_transform.a)),
        }

    except Exception as e:
        return {"tile_id": tile_id, "error": str(e)}


def create_mosaic_from_tiles(tile_results: List[Dict], output_dir: Path) -> Optional[Path]:
    """
    Mosaic all per-tile DTM metrics into a single multiband raster.
    """
    try:
        output_dir = Path(output_dir)
        _ensure_dir(output_dir)

        ordered_results = sorted(tile_results, key=lambda item: item.get("tile_id", ""))
        tile_files = [Path(r["output_file"]) for r in ordered_results if "output_file" in r]
        if not tile_files:
            return None

        srcs = [rasterio.open(p) for p in tile_files]
        try:
            mosaic, out_transform = merge(srcs)
            meta = srcs[0].meta.copy()
        finally:
            for s in srcs:
                s.close()

        meta.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            count=mosaic.shape[0],
            dtype="float32",
            compress="lzw",
            tiled=True,
            blockxsize=1024,
            blockysize=1024,
            BIGTIFF="YES",
        )

        # Write mosaic directly into the chosen output directory
        mosaic_path = output_dir / "dtm_metrics.tif"
        with rasterio.open(mosaic_path, "w", **meta) as dst:
            dst.write(mosaic)
            dst.set_band_description(1, "elevation_m")
            dst.set_band_description(2, "slope_deg")
            dst.set_band_description(3, "aspect_deg")
            dst.set_band_description(4, "cti_twi")
            dst.update_tags(
                title="DTM metrics mosaic for wet woodland potential mapping",
                processing_date=datetime.now().isoformat(),
            )

        return mosaic_path
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    defaults = _default_dtm_stage_paths()
    p = argparse.ArgumentParser(
        description="Build the DTM metrics raster (elevation, slope, aspect, CTI)."
    )

    p.add_argument(
        "--dtm-dir",
        default=str(defaults["dtm_dir"]),
        help=f"Directory containing DTM tiles (default: {defaults['dtm_dir']})",
    )
    p.add_argument(
        "--outdir",
        default=str(defaults["outdir"]),
        help=(
            "Output directory for per-tile metrics and optional mosaic. "
            f"Default: {defaults['outdir']} (mosaic will be written here as dtm_metrics.tif)."
        ),
    )
    p.add_argument(
        "--coverage-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a preflight DTM VRT coverage diagnostic inside a mask polygon (default: enabled).",
    )
    p.add_argument(
        "--preview-resolution",
        type=float,
        default=None,
        help=(
            "Optional quick-look elevation preview resolution in meters, written directly "
            "from the VRT before full tile processing (e.g. 250 or 500). Default: off."
        ),
    )
    p.add_argument(
        "--coverage-mask-shp",
        default=str(defaults["england_shp"]),
        help=(
            "Polygon mask used for VRT extent pruning, tile-grid pruning, "
            "preview clipping, coverage preflight, and final mosaic clipping. "
            f"Default: {defaults['england_shp']} (skipped if missing)"
        ),
    )
    p.add_argument(
        "--clip-mask-buffer-m",
        type=float,
        default=-100.0,
        help=(
            "Buffer in metres applied only when clipping the final mosaic. "
            "Negative shrinks inland, positive expands seaward. Default: -100."
        ),
    )
    p.add_argument(
        "--coverage-check-resolution",
        type=float,
        default=None,
        help=(
            "Resolution in meters for the VRT coverage preflight raster. "
            "Default: auto = max(output_resolution, 500)."
        ),
    )
    p.add_argument(
        "--coverage-hole-threshold-ha",
        type=float,
        default=250.0,
        help="Report missing-data holes at or above this area in hectares. Default: 250",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=8192,
        help="Tile size in pixels for processing (default: 8192).",
    )
    p.add_argument(
        "--buffer",
        type=int,
        default=512,
        help="Buffer pixels around each tile for flow routing edge effects (default: 512). "
             "CTI requires upstream context — too small a buffer corrupts accumulation at tile edges.",
    )
    p.add_argument(
        "--pixel-size",
        type=float,
        default=10.0,
        help="DTM pixel size in meters (native resolution; default: 10).",
    )
    p.add_argument(
        "--output-resolution",
        type=float,
        default=100.0,
        help="Output resolution in meters (e.g. 100 for 100 m grid; default: 100).",
    )
    p.add_argument(
        "--workers",
        "--cores",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    p.add_argument(
        "--pattern",
        default="*.tif",
        help="File pattern for DTM tiles (default: *.tif).",
    )
    p.add_argument(
        "--mosaic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a mosaic GeoTIFF from per-tile outputs (default: enabled).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    dtm_dir = Path(args.dtm_dir).expanduser()
    outdir = Path(args.outdir).expanduser()
    _ensure_dir(outdir)

    if not dtm_dir.exists():
        print(f"❌ DTM directory not found: {dtm_dir}")
        return 1

    print("🧭 Tiled DTM preprocessing for potential mapping")
    print("=" * 60)
    print(f"📁 DTM tiles: {dtm_dir}")
    print(f"📂 Output dir: {outdir}")
    print(f"🧱 Mosaic path: {outdir / 'dtm_metrics.tif'}")
    print(f"🔲 Tile size: {args.tile_size}px")
    print(f"📏 Buffer: {args.buffer}px")
    print(f"📤 Target output resolution: {args.output_resolution} m")
    print(f"⚡ Workers: {args.workers}")
    print(f"🖼️  Preview: {f'{args.preview_resolution} m' if args.preview_resolution else 'off'}")
    print(f"🔎 Coverage check: {'on' if args.coverage_check else 'off'}")
    print(
        "✂️  Mosaic clip mask: "
        f"{args.coverage_mask_shp if args.coverage_mask_shp else 'off'}"
    )
    print(f"↔️  Mosaic clip buffer: {args.clip_mask_buffer_m:g} m")

    # 1) Find DTM tiles
    dtm_files = find_dtm_tiles(dtm_dir, args.pattern)
    if not dtm_files:
        print(f"❌ No DTM tiles found in {dtm_dir} with pattern {args.pattern}")
        return 1
    print(f"📄 Found {len(dtm_files)} DTM tiles")
    non_dtm_like = [p for p in dtm_files if "dtm" not in p.stem.lower()]
    if len(non_dtm_like) == len(dtm_files):
        sample_names = ", ".join(p.name for p in dtm_files[:5])
        print("⚠️  None of the matched files look DTM-like by filename.")
        print(f"   Sample matches: {sample_names}")
        print("   If this is unintended, set a stricter --dtm-dir or --pattern (e.g. 'DTM*.tif').")
    valid_dtm_files, rejected_dtm_files = validate_dtm_inputs(dtm_files)
    print(
        f"🧪 Source tile preflight: {len(valid_dtm_files)} valid, "
        f"{len(rejected_dtm_files)} rejected"
    )
    if rejected_dtm_files:
        print("❌ Source tiles rejected before VRT creation:")
        for item in rejected_dtm_files:
            print(f"   {item['path']}: {item['reason']}")
        print("Refusing to build a VRT when one or more source tiles would be skipped.")
        return 1
    dtm_files = valid_dtm_files

    coverage_mask_path = Path(args.coverage_mask_shp).expanduser() if args.coverage_mask_shp else None
    mask_bounds_for_vrt: Optional[rasterio.coords.BoundingBox] = None
    tile_mask_bounds: Optional[rasterio.coords.BoundingBox] = None
    prepared_tile_mask = None
    if coverage_mask_path is not None and coverage_mask_path.is_file():
        try:
            with rasterio.open(dtm_files[0]) as ref:
                source_pixel_size = max(abs(ref.transform.a), abs(ref.transform.e))
                tile_mask_bounds, prepared_tile_mask = _prepare_mask_filter(
                    coverage_mask_path,
                    ref.crs,
                )
            processing_margin_m = float(args.buffer) * float(source_pixel_size)
            mask_bounds_for_vrt = _expand_bounds(tile_mask_bounds, processing_margin_m)
            print(
                "🧭 Processing extent limited to mask bounds "
                f"plus {processing_margin_m:.1f} m buffer context"
            )
        except Exception as exc:
            print(f"⚠️  Mask-guided VRT/tile pruning disabled: {exc}")
            mask_bounds_for_vrt = None
            tile_mask_bounds = None
            prepared_tile_mask = None

    # 2) Build VRT over all tiles
    vrt_path = outdir / "temp_dtm.vrt"
    create_virtual_raster(dtm_files, vrt_path, mask_bounds=mask_bounds_for_vrt)
    preview_summary: Optional[Dict[str, object]] = None
    if args.preview_resolution:
        preview_mask_path = None
        if coverage_mask_path is not None and coverage_mask_path.is_file():
            preview_mask_path = coverage_mask_path
        try:
            preview_summary = write_vrt_elevation_preview(
                vrt_path,
                outdir,
                resolution_m=float(args.preview_resolution),
                mask_path=preview_mask_path,
            )
        except Exception as exc:
            print(f"⚠️  Preview export failed: {exc}")
    coverage_summary: Optional[Dict[str, object]] = None
    if args.coverage_check and args.coverage_mask_shp:
        if coverage_mask_path.is_file():
            coverage_resolution = (
                float(args.coverage_check_resolution)
                if args.coverage_check_resolution
                else max(float(args.output_resolution), 500.0)
            )
            try:
                coverage_summary = check_vrt_coverage(
                    vrt_path,
                    coverage_mask_path,
                    outdir,
                    resolution_m=coverage_resolution,
                    min_hole_area_ha=float(args.coverage_hole_threshold_ha),
                )
            except Exception as exc:
                print(f"⚠️  Coverage check failed: {exc}")
        else:
            print(f"⚠️  Coverage check skipped: mask not found at {coverage_mask_path}")

    # 3) Define tile windows over the VRT
    with rasterio.open(vrt_path) as vrt:
        print(f"🗺️  VRT dimensions: {vrt.width} x {vrt.height}")
        print(f"🗺️  VRT CRS: {vrt.crs}")
        print(f"🗺️  VRT pixel size: {abs(vrt.transform[0]):.1f} x {abs(vrt.transform[4]):.1f} m")

        n_tiles_x = (vrt.width + args.tile_size - 1) // args.tile_size
        n_tiles_y = (vrt.height + args.tile_size - 1) // args.tile_size
        total_tiles = n_tiles_x * n_tiles_y
        print(f"📐 Tile grid: {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles")

        tile_args = []
        masked_out_tiles = 0
        for row in range(n_tiles_y):
            for col in range(n_tiles_x):
                _, core_bounds, window, _ = get_tile_bounds_with_buffer(
                    vrt, args.tile_size, args.buffer, row, col
                )
                if (
                    tile_mask_bounds is not None
                    and not _tile_intersects_mask(core_bounds, tile_mask_bounds, prepared_tile_mask)
                ):
                    masked_out_tiles += 1
                    continue
                tile_id = f"{row:04d}_{col:04d}"
                tile_args.append(
                    (
                        str(vrt_path),
                        window,
                        tile_id,
                        str(outdir),
                        float(args.pixel_size),
                        float(args.output_resolution),
                    )
                )
        if tile_mask_bounds is not None:
            kept_tiles = len(tile_args)
            print(
                f"🧭 Tile pruning by mask: kept {kept_tiles} tiles, "
                f"skipped {masked_out_tiles} outside-mask tiles"
            )

    # 4) Process tiles in parallel
    print(f"\n🚀 Processing {len(tile_args)} tiles with {args.workers} workers...")
    results: List[Dict] = []
    skipped_tiles: List[Dict] = []
    failed: List[Dict] = []
    fatal_error: Optional[Dict] = None

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker_vrt,
        initargs=(str(vrt_path),),
    ) as ex:
        futures = {ex.submit(process_single_tile_potential, ta): ta[2] for ta in tile_args}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Tiles"):
            tile_id = futures[fut]
            try:
                res = fut.result()
                if res is None:
                    skipped_tiles.append({"tile_id": tile_id, "reason": "worker_returned_none"})
                    continue
                if "error" in res:
                    failed.append(res)
                    fatal_error = res
                    print(f"\n❌ Tile {tile_id} failed: {res['error']}")
                    print("Aborting remaining tiles due to first error.")
                    break
                elif res.get("skipped"):
                    skipped_tiles.append(res)
                else:
                    results.append(res)
                    if len(results) == 1 and "output_pixel_size_m" in res:
                        print(f"\n✅ First tile done → output pixel size: {res['output_pixel_size_m']:.1f} m")
            except Exception as e:
                err = {"tile_id": tile_id, "error": str(e)}
                failed.append(err)
                fatal_error = err
                print(f"\n❌ Tile {tile_id} raised exception: {e}")
                print("Aborting remaining tiles due to first error.")
                break

    # Clean up VRT
    try:
        vrt_path.unlink()
    except Exception:
        pass

    # 5) If we bailed early on a fatal error, stop now
    if fatal_error is not None and not results:
        print("\n❌ FATAL: Aborted due to first tile error.")
        print(f"   Tile {fatal_error['tile_id']}: {fatal_error['error']}")
        return 1

    # 6) Summary
    print("\n📊 Summary")
    print(f"   Total tiles: {len(tile_args)}")
    print(f"   Successful tiles: {len(results)}")
    print(f"   Skipped tiles: {len(skipped_tiles)}")
    print(f"   Failed tiles: {len(failed)}")
    if skipped_tiles:
        print("   Example skipped tiles:")
        for item in skipped_tiles[:10]:
            print(f"     Tile {item['tile_id']}: {item.get('reason', 'unknown')}")
    if failed:
        print("   Example errors:")
        for f in failed[:5]:
            print(f"     Tile {f['tile_id']}: {f['error']}")

    # Save a JSON manifest
    manifest = {
        "processing_date": datetime.now().isoformat(),
        "dtm_dir": str(dtm_dir),
        "outdir": str(outdir),
        "tile_size": args.tile_size,
        "buffer": args.buffer,
        "pixel_size": args.pixel_size,
        "output_resolution": args.output_resolution,
        "successful_tiles": len(results),
        "skipped_tiles": len(skipped_tiles),
        "failed_tiles": len(failed),
        "preview": preview_summary,
        "coverage_check": coverage_summary,
        "source_tile_rejections": rejected_dtm_files,
        "skipped_tile_details": skipped_tiles,
        "tiles": results,
    }
    manifest_path = outdir / "dtm_metrics_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 Manifest: {manifest_path}")

    if fatal_error is not None:
        print("\n❌ FATAL: Aborted after a tile failure; refusing to mosaic a partial tile set.")
        print(f"   Tile {fatal_error['tile_id']}: {fatal_error['error']}")
        return 1

    # 6) Mosaic, if requested
    if args.mosaic and results:
        print("\n🧩 Creating mosaic...")
        mosaic_path = create_mosaic_from_tiles(results, outdir)
        if mosaic_path:
            print(f"✅ Mosaic written to {mosaic_path}")
            if args.coverage_mask_shp:
                clip_mask_path = Path(args.coverage_mask_shp).expanduser()
                if clip_mask_path.is_file():
                    try:
                        clipped = apply_polygon_mask_to_raster(
                            mosaic_path,
                            clip_mask_path,
                            buffer_m=float(args.clip_mask_buffer_m),
                        )
                        if clipped:
                            print(
                                f"✂️  Clipped mosaic to {clip_mask_path} "
                                f"(buffer {args.clip_mask_buffer_m:g} m)"
                            )
                    except Exception as exc:
                        print(f"⚠️  Mosaic clip failed: {exc}")
                else:
                    print(f"⚠️  Mosaic clip skipped: mask not found at {clip_mask_path}")
            # Optionally clean up per-tile rasters to avoid clutter/disk use
            print("🧹 Removing per-tile DTM metric files...")
            for r in results:
                tile_path = Path(r.get("output_file", ""))
                if tile_path.is_file():
                    try:
                        tile_path.unlink()
                    except Exception:
                        pass
        else:
            print("⚠️  Mosaic creation failed or returned no output")

    print("\n✅ Finished tiled DTM preprocessing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
