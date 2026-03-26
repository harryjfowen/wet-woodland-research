#!/usr/bin/env python3
"""
Wet Woodland Statistics Pipeline

This script computes spatial statistics for wet woodland data:
 1) Total area (ha), split on-peat vs off-peat
 2) Contiguous patches and patch areas (ha)
 3) Patch size distribution, including TOW categories (Lone Tree, Group of Trees, Small Woodland)
 4) Nearest-neighbour distances between patches (m)
 5) Density per km² and per 10 km cells
 6) Aggregation by LNRS region polygons
 7) Distance to rivers/lakes (m) for wet woodland pixels
 8) Elevation and floodplain summaries
 9) Export summary tables and rasters

Input format: Multi-band GeoTIFF (band 1=binary 0/1, band 2=probabilities 0-1, 255=nodata)
- Without --threshold: Uses band 1 binary classifications
- With --threshold: Uses band 2 probabilities with threshold applied
Urban areas can be optionally masked out.
The script is designed to be robust to missing optional inputs (it will skip those metrics).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import glob
import tempfile
import warnings
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.transform import rowcol
from rasterio.warp import calculate_default_transform, reproject
from rasterio.merge import merge as rio_merge
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, box


# ----------------------
# Data classes
# ----------------------

@dataclass
class RasterData:
    array: np.ndarray  # 2D
    transform: Affine
    crs: str
    nodata: Optional[float]
    dtype: str
    valid_mask: Optional[np.ndarray] = None

    @property
    def pixel_width(self) -> float:
        return abs(self.transform.a)

    @property
    def pixel_height(self) -> float:
        return abs(self.transform.e)

    @property
    def pixel_area_m2(self) -> float:
        return self.pixel_width * self.pixel_height

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape


# ----------------------
# Utilities
# ----------------------

def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _read_binary_raster(path: str, threshold: Optional[float] = None) -> RasterData:
    with rasterio.open(path) as ds:
        # Choose which band to use.
        if threshold is not None:
            # If 2 bands exist, use band 2 as probability; otherwise use band 1.
            prob_band = 2 if ds.count >= 2 else 1
            print(f"  Using band {prob_band} (probabilities) with threshold {threshold}")
            arr = ds.read(prob_band, masked=False).astype(np.float32)
            wet_mask = arr >= float(threshold)
        else:
            print("  Using band 1 (binary classifications)")
            arr = ds.read(1, masked=False)
            # Binarize:
            # - if integer, treat exactly 1 as wet; 0 as non-wet
            # - if float, treat >0.5 as wet
            if np.issubdtype(arr.dtype, np.integer):
                wet_mask = (arr == 1)
            else:
                wet_mask = (arr > 0.5)

        # Determine valid pixels (exclude nodata); default nodata=255 for uint8 if missing
        nodata_value = ds.nodata
        if nodata_value is None and arr.dtype == np.uint8:
            nodata_value = 255
        if nodata_value is not None:
            valid_mask = (arr != nodata_value)
        else:
            valid_mask = np.ones_like(arr, dtype=bool)

        binary_unmasked = np.where(valid_mask & wet_mask, 1, 0).astype(np.uint8)
        return RasterData(
            array=binary_unmasked,
            transform=ds.transform,
            crs=str(ds.crs) if ds.crs is not None else "",
            nodata=nodata_value,
            dtype=str(binary_unmasked.dtype),
            valid_mask=valid_mask.astype(bool),
        )


def _read_raster_float(path: str) -> RasterData:
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=False).astype(np.float32)
        ndv = ds.nodata
        valid_mask = None
        if ndv is not None:
            valid_mask = (arr != ndv)
            arr = np.where(valid_mask, arr, np.float32(np.nan))
        return RasterData(
            array=arr,
            transform=ds.transform,
            crs=str(ds.crs) if ds.crs is not None else "",
            nodata=ds.nodata,
            dtype=str(arr.dtype),
            valid_mask=valid_mask if valid_mask is not None else None,
        )


def _reproject_match(src: RasterData, dst_profile: dict, resampling=Resampling.nearest) -> RasterData:
    # Use appropriate nodata value (avoid 0.0 which could be valid data like sea level)
    dst_nodata = -9999.0 if src.nodata is None else src.nodata

    dst_arr = np.full((dst_profile["height"], dst_profile["width"]), dst_nodata, dtype=np.float32)
    reproject(
        source=src.array.astype(np.float32),
        destination=dst_arr,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        resampling=resampling,
        src_nodata=src.nodata,
        dst_nodata=dst_nodata,
    )
    dst_arr = np.nan_to_num(dst_arr, nan=dst_nodata)
    # Binary if we originated from a binary; otherwise leave float
    # Heuristic: if values are 0/1, keep as uint8
    valid_mask = (dst_arr != dst_nodata)
    if valid_mask.any():
        unique_vals = np.unique(dst_arr[valid_mask])
    if unique_vals.size <= 3 and set(np.round(unique_vals).tolist()).issubset({0, 1}):
            # Convert to uint8 binary, but preserve nodata (use 255 to avoid conflict with 0/1 data)
            binary_arr = np.full_like(dst_arr, 255, dtype=np.uint8)
            binary_arr[valid_mask] = (dst_arr[valid_mask] > 0.5).astype(np.uint8)
            binary_arr[~valid_mask] = 255  # nodata = 255 for uint8 (consistent with predictions)
            return RasterData(
                array=binary_arr,
                transform=dst_profile["transform"],
                crs=str(dst_profile["crs"]),
                nodata=255,  # 255 as nodata for uint8 (0 and 1 are valid data)
                dtype="uint8",
            )
    # Keep as float32
    return RasterData(
        array=dst_arr,
        transform=dst_profile["transform"],
        crs=str(dst_profile["crs"]),
        nodata=dst_nodata,
        dtype="float32",
    )


def _vector_to_raster_mask(
    gdf: gpd.GeoDataFrame, like: RasterData, burn_value: int = 1
) -> np.ndarray:
    if gdf.empty:
        return np.zeros(like.shape, dtype=np.uint8)
    gdf = gdf.to_crs(like.crs)
    mask = rasterize(
        ((geom, burn_value) for geom in gdf.geometry if geom is not None),
        out_shape=like.shape,
        transform=like.transform,
        fill=0,
        all_touched=False,
        dtype=np.uint8,
    )
    return mask


def _load_vector(path: str, target_crs: Optional[str]) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(path)
    except UnicodeDecodeError:
        warnings.warn(
            f"Vector file {path} has non-UTF-8 text fields. Falling back to geometry-only read."
        )
        gdf = gpd.read_file(path, columns=[])
    if gdf.crs is None:
        warnings.warn(f"Vector file {path} has no CRS defined. Assuming it matches target CRS.")
    if target_crs:
        if gdf.crs is not None and str(gdf.crs) != str(target_crs):
            print(f"  Reprojecting {os.path.basename(path)}: {gdf.crs} → {target_crs}")
        gdf = gdf.to_crs(target_crs)
    return gdf


def _write_geotiff(
    path: str, array: np.ndarray, transform: Affine, crs: str, nodata: Optional[float] = None, dtype: Optional[str] = None
) -> None:
    if dtype is None:
        dtype = str(array.dtype)
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "BIGTIFF": "YES",  # Enable BigTIFF for large files (>4GB)
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def _collect_tiles(tiles_dir: str, tiles_glob: str) -> List[str]:
    pattern = os.path.join(tiles_dir, tiles_glob)
    paths = sorted(glob.glob(pattern))
    # Filter out .aux.xml sidecars if pattern picks them up
    paths = [p for p in paths if not p.lower().endswith(".aux.xml")]
    return paths


def _try_build_vrt(tile_paths: List[str], vrt_path: str) -> bool:
    try:
        from osgeo import gdal  # type: ignore
        gdal.UseExceptions()
    except ImportError:
        # GDAL Python bindings not installed - silently skip, will try CLI
        return False
    except Exception as e:
        print(f"Warning: GDAL import error: {e}")
        return False
    try:
        # Overwrite if exists
        if os.path.exists(vrt_path):
            try:
                os.remove(vrt_path)
            except Exception:
                pass
        # BuildVRT options with explicit nodata (255) so background never counts
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False, srcNodata=255, VRTNodata=255)
        vrt = gdal.BuildVRT(vrt_path, tile_paths, options=vrt_options)
        if vrt is None:
            print(f"Warning: gdal.BuildVRT returned None for {vrt_path}")
            return False
        vrt.FlushCache()
        vrt = None
        success = os.path.exists(vrt_path)
        if not success:
            print(f"Warning: VRT file was not created at {vrt_path}")
        return success
    except Exception as e:
        print(f"Warning: Failed to build VRT with Python GDAL: {e}")
        return False


def _merge_tiles_to_tiff(tile_paths: List[str], out_tif: str) -> str:
    if len(tile_paths) == 0:
        raise ValueError("No tile rasters found to merge.")
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        # Use nodata=255 to align with classification scheme
        mosaic, transform = rio_merge(datasets, nodata=255)
        crs = datasets[0].crs
        profile = {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "count": mosaic.shape[0],
            "dtype": str(mosaic.dtype),
            "crs": str(crs),
            "transform": transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "BIGTIFF": "YES",
            "nodata": 255,
        }
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(mosaic)
        return out_tif
    finally:
        for ds in datasets:
            ds.close()


def _try_build_vrt_cli(tile_paths: List[str], vrt_path: str) -> bool:
    """
    Attempt to build a VRT using the gdalbuildvrt command-line tool.
    Returns True on success.
    """
    try:
        # Overwrite if exists
        if os.path.exists(vrt_path):
            try:
                os.remove(vrt_path)
            except Exception:
                pass
        cmd = ["gdalbuildvrt", "-overwrite", "-srcnodata", "255", "-vrtnodata", "255", vrt_path] + tile_paths
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode == 0 and os.path.exists(vrt_path):
            print(f"Successfully created VRT using gdalbuildvrt CLI: {vrt_path}")
            return True
        else:
            print(f"Warning: gdalbuildvrt CLI failed with return code {proc.returncode}")
            if proc.stderr:
                print(f"  stderr: {proc.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("Warning: gdalbuildvrt command not found in PATH")
        return False
    except Exception as e:
        print(f"Warning: Failed to build VRT with gdalbuildvrt CLI: {e}")
        return False


def _compute_pixel_centroids(rows: np.ndarray, cols: np.ndarray, transform: Affine) -> Tuple[np.ndarray, np.ndarray]:
    # Convert row/col indices to x,y at pixel centers
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return np.asarray(xs), np.asarray(ys)


def _label_patches(binary: np.ndarray, connectivity: int = 2) -> Tuple[np.ndarray, int]:
    # connectivity=2 => 8-connected; connectivity=1 => 4-connected
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 2 else 1)
    labeled, num = ndimage.label(binary.astype(bool), structure=structure)
    return labeled, num


def make_connectivity_structure(connectivity: int):
    return ndimage.generate_binary_structure(2, 2 if connectivity == 2 else 1)


def label_with_bridge(binary_mask: np.ndarray, connectivity: int, bridge_pixels: int) -> np.ndarray:
    """
    Label connected components with optional bridging via morphological dilation.
    - If bridge_pixels > 0, dilate mask for N iterations using connectivity struct,
      label the dilated mask, then mask labels back to original wet pixels
      so areas are not inflated.
    - If bridge_pixels == 0, label the original mask directly.
    Returns label array aligned to original grid (0 for background).
    """
    structure = make_connectivity_structure(connectivity)
    mask = binary_mask.astype(bool)
    if bridge_pixels and bridge_pixels > 0:
        dilated = ndimage.binary_dilation(mask, structure=structure, iterations=int(bridge_pixels))
        labeled_dilated, _ = ndimage.label(dilated, structure=structure)
        labels = labeled_dilated.astype(np.int32)
        labels[~mask] = 0
        return labels
    labeled, _ = ndimage.label(mask, structure=structure)
    return labeled.astype(np.int32)


def _block_reduce_sum(arr: np.ndarray, factor_y: int, factor_x: int) -> np.ndarray:
    """
    Reduce a 2D array by summing non-overlapping blocks of size factor_y x factor_x.
    The output shape is floor(H/factor_y) x floor(W/factor_x).
    """
    H, W = arr.shape
    H2 = (H // factor_y) * factor_y
    W2 = (W // factor_x) * factor_x
    if H2 == 0 or W2 == 0:
        return np.zeros((0, 0), dtype=arr.dtype)
    arr_c = arr[:H2, :W2]
    reshaped = arr_c.reshape(H2 // factor_y, factor_y, W2 // factor_x, factor_x)
    return reshaped.sum(axis=(1, 3))


def _grid_transform_from_block_reduce(base_transform: Affine, factor_y: int, factor_x: int) -> Affine:
    return Affine(
        base_transform.a * factor_x,
        base_transform.b,
        base_transform.c,
        base_transform.d,
        base_transform.e * factor_y,
        base_transform.f,
    )


def _modal_value(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    unique, counts = np.unique(values, return_counts=True)
    idx = np.argmax(counts)
    return float(unique[idx])


def _write_summary_report(
    path: str,
    summary_dict: Dict[str, float],
    df_patches: pd.DataFrame,
    df_lnrs: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    """
    Write a comprehensive, human-readable text report suitable for scientific papers.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("WET WOODLAND LANDSCAPE STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Study area info
        f.write("STUDY AREA INFORMATION\n")
        f.write("-" * 80 + "\n")
        pixel_area = summary_dict.get("pixel_area_m2", 0)
        f.write(f"Raster Resolution: {math.sqrt(pixel_area):.1f} m\n")
        if hasattr(args, 'connectivity'):
            conn_type = "8-connected" if args.connectivity == 2 else "4-connected"
            f.write(f"Patch Connectivity: {conn_type}\n")
        f.write("\n")

        # Total area
        f.write("TOTAL WET WOODLAND AREA\n")
        f.write("-" * 80 + "\n")
        total_ha = summary_dict.get("total_wet_woodland_ha", 0)
        f.write(f"Total Wet Woodland: {total_ha:,.2f} ha ({total_ha/100:,.2f} km²)\n")
        # Global percentages
        if "study_area_ha_footprint" in summary_dict and "wet_pct_of_footprint" in summary_dict:
            footprint_ha = summary_dict.get("study_area_ha_footprint", 0)
            wet_pct_fp = summary_dict.get("wet_pct_of_footprint", float("nan"))
            f.write(f"Study Area (raster footprint): {footprint_ha:,.2f} ha\n")
            f.write(f"Wet Woodland as % of footprint: {wet_pct_fp:.4f}%\n")
        if ("lnrs_union_area_ha" in summary_dict) and ("wet_area_lnrs_ha" in summary_dict) and ("wet_pct_of_lnrs_union" in summary_dict):
            lnrs_ha = summary_dict.get("lnrs_union_area_ha", 0)
            wet_lnrs = summary_dict.get("wet_area_lnrs_ha", 0)
            wet_pct_lnrs = summary_dict.get("wet_pct_of_lnrs_union", float("nan"))
            f.write(f"LNRS Union Area: {lnrs_ha:,.2f} ha\n")
            f.write(f"Wet Woodland within LNRS: {wet_lnrs:,.2f} ha ({wet_pct_lnrs:.4f}%)\n")
        if ("england_land_area_km2" in summary_dict) and ("wet_pct_of_england_land" in summary_dict):
            eng_km2 = summary_dict.get("england_land_area_km2", 0.0)
            wet_pct_eng = summary_dict.get("wet_pct_of_england_land", float("nan"))
            f.write(f"England Land Area (fixed): {eng_km2:,.0f} km²\n")
            f.write(f"Wet Woodland as % of England: {wet_pct_eng:.4f}%\n")
        # Area estimates block
        f.write("\nArea estimates:\n")
        f.write(f"  Wet woodland: {total_ha:,.1f} ha ({total_ha/100:,.1f} km²)\n")
        if "forest_total_area_ha" in summary_dict:
            forest_ha = summary_dict.get("forest_total_area_ha", float("nan"))
            f.write(f"  Total forest (wet + non-wet): {forest_ha:,.1f} ha ({forest_ha/100:,.1f} km²)\n")
        if "wet_pct_of_forest" in summary_dict:
            f.write(f"  Wet woodland as % of forest: {summary_dict.get('wet_pct_of_forest', float('nan')):.4f}%\n")
        if "on_peat_ha" in summary_dict:
            on_peat = summary_dict.get("on_peat_ha", 0)
            off_peat = summary_dict.get("off_peat_ha", 0)
            on_peat_pct = (on_peat / total_ha * 100) if total_ha > 0 else 0
            off_peat_pct = (off_peat / total_ha * 100) if total_ha > 0 else 0
            f.write(f"  On Peat: {on_peat:,.2f} ha ({on_peat_pct:.4f}%)\n")
            f.write(f"  Off Peat: {off_peat:,.2f} ha ({off_peat_pct:.4f}%)\n")
        f.write("\n")

        # Patch statistics
        f.write("PATCH STATISTICS\n")
        f.write("-" * 80 + "\n")
        num_patches = int(summary_dict.get("num_patches", 0))
        f.write(f"Total Number of Patches: {num_patches:,}\n")
        f.write(f"Mean Patch Size: {summary_dict.get('mean_patch_size_ha', 0):.3f} ha\n")
        f.write(f"Median Patch Size: {summary_dict.get('median_patch_size_ha', 0):.3f} ha\n")
        f.write(f"Modal Patch Size: {summary_dict.get('modal_patch_size_ha', 0):.3f} ha\n")
        f.write("\n")

        # Patch size distribution
        f.write("Patch Size Distribution:\n")
        lt0_01 = int(summary_dict.get("patches_lt_0_01ha", 0))
        bet0_01_0_1 = int(summary_dict.get("patches_0_01_to_0_1ha", 0))
        bet0_1_1 = int(summary_dict.get("patches_0_1_to_1ha", 0))
        bet1_5 = int(summary_dict.get("patches_1_to_5ha", 0))
        bet5_10 = int(summary_dict.get("patches_5_to_10ha", 0))
        gte10 = int(summary_dict.get("patches_gte_10ha", 0))
        # Areas per size band
        area_lt0_01 = summary_dict.get("area_lt_0_01ha", 0)
        area_0_01_0_1 = summary_dict.get("area_0_01_to_0_1ha", 0)
        area_0_1_1 = summary_dict.get("area_0_1_to_1ha", 0)
        area_1_5 = summary_dict.get("area_1_to_5ha", 0)
        area_5_10 = summary_dict.get("area_5_to_10ha", 0)
        area_gte10 = summary_dict.get("area_gte_10ha", 0)
        # Area percentages
        total_area_ha = area_lt0_01 + area_0_01_0_1 + area_0_1_1 + area_1_5 + area_5_10 + area_gte10
        area_lt0_01_pct = (area_lt0_01 / total_area_ha * 100) if total_area_ha > 0 else 0
        area_0_01_0_1_pct = (area_0_01_0_1 / total_area_ha * 100) if total_area_ha > 0 else 0
        area_0_1_1_pct = (area_0_1_1 / total_area_ha * 100) if total_area_ha > 0 else 0
        area_1_5_pct = (area_1_5 / total_area_ha * 100) if total_area_ha > 0 else 0
        area_5_10_pct = (area_5_10 / total_area_ha * 100) if total_area_ha > 0 else 0
        area_gte10_pct = (area_gte10 / total_area_ha * 100) if total_area_ha > 0 else 0
        f.write(f"  ≤ 0.01 ha: {area_lt0_01:,.2f} ha ({area_lt0_01_pct:.2f}%) — {lt0_01:,} patches\n")
        f.write(f"  0.01-0.1 ha: {area_0_01_0_1:,.2f} ha ({area_0_01_0_1_pct:.2f}%) — {bet0_01_0_1:,} patches\n")
        f.write(f"  0.1-1 ha: {area_0_1_1:,.2f} ha ({area_0_1_1_pct:.2f}%) — {bet0_1_1:,} patches\n")
        f.write(f"  1-5 ha: {area_1_5:,.2f} ha ({area_1_5_pct:.2f}%) — {bet1_5:,} patches\n")
        f.write(f"  5-10 ha: {area_5_10:,.2f} ha ({area_5_10_pct:.2f}%) — {bet5_10:,} patches\n")
        f.write(f"  ≥ 10 ha: {area_gte10:,.2f} ha ({area_gte10_pct:.2f}%) — {gte10:,} patches\n")
        f.write("\n")

        # TOW categories
        if "num_lone_tree_patches" in summary_dict:
            f.write("TREES OUTSIDE WOODLAND (TOW) CLASSIFICATION\n")
            f.write("-" * 80 + "\n")
            lone_n = int(summary_dict.get("num_lone_tree_patches", 0))
            lone_ha = summary_dict.get("area_ha_lone_tree", 0)
            group_n = int(summary_dict.get("num_group_of_trees_patches", 0))
            group_ha = summary_dict.get("area_ha_group_of_trees", 0)
            small_n = int(summary_dict.get("num_small_woodland_patches", 0))
            small_ha = summary_dict.get("area_ha_small_woodland", 0)

            f.write(f"Lone Trees (5-350 m²):\n")
            f.write(f"  Count: {lone_n:,} patches\n")
            f.write(f"  Area: {lone_ha:,.2f} ha ({(lone_ha/total_ha*100):.4f}% of total)\n")
            f.write(f"\n")
            f.write(f"Group of Trees (350-1,000 m²):\n")
            f.write(f"  Count: {group_n:,} patches\n")
            f.write(f"  Area: {group_ha:,.2f} ha ({(group_ha/total_ha*100):.4f}% of total)\n")
            f.write(f"\n")
            f.write(f"Small Woodlands (> 1,000 m²):\n")
            f.write(f"  Count: {small_n:,} patches\n")
            f.write(f"  Area: {small_ha:,.2f} ha ({(small_ha/total_ha*100):.4f}% of total)\n")
            f.write("\n")

        # Landscape metrics
        f.write("LANDSCAPE FRAGMENTATION METRICS\n")
        f.write("-" * 80 + "\n")
        f.write("NOTE: These metrics are computed across the ENTIRE study area (landscape-scale).\n\n")

        ldi = summary_dict.get("landscape_division_index", float('nan'))
        mesh = summary_dict.get("effective_mesh_size_ha", float('nan'))
        largest_pct = summary_dict.get("largest_patch_pct_of_total", float('nan'))
        top10_pct = summary_dict.get("top10pct_patches_area_pct", float('nan'))

        f.write(f"Landscape Division Index: {ldi:.4f}\n")
        f.write(f"  (0 = single continuous patch, 1 = completely fragmented)\n")
        f.write(f"  Interpretation: {(ldi*100):.1f}% of landscape is 'divided'\n\n")

        f.write(f"Effective Mesh Size: {mesh:,.2f} ha ({mesh/100:,.2f} km²)\n")
        f.write(f"  (Larger = less fragmented; equals mean patch size if equally divided)\n\n")

        f.write(f"Patch Size Concentration:\n")
        f.write(f"  Largest single patch: {largest_pct:.1f}% of total wet woodland area\n")
        f.write(f"  Top 10% of patches: {top10_pct:.1f}% of total wet woodland area\n")

        # Interpretation help
        if not math.isnan(largest_pct):
            if largest_pct > 80:
                f.write(f"\n  ⚠ INTERPRETATION: Landscape dominated by 1 massive patch!\n")
                f.write(f"     Fragmentation metrics may be misleading. Most area is in one patch,\n")
                f.write(f"     but there may be many small isolated patches elsewhere.\n")
            elif largest_pct > 50:
                f.write(f"\n  ⚠ INTERPRETATION: Largest patch contains majority of area.\n")
                f.write(f"     Fragmentation metrics reflect this large patch dominance.\n")
            elif top10_pct > 90:
                f.write(f"\n  INTERPRETATION: Top 10% of patches contain most area.\n")
                f.write(f"     Landscape has few large patches and many tiny ones.\n")
            else:
                f.write(f"\n  INTERPRETATION: Area is distributed across patches.\n")
                f.write(f"     Fragmentation metrics reflect true landscape pattern.\n")
        f.write("\n")

        # Nearest neighbor
        if len(df_patches) > 0 and "nearest_neighbour_m" in df_patches.columns:
            nn_mean = df_patches["nearest_neighbour_m"].mean()
            nn_median = df_patches["nearest_neighbour_m"].median()
            nn_min = df_patches["nearest_neighbour_m"].min()
            nn_max = df_patches["nearest_neighbour_m"].max()
            f.write("Nearest Neighbor Distances:\n")
            f.write(f"  Mean: {nn_mean:.1f} m\n")
            f.write(f"  Median: {nn_median:.1f} m\n")
            f.write(f"  Range: {nn_min:.1f} - {nn_max:.1f} m\n")
            f.write("\n")

        # Proximity index
        if len(df_patches) > 0 and "proximity_index" in df_patches.columns:
            prox_mean = df_patches["proximity_index"].mean()
            prox_median = df_patches["proximity_index"].median()
            f.write("Proximity Index (within 1 km):\n")
            f.write(f"  Mean: {prox_mean:.2f}\n")
            f.write(f"  Median: {prox_median:.2f}\n")
            f.write(f"  (Higher values indicate patches near other large patches)\n")
            f.write("\n")

        # Distance to water
        if "wet_to_water_mean_m" in summary_dict:
            f.write("DISTANCE TO WATER FEATURES\n")
            f.write("-" * 80 + "\n")
            water_px = int(summary_dict.get("water_pixels_rasterized", 0))
            f.write(f"Water Pixels Rasterized: {water_px:,}\n")
            f.write(f"\nDistance from Wet Woodland to Nearest Water Feature:\n")
            f.write(f"  Minimum: {summary_dict.get('wet_to_water_min_m', float('nan')):.1f} m\n")
            f.write(f"  10th Percentile: {summary_dict.get('wet_to_water_p10_m', float('nan')):.1f} m\n")
            f.write(f"  Median: {summary_dict.get('wet_to_water_median_m', float('nan')):.1f} m\n")
            f.write(f"  Mean: {summary_dict.get('wet_to_water_mean_m', float('nan')):.1f} m\n")
            f.write(f"  90th Percentile: {summary_dict.get('wet_to_water_p90_m', float('nan')):.1f} m\n")
            f.write(f"  Maximum: {summary_dict.get('wet_to_water_max_m', float('nan')):.1f} m\n")
            f.write("\n")

        # Elevation
        if "elev_mean_m" in summary_dict:
            f.write("ELEVATION STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Elevation: {summary_dict.get('elev_mean_m', float('nan')):.1f} m\n")
            f.write(f"Median Elevation: {summary_dict.get('elev_median_m', float('nan')):.1f} m\n")
            f.write(f"90th Percentile: {summary_dict.get('elev_p90_m', float('nan')):.1f} m\n")
            f.write("\n")

        # Floodplain
        if "wet_in_floodplain_pct" in summary_dict:
            f.write("FLOODPLAIN ASSOCIATION\n")
            f.write("-" * 80 + "\n")
            fp_pct = summary_dict.get("wet_in_floodplain_pct", float('nan'))
            f.write(f"Wet Woodland within Floodplain: {fp_pct:.4f}%\n")
            if not math.isnan(fp_pct):
                fp_ha = total_ha * fp_pct / 100
                f.write(f"  ({fp_ha:,.2f} ha)\n")
            f.write("\n")

        # Summary for paper
        f.write("=" * 80 + "\n")
        f.write("SUMMARY FOR SCIENTIFIC PAPER\n")
        f.write("=" * 80 + "\n")
        f.write(f"The study area contained {total_ha:,.1f} ha of wet woodland distributed across\n")
        f.write(f"{num_patches:,} discrete patches. The landscape was highly fragmented (Landscape\n")
        f.write(f"Division Index = {ldi:.3f}), with a mean patch size of {summary_dict.get('mean_patch_size_ha', 0):.2f} ha and\n")
        lt0_1_area_combined_pct = area_lt0_01_pct + area_0_01_0_1_pct
        f.write(f"median of {summary_dict.get('median_patch_size_ha', 0):.2f} ha. {lt0_1_area_combined_pct:.2f}% of wet woodland area was in patches smaller than 0.1 ha\n")
        f.write(f"({area_lt0_01_pct:.2f}% ≤ 0.01 ha, {area_0_01_0_1_pct:.2f}% 0.01-0.1 ha), {area_0_1_1_pct:.2f}% in 0.1-1 ha patches, and {area_gte10_pct:.2f}% in patches exceeding 10 ha. ")

        if "on_peat_ha" in summary_dict:
            on_peat_pct = (summary_dict.get("on_peat_ha", 0) / total_ha * 100) if total_ha > 0 else 0
            f.write(f"Approximately {on_peat_pct:.4f}% of wet\n")
            f.write(f"woodland occurred on peat soils. ")

        if "wet_to_water_median_m" in summary_dict:
            median_dist = summary_dict.get("wet_to_water_median_m", float('nan'))
            f.write(f"The median distance to water features\n")
            f.write(f"was {median_dist:.0f} m. ")

        if "wet_in_floodplain_pct" in summary_dict:
            fp_pct = summary_dict.get("wet_in_floodplain_pct", float('nan'))
            if not math.isnan(fp_pct):
                f.write(f"{fp_pct:.4f}% of wet woodland was located within mapped\n")
                f.write(f"floodplains.")

        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")

        # LNRS regional summary (optional)
        if df_lnrs is not None and not df_lnrs.empty:
            f.write("\n")
            f.write("LNRS REGIONAL SUMMARY (WET WOODLAND AREA AND PROPORTIONS)\n")
            f.write("-" * 80 + "\n")
            f.write("For each LNRS region, we report the total wet woodland area (ha) and the\n")
            f.write("proportion of the reference area that is wet woodland. If a column named\n")
            f.write("'forest_area_ha' is present in the LNRS dataset, the proportion is computed\n")
            f.write("relative to that forest area; otherwise it is computed relative to the total\n")
            f.write("LNRS region area.\n\n")
            use_forest = "forest_area_ha" in df_lnrs.columns
            # Prepare and sort by wet area descending for readability
            cols = ["lnrs_id", "wet_area_ha", "region_area_ha"]
            if "NAME" in df_lnrs.columns:
                cols.append("NAME")
            if use_forest:
                cols.append("forest_area_ha")
            df_list = df_lnrs[cols].copy()
            if use_forest:
                df_list["prop_pct"] = np.where(
                    df_list["forest_area_ha"].notna() & (df_list["forest_area_ha"] > 0),
                    (df_list["wet_area_ha"] / df_list["forest_area_ha"]) * 100.0,
                    np.nan,
                )
            else:
                df_list["prop_pct"] = np.where(
                    df_list["region_area_ha"] > 0,
                    (df_list["wet_area_ha"] / df_list["region_area_ha"]) * 100.0,
                    np.nan,
                )
            if "wet_area_ha" in df_list.columns:
                df_list = df_list.sort_values("wet_area_ha", ascending=False, na_position="last")
            # Header
            header = ["LNRS", "Wet_ha", "RefArea_ha", "Prop_%"]
            f.write(f"{header[0]:<40} {header[1]:>15} {header[2]:>15} {header[3]:>10}\n")
            f.write("-" * 85 + "\n")
            for r in df_list.itertuples(index=False):
                name = (
                    r.NAME
                    if ("NAME" in df_list.columns and pd.notna(getattr(r, "NAME", None)) and str(getattr(r, "NAME", "")).strip() != "")
                    else f"LNRS {int(r.lnrs_id)}"
                )
                wet = float(r.wet_area_ha) if hasattr(r, "wet_area_ha") else 0.0
                if use_forest and hasattr(r, "forest_area_ha") and pd.notnull(r.forest_area_ha):
                    ref = float(r.forest_area_ha)
                else:
                    ref = float(r.region_area_ha) if hasattr(r, "region_area_ha") else 0.0
                prop = float(r.prop_pct) if hasattr(r, "prop_pct") and pd.notnull(r.prop_pct) else float("nan")
                f.write(f"{name:<40} {wet:15,.2f} {ref:15,.2f} {prop:10.4f}\n")
            f.write("\n")


# ----------------------
# Core computations
# ----------------------

def compute_area_split_on_off_peat(binary: RasterData, peat_gdf: gpd.GeoDataFrame) -> Dict[str, float]:
    peat_mask = _vector_to_raster_mask(peat_gdf, binary, burn_value=1)
    wet_mask = (binary.array == 1)
    pixel_area_ha = binary.pixel_area_m2 / 10000.0
    on_peat_ha = float((wet_mask & (peat_mask == 1)).sum()) * pixel_area_ha
    off_peat_ha = float((wet_mask & (peat_mask == 0)).sum()) * pixel_area_ha
    total_ha = on_peat_ha + off_peat_ha
    return {
        "total_wet_woodland_ha": total_ha,
        "on_peat_ha": on_peat_ha,
        "off_peat_ha": off_peat_ha,
    }


def compute_patches_and_areas(binary: RasterData, connectivity: int = 2) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Label patches and compute areas and centroids.
    Optimized using ndimage.center_of_mass for centroid calculation.
    """
    labeled, num = _label_patches(binary.array, connectivity=connectivity)
    # Compute areas by label
    labels, counts = np.unique(labeled[labeled > 0], return_counts=True)
    pixel_area_ha = binary.pixel_area_m2 / 10000.0
    areas_ha = counts.astype(np.float64) * pixel_area_ha

    # Use ndimage.center_of_mass for efficient centroid calculation
    centroids = ndimage.center_of_mass(
        input=np.ones_like(labeled, dtype=np.uint8),
        labels=labeled,
        index=labels
    )
    centroids_arr = np.array(centroids, dtype=np.float64)  # shape: (n_labels, 2)
    mean_row = centroids_arr[:, 0]
    mean_col = centroids_arr[:, 1]

    # Convert mean row/col to coordinates
    xs, ys = rasterio.transform.xy(binary.transform, mean_row, mean_col, offset="center")
    df = pd.DataFrame({
        "patch_id": labels.astype(np.int64),
        "area_ha": areas_ha,
        "centroid_x": np.asarray(xs, dtype=np.float64),
        "centroid_y": np.asarray(ys, dtype=np.float64),
        "pixel_count": counts.astype(np.int64),
    })
    return labeled, df.sort_values("patch_id").reset_index(drop=True)


def summarise_patch_sizes(df_patches: pd.DataFrame) -> Dict[str, float]:
    areas = df_patches["area_ha"].to_numpy()

    # Patch counts per size band (use <= 0.01 to capture single 10m pixels = 0.01 ha)
    mask_lt0_01 = areas <= 0.01
    mask_0_01_0_1 = (areas > 0.01) & (areas < 0.1)
    mask_0_1_1 = (areas >= 0.1) & (areas < 1.0)
    mask_1_5 = (areas >= 1.0) & (areas < 5.0)
    mask_5_10 = (areas >= 5.0) & (areas < 10.0)
    mask_gte10 = areas >= 10.0

    lt0_01 = float(mask_lt0_01.sum())
    bet0_01_0_1 = float(mask_0_01_0_1.sum())
    bet0_1_1 = float(mask_0_1_1.sum())
    bet1_5 = float(mask_1_5.sum())
    bet5_10 = float(mask_5_10.sum())
    gte10 = float(mask_gte10.sum())

    # Total area (ha) per size band
    area_lt0_01 = float(areas[mask_lt0_01].sum()) if mask_lt0_01.any() else 0.0
    area_0_01_0_1 = float(areas[mask_0_01_0_1].sum()) if mask_0_01_0_1.any() else 0.0
    area_0_1_1 = float(areas[mask_0_1_1].sum()) if mask_0_1_1.any() else 0.0
    area_1_5 = float(areas[mask_1_5].sum()) if mask_1_5.any() else 0.0
    area_5_10 = float(areas[mask_5_10].sum()) if mask_5_10.any() else 0.0
    area_gte10 = float(areas[mask_gte10].sum()) if mask_gte10.any() else 0.0

    modal = _modal_value(np.round(areas, 1))
    return {
        "num_patches": float(areas.size),
        "patches_lt_0_01ha": lt0_01,
        "patches_0_01_to_0_1ha": bet0_01_0_1,
        "patches_0_1_to_1ha": bet0_1_1,
        "patches_1_to_5ha": bet1_5,
        "patches_5_to_10ha": bet5_10,
        "patches_gte_10ha": gte10,
        "area_lt_0_01ha": area_lt0_01,
        "area_0_01_to_0_1ha": area_0_01_0_1,
        "area_0_1_to_1ha": area_0_1_1,
        "area_1_to_5ha": area_1_5,
        "area_5_to_10ha": area_5_10,
        "area_gte_10ha": area_gte10,
        "modal_patch_size_ha": float(modal) if modal is not None else float("nan"),
        "median_patch_size_ha": float(np.median(areas)) if areas.size else float("nan"),
        "mean_patch_size_ha": float(np.mean(areas)) if areas.size else float("nan"),
    }


def compute_patches_and_areas_from_labels(binary: RasterData, labels: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute patch metrics given a precomputed label raster (aligned to binary).
    Areas and centroids are computed on original wet pixels only (labels>0).
    Optimized using ndimage.center_of_mass for centroid calculation.
    """
    labels_arr = labels.astype(np.int32)
    valid = labels_arr > 0
    if not np.any(valid):
        return labels_arr, pd.DataFrame(columns=["patch_id", "area_ha", "centroid_x", "centroid_y", "pixel_count"])

    # Get unique labels and counts
    lbls, counts = np.unique(labels_arr[valid], return_counts=True)
    pixel_area_ha = binary.pixel_area_m2 / 10000.0
    areas_ha = counts.astype(np.float64) * pixel_area_ha

    # Use ndimage.center_of_mass for efficient centroid calculation
    # This is much faster than pandas groupby for large numbers of patches
    centroids = ndimage.center_of_mass(
        input=np.ones_like(labels_arr, dtype=np.uint8),
        labels=labels_arr,
        index=lbls
    )
    # centroids is a list of (row, col) tuples
    centroids_arr = np.array(centroids, dtype=np.float64)  # shape: (n_labels, 2)
    mean_row = centroids_arr[:, 0]
    mean_col = centroids_arr[:, 1]

    # Convert row/col to geographic coordinates
    xs, ys = rasterio.transform.xy(binary.transform, mean_row, mean_col, offset="center")

    df = pd.DataFrame({
        "patch_id": lbls.astype(np.int64),
        "area_ha": areas_ha,
        "centroid_x": np.asarray(xs, dtype=np.float64),
        "centroid_y": np.asarray(ys, dtype=np.float64),
        "pixel_count": counts.astype(np.int64),
    })
    return labels_arr, df.sort_values("patch_id").reset_index(drop=True)


def compute_nearest_neighbour(df_patches: pd.DataFrame) -> pd.DataFrame:
    if df_patches.empty or df_patches.shape[0] < 2:
        df_patches["nearest_neighbour_m"] = np.nan
        return df_patches
    coords = df_patches[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    tree = cKDTree(coords)
    # Query k=2: first result is the point itself (distance 0), second is nearest other
    dists, idxs = tree.query(coords, k=2)
    nn_d = dists[:, 1]
    df = df_patches.copy()
    df["nearest_neighbour_m"] = nn_d.astype(np.float64)
    return df


def compute_proximity_index(
    df_patches: pd.DataFrame,
    radius_m: float = 1000.0,
    power: float = 2.0,
) -> pd.DataFrame:
    """
    Proximity/Connectivity index per patch:
      PROX_i = sum over j != i, d_ij <= radius ( area_j / d_ij^power )
    Uses centroid-to-centroid distances and patch areas in hectares.
    Fully vectorized implementation for performance.
    """
    if df_patches.empty or df_patches.shape[0] < 2:
        df_patches["proximity_index"] = np.nan
        df_patches["proximity_neighbors"] = 0
        return df_patches
    coords = df_patches[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    areas_ha = df_patches["area_ha"].to_numpy(dtype=np.float64)
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=float(radius_m))

    # Vectorized computation
    n = len(df_patches)
    prox_values = np.zeros(n, dtype=np.float64)
    nbr_counts = np.zeros(n, dtype=np.int32)

    for i, nbrs in enumerate(neighbors):
        if len(nbrs) <= 1:  # Only self or empty
                continue
        # Convert to array and remove self
        nbrs_arr = np.array(nbrs, dtype=np.int32)
        mask = nbrs_arr != i
        nbrs_arr = nbrs_arr[mask]

        if len(nbrs_arr) == 0:
                continue

        # Vectorized distance calculation
        dx = coords[nbrs_arr, 0] - coords[i, 0]
        dy = coords[nbrs_arr, 1] - coords[i, 1]
        dists = np.hypot(dx, dy)

        # Filter out zero distances (shouldn't happen, but safety check)
        valid = dists > 0
        if not np.any(valid):
            continue

        dists = dists[valid]
        nbrs_arr = nbrs_arr[valid]

        # Vectorized proximity calculation: area_j / dist^power
        prox_contributions = areas_ha[nbrs_arr] / (dists ** power)
        prox_values[i] = np.sum(prox_contributions)
        nbr_counts[i] = len(nbrs_arr)

    df = df_patches.copy()
    df["proximity_index"] = prox_values
    df["proximity_neighbors"] = nbr_counts
    return df


def compute_landscape_division_metrics(df_patches: pd.DataFrame) -> Dict[str, float]:
    """
    Landscape Division Index (LDI) and Effective Mesh Size for the wet woodland class:
      - Let A = total wet woodland area, a_i = area of patch i
      - LDI = 1 - sum(a_i^2) / A^2
      - Effective mesh size (m_eff) = sum(a_i^2) / A
    Areas can be in any consistent unit; we use hectares.

    NOTE: These are LANDSCAPE-SCALE metrics computed across the ENTIRE study area.
    """
    if df_patches.empty:
        return {"landscape_division_index": float("nan"), "effective_mesh_size_ha": float("nan")}
    areas = df_patches["area_ha"].to_numpy(dtype=np.float64)
    A = float(np.sum(areas))
    if A <= 0:
        return {"landscape_division_index": float("nan"), "effective_mesh_size_ha": float("nan")}
    sum_sq = float(np.sum(areas * areas))
    ldi = 1.0 - (sum_sq / (A * A))
    meff = sum_sq / A

    # Additional metrics to understand the distribution
    # Contribution of largest patch to total area
    largest_area = float(np.max(areas)) if areas.size > 0 else 0.0
    largest_pct = (largest_area / A * 100) if A > 0 else 0.0

    # Contribution of top 10% patches to total area
    n_patches = len(areas)
    n_top10 = max(1, int(n_patches * 0.1))
    top10_areas = np.sort(areas)[-n_top10:]
    top10_total = float(np.sum(top10_areas))
    top10_pct = (top10_total / A * 100) if A > 0 else 0.0

    return {
        "landscape_division_index": ldi,
        "effective_mesh_size_ha": meff,
        "largest_patch_pct_of_total": largest_pct,
        "top10pct_patches_area_pct": top10_pct,
    }


def _derive_block_factors(pixel_size_m: float, target_cell_m: float) -> int:
    # Returns the integer downsample factor closest to target_cell_m
    if pixel_size_m <= 0:
        return 0
    factor = target_cell_m / pixel_size_m
    # Round to nearest positive integer, at least 1
    return max(1, int(round(factor)))


def density_grids(binary: RasterData, km1: bool = True, km10: bool = True) -> Dict[str, Tuple[np.ndarray, Affine]]:
    results: Dict[str, Tuple[np.ndarray, Affine]] = {}
    wet = (binary.array == 1).astype(np.uint32)
    px_area_m2 = binary.pixel_area_m2
    # 1 km
    if km1:
        f1 = _derive_block_factors(binary.pixel_width, 1000.0)
        sum1 = _block_reduce_sum(wet, f1, f1).astype(np.float32)
        # density in ha per km^2 (km^2 = 1e6 m^2)
        area1_m2 = sum1 * px_area_m2
        density1_ha_per_km2 = (area1_m2 / 10000.0) / ( (f1 * binary.pixel_width) * (f1 * binary.pixel_height) / 1_000_000.0 )
        transform1 = _grid_transform_from_block_reduce(binary.transform, f1, f1)
        results["density_1km"] = (density1_ha_per_km2, transform1)
    # 10 km
    if km10:
        f10 = _derive_block_factors(binary.pixel_width, 10_000.0)
        sum10 = _block_reduce_sum(wet, f10, f10).astype(np.float32)
        area10_m2 = sum10 * px_area_m2
        density10_ha_per_km2 = (area10_m2 / 10000.0) / ( (f10 * binary.pixel_width) * (f10 * binary.pixel_height) / 1_000_000.0 )
        transform10 = _grid_transform_from_block_reduce(binary.transform, f10, f10)
        results["density_10km"] = (density10_ha_per_km2, transform10)
    return results


def aggregate_by_lnrs(binary: RasterData, lnrs_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if lnrs_gdf.empty:
        return pd.DataFrame()
    lnrs_gdf = lnrs_gdf.to_crs(binary.crs).reset_index(drop=True).copy()
    # Reserve 0 for background so region IDs are always valid in raster space.
    lnrs_gdf["lnrs_id"] = np.arange(1, len(lnrs_gdf) + 1, dtype=np.int32)
    region_raster = rasterize(
        ((geom, rid) for geom, rid in zip(lnrs_gdf.geometry, lnrs_gdf["lnrs_id"])),
        out_shape=binary.shape,
        transform=binary.transform,
        fill=0,
        all_touched=False,
        dtype=np.int32,
    )
    wet_mask = (binary.array == 1)
    ids, wet_counts = np.unique(region_raster[wet_mask], return_counts=True)
    pixel_area_ha = binary.pixel_area_m2 / 10000.0
    wet_area_ha = wet_counts.astype(np.float64) * pixel_area_ha
    # Region areas (in hectares) from raster cell counts per region
    ids_all, cell_counts = np.unique(region_raster, return_counts=True)
    region_area_ha = (cell_counts.astype(np.float64) * pixel_area_ha)
    df_area = pd.DataFrame({"lnrs_id": ids_all.astype(np.int64), "region_area_ha": region_area_ha})
    df_wet = pd.DataFrame({"lnrs_id": ids.astype(np.int64), "wet_area_ha": wet_area_ha})
    df = df_area.merge(df_wet, on="lnrs_id", how="left").fillna({"wet_area_ha": 0.0})
    df["wet_density_ha_per_km2"] = df["wet_area_ha"] / (df["region_area_ha"] / 100.0)
    # Bring attributes if present (like names)
    if "NAME" in lnrs_gdf.columns:
        names = lnrs_gdf[["lnrs_id", "NAME"]]
        df = df.merge(names, on="lnrs_id", how="left")
    return df.sort_values("lnrs_id").reset_index(drop=True)


def _prepare_lnrs_overlay(
    lnrs_gdf: gpd.GeoDataFrame,
    binary: RasterData,
    all_touched: bool = True,
    buffer_m: float = 0.0,
) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
    """
    Prepare LNRS polygons for raster analysis once per run.
    - Reprojects to the wet woodland CRS
    - Applies any requested buffer
    - Assigns 1-based lnrs_id values so 0 remains reserved for background
    - Rasterizes the regions to the wet woodland grid
    """
    if lnrs_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=binary.crs), np.zeros(binary.shape, dtype=np.int32)

    lnrs = lnrs_gdf.to_crs(binary.crs).reset_index(drop=True).copy()
    if buffer_m and buffer_m != 0.0:
        lnrs["geometry"] = lnrs.geometry.buffer(buffer_m)
    lnrs["lnrs_id"] = np.arange(1, len(lnrs) + 1, dtype=np.int32)
    lnrs["region_area_ha"] = lnrs.geometry.area.astype(np.float64) / 10000.0

    region_raster = rasterize(
        ((geom, int(rid)) for geom, rid in zip(lnrs.geometry, lnrs["lnrs_id"])),
        out_shape=binary.shape,
        transform=binary.transform,
        fill=0,
        all_touched=all_touched,
        dtype=np.int32,
    )
    return lnrs, region_raster


def _count_pixels_by_region(region_raster: np.ndarray, mask: np.ndarray, n_regions: int) -> np.ndarray:
    if n_regions <= 0:
        return np.zeros(0, dtype=np.int64)
    region_ids = region_raster[mask].astype(np.int64, copy=False)
    if region_ids.size == 0:
        return np.zeros(n_regions, dtype=np.int64)
    counts = np.bincount(region_ids, minlength=n_regions + 1)
    return counts[1 : n_regions + 1]


def aggregate_by_lnrs_vector(
    binary: RasterData,
    lnrs_gdf: gpd.GeoDataFrame,
    all_touched: bool = True,
    buffer_m: float = 0.0,
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, np.ndarray]:
    """
    Aggregation by LNRS regions without creating a background class:
      - Region area is computed from vector polygon geometry area (ha)
      - Wet woodland area is computed from one shared rasterized LNRS grid
    This avoids per-polygon rasterization and keeps vector-based region areas.
    """
    if lnrs_gdf.empty:
        return pd.DataFrame(), gpd.GeoDataFrame(geometry=[], crs=binary.crs), np.zeros(binary.shape, dtype=np.int32)

    lnrs, region_raster = _prepare_lnrs_overlay(
        lnrs_gdf,
        binary,
        all_touched=all_touched,
        buffer_m=buffer_m,
    )
    n_regions = len(lnrs)
    pixel_area_ha = binary.pixel_area_m2 / 10000.0
    wet_counts = _count_pixels_by_region(region_raster, binary.array == 1, n_regions)

    df = lnrs.drop(columns="geometry").copy()
    df["wet_area_ha"] = wet_counts.astype(np.float64) * pixel_area_ha
    df["wet_density_ha_per_km2"] = df["wet_area_ha"] / (df["region_area_ha"] / 100.0).replace({0.0: np.nan})
    cols = ["lnrs_id", "region_area_ha", "wet_area_ha", "wet_density_ha_per_km2"]
    if "NAME" in df.columns:
        cols.append("NAME")
    return df[cols].sort_values("wet_area_ha", ascending=False).reset_index(drop=True), lnrs, region_raster


def assign_patches_to_lnrs(
    labeled: np.ndarray,
    df_patches: pd.DataFrame,
    region_raster: np.ndarray,
    binary: RasterData,
) -> pd.DataFrame:
    """
    Assign each patch to the LNRS region that contains the majority of its pixels.
    Adds column lnrs_id to df_patches. Patches with no overlap get lnrs_id = -1.
    """
    if df_patches.empty:
        df = df_patches.copy()
        df["lnrs_id"] = -1
        return df

    wet_mask = (binary.array == 1)
    flat_label = labeled[wet_mask].astype(np.int64)
    flat_region = region_raster[wet_mask].astype(np.int64)
    valid = (flat_label > 0) & (flat_region > 0)
    if not np.any(valid):
        df = df_patches.copy()
        df["lnrs_id"] = -1
        return df

    pairs = np.column_stack((flat_label[valid], flat_region[valid]))
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    patch_ids = unique_pairs[:, 0]
    region_ids = unique_pairs[:, 1]

    # Sort so the first row for each patch is the majority region.
    order = np.lexsort((region_ids, -counts, patch_ids))
    patch_ids_sorted = patch_ids[order]
    region_ids_sorted = region_ids[order]
    first_idx = np.unique(patch_ids_sorted, return_index=True)[1]
    lnrs_per_patch = pd.Series(region_ids_sorted[first_idx], index=patch_ids_sorted[first_idx])

    df = df_patches.copy()
    df["lnrs_id"] = df["patch_id"].map(lnrs_per_patch).fillna(-1).astype(np.int64)
    return df


def _summarise_patch_sizes_for_region(df_sub: pd.DataFrame) -> Dict[str, float]:
    """Same bands as summarise_patch_sizes; returns dict suitable for one row per region."""
    if df_sub.empty:
        return {
            "area_leq_0_01_ha": 0.0, "area_0_01_0_1_ha": 0.0, "area_0_1_1_ha": 0.0,
            "area_1_5_ha": 0.0, "area_5_10_ha": 0.0, "area_gte_10_ha": 0.0,
            "pct_leq_0_01": 0.0, "pct_0_01_0_1": 0.0, "pct_0_1_1": 0.0,
            "pct_1_5": 0.0, "pct_5_10": 0.0, "pct_gte_10": 0.0,
            "median_patch_ha": float("nan"), "largest_patch_ha": float("nan"),
            "num_patches": 0,
        }
    areas = df_sub["area_ha"].to_numpy()
    mask_lt0_01 = areas <= 0.01
    mask_0_01_0_1 = (areas > 0.01) & (areas < 0.1)
    mask_0_1_1 = (areas >= 0.1) & (areas < 1.0)
    mask_1_5 = (areas >= 1.0) & (areas < 5.0)
    mask_5_10 = (areas >= 5.0) & (areas < 10.0)
    mask_gte10 = areas >= 10.0
    area_leq_0_01 = float(areas[mask_lt0_01].sum()) if mask_lt0_01.any() else 0.0
    area_0_01_0_1 = float(areas[mask_0_01_0_1].sum()) if mask_0_01_0_1.any() else 0.0
    area_0_1_1 = float(areas[mask_0_1_1].sum()) if mask_0_1_1.any() else 0.0
    area_1_5 = float(areas[mask_1_5].sum()) if mask_1_5.any() else 0.0
    area_5_10 = float(areas[mask_5_10].sum()) if mask_5_10.any() else 0.0
    area_gte10 = float(areas[mask_gte10].sum()) if mask_gte10.any() else 0.0
    total = area_leq_0_01 + area_0_01_0_1 + area_0_1_1 + area_1_5 + area_5_10 + area_gte10
    pct = (lambda a, t: (a / t * 100.0) if t > 0 else 0.0)
    return {
        "area_leq_0_01_ha": area_leq_0_01, "area_0_01_0_1_ha": area_0_01_0_1,
        "area_0_1_1_ha": area_0_1_1, "area_1_5_ha": area_1_5,
        "area_5_10_ha": area_5_10, "area_gte_10_ha": area_gte10,
        "pct_leq_0_01": pct(area_leq_0_01, total), "pct_0_01_0_1": pct(area_0_01_0_1, total),
        "pct_0_1_1": pct(area_0_1_1, total), "pct_1_5": pct(area_1_5, total),
        "pct_5_10": pct(area_5_10, total), "pct_gte_10": pct(area_gte10, total),
        "median_patch_ha": float(np.median(areas)) if areas.size else float("nan"),
        "largest_patch_ha": float(np.max(areas)) if areas.size else float("nan"),
        "num_patches": int(areas.size),
    }


def compute_lnrs_region_stats(
    lnrs_gdf: gpd.GeoDataFrame,
    df_lnrs: pd.DataFrame,
    df_patches: pd.DataFrame,
    binary: RasterData,
    region_raster: np.ndarray,
    peat_gdf: Optional[gpd.GeoDataFrame] = None,
    forest: Optional[RasterData] = None,
) -> pd.DataFrame:
    """
    Compute full coverage, patch distribution, and fragmentation stats per LNRS region.
    df_lnrs must have lnrs_id, wet_area_ha, region_area_ha.
    df_patches must have lnrs_id, area_ha, centroid_x, centroid_y (and will be used to recompute NN per region).
    region_raster: int array same shape as binary, 0 = no region.
    Returns DataFrame with one row per lnrs_id and all stat columns.
    """
    if df_lnrs.empty:
        return pd.DataFrame()

    pixel_area_ha = binary.pixel_area_m2 / 10000.0
    wet_mask = (binary.array == 1)
    n_regions = len(lnrs_gdf)

    wet_counts = _count_pixels_by_region(region_raster, wet_mask, n_regions)

    peat_mask = None
    on_peat_counts = off_peat_counts = None
    if peat_gdf is not None and not peat_gdf.empty:
        peat_mask = _vector_to_raster_mask(peat_gdf, binary, burn_value=1)
        on_peat_counts = _count_pixels_by_region(region_raster, wet_mask & (peat_mask == 1), n_regions)
        off_peat_counts = _count_pixels_by_region(region_raster, wet_mask & (peat_mask == 0), n_regions)

    forest_mask = None
    forest_counts = None
    if forest is not None:
        forest_aligned = _align_float_raster_like(forest, binary, resampling=Resampling.nearest)
        forest_mask = (forest_aligned.array > 0)
        forest_counts = _count_pixels_by_region(region_raster, forest_mask & (region_raster > 0), n_regions)

    empty_patch_metrics = _summarise_patch_sizes_for_region(pd.DataFrame())
    region_patch_metrics: Dict[int, Dict[str, float]] = {}
    if not df_patches.empty and "lnrs_id" in df_patches.columns:
        grouped = df_patches.loc[df_patches["lnrs_id"] > 0].groupby("lnrs_id", sort=False)
        for lnrs_id, df_sub in grouped:
            df_sub = df_sub.copy()
            patch_stats = _summarise_patch_sizes_for_region(df_sub)
            if len(df_sub) >= 2:
                df_sub = compute_nearest_neighbour(df_sub)
                mean_nn_m = float(df_sub["nearest_neighbour_m"].mean())
            else:
                mean_nn_m = float("nan")
            div = compute_landscape_division_metrics(df_sub)
            effective_mesh_ha = div.get("effective_mesh_size_ha", float("nan"))
            region_patch_metrics[int(lnrs_id)] = {
                **patch_stats,
                "mean_nn_m": mean_nn_m,
                "effective_mesh_size_ha": effective_mesh_ha,
                "effective_mesh_size_km2": (
                    effective_mesh_ha / 100.0 if not math.isnan(effective_mesh_ha) else float("nan")
                ),
            }

    rows = []
    for row_lnrs in df_lnrs.itertuples(index=False):
        lnrs_id = int(row_lnrs.lnrs_id)
        idx = lnrs_id - 1
        wet_area_ha = float(row_lnrs.wet_area_ha)
        region_area_ha = float(row_lnrs.region_area_ha)

        on_peat_ha = off_peat_ha = float("nan")
        if on_peat_counts is not None and off_peat_counts is not None:
            on_peat_ha = float(on_peat_counts[idx]) * pixel_area_ha
            off_peat_ha = float(off_peat_counts[idx]) * pixel_area_ha

        forest_cover_pct = float("nan")
        if forest_counts is not None:
            forest_in_region = int(forest_counts[idx])
            forest_cover_pct = (
                float(wet_counts[idx]) / float(forest_in_region) * 100.0
                if forest_in_region > 0
                else float("nan")
            )

        patch_metrics = region_patch_metrics.get(
            lnrs_id,
            {
                **empty_patch_metrics,
                "mean_nn_m": float("nan"),
                "effective_mesh_size_ha": float("nan"),
                "effective_mesh_size_km2": float("nan"),
            },
        )

        out = {
            "lnrs_id": lnrs_id,
            "total_area_ha": wet_area_ha,
            "total_area_km2": wet_area_ha / 100.0,
            "region_area_ha": region_area_ha,
            "forest_cover_pct": forest_cover_pct,
            "on_peat_ha": on_peat_ha,
            "off_peat_ha": off_peat_ha,
            "on_peat_pct": (on_peat_ha / wet_area_ha * 100.0) if (peat_mask is not None and wet_area_ha > 0) else float("nan"),
            "off_peat_pct": (off_peat_ha / wet_area_ha * 100.0) if (peat_mask is not None and wet_area_ha > 0) else float("nan"),
            "effective_mesh_size_ha": patch_metrics["effective_mesh_size_ha"],
            "effective_mesh_size_km2": patch_metrics["effective_mesh_size_km2"],
            "mean_nn_m": patch_metrics["mean_nn_m"],
        }
        out.update(empty_patch_metrics)
        out.update(patch_metrics)
        rows.append(out)
    return pd.DataFrame(rows)


def distance_to_water(binary: RasterData, rivers_gdf: gpd.GeoDataFrame, lakes_gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, Dict[str, float]]:
    water_mask = np.zeros(binary.shape, dtype=np.uint8)
    if not rivers_gdf.empty:
        water_mask |= _vector_to_raster_mask(rivers_gdf, binary, burn_value=1)
    if not lakes_gdf.empty:
        water_mask |= _vector_to_raster_mask(lakes_gdf, binary, burn_value=1)

    # Check if water features were successfully rasterized
    water_pixel_count = int(np.sum(water_mask))
    if water_pixel_count == 0:
        warnings.warn("No water pixels were rasterized. Check that rivers/lakes overlap with study area and CRS matches.")

    # EDT distance transform on the inverse of water mask (distance from each pixel to nearest water pixel)
    # Compute distances in pixels then scale by pixel size
    inv = (water_mask == 0)
    # distance_transform_edt computes distance to nearest False when input is boolean;
    # we want distances to water (water_mask==1), so pass inverse
    dist_pixels = ndimage.distance_transform_edt(inv)
    # Convert to meters using average pixel size; assumes square pixels or uses geometric mean
    pixel_size_m = math.sqrt(binary.pixel_width * binary.pixel_height)
    dist_m = dist_pixels * pixel_size_m
    wet_mask = (binary.array == 1)
    wet_dists = dist_m[wet_mask]
    summary = {
        "wet_to_water_mean_m": float(np.mean(wet_dists)) if wet_dists.size else float("nan"),
        "wet_to_water_median_m": float(np.median(wet_dists)) if wet_dists.size else float("nan"),
        "wet_to_water_min_m": float(np.min(wet_dists)) if wet_dists.size else float("nan"),
        "wet_to_water_p10_m": float(np.percentile(wet_dists, 10)) if wet_dists.size else float("nan"),
        "wet_to_water_p90_m": float(np.percentile(wet_dists, 90)) if wet_dists.size else float("nan"),
        "wet_to_water_max_m": float(np.max(wet_dists)) if wet_dists.size else float("nan"),
        "water_pixels_rasterized": float(water_pixel_count),
    }
    return dist_m.astype(np.float32), summary


def sample_elevation(binary: RasterData, elev: RasterData) -> Dict[str, float]:
    # Reproject elevation to binary raster grid if needed
    if (elev.crs != binary.crs) or (elev.transform != binary.transform) or (elev.array.shape != binary.array.shape):
        dst_profile = {
            "height": binary.array.shape[0],
            "width": binary.array.shape[1],
            "crs": binary.crs,
            "transform": binary.transform,
        }
        elev_aligned = _reproject_match(elev, dst_profile, resampling=Resampling.bilinear)
    else:
        elev_aligned = elev
    wet_mask = (binary.array == 1)
    vals = elev_aligned.array[wet_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"elev_mean_m": float("nan"), "elev_median_m": float("nan"), "elev_p90_m": float("nan")}
    return {
        "elev_mean_m": float(np.mean(vals)),
        "elev_median_m": float(np.median(vals)),
        "elev_p90_m": float(np.percentile(vals, 90)),
    }


def floodplain_association(binary: RasterData, floodplain_gdf: gpd.GeoDataFrame) -> Dict[str, float]:
    if floodplain_gdf.empty:
        return {"wet_in_floodplain_pct": float("nan")}
    fp_mask = _vector_to_raster_mask(floodplain_gdf, binary, burn_value=1)
    wet_mask = (binary.array == 1)
    wet_in_fp = float((wet_mask & (fp_mask == 1)).sum())
    wet_total = float(wet_mask.sum())
    pct = (wet_in_fp / wet_total * 100.0) if wet_total > 0 else float("nan")
    return {"wet_in_floodplain_pct": pct}


def apply_urban_mask(binary: RasterData, urban_gdf: Optional[gpd.GeoDataFrame]) -> RasterData:
    if urban_gdf is None or urban_gdf.empty:
        return binary
    mask = _vector_to_raster_mask(urban_gdf, binary, burn_value=1)
    arr = binary.array.copy()
    arr[mask == 1] = 0
    return RasterData(array=arr, transform=binary.transform, crs=binary.crs, nodata=binary.nodata, dtype=binary.dtype)


def _align_float_raster_like(src: RasterData, like: RasterData, resampling=Resampling.bilinear) -> RasterData:
    if (src.crs == like.crs) and (src.transform == like.transform) and (src.array.shape == like.array.shape):
        return src
    dst_profile = {
        "height": like.array.shape[0],
        "width": like.array.shape[1],
        "crs": like.crs,
        "transform": like.transform,
    }
    return _reproject_match(src, dst_profile, resampling=resampling)


def sieve_small_patches(binary: RasterData, connectivity: int, min_pixels: int) -> RasterData:
    """
    Remove connected components smaller than min_pixels by zeroing them out.
    """
    if min_pixels is None or min_pixels <= 1:
        return binary
    labeled, _ = _label_patches(binary.array, connectivity=connectivity)
    counts = np.bincount(labeled.ravel())
    if counts.size <= 1:
        return binary
    counts[0] = 0
    mask_small = (labeled > 0) & (counts[labeled] < int(min_pixels))
    if not np.any(mask_small):
        return binary
    arr = binary.array.copy()
    arr[mask_small] = 0
    return RasterData(array=arr.astype(np.uint8), transform=binary.transform, crs=binary.crs, nodata=binary.nodata, dtype="uint8")


def classify_patches_tow(
    df_patches: pd.DataFrame,
    labeled: np.ndarray,
    binary: RasterData,
    height_raster: Optional[RasterData],
    height_min_m: float = 3.0,
    height_coverage: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Apply TOW thresholds:
      - Lone Tree:       area 5–350 m²,  min height 3 m
      - Group of Trees:  area >350–1000 m², min height 3 m
      - Small Woodland:  area >1000 m², min height 3 m
    If a height raster is provided, require that the fraction of pixels with height >= height_min_m
    within a patch is >= height_coverage (default 0.0 means at least one pixel).
    If height raster is not provided, only area thresholds are applied.
    """
    df = df_patches.copy()
    # Area in m²
    df["area_m2"] = df["area_ha"] * 10000.0
    wet_mask = (binary.array == 1)
    # Height coverage
    if height_raster is not None:
        height_aligned = _align_float_raster_like(height_raster, binary, resampling=Resampling.bilinear)
        height_ok = np.asarray(height_aligned.array >= float(height_min_m))
        # Count per label: number of wet pixels in patch with height_ok
        labels = df["patch_id"].to_numpy(dtype=np.int64)
        # Ensure labels are valid indices
        max_label = int(labeled.max()) if labeled.size else 0
        # boolean array of same shape
        ok_and_wet = (height_ok & wet_mask)
        # Sum ok pixels per patch
        ok_counts = ndimage.sum(ok_and_wet.astype(np.uint32), labels=labeled, index=labels).astype(np.float64)
        pixel_counts = df["pixel_count"].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            cover_frac = np.where(pixel_counts > 0, ok_counts / pixel_counts, 0.0)
        df["height_cover_frac"] = cover_frac
        df["height_pass"] = df["height_cover_frac"] >= float(height_coverage)
    else:
        df["height_cover_frac"] = np.nan
        df["height_pass"] = True
    # Category assignment by area and height pass - fully vectorized
    area_m2 = df["area_m2"].to_numpy()
    height_pass = df["height_pass"].to_numpy()

    # Initialize all as None (will use empty string for pandas compatibility)
    categories = np.full(len(df), None, dtype=object)

    # Apply conditions in reverse order (most general to most specific)
    # This way more specific conditions overwrite general ones
    valid_mask = (area_m2 >= 5.0) & height_pass

    # Small Woodland: > 1000 m²
    mask_woodland = valid_mask & (area_m2 > 1000.0)
    categories[mask_woodland] = "Small Woodland"

    # Group of Trees: 350 < area <= 1000 m²
    mask_group = valid_mask & (area_m2 > 350.0) & (area_m2 <= 1000.0)
    categories[mask_group] = "Group of Trees"

    # Lone Tree: 5 <= area <= 350 m²
    mask_lone = valid_mask & (area_m2 >= 5.0) & (area_m2 <= 350.0)
    categories[mask_lone] = "Lone Tree"

    df["category_tow"] = categories
    # Summaries
    summary: Dict[str, float] = {}
    for cat in ["Lone Tree", "Group of Trees", "Small Woodland"]:
        mask = (df["category_tow"] == cat)
        summary[f"num_{cat.lower().replace(' ', '_')}_patches"] = float(mask.sum())
        area_ha = float(df.loc[mask, "area_ha"].sum())
        summary[f"area_ha_{cat.lower().replace(' ', '_')}"] = area_ha
    return df, summary


# ----------------------
# CLI
# ----------------------

def _default_stats_paths() -> Dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    input_root = repo_root / "data" / "input"
    output_root = repo_root / "data" / "output"
    return {
        "wet_woodland_raster": output_root / "postprocess" / "wet_woodland_mosaic_hysteresis.tif",
        "tiles_dir": output_root / "predictions" / "tiles",
        "peat": input_root / "peat" / "peat_extent.shp",
        "lnrs": input_root / "boundaries" / "lnrs_areas.shp",
        "rivers": input_root / "hydro" / "rivers.shp",
        "lakes": input_root / "hydro" / "lakes.gpkg",
        "urban": input_root / "boundaries" / "england_urban.shp",
        "outdir": output_root / "stats" / "wet_woodland_stats",
        "report_file": output_root / "reports" / "wetwoodland_stats.txt",
    }


def _resolve_optional_source(
    value: Optional[str],
    label: str,
    default_path: Optional[Path] = None,
) -> Optional[str]:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate)
    if default_path is not None:
        candidate_norm = os.path.normpath(str(candidate.resolve(strict=False)))
        default_norm = os.path.normpath(str(default_path.resolve(strict=False)))
        if candidate_norm == default_norm:
            print(f"{label}: skipped (not found at {candidate})")
            return None
    raise SystemExit(f"{label} not found: {value}")


def _paths_match(path_str: str, default_path: Optional[Path]) -> bool:
    if default_path is None:
        return False
    return os.path.normpath(str(Path(path_str).expanduser().resolve(strict=False))) == os.path.normpath(
        str(default_path.resolve(strict=False))
    )


def _load_optional_vector_source(
    path: Optional[str],
    label: str,
    target_crs: str,
    default_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    if not path:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    try:
        return _load_vector(path, target_crs)
    except Exception as exc:
        if _paths_match(path, default_path):
            print(f"  {label}: skipped (unreadable default at {path}: {exc})")
            return gpd.GeoDataFrame(geometry=[], crs=target_crs)
        raise SystemExit(f"{label} could not be read: {path}\n{exc}") from exc


def _load_optional_raster_source(
    path: Optional[str],
    label: str,
    loader,
    default_path: Optional[Path] = None,
):
    if not path:
        return None
    try:
        return loader(path)
    except Exception as exc:
        if _paths_match(path, default_path):
            print(f"  {label}: skipped (unreadable default at {path}: {exc})")
            return None
        raise SystemExit(f"{label} could not be read: {path}\n{exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    defaults = _default_stats_paths()
    p = argparse.ArgumentParser(description="Wet Woodland Statistics Pipeline")
    core = p.add_argument_group("Core inputs")
    core.add_argument(
        "--wet-woodland-raster",
        default=str(defaults["wet_woodland_raster"]),
        help=(
            "Primary wet woodland raster input (band 1=binary 0/1, optional band 2=probabilities 0-1, 255=nodata). "
            f"Default: {defaults['wet_woodland_raster']}"
        ),
    )
    core.add_argument(
        "--tiles-dir",
        default=None,
        help=(
            "Optional prediction tiles directory. If set, tiles are mosaicked and used instead of --wet-woodland-raster. "
            f"Canonical location: {defaults['tiles_dir']}"
        ),
    )
    core.add_argument("--tiles-glob", default="*.tif", help="Glob pattern inside --tiles-dir (default: *.tif)")
    core.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold for band 2 probabilities. If omitted, uses band 1 binary classifications.",
    )

    overlays = p.add_argument_group("Standard overlays")
    overlays.add_argument(
        "--peat",
        default=str(defaults["peat"]),
        help=f"Peat polygon layer (default: {defaults['peat']}; skipped if missing)",
    )
    overlays.add_argument(
        "--lnrs",
        default=str(defaults["lnrs"]),
        help=f"LNRS polygons (default: {defaults['lnrs']}; skipped if missing)",
    )
    overlays.add_argument(
        "--rivers",
        default=str(defaults["rivers"]),
        help=f"Rivers line layer (default: {defaults['rivers']}; skipped if missing)",
    )
    overlays.add_argument(
        "--lakes",
        default=str(defaults["lakes"]),
        help=f"Lakes polygon layer (default: {defaults['lakes']}; skipped if missing)",
    )
    overlays.add_argument(
        "--urban",
        default=str(defaults["urban"]),
        help=f"Urban polygons used when --mask-urban is set (default: {defaults['urban']}; skipped if missing)",
    )
    overlays.add_argument("--lnrs-all-touched", action="store_true", help="Treat pixels touched by LNRS polygon edges as inside")
    overlays.add_argument("--lnrs-buffer-m", type=float, default=0.0, help="Buffer LNRS polygons by this many meters before aggregation")
    overlays.add_argument("--mask-urban", action="store_true", help="Mask urban polygons out of the wet woodland raster before analysis")

    optional_inputs = p.add_argument_group("Optional extra inputs")
    optional_inputs.add_argument("--elevation", required=False, help="Optional elevation raster (m)")
    optional_inputs.add_argument("--floodplain", required=False, help="Optional floodplain polygon layer")
    optional_inputs.add_argument("--height-raster", required=False, help="Optional canopy height raster (m) for TOW thresholds")
    optional_inputs.add_argument("--forest-raster", required=False, help="Optional forest/woodland raster; values >0 treated as forest")

    analysis = p.add_argument_group("Analysis controls")
    analysis.add_argument("--sieve-min-area-m2", type=float, default=None, help="Remove connected components smaller than this area (m²) before analysis")
    analysis.add_argument("--sieve-min-pixels", type=int, default=None, help="Alternative sieve threshold in pixels")
    analysis.add_argument("--height-min", type=float, default=3.0, help="Minimum height (m) for TOW categories (default: 3.0)")
    analysis.add_argument("--height-coverage", type=float, default=0.0, help="Required fraction [0-1] of pixels >= min height within a patch (default: any)")
    analysis.add_argument("--bridge-pixels", type=int, default=0, help="Bridge connectivity by dilating up to N pixels before labeling")
    analysis.add_argument("--bridge-meters", type=float, default=None, help="Bridge connectivity distance in meters; converted to pixels using raster resolution")
    analysis.add_argument("--prox-radius-m", type=float, default=1000.0, help="Radius (m) for proximity index (default: 1000)")
    analysis.add_argument("--prox-power", type=float, default=2.0, help="Distance decay power for proximity index (default: 2)")
    analysis.add_argument("--connectivity", type=int, default=2, choices=[1, 2], help="Patch connectivity: 1=4-connected, 2=8-connected (default)")

    outputs = p.add_argument_group("Outputs")
    outputs.add_argument(
        "--outdir",
        default=str(defaults["outdir"]),
        help=f"Output directory for exported rasters/tables (default: {defaults['outdir']})",
    )
    outputs.add_argument(
        "--report-file",
        default=None,
        help=f"Optional text report path. Default: {defaults['report_file']}",
    )
    outputs.add_argument("--prefix", default="wet_woodland", help="Output filename prefix")
    outputs.add_argument("--export-all", action="store_true", help="Export all outputs: CSVs, rasters, GeoPackage, manifest (default: report only)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    defaults = _default_stats_paths()
    args = build_parser().parse_args(argv)
    _ensure_dir(args.outdir)
    report_path = args.report_file if args.report_file else str(defaults["report_file"])
    _ensure_dir(str(Path(report_path).parent))

    # Resolve primary input: prefer an explicit tiles dir, otherwise use the cleaned hysteresis mosaic.
    wet_input_path: Optional[str] = None
    if args.tiles_dir:
        if not os.path.isdir(args.tiles_dir):
            raise SystemExit(f"--tiles-dir is not a directory: {args.tiles_dir}")
        tiles = _collect_tiles(args.tiles_dir, args.tiles_glob)
        if not tiles:
            raise SystemExit(f"No tiles found in {args.tiles_dir} matching {args.tiles_glob}")
        print(f"Found {len(tiles)} tile(s) in {args.tiles_dir}")
        # Prefer VRT for scalability; fallback to merge
        vrt_path = os.path.join(args.outdir, f"{args.prefix}_tiles.vrt")
        print(f"Attempting to build VRT at {vrt_path}...")
        vrt_success = _try_build_vrt(tiles, vrt_path)
        if not vrt_success:
            vrt_success = _try_build_vrt_cli(tiles, vrt_path)
        if vrt_success:
            wet_input_path = vrt_path
            print(f"✓ Successfully created VRT: {vrt_path}")
        else:
            print("! VRT creation failed (GDAL not available), falling back to merging tiles...")
            print("  Note: To enable VRT (faster), install GDAL: pip install gdal or conda install gdal")
            merged_path = os.path.join(args.outdir, f"{args.prefix}_tiles_merged.tif")
            print(f"  Merging {len(tiles)} tiles to {merged_path}...")
            wet_input_path = _merge_tiles_to_tiff(tiles, merged_path)
            print(f"✓ Successfully merged tiles to {merged_path}")
    else:
        wet_input_path = args.wet_woodland_raster
        if not wet_input_path or not os.path.isfile(wet_input_path):
            raise SystemExit(
                "Wet woodland raster not found. "
                f"Expected cleaned hysteresis mosaic at {args.wet_woodland_raster} "
                "or provide --tiles-dir."
            )

    # Resolve optional canonical sources. Missing defaults are skipped; bad explicit paths still error.
    args.peat = _resolve_optional_source(args.peat, "Peat polygons", defaults["peat"])
    args.lnrs = _resolve_optional_source(args.lnrs, "LNRS polygons", defaults["lnrs"])
    args.rivers = _resolve_optional_source(args.rivers, "Rivers", defaults["rivers"])
    args.lakes = _resolve_optional_source(args.lakes, "Lakes", defaults["lakes"])
    args.urban = (
        _resolve_optional_source(args.urban, "Urban polygons", defaults["urban"])
        if args.mask_urban
        else None
    )
    args.elevation = _resolve_optional_source(args.elevation, "Elevation raster")
    args.floodplain = _resolve_optional_source(args.floodplain, "Floodplain polygons")
    args.height_raster = _resolve_optional_source(args.height_raster, "Height raster")
    args.forest_raster = _resolve_optional_source(args.forest_raster, "Forest raster")

    print("Wet woodland stats")
    print("=" * 60)
    print(f"Input raster:      {wet_input_path}")
    if args.tiles_dir:
        print(f"Tiles dir:         {args.tiles_dir}")
    print(f"Output dir:        {args.outdir}")
    print(f"Report file:       {report_path}")
    print(f"Peat source:       {args.peat if args.peat else 'Skipped'}")
    print(f"LNRS source:       {args.lnrs if args.lnrs else 'Skipped'}")
    print(f"Rivers source:     {args.rivers if args.rivers else 'Skipped'}")
    print(f"Lakes source:      {args.lakes if args.lakes else 'Skipped'}")
    print(f"Urban mask:        {args.urban if args.urban else 'Off'}")
    if args.export_all:
        print(f"Output prefix:     {args.prefix}")

    # Load binary wet woodland raster (VRT or GeoTIFF both supported)
    print(f"\nLoading wet woodland raster from {wet_input_path}...")

    # First read to get raw data for nodata detection
    with rasterio.open(wet_input_path) as ds:
        arr_raw = ds.read(1, masked=False)
        nodata_val = ds.nodata if ds.nodata is not None else 255

    # Count before processing
    nodata_pixels = int(np.sum(arr_raw == nodata_val))

    # Now process
    wet = _read_binary_raster(wet_input_path, threshold=args.threshold)
    # Valid footprint (exclude nodata)
    valid_count = int(np.sum(wet.valid_mask)) if (hasattr(wet, "valid_mask") and wet.valid_mask is not None) else wet.array.size
    wet_pixels = int(np.sum((wet.array == 1)))
    nonwet_pixels = int(np.sum((wet.array == 0) & (wet.valid_mask if wet.valid_mask is not None else np.ones_like(wet.array, dtype=bool))))
    valid_forest_pixels = valid_count  # all valid pixels are forest by construction (0 or 1)

    # Percentages and areas
    wet_pct_of_forest = (wet_pixels / valid_forest_pixels * 100) if valid_forest_pixels > 0 else 0
    nodata_pct = (nodata_pixels / (valid_count + nodata_pixels) * 100) if (valid_count + nodata_pixels) > 0 else 0
    pixel_area_m2 = wet.pixel_area_m2
    wet_area_ha = wet_pixels * pixel_area_m2 / 10000
    wet_area_km2 = wet_area_ha / 100
    forest_area_ha = valid_forest_pixels * pixel_area_m2 / 10000
    forest_area_km2 = forest_area_ha / 100
    england_land_area_km2 = 130_279.0
    england_land_area_ha = england_land_area_km2 * 100.0
    wet_pct_england = (wet_area_ha / england_land_area_ha * 100.0) if england_land_area_ha > 0 else 0.0

    print(f"  Raster shape: {wet.shape}, CRS: {wet.crs}, pixel size: {wet.pixel_width:.2f} x {wet.pixel_height:.2f} m")
    print(f"  Pixel classification:")
    print(f"    Wet woodland (1): {wet_pixels:,}")
    print(f"    Non-wet forest (0): {nonwet_pixels:,}")
    print(f"    Nodata (255): {nodata_pixels:,} ({nodata_pct:.1f}% of raster)")
    print(f"  ")
    print(f"  Proportional statistics:")
    print(f"    Wet woodland: {wet_pct_of_forest:.4f}% of forest pixels")
    print(f"    Wet woodland: {wet_pct_england:.4f}% of England land area")

    # Area estimates
    print(f"  ")
    print(f"  Area estimates:")
    print(f"    Wet woodland: {wet_area_ha:,.1f} ha ({wet_area_km2:,.1f} km²)")
    print(f"    Total forest (wet + non-wet): {forest_area_ha:,.1f} ha ({forest_area_km2:,.1f} km²)")

    # Optional vectors
    print("\nLoading vector datasets...")
    peat_gdf = _load_optional_vector_source(args.peat, "Peat", wet.crs, defaults["peat"])
    if args.peat:
        print(f"  Peat: {len(peat_gdf)} features")
    lnrs_gdf = _load_optional_vector_source(args.lnrs, "LNRS", wet.crs, defaults["lnrs"])
    if args.lnrs and not lnrs_gdf.empty:
        print(f"  LNRS: {len(lnrs_gdf)} regions")
    rivers_gdf = _load_optional_vector_source(args.rivers, "Rivers", wet.crs, defaults["rivers"])
    if args.rivers:
        print(f"  Rivers: {len(rivers_gdf)} features")
    lakes_gdf = _load_optional_vector_source(args.lakes, "Lakes", wet.crs, defaults["lakes"])
    if args.lakes:
        print(f"  Lakes: {len(lakes_gdf)} features")
    urban_gdf = _load_optional_vector_source(args.urban, "Urban", wet.crs, defaults["urban"])
    if args.urban:
        print(f"  Urban: {len(urban_gdf)} features")
    floodplain_gdf = _load_optional_vector_source(args.floodplain, "Floodplain", wet.crs)
    if args.floodplain:
        print(f"  Floodplain: {len(floodplain_gdf)} features")

    # Optional rasters
    print("\nLoading optional rasters...")
    elev = _load_optional_raster_source(args.elevation, "Elevation", _read_raster_float)
    if elev:
        print(f"  Elevation: {elev.shape}")
    height = _load_optional_raster_source(args.height_raster, "Height", _read_raster_float)
    if height:
        print(f"  Height: {height.shape}")
    forest = _load_optional_raster_source(args.forest_raster, "Forest", _read_raster_float)
    if forest:
        print(f"  Forest: {forest.shape}")

    # Urban mask
    if args.mask_urban and not urban_gdf.empty:
        print("\nApplying urban mask...")
        wet = apply_urban_mask(wet, urban_gdf)
        print(f"  Urban mask applied")

    # Optional sieve pre-filter
    if args.sieve_min_pixels is not None or args.sieve_min_area_m2 is not None:
        min_pixels = None
        if args.sieve_min_pixels is not None:
            min_pixels = int(args.sieve_min_pixels)
        elif args.sieve_min_area_m2 is not None:
            px_area = wet.pixel_area_m2
            if px_area > 0:
                min_pixels = int(math.ceil(float(args.sieve_min_area_m2) / px_area))
        if min_pixels and min_pixels > 1:
            print(f"\nSieving small patches (< {min_pixels} pixels)...")
            wet = sieve_small_patches(wet, connectivity=args.connectivity, min_pixels=min_pixels)
            print(f"  Sieve complete")

    print("\n" + "="*60)
    print("COMPUTING WET WOODLAND STATISTICS")
    print("="*60)

    # 1) Area split on/off peat
    print("\n[1/9] Computing total area (on-peat vs off-peat)...")
    totals = {}
    if not peat_gdf.empty:
        totals = compute_area_split_on_off_peat(wet, peat_gdf)
        total_ha = totals.get('total_wet_woodland_ha', 0)
        on_peat_ha = totals.get('on_peat_ha', 0)
        off_peat_ha = totals.get('off_peat_ha', 0)
        on_peat_pct = (on_peat_ha / total_ha * 100) if total_ha > 0 else 0
        off_peat_pct = (off_peat_ha / total_ha * 100) if total_ha > 0 else 0
        print(f"  Total wet woodland: {total_ha:,.2f} ha ({total_ha/100:,.2f} km²)")
        print(f"  On peat: {on_peat_ha:,.2f} ha ({on_peat_pct:.4f}%)")
        print(f"  Off peat: {off_peat_ha:,.2f} ha ({off_peat_pct:.4f}%)")
    else:
        # Fallback total
        pixel_area_ha = wet.pixel_area_m2 / 10000.0
        totals = {"total_wet_woodland_ha": float((wet.array == 1).sum()) * pixel_area_ha}
        print(f"  Total wet woodland: {totals['total_wet_woodland_ha']:,.2f} ha ({totals['total_wet_woodland_ha']/100:,.2f} km²)")

    # 2) Patches and areas (with optional bridging)
    print("\n[2/9] Labeling patches and computing areas...")
    bridge_pixels = int(args.bridge_pixels) if getattr(args, "bridge_pixels", 0) else 0
    if getattr(args, "bridge_meters", None) is not None and args.bridge_meters and args.bridge_meters > 0:
        px = math.sqrt(wet.pixel_width * wet.pixel_height)
        if px > 0:
            bridge_pixels = max(bridge_pixels, int(round(float(args.bridge_meters) / px)))
    if bridge_pixels > 0:
        bridge_m = bridge_pixels * math.sqrt(wet.pixel_width * wet.pixel_height)
        print(f"  Bridging enabled: {bridge_pixels} pixels (~{bridge_m:.1f} m)")
        print(f"  NOTE: To disable bridging, remove --bridge-pixels and --bridge-meters flags")
    else:
        print(f"  Bridging disabled (patches must be directly connected)")
    labels = label_with_bridge(wet.array, connectivity=args.connectivity, bridge_pixels=bridge_pixels)
    labeled, df_patches = compute_patches_and_areas_from_labels(wet, labels)
    print(f"  Found {len(df_patches)} patches")

    # 3) Patch size distribution
    print("\n[3/9] Computing patch size distribution...")
    patch_summary = summarise_patch_sizes(df_patches)
    num_patches = int(patch_summary.get('num_patches', 0))
    lt0_01 = int(patch_summary.get('patches_lt_0_01ha', 0))
    bet0_01_0_1 = int(patch_summary.get('patches_0_01_to_0_1ha', 0))
    bet0_1_1 = int(patch_summary.get('patches_0_1_to_1ha', 0))
    bet1_5 = int(patch_summary.get('patches_1_to_5ha', 0))
    bet5_10 = int(patch_summary.get('patches_5_to_10ha', 0))
    gte10 = int(patch_summary.get('patches_gte_10ha', 0))
    # Areas per size band
    area_lt0_01 = patch_summary.get('area_lt_0_01ha', 0)
    area_0_01_0_1 = patch_summary.get('area_0_01_to_0_1ha', 0)
    area_0_1_1 = patch_summary.get('area_0_1_to_1ha', 0)
    area_1_5 = patch_summary.get('area_1_to_5ha', 0)
    area_5_10 = patch_summary.get('area_5_to_10ha', 0)
    area_gte10 = patch_summary.get('area_gte_10ha', 0)

    # Area percentages
    total_area_ha = area_lt0_01 + area_0_01_0_1 + area_0_1_1 + area_1_5 + area_5_10 + area_gte10
    area_lt0_01_pct = (area_lt0_01 / total_area_ha * 100) if total_area_ha > 0 else 0
    area_0_01_0_1_pct = (area_0_01_0_1 / total_area_ha * 100) if total_area_ha > 0 else 0
    area_0_1_1_pct = (area_0_1_1 / total_area_ha * 100) if total_area_ha > 0 else 0
    area_1_5_pct = (area_1_5 / total_area_ha * 100) if total_area_ha > 0 else 0
    area_5_10_pct = (area_5_10 / total_area_ha * 100) if total_area_ha > 0 else 0
    area_gte10_pct = (area_gte10 / total_area_ha * 100) if total_area_ha > 0 else 0

    print(f"  ≤ 0.01 ha: {area_lt0_01:,.2f} ha ({area_lt0_01_pct:.2f}%) — {lt0_01:,} patches")
    print(f"  0.01-0.1 ha: {area_0_01_0_1:,.2f} ha ({area_0_01_0_1_pct:.2f}%) — {bet0_01_0_1:,} patches")
    print(f"  0.1-1 ha: {area_0_1_1:,.2f} ha ({area_0_1_1_pct:.2f}%) — {bet0_1_1:,} patches")
    print(f"  1-5 ha: {area_1_5:,.2f} ha ({area_1_5_pct:.2f}%) — {bet1_5:,} patches")
    print(f"  5-10 ha: {area_5_10:,.2f} ha ({area_5_10_pct:.2f}%) — {bet5_10:,} patches")
    print(f"  ≥ 10 ha: {area_gte10:,.2f} ha ({area_gte10_pct:.2f}%) — {gte10:,} patches")
    print(f"  Median patch size: {patch_summary.get('median_patch_size_ha', 0):.2f} ha")
    if len(df_patches) > 0:
        largest_patch_ha = df_patches['area_ha'].max()
        print(f"  Largest patch: {largest_patch_ha:,.2f} ha ({largest_patch_ha/100:,.2f} km²)")

    # 4) Nearest neighbour distances
    print("\n[4/9] Computing nearest neighbour distances...")
    df_patches = compute_nearest_neighbour(df_patches)
    if len(df_patches) > 0:
        print(f"  Mean NN distance: {df_patches['nearest_neighbour_m'].mean():.1f} m")

    # 4a) Proximity index within radius
    print(f"\n[5/9] Computing proximity index (radius={args.prox_radius_m}m)...")
    df_patches = compute_proximity_index(
        df_patches,
        radius_m=float(args.prox_radius_m),
        power=float(args.prox_power),
    )
    division_metrics = compute_landscape_division_metrics(df_patches)
    ldi = division_metrics.get('landscape_division_index', float('nan'))
    mesh = division_metrics.get('effective_mesh_size_ha', float('nan'))
    largest_pct = division_metrics.get('largest_patch_pct_of_total', float('nan'))
    top10_pct = division_metrics.get('top10pct_patches_area_pct', float('nan'))

    print(f"  Landscape Division Index: {ldi:.4f} (landscape-scale: 0=connected, 1=fragmented)")
    print(f"  Effective Mesh Size: {mesh:,.2f} ha ({mesh/100:,.2f} km²)")
    print(f"  Largest patch: {largest_pct:.4f}% of total area")
    print(f"  Top 10% of patches: {top10_pct:.4f}% of total area")

    # Warning for unrealistic values or dominance by large patches
    if not math.isnan(mesh) and mesh > 10000:  # >100 km²
        warnings.warn(
            f"Effective mesh size ({mesh:,.0f} ha = {mesh/100:,.0f} km²) is extremely large. "
            f"This suggests one or more very large patches dominating the landscape. "
            f"Largest patch accounts for {largest_pct:.4f}% of total area. "
            f"Check the largest patch size above. If bridging is enabled, consider disabling it."
        )
    elif not math.isnan(largest_pct) and largest_pct > 50:
        print(f"  Note: Largest patch dominates ({largest_pct:.4f}% of area) - fragmentation metrics may be misleading")

    # 4b) TOW categories by area (m²) and optional height thresholds
    print("\n[6/9] Classifying patches by TOW categories...")
    tow_summary = {}
    df_patches, tow_summary = classify_patches_tow(
        df_patches,
        labeled=labeled,
        binary=wet,
        height_raster=height,
        height_min_m=float(args.height_min),
        height_coverage=float(args.height_coverage),
    )

    # Calculate percentages for TOW categories
    total_wet_ha = totals.get('total_wet_woodland_ha', 0)
    lone_n = int(tow_summary.get('num_lone_tree_patches', 0))
    lone_ha = tow_summary.get('area_ha_lone_tree', 0)
    lone_pct = (lone_ha / total_wet_ha * 100) if total_wet_ha > 0 else 0

    group_n = int(tow_summary.get('num_group_of_trees_patches', 0))
    group_ha = tow_summary.get('area_ha_group_of_trees', 0)
    group_pct = (group_ha / total_wet_ha * 100) if total_wet_ha > 0 else 0

    small_n = int(tow_summary.get('num_small_woodland_patches', 0))
    small_ha = tow_summary.get('area_ha_small_woodland', 0)
    small_pct = (small_ha / total_wet_ha * 100) if total_wet_ha > 0 else 0

    print(f"  Lone Trees: {lone_n:,} patches ({lone_ha:,.2f} ha, {lone_pct:.1f}% of area)")
    print(f"  Group of Trees: {group_n:,} patches ({group_ha:,.2f} ha, {group_pct:.1f}% of area)")
    print(f"  Small Woodlands: {small_n:,} patches ({small_ha:,.2f} ha, {small_pct:.1f}% of area)")

    # 5) Density grids (1 km, 10 km)
    print("\n[7/9] Computing density grids (1km and 10km)...")
    densities = density_grids(wet, km1=True, km10=True)
    print(f"  1km grid: {densities.get('density_1km', (np.array([]), None))[0].shape if 'density_1km' in densities else 'N/A'}")
    print(f"  10km grid: {densities.get('density_10km', (np.array([]), None))[0].shape if 'density_10km' in densities else 'N/A'}")

    # 6) LNRS aggregation
    print("\n[8/9] Computing LNRS regional aggregation...")
    df_lnrs = pd.DataFrame()
    lnrs_overlay = gpd.GeoDataFrame(geometry=[], crs=wet.crs)
    region_raster: Optional[np.ndarray] = None
    if not lnrs_gdf.empty:
        # Prepare one shared LNRS raster and reuse it for every downstream LNRS computation.
        df_lnrs, lnrs_overlay, region_raster = aggregate_by_lnrs_vector(
            wet,
            lnrs_gdf,
            all_touched=bool(getattr(args, "lnrs_all_touched", False)),
            buffer_m=float(getattr(args, "lnrs_buffer_m", 0.0)),
        )
        print(f"  Aggregated across {len(df_lnrs)} LNRS regions")
        df_patches = assign_patches_to_lnrs(
            labeled,
            df_patches,
            region_raster,
            wet,
        )
        df_lnrs_stats = compute_lnrs_region_stats(
            lnrs_overlay,
            df_lnrs,
            df_patches,
            wet,
            region_raster,
            peat_gdf=peat_gdf if not peat_gdf.empty else None,
            forest=forest if (forest is not None) else None,
        )
    else:
        df_lnrs_stats = pd.DataFrame()
        print("  Skipped (no LNRS data)")

    # 7) Distances to water
    print("\n[9/9] Computing additional metrics...")
    water_dist_summary = {}
    dist_water_raster = None
    if (not rivers_gdf.empty) or (not lakes_gdf.empty):
        print("  - Distance to water...")
        dist_water_raster, water_dist_summary = distance_to_water(wet, rivers_gdf, lakes_gdf)
        print(f"    Water pixels rasterized: {int(water_dist_summary.get('water_pixels_rasterized', 0))}")
        print(f"    Min distance: {water_dist_summary.get('wet_to_water_min_m', float('nan')):.1f} m")
        print(f"    10th percentile: {water_dist_summary.get('wet_to_water_p10_m', float('nan')):.1f} m")
        print(f"    Median distance: {water_dist_summary.get('wet_to_water_median_m', float('nan')):.1f} m")
        print(f"    Mean distance: {water_dist_summary.get('wet_to_water_mean_m', float('nan')):.1f} m")
        print(f"    90th percentile: {water_dist_summary.get('wet_to_water_p90_m', float('nan')):.1f} m")
        print(f"    Max distance: {water_dist_summary.get('wet_to_water_max_m', float('nan')):.1f} m")
    else:
        print("  - Distance to water: Skipped (no water data)")

    # 8) Elevation and floodplain
    if elev is not None:
        print("  - Elevation statistics...")
        elev_summary = sample_elevation(wet, elev)
        print(f"    Mean elevation: {elev_summary.get('elev_mean_m', float('nan')):.1f} m")
    else:
        elev_summary = {}
        print("  - Elevation: Skipped (no elevation data)")

    if not floodplain_gdf.empty:
        print("  - Floodplain association...")
        flood_summary = floodplain_association(wet, floodplain_gdf)
        print(f"    Wet woodland in floodplain: {flood_summary.get('wet_in_floodplain_pct', float('nan')):.1f}%")
    else:
        flood_summary = {}
        print("  - Floodplain: Skipped (no floodplain data)")

    # 9) Exports
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)

    # Summary dictionary and global percentages
    pixel_area_ha = wet.pixel_area_m2 / 10000.0
    # Footprint: only valid (non-nodata) pixels count
    if hasattr(wet, "valid_mask") and wet.valid_mask is not None:
        valid_count = int(np.sum(wet.valid_mask))
    else:
        valid_count = int(wet.array.size)
    study_area_ha_footprint = float(valid_count) * pixel_area_ha
    wet_total_ha = float(totals.get("total_wet_woodland", totals.get("total_wet_woodland_ha", 0.0))) if isinstance(totals, dict) else 0.0
    if wet_total_ha == 0.0:
        wet_total_ha = float(totals.get("total_wet_woodland_ha", 0.0)) if isinstance(totals, dict) else 0.0
    wet_pct_footprint = (wet_total_ha / study_area_ha_footprint * 100.0) if study_area_ha_footprint > 0 else float("nan")

    # Forest totals:
    # If a forest raster is provided, use it; otherwise assume the prediction footprint (0/1)
    # already corresponds to "all forest" as per user workflow.
    forest_total_ha = float("nan")
    if 'forest' in locals() and forest is not None:
        # Align to wet grid
        forest_aligned = _align_float_raster_like(forest, wet, resampling=Resampling.nearest)
        forest_mask = forest_aligned.array > 0
        forest_total_ha = float(forest_mask.sum()) * pixel_area_ha
    else:
        # Use the valid raster footprint as total forest area (assumes predictions exist only over forest)
        forest_total_ha = study_area_ha_footprint

    lnrs_area_ha = float(df_lnrs["region_area_ha"].sum()) if not df_lnrs.empty and "region_area_ha" in df_lnrs.columns else float("nan")
    wet_area_lnrs_ha = float(df_lnrs["wet_area_ha"].sum()) if not df_lnrs.empty and "wet_area_ha" in df_lnrs.columns else float("nan")
    wet_pct_lnrs = (wet_area_lnrs_ha / lnrs_area_ha * 100.0) if (isinstance(lnrs_area_ha, float) and lnrs_area_ha and lnrs_area_ha > 0) else float("nan")

    # England fixed land area (requested): 130,279 km²
    england_land_area_km2 = 130_279.0
    england_land_area_ha = england_land_area_km2 * 100.0
    wet_pct_england = (wet_total_ha / england_land_area_ha * 100.0) if england_land_area_ha > 0 else float("nan")

    summary = {
        "pixel_area_m2": wet.pixel_area_m2,
        "study_area_ha_footprint": study_area_ha_footprint,
        "wet_pct_of_footprint": wet_pct_footprint,
    }
    if not math.isnan(lnquiv := lnrs_area_ha):
        summary["lnrs_union_area_ha"] = lnrs_area_ha
    if not math.isnan(wet_area_lnrs_ha):
        summary["wet_area_lnrs_ha"] = wet_area_lnrs_ha
    if not math.isnan(wet_pct_lnrs):
        summary["wet_pct_of_lnrs_union"] = wet_pct_lnrs
    summary["forest_total_area_ha"] = forest_total_ha
    summary["forest_total_area_km2"] = forest_total_ha / 100.0
    # Wet woodland as % of forest
    wet_pct_forest = (wet_total_ha / forest_total_ha * 100.0) if (forest_total_ha and forest_total_ha > 0) else float("nan")
    summary["wet_pct_of_forest"] = wet_pct_forest
    # Add England-wide fixed denominator stats
    summary["england_land_area_km2"] = england_land_area_km2
    summary["england_land_area_ha"] = england_land_area_ha
    summary["wet_pct_of_england_land"] = wet_pct_england

    summary.update(totals)
    summary.update(patch_summary)
    summary.update(tow_summary)
    summary.update(division_metrics)
    summary.update(water_dist_summary)
    summary.update(elev_summary)
    summary.update(flood_summary)

    # Human-readable text report (for papers)
    print("\nWriting summary report...")
    _write_summary_report(report_path, summary, df_patches, df_lnrs, args)
    print(f"  Report: {report_path}")

    if args.export_all:
        # CSV files
        print("\nWriting CSV files...")
        summary_path = os.path.join(args.outdir, f"{args.prefix}_summary.csv")
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        print(f"  Summary CSV: {summary_path}")

        patches_path = os.path.join(args.outdir, f"{args.prefix}_patch_metrics.csv")
        df_patches.to_csv(patches_path, index=False)
        print(f"  Patch metrics: {patches_path}")

        if not df_lnrs.empty:
            lnrs_path = os.path.join(args.outdir, f"{args.prefix}_lnrs_aggregation.csv")
            df_lnrs.to_csv(lnrs_path, index=False)
            print(f"  LNRS aggregation: {lnrs_path}")

        if not df_lnrs_stats.empty and not lnrs_overlay.empty:
            lnrs_export = lnrs_overlay.drop(columns=["region_area_ha"], errors="ignore")
            gdf_lnrs_out = lnrs_export.merge(df_lnrs_stats, on="lnrs_id", how="left")
            gdf_lnrs_out = gpd.GeoDataFrame(gdf_lnrs_out, geometry="geometry", crs=lnrs_overlay.crs)
            gdf_lnrs_out = gdf_lnrs_out.rename(columns={"lnrs_id": "region_id"})
            lnrs_gpkg_path = os.path.join(args.outdir, f"{args.prefix}_lnrs_regions.gpkg")
            gdf_lnrs_out.to_file(lnrs_gpkg_path, driver="GPKG", index=False)
            print(f"  LNRS regions (with stats): {lnrs_gpkg_path}")

        # Rasters
        print("\nWriting raster files...")
        if "density_1km" in densities:
            arr, tr = densities["density_1km"]
            dens1_path = os.path.join(args.outdir, f"{args.prefix}_density_1km_ha_per_km2.tif")
            _write_geotiff(dens1_path, arr.astype(np.float32), tr, wet.crs, nodata=np.nan, dtype="float32")
            print(f"  Density 1km: {dens1_path}")
        if "density_10km" in densities:
            arr, tr = densities["density_10km"]
            dens10_path = os.path.join(args.outdir, f"{args.prefix}_density_10km_ha_per_km2.tif")
            _write_geotiff(dens10_path, arr.astype(np.float32), tr, wet.crs, nodata=np.nan, dtype="float32")
            print(f"  Density 10km: {dens10_path}")
        if dist_water_raster is not None:
            water_dist_path = os.path.join(args.outdir, f"{args.prefix}_distance_to_water_m.tif")
            _write_geotiff(water_dist_path, dist_water_raster, wet.transform, wet.crs, nodata=np.nan, dtype="float32")
            print(f"  Distance to water: {water_dist_path}")
        labels_path = os.path.join(args.outdir, f"{args.prefix}_patch_labels.tif")
        _write_geotiff(labels_path, labeled.astype(np.int32), wet.transform, wet.crs, nodata=0, dtype="int32")
        print(f"  Patch labels: {labels_path}")

        # Manifest
        manifest_path = os.path.join(args.outdir, f"{args.prefix}_manifest.json")
        manifest = {
            "summary_csv": os.path.join(args.outdir, f"{args.prefix}_summary.csv"),
            "patch_metrics_csv": os.path.join(args.outdir, f"{args.prefix}_patch_metrics.csv"),
            "lnrs_csv": os.path.join(args.outdir, f"{args.prefix}_lnrs_aggregation.csv") if not df_lnrs.empty else None,
            "lnrs_regions_gpkg": os.path.join(args.outdir, f"{args.prefix}_lnrs_regions.gpkg") if not df_lnrs_stats.empty else None,
            "density_1km": os.path.join(args.outdir, f"{args.prefix}_density_1km_ha_per_km2.tif") if "density_1km" in densities else None,
            "density_10km": os.path.join(args.outdir, f"{args.prefix}_density_10km_ha_per_km2.tif") if "density_10km" in densities else None,
            "distance_to_water": os.path.join(args.outdir, f"{args.prefix}_distance_to_water_m.tif") if dist_water_raster is not None else None,
            "patch_labels": os.path.join(args.outdir, f"{args.prefix}_patch_labels.tif"),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Manifest: {manifest_path}")
    else:
        print("  (Pass --export-all to write CSVs, rasters, GeoPackage and manifest)")

    print("\n" + "="*60)
    print("COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nAll outputs written to: {args.outdir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
