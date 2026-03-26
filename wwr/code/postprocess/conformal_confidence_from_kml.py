#!/usr/bin/env python3
"""
Build a calibrated confidence map from a wet woodland score raster using KML
positives and, for the 3-class mode, discarded-background negatives.

Supported output modes:

1) three-class
   0 = confident_non_wet
   1 = uncertain
   2 = confident_wet
   255 = nodata

   Calibration uses split-conformal nonconformity scores:
     score = 1 - s  if y == 1
     score = s      if y == 0

   Thresholds:
     qhat   = conformal quantile at 1 - alpha
     t_low  = qhat
     t_high = 1 - qhat

2) wet-other
   1 = confident_wet
   255 = other_or_nodata

   Calibration is one-sided on positive sites only:
     score_pos = 1 - s
     t_wet     = 1 - qhat_pos

3) wet-fpr
   1 = confident_wet
   255 = other_or_nodata

   Calibration is one-sided on held-out negatives only:
     score_neg = s
     t_wet     = qhat_neg
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge as rio_merge
from rasterio.transform import xy
from rasterio.windows import Window, from_bounds, transform as window_transform

PRED_NODATA = 255.0
OUT_NODATA = 255


def _normalized_path_str(value: Optional[object]) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        return os.path.normpath(os.path.abspath(str(Path(str(value)).expanduser())))
    except Exception:
        return os.path.normpath(os.path.abspath(str(value)))


def _run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


def _collect_tiles(tiles_dir: Path, pattern: str) -> List[Path]:
    tiles = sorted(tiles_dir.glob(pattern))
    return [p for p in tiles if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]


def _detect_probability_band(path: Path) -> int:
    with rasterio.open(path) as ds:
        return 2 if ds.count >= 2 else 1


def _get_probability_band(ds: rasterio.DatasetReader) -> int:
    return 2 if ds.count >= 2 else 1


def _aligned_window(window: Window, width: int, height: int) -> Window:
    col_off = max(0, int(math.floor(window.col_off)))
    row_off = max(0, int(math.floor(window.row_off)))
    col_end = min(width, int(math.ceil(window.col_off + window.width)))
    row_end = min(height, int(math.ceil(window.row_off + window.height)))
    return Window(col_off, row_off, max(0, col_end - col_off), max(0, row_end - row_off))


def _build_probability_mosaic(
    tile_paths: List[Path],
    tile_prob_bands: Dict[Path, int],
    nodata_value: float,
    temp_dir: Path,
) -> Path:
    """
    Build a single-band probability mosaic from prediction tiles. Prefers GDAL VRT.
    """
    unique_bands = sorted({int(tile_prob_bands[p]) for p in tile_paths})
    uniform_band = len(unique_bands) == 1
    vrt_path = temp_dir / "probability_mosaic.vrt"
    file_list = temp_dir / "tile_list.txt"
    nd = str(float(nodata_value))

    if uniform_band:
        with open(file_list, "w", encoding="utf-8") as f:
            for p in tile_paths:
                f.write(f"{p}\n")

        cmd = ["gdalbuildvrt", "-overwrite", "-b", str(unique_bands[0]), "-srcnodata", nd, "-vrtnodata", nd]
        cmd.extend(["-input_file_list", str(file_list), str(vrt_path)])
        proc = _run_command(cmd)
        if proc.returncode == 0 and vrt_path.exists():
            return vrt_path
    else:
        tile_vrt_dir = temp_dir / "tile_prob_vrts"
        tile_vrt_dir.mkdir(parents=True, exist_ok=True)
        per_tile_vrts: List[Path] = []
        for i, p in enumerate(tile_paths):
            band = int(tile_prob_bands[p])
            out_vrt = tile_vrt_dir / f"tile_prob_{i:06d}.vrt"
            cmd = ["gdal_translate", "-of", "VRT", "-b", str(band), "-a_nodata", nd, str(p), str(out_vrt)]
            proc = _run_command(cmd)
            if proc.returncode != 0 or not out_vrt.exists():
                per_tile_vrts = []
                break
            per_tile_vrts.append(out_vrt)

        if per_tile_vrts:
            with open(file_list, "w", encoding="utf-8") as f:
                for p in per_tile_vrts:
                    f.write(f"{p}\n")
            cmd = ["gdalbuildvrt", "-overwrite", "-srcnodata", nd, "-vrtnodata", nd]
            cmd.extend(["-input_file_list", str(file_list), str(vrt_path)])
            proc = _run_command(cmd)
            if proc.returncode == 0 and vrt_path.exists():
                return vrt_path

    print("⚠️  GDAL probability VRT failed; falling back to rasterio.merge (higher memory).")
    out_tif = temp_dir / "probability_mosaic.tif"

    if uniform_band:
        source_paths = tile_paths
        merge_band = unique_bands[0]
    else:
        temp_tiles_dir = temp_dir / "single_band_tiles"
        temp_tiles_dir.mkdir(parents=True, exist_ok=True)
        source_paths = []
        for i, p in enumerate(tile_paths):
            band = int(tile_prob_bands[p])
            single_path = temp_tiles_dir / f"tile_prob_{i:06d}.tif"
            with rasterio.open(p) as src:
                arr = src.read(band, masked=False)
                profile = src.profile.copy()
                profile.update(
                    {
                        "driver": "GTiff",
                        "count": 1,
                        "dtype": arr.dtype,
                        "compress": "LZW",
                        "tiled": True,
                        "BIGTIFF": "IF_SAFER",
                        "nodata": float(nodata_value),
                    }
                )
                with rasterio.open(single_path, "w", **profile) as dst:
                    dst.write(arr, 1)
            source_paths.append(single_path)
        merge_band = 1

    srcs = [rasterio.open(p) for p in source_paths]
    try:
        mosaic, out_transform = rio_merge(srcs, indexes=[merge_band], nodata=nodata_value, method="first")
        profile = srcs[0].profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "count": 1,
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "dtype": "float32",
                "compress": "LZW",
                "tiled": True,
                "BIGTIFF": "YES",
                "nodata": float(nodata_value),
            }
        )
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(mosaic.astype(np.float32))
    finally:
        for src in srcs:
            src.close()

    return out_tif


def load_kml_polygons(kml_path: Path, target_crs) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(kml_path)
    if gdf.empty:
        return gdf
    poly_mask = gdf.geometry.geom_type.isin(("Polygon", "MultiPolygon"))
    gdf = gdf.loc[poly_mask].copy()
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    if target_crs and str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    return gdf.reset_index(drop=True)


def warn_if_calibration_kml_matches_training_metadata(
    kml_path: Path,
    *,
    score_raster: Path,
    background_raster: Path,
) -> None:
    """
    Warn if raster metadata suggests the supplied KML was also used during training.
    """
    calibration_kml = _normalized_path_str(kml_path)
    if calibration_kml is None:
        return

    checks = [
        ("score raster", score_raster),
        ("background raster", background_raster),
    ]
    for label, path in checks:
        try:
            with rasterio.open(path) as ds:
                recorded = ds.tags().get("wet_woodland_kml")
        except Exception:
            continue
        recorded_norm = _normalized_path_str(recorded)
        if recorded_norm and recorded_norm == calibration_kml:
            print(
                "⚠️  Calibration KML matches the training KML recorded in the "
                f"{label} metadata: {recorded}"
            )
            print(
                "   Conformal thresholds from this KML are not independent of training "
                "and will be optimistically calibrated."
            )


def _read_probability(ds: rasterio.DatasetReader, band: int, window: Window) -> np.ndarray:
    arr = ds.read(band, window=window, masked=False).astype(np.float32)
    nodata = ds.nodata if ds.nodata is not None else PRED_NODATA
    arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where((arr >= 0.0) & (arr <= 1.0), arr, np.nan)
    return arr


def sample_positive_calibration(
    prediction_path: Path,
    kml_gdf: gpd.GeoDataFrame,
    *,
    max_per_polygon: int,
    all_touched: bool,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    probs_all: List[np.ndarray] = []
    polygon_counts: List[Tuple[int, int, int]] = []

    with rasterio.open(prediction_path) as ds:
        prob_band = _get_probability_band(ds)
        pixel_size = max(abs(ds.transform.a), abs(ds.transform.e))

        for poly_idx, geom in enumerate(kml_gdf.geometry, start=1):
            if geom is None or geom.is_empty:
                polygon_counts.append((poly_idx, 0, 0))
                continue

            minx, miny, maxx, maxy = geom.bounds
            window = from_bounds(
                minx - pixel_size,
                miny - pixel_size,
                maxx + pixel_size,
                maxy + pixel_size,
                ds.transform,
            )
            window = _aligned_window(window, ds.width, ds.height)
            if window.width < 1 or window.height < 1:
                polygon_counts.append((poly_idx, 0, 0))
                continue

            probs = _read_probability(ds, prob_band, window)
            tile_transform = window_transform(window, ds.transform)
            mask = rasterize(
                [(geom, 1)],
                out_shape=(int(window.height), int(window.width)),
                transform=tile_transform,
                fill=0,
                all_touched=all_touched,
                dtype=np.uint8,
            ).astype(bool)

            rows, cols = np.where(mask & np.isfinite(probs))
            available = int(rows.size)
            if available == 0:
                polygon_counts.append((poly_idx, 0, 0))
                continue

            if max_per_polygon > 0 and available > max_per_polygon:
                chosen = rng.choice(available, size=max_per_polygon, replace=False)
                rows = rows[chosen]
                cols = cols[chosen]
            sampled = int(rows.size)
            probs_all.append(probs[rows, cols].astype(np.float32))
            polygon_counts.append((poly_idx, available, sampled))

    if probs_all:
        probs = np.concatenate(probs_all)
    else:
        probs = np.array([], dtype=np.float32)

    return {
        "probs": probs,
        "polygon_counts": polygon_counts,
        "n_positive": int(probs.size),
    }


def _parse_negative_classes(raw: str) -> List[int]:
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def _sample_spatially_stratified(
    xs: np.ndarray,
    ys: np.ndarray,
    target_n: int,
    grid_size_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(xs)
    if target_n <= 0 or n == 0:
        return np.array([], dtype=np.int64)
    if target_n >= n:
        return np.arange(n, dtype=np.int64)

    x0 = float(xs.min())
    y0 = float(ys.min())
    gx = np.floor((xs - x0) / grid_size_m).astype(np.int64)
    gy = np.floor((ys - y0) / grid_size_m).astype(np.int64)

    cells: Dict[Tuple[int, int], List[int]] = {}
    for idx, key in enumerate(zip(gx, gy)):
        cells.setdefault(key, []).append(idx)

    cell_keys = list(cells.keys())
    rng.shuffle(cell_keys)
    n_cells = len(cell_keys)
    if n_cells == 0:
        return np.array([], dtype=np.int64)

    selected: List[int] = []
    leftovers: List[int] = []

    if target_n <= n_cells:
        for key in cell_keys[:target_n]:
            idxs = np.asarray(cells[key], dtype=np.int64)
            selected.append(int(rng.choice(idxs)))
        return np.asarray(selected, dtype=np.int64)

    base = target_n // n_cells
    remainder = target_n % n_cells
    for i, key in enumerate(cell_keys):
        idxs = np.asarray(cells[key], dtype=np.int64)
        rng.shuffle(idxs)
        take = min(len(idxs), base + (1 if i < remainder else 0))
        if take > 0:
            selected.extend(idxs[:take].tolist())
        if take < len(idxs):
            leftovers.extend(idxs[take:].tolist())

    if len(selected) < target_n and leftovers:
        leftovers_arr = np.asarray(leftovers, dtype=np.int64)
        rng.shuffle(leftovers_arr)
        need = min(target_n - len(selected), len(leftovers_arr))
        selected.extend(leftovers_arr[:need].tolist())

    return np.asarray(selected[:target_n], dtype=np.int64)


def sample_negative_calibration(
    prediction_path: Path,
    background_path: Path,
    kml_gdf: gpd.GeoDataFrame,
    *,
    target_n: int,
    negative_classes: Sequence[int],
    exclusion_buffer_m: float,
    grid_size_m: float,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)

    with rasterio.open(background_path) as bg:
        bg_arr = bg.read(1, masked=False)
        bg_nodata = bg.nodata if bg.nodata is not None else OUT_NODATA
        kml_bg = kml_gdf.to_crs(bg.crs)
        geoms = []
        for geom in kml_bg.geometry:
            if geom is None or geom.is_empty:
                continue
            if exclusion_buffer_m != 0.0:
                geom = geom.buffer(exclusion_buffer_m)
            if geom is None or geom.is_empty:
                continue
            geoms.append(geom)
        exclusion_mask = rasterize(
            [(geom, 1) for geom in geoms],
            out_shape=(bg.height, bg.width),
            transform=bg.transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        ).astype(bool)
        eligible = (bg_arr != bg_nodata) & np.isin(bg_arr, list(negative_classes)) & (~exclusion_mask)
        rows, cols = np.where(eligible)
        available = int(rows.size)
        if available == 0:
            return {
                "probs": np.array([], dtype=np.float32),
                "n_available": 0,
                "n_negative": 0,
                "class_counts": {},
            }

        xs, ys = xy(bg.transform, rows, cols, offset="center")
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        selected_local = _sample_spatially_stratified(xs, ys, target_n, grid_size_m, rng)
        rows = rows[selected_local]
        cols = cols[selected_local]
        xs = xs[selected_local]
        ys = ys[selected_local]
        labels = bg_arr[rows, cols].astype(np.int32)

    sampled_probs: List[np.ndarray] = []
    with rasterio.open(prediction_path) as ds:
        prob_band = _get_probability_band(ds)
        coords = np.column_stack([xs, ys])
        chunk_size = 50_000
        for start in range(0, len(coords), chunk_size):
            chunk = coords[start : start + chunk_size]
            values = np.asarray([v[0] for v in ds.sample((tuple(xy_) for xy_ in chunk), indexes=prob_band)], dtype=np.float32)
            nodata = ds.nodata if ds.nodata is not None else PRED_NODATA
            values = np.where(values == nodata, np.nan, values)
            values = np.where((values >= 0.0) & (values <= 1.0), values, np.nan)
            sampled_probs.append(values)

    probs = np.concatenate(sampled_probs) if sampled_probs else np.array([], dtype=np.float32)
    valid = np.isfinite(probs)
    probs = probs[valid]
    labels = labels[valid]

    class_counts = Counter(int(v) for v in labels.tolist())
    return {
        "probs": probs.astype(np.float32),
        "n_available": available,
        "n_negative": int(probs.size),
        "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
    }


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError("--alpha must be between 0 and 1.")
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("No finite conformal scores available.")
    scores.sort()
    rank = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), scores.size)
    return float(scores[rank - 1])


def classify_probability_block(probs: np.ndarray, qhat: float) -> np.ndarray:
    out = np.full(probs.shape, OUT_NODATA, dtype=np.uint8)
    valid = np.isfinite(probs)
    if not np.any(valid):
        return out
    accept_nonwet = probs <= qhat
    accept_wet = probs >= (1.0 - qhat)
    out[valid & accept_nonwet & ~accept_wet] = 0
    out[valid & accept_wet & ~accept_nonwet] = 2
    out[valid & ~((accept_nonwet & ~accept_wet) | (accept_wet & ~accept_nonwet))] = 1
    return out


def classify_confident_wet_block(probs: np.ndarray, wet_threshold: float) -> np.ndarray:
    out = np.full(probs.shape, OUT_NODATA, dtype=np.uint8)
    valid = np.isfinite(probs)
    if not np.any(valid):
        return out
    out[valid & (probs >= wet_threshold)] = 1
    return out


def write_uncertainty_raster(
    prediction_path: Path,
    output_path: Path,
    *,
    qhat: float,
    alpha: float,
) -> Dict[str, int]:
    with rasterio.open(prediction_path) as src:
        prob_band = _get_probability_band(src)
        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "count": 1,
                "dtype": "uint8",
                "nodata": OUT_NODATA,
                "compress": "lzw",
            }
        )
        block_x = min(512, int(src.width))
        block_y = min(512, int(src.height))
        block_x = (block_x // 16) * 16
        block_y = (block_y // 16) * 16
        if block_x >= 16 and block_y >= 16:
            profile.update(
                {
                    "tiled": True,
                    "blockxsize": block_x,
                    "blockysize": block_y,
                }
            )
        else:
            profile["tiled"] = False
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        counts = Counter()

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.set_band_description(1, "conformal_confidence")
            dst.update_tags(
                alpha=str(alpha),
                qhat=str(qhat),
                threshold_low=str(qhat),
                threshold_high=str(1.0 - qhat),
                class_codes="0=confident_non_wet,1=uncertain,2=confident_wet,255=nodata",
            )
            for _, window in src.block_windows(prob_band):
                probs = _read_probability(src, prob_band, window)
                out = classify_probability_block(probs, qhat)
                dst.write(out, 1, window=window)
                unique, n = np.unique(out, return_counts=True)
                counts.update({int(k): int(v) for k, v in zip(unique, n)})

    return {
        "confident_non_wet": int(counts.get(0, 0)),
        "uncertain": int(counts.get(1, 0)),
        "confident_wet": int(counts.get(2, 0)),
        "nodata": int(counts.get(OUT_NODATA, 0)),
    }


def write_confident_wet_raster(
    prediction_path: Path,
    output_path: Path,
    *,
    wet_threshold: float,
    alpha: float,
    qhat: float,
    mode: str,
    extra_tags: Optional[Dict[str, object]] = None,
) -> Dict[str, int]:
    with rasterio.open(prediction_path) as src:
        prob_band = _get_probability_band(src)
        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "count": 1,
                "dtype": "uint8",
                "nodata": OUT_NODATA,
                "compress": "lzw",
            }
        )
        block_x = min(512, int(src.width))
        block_y = min(512, int(src.height))
        block_x = (block_x // 16) * 16
        block_y = (block_y // 16) * 16
        if block_x >= 16 and block_y >= 16:
            profile.update(
                {
                    "tiled": True,
                    "blockxsize": block_x,
                    "blockysize": block_y,
                }
            )
        else:
            profile["tiled"] = False
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        confident_wet = 0
        valid_pixels = 0
        total_pixels = 0

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.set_band_description(1, "conformal_confident_wet_mask")
            tags = {
                "alpha": str(alpha),
                "threshold_wet": str(wet_threshold),
                "class_codes": "1=confident_wet,255=other_or_nodata",
                "conformal_mode": mode,
            }
            if mode == "wet-other":
                tags["qhat_positive"] = str(qhat)
            else:
                tags["qhat_negative"] = str(qhat)
            if extra_tags:
                tags.update({k: str(v) for k, v in extra_tags.items() if v is not None})
            dst.update_tags(**tags)
            for _, window in src.block_windows(prob_band):
                probs = _read_probability(src, prob_band, window)
                valid = np.isfinite(probs)
                out = classify_confident_wet_block(probs, wet_threshold)
                dst.write(out, 1, window=window)
                total_pixels += int(probs.size)
                valid_pixels += int(valid.sum())
                confident_wet += int(np.sum(valid & (probs >= wet_threshold)))

    return {
        "confident_wet": confident_wet,
        "other_valid_masked": int(valid_pixels - confident_wet),
        "input_nodata": int(total_pixels - valid_pixels),
    }


def write_report(report_path: Path, summary: Dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    mode = str(summary.get("output_mode", "three-class"))
    lines = [
        "=" * 70,
        "WET WOODLAND CONFORMAL CONFIDENCE REPORT",
        f"Generated: {summary['generated_at']}",
        "=" * 70,
        "",
        "Inputs",
        "-" * 70,
        f"Score source: {summary['prediction_source']}",
        f"KML: {summary['kml_path']}",
        f"Background raster: {summary['background_raster']}",
        "",
        "Calibration",
        "-" * 70,
        f"Mode: {mode}",
        f"Alpha: {summary['alpha']}",
        f"qhat: {summary['qhat']:.6f}",
        f"Positive samples: {summary['n_positive']:,}",
        "",
        "Output",
        "-" * 70,
        f"Output raster: {summary['output_raster']}",
    ]
    if mode in {"wet-other", "wet-fpr"}:
        lines.extend(
            [
                f"Wet threshold: {summary['threshold_wet']:.6f}",
                f"Confident wet pixels: {summary['output_counts']['confident_wet']:,}",
                f"Other valid pixels masked to nodata: {summary['output_counts']['other_valid_masked']:,}",
                f"Input nodata pixels: {summary['output_counts']['input_nodata']:,}",
            ]
        )
        if mode == "wet-fpr":
            lines.extend(
                [
                    f"Negative samples: {summary['n_negative']:,}",
                    f"Available negative candidates: {summary['n_negative_available']:,}",
                    f"Background class counts used: {summary['negative_class_counts']}",
                    f"Empirical background exceedance: {summary['background_exceedance_rate']:.6f}",
                    f"Positive hit rate at wet threshold: {summary['positive_hit_rate']:.6f}",
                ]
            )
    else:
        lines.extend(
            [
                f"Threshold low: {summary['threshold_low']:.6f}",
                f"Threshold high: {summary['threshold_high']:.6f}",
                f"Negative samples: {summary['n_negative']:,}",
                f"Available negative candidates: {summary['n_negative_available']:,}",
                f"Background class counts used: {summary['negative_class_counts']}",
                f"Confident non-wet pixels: {summary['output_counts']['confident_non_wet']:,}",
                f"Uncertain pixels: {summary['output_counts']['uncertain']:,}",
                f"Confident wet pixels: {summary['output_counts']['confident_wet']:,}",
                f"Nodata pixels: {summary['output_counts']['nodata']:,}",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    default_tiles = repo_root / "data" / "output" / "predictions" / "tiles"
    default_bg = repo_root / "data" / "validation" / "eval_background.tif"
    default_kml = repo_root / "data" / "validation" / "wetwoodlands.kml"
    default_output = repo_root / "data" / "output" / "postprocess" / "wet_woodland_conformal_confidence.tif"
    default_report = repo_root / "data" / "output" / "reports" / "wet_woodland_conformal_confidence.report.txt"

    p = argparse.ArgumentParser(
        description="Create a conformal 3-class confidence raster from a wet woodland probability/suitability score raster, KML positives, and discarded-background negatives."
    )
    p.add_argument("--tiles-dir", default=str(default_tiles), help=f"Prediction tiles directory (default: {default_tiles})")
    p.add_argument("--tiles-glob", default="*.tif", help="Glob for prediction tiles (default: *.tif)")
    p.add_argument(
        "--prediction-raster",
        "--score-raster",
        default=None,
        help="Optional single probability/suitability score raster or VRT instead of tiles. If set, overrides --tiles-dir.",
    )
    p.add_argument("--background-raster", default=str(default_bg), help=f"Discarded background raster (default: {default_bg})")
    p.add_argument("--kml", default=str(default_kml), help=f"KML polygons for wet woodland positives (default: {default_kml})")
    p.add_argument("--output", default=str(default_output), help=f"Output uncertainty raster (default: {default_output})")
    p.add_argument("--report-file", default=str(default_report), help=f"Text report path (default: {default_report})")
    p.add_argument(
        "--output-mode",
        choices=["three-class", "wet-other", "wet-fpr"],
        default="three-class",
        help=(
            "Output calibration mode. "
            "'three-class' uses positives plus negatives to make non-wet / uncertain / wet. "
            "'wet-other' uses positive-only one-sided conformal to make a sparse confident_wet mask "
            "(all other valid pixels are written as nodata). "
            "'wet-fpr' uses held-out negatives to set a sparse confident_wet threshold at "
            "approximately the chosen background false-positive rate. "
            "Default: three-class"
        ),
    )
    p.add_argument("--alpha", type=float, default=0.10, help="Miscoverage level alpha (default: 0.10 for 90%% confidence)")
    p.add_argument(
        "--max-pos-per-polygon",
        type=int,
        default=500,
        help="Maximum positive calibration pixels sampled per polygon. Use 0 for all. Default: 500",
    )
    p.add_argument(
        "--negative-to-positive-ratio",
        type=float,
        default=1.0,
        help="Target negative:positive calibration ratio. Default: 1.0",
    )
    p.add_argument(
        "--negative-classes",
        default="0,1",
        help=(
            "Comma-separated discarded-background classes to use as negatives. "
            "Default: 0,1. For Elapid suitability_eval_background.tif use 0 to keep "
            "only held-out background and exclude sampled training background."
        ),
    )
    p.add_argument(
        "--negative-grid-size-m",
        type=float,
        default=2000.0,
        help="Grid size for spatially stratified negative sampling. Default: 2000",
    )
    p.add_argument(
        "--negative-exclusion-buffer-m",
        type=float,
        default=100.0,
        help="Buffer around KML polygons excluded from negative sampling. Default: 100",
    )
    p.add_argument("--all-touched", action="store_true", default=False, help="Use all_touched=True when rasterizing KML positives")
    p.add_argument("--seed", type=int, default=42, help="Random seed for calibration sampling")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    prediction_raster = Path(args.prediction_raster).expanduser() if args.prediction_raster else None
    tiles_dir = Path(args.tiles_dir).expanduser()
    background_raster = Path(args.background_raster).expanduser()
    kml_path = Path(args.kml).expanduser()
    output_path = Path(args.output).expanduser()
    report_path = Path(args.report_file).expanduser()

    if not kml_path.is_file():
        raise SystemExit(f"KML not found: {kml_path}")
    if not background_raster.is_file():
        raise SystemExit(f"Background raster not found: {background_raster}")

    temp_ctx: Optional[tempfile.TemporaryDirectory[str]] = None
    source_label = ""
    try:
        if prediction_raster is not None:
            if not prediction_raster.is_file():
                raise SystemExit(f"Prediction raster not found: {prediction_raster}")
            source_path = prediction_raster
            source_label = str(prediction_raster)
        else:
            if not tiles_dir.is_dir():
                raise SystemExit(f"Prediction tiles directory not found: {tiles_dir}")
            tiles = _collect_tiles(tiles_dir, args.tiles_glob)
            if not tiles:
                raise SystemExit(f"No prediction tiles found in {tiles_dir} matching {args.tiles_glob}")
            print(f"📄 Found {len(tiles)} prediction tile(s)")
            tile_prob_bands = {p: _detect_probability_band(p) for p in tiles}
            nodata_value = PRED_NODATA
            with rasterio.open(tiles[0]) as ds0:
                if ds0.nodata is not None:
                    nodata_value = float(ds0.nodata)
            temp_ctx = tempfile.TemporaryDirectory(prefix="conformal_confidence_")
            source_path = _build_probability_mosaic(tiles, tile_prob_bands, nodata_value, Path(temp_ctx.name))
            source_label = str(tiles_dir)

        with rasterio.open(source_path) as ds:
            crs = ds.crs
            if crs is None:
                raise SystemExit(f"Prediction source has no CRS: {source_path}")
            prob_band = _get_probability_band(ds)
            print("Conformal Confidence")
            print("=" * 70)
            print(f"Score source:        {source_label}")
            print(f"Score raster:        {source_path}")
            print(f"Score band:          {prob_band}")
            print(f"KML:                 {kml_path}")
            print(f"Background raster:   {background_raster}")
            print(f"Output raster:       {output_path}")
            print(f"Output mode:         {args.output_mode}")
            print(f"Alpha:               {args.alpha}")

        warn_if_calibration_kml_matches_training_metadata(
            kml_path,
            score_raster=Path(source_path),
            background_raster=background_raster,
        )

        kml_gdf = load_kml_polygons(kml_path, crs)
        if kml_gdf.empty:
            raise SystemExit("No polygon geometries found in KML.")
        print(f"KML polygons:        {len(kml_gdf)}")

        pos = sample_positive_calibration(
            Path(source_path),
            kml_gdf,
            max_per_polygon=int(args.max_pos_per_polygon),
            all_touched=bool(args.all_touched),
            seed=int(args.seed),
        )
        n_positive = int(pos["n_positive"])
        if n_positive == 0:
            raise SystemExit("No positive calibration pixels sampled from KML polygons.")
        print(f"Positive samples:    {n_positive:,}")
        pos_probs = np.asarray(pos["probs"], dtype=np.float32)

        n_negative = 0
        neg: Dict[str, object] = {
            "n_available": 0,
            "class_counts": {},
        }
        threshold_low = None
        threshold_high = None
        threshold_wet = None
        background_exceedance_rate = None
        positive_hit_rate = None

        if args.output_mode == "wet-other":
            scores = (1.0 - pos_probs).astype(np.float32)
            qhat = conformal_quantile(scores, float(args.alpha))
            threshold_wet = 1.0 - qhat
            print(f"qhat_positive:       {qhat:.6f}")
            print(f"Wet threshold:       {threshold_wet:.6f}")
            if threshold_wet <= 0.05:
                print(
                    "⚠️  Wet threshold is very low. Even as a sparse mask, this may still "
                    "flag most valid pixels as confident wet."
                )
            output_counts = write_confident_wet_raster(
                Path(source_path),
                output_path,
                wet_threshold=float(threshold_wet),
                qhat=float(qhat),
                alpha=float(args.alpha),
                mode="wet-other",
            )
            print(f"✅ Wrote confident-wet raster: {output_path}")
            print(
                "Class counts:        "
                f"wet={output_counts['confident_wet']:,}  "
                f"other_valid_masked={output_counts['other_valid_masked']:,}"
            )
        else:
            target_neg = max(1, int(round(n_positive * float(args.negative_to_positive_ratio))))
            neg = sample_negative_calibration(
                Path(source_path),
                background_raster,
                kml_gdf,
                target_n=target_neg,
                negative_classes=_parse_negative_classes(args.negative_classes),
                exclusion_buffer_m=float(args.negative_exclusion_buffer_m),
                grid_size_m=float(args.negative_grid_size_m),
                seed=int(args.seed),
            )
            n_negative = int(neg["n_negative"])
            if n_negative == 0:
                raise SystemExit("No negative calibration pixels sampled from discarded background.")
            print(f"Negative samples:    {n_negative:,} (available {int(neg['n_available']):,})")
            print(f"Negative classes:    {neg['class_counts']}")

            neg_probs = np.asarray(neg["probs"], dtype=np.float32)
            if args.output_mode == "wet-fpr":
                qhat = conformal_quantile(neg_probs, float(args.alpha))
                threshold_wet = qhat
                background_exceedance_rate = float(np.mean(neg_probs >= threshold_wet))
                positive_hit_rate = float(np.mean(pos_probs >= threshold_wet))
                print(f"qhat_negative:       {qhat:.6f}")
                print(f"Wet threshold:       {threshold_wet:.6f}")
                print(f"Background exceed.:  {background_exceedance_rate:.6f}")
                print(f"Positive hit rate:   {positive_hit_rate:.6f}")
                if threshold_wet <= 0.05:
                    print(
                        "⚠️  Wet threshold is still very low. This mask may still be broad, "
                        "which means the held-out background also scores wet-like too often."
                    )
                output_counts = write_confident_wet_raster(
                    Path(source_path),
                    output_path,
                    wet_threshold=float(threshold_wet),
                    qhat=float(qhat),
                    alpha=float(args.alpha),
                    mode="wet-fpr",
                    extra_tags={
                        "background_exceedance_rate": background_exceedance_rate,
                        "positive_hit_rate": positive_hit_rate,
                    },
                )
                print(f"✅ Wrote confident-wet raster: {output_path}")
                print(
                    "Class counts:        "
                    f"wet={output_counts['confident_wet']:,}  "
                    f"other_valid_masked={output_counts['other_valid_masked']:,}"
                )
            else:
                y = np.concatenate(
                    [
                        np.ones(pos_probs.shape[0], dtype=np.uint8),
                        np.zeros(neg_probs.shape[0], dtype=np.uint8),
                    ]
                )
                p = np.concatenate([pos_probs, neg_probs]).astype(np.float32)
                scores = np.where(y == 1, 1.0 - p, p).astype(np.float32)
                qhat = conformal_quantile(scores, float(args.alpha))
                threshold_low = qhat
                threshold_high = 1.0 - qhat
                print(f"qhat:                {qhat:.6f}")
                print(f"Thresholds:          low={threshold_low:.6f}  high={threshold_high:.6f}")
                if qhat >= 0.5:
                    print(
                        "⚠️  qhat >= 0.5, so the two-sided conformal regions overlap heavily. "
                        "Expect an almost-all-uncertain map. Consider --output-mode wet-fpr."
                    )

                output_counts = write_uncertainty_raster(
                    Path(source_path),
                    output_path,
                    qhat=float(qhat),
                    alpha=float(args.alpha),
                )
                print(f"✅ Wrote uncertainty raster: {output_path}")
                print(
                    "Class counts:        "
                    f"non-wet={output_counts['confident_non_wet']:,}  "
                    f"uncertain={output_counts['uncertain']:,}  "
                    f"wet={output_counts['confident_wet']:,}"
                )

        summary = {
            "generated_at": datetime.now().isoformat(),
            "output_mode": args.output_mode,
            "prediction_source": source_label,
            "kml_path": str(kml_path),
            "background_raster": str(background_raster),
            "output_raster": str(output_path),
            "alpha": float(args.alpha),
            "qhat": float(qhat),
            "threshold_low": float(threshold_low) if threshold_low is not None else None,
            "threshold_high": float(threshold_high) if threshold_high is not None else None,
            "threshold_wet": float(threshold_wet) if threshold_wet is not None else None,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_negative_available": int(neg["n_available"]),
            "negative_class_counts": neg["class_counts"],
            "background_exceedance_rate": background_exceedance_rate,
            "positive_hit_rate": positive_hit_rate,
            "output_counts": output_counts,
            "polygon_sampling": [
                {
                    "polygon_id": int(poly_id),
                    "available_pixels": int(available),
                    "sampled_pixels": int(sampled),
                }
                for poly_id, available, sampled in pos["polygon_counts"]
            ],
        }
        write_report(report_path, summary)
        json_path = report_path.with_suffix(".json")
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"📝 Wrote report: {report_path}")
        print(f"📝 Wrote summary JSON: {json_path}")
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
