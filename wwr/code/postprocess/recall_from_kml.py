#!/usr/bin/env python3
"""
Recall evaluation: compare tiled wet woodland predictions to ground-truth KML polygons.

Loads a KML file containing wet woodland polygons (shapes only; points are ignored),
builds the prediction raster from tiles (same format as wet_woodland_stats: band 1 =
binary 0/1, band 2 = probabilities, 255 = nodata), and computes recall:

  Recall = TP / (TP + FN) = (ground-truth wet pixels predicted as wet) / (all ground-truth wet pixels)

So we measure: of all known wet woodland area in the KML, what fraction did our
tiles predict as wet? Pixels where the prediction is nodata count as false negatives.

Usage:
  python recall_from_kml.py --kml data/validation/wetwoodlands.kml --tiles-dir data/output/predictions/tiles --outdir data/output/reports
  python recall_from_kml.py --kml ground_truth.kml --wet-woodland-raster data/output/postprocess/wet_woodland_mosaic_hysteresis.tif --outdir data/output/reports [--threshold 0.5]

Note: --outdir is a directory; outputs go inside it. Use a folder name, not a .txt file.
By default this writes a single text report. Use --export-detail-files to also
write the per-polygon CSV and IoU text sidecars.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_erosion
from rasterio.merge import merge as rio_merge
from rasterio.windows import Window, from_bounds, transform as window_transform

# Raster format: 0 = non-wet, 1 = wet, 255 = nodata
NODATA = 255


def _collect_tiles(tiles_dir: str, tiles_glob: str) -> List[str]:
    pattern = os.path.join(tiles_dir, tiles_glob)
    paths = sorted(glob.glob(pattern))
    paths = [p for p in paths if not p.lower().endswith(".aux.xml")]
    return paths


def _tiles_overlapping_bounds(
    tile_paths: List[str],
    bounds: Tuple[float, float, float, float],
) -> List[str]:
    """Return tile paths whose bounding box intersects the given (minx, miny, maxx, maxy).
    Uses rtree for fast indexing when available; otherwise falls back to checking each tile.
    """
    minx, miny, maxx, maxy = bounds
    try:
        import rtree  # type: ignore
    except ImportError:
        # No rtree: check each tile (slower but no extra dependency)
        out = []
        for path in tile_paths:
            try:
                with rasterio.open(path) as ds:
                    b = ds.bounds
                if not (b.right < minx or b.left > maxx or b.top < miny or b.bottom > maxy):
                    out.append(path)
            except Exception:
                continue
        return out

    # Build rtree index: id -> (left, bottom, right, top)
    idx = rtree.index.Index()
    for i, path in enumerate(tile_paths):
        try:
            with rasterio.open(path) as ds:
                b = ds.bounds
            idx.insert(i, (b.left, b.bottom, b.right, b.top))
        except Exception:
            continue
    # Query: intersection with polygon extent
    hits = list(idx.intersection((minx, miny, maxx, maxy)))
    return [tile_paths[i] for i in hits]


def _try_build_vrt(tile_paths: List[str], vrt_path: str) -> bool:
    try:
        from osgeo import gdal  # type: ignore
        gdal.UseExceptions()
    except ImportError:
        return False
    except Exception:
        return False
    try:
        if os.path.exists(vrt_path):
            try:
                os.remove(vrt_path)
            except Exception:
                pass
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=False, srcNodata=255, VRTNodata=255)
        vrt = gdal.BuildVRT(vrt_path, tile_paths, options=vrt_options)
        if vrt is None:
            return False
        vrt.FlushCache()
        vrt = None
        return os.path.exists(vrt_path)
    except Exception:
        return False


def _try_build_vrt_cli(tile_paths: List[str], vrt_path: str) -> bool:
    try:
        if os.path.exists(vrt_path):
            try:
                os.remove(vrt_path)
            except Exception:
                pass
        cmd = ["gdalbuildvrt", "-overwrite", "-srcnodata", "255", "-vrtnodata", "255", vrt_path] + tile_paths
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return proc.returncode == 0 and os.path.exists(vrt_path)
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _merge_tiles_to_tiff(tile_paths: List[str], out_tif: str) -> str:
    if not tile_paths:
        raise ValueError("No tile rasters found to merge.")
    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = rio_merge(datasets, nodata=255)
        if mosaic.shape[0] != 1:
            mosaic = mosaic[0:1, ...]
        arr = mosaic[0].astype(np.uint8)
        crs = datasets[0].crs
        with rasterio.open(
            out_tif, "w",
            driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
            dtype=arr.dtype, crs=crs, transform=transform, nodata=255,
            compress="lzw", tiled=True,
        ) as dst:
            dst.write(arr, 1)
        return out_tif
    finally:
        for ds in datasets:
            ds.close()


def _get_raster_profile(path: str) -> Tuple[str, object, int, int]:
    """Return (crs, transform, width, height) for the raster."""
    with rasterio.open(path) as ds:
        return (
            str(ds.crs) if ds.crs else "",
            ds.transform,
            ds.width,
            ds.height,
        )


def _read_binary_raster(
    path: str,
    threshold: Optional[float] = None,
    window: Optional[Window] = None,
) -> Tuple[np.ndarray, object, str, float]:
    """Returns (binary_array, transform, crs, pixel_area_m2). Values: 0, 1, NODATA.
    If window is given, read only that window; transform is the windowed transform.
    """
    with rasterio.open(path) as ds:
        w = window
        if w is None:
            w = Window(0, 0, ds.width, ds.height)
        nodata_val = getattr(ds, "nodata", None)
        if nodata_val is None:
            nodata_val = NODATA
        if threshold is not None and ds.count >= 2:
            # 2+ bands: threshold applies to band 2
            arr = ds.read(2, window=w, masked=False).astype(np.float32)
            valid = (arr != nodata_val) & (arr >= 0) & (arr <= 1)
            wet = (arr >= threshold) & valid
        elif threshold is not None:
            # 1 band: threshold applies to band 1
            arr = ds.read(1, window=w, masked=False).astype(np.float32)
            valid = (arr != nodata_val) & (arr >= 0) & (arr <= 1)
            wet = (arr >= threshold) & valid
        else:
            arr = ds.read(1, window=w, masked=False)
            if np.issubdtype(arr.dtype, np.integer):
                wet = (arr == 1)
            else:
                wet = (arr > 0.5)
            valid = (arr != nodata_val) if nodata_val is not None else np.ones_like(arr, dtype=bool)
        out = np.where(valid & wet, 1, np.where(valid, 0, NODATA)).astype(np.uint8)
        transform = window_transform(w, ds.transform)
        crs = str(ds.crs) if ds.crs else ""
        res = abs(ds.transform.a) * abs(ds.transform.e)
        return out, transform, crs, res


def load_kml_polygons(kml_path: str, target_crs: Optional[str]) -> gpd.GeoDataFrame:
    """Load KML and keep only polygon geometries (drop points, lines).
    KML from Google Earth is WGS84 (EPSG:4326); we reproject to target_crs to match the tiles.
    """
    gdf = gpd.read_file(kml_path)
    if gdf.empty:
        return gdf
    # Keep Polygon and MultiPolygon only
    geom_types = gdf.geometry.geom_type
    poly_mask = geom_types.isin(("Polygon", "MultiPolygon"))
    gdf = gdf.loc[poly_mask].copy()
    if gdf.empty:
        return gdf
    if target_crs:
        src_crs = gdf.crs
        if src_crs is None:
            # Google Earth KML is WGS84; assume that when CRS is missing
            gdf.set_crs("EPSG:4326", inplace=True)
            src_crs = gdf.crs
        if str(src_crs) != str(target_crs):
            print(f"  Reprojecting KML: {src_crs} → {target_crs}")
            gdf = gdf.to_crs(target_crs)
    return gdf.reset_index(drop=True)


def rasterize_truth(
    gdf: gpd.GeoDataFrame,
    shape: Tuple[int, int],
    transform,
    crs: str,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize polygons to same grid; 1 = inside polygon, 0 = outside."""
    if gdf.empty:
        return np.zeros(shape, dtype=np.uint8)
    gdf = gdf.to_crs(crs)
    mask = rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype=np.uint8,
    )
    return mask


def rasterize_truth_per_polygon(
    gdf: gpd.GeoDataFrame,
    shape: Tuple[int, int],
    transform,
    crs: str,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize with burn value = 1-based index per polygon. 0 = no polygon."""
    if gdf.empty:
        return np.zeros(shape, dtype=np.int32)
    gdf = gdf.to_crs(crs)
    shapes_with_id = [
        (geom, i + 1)
        for i, geom in enumerate(gdf.geometry)
        if geom is not None and not geom.is_empty
    ]
    if not shapes_with_id:
        return np.zeros(shape, dtype=np.int32)
    out = rasterize(
        shapes_with_id,
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype=np.int32,
    )
    return out


def _read_prob_raster(path: str, window: Optional[Window] = None) -> Optional[np.ndarray]:
    """Read band 2 (probabilities 0–1) if present; return float array, nodata = np.nan."""
    with rasterio.open(path) as ds:
        if ds.count < 2:
            return None
        w = window if window is not None else Window(0, 0, ds.width, ds.height)
        arr = ds.read(2, window=w, masked=False).astype(np.float32)
        nodata_val = getattr(ds, "nodata", None)
        if nodata_val is not None:
            arr = np.where(arr == nodata_val, np.nan, arr)
        arr = np.where((arr >= 0) & (arr <= 1), arr, np.nan)
        return arr


def _read_binary_and_prob(
    path: str,
    threshold: Optional[float] = None,
    window: Optional[Window] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], object, str, float]:
    """Read binary pred and prob (band 2) in one open. Returns (pred, prob, transform, crs, pixel_area_m2)."""
    w = window if window is not None else Window(0, 0, 0, 0)  # will be replaced inside open
    with rasterio.open(path) as ds:
        w = window if window is not None else Window(0, 0, ds.width, ds.height)
        nodata_val = getattr(ds, "nodata", None) or NODATA
        if threshold is not None and ds.count >= 2:
            # 2+ bands: threshold applies to band 2 (probabilities)
            arr = ds.read(2, window=w, masked=False).astype(np.float32)
            valid = (arr != nodata_val) & (arr >= 0) & (arr <= 1)
            wet = (arr >= threshold) & valid
            pred = np.where(valid & wet, 1, np.where(valid, 0, NODATA)).astype(np.uint8)
            prob = np.where(valid, arr, np.nan).astype(np.float32)
        elif threshold is not None and ds.count == 1:
            # 1 band only: threshold applies to band 1 (probabilities)
            arr1 = ds.read(1, window=w, masked=False).astype(np.float32)
            valid = (arr1 != nodata_val) & (arr1 >= 0) & (arr1 <= 1)
            wet = (arr1 >= threshold) & valid
            pred = np.where(valid & wet, 1, np.where(valid, 0, NODATA)).astype(np.uint8)
            prob = np.where(valid, arr1, np.nan).astype(np.float32)
        else:
            if ds.count >= 2:
                arr1, arr2 = ds.read([1, 2], window=w, masked=False)
            else:
                arr1 = ds.read(1, window=w, masked=False)
                arr2 = None
            valid = (arr1 != nodata_val) if nodata_val is not None else np.ones_like(arr1, dtype=bool)
            if np.issubdtype(arr1.dtype, np.integer):
                wet = (arr1 == 1)
            else:
                wet = (arr1 > 0.5)
            pred = np.where(valid & wet, 1, np.where(valid, 0, NODATA)).astype(np.uint8)
            if arr2 is not None:
                prob = np.where((arr2 != nodata_val) & (arr2 >= 0) & (arr2 <= 1), arr2.astype(np.float32), np.nan).astype(np.float32)
            else:
                prob = None
        transform = window_transform(w, ds.transform)
        crs = str(ds.crs) if ds.crs else ""
        res = abs(ds.transform.a) * abs(ds.transform.e)
        return pred, prob, transform, crs, res


# Block size for streaming read (only load tiles this size, extract polygon pixels)
_STREAM_TILE = 2048


def _extract_polygon_pixels_streaming(
    path: str,
    bbox_window: Window,
    full_transform,
    gdf: gpd.GeoDataFrame,
    crs: str,
    all_touched: bool,
    threshold: Optional[float],
    eroded_bbox: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, float, str, bool]:
    """Read raster in blocks; keep only pixels inside polygons (or inside eroded mask if given). Returns (pred_1d, prob_1d, poly_id_1d, pixel_area_m2, crs, has_prob)."""
    pred_list: List[np.ndarray] = []
    prob_list: List[np.ndarray] = []
    poly_list: List[np.ndarray] = []
    with rasterio.open(path) as ds:
        nodata_val = getattr(ds, "nodata", None) or NODATA
        has_prob = ds.count >= 2
        w = bbox_window
        for row in range(0, int(w.height), _STREAM_TILE):
            for col in range(0, int(w.width), _STREAM_TILE):
                tw = min(_STREAM_TILE, int(w.width) - col)
                th = min(_STREAM_TILE, int(w.height) - row)
                if tw < 1 or th < 1:
                    continue
                sub = Window(w.col_off + col, w.row_off + row, tw, th)
                tile_transform = window_transform(sub, ds.transform)
                truth_per_poly_tile = rasterize_truth_per_polygon(gdf, (th, tw), tile_transform, crs, all_touched)
                if eroded_bbox is not None:
                    eroded_tile = eroded_bbox[row : row + th, col : col + tw]
                    mask = (eroded_tile.ravel() == 1)
                else:
                    mask = (truth_per_poly_tile > 0).ravel()
                n = int(np.sum(mask))
                if n == 0:
                    continue
                if threshold is not None and has_prob:
                    # 2+ bands: threshold applies to band 2
                    arr = ds.read(2, window=sub, masked=False).astype(np.float32)
                    valid = (arr != nodata_val) & (arr >= 0) & (arr <= 1)
                    wet = (arr >= threshold) & valid
                    pred_tile = np.where(valid & wet, 1, np.where(valid, 0, NODATA)).astype(np.uint8)
                    prob_tile = np.where(valid, arr, np.nan).astype(np.float32)
                elif threshold is not None and not has_prob:
                    # 1 band only: threshold applies to band 1
                    arr1 = ds.read(1, window=sub, masked=False).astype(np.float32)
                    valid = (arr1 != nodata_val) & (arr1 >= 0) & (arr1 <= 1)
                    wet = (arr1 >= threshold) & valid
                    pred_tile = np.where(valid & wet, 1, np.where(valid, 0, NODATA)).astype(np.uint8)
                    prob_tile = np.where(valid, arr1, np.nan).astype(np.float32)
                else:
                    arr1 = ds.read(1, window=sub, masked=False)
                    valid = (arr1 != nodata_val) if nodata_val is not None else np.ones_like(arr1, dtype=bool)
                    wet = (arr1 == 1) if np.issubdtype(arr1.dtype, np.integer) else (arr1 > 0.5)
                    pred_tile = np.where(valid & wet, 1, np.where(valid, 0, NODATA)).astype(np.uint8)
                    if has_prob:
                        arr2 = ds.read(2, window=sub, masked=False).astype(np.float32)
                        prob_tile = np.where((arr2 != nodata_val) & (arr2 >= 0) & (arr2 <= 1), arr2, np.nan).astype(np.float32)
                    else:
                        prob_tile = None
                pred_list.append(pred_tile.ravel()[mask])
                poly_list.append(truth_per_poly_tile.ravel()[mask].astype(np.int32))
                if prob_tile is not None:
                    prob_list.append(prob_tile.ravel()[mask])
        pixel_area_m2 = abs(ds.transform.a) * abs(ds.transform.e)
        out_crs = str(ds.crs) if ds.crs else ""
    pred_1d = np.concatenate(pred_list) if pred_list else np.array([], dtype=np.uint8)
    poly_id_1d = np.concatenate(poly_list) if poly_list else np.array([], dtype=np.int32)
    prob_1d = np.concatenate(prob_list) if prob_list else None
    return pred_1d, prob_1d, poly_id_1d, pixel_area_m2, out_crs, has_prob


def compute_recall(pred: np.ndarray, truth: np.ndarray) -> Tuple[float, int, int, int]:
    """Recall = TP / (TP + FN). truth and pred: 1 = wet, 0 = not, pred 255 = nodata."""
    truth_wet = (truth == 1)
    pred_wet = (pred == 1)
    tp = int(np.sum(truth_wet & pred_wet))
    fn = int(np.sum(truth_wet & ~pred_wet))  # includes where pred is nodata
    total_truth = int(np.sum(truth_wet))
    recall = (tp / total_truth) if total_truth > 0 else float("nan")
    return recall, tp, fn, total_truth


def compute_iou_polygon_areas_only(pred: np.ndarray, truth: np.ndarray) -> Tuple[float, int, int]:
    """IoU over polygon areas only: intersection/union = TP/(TP+FN). Background (pixels outside polygons) ignored."""
    truth_wet = (truth == 1)
    pred_wet = (pred == 1)
    intersection = int(np.sum(truth_wet & pred_wet))
    union = int(np.sum(truth_wet))  # polygon pixels only (TP+FN)
    iou = (intersection / union) if union > 0 else float("nan")
    return iou, intersection, union


def polygon_mean_prediction(prob: np.ndarray, poly_mask: np.ndarray) -> Tuple[float, float]:
    """Mean and median prediction (band 2 prob) over pixels inside polygon; exclude nan."""
    vals = prob[poly_mask]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.median(vals))


def main(argv: Optional[List[str]] = None) -> int:
    tow_root = Path(__file__).resolve().parents[2]
    default_kml = tow_root / "data" / "validation" / "wetwoodlands.kml"
    default_wet_raster = tow_root / "data" / "output" / "postprocess" / "wet_woodland_mosaic_hysteresis.tif"
    default_outdir = tow_root / "data" / "output" / "reports"
    parser = argparse.ArgumentParser(
        description="Compute recall of tiled wet woodland predictions against KML ground-truth polygons.",
    )
    parser.add_argument(
        "--kml",
        default=str(default_kml),
        help=f"Path to KML file with wet woodland polygons (shapes only) (default: {default_kml})",
    )
    parser.add_argument("--tiles-dir", help="Directory containing prediction GeoTIFF tiles")
    parser.add_argument("--tiles-glob", default="*.tif", help="Glob for tiles (default: *.tif)")
    parser.add_argument(
        "--wet-woodland-raster",
        help=(
            "Single raster instead of tiles (band 1 binary, band 2 probs). "
            f"If omitted and --tiles-dir is not provided, defaults to {default_wet_raster}"
        ),
    )
    parser.add_argument("--threshold", type=float, default=None, help="Use band 2 with this threshold (0–1)")
    parser.add_argument(
        "--outdir",
        default=str(default_outdir),
        help=f"Output directory for recall outputs (default: {default_outdir})",
    )
    parser.add_argument("--prefix", default="recall_kml", help="Output filename prefix")
    parser.add_argument(
        "--export-detail-files",
        action="store_true",
        default=False,
        help="Also write per-polygon CSV and IoU text files (default: report only)",
    )
    parser.add_argument("--all-touched", action="store_true", default=False, help="Include pixels that touch polygon edges (default: center-only)")
    parser.add_argument("--no-all-touched", action="store_false", dest="all_touched", help="Only pixels whose center is inside polygon (default; true 10m resolution)")
    parser.add_argument("--erode-pixels", type=int, default=0, metavar="N", help="Erode ground-truth by N pixels to exclude edge pixels from recall/IoU (default: 0)")
    parser.add_argument("--masked", action="store_true", default=True, help="Stream in tiles and keep only polygon pixels (default; lower memory)")
    parser.add_argument("--no-masked", action="store_false", dest="masked", help="Load full extent window (higher memory, faster for small extents)")
    args = parser.parse_args(argv)

    if args.outdir.strip().lower().endswith(".txt"):
        print("Note: --outdir should be a directory. Creating directory with that name.")
    os.makedirs(args.outdir, exist_ok=True)
    temp_ctx: Optional[tempfile.TemporaryDirectory[str]] = None
    report_path = os.path.join(args.outdir, f"{args.prefix}_report.txt")
    csv_path = os.path.join(args.outdir, f"{args.prefix}_per_polygon.csv")
    iou_txt_path = os.path.join(args.outdir, f"{args.prefix}_polygon_iou.txt")
    print("Recall from KML")
    print("=" * 60)
    print(f"KML:              {args.kml}")
    print(f"Output dir:       {args.outdir}")
    print(f"Report file:      {report_path}")
    if args.export_detail_files:
        print(f"Per-polygon CSV:  {csv_path}")
        print(f"Polygon IoU txt:  {iou_txt_path}")

    # Resolve prediction raster
    wet_path: Optional[str] = args.wet_woodland_raster
    if wet_path is None and not args.tiles_dir:
        if default_wet_raster.exists():
            wet_path = str(default_wet_raster)
        else:
            raise SystemExit(
                "No prediction source provided and default hysteresis raster not found at "
                f"{default_wet_raster}. Provide --wet-woodland-raster or --tiles-dir."
            )

    if args.tiles_dir and args.wet_woodland_raster is None:
        tiles = _collect_tiles(args.tiles_dir, args.tiles_glob)
        if not tiles:
            raise SystemExit(f"No tiles in {args.tiles_dir} matching {args.tiles_glob}")
        # Get CRS from first tile so we can load KML and query overlapping tiles
        with rasterio.open(tiles[0]) as ds:
            tile_crs = str(ds.crs) if ds.crs else ""
        print(f"Loading KML polygons from {args.kml}")
        gdf = load_kml_polygons(args.kml, target_crs=tile_crs)
        if gdf.empty:
            raise SystemExit("No polygon geometries found in KML (only shapes are used; points/lines ignored).")
        print(f"  Loaded {len(gdf)} polygon(s)")
        # Only load tiles that overlap the polygon extent (rtree for fast indexing)
        poly_bounds = gdf.total_bounds
        overlapping_tiles = _tiles_overlapping_bounds(tiles, poly_bounds)
        if not overlapping_tiles:
            raise SystemExit("No prediction tiles overlap the polygon extent. Check CRS and extent.")
        print(f"  Tiles overlapping polygons: {len(overlapping_tiles)} / {len(tiles)}")
        temp_ctx = tempfile.TemporaryDirectory(prefix=f"{args.prefix}_")
        scratch_dir = temp_ctx.name
        vrt_path = os.path.join(scratch_dir, f"{args.prefix}_pred.vrt")
        if _try_build_vrt(overlapping_tiles, vrt_path) or _try_build_vrt_cli(overlapping_tiles, vrt_path):
            wet_path = vrt_path
        else:
            merged_path = os.path.join(scratch_dir, f"{args.prefix}_pred_merged.tif")
            wet_path = _merge_tiles_to_tiff(overlapping_tiles, merged_path)
    else:
        # Single raster: load KML after we have CRS from raster
        wet_path = wet_path
        if wet_path is None:
            raise SystemExit("Provide --tiles-dir or --wet-woodland-raster.")
        if not os.path.exists(wet_path):
            raise SystemExit(f"Prediction raster not found: {wet_path}")
        crs, full_transform, width, height = _get_raster_profile(wet_path)
        print(f"Loading KML polygons from {args.kml}")
        gdf = load_kml_polygons(args.kml, target_crs=crs)
        if gdf.empty:
            raise SystemExit("No polygon geometries found in KML (only shapes are used; points/lines ignored).")
        print(f"  Loaded {len(gdf)} polygon(s)")

    # Raster profile (for tiles path we haven't read it yet)
    if args.tiles_dir and args.wet_woodland_raster is None:
        crs, full_transform, width, height = _get_raster_profile(wet_path)

    # Bbox of polygon extent (for reading)
    minx, miny, maxx, maxy = gdf.total_bounds
    res = abs(full_transform.a)
    minx -= res
    miny -= res
    maxx += res
    maxy += res
    bbox_window = from_bounds(minx, miny, maxx, maxy, full_transform)
    bbox_window = bbox_window.intersection(Window(0, 0, width, height))
    if bbox_window.width < 1 or bbox_window.height < 1:
        raise SystemExit("Polygons do not overlap the prediction raster.")
    extent_px = int(bbox_window.width) * int(bbox_window.height)
    use_masked = args.masked or (extent_px > 1_000_000)

    mean_threshold = args.threshold if args.threshold is not None else 0.5
    name_col = next((c for c in gdf.columns if c.lower() == "name"), None)
    if name_col is None:
        name_col = next((c for c in gdf.columns if "name" in c.lower()), None)
    if name_col is not None:
        polygon_names = [str(gdf.iloc[j][name_col]).strip() or f"polygon_{j+1}" for j in range(len(gdf))]
    else:
        polygon_names = [f"polygon_{j+1}" for j in range(len(gdf))]

    if use_masked:
        eroded_bbox = None
        if args.erode_pixels > 0:
            bbox_transform = window_transform(bbox_window, full_transform)
            truth_per_poly_bbox = rasterize_truth_per_polygon(
                gdf, (int(bbox_window.height), int(bbox_window.width)), bbox_transform, crs, args.all_touched
            )
            truth_bbox = (truth_per_poly_bbox > 0).astype(np.uint8)
            eroded_bbox = binary_erosion(truth_bbox.astype(bool), iterations=args.erode_pixels).astype(np.uint8)
        print(f"  Extent bbox: {int(bbox_window.width):,} x {int(bbox_window.height):,} px (full raster: {width} x {height})")
        print(f"  Streaming in tiles (masked): keeping only pixels inside polygons (~{extent_px:,} px in bbox)")
        if args.erode_pixels > 0:
            n_interior = int(np.sum(eroded_bbox))
            print(f"  Eroded ground-truth by {args.erode_pixels} px (interior only): {n_interior:,} pixels")
        pred_1d, prob_1d, poly_id_1d, pixel_area_m2, crs, has_prob = _extract_polygon_pixels_streaming(
            wet_path, bbox_window, full_transform, gdf, crs, args.all_touched, args.threshold, eroded_bbox=eroded_bbox
        )
        total_truth_px = len(pred_1d)
        total_truth_ha = total_truth_px * pixel_area_m2 / 10000.0
        res_m = (pixel_area_m2 ** 0.5) if pixel_area_m2 > 0 else 0
        vector_area_ha = float(gdf.geometry.area.sum()) / 10000.0
        print(f"  Polygon pixels extracted: {total_truth_px:,} ({total_truth_ha:.2f} ha)")
        print(f"  Resolution: {res_m:.0f} m (pixel area {pixel_area_m2:.0f} m²)")
        if not args.all_touched:
            print(f"  (center-only; vector area {vector_area_ha:.2f} ha for comparison)")
        if has_prob and prob_1d is not None:
            print(f"  Band 2 (probabilities) loaded for mean/median")

        tp = int(np.sum(pred_1d == 1))
        fn = total_truth_px - tp
        recall = (tp / total_truth_px) if total_truth_px > 0 else float("nan")
        iou = recall
        intersection = tp
        union_px = total_truth_px
        tp_ha = tp * pixel_area_m2 / 10000.0
        fn_ha = fn * pixel_area_m2 / 10000.0
        print(f"\nOver polygon areas only (background ignored):")
        print(f"  Recall / IoU: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  TP (predicted wet): {tp:,} px ({tp_ha:.2f} ha)")
        print(f"  FN (missed):        {fn:,} px ({fn_ha:.2f} ha)")
        print(f"  Intersection: {intersection:,} px  Union (polygon px): {union_px:,} px")

        max_id = int(np.max(poly_id_1d)) if len(poly_id_1d) > 0 else 0
        labels = np.arange(1, max_id + 1, dtype=np.int32)
        tot_per = np.bincount(poly_id_1d, minlength=max_id + 1)[1:]
        tp_per = np.bincount(poly_id_1d, weights=(pred_1d == 1).astype(np.int32), minlength=max_id + 1)[1:]
        recall_per = np.full(len(tot_per), np.nan, dtype=np.float64)
        valid = tot_per > 0
        recall_per[valid] = tp_per[valid].astype(np.float64) / tot_per[valid]
        iou_per = recall_per
        detected_any_per = (tp_per > 0).astype(np.int32)

        if prob_1d is not None and len(prob_1d) > 0:
            valid = np.isfinite(prob_1d)
            sum_probs = np.bincount(poly_id_1d, weights=np.where(valid, prob_1d, 0.0), minlength=max_id + 1)[1:]
            count_valid = np.bincount(poly_id_1d, weights=valid.astype(np.float64), minlength=max_id + 1)[1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_per = np.where(count_valid > 0, sum_probs / count_valid, np.nan)
            median_per = np.array([
                float(np.nanmedian(prob_1d[poly_id_1d == i])) if (tot_per[i - 1] > 0 and np.any(np.isfinite(prob_1d[poly_id_1d == i]))) else np.nan
                for i in labels
            ])
            wet_by_mean_per = np.where(np.isnan(mean_per), 0, (mean_per >= mean_threshold).astype(np.int32))
        else:
            mean_per = np.full(len(labels), np.nan, dtype=np.float64)
            median_per = np.full(len(labels), np.nan, dtype=np.float64)
            wet_by_mean_per = np.zeros(len(labels), dtype=np.int32)
        prob = prob_1d  # for later "if prob is not None" checks

        rows = []
        for idx, i in enumerate(labels):
            tot_i = int(tot_per[idx])
            if tot_i == 0:
                continue
            poly_name = polygon_names[i - 1] if (i - 1) < len(polygon_names) else f"polygon_{i}"
            row = {
                "polygon_id": int(i),
                "polygon_name": poly_name,
                "truth_pixels": tot_i,
                "truth_ha": tot_i * pixel_area_m2 / 10000.0,
                "tp_pixels": int(tp_per[idx]),
                "fn_pixels": tot_i - int(tp_per[idx]),
                "recall": float(recall_per[idx]),
                "iou": float(iou_per[idx]),
                "detected_any": int(detected_any_per[idx]),
            }
            if prob_1d is not None:
                row["mean_pred"] = float(mean_per[idx])
                row["median_pred"] = float(median_per[idx])
                row["wet_by_mean"] = int(wet_by_mean_per[idx])
            rows.append(row)
        df_poly = pd.DataFrame(rows)
    else:
        print(f"  Cropping to polygon extent: {int(bbox_window.width):,} x {int(bbox_window.height):,} px (full raster: {width} x {height})")
        print(f"Loading predictions from {wet_path}")
        pred, prob, transform, crs, pixel_area_m2 = _read_binary_and_prob(wet_path, threshold=args.threshold, window=bbox_window)
        shape = pred.shape
        total_truth_px = int(np.sum(pred.shape))  # placeholder, will set below
        res_m = (pixel_area_m2 ** 0.5) if pixel_area_m2 > 0 else 0
        vector_area_ha = float(gdf.geometry.area.sum()) / 10000.0
        print(f"  Shape: {shape}, CRS: {crs}, pixel area: {pixel_area_m2:.2f} m²")
        truth_per_poly = rasterize_truth_per_polygon(gdf, shape, transform, crs, all_touched=args.all_touched)
        truth = (truth_per_poly > 0).astype(np.uint8)
        if args.erode_pixels > 0:
            truth = (binary_erosion(truth.astype(bool), iterations=args.erode_pixels).astype(np.uint8))
            total_truth_px = int(np.sum(truth))
            print(f"  Eroded ground-truth by {args.erode_pixels} px (interior only): {total_truth_px:,} pixels ({total_truth_px * pixel_area_m2 / 10000.0:.2f} ha)")
        else:
            total_truth_px = int(np.sum(truth))
        total_truth_ha = total_truth_px * pixel_area_m2 / 10000.0
        if args.erode_pixels == 0:
            print(f"  Ground-truth: {total_truth_px:,} pixels ({total_truth_ha:.2f} ha)")
        if not args.all_touched:
            print(f"  (center-only; vector area {vector_area_ha:.2f} ha)")

        recall, tp, fn, total_truth = compute_recall(pred, truth)
        iou, intersection, union_px = compute_iou_polygon_areas_only(pred, truth)
        tp_ha = tp * pixel_area_m2 / 10000.0
        fn_ha = fn * pixel_area_m2 / 10000.0
        print(f"\nOver polygon areas only (background ignored):")
        print(f"  Recall / IoU: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  TP: {tp:,} px ({tp_ha:.2f} ha)  FN: {fn:,} px ({fn_ha:.2f} ha)")
        print(f"  Intersection: {intersection:,} px  Union: {union_px:,} px")

        max_id = int(np.max(truth_per_poly))
        labels = np.arange(1, max_id + 1, dtype=np.int32)
        flat_labels = truth_per_poly.ravel()
        interior = (truth.ravel() == 1).astype(np.int32)  # eroded: only interior pixels count
        pred_wet = (pred == 1).ravel()
        tot_per = np.bincount(flat_labels, weights=interior, minlength=max_id + 1)[1:]
        tp_per = np.bincount(flat_labels, weights=((pred_wet.astype(np.int32)) * interior), minlength=max_id + 1)[1:]
        recall_per = np.full(len(tot_per), np.nan, dtype=np.float64)
        valid = tot_per > 0
        recall_per[valid] = tp_per[valid].astype(np.float64) / tot_per[valid]
        iou_per = recall_per
        detected_any_per = (tp_per > 0).astype(np.int32)

        if prob is not None:
            prob_flat = prob.ravel()
            valid = ~np.isnan(prob_flat) & (flat_labels > 0) & (interior > 0)
            sum_probs = np.bincount(flat_labels, weights=np.where(np.isnan(prob_flat), 0.0, prob_flat).astype(np.float64) * interior, minlength=max_id + 1)[1:]
            count_valid = np.bincount(flat_labels, weights=valid.astype(np.float64), minlength=max_id + 1)[1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_per = np.where(count_valid > 0, sum_probs / count_valid, np.nan)
            median_per = np.array([
                float(np.nanmedian(prob[(truth_per_poly == i) & (truth == 1)])) if (tot_per[i - 1] > 0 and np.any(np.isfinite(prob[(truth_per_poly == i) & (truth == 1)]))) else np.nan
                for i in labels
            ])
            wet_by_mean_per = np.where(np.isnan(mean_per), 0, (mean_per >= mean_threshold).astype(np.int32))
        else:
            mean_per = np.full(len(labels), np.nan, dtype=np.float64)
            median_per = np.full(len(labels), np.nan, dtype=np.float64)
            wet_by_mean_per = np.zeros(len(labels), dtype=np.int32)

        rows = []
        for idx, i in enumerate(labels):
            tot_i = int(tot_per[idx])
            if tot_i == 0:
                continue
            poly_name = polygon_names[i - 1] if (i - 1) < len(polygon_names) else f"polygon_{i}"
            row = {
                "polygon_id": int(i),
                "polygon_name": poly_name,
                "truth_pixels": tot_i,
                "truth_ha": tot_i * pixel_area_m2 / 10000.0,
                "tp_pixels": int(tp_per[idx]),
                "fn_pixels": tot_i - int(tp_per[idx]),
                "recall": float(recall_per[idx]),
                "iou": float(iou_per[idx]),
                "detected_any": int(detected_any_per[idx]),
            }
            if prob is not None:
                row["mean_pred"] = float(mean_per[idx])
                row["median_pred"] = float(median_per[idx])
                row["wet_by_mean"] = int(wet_by_mean_per[idx])
            rows.append(row)
        df_poly = pd.DataFrame(rows)

    n_poly = len(df_poly)
    n_total_poly = len(gdf)
    n_excluded = n_total_poly - n_poly if (args.erode_pixels > 0 and n_total_poly >= n_poly) else 0
    n_detected_any = int(df_poly["detected_any"].sum()) if not df_poly.empty else 0
    n_wet_by_mean = int(df_poly["wet_by_mean"].sum()) if "wet_by_mean" in df_poly.columns and not df_poly.empty else 0
    miou = float(df_poly["iou"].mean()) if not df_poly.empty and "iou" in df_poly.columns else float("nan")
    iou_valid = df_poly["iou"].dropna() if not df_poly.empty else pd.Series(dtype=float)
    n_iou_50 = int((iou_valid >= 0.5).sum()) if not iou_valid.empty else 0
    pct_iou_50 = (100.0 * n_iou_50 / n_poly) if n_poly > 0 else float("nan")
    if n_poly > 0:
        if n_excluded > 0:
            print(f"Polygons excluded (no interior after erosion): {n_excluded}. Stats below for {n_poly} polygons.")
        print(f"mIoU (mean IoU over polygons): {miou:.4f}")
        frac_iou50 = n_iou_50 / n_poly if n_poly > 0 else float("nan")
        print(f"  Polygons with IoU ≥ 0.5: {n_iou_50}/{n_poly} ({frac_iou50:.3f})")
        pct_any = 100.0 * n_detected_any / n_poly
        frac_detected = n_detected_any / n_poly if n_poly > 0 else float("nan")
        print(f"  Polygons with ≥1 pixel above threshold (detected): {n_detected_any}/{n_poly} ({frac_detected:.3f})")
        if "wet_by_mean" in df_poly.columns:
            pct_mean = 100.0 * n_wet_by_mean / n_poly
            frac_wet_mean = n_wet_by_mean / n_poly if n_poly > 0 else float("nan")
            print(f"  Polygons with mean pred ≥ {mean_threshold} (wet by mean): {n_wet_by_mean}/{n_poly} ({frac_wet_mean:.3f})")

    # Keep the full per-polygon table in the report so the text output is self-contained.
    report_table = df_poly.copy() if not df_poly.empty else pd.DataFrame()

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Recall evaluation: predictions vs KML ground-truth (interior pixels only)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Setup: polygon areas only; wet = pred >= threshold (band 2). Recall = IoU = TP/(TP+FN).\n")
        if args.erode_pixels > 0:
            f.write(f"Eroded by {args.erode_pixels} px (interior only). ")
            if n_excluded > 0:
                f.write(f"Excluded {n_excluded} polygons (no interior). ")
        f.write(f"Resolution {res_m:.0f} m.\n\n")
        f.write(f"Inputs: KML {args.kml}\n")
        f.write(f"       Raster {wet_path}\n")
        f.write(f"       Threshold {args.threshold if args.threshold is not None else 'band 1 binary'}\n\n")
        f.write("Summary\n")
        f.write("-" * 40 + "\n")
        f.write(f"Ground-truth: {total_truth_px:,} px ({total_truth_ha:.2f} ha) in {n_poly} polygons\n")
        f.write(f"Recall / IoU: {recall:.4f} ({recall*100:.2f}%)\n")
        f.write(f"  TP (predicted wet): {tp:,} px ({tp_ha:.2f} ha)\n")
        f.write(f"  FN (missed):         {fn:,} px ({fn_ha:.2f} ha)\n")
        f.write(f"  Intersection: {intersection:,} px  Union (polygon px): {union_px:,} px\n\n")
        if n_poly > 0:
            f.write(f"mIoU (mean IoU over polygons): {miou:.4f}\n")
            frac_iou50 = n_iou_50 / n_poly if n_poly > 0 else float("nan")
            frac_detected = n_detected_any / n_poly if n_poly > 0 else float("nan")
            frac_wet_mean = n_wet_by_mean / n_poly if n_poly > 0 else float("nan")
            f.write(f"  Polygons with IoU ≥ 0.5: {n_iou_50}/{n_poly} ({frac_iou50:.3f})\n")
            f.write(f"  Polygons with ≥1 pixel above threshold (detected): {n_detected_any}/{n_poly} ({frac_detected:.3f})\n")
            if "wet_by_mean" in df_poly.columns:
                f.write(f"  Polygons with mean pred ≥ {mean_threshold} (wet by mean): {n_wet_by_mean}/{n_poly} ({frac_wet_mean:.3f})\n")
            f.write("\n")
        if not report_table.empty:
            f.write(f"Per-polygon detail ({len(report_table)} polygons)\n")
            f.write("-" * 40 + "\n")
            report_table_fmt = report_table.copy()
            for col in report_table_fmt.columns:
                if report_table_fmt[col].dtype in (np.float64, np.float32):
                    report_table_fmt[col] = report_table_fmt[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
            f.write(report_table_fmt.to_string(index=False) + "\n")
    print(f"\nReport written to {report_path}")

    if args.export_detail_files and not df_poly.empty:
        df_poly.to_csv(csv_path, index=False)
        print(f"Per-polygon CSV: {csv_path}")
        # One line per polygon: polygon_name, IoU (tab-separated)
        names = df_poly["polygon_name"] if "polygon_name" in df_poly.columns else df_poly["polygon_id"].astype(str)
        ious = df_poly["iou"]
        with open(iou_txt_path, "w", encoding="utf-8") as f:
            for name, iou_val in zip(names, ious):
                iou_str = f"{float(iou_val):.6f}" if pd.notna(iou_val) else "nan"
                f.write(f"{name}\t{iou_str}\n")
        print(f"Polygon IoU list: {iou_txt_path}")
    if temp_ctx is not None:
        temp_ctx.cleanup()

    # Compute interior polygon size stats (1-pixel erosion) to characterize validation fragments
    print(f"\n📏 Validation polygon size analysis (1-pixel interior erosion):")
    try:
        # Rasterize each polygon, erode by 1 pixel, count interior pixels
        interior_areas_ha = []
        for idx in range(len(gdf)):
            geom = gdf.geometry.iloc[idx]
            if geom is None or geom.is_empty:
                continue
            # Get bounds for this polygon
            minx_p, miny_p, maxx_p, maxy_p = geom.bounds
            # Add buffer
            minx_p -= res * 2
            miny_p -= res * 2
            maxx_p += res * 2
            maxy_p += res * 2
            # Create small window for this polygon
            poly_window = from_bounds(minx_p, miny_p, maxx_p, maxy_p, full_transform)
            poly_window = poly_window.intersection(Window(0, 0, width, height))
            if poly_window.width < 3 or poly_window.height < 3:
                interior_areas_ha.append(0.0)
                continue
            poly_transform = window_transform(poly_window, full_transform)
            poly_shape = (int(poly_window.height), int(poly_window.width))
            # Rasterize this polygon
            poly_mask = rasterize(
                [(geom, 1)],
                out_shape=poly_shape,
                transform=poly_transform,
                fill=0,
                all_touched=False,
                dtype=np.uint8,
            )
            # Erode by 1 pixel
            eroded = binary_erosion(poly_mask.astype(bool), iterations=1)
            interior_px = int(np.sum(eroded))
            interior_ha = interior_px * pixel_area_m2 / 10000.0
            interior_areas_ha.append(interior_ha)

        interior_areas_ha = np.array(interior_areas_ha)
        n_with_interior = np.sum(interior_areas_ha > 0)
        n_no_interior = len(interior_areas_ha) - n_with_interior

        if n_with_interior > 0:
            median_ha = float(np.median(interior_areas_ha[interior_areas_ha > 0]))
            mean_ha = float(np.mean(interior_areas_ha[interior_areas_ha > 0]))
            min_ha = float(np.min(interior_areas_ha[interior_areas_ha > 0]))
            max_ha = float(np.max(interior_areas_ha[interior_areas_ha > 0]))
            q25_ha = float(np.percentile(interior_areas_ha[interior_areas_ha > 0], 25))
            q75_ha = float(np.percentile(interior_areas_ha[interior_areas_ha > 0], 75))

            print(f"   Polygons with interior (after 1px erosion): {n_with_interior}/{len(gdf)}")
            print(f"   Polygons with no interior (edge-only): {n_no_interior}/{len(gdf)}")
            print(f"   Interior area statistics (ha):")
            print(f"      Median: {median_ha:.3f} ha ({median_ha * 10000:.0f} m²)")
            print(f"      Mean:   {mean_ha:.3f} ha ({mean_ha * 10000:.0f} m²)")
            print(f"      Range:  {min_ha:.3f} - {max_ha:.3f} ha")
            print(f"      IQR:    {q25_ha:.3f} - {q75_ha:.3f} ha")

            # Context for MMU
            mmu_ha = 0.03  # 300 m² = 0.03 ha
            n_below_mmu = np.sum((interior_areas_ha > 0) & (interior_areas_ha < mmu_ha))
            print(f"   Polygons with interior < 300m² MMU: {n_below_mmu}/{n_with_interior}")
        else:
            print(f"   No polygons have interior area after 1px erosion")
    except Exception as e:
        print(f"   (Could not compute interior stats: {e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
