#!/usr/bin/env python3
"""
Render predictions and suitability GeoTIFFs (0–1) as PNGs using Datashader:
fire colormap, white background. Optional hillshade underlay from a DEM (earthpy).
Run from repo root or static/ (paths relative to repo).

Modes:
  direct   – map pixel values 0–1 directly (full resolution, or optional --size).
  density  – coarser grid; each output pixel = mean of overlapping raster cells (0–1).
"""

import argparse
import math
import os
from collections import deque
from pathlib import Path

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
for env_name, subdir in (
    ("NUMBA_CACHE_DIR", "numba_cache"),
    ("MPLCONFIGDIR", "mpl"),
    ("XDG_CACHE_HOME", "xdg_cache"),
):
    os.environ.setdefault(env_name, str(TMP_ROOT / subdir))
    Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
import xarray as xr
import datashader as ds
from datashader import transfer_functions as tf
import colorcet as cc
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform as shapely_transform
try:
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

try:
    import fiona
    FIONA_AVAILABLE = True
except ImportError:
    FIONA_AVAILABLE = False

try:
    import earthpy.spatial as es
    EARTHPY_AVAILABLE = True
except ImportError:
    EARTHPY_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def _tiled_connectivity_filter(
    src,
    read_window,
    out_h: int,
    out_w: int,
    threshold: float,
    min_patch_size: int,
    tile_size: int = 4096,
    overlap: int = 1,
) -> np.ndarray:
    """
    Perform connectivity filtering on large rasters using a tiled approach.

    Uses union-find to merge connected components across tile boundaries.
    Returns a boolean mask of pixels to keep.
    """
    from scipy import ndimage

    # Union-find data structure for merging labels across tiles
    class UnionFind:
        def __init__(self):
            self.parent = {}
            self.size = {}

        def find(self, x):
            if x not in self.parent:
                self.parent[x] = x
                self.size[x] = 0
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px != py:
                # Merge smaller into larger
                if self.size[px] < self.size[py]:
                    px, py = py, px
                self.parent[py] = px
                self.size[px] += self.size[py]

        def add_size(self, x, count):
            root = self.find(x)
            self.size[root] += count

    uf = UnionFind()
    struct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)

    # Track global label assignments: (tile_row, tile_col, local_label) -> global_label
    global_label_counter = [0]
    tile_labels = {}  # (tr, tc) -> local label array

    def get_global_label(tr, tc, local_label):
        key = (tr, tc, local_label)
        if key not in tile_labels:
            global_label_counter[0] += 1
            tile_labels[key] = global_label_counter[0]
        return tile_labels[key]

    n_tiles_y = (out_h + tile_size - 1) // tile_size
    n_tiles_x = (out_w + tile_size - 1) // tile_size
    print(f"    Processing {n_tiles_y}x{n_tiles_x} tiles...")

    # Calculate the scale factor from source to output
    if read_window is not None:
        src_h, src_w = int(read_window.height), int(read_window.width)
        src_y0, src_x0 = int(read_window.row_off), int(read_window.col_off)
    else:
        src_h, src_w = src.height, src.width
        src_y0, src_x0 = 0, 0

    scale_y = src_h / out_h
    scale_x = src_w / out_w

    # First pass: label each tile and record boundary connections
    tile_data = {}
    for tr in range(n_tiles_y):
        for tc in range(n_tiles_x):
            y0 = tr * tile_size
            y1 = min((tr + 1) * tile_size, out_h)
            x0 = tc * tile_size
            x1 = min((tc + 1) * tile_size, out_w)
            tile_h, tile_w = y1 - y0, x1 - x0

            # Calculate source window for this tile
            src_row_start = int(y0 * scale_y) + src_y0
            src_row_end = int(y1 * scale_y) + src_y0
            src_col_start = int(x0 * scale_x) + src_x0
            src_col_end = int(x1 * scale_x) + src_x0
            tile_window = rasterio.windows.Window(
                src_col_start, src_row_start,
                src_col_end - src_col_start, src_row_end - src_row_start
            )

            # Read just this tile, resampled to tile dimensions
            tile = src.read(
                1,
                window=tile_window,
                out_shape=(tile_h, tile_w),
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)

            if src.nodata is not None:
                tile = np.where(tile == src.nodata, np.nan, tile)

            valid = np.isfinite(tile)
            binary = (tile >= threshold) & valid
            labels, num_features = ndimage.label(binary, structure=struct)

            # Store tile labels and count sizes
            tile_data[(tr, tc)] = labels
            for local_label in range(1, num_features + 1):
                gl = get_global_label(tr, tc, local_label)
                count = np.sum(labels == local_label)
                uf.add_size(gl, count)

            # Connect to left neighbor
            if tc > 0 and (tr, tc - 1) in tile_data:
                left_labels = tile_data[(tr, tc - 1)]
                # Right edge of left tile connects to left edge of this tile
                for y in range(labels.shape[0]):
                    left_label = left_labels[y, -1] if y < left_labels.shape[0] else 0
                    this_label = labels[y, 0]
                    if left_label > 0 and this_label > 0:
                        gl_left = get_global_label(tr, tc - 1, left_label)
                        gl_this = get_global_label(tr, tc, this_label)
                        uf.union(gl_left, gl_this)

            # Connect to top neighbor
            if tr > 0 and (tr - 1, tc) in tile_data:
                top_labels = tile_data[(tr - 1, tc)]
                # Bottom edge of top tile connects to top edge of this tile
                for x in range(labels.shape[1]):
                    top_label = top_labels[-1, x] if x < top_labels.shape[1] else 0
                    this_label = labels[0, x]
                    if top_label > 0 and this_label > 0:
                        gl_top = get_global_label(tr - 1, tc, top_label)
                        gl_this = get_global_label(tr, tc, this_label)
                        uf.union(gl_top, gl_this)

    # Compute final sizes for each root
    root_sizes = {}
    for key, gl in tile_labels.items():
        root = uf.find(gl)
        if root not in root_sizes:
            root_sizes[root] = 0
        root_sizes[root] += uf.size.get(gl, 0)

    # Determine which roots to keep
    keep_roots = {root for root, size in root_sizes.items() if size >= min_patch_size}
    kept = len(keep_roots)
    total = len(set(uf.find(gl) for gl in tile_labels.values()))
    print(f"    Kept {kept}/{total} patches with >= {min_patch_size} pixels")

    # Second pass: build output mask
    result = np.zeros((out_h, out_w), dtype=bool)
    for (tr, tc), labels in tile_data.items():
        y0 = tr * tile_size
        y1 = min((tr + 1) * tile_size, out_h)
        x0 = tc * tile_size
        x1 = min((tc + 1) * tile_size, out_w)

        tile_mask = np.zeros_like(labels, dtype=bool)
        for local_label in np.unique(labels):
            if local_label == 0:
                continue
            gl = get_global_label(tr, tc, local_label)
            root = uf.find(gl)
            if root in keep_roots:
                tile_mask |= (labels == local_label)

        result[y0:y1, x0:x1] = tile_mask

    return result


def raster_to_dataarray(
    raster_path: Path,
    band_index: int = 1,
    nodata_to_nan: bool = True,
    max_size: int | None = None,
    target_res_m: float | None = None,
    clip_geom: dict | None = None,
    region_bounds: tuple[float, float, float, float] | None = None,
    connectivity_threshold: float = 0.0,
    min_patch_size: int = 0,
    pre_filter_res_m: float = 0.0,
    pre_filter_threshold: float = 0.0,
    pre_filter_min_patch: int = 0,
    fill_interior_nodata: bool = False,
) -> tuple[xr.DataArray, rasterio.Affine, xr.DataArray]:
    """Load a single-band GeoTIFF into an xarray DataArray with (y, x) coords for Datashader.
    Returns (da, transform, clip_mask_da), where clip_mask_da is 1 inside clip geometry
    (or everywhere if no clip). If target_res_m is set, resample to that map resolution
    first; if max_size is set, an additional decimation may be applied.
    """
    with rasterio.open(raster_path) as src:
        read_window = None
        base_transform = src.transform
        base_h, base_w = src.height, src.width

        if region_bounds is not None:
            left, bottom, right, top = region_bounds
            win = rasterio.windows.from_bounds(
                left,
                bottom,
                right,
                top,
                transform=src.transform,
            )
            win = win.round_offsets().round_lengths()
            full = rasterio.windows.Window(0, 0, src.width, src.height)
            try:
                win = win.intersection(full)
            except Exception as exc:
                raise ValueError(f"Requested test window is outside raster extent: {raster_path}") from exc
            if win.width < 1 or win.height < 1:
                raise ValueError(f"Requested test window is empty for raster: {raster_path}")
            read_window = win
            base_transform = src.window_transform(read_window)
            base_h, base_w = int(read_window.height), int(read_window.width)

        # Pre-filter connectivity at intermediate resolution before final downsampling
        connectivity_mask = None
        if pre_filter_res_m > 0 and pre_filter_threshold > 0 and pre_filter_min_patch > 0:
            from scipy import ndimage
            from scipy.ndimage import zoom

            # Calculate dimensions at pre-filter resolution
            native_res = abs(src.transform.a)
            bounds = src.bounds if read_window is None else rasterio.windows.bounds(read_window, src.transform)
            width_m = bounds[2] - bounds[0]
            height_m = bounds[3] - bounds[1]
            pf_w = max(1, int(np.ceil(width_m / pre_filter_res_m)))
            pf_h = max(1, int(np.ceil(height_m / pre_filter_res_m)))
            total_pixels = pf_h * pf_w

            # Use tiled approach for large arrays (>500M pixels)
            if total_pixels > 500_000_000:
                print(f"  Pre-filtering connectivity at {pre_filter_res_m:.0f}m ({pf_h}x{pf_w}) using tiled approach...")
                connectivity_mask = _tiled_connectivity_filter(
                    src, read_window, pf_h, pf_w,
                    pre_filter_threshold, pre_filter_min_patch
                )
            else:
                print(f"  Pre-filtering connectivity at {pre_filter_res_m:.0f}m ({pf_h}x{pf_w})...")

                # Read at pre-filter resolution
                pf_data = src.read(
                    band_index,
                    window=read_window,
                    out_shape=(pf_h, pf_w),
                    resampling=rasterio.enums.Resampling.average,
                ).astype(np.float32)  # Use float32 to save memory

                if src.nodata is not None:
                    pf_data = np.where(pf_data == src.nodata, np.nan, pf_data)

                # 8-connectivity analysis
                valid = np.isfinite(pf_data)
                binary = (pf_data >= pre_filter_threshold) & valid
                del pf_data  # Free memory
                struct = np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]], dtype=np.uint8)
                labels, num_features = ndimage.label(binary, structure=struct)
                del binary  # Free memory

                if num_features > 0:
                    patch_sizes = ndimage.sum_labels(np.ones_like(labels, dtype=np.uint8), labels, index=np.arange(1, num_features + 1))
                    keep_labels = np.where(patch_sizes >= pre_filter_min_patch)[0] + 1
                    keep_mask = np.isin(labels, keep_labels)
                    del labels  # Free memory
                    kept = len(keep_labels)
                    print(f"  Pre-filter: kept {kept}/{num_features} patches with >= {pre_filter_min_patch} pixels at {pre_filter_res_m:.0f}m")
                    connectivity_mask = keep_mask
                else:
                    connectivity_mask = np.ones((pf_h, pf_w), dtype=bool)

        read_h, read_w = base_h, base_w
        read_resampling = rasterio.enums.Resampling.bilinear
        target_res_applied = False
        if target_res_m and target_res_m > 0:
            bounds = src.bounds if read_window is None else rasterio.windows.bounds(read_window, src.transform)
            width_m = bounds[2] - bounds[0]
            height_m = bounds[3] - bounds[1]
            read_w = max(1, int(np.ceil(width_m / target_res_m)))
            read_h = max(1, int(np.ceil(height_m / target_res_m)))
            native_res_m = float(abs(src.transform.a))
            if target_res_m >= native_res_m:
                read_resampling = rasterio.enums.Resampling.average
            target_res_applied = True
            print(
                f"  Resampling to ~{target_res_m:.0f}m "
                f"({base_h}x{base_w} -> {read_h}x{read_w})"
            )

        if max_size and max(read_h, read_w) > max_size:
            scale = max_size / max(read_h, read_w)
            out_h = max(1, int(read_h * scale))
            out_w = max(1, int(read_w * scale))
            data = src.read(
                band_index,
                window=read_window,
                out_shape=(out_h, out_w),
                resampling=read_resampling,
            )
            t = base_transform * base_transform.scale(base_w / out_w, base_h / out_h)
            h, w = out_h, out_w
        elif target_res_applied:
            data = src.read(
                band_index,
                window=read_window,
                out_shape=(read_h, read_w),
                resampling=read_resampling,
            )
            t = base_transform * base_transform.scale(base_w / read_w, base_h / read_h)
            h, w = read_h, read_w
        else:
            data = src.read(band_index, window=read_window)
            t = base_transform
            h, w = base_h, base_w
        nodata = src.nodata
    if nodata_to_nan and nodata is not None:
        data = np.where(data == nodata, np.nan, data).astype(np.float64)

    # Apply pre-computed connectivity mask (resized to output dimensions)
    if connectivity_mask is not None:
        from scipy.ndimage import zoom
        if connectivity_mask.shape != (h, w):
            mask_resized = zoom(connectivity_mask.astype(np.float32),
                               (h / connectivity_mask.shape[0], w / connectivity_mask.shape[1]),
                               order=0) > 0.5
        else:
            mask_resized = connectivity_mask
        data = np.where(mask_resized, data, np.nan)

    # Connectivity filter: only keep pixels in patches >= min_patch_size
    if min_patch_size > 0 and connectivity_threshold > 0:
        from scipy import ndimage
        valid = np.isfinite(data)
        binary = (data >= connectivity_threshold) & valid
        # 8-connectivity structure
        struct = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], dtype=np.uint8)
        labels, num_features = ndimage.label(binary, structure=struct)
        if num_features > 0:
            patch_sizes = ndimage.sum(binary, labels, index=np.arange(1, num_features + 1))
            # Create mask of pixels in large enough patches
            keep_mask = np.zeros_like(data, dtype=bool)
            for i, size in enumerate(patch_sizes, 1):
                if size >= min_patch_size:
                    keep_mask[labels == i] = True
            # Set pixels not in large patches to NaN
            data = np.where(keep_mask, data, np.nan)
            kept = np.sum(patch_sizes >= min_patch_size)
            print(f"  Connectivity filter: kept {kept}/{num_features} patches with >= {min_patch_size} pixels")

    clip_inside = np.ones((h, w), dtype=np.float32)
    if clip_geom is not None:
        inside = geometry_mask(
            [clip_geom],
            out_shape=(h, w),
            transform=t,
            invert=True,
        )
        clip_inside = inside.astype(np.float32)
        if fill_interior_nodata:
            # Fill nodata holes inside the clip polygon using the in-polygon minimum value.
            inside_valid = inside & np.isfinite(data)
            if inside_valid.any():
                min_inside = float(np.nanmin(data[inside_valid]))
                data = np.where(inside & ~np.isfinite(data), min_inside, data)
        data = np.where(inside, data, np.nan)
    # Pixel-center coordinates (rasterio: row 0 = top, y decreases downward)
    xs = np.arange(w) * t.a + t.c + 0.5 * t.a
    ys = np.arange(h) * t.e + t.f + 0.5 * t.e
    # Flip y so north is up: GeoTIFF has row 0 = north (large y); Datashader/PNG
    # draw row 0 = south, so we need row 0 = south (small y), i.e. y increasing.
    data = np.ascontiguousarray(data[::-1, :])
    clip_inside = np.ascontiguousarray(clip_inside[::-1, :])
    ys = ys[::-1].copy()
    da = xr.DataArray(
        data,
        coords=[("y", ys), ("x", xs)],
        dims=["y", "x"],
    )
    clip_mask_da = xr.DataArray(
        clip_inside,
        coords=[("y", ys), ("x", xs)],
        dims=["y", "x"],
    )
    return da, t, clip_mask_da


def raster_to_density(
    raster_path: Path,
    cell_size_m: float = 1000.0,
    threshold: float = 0.0,
    min_density: float = 0.0,
    normalize_to_max: bool = False,
    clip_geom: dict | None = None,
    region_bounds: tuple[float, float, float | None] | None = None,
    pre_filter_threshold: float = 0.0,
    pre_filter_min_patch: int = 0,
    pre_filter_res_m: float = 100.0,
) -> tuple[xr.DataArray, rasterio.Affine, xr.DataArray]:
    """
    Aggregate a raster to a coarser density grid using GDAL's efficient resampling.

    Each output pixel = mean of input values in that cell (memory efficient).
    For predictions, this creates a density/intensity map at 1km resolution.

    Uses a WarpedVRT to ensure GDAL handles nodata correctly during average
    resampling (nodata pixels are excluded from the mean, not averaged as zeros).
    The VRT also enables GDAL's internal overview machinery for faster reads
    when downsampling large rasters (e.g. 52741x58127 @ 10m -> ~700x900 @ 1km).

    min_density filters out cells where the aggregated mean is below this value,
    which effectively removes cells with only scattered/isolated pixels.

    normalize_to_max: if True, normalize all values so the global max becomes 1.0.
    This highlights relative density differences across the whole map.

    pre_filter_threshold and pre_filter_min_patch: if both > 0, apply 8-connectivity
    filtering at pre_filter_res_m resolution BEFORE final density aggregation.
    Removes isolated patches smaller than min_patch pixels at that resolution.
    The filter result is then used to mask the final density aggregation.

    Returns (density_da, transform, clip_mask_da).
    """
    from rasterio.vrt import WarpedVRT

    with rasterio.open(raster_path) as src:
        native_res = abs(src.transform.a)
        nodata = src.nodata

        # Determine bounds
        if region_bounds is not None:
            left, bottom, right, top = region_bounds
        else:
            left, bottom, right, top = src.bounds

        # Calculate output dimensions for target cell size
        width_m = right - left
        height_m = top - bottom
        out_w = max(1, int(np.ceil(width_m / cell_size_m)))
        out_h = max(1, int(np.ceil(height_m / cell_size_m)))

        # Build a VRT that tells GDAL to treat nodata properly during resampling.
        # This ensures nodata pixels are excluded from the average (not averaged
        # as zeros), and lets GDAL use its internal decimation fast-path.
        vrt_opts = dict(
            crs=src.crs,
            resampling=rasterio.enums.Resampling.average,
        )
        if nodata is not None:
            vrt_opts["src_nodata"] = nodata
            vrt_opts["nodata"] = np.nan

        with WarpedVRT(src, **vrt_opts) as vrt:
            # Get window for region within the VRT
            win = rasterio.windows.from_bounds(
                left, bottom, right, top, transform=vrt.transform,
            )
            win = win.round_offsets().round_lengths()
            full = rasterio.windows.Window(0, 0, vrt.width, vrt.height)
            try:
                win = win.intersection(full)
            except Exception:
                win = full

            # Pre-filter: connectivity at intermediate resolution before density
            connectivity_mask = None
            if pre_filter_threshold > 0 and pre_filter_min_patch > 0:
                from scipy import ndimage

                conn_w = max(1, int(np.ceil(width_m / pre_filter_res_m)))
                conn_h = max(1, int(np.ceil(height_m / pre_filter_res_m)))
                print(f"  Pre-filtering: loading at {pre_filter_res_m:.0f}m ({conn_h}x{conn_w}) for connectivity...")

                conn_data = vrt.read(
                    1,
                    window=win,
                    out_shape=(conn_h, conn_w),
                    resampling=rasterio.enums.Resampling.average,
                ).astype(np.float32)

                # VRT maps nodata -> NaN; also catch any residual nodata
                conn_data[~np.isfinite(conn_data)] = np.nan

                # 8-connectivity (queen's case)
                binary = conn_data >= pre_filter_threshold
                del conn_data
                struct = np.array([[1, 1, 1],
                                   [1, 1, 1],
                                   [1, 1, 1]], dtype=np.uint8)
                labels, num_features = ndimage.label(binary, structure=struct)
                del binary

                if num_features > 0:
                    patch_sizes = ndimage.sum(
                        np.ones(labels.shape, dtype=np.uint8),
                        labels,
                        index=np.arange(1, num_features + 1),
                    )
                    keep_labels = np.where(patch_sizes >= pre_filter_min_patch)[0] + 1
                    keep_mask = np.isin(labels, keep_labels)
                    kept = len(keep_labels)
                    del labels, patch_sizes
                    print(f"  Pre-filter: kept {kept}/{num_features} patches with >= {pre_filter_min_patch} pixels at {pre_filter_res_m:.0f}m")

                    # Resize mask to final density resolution using nearest-neighbor
                    from scipy.ndimage import zoom
                    scale_h = out_h / conn_h
                    scale_w = out_w / conn_w
                    connectivity_mask = zoom(
                        keep_mask.astype(np.float32),
                        (scale_h, scale_w),
                        order=0,
                    ) > 0.5
                    del keep_mask

            # --- Core density read: single GDAL call, no full-res load ---
            # out_shape tells GDAL to resample directly to ~700x900 for 1km cells.
            # With the VRT wrapper, nodata is properly excluded from the average.
            density = vrt.read(
                1,
                window=win,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)

        # VRT already mapped nodata -> NaN; catch any remaining non-finite values
        density[~np.isfinite(density)] = np.nan

        # Apply connectivity mask if computed
        if connectivity_mask is not None:
            if connectivity_mask.shape != density.shape:
                from scipy.ndimage import zoom
                connectivity_mask = zoom(
                    connectivity_mask.astype(np.float32),
                    (density.shape[0] / connectivity_mask.shape[0],
                     density.shape[1] / connectivity_mask.shape[1]),
                    order=0,
                ) > 0.5
            density[~connectivity_mask] = np.nan
            del connectivity_mask

        print(f"  Density aggregation: {native_res:.1f}m -> {out_h}x{out_w} @ {cell_size_m:.0f}m")

    if threshold > 0:
        density[density <= threshold] = np.nan

    # Filter out cells with low density (scattered isolated pixels)
    if min_density > 0:
        density[density < min_density] = np.nan

    # Normalize to global max so highest density = 1.0
    if normalize_to_max:
        global_max = np.nanmax(density)
        if global_max > 0:
            density = density / global_max
            print(f"  Normalized to global max: {global_max:.4f}")

    new_h, new_w = density.shape

    # Build output transform
    out_transform = rasterio.Affine(
        cell_size_m, 0, left,
        0, -cell_size_m, top
    )

    # Build coordinates
    xs = np.arange(new_w) * cell_size_m + left + 0.5 * cell_size_m
    ys = np.arange(new_h) * (-cell_size_m) + top - 0.5 * cell_size_m

    # Clip mask
    clip_inside = np.ones((new_h, new_w), dtype=np.float32)
    if clip_geom is not None:
        inside = geometry_mask(
            [clip_geom],
            out_shape=(new_h, new_w),
            transform=out_transform,
            invert=True,
        )
        clip_inside = inside.astype(np.float32)
        density[~inside] = np.nan

    # Flip for datashader (row 0 = south)
    density = np.ascontiguousarray(density[::-1, :])
    clip_inside = np.ascontiguousarray(clip_inside[::-1, :])
    ys = ys[::-1].copy()

    da = xr.DataArray(
        density,
        coords=[("y", ys), ("x", xs)],
        dims=["y", "x"],
    )
    clip_mask_da = xr.DataArray(
        clip_inside,
        coords=[("y", ys), ("x", xs)],
        dims=["y", "x"],
    )

    print(f"  Density aggregation: {native_res:.1f}m -> {new_h}x{new_w} @ {cell_size_m:.0f}m")
    return da, out_transform, clip_mask_da


def raster_to_hexbin(
    raster_path: Path,
    hex_size_m: float = 1000.0,
    threshold: float = 0.0,
    normalize_to_max: bool = False,
    clip_geom: dict | None = None,
    region_bounds: tuple | None = None,
    output_bounds: tuple | None = None,
    band_index: int = 1,
    read_res_m: float = 100.0,
    out_px_per_hex: int = 4,
) -> tuple:
    """
    Aggregate raster band to pointy-top hexagonal bins at ~hex_size_m flat-to-flat diameter.

    Algorithm (fully vectorised — no Python loops):
      1. Read at read_res_m via GDAL (100 m default → 100× faster than native 10 m).
      2. Assign each cell to its hex using axial cube-coordinate rounding (numpy).
      3. Compute mean per hex with np.bincount — O(N) with no Python overhead.
      4. Rasterise: each output pixel looks up its hex and takes that hex's value.
         At out_px_per_hex=4 each hex is ~8 px in diameter so shapes are visible.

    Returns (da, transform, clip_mask_da) — same API as raster_to_density so the
    datashader / hillshade / decoration pipeline works without further changes.

    hex_size_m    : flat-to-flat hex diameter in metres (1000 → ~0.87 km² per hex).
    read_res_m    : GDAL read resolution in metres (default 100).
    out_px_per_hex: pixels per outer hex radius in the output raster (4 → ~8 px Ø).
    output_bounds : optional (left, bottom, right, top) for the output canvas extent.
                    When larger than the raster (e.g. the DEM extent), areas with no
                    data are NaN so the output matches the hillshade/suitability extent.
    """
    import math
    from rasterio.vrt import WarpedVRT

    sqrt3 = math.sqrt(3)
    R     = hex_size_m / sqrt3   # outer radius (centre → vertex), m
    col_w = sqrt3 * R            # x centre-to-centre spacing, m  # noqa: F841
    row_h = 1.5 * R              # y centre-to-centre spacing, m  # noqa: F841

    # ── 1. Determine read bounds and output bounds ────────────────────────────
    with rasterio.open(raster_path) as src:
        native_res = abs(src.transform.a)
        nodata     = src.nodata
        crs        = src.crs
        raster_bounds = src.bounds

        # Output canvas extent (may be larger than the raster, e.g. DEM extent)
        if output_bounds is not None:
            out_left, out_bottom, out_right, out_top = output_bounds
        elif region_bounds is not None:
            out_left, out_bottom, out_right, out_top = region_bounds
        else:
            out_left, out_bottom, out_right, out_top = raster_bounds

        # Data read extent = intersection of requested region with the raster
        if region_bounds is not None:
            req_l, req_b, req_r, req_t = region_bounds
        else:
            req_l, req_b, req_r, req_t = raster_bounds
        read_left   = max(req_l, raster_bounds.left)
        read_bottom = max(req_b, raster_bounds.bottom)
        read_right  = min(req_r, raster_bounds.right)
        read_top    = min(req_t, raster_bounds.top)
        # Also clamp to output bounds so we never read outside the canvas
        read_left   = max(read_left,   out_left)
        read_bottom = max(read_bottom, out_bottom)
        read_right  = min(read_right,  out_right)
        read_top    = min(read_top,    out_top)

        read_width_m  = read_right - read_left
        read_height_m = read_top   - read_bottom
        read_w = max(1, int(math.ceil(read_width_m  / read_res_m)))
        read_h = max(1, int(math.ceil(read_height_m / read_res_m)))

        vrt_opts: dict = dict(crs=crs, resampling=rasterio.enums.Resampling.average)
        if nodata is not None:
            vrt_opts["src_nodata"] = nodata
            vrt_opts["nodata"]     = np.nan

        with WarpedVRT(src, **vrt_opts) as vrt:
            win  = rasterio.windows.from_bounds(
                read_left, read_bottom, read_right, read_top, vrt.transform)
            win  = win.round_offsets().round_lengths()
            full = rasterio.windows.Window(0, 0, vrt.width, vrt.height)
            try:
                win = win.intersection(full)
            except Exception:
                win = full
            data = vrt.read(
                band_index, window=win,
                out_shape=(read_h, read_w),
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)

    data[~np.isfinite(data)] = np.nan
    if threshold > 0.0:
        data[data <= threshold] = np.nan
    print(f"  Hexbin: read {native_res:.1f} m → {read_h}×{read_w} at {read_res_m:.0f} m/px")

    # ── 2. Pixel centres for the read area ────────────────────────────────────
    # px_y is TOP-DOWN so row 0 = northern edge (high northing).
    actual_rx = read_width_m  / read_w
    actual_ry = read_height_m / read_h
    px_x = read_left + (np.arange(read_w, dtype=np.float64) + 0.5) * actual_rx
    px_y = read_top  - (np.arange(read_h, dtype=np.float64) + 0.5) * actual_ry
    PX, PY = np.meshgrid(px_x, px_y)    # (read_h, read_w); PY[0] near top

    # Relative coords use the OUTPUT origin so hex keys are consistent
    # between data pixels and output grid pixels.
    X_rel = (PX - out_left  ).astype(np.float32)
    Y_rel = (PY - out_bottom).astype(np.float32)

    # ── 3. Assign each pixel to its hex (cube-coordinate rounding) ───────────
    # Pointy-top axial formulae: q = (√3/3·x − 1/3·y)/R,  r = (2/3·y)/R
    inv_R = np.float32(1.0 / R)
    fq = (np.float32(sqrt3 / 3.0) * X_rel - np.float32(1.0 / 3.0) * Y_rel) * inv_R
    fr = np.float32(2.0 / 3.0) * Y_rel * inv_R
    fs = -fq - fr

    qi = np.round(fq).astype(np.int32)
    ri = np.round(fr).astype(np.int32)
    si = np.round(fs).astype(np.int32)

    dq = np.abs(qi.astype(np.float32) - fq)
    dr = np.abs(ri.astype(np.float32) - fr)
    ds = np.abs(si.astype(np.float32) - fs)

    fix_q = (dq > dr) & (dq > ds)
    fix_r = (~fix_q) & (dr > ds)
    qi_f  = np.where(fix_q, -ri - si, qi)
    ri_f  = np.where(fix_r, -qi - si, ri)

    # ── 4. Group pixels by hex and compute mean (np.bincount — zero Python loops)
    valid_mask = np.isfinite(data)
    flat_q   = qi_f[valid_mask].astype(np.int64)
    flat_r   = ri_f[valid_mask].astype(np.int64)
    flat_val = data[valid_mask].astype(np.float64)

    # Encode (q, r) → int64.  England hex range ≈ ±700 q/r so 1.5M shift is safe.
    Q_SHIFT: int = 1_500_000
    R_SHIFT: int = 1_500_000
    MULT:    int = 4_000_000
    hex_key = (flat_q + Q_SHIFT) * MULT + (flat_r + R_SHIFT)

    unique_keys, inv = np.unique(hex_key, return_inverse=True)
    sums   = np.bincount(inv, weights=flat_val)
    counts = np.bincount(inv).astype(np.float64)
    means  = np.where(counts > 0, sums / counts, np.nan).astype(np.float32)

    uq = (unique_keys // MULT - Q_SHIFT).astype(np.int32)  # noqa: F841
    ur = (unique_keys  % MULT - R_SHIFT).astype(np.int32)  # noqa: F841
    del flat_q, flat_r, flat_val, hex_key, sums, counts, inv

    if normalize_to_max:
        valid_m = np.isfinite(means)
        if valid_m.any():
            gmax = float(means[valid_m].max())
            if gmax > 0.0:
                means = np.where(valid_m, means / gmax, np.nan).astype(np.float32)
                print(f"  Hexbin: normalised to global max {gmax:.4f}")

    n_valid = int(np.isfinite(means).sum())
    print(f"  Hexbin: {n_valid:,} filled hexes of {len(uq):,} total")

    # ── 5. Rasterise — output grid covers the full output_bounds ─────────────
    # Each output pixel looks up its hex; pixels outside the data area stay NaN.
    out_width_m  = out_right - out_left
    out_height_m = out_top   - out_bottom
    pixel_m = float(R) / max(1, out_px_per_hex)
    out_w = max(1, int(math.ceil(out_width_m  / pixel_m)))
    out_h = max(1, int(math.ceil(out_height_m / pixel_m)))
    MAX_SIDE = 8000
    if max(out_w, out_h) > MAX_SIDE:
        sc    = MAX_SIDE / float(max(out_w, out_h))
        out_w = max(1, int(out_w * sc))
        out_h = max(1, int(out_h * sc))
        pixel_m = out_width_m / out_w

    actual_ox = out_width_m  / out_w
    actual_oy = out_height_m / out_h
    op_x = out_left + (np.arange(out_w, dtype=np.float64) + 0.5) * actual_ox
    op_y = out_top  - (np.arange(out_h, dtype=np.float64) + 0.5) * actual_oy
    OPX, OPY = np.meshgrid(op_x, op_y)   # (out_h, out_w); OPY[0] near top

    # Relative to OUTPUT origin — same origin used for data pixels above
    OX_rel = (OPX - out_left  ).astype(np.float32)
    OY_rel = (OPY - out_bottom).astype(np.float32)

    ofq = (np.float32(sqrt3 / 3.0) * OX_rel - np.float32(1.0 / 3.0) * OY_rel) * inv_R
    ofr = np.float32(2.0 / 3.0) * OY_rel * inv_R
    ofs = -ofq - ofr

    oqi = np.round(ofq).astype(np.int32)
    ori = np.round(ofr).astype(np.int32)
    osi = np.round(ofs).astype(np.int32)

    odq = np.abs(oqi.astype(np.float32) - ofq)
    odr = np.abs(ori.astype(np.float32) - ofr)
    ods = np.abs(osi.astype(np.float32) - ofs)

    ofix_q = (odq > odr) & (odq > ods)
    ofix_r = (~ofix_q) & (odr > ods)
    oqi_f  = np.where(ofix_q, -ori - osi, oqi).astype(np.int64)
    ori_f  = np.where(ofix_r, -oqi - osi, ori).astype(np.int64)
    del OPX, OPY, OX_rel, OY_rel, ofq, ofr, ofs, oqi, ori, osi, odq, odr, ods

    out_key  = (oqi_f + Q_SHIFT) * MULT + (ori_f + R_SHIFT)
    key_flat = out_key.ravel()
    idx      = np.searchsorted(unique_keys, key_flat).clip(0, len(unique_keys) - 1)
    matched  = unique_keys[idx] == key_flat

    canvas = np.full(out_h * out_w, np.nan, dtype=np.float32)
    canvas[matched] = means[idx[matched]]
    canvas = canvas.reshape(out_h, out_w)
    print(f"  Hexbin: output {out_h}×{out_w} at {pixel_m:.0f} m/px "
          f"(~{2 * R / pixel_m:.1f} px hex diameter)")

    # ── 6. Clip mask and DataArray ────────────────────────────────────────────
    out_transform = rasterio.Affine(pixel_m, 0.0, out_left, 0.0, -pixel_m, out_top)
    clip_inside   = np.ones((out_h, out_w), dtype=np.float32)
    if clip_geom is not None:
        inside      = geometry_mask([clip_geom], out_shape=(out_h, out_w),
                                    transform=out_transform, invert=True)
        clip_inside = inside.astype(np.float32)
        canvas[~inside] = np.nan

    xs = out_left + (np.arange(out_w) + 0.5) * actual_ox
    ys = out_top  - (np.arange(out_h) + 0.5) * actual_oy

    # Flip to datashader convention (row 0 = south)
    canvas      = np.ascontiguousarray(canvas[::-1, :])
    clip_inside = np.ascontiguousarray(clip_inside[::-1, :])
    ys          = ys[::-1].copy()

    da = xr.DataArray(canvas,      coords=[("y", ys), ("x", xs)], dims=["y", "x"])
    cl = xr.DataArray(clip_inside, coords=[("y", ys), ("x", xs)], dims=["y", "x"])
    return da, out_transform, cl


def raster_to_density_connectivity(
    raster_path: Path,
    cell_size_m: float = 500.0,
    prob_threshold: float = 0.15,
    density_threshold: float = 0.02,
    min_patch_size: int = 1,
    connectivity_power: float = 0.3,
    isolated_weight: float = 0.2,
    output_gamma: float = 1.0,
    final_mask: float = 0.0,
    smooth_sigma: float = 0.0,
    clip_geom: dict | None = None,
    region_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[xr.DataArray, rasterio.Affine, xr.DataArray]:
    """
    Combine density aggregation with connectivity weighting.

    1. Aggregate 10m predictions to cell_size_m (e.g., 500m) using mean of values > prob_threshold
    2. Apply 8-connectivity analysis on cells with density > density_threshold
    3. Weight by patch size: isolated cells get isolated_weight, large clusters approach 1.0

    Final value = density × (isolated_weight + (1-isolated_weight) × connectivity_factor)

    Parameters:
    - cell_size_m: aggregation cell size (e.g., 500m)
    - prob_threshold: minimum probability to count in density
    - density_threshold: minimum density to consider "connected" for patch analysis
    - min_patch_size: patches smaller than this are zeroed out
    - connectivity_power: exponent for patch size weighting (lower = less extreme)
    - isolated_weight: base weight for isolated pixels (0.2 = isolated get 20% of their density)
    """
    from scipy import ndimage

    with rasterio.open(raster_path) as src:
        native_res = abs(src.transform.a)

        # Determine bounds
        if region_bounds is not None:
            left, bottom, right, top = region_bounds
        else:
            left, bottom, right, top = src.bounds

        # Calculate output dimensions for target cell size
        width_m = right - left
        height_m = top - bottom
        out_w = max(1, int(np.ceil(width_m / cell_size_m)))
        out_h = max(1, int(np.ceil(height_m / cell_size_m)))

        # Get window for region
        win = rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
        win = win.round_offsets().round_lengths()
        full = rasterio.windows.Window(0, 0, src.width, src.height)
        try:
            win = win.intersection(full)
        except:
            win = full

        # Read directly at target resolution using GDAL's average resampling
        density = src.read(
            1,
            window=win,
            out_shape=(out_h, out_w),
            resampling=rasterio.enums.Resampling.average,
        ).astype(np.float64)

        nodata = src.nodata

    # Apply nodata and probability threshold
    if nodata is not None:
        density = np.where(density == nodata, np.nan, density)
    if prob_threshold > 0:
        # The density represents mean value; values below threshold contribute 0
        # So a cell with density X had X fraction of pixels above threshold
        density = np.where(density > prob_threshold, density, np.nan)

    print(f"  Density aggregation: {native_res:.1f}m → {out_h}x{out_w} @ {cell_size_m:.0f}m")

    # --- Connectivity analysis on density grid ---
    valid = np.isfinite(density)
    binary = (density >= density_threshold) & valid

    # 8-connectivity structure
    struct = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.uint8)

    labels, num_features = ndimage.label(binary, structure=struct)
    print(f"  Found {num_features} connected density patches (density >= {density_threshold})")

    # Calculate patch sizes
    if num_features > 0:
        patch_sizes = ndimage.sum(binary, labels, index=np.arange(1, num_features + 1))

        # Create size map
        size_map = np.zeros_like(density)
        for i, size in enumerate(patch_sizes, 1):
            size_map[labels == i] = size

        # Filter small patches
        if min_patch_size > 1:
            small_mask = size_map < min_patch_size
            size_map[small_mask] = 0
            density[small_mask] = np.nan
            kept = np.sum(patch_sizes >= min_patch_size)
            print(f"  Kept {kept} patches with size >= {min_patch_size} cells")

        # Connectivity weight: isolated_weight for size=1, up to 1.0 for large patches
        # Formula: isolated_weight + (1 - isolated_weight) × (log(size) / log(max_size))^power
        max_size = max(size_map.max(), 1)
        if max_size > 1:
            log_sizes = np.log(np.maximum(size_map, 1))
            log_max = np.log(max_size)
            conn_factor = np.power(log_sizes / log_max, connectivity_power)
        else:
            conn_factor = np.ones_like(size_map)

        weights = np.where(
            size_map > 0,
            isolated_weight + (1.0 - isolated_weight) * conn_factor,
            0.0
        )
    else:
        weights = np.zeros_like(density)

    # Final: density × connectivity weight
    result = np.where(valid & (weights > 0), density * weights, np.nan)

    # Apply output gamma to stretch contrast (gamma < 1 boosts mid-tones)
    if output_gamma != 1.0 and np.nanmax(result) > 0:
        # Normalize to 0-1, apply gamma, then scale back
        rmax = np.nanmax(result)
        result = np.where(np.isfinite(result), np.power(result / rmax, output_gamma) * rmax, np.nan)

    # Hard mask: values below final_mask become NaN (white/transparent)
    if final_mask > 0:
        result = np.where(result >= final_mask, result, np.nan)
        print(f"  Masked values below {final_mask}")

    # Gaussian smoothing to blend sparse pixels
    if smooth_sigma > 0:
        # Replace NaN with 0 for smoothing, then restore NaN where originally invalid
        valid_for_smooth = np.isfinite(result)
        result_filled = np.nan_to_num(result, nan=0.0)
        result_smoothed = ndimage.gaussian_filter(result_filled, sigma=smooth_sigma)
        # Also smooth the mask to get proper blending at edges
        mask_smoothed = ndimage.gaussian_filter(valid_for_smooth.astype(float), sigma=smooth_sigma)
        # Normalize by mask to avoid darkening at edges
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(mask_smoothed > 0.01, result_smoothed / mask_smoothed, np.nan)
        print(f"  Gaussian smoothing sigma={smooth_sigma}")

    # Build output transform and coordinates
    out_transform = rasterio.Affine(cell_size_m, 0, left, 0, -cell_size_m, top)
    xs = np.arange(out_w) * cell_size_m + left + 0.5 * cell_size_m
    ys = np.arange(out_h) * (-cell_size_m) + top - 0.5 * cell_size_m

    # Clip mask
    clip_inside = np.ones((out_h, out_w), dtype=np.float32)
    if clip_geom is not None:
        inside = geometry_mask(
            [clip_geom],
            out_shape=(out_h, out_w),
            transform=out_transform,
            invert=True,
        )
        clip_inside = inside.astype(np.float32)
        result = np.where(inside, result, np.nan)

    # Flip for datashader
    result = np.ascontiguousarray(result[::-1, :])
    clip_inside = np.ascontiguousarray(clip_inside[::-1, :])
    ys = ys[::-1].copy()

    da = xr.DataArray(result, coords=[("y", ys), ("x", xs)], dims=["y", "x"])
    clip_mask_da = xr.DataArray(clip_inside, coords=[("y", ys), ("x", xs)], dims=["y", "x"])

    print(f"  Density+connectivity: isolated_weight={isolated_weight}, power={connectivity_power}")
    return da, out_transform, clip_mask_da


def raster_to_connectivity_weighted(
    raster_path: Path,
    max_size: int | None = None,
    threshold: float = 0.15,
    min_patch_size: int = 1,
    patch_weight_power: float = 0.5,
    clip_geom: dict | None = None,
    region_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[xr.DataArray, rasterio.Affine, xr.DataArray]:
    """
    Weight probability values by patch connectivity (8-connectivity).

    Pixels in larger connected patches get boosted, isolated pixels get reduced.
    Final value = probability × (patch_size ^ patch_weight_power) / max_weight

    Parameters:
    - threshold: minimum probability to consider "connected" (binary threshold)
    - min_patch_size: patches smaller than this are set to 0 (removes isolated pixels)
    - patch_weight_power: exponent for patch size weighting (0.5=sqrt, 1.0=linear, 0.3=cube root)

    Returns (weighted_da, transform, clip_mask_da).
    """
    from scipy import ndimage

    with rasterio.open(raster_path) as src:
        crs = src.crs
        native_transform = src.transform
        nodata = src.nodata

        # Determine bounds and read window
        if region_bounds is not None:
            left, bottom, right, top = region_bounds
            win = rasterio.windows.from_bounds(left, bottom, right, top, transform=native_transform)
            win = win.round_offsets().round_lengths()
            full = rasterio.windows.Window(0, 0, src.width, src.height)
            try:
                win = win.intersection(full)
            except:
                win = full
        else:
            left, bottom, right, top = src.bounds
            win = None

        # Calculate output shape
        if win:
            native_h, native_w = int(win.height), int(win.width)
        else:
            native_h, native_w = src.height, src.width

        # Determine target size
        if max_size and max(native_h, native_w) > max_size:
            scale = max_size / max(native_h, native_w)
            out_h = max(1, int(native_h * scale))
            out_w = max(1, int(native_w * scale))
        else:
            out_h, out_w = native_h, native_w

        # Read at target resolution using nearest resampling to preserve sparse structure
        # (bilinear would blur isolated predictions into continuous low-value field)
        data = src.read(
            1,
            window=win,
            out_shape=(out_h, out_w),
            resampling=rasterio.enums.Resampling.nearest,
        ).astype(np.float64)

        # Build transform for output grid
        if win:
            win_transform = rasterio.windows.transform(win, native_transform)
            pixel_w = (right - left) / out_w
            pixel_h = (top - bottom) / out_h
        else:
            pixel_w = abs(native_transform.a) * native_w / out_w
            pixel_h = abs(native_transform.e) * native_h / out_h
            left, top = native_transform.c, native_transform.f

        out_transform = rasterio.Affine(pixel_w, 0, left, 0, -pixel_h, top)

    # Handle nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    # Create binary mask for connectivity analysis
    valid = np.isfinite(data)
    binary = (data >= threshold) & valid

    # 8-connectivity structure (queen's case)
    struct = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Label connected components
    labels, num_features = ndimage.label(binary, structure=struct)
    print(f"  Found {num_features} connected patches above threshold {threshold}")

    # Calculate patch sizes
    patch_sizes = ndimage.sum(binary, labels, index=np.arange(1, num_features + 1))

    # Create size map (each pixel gets its patch size)
    size_map = np.zeros_like(data)
    for i, size in enumerate(patch_sizes, 1):
        size_map[labels == i] = size

    # Filter small patches
    if min_patch_size > 1:
        small_mask = size_map < min_patch_size
        size_map[small_mask] = 0
        data[small_mask] = np.nan
        kept = np.sum(patch_sizes >= min_patch_size)
        print(f"  Kept {kept} patches with size >= {min_patch_size} pixels")

    # Weight by patch size: prob × (size ^ power) / max_weight
    # This boosts pixels in large patches, reduces isolated ones
    if size_map.max() > 0:
        weights = np.power(size_map, patch_weight_power)
        max_weight = weights.max()
        weights = weights / max_weight  # normalize to 0-1
    else:
        weights = np.zeros_like(size_map)

    # Final weighted value: probability × connectivity_weight
    # Or use: sqrt(prob × weight) to balance them
    weighted = np.where(valid & (size_map > 0), data * weights, np.nan)

    # Build coordinates
    h, w = weighted.shape
    xs = np.arange(w) * pixel_w + left + 0.5 * pixel_w
    ys = np.arange(h) * (-pixel_h) + top - 0.5 * pixel_h

    # Clip mask
    clip_inside = np.ones((h, w), dtype=np.float32)
    if clip_geom is not None:
        inside = geometry_mask(
            [clip_geom],
            out_shape=(h, w),
            transform=out_transform,
            invert=True,
        )
        clip_inside = inside.astype(np.float32)
        weighted = np.where(inside, weighted, np.nan)

    # Flip for datashader (row 0 = south)
    weighted = np.ascontiguousarray(weighted[::-1, :])
    clip_inside = np.ascontiguousarray(clip_inside[::-1, :])
    ys = ys[::-1].copy()

    da = xr.DataArray(
        weighted,
        coords=[("y", ys), ("x", xs)],
        dims=["y", "x"],
    )
    clip_mask_da = xr.DataArray(
        clip_inside,
        coords=[("y", ys), ("x", xs)],
        dims=["y", "x"],
    )

    print(f"  Connectivity weighting: {h}x{w}, power={patch_weight_power}")
    return da, out_transform, clip_mask_da


def _raster_same_grid(path_a: Path, path_b: Path, tol: float = 1e-6) -> bool:
    """True if the two rasters have the same CRS, shape, and transform (within tol)."""
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        if a.crs != b.crs or a.height != b.height or a.width != b.width:
            return False
        ta, tb = a.transform, b.transform
        return (
            abs(ta.a - tb.a) <= tol and abs(ta.b - tb.b) <= tol and abs(ta.c - tb.c) <= tol
            and abs(ta.d - tb.d) <= tol and abs(ta.e - tb.e) <= tol and abs(ta.f - tb.f) <= tol
        )


def load_clip_geometry(
    boundary_path: Path,
    target_crs,
    buffer_m: float = 0.0,
) -> dict:
    """Load and optionally buffer a boundary, returned as GeoJSON geometry in target CRS."""
    if not FIONA_AVAILABLE:
        raise ValueError("Clip boundary requested but fiona is not installed. Install fiona or omit --clip-boundary.")
    with fiona.open(boundary_path) as src:
        src_crs_raw = src.crs_wkt or src.crs
        geoms = [shape(feat["geometry"]) for feat in src if feat.get("geometry") is not None]
    if not geoms:
        raise ValueError(f"No geometries found in clip boundary: {boundary_path}")

    merged = unary_union(geoms)
    if src_crs_raw:
        src_crs = rasterio.crs.CRS.from_user_input(src_crs_raw)
    else:
        src_crs = None

    if src_crs is not None and target_crs is not None and src_crs != target_crs:
        transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
        merged = shapely_transform(transformer.transform, merged)

    if buffer_m != 0:
        merged = merged.buffer(buffer_m)
        if merged.is_empty:
            raise ValueError(
                f"Clip buffer {buffer_m} produced empty geometry. "
                "Use a smaller magnitude for --clip-buffer-m."
            )

    if not merged.is_valid:
        merged = merged.buffer(0)
    return mapping(merged)


def _hillshade_from_slope_aspect(
    slope_deg: np.ndarray,
    aspect_deg: np.ndarray,
    azimuth: float,
    altitude: float,
) -> np.ndarray:
    """Compute hillshade (0–255) from slope and aspect in degrees. Flat / nodata → 0."""
    zenith_deg = 90.0 - altitude
    zenith_rad = np.deg2rad(zenith_deg)
    slope_rad = np.deg2rad(np.nan_to_num(slope_deg, nan=0.0))
    aspect_rad = np.deg2rad(np.nan_to_num(aspect_deg, nan=0.0))
    azimuth_rad = np.deg2rad(azimuth)
    # Hillshade = 255 * (cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect))
    out = 255.0 * (
        np.cos(zenith_rad) * np.cos(slope_rad)
        + np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _parse_float_list(raw: str) -> list[float]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one numeric value.")
    return [float(v) for v in vals]


def _resolve_md_pairs(azimuths: list[float], altitudes: list[float]) -> list[tuple[float, float]]:
    if len(altitudes) == 1 and len(azimuths) > 1:
        altitudes = altitudes * len(azimuths)
    if len(azimuths) != len(altitudes):
        raise ValueError(
            "Multidirectional hillshade requires equal counts of azimuths/altitudes, "
            "or a single altitude value."
        )
    return list(zip(azimuths, altitudes))


def _combine_single_and_multidirectional(
    single_hillshade: np.ndarray,
    slope_deg: np.ndarray | None,
    aspect_deg: np.ndarray | None,
    dem_fill: np.ndarray | None,
    use_multidirectional: bool,
    md_pairs: list[tuple[float, float]],
    md_strength: float,
    md_mode: str,
) -> np.ndarray:
    """Combine single-direction hillshade with multidirectional outline shading."""
    base = single_hillshade.astype(np.float64) / 255.0
    if not use_multidirectional or not md_pairs:
        return single_hillshade

    md_stack = []
    if slope_deg is not None and aspect_deg is not None:
        for az, alt in md_pairs:
            hs = _hillshade_from_slope_aspect(slope_deg, aspect_deg, azimuth=az, altitude=alt)
            md_stack.append(hs.astype(np.float64) / 255.0)
    elif dem_fill is not None:
        for az, alt in md_pairs:
            hs = es.hillshade(dem_fill, azimuth=az, altitude=alt).astype(np.float64) / 255.0
            md_stack.append(hs)

    if not md_stack:
        return single_hillshade

    md = np.mean(np.stack(md_stack, axis=0), axis=0)
    if md_mode == "replace":
        combined = ((1.0 - md_strength) * base) + (md_strength * md)
    else:
        multiplied = base * md  # ImageChops.multiply equivalent for grayscale
        combined = ((1.0 - md_strength) * base) + (md_strength * multiplied)
    return np.clip(combined * 255.0, 0, 255).astype(np.uint8)


def build_multidirectional_terrain(
    dem: np.ndarray,
    x_res: float,
    y_res: float,
    azimuths: list[float] = [45, 135, 225, 315],
    altitude: float = 45,
    gamma: float = 0.7,
    z_factor: float = 1.0,
) -> np.ndarray:
    """
    Build multidirectional hillshade terrain visualization.

    Averages hillshades from 4 directions (45, 135, 225, 315 degrees) to create
    soft outline effect where terrain edges are defined from all angles.

    z_factor: vertical exaggeration factor (1.5 = 50% more pronounced terrain)

    Returns luminance array (0-1) where 0=dark shadows, 1=bright highlights.
    """
    dem_f = dem.astype(np.float64)
    valid = np.isfinite(dem_f)
    if not valid.any():
        return np.ones_like(dem_f)

    dem_fill = np.where(valid, dem_f, np.nanmedian(dem_f[valid]))

    # Compute slope and aspect from gradients
    gy, gx = np.gradient(dem_fill, y_res, x_res)
    # Apply z-factor for terrain exaggeration
    slope_rad = np.arctan(z_factor * np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)

    altitude_rad = np.deg2rad(altitude)

    # Compute hillshade from each azimuth direction
    hillshades = []
    for az in azimuths:
        az_rad = np.deg2rad(az)
        shade = (np.cos(altitude_rad) * np.cos(slope_rad) +
                 np.sin(altitude_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect))
        hillshades.append(np.clip(shade, 0, 1))

    # Average all directions for soft outline effect
    md_mean = np.mean(np.stack(hillshades, axis=0), axis=0)

    # Stretch contrast
    md_lo = np.nanpercentile(md_mean[valid], 2)
    md_hi = np.nanpercentile(md_mean[valid], 98)
    luminance = np.clip((md_mean - md_lo) / max(md_hi - md_lo, 1e-6), 0, 1)

    # Apply gamma for contrast control
    luminance = np.power(luminance, gamma)

    # Mask invalid areas
    luminance = np.where(valid, luminance, np.nan)
    return luminance


def hillshade_dataarray_on_da_grid(
    dem_path: Path,
    da: xr.DataArray,
    extent_crs,
    thematic_transform: rasterio.Affine,
    azimuth: float = 240,
    altitude: float = 30,
    dem_band: int = 1,
    slope_band: int | None = None,
    aspect_band: int | None = None,
    thematic_path: Path | None = None,
    use_multidirectional: bool = False,
    md_pairs: list[tuple[float, float]] | None = None,
    md_strength: float = 0.0,
    md_mode: str = "multiply",
    z_factor: float = 1.0,
) -> xr.DataArray:
    """Build a hillshade DataArray on the exact same grid as the thematic.
    If thematic_path and dem_path share the same grid (same CRS, size, transform), we read
    the DTM at native resolution (no reprojection) for pixel-perfect alignment.
    Otherwise we reproject the DTM using the thematic's transform.
    """
    ny, nx = int(da.sizes["y"]), int(da.sizes["x"])
    xs = da.x.values
    ys = da.y.values
    dst_transform = thematic_transform
    md_pairs = md_pairs or []

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata
        same_grid = (
            src.crs == extent_crs and src.height == ny and src.width == nx
            and all(abs(a - b) <= 1e-6 for a, b in zip(thematic_transform, src_transform))
        )
        if same_grid and slope_band is not None and aspect_band is not None:
            print("  Using native DTM grid (no reprojection) for pixel-perfect overlay")
            slope = src.read(slope_band)
            aspect = src.read(aspect_band)
            slope = np.where(slope == src_nodata, np.nan, slope.astype(np.float64))
            aspect = np.nan_to_num(np.where(aspect == src_nodata, np.nan, aspect.astype(np.float64)), nan=0.0)
            # Apply z-factor to exaggerate terrain (transform slope in degrees)
            if z_factor != 1.0:
                slope_rad = np.deg2rad(slope)
                slope = np.rad2deg(np.arctan(z_factor * np.tan(slope_rad)))
            single = _hillshade_from_slope_aspect(slope, aspect, azimuth=azimuth, altitude=altitude)
            hillshade = _combine_single_and_multidirectional(
                single,
                slope_deg=slope,
                aspect_deg=aspect,
                dem_fill=None,
                use_multidirectional=use_multidirectional,
                md_pairs=md_pairs,
                md_strength=md_strength,
                md_mode=md_mode,
            )
            hillshade = np.ascontiguousarray(hillshade[::-1, :])
            return xr.DataArray(hillshade, coords=[("y", ys), ("x", xs)], dims=["y", "x"])
        dem = src.read(dem_band)
        if slope_band is not None and aspect_band is not None:
            slope = src.read(slope_band)
            aspect = src.read(aspect_band)

    dem = np.where(dem == src_nodata, np.nan, dem.astype(np.float64))
    dest = np.empty((ny, nx), dtype=np.float64)
    reproject(
        dem,
        dest,
        src_transform=src_transform,
        src_crs=dem_crs,
        dst_transform=dst_transform,
        dst_crs=extent_crs,
        resampling=Resampling.bilinear,
    )

    if slope_band is not None and aspect_band is not None:
        slope_dest = np.empty((ny, nx), dtype=np.float64)
        aspect_dest = np.empty((ny, nx), dtype=np.float64)
        reproject(
            slope.astype(np.float64),
            slope_dest,
            src_transform=src_transform,
            src_crs=dem_crs,
            dst_transform=dst_transform,
            dst_crs=extent_crs,
            resampling=Resampling.bilinear,
        )
        reproject(
            aspect.astype(np.float64),
            aspect_dest,
            src_transform=src_transform,
            src_crs=dem_crs,
            dst_transform=dst_transform,
            dst_crs=extent_crs,
            resampling=Resampling.bilinear,
        )
        # Apply z-factor to exaggerate terrain (transform slope in degrees)
        if z_factor != 1.0:
            slope_rad = np.deg2rad(slope_dest)
            slope_dest = np.rad2deg(np.arctan(z_factor * np.tan(slope_rad)))
        single = _hillshade_from_slope_aspect(
            slope_dest, aspect_dest, azimuth=azimuth, altitude=altitude
        )
        hillshade = _combine_single_and_multidirectional(
            single,
            slope_deg=slope_dest,
            aspect_deg=aspect_dest,
            dem_fill=None,
            use_multidirectional=use_multidirectional,
            md_pairs=md_pairs,
            md_strength=md_strength,
            md_mode=md_mode,
        )
    else:
        dem_fill = np.nan_to_num(dest, nan=0.0)
        # Apply z-factor for terrain exaggeration
        if z_factor != 1.0:
            dem_fill = dem_fill * z_factor
        single = es.hillshade(dem_fill, azimuth=azimuth, altitude=altitude).astype(np.uint8)
        hillshade = _combine_single_and_multidirectional(
            single,
            slope_deg=None,
            aspect_deg=None,
            dem_fill=dem_fill,
            use_multidirectional=use_multidirectional,
            md_pairs=md_pairs,
            md_strength=md_strength,
            md_mode=md_mode,
        )

    # Rasterio dest has row 0 = north; da has row 0 = south (y increasing)
    hillshade = np.ascontiguousarray(hillshade[::-1, :])
    return xr.DataArray(
        hillshade,
        coords=[("y", ys), ("x", xs)],
        dims=["y", "x"],
    )


def _apply_nodata(arr: np.ndarray, nodata) -> np.ndarray:
    """Convert nodata to NaN for robust numeric processing."""
    out = arr.astype(np.float64, copy=False)
    if nodata is None:
        return out
    try:
        if np.isnan(nodata):
            return np.where(np.isfinite(out), out, np.nan)
    except TypeError:
        pass
    return np.where(out == nodata, np.nan, out)


def dem_dataarray_on_da_grid(
    dem_path: Path,
    da: xr.DataArray,
    extent_crs,
    thematic_transform: rasterio.Affine,
    dem_band: int = 1,
) -> xr.DataArray:
    """Read/reproject DEM values onto the exact same (y, x) grid as da."""
    ny, nx = int(da.sizes["y"]), int(da.sizes["x"])
    xs = da.x.values
    ys = da.y.values
    dst_transform = thematic_transform

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata
        same_grid = (
            src.crs == extent_crs and src.height == ny and src.width == nx
            and all(abs(a - b) <= 1e-6 for a, b in zip(thematic_transform, src_transform))
        )
        if same_grid:
            dem = _apply_nodata(src.read(dem_band), src_nodata)
            dem = np.ascontiguousarray(dem[::-1, :])
            return xr.DataArray(dem, coords=[("y", ys), ("x", xs)], dims=["y", "x"])
        dem = _apply_nodata(src.read(dem_band), src_nodata)

    dest = np.empty((ny, nx), dtype=np.float64)
    reproject(
        dem,
        dest,
        src_transform=src_transform,
        src_crs=dem_crs,
        dst_transform=dst_transform,
        dst_crs=extent_crs,
        resampling=Resampling.bilinear,
    )
    dest = np.ascontiguousarray(dest[::-1, :])
    return xr.DataArray(dest, coords=[("y", ys), ("x", xs)], dims=["y", "x"])


def resolve_output_dims(
    ny: int,
    nx: int,
    mode: str,
    req_width: int | None,
    req_height: int | None,
    req_size: int | None,
    square_size: bool,
) -> tuple[int, int]:
    """Return output (width, height), preserving raster aspect by default."""
    if req_width is not None and req_height is not None:
        return req_width, req_height

    if req_width is not None:
        out_h = max(1, int(round(req_width * ny / nx)))
        return req_width, out_h

    if req_height is not None:
        out_w = max(1, int(round(req_height * nx / ny)))
        return out_w, req_height

    if req_size is not None:
        if square_size:
            return req_size, req_size
        if nx >= ny:
            out_w = req_size
            out_h = max(1, int(round(req_size * ny / nx)))
        else:
            out_h = req_size
            out_w = max(1, int(round(req_size * nx / ny)))
        return out_w, out_h

    if mode == "density":
        # Default density size when no explicit dimensions are provided.
        default_size = 1200
        if nx >= ny:
            out_w = default_size
            out_h = max(1, int(round(default_size * ny / nx)))
        else:
            out_h = default_size
            out_w = max(1, int(round(default_size * nx / ny)))
        return out_w, out_h

    # Direct mode defaults to native raster size.
    return nx, ny


def apply_pillow_enhancements(
    image: Image.Image,
    contrast: float,
    color: float,
    sharpness: float,
    autocontrast_cutoff: float,
    aa_scale: int,
    smooth_radius: float,
    smooth_restore: int,
    clarity_radius: float,
    clarity_percent: int,
    tone_curve_strength: float,
    grain_amount: float,
) -> Image.Image:
    """Optional visual polish pass using Pillow."""

    def _apply_rgb_preserve_alpha(img: Image.Image, fn) -> Image.Image:
        if "A" in img.getbands():
            alpha = img.getchannel("A")
            rgb = img.convert("RGB")
            rgb_out = fn(rgb)
            out_rgba = rgb_out.convert("RGBA")
            out_rgba.putalpha(alpha)
            return out_rgba
        return fn(img.convert("RGB")) if img.mode != "RGB" else fn(img)

    def _clarity_rgb(rgb: Image.Image) -> Image.Image:
        if clarity_radius <= 0 or clarity_percent <= 0:
            return rgb
        # Apply local contrast to luminance only to avoid color halos.
        ycbcr = rgb.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        y = y.filter(
            ImageFilter.UnsharpMask(
                radius=clarity_radius,
                percent=clarity_percent,
                threshold=2,
            )
        )
        return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")

    def _tone_curve_rgb(rgb: Image.Image) -> Image.Image:
        if tone_curve_strength <= 0:
            return rgb
        x = np.linspace(0.0, 1.0, 256)
        k = 8.0
        s = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
        s = (s - s[0]) / max(1e-9, (s[-1] - s[0]))
        y = np.clip((1.0 - tone_curve_strength) * x + tone_curve_strength * s, 0.0, 1.0)
        lut = np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8).tolist()
        return rgb.point(lut * 3)

    def _grain_rgb(rgb: Image.Image) -> Image.Image:
        if grain_amount <= 0:
            return rgb
        arr = np.asarray(rgb).astype(np.int16)
        h, w = arr.shape[:2]
        noise = np.random.normal(loc=0.0, scale=255.0 * grain_amount, size=(h, w, 1))
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    out = image
    if aa_scale > 1:
        w, h = out.size
        out = out.resize((w * aa_scale, h * aa_scale), Image.Resampling.BICUBIC)
        out = out.resize((w, h), Image.Resampling.LANCZOS)
    if smooth_radius > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
        if smooth_restore > 0:
            out = out.filter(
                ImageFilter.UnsharpMask(
                    radius=max(0.6, smooth_radius * 2.4),
                    percent=smooth_restore,
                    threshold=2,
                )
            )
    if autocontrast_cutoff > 0:
        if "A" in out.getbands():
            alpha = out.getchannel("A")
            rgb = out.convert("RGB")
            rgb = ImageOps.autocontrast(rgb, cutoff=autocontrast_cutoff)
            out = rgb.convert("RGBA")
            out.putalpha(alpha)
        else:
            out = ImageOps.autocontrast(out, cutoff=autocontrast_cutoff)
    out = _apply_rgb_preserve_alpha(out, _clarity_rgb)
    out = _apply_rgb_preserve_alpha(out, _tone_curve_rgb)
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    if color != 1.0:
        out = ImageEnhance.Color(out).enhance(color)
    if sharpness != 1.0:
        out = ImageEnhance.Sharpness(out).enhance(sharpness)
    out = _apply_rgb_preserve_alpha(out, _grain_rgb)
    return out


def style_agg_for_color(
    agg: xr.DataArray,
    value_mask_below: float,
    value_gamma: float,
) -> xr.DataArray:
    """Prepare aggregated values for color mapping (mask low values, optional gamma)."""
    vals = agg.values
    if not np.issubdtype(vals.dtype, np.floating):
        return agg

    out = vals.astype(np.float64, copy=True)
    finite = np.isfinite(out)

    if value_mask_below > 0:
        out = np.where(finite & (out >= value_mask_below), out, np.nan)
        finite = np.isfinite(out)

    if value_gamma != 1.0:
        denom = max(1e-9, 1.0 - value_mask_below)
        norm = np.where(finite, np.clip((out - value_mask_below) / denom, 0, 1), np.nan)
        out = np.where(np.isfinite(norm), np.power(norm, value_gamma), np.nan)

    return xr.DataArray(out, coords=agg.coords, dims=agg.dims)


def _sample_mpl_cmap(name: str, n: int = 256) -> list[str]:
    if not MATPLOTLIB_AVAILABLE:
        raise ValueError(f"Matplotlib is required for colormap '{name}'.")
    cmap = colormaps[name]
    return [to_hex(cmap(i / (n - 1)), keep_alpha=False) for i in range(n)]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def _rgb_to_hex(rgb: np.ndarray) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _interpolate_hex_ramp(anchors: list[str], n: int = 256) -> list[str]:
    """Linearly interpolate anchor hex colors into an n-color ramp."""
    if len(anchors) < 2:
        return anchors
    rgb = np.array([_hex_to_rgb(c) for c in anchors], dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(anchors))
    dst = np.linspace(0.0, 1.0, n)
    out = np.empty((n, 3), dtype=np.float64)
    for ch in range(3):
        out[:, ch] = np.interp(dst, src, rgb[:, ch])
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return [_rgb_to_hex(c) for c in out]


def get_color_ramp(name: str) -> list[str]:
    """Return a list of hex colors for Datashader shade()."""
    if name == "fire":
        return list(cc.fire)
    if name == "viridis":
        return _sample_mpl_cmap("viridis")
    if name == "cividis":
        return _sample_mpl_cmap("cividis")
    if name == "spectral":
        if MATPLOTLIB_AVAILABLE:
            return _sample_mpl_cmap("Spectral")
        # ColorBrewer Spectral 11 fallback (low=red, high=blue)
        return _interpolate_hex_ramp(
            ["#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b", "#ffffbf",
             "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2"]
        )
    if name == "spectral_r":
        if MATPLOTLIB_AVAILABLE:
            return _sample_mpl_cmap("Spectral_r")
        # Inverted Spectral fallback (high=red).
        return _interpolate_hex_ramp(
            ["#5e4fa2", "#3288bd", "#66c2a5", "#abdda4", "#e6f598", "#ffffbf",
             "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#9e0142"]
        )
    if name == "ylgnbu":
        return _sample_mpl_cmap("YlGnBu")
    if name == "ylgnbu_r":
        return _sample_mpl_cmap("YlGnBu_r")
    if name == "rocket":
        # Seaborn-like rocket anchors (dark navy -> magenta -> peach) for strong value contrast.
        return [
            "#03051A",
            "#180F3D",
            "#3B0F70",
            "#641A80",
            "#8C2981",
            "#B73779",
            "#DE4968",
            "#F66E5B",
            "#FE9F6D",
            "#FEC98D",
            "#FCFDBF",
        ]
    if name == "sunset":
        # Beautiful sunset sky: white → cream → gold → orange → coral → magenta → purple → deep blue
        return _interpolate_hex_ramp([
            "#FFFFFF",  # white (low)
            "#FFFEE0",  # cream/light yellow
            "#FFE4A0",  # light gold
            "#FFCC66",  # golden yellow
            "#FFAA44",  # orange
            "#FF7744",  # coral orange
            "#E85555",  # coral red
            "#CC3366",  # magenta
            "#993388",  # purple magenta
            "#6633AA",  # purple
            "#3322AA",  # deep purple blue
            "#1A1155",  # deep night blue (high)
        ])
    if name == "sunset_r":
        # Reversed sunset (deep blue → cream)
        return list(reversed(_interpolate_hex_ramp([
            "#FFFEE0",
            "#FFE4A0",
            "#FFCC66",
            "#FFAA44",
            "#FF7744",
            "#E85555",
            "#CC3366",
            "#993388",
            "#6633AA",
            "#3322AA",
            "#1A1155",
        ])))
    if name == "sunset_helix":
        # Cubehelix-style sunset: white → dark blue → purple → dusty rose → coral
        return _interpolate_hex_ramp([
            "#FFFFFF",  # white (low)
            "#193a71",  # dark blue
            "#484c7a",  # purple-blue
            "#725a84",  # purple
            "#986981",  # dusty rose/mauve
            "#b5667b",  # coral/pink (high)
        ])
    if name == "sunset_helix_r":
        # Reversed: coral → dark blue
        return list(reversed(_interpolate_hex_ramp([
            "#193a71",
            "#484c7a",
            "#725a84",
            "#986981",
            "#b5667b",
        ])))
    if name == "earth":
        # Slate blue → cream → brown
        return _interpolate_hex_ramp([
            "#4f7288",  # slate blue
            "#6c909f",  # light blue-gray
            "#f5e6c4",  # cream
            "#a68c69",  # tan
            "#64504b",  # dark brown
        ])
    if name == "slate":
        # White → cream → mauve-gray → blue-gray → dark slate
        return _interpolate_hex_ramp([
            "#FFFFFF",  # white (low)
            "#d7cdb3",  # cream/beige
            "#b0a5ad",  # light mauve-gray
            "#6c7885",  # gray-blue
            "#546871",  # blue-gray
            "#2f3941",  # dark slate (high)
        ])
    if name == "ember":
        # White → golden → orange → burnt orange → dark red
        return _interpolate_hex_ramp([
            "#FFFFFF",  # white (low)
            "#f4ac54",  # golden yellow/orange
            "#d78140",  # orange
            "#bb562d",  # burnt orange
            "#9f2b19",  # dark red/brown
            "#830106",  # deep red/maroon (high)
        ])
    if name == "heat":
        # White → golden orange → orange → red-orange → bright red
        return _interpolate_hex_ramp([
            "#FFFFFF",  # white (low)
            "#f39131",  # golden orange
            "#f17625",  # orange
            "#ee5e14",  # orange-red
            "#e5451d",  # red-orange
            "#ea1e1e",  # bright red (high)
        ])
    if name == "inferno":
        # White → dark red → very dark → near black
        return _interpolate_hex_ramp([
            "#FFFFFF",  # white (low)
            "#b2060c",  # dark red
            "#880509",  # darker red
            "#5e0507",  # very dark red/brown
            "#340405",  # almost black
            "#090403",  # near black (high)
        ])
    if name == "mako":
        if SEABORN_AVAILABLE:
            return [to_hex(c, keep_alpha=False) for c in sns.color_palette("mako", n_colors=256)]
        # Fallback: colorcet kbc is visually close to mako.
        return list(cc.kbc)
    if name == "mako_r":
        if SEABORN_AVAILABLE:
            ramp = [to_hex(c, keep_alpha=False) for c in sns.color_palette("mako", n_colors=256)]
        else:
            ramp = list(cc.kbc)
        return list(reversed(ramp))
    if name == "mako_viridis":
        if SEABORN_AVAILABLE:
            mako = [to_hex(c, keep_alpha=False) for c in sns.color_palette("mako", n_colors=256)]
        else:
            mako = list(cc.kbc)
        viridis = _sample_mpl_cmap("viridis", n=256)
        return mako[:128] + viridis[128:]
    if name == "wetland":
        # Custom palette: cream → yellow → green → teal → blue
        # Designed for wet woodland suitability maps
        return _interpolate_hex_ramp([
            "#FFFEF5",  # cream/off-white (low)
            "#FEF6B5",  # pale yellow
            "#D9F0A3",  # yellow-green
            "#78C679",  # green
            "#31A354",  # darker green
            "#006837",  # forest green
            "#004529",  # dark teal-green (high)
        ])
    if name == "wetland_blue":
        # Cream → yellow → green → blue variant
        return _interpolate_hex_ramp([
            "#FFFEF5",  # cream (low)
            "#FEF6B5",  # pale yellow
            "#D9F0A3",  # yellow-green
            "#7FCDBB",  # teal
            "#41B6C4",  # cyan-blue
            "#2C7FB8",  # blue
            "#253494",  # dark blue (high)
        ])
    if name == "cream_blue_black":
        # Cream -> blue (mid) -> dark blue -> blue-black
        # Tuned so blue appears around the midpoint of the value range.
        return _interpolate_hex_ramp([
            "#FFF9EE",  # cream (low)
            "#D8EAF8",  # pale blue
            "#74A9CF",  # medium blue (mid-low)
            "#2B8CBE",  # blue (mid)
            "#225EA8",  # deep blue (mid-high)
            "#1D3F7A",  # dark blue
            "#0A1E3F",  # blue-black (high)
        ])
    if name == "cream_cyan_blue":
        # White -> cream -> cyan -> dark blue
        return _interpolate_hex_ramp([
            "#FFFFFF",  # white (low)
            "#FFF6E6",  # warm cream
            "#DDF3F7",  # pale cyan
            "#8ED3E8",  # cyan
            "#3EA4D6",  # blue-cyan
            "#1F6FB3",  # deep blue
            "#0A1E3F",  # dark blue (high)
        ])
    if name == "cubehelix":
        # Seaborn cubehelix - perceptually uniform, good for terrain
        if SEABORN_AVAILABLE:
            return [to_hex(c, keep_alpha=False) for c in sns.cubehelix_palette(256)]
        # Fallback anchors approximating cubehelix
        return _interpolate_hex_ramp([
            "#1A1530",  # dark purple-black
            "#3D2B56",  # purple
            "#5A4F7E",  # blue-purple
            "#6B79A8",  # blue
            "#72A6C6",  # cyan
            "#7DCFB6",  # teal-green
            "#A8E890",  # lime green
            "#E8F576",  # yellow-green
            "#FCFDBF",  # cream (high)
        ])
    if name == "cubehelix_r":
        # Reversed cubehelix - dark at high values
        if SEABORN_AVAILABLE:
            return [to_hex(c, keep_alpha=False) for c in sns.cubehelix_palette(256, reverse=True)]
        return list(reversed(_interpolate_hex_ramp([
            "#1A1530", "#3D2B56", "#5A4F7E", "#6B79A8",
            "#72A6C6", "#7DCFB6", "#A8E890", "#E8F576", "#FCFDBF",
        ])))
    if name == "cubehelix_green":
        # Custom cubehelix variant: cream → green → teal emphasis
        if SEABORN_AVAILABLE:
            return [to_hex(c, keep_alpha=False) for c in sns.cubehelix_palette(256, start=0.5, rot=-0.5, light=0.95, dark=0.15)]
        return _interpolate_hex_ramp([
            "#F5FBF2",  # light cream-green
            "#D4EFD0",  # pale green
            "#9DD4A5",  # green
            "#5BB07C",  # darker green
            "#2C8B6B",  # teal-green
            "#1A6B6B",  # teal
            "#154A5A",  # dark teal
            "#122E3D",  # dark blue-green
        ])
    if name == "cubehelix_ylbu":
        # Sandy grey → muted green → dark blue. Matches the bivariate/ALC suitability map palette.
        if SEABORN_AVAILABLE:
            import numpy as _np
            _colors = sns.cubehelix_palette(256, start=2.0, rot=-1.0, light=0.96, dark=0.05, hue=1.8)
            _sandy = _np.array([0.76, 0.72, 0.65])
            _n_blend = 60
            for _i in range(_n_blend):
                _t = _i / _n_blend
                _colors[_i] = (1 - _t) * _sandy + _t * _np.array(_colors[_i])
            return [to_hex(c, keep_alpha=False) for c in _colors]
        return _interpolate_hex_ramp([
            "#C2B8A6",  # sandy grey (low)
            "#D7EDDB",  # muted green
            "#7EC8C9",  # teal
            "#2E88BD",  # blue
            "#163E82",  # dark blue
            "#081D58",  # very dark blue (high)
        ])
    if name == "bivariate_landvalue":
        # Pale grey-green -> muted teal -> deep blue-navy.
        # Derived from the bivariate suitability x land-value palette averages.
        return _interpolate_hex_ramp([
            "#FBFCFB",  # near-white
            "#EEF2EE",  # cool off-white
            "#DBE2DB",  # pale grey-green
            "#B9C4B7",  # soft sage
            "#83BCB6",  # muted teal
            "#6FA89E",  # deeper teal-green
            "#4E6486",  # slate blue
            "#334869",  # dark blue
            "#223955",  # navy
        ])
    raise ValueError(f"Unknown colormap: {name}")


def stretch_hillshade(
    hillshade_vals: np.ndarray,
    low_pct: float,
    high_pct: float,
    gamma: float,
) -> np.ndarray:
    """Percentile stretch + gamma for stronger terrain relief."""
    v = hillshade_vals.astype(np.float64, copy=False)
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros_like(v, dtype=np.float64)

    lo = float(np.nanpercentile(v[finite], low_pct))
    hi = float(np.nanpercentile(v[finite], high_pct))
    if hi <= lo:
        hi = lo + 1.0

    out = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        out = np.power(out, gamma)
    return out


def _safe_percentile(arr: np.ndarray, pct: float, fallback: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return fallback
    return float(np.nanpercentile(finite, pct))


def _grid_cellsize_from_agg(agg: xr.DataArray, fallback_transform: rasterio.Affine) -> tuple[float, float]:
    """Estimate map cell size (meters) from aggregated grid coordinates."""
    x_res = abs(float(fallback_transform.a))
    y_res = abs(float(fallback_transform.e))
    if "x" in agg.coords and agg.sizes.get("x", 0) > 1:
        dx = np.diff(agg.x.values.astype(np.float64))
        dx = dx[np.isfinite(dx)]
        if dx.size:
            x_res = abs(float(np.nanmedian(dx)))
    if "y" in agg.coords and agg.sizes.get("y", 0) > 1:
        dy = np.diff(agg.y.values.astype(np.float64))
        dy = dy[np.isfinite(dy)]
        if dy.size:
            y_res = abs(float(np.nanmedian(dy)))
    return max(x_res, 1e-6), max(y_res, 1e-6)


def _box_blur_nan(arr: np.ndarray, radius_px: int) -> np.ndarray:
    """Fast NaN-aware box blur using integral images."""
    if radius_px <= 0:
        return arr.astype(np.float64, copy=True)

    r = int(radius_px)
    k = 2 * r + 1
    src = arr.astype(np.float64, copy=False)
    pad = np.pad(src, ((r, r), (r, r)), mode="edge")

    valid = np.isfinite(pad).astype(np.float64)
    data = np.where(np.isfinite(pad), pad, 0.0)

    s_data = np.pad(data, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    s_valid = np.pad(valid, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    sum_data = s_data[k:, k:] - s_data[:-k, k:] - s_data[k:, :-k] + s_data[:-k, :-k]
    sum_valid = s_valid[k:, k:] - s_valid[:-k, k:] - s_valid[k:, :-k] + s_valid[:-k, :-k]

    out = np.where(sum_valid > 0, sum_data / np.maximum(sum_valid, 1.0), np.nan)
    return out


def build_pro_terrain_luminance(
    hillshade_vals: np.ndarray,
    dem_vals: np.ndarray,
    x_res_m: float,
    y_res_m: float,
    clip_low: float,
    clip_high: float,
    final_gamma: float,
    lrm_radius_px: int,
    lrm_weight: float,
    slope_weight: float,
    curvature_weight: float,
) -> np.ndarray:
    """Multi-scale terrain composite: hillshade + LRM + slope/curvature ambient terms."""
    hs = stretch_hillshade(hillshade_vals, low_pct=clip_low, high_pct=clip_high, gamma=1.0)
    dem = dem_vals.astype(np.float64, copy=False)
    dem_valid = np.isfinite(dem)
    if not dem_valid.any():
        return stretch_hillshade(hillshade_vals, low_pct=clip_low, high_pct=clip_high, gamma=final_gamma)

    # Local Relief Model (high-pass terrain at chosen scale).
    dem_blur = _box_blur_nan(dem, radius_px=max(1, int(lrm_radius_px)))
    lrm = dem - dem_blur
    lrm_scale = _safe_percentile(np.abs(lrm), 95.0, 1.0)
    if lrm_scale <= 1e-9:
        lrm_scale = 1.0
    lrm_norm = np.clip(0.5 + 0.5 * (lrm / lrm_scale), 0.0, 1.0)

    # Gradient-derived ambient shading terms.
    dem_fill = np.where(dem_valid, dem, _safe_percentile(dem, 50.0, 0.0))
    gy, gx = np.gradient(dem_fill, y_res_m, x_res_m)
    slope_mag = np.sqrt(gx * gx + gy * gy)
    slope_scale = _safe_percentile(slope_mag, 98.5, 1.0)
    slope_norm = np.clip(slope_mag / max(slope_scale, 1e-9), 0.0, 1.0)
    slope_term = 1.0 - np.clip(slope_weight, 0.0, 1.0) * slope_norm

    gxx = np.gradient(gx, x_res_m, axis=1)
    gyy = np.gradient(gy, y_res_m, axis=0)
    curvature = gxx + gyy
    curv_scale = _safe_percentile(np.abs(curvature), 98.5, 1.0)
    curv_norm = np.clip(np.abs(curvature) / max(curv_scale, 1e-9), 0.0, 1.0)
    curv_term = 1.0 - np.clip(curvature_weight, 0.0, 1.0) * curv_norm

    # Composite relief: preserve classic hillshade while injecting multi-scale structure.
    terrain = (1.0 - np.clip(lrm_weight, 0.0, 1.0)) * hs + np.clip(lrm_weight, 0.0, 1.0) * lrm_norm
    terrain = terrain * slope_term * curv_term
    terrain = np.where(dem_valid, terrain, np.nan)

    return stretch_hillshade(terrain * 255.0, low_pct=1.0, high_pct=99.0, gamma=final_gamma)


def exterior_nodata_mask(valid_mask: np.ndarray) -> np.ndarray:
    """Return nodata cells connected to image edges (treat as outside background)."""
    h, w = valid_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def try_seed(y: int, x: int) -> None:
        if not valid_mask[y, x] and not outside[y, x]:
            outside[y, x] = True
            q.append((y, x))

    for x in range(w):
        try_seed(0, x)
        try_seed(h - 1, x)
    for y in range(h):
        try_seed(y, 0)
        try_seed(y, w - 1)

    while q:
        y, x = q.popleft()
        if y > 0:
            ny, nx = y - 1, x
            if not valid_mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
        if y < h - 1:
            ny, nx = y + 1, x
            if not valid_mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
        if x > 0:
            ny, nx = y, x - 1
            if not valid_mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))
        if x < w - 1:
            ny, nx = y, x + 1
            if not valid_mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))

    return outside


def _nice_scale_km(target_km: float) -> float:
    """Round target to a cartographic-friendly bar length in km."""
    if target_km <= 0:
        return 1.0
    exp = math.floor(math.log10(target_km))
    base = 10 ** exp
    mantissa = target_km / base
    if mantissa >= 5:
        nice = 5
    elif mantissa >= 2:
        nice = 2
    else:
        nice = 1
    return nice * base


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a readable font with fallback."""
    candidates = []
    if bold:
        candidates = [
            "DejaVuSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ]
    else:
        candidates = [
            "DejaVuSans.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def add_map_decorations(
    image: Image.Image,
    cmap: list[str],
    map_width_m: float | None,
    add_scale_bar: bool,
    add_legend: bool,
    scale_bar_km: float | None = None,
    legend_title: str = "Suitability",
) -> Image.Image:
    """Draw simple cartographic overlays: top-left scale bar and 0-1 legend."""
    if not add_scale_bar and not add_legend:
        return image

    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    h = out.size[1]
    font = _load_font(max(16, int(h * 0.018)), bold=False)
    font_bold = _load_font(max(18, int(h * 0.02)), bold=True)
    w, h = out.size

    if add_scale_bar and map_width_m and map_width_m > 0:
        map_width_km = map_width_m / 1000.0
        if scale_bar_km is not None and scale_bar_km > 0:
            scale_km = float(scale_bar_km)
        else:
            target_km = max(1.0, map_width_km * 0.14)
            scale_km = _nice_scale_km(target_km)
        bar_px = max(40, int(round((scale_km / map_width_km) * w)))
        margin = max(16, int(0.02 * min(w, h)))
        x0 = max(margin, (w - bar_px) // 2)
        y0 = h - margin - 40
        x1 = min(w - margin, x0 + bar_px)
        # Bold classic black line with subtle white halo.
        draw.line((x0, y0, x1, y0), fill=(255, 255, 255), width=10)
        draw.line((x0, y0, x1, y0), fill=(0, 0, 0), width=6)
        tick_h = 10
        for x in (x0, x1):
            draw.line((x, y0 - tick_h, x, y0 + tick_h), fill=(255, 255, 255), width=8)
            draw.line((x, y0 - tick_h, x, y0 + tick_h), fill=(0, 0, 0), width=4)
        label = f"{scale_km:g} km"
        # text below the scale bar
        tx = (x0 + x1) // 2 - 20
        ty = y0 + tick_h + 8
        for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((tx + ox, ty + oy), label, fill=(255, 255, 255), font=font_bold)
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font_bold)

    if add_legend and cmap:
        bar_h = max(120, int(0.18 * h))
        bar_w = max(22, int(0.018 * w))
        bar_x0 = int(0.20 * w)  # 20% from left
        bar_y0 = int(0.10 * h)  # 10% from top
        bar = Image.new("RGB", (bar_w, bar_h))
        bar_px_img = bar.load()
        n = len(cmap)
        for yy in range(bar_h):
            t = 1.0 - (yy / max(1, bar_h - 1))
            idx = min(n - 1, max(0, int(round(t * (n - 1)))))
            color = cmap[idx]
            if isinstance(color, str):
                # Supports '#RRGGBB' from matplotlib/colorcet
                rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
            else:
                rgb = tuple(color[:3])
            for xx in range(bar_w):
                bar_px_img[xx, yy] = rgb
        out.paste(bar, (bar_x0, bar_y0))
        # Title above bar, values to the right
        title_bbox = draw.textbbox((0, 0), legend_title, font=font_bold)
        title_h = max(1, title_bbox[3] - title_bbox[1])
        title_gap = max(10, int(0.008 * h))
        title_y = bar_y0 - title_h - title_gap
        for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((bar_x0 + ox, title_y + oy), legend_title, fill=(255, 255, 255), font=font_bold)
        draw.text((bar_x0, title_y), legend_title, fill=(0, 0, 0), font=font_bold)
        for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((bar_x0 + bar_w + 8 + ox, bar_y0 + oy), "1.0", fill=(255, 255, 255), font=font)
        draw.text((bar_x0 + bar_w + 8, bar_y0), "1.0", fill=(0, 0, 0), font=font)
        for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((bar_x0 + bar_w + 8 + ox, bar_y0 + bar_h - 16 + oy), "0.0", fill=(255, 255, 255), font=font)
        draw.text((bar_x0 + bar_w + 8, bar_y0 + bar_h - 16), "0.0", fill=(0, 0, 0), font=font)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Render predictions and suitability TIFs with Datashader (fire cmap, white bg)"
    )
    parser.add_argument(
        "--predictions",
        default="data/output/postprocess/wet_woodland_mosaic_hysteresis.tif",
        help="Predictions GeoTIFF (band 1 is the binary extent surface by default)",
    )
    parser.add_argument(
        "--suitability",
        default="data/output/potential/maxent/wet_woodland_potential.tif",
        help="Suitability GeoTIFF (0–1)",
    )
    parser.add_argument(
        "--predictions-band",
        type=int,
        default=1,
        metavar="N",
        help="1-based raster band for predictions input (default 1).",
    )
    parser.add_argument(
        "--suitability-band",
        type=int,
        default=1,
        metavar="N",
        help="1-based raster band for suitability input (default 1).",
    )
    parser.add_argument(
        "--predictions-resample-m",
        type=float,
        default=0.0,
        metavar="M",
        help="Optional predictions-only target resolution in meters before rendering (e.g. 100). Set 0 to keep native.",
    )
    parser.add_argument(
        "--output-dir",
        default="visualise/output",
        help="Directory for output PNGs",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "density", "connectivity", "density_connectivity", "hexbin"],
        default="direct",
        help="direct = map values; density = coarser grid; connectivity = patch weighting; density_connectivity = density + connectivity combined; hexbin = 1 km pointy-top hexagonal bins (vectorised, fast)",
    )
    parser.add_argument(
        "--density-cell-m",
        type=float,
        default=1000.0,
        metavar="M",
        help="Cell size in meters for density aggregation (default 1000 = 1km pixels).",
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=0.0,
        metavar="V",
        help="Only count values above this threshold in density aggregation (default 0).",
    )
    parser.add_argument(
        "--density-min",
        type=float,
        default=0.0,
        metavar="V",
        help="Filter out cells with aggregated density below this value (removes scattered isolated pixels).",
    )
    parser.add_argument(
        "--density-normalize-max",
        action="store_true",
        help="Normalize density to global max (highest density = 1.0) to highlight relative differences.",
    )
    parser.add_argument(
        "--density-pre-filter-threshold",
        type=float,
        default=0.0,
        metavar="V",
        help="Threshold for pre-filtering raw pixels before density (values >= this are considered valid).",
    )
    parser.add_argument(
        "--density-pre-filter-min-patch",
        type=int,
        default=0,
        metavar="N",
        help="Min patch size for pre-filtering (8-connectivity at pre-filter resolution before density aggregation).",
    )
    parser.add_argument(
        "--density-pre-filter-res",
        type=float,
        default=100.0,
        metavar="M",
        help="Resolution in meters for pre-filtering connectivity analysis (default 100m).",
    )
    # Connectivity mode arguments
    parser.add_argument(
        "--connectivity-threshold",
        type=float,
        default=0.15,
        metavar="V",
        help="Probability threshold for connectivity analysis (pixels above this are 'connected').",
    )
    parser.add_argument(
        "--connectivity-min-patch",
        type=int,
        default=1,
        metavar="N",
        help="Minimum patch size in pixels (smaller patches are filtered out).",
    )
    parser.add_argument(
        "--predictions-connectivity-min-patch",
        type=int,
        default=None,
        metavar="N",
        help="Optional direct-mode override for predictions patch size filter. Defaults to 3 when global defaults are used.",
    )
    parser.add_argument(
        "--suitability-connectivity-min-patch",
        type=int,
        default=None,
        metavar="N",
        help="Optional direct-mode override for suitability patch size filter. Defaults to 1 when global defaults are used.",
    )
    parser.add_argument(
        "--pre-filter-res",
        type=float,
        default=0.0,
        metavar="M",
        help="Resolution in meters for pre-filtering connectivity in direct mode (0 = disabled). E.g., 100 for 100m.",
    )
    parser.add_argument(
        "--pre-filter-threshold",
        type=float,
        default=0.15,
        metavar="V",
        help="Threshold for pre-filtering (values >= this are considered connected).",
    )
    parser.add_argument(
        "--pre-filter-min-patch",
        type=int,
        default=3,
        metavar="N",
        help="Minimum patch size for pre-filtering at pre-filter resolution.",
    )
    parser.add_argument(
        "--connectivity-power",
        type=float,
        default=0.5,
        metavar="P",
        help="Patch size weighting power (0.5=sqrt, 0.3=cube root, 1.0=linear). Lower = less extreme weighting.",
    )
    parser.add_argument(
        "--isolated-weight",
        type=float,
        default=0.2,
        metavar="W",
        help="Base weight for isolated cells in density_connectivity mode (0.2 = isolated get 20%% of value).",
    )
    parser.add_argument(
        "--output-gamma",
        type=float,
        default=1.0,
        metavar="G",
        help="Gamma correction for output values (< 1 boosts contrast, e.g., 0.5 for sqrt stretch).",
    )
    parser.add_argument(
        "--final-mask",
        type=float,
        default=0.0,
        metavar="V",
        help="Hard mask: values below this become transparent/white (removes low-value smear).",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        metavar="S",
        help="Gaussian smoothing sigma to blend sparse pixels (0=off, try 1-3 for soft blending).",
    )
    parser.add_argument(
        "--colormap",
        choices=[
            "fire",
            "viridis",
            "cividis",
            "spectral",
            "spectral_r",
            "ylgnbu",
            "ylgnbu_r",
            "rocket",
            "sunset",
            "sunset_r",
            "sunset_helix",
            "sunset_helix_r",
            "earth",
            "slate",
            "ember",
            "heat",
            "inferno",
            "mako",
            "mako_r",
            "mako_viridis",
            "wetland",
            "wetland_blue",
            "cream_blue_black",
            "cream_cyan_blue",
            "cubehelix",
            "cubehelix_r",
            "cubehelix_green",
            "cubehelix_ylbu",
            "bivariate_landvalue",
        ],
        default="cubehelix_ylbu",
        help="Thematic colormap. cubehelix: perceptually uniform; wetland: cream→green→blue.",
    )
    parser.add_argument(
        "--shade-how",
        choices=["linear", "eq_hist", "log", "cbrt"],
        default="linear",
        help="Datashader value transform before color mapping. eq_hist often makes values stand out more.",
    )
    parser.add_argument(
        "--raster-interpolate",
        choices=["nearest", "linear"],
        default="linear",
        help="Interpolation used when raster needs resampling to output grid.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        metavar="N",
        help="Output size in pixels for the longest image side (preserves aspect ratio by default).",
    )
    parser.add_argument(
        "--square-size",
        action="store_true",
        help="When using --size, force square output instead of preserving raster aspect ratio.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width (overrides --size if set)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height (overrides --size if set)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=2000,
        metavar="N",
        help="Max raster side when loading (decimate to save memory). Default 2000.",
    )
    parser.add_argument(
        "--test-window-km",
        type=float,
        default=None,
        metavar="KM",
        help="Optional square test window size in km (e.g. 100 for fast iteration).",
    )
    parser.add_argument(
        "--test-center-lon",
        type=float,
        default=-3.95,
        help="Test window center longitude in EPSG:4326 (default Dartmoor area).",
    )
    parser.add_argument(
        "--test-center-lat",
        type=float,
        default=50.56,
        help="Test window center latitude in EPSG:4326 (default Dartmoor area).",
    )
    parser.add_argument(
        "--test-center-x",
        type=float,
        default=None,
        help="Optional test window center X in raster CRS units (e.g. EPSG:27700 eastings).",
    )
    parser.add_argument(
        "--test-center-y",
        type=float,
        default=None,
        help="Optional test window center Y in raster CRS units (e.g. EPSG:27700 northings).",
    )
    parser.add_argument(
        "--dem",
        default="data/output/potential/potential_predictors_100m.tif",
        metavar="PATH",
        help="DEM/DTM GeoTIFF for hillshade underlay (requires earthpy). Default: data/output/potential/potential_predictors_100m.tif. Set to empty or missing to disable.",
    )
    parser.add_argument(
        "--dem-band",
        type=int,
        default=1,
        metavar="N",
        help="1-based band index for elevation in multi-band DTM (default 1).",
    )
    parser.add_argument(
        "--dem-slope-band",
        type=int,
        default=2,
        metavar="N",
        help="1-based band for slope in degrees (default 2). Set 0 to derive from elevation.",
    )
    parser.add_argument(
        "--dem-aspect-band",
        type=int,
        default=3,
        metavar="N",
        help="1-based band for aspect in degrees (default 3). Set 0 to derive from elevation.",
    )
    parser.add_argument(
        "--hillshade-azimuth",
        type=float,
        default=240,
        help="Sun azimuth for hillshade (degrees, 0–360). Default 240 (NW).",
    )
    parser.add_argument(
        "--hillshade-altitude",
        type=float,
        default=30,
        help="Sun altitude for hillshade (degrees, 0–90). Default 30.",
    )
    parser.add_argument(
        "--hillshade-z-factor",
        type=float,
        default=1.5,
        help="Vertical exaggeration factor for terrain (1.5 = 50%% more pronounced). Default 1.5.",
    )
    parser.add_argument(
        "--hillshade-multidirectional",
        action="store_true",
        help="Use multidirectional hillshade outlines (45,135,225,315 style) blended with primary hillshade.",
    )
    parser.add_argument(
        "--hillshade-md-azimuths",
        default="45,135,225,315",
        help="Comma list of azimuths for multidirectional hillshade.",
    )
    parser.add_argument(
        "--hillshade-md-altitudes",
        default="45",
        help="Comma list of altitudes for multidirectional hillshade (or one value for all azimuths).",
    )
    parser.add_argument(
        "--hillshade-md-strength",
        type=float,
        default=0.55,
        metavar="S",
        help="Blend strength of multiplied multidirectional hillshade (0–1).",
    )
    parser.add_argument(
        "--hillshade-md-mode",
        choices=["multiply", "replace"],
        default="multiply",
        help="How multidirectional hillshade combines with primary hillshade.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.7,
        metavar="A",
        help="Maximum thematic opacity over hillshade (0–1). Default 0.7.",
    )
    parser.add_argument(
        "--terrain-blend-mode",
        choices=["drape", "alpha_only"],
        default="drape",
        help="drape = color modulated by hillshade; alpha_only = plain color alpha over terrain.",
    )
    parser.add_argument(
        "--terrain-only",
        action="store_true",
        help="Render terrain/hillshade only (no thematic colors). Uses predictions raster only for extent/window.",
    )
    parser.add_argument(
        "--terrain-palette",
        choices=["black_white", "white_black"],
        default="black_white",
        help="Grayscale palette for --terrain-only. black_white keeps highlights bright; white_black inverts.",
    )
    parser.add_argument(
        "--outline-mode",
        choices=["off", "multidirectional"],
        default="multidirectional",
        help="Use multidirectional hillshade terrain (4-direction average for soft outline effect). Default multidirectional.",
    )
    parser.add_argument(
        "--outline-gamma",
        type=float,
        default=0.7,
        metavar="G",
        help="Gamma for terrain contrast (0.3-2). <1 = more contrast. Default 0.7.",
    )
    parser.add_argument(
        "--clip-boundary",
        default="data/input/boundaries/england.shp",
        metavar="PATH",
        help="Optional boundary vector to clip raster outputs. Default: data/input/boundaries/england.shp",
    )
    parser.add_argument(
        "--clip-buffer-m",
        type=float,
        default=0.0,
        metavar="M",
        help="Buffer distance for clip geometry in CRS units (meters in EPSG:27700). Use -100 to shrink inward.",
    )
    parser.add_argument(
        "--fill-interior-nodata",
        action="store_true",
        default=False,
        help="Fill nodata holes inside the clip boundary with the in-polygon minimum value. Off by default (holes show as hillshade/white).",
    )
    parser.add_argument(
        "--value-mask-below",
        type=float,
        default=0.0,
        metavar="V",
        help="Mask thematic values below V (0–1) so hillshade shows through. Default 0 (off).",
    )
    parser.add_argument(
        "--predictions-value-mask-below",
        type=float,
        default=None,
        metavar="V",
        help="Optional predictions-only cutoff (0–1). Overrides --value-mask-below for predictions.",
    )
    parser.add_argument(
        "--suitability-value-mask-below",
        type=float,
        default=None,
        metavar="V",
        help="Optional suitability-only cutoff (0–1). Overrides --value-mask-below for suitability.",
    )
    parser.add_argument(
        "--value-gamma",
        type=float,
        default=1.0,
        metavar="G",
        help="Gamma for thematic values after masking (0.2–5). >1 emphasizes higher values.",
    )
    parser.add_argument(
        "--predictions-value-gamma",
        type=float,
        default=None,
        metavar="G",
        help="Optional predictions-only gamma (0.2–5). Overrides --value-gamma for predictions.",
    )
    parser.add_argument(
        "--suitability-value-gamma",
        type=float,
        default=None,
        metavar="G",
        help="Optional suitability-only gamma (0.2–5). Overrides --value-gamma for suitability.",
    )
    parser.add_argument(
        "--hillshade-strength",
        type=float,
        default=0.85,
        metavar="S",
        help="Multiply blend: thematic * ((1-S) + S*hillshade). Higher = stronger relief (0–1). Default 0.85.",
    )
    parser.add_argument(
        "--hillshade-clip-low",
        type=float,
        default=2.0,
        metavar="P",
        help="Low percentile for hillshade contrast stretch (0–100). Default 2.",
    )
    parser.add_argument(
        "--hillshade-clip-high",
        type=float,
        default=98.0,
        metavar="P",
        help="High percentile for hillshade contrast stretch (0–100). Default 98.",
    )
    parser.add_argument(
        "--hillshade-gamma",
        type=float,
        default=0.9,
        metavar="G",
        help="Gamma for stretched hillshade (>0). <1 brightens terrain. Default 0.9.",
    )
    parser.add_argument(
        "--terrain-composite",
        choices=["classic", "pro"],
        default="classic",
        help="classic: hillshade stretch only; pro: adds local relief + slope/curvature ambient terms.",
    )
    parser.add_argument(
        "--pro-lrm-radius-px",
        type=int,
        default=14,
        metavar="PX",
        help="Radius (pixels) for local relief model blur in pro terrain mode.",
    )
    parser.add_argument(
        "--pro-lrm-weight",
        type=float,
        default=0.34,
        metavar="W",
        help="Blend weight of local relief model in pro terrain mode (0-1).",
    )
    parser.add_argument(
        "--pro-slope-weight",
        type=float,
        default=0.18,
        metavar="W",
        help="Ambient darkening strength from slope in pro terrain mode (0-1).",
    )
    parser.add_argument(
        "--pro-curvature-weight",
        type=float,
        default=0.14,
        metavar="W",
        help="Ambient outline strength from curvature in pro terrain mode (0-1).",
    )
    parser.add_argument(
        "--pro-terrain-gamma",
        type=float,
        default=0.92,
        metavar="G",
        help="Final gamma applied to pro terrain composite (>0).",
    )
    parser.add_argument(
        "--thematic-alpha-min",
        type=float,
        default=0.2,
        metavar="A",
        help="Minimum thematic alpha for valid cells (0–1). Default 0.2.",
    )
    parser.add_argument(
        "--thematic-alpha-gamma",
        type=float,
        default=0.8,
        metavar="G",
        help="Gamma for thematic alpha ramp (>0). <1 lifts mid-values. Default 0.8.",
    )
    parser.add_argument(
        "--pillow-contrast",
        type=float,
        default=1.5,
        metavar="F",
        help="Pillow contrast enhancement factor (>0). Default 1.5.",
    )
    parser.add_argument(
        "--pillow-color",
        type=float,
        default=1.6,
        metavar="F",
        help="Pillow color/saturation enhancement factor (>0). Default 1.6.",
    )
    parser.add_argument(
        "--pillow-sharpness",
        type=float,
        default=1.0,
        metavar="F",
        help="Pillow sharpness enhancement factor (>0). 1.0 = unchanged.",
    )
    parser.add_argument(
        "--pillow-autocontrast-cutoff",
        type=float,
        default=1.0,
        metavar="P",
        help="Pillow autocontrast cutoff percent (0–49). Default 1.",
    )
    parser.add_argument(
        "--pillow-aa-scale",
        type=int,
        default=1,
        metavar="N",
        help="Optional anti-alias supersample factor (1=off, 2 recommended).",
    )
    parser.add_argument(
        "--pillow-smooth-radius",
        type=float,
        default=0.0,
        metavar="R",
        help="Gaussian smoothing radius for jagged edges (0=off, try 0.25-0.6).",
    )
    parser.add_argument(
        "--pillow-smooth-restore",
        type=int,
        default=0,
        metavar="PCT",
        help="Unsharp restore strength after smoothing (0-300, 0=off; try 70-110).",
    )
    parser.add_argument(
        "--pillow-clarity-radius",
        type=float,
        default=0.0,
        metavar="R",
        help="Luminance-only local contrast radius (0=off; try 6-10).",
    )
    parser.add_argument(
        "--pillow-clarity-percent",
        type=int,
        default=0,
        metavar="PCT",
        help="Luminance-only local contrast strength (0-300; try 25-45).",
    )
    parser.add_argument(
        "--pillow-tone-curve",
        type=float,
        default=0.0,
        metavar="S",
        help="S-curve tone mapping strength (0-1).",
    )
    parser.add_argument(
        "--pillow-grain",
        type=float,
        default=0.0,
        metavar="A",
        help="Monochrome grain amount (0-0.2; try 0.008-0.02).",
    )
    parser.add_argument(
        "--add-scale-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a simple black km scale bar at bottom-center. Enabled by default.",
    )
    parser.add_argument(
        "--add-legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a top-left legend with the active color palette and 0-1 range. Enabled by default.",
    )
    parser.add_argument(
        "--legend-title",
        default=None,
        help="Optional global legend title override for all outputs.",
    )
    parser.add_argument(
        "--predictions-legend-title",
        default="Prediction density",
        help="Legend title for predictions output.",
    )
    parser.add_argument(
        "--suitability-legend-title",
        default="Restoration suitability",
        help="Legend title for suitability output.",
    )
    parser.add_argument(
        "--scale-bar-km",
        type=float,
        default=None,
        metavar="KM",
        help="Optional fixed scale bar length in km (e.g. 100).",
    )
    parser.add_argument(
        "--output-dpi",
        type=int,
        default=300,
        metavar="DPI",
        help="PNG DPI metadata to embed (e.g. 300 or 600).",
    )
    args = parser.parse_args()

    if args.pillow_contrast <= 0:
        raise ValueError("--pillow-contrast must be > 0")
    if args.pillow_color <= 0:
        raise ValueError("--pillow-color must be > 0")
    if args.pillow_sharpness <= 0:
        raise ValueError("--pillow-sharpness must be > 0")
    if not (0 <= args.pillow_autocontrast_cutoff < 50):
        raise ValueError("--pillow-autocontrast-cutoff must be in [0, 50)")
    if args.pillow_aa_scale < 1 or args.pillow_aa_scale > 4:
        raise ValueError("--pillow-aa-scale must be in [1, 4]")
    if args.pillow_smooth_radius < 0:
        raise ValueError("--pillow-smooth-radius must be >= 0")
    if args.pillow_smooth_restore < 0 or args.pillow_smooth_restore > 300:
        raise ValueError("--pillow-smooth-restore must be in [0, 300]")
    if args.pillow_clarity_radius < 0:
        raise ValueError("--pillow-clarity-radius must be >= 0")
    if args.pillow_clarity_percent < 0 or args.pillow_clarity_percent > 300:
        raise ValueError("--pillow-clarity-percent must be in [0, 300]")
    if args.pillow_tone_curve < 0 or args.pillow_tone_curve > 1:
        raise ValueError("--pillow-tone-curve must be in [0, 1]")
    if args.pillow_grain < 0 or args.pillow_grain > 0.2:
        raise ValueError("--pillow-grain must be in [0, 0.2]")
    if not (0 <= args.value_mask_below < 1):
        raise ValueError("--value-mask-below must be in [0, 1)")
    if args.predictions_value_mask_below is not None and not (0 <= args.predictions_value_mask_below < 1):
        raise ValueError("--predictions-value-mask-below must be in [0, 1)")
    if args.suitability_value_mask_below is not None and not (0 <= args.suitability_value_mask_below < 1):
        raise ValueError("--suitability-value-mask-below must be in [0, 1)")
    if not (0.2 <= args.value_gamma <= 5.0):
        raise ValueError("--value-gamma must be in [0.2, 5.0]")
    if args.predictions_value_gamma is not None and not (0.2 <= args.predictions_value_gamma <= 5.0):
        raise ValueError("--predictions-value-gamma must be in [0.2, 5.0]")
    if args.suitability_value_gamma is not None and not (0.2 <= args.suitability_value_gamma <= 5.0):
        raise ValueError("--suitability-value-gamma must be in [0.2, 5.0]")
    if not (0 <= args.overlay_alpha <= 1):
        raise ValueError("--overlay-alpha must be in [0, 1]")
    if not (0 <= args.hillshade_clip_low < 100):
        raise ValueError("--hillshade-clip-low must be in [0, 100)")
    if not (0 < args.hillshade_clip_high <= 100):
        raise ValueError("--hillshade-clip-high must be in (0, 100]")
    if not (args.hillshade_clip_low < args.hillshade_clip_high):
        raise ValueError("--hillshade-clip-low must be < --hillshade-clip-high")
    if args.hillshade_gamma <= 0:
        raise ValueError("--hillshade-gamma must be > 0")
    if args.pro_lrm_radius_px < 1 or args.pro_lrm_radius_px > 400:
        raise ValueError("--pro-lrm-radius-px must be in [1, 400]")
    if not (0 <= args.pro_lrm_weight <= 1):
        raise ValueError("--pro-lrm-weight must be in [0, 1]")
    if not (0 <= args.pro_slope_weight <= 1):
        raise ValueError("--pro-slope-weight must be in [0, 1]")
    if not (0 <= args.pro_curvature_weight <= 1):
        raise ValueError("--pro-curvature-weight must be in [0, 1]")
    if args.pro_terrain_gamma <= 0:
        raise ValueError("--pro-terrain-gamma must be > 0")
    if not (0 <= args.hillshade_md_strength <= 1):
        raise ValueError("--hillshade-md-strength must be in [0, 1]")
    if not (0 <= args.thematic_alpha_min <= 1):
        raise ValueError("--thematic-alpha-min must be in [0, 1]")
    if args.thematic_alpha_gamma <= 0:
        raise ValueError("--thematic-alpha-gamma must be > 0")
    if args.scale_bar_km is not None and args.scale_bar_km <= 0:
        raise ValueError("--scale-bar-km must be > 0")
    if args.test_window_km is not None and args.test_window_km <= 0:
        raise ValueError("--test-window-km must be > 0")
    if args.output_dpi <= 0:
        raise ValueError("--output-dpi must be > 0")
    if not (0.3 <= args.outline_gamma <= 2.0):
        raise ValueError("--outline-gamma must be in [0.3, 2.0]")
    if args.predictions_connectivity_min_patch is not None and args.predictions_connectivity_min_patch < 0:
        raise ValueError("--predictions-connectivity-min-patch must be >= 0")
    if args.suitability_connectivity_min_patch is not None and args.suitability_connectivity_min_patch < 0:
        raise ValueError("--suitability-connectivity-min-patch must be >= 0")
    if args.predictions_band < 1:
        raise ValueError("--predictions-band must be >= 1")
    if args.suitability_band < 1:
        raise ValueError("--suitability-band must be >= 1")
    if args.predictions_resample_m < 0:
        raise ValueError("--predictions-resample-m must be >= 0")

    # Keep the tuned look by default:
    # predictions use a stronger direct-mode patch filter; suitability stays lighter.
    default_connectivity_min_patch = parser.get_default("connectivity_min_patch")
    if args.predictions_connectivity_min_patch is None:
        if args.connectivity_min_patch == default_connectivity_min_patch:
            predictions_connectivity_min_patch = 3
        else:
            predictions_connectivity_min_patch = args.connectivity_min_patch
    else:
        predictions_connectivity_min_patch = args.predictions_connectivity_min_patch

    if args.suitability_connectivity_min_patch is None:
        if args.connectivity_min_patch == default_connectivity_min_patch:
            suitability_connectivity_min_patch = 1
        else:
            suitability_connectivity_min_patch = args.connectivity_min_patch
    else:
        suitability_connectivity_min_patch = args.suitability_connectivity_min_patch

    if args.terrain_only:
        cmap = ["#FFFFFF", "#000000"] if args.terrain_palette == "white_black" else ["#000000", "#FFFFFF"]
    else:
        cmap = get_color_ramp(args.colormap)
    md_pairs = _resolve_md_pairs(
        _parse_float_list(args.hillshade_md_azimuths),
        _parse_float_list(args.hillshade_md_altitudes),
    )

    repo = Path(__file__).resolve().parent
    if (repo / "data").exists():
        base = repo
    else:
        base = repo.parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = base / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = base / args.predictions
    suit_path = base / args.suitability
    dem_path = (base / args.dem) if (args.dem and args.dem.strip()) else None
    clip_boundary_path = None
    if args.clip_boundary and args.clip_boundary.strip():
        clip_boundary_path = Path(args.clip_boundary)
        if not clip_boundary_path.is_absolute():
            clip_boundary_path = base / clip_boundary_path
        if not clip_boundary_path.exists():
            print(f"Warning: clip boundary not found at {clip_boundary_path}; clipping disabled.")
            clip_boundary_path = None

    if dem_path and not dem_path.exists():
        dem_path = None
        print("Warning: DEM path not found, skipping hillshade.")
    # Only need earthpy when deriving hillshade from elevation (no slope/aspect bands) and not using outline mode
    need_earthpy = (args.dem_slope_band == 0 or args.dem_aspect_band == 0) and args.outline_mode == "off"
    if dem_path and need_earthpy and not EARTHPY_AVAILABLE:
        dem_path = None
        print("Warning: earthpy not installed, skipping hillshade (needed for elevation-only). pip install earthpy")

    def render_one(label: str, path: Path, out_name: str) -> None:
        if not path.exists():
            print(f"Skipping {label}: not found at {path}")
            return
        print(f"Loading {path}...")
        raster_band = 1
        if label == "predictions":
            raster_band = args.predictions_band
        elif label == "suitability":
            raster_band = args.suitability_band
        outside_white_mask = None
        raw_valid_mask_img = None
        with rasterio.open(path) as src:
            if raster_band > src.count:
                raise ValueError(
                    f"{label} band {raster_band} requested but {path} only has {src.count} band(s)."
                )
            extent_crs = src.crs
        clip_geom = None
        if clip_boundary_path is not None:
            clip_geom = load_clip_geometry(
                clip_boundary_path,
                target_crs=extent_crs,
                buffer_m=args.clip_buffer_m,
            )
        region_bounds = None
        if args.test_window_km is not None:
            if args.test_center_x is not None and args.test_center_y is not None:
                cx, cy = float(args.test_center_x), float(args.test_center_y)
            else:
                if not PYPROJ_AVAILABLE:
                    raise ValueError(
                        "pyproj is required for lon/lat test centers. "
                        "Install pyproj or pass --test-center-x/--test-center-y in raster CRS."
                    )
                to_src = Transformer.from_crs("EPSG:4326", extent_crs, always_xy=True)
                cx, cy = to_src.transform(args.test_center_lon, args.test_center_lat)
            half_m = float(args.test_window_km) * 500.0
            region_bounds = (cx - half_m, cy - half_m, cx + half_m, cy + half_m)
            print(
                f"  Test window {args.test_window_km:g} km around "
                f"({cx:.2f}, {cy:.2f}) in {extent_crs}"
            )

        # For fast iterations, always honor --max-size even when thematic/DTM share a grid.
        max_size_use = args.max_size
        if label == "predictions" and args.predictions_resample_m > 0:
            default_max_size = parser.get_default("max_size")
            if args.max_size == default_max_size:
                max_size_use = None

        # Use density aggregation if mode is density, connectivity weighting if mode is connectivity
        if args.mode == "density":
            da, thematic_transform, clip_mask_da = raster_to_density(
                path,
                cell_size_m=args.density_cell_m,
                threshold=args.density_threshold,
                min_density=args.density_min,
                normalize_to_max=args.density_normalize_max,
                clip_geom=clip_geom,
                region_bounds=region_bounds,
                pre_filter_threshold=args.density_pre_filter_threshold,
                pre_filter_min_patch=args.density_pre_filter_min_patch,
                pre_filter_res_m=args.density_pre_filter_res,
            )
        elif args.mode == "connectivity":
            da, thematic_transform, clip_mask_da = raster_to_connectivity_weighted(
                path,
                max_size=max_size_use,
                threshold=args.connectivity_threshold,
                min_patch_size=args.connectivity_min_patch,
                patch_weight_power=args.connectivity_power,
                clip_geom=clip_geom,
                region_bounds=region_bounds,
            )
        elif args.mode == "density_connectivity":
            da, thematic_transform, clip_mask_da = raster_to_density_connectivity(
                path,
                cell_size_m=args.density_cell_m,
                prob_threshold=args.density_threshold,
                density_threshold=args.density_min,
                min_patch_size=args.connectivity_min_patch,
                connectivity_power=args.connectivity_power,
                isolated_weight=args.isolated_weight,
                output_gamma=args.output_gamma,
                final_mask=args.final_mask,
                smooth_sigma=args.smooth_sigma,
                clip_geom=clip_geom,
                region_bounds=region_bounds,
            )
        elif args.mode == "hexbin":
            hexbin_output_bounds = None
            if dem_path is not None and dem_path.exists():
                with rasterio.open(dem_path) as _dem_src:
                    _b = _dem_src.bounds
                    hexbin_output_bounds = (_b.left, _b.bottom, _b.right, _b.top)
            da, thematic_transform, clip_mask_da = raster_to_hexbin(
                path,
                hex_size_m=args.density_cell_m,
                threshold=args.density_threshold,
                normalize_to_max=args.density_normalize_max,
                clip_geom=clip_geom,
                region_bounds=region_bounds,
                output_bounds=hexbin_output_bounds,
                band_index=raster_band,
            )
        else:
            direct_min_patch = args.connectivity_min_patch
            if label == "predictions":
                direct_min_patch = predictions_connectivity_min_patch
            elif label == "suitability":
                direct_min_patch = suitability_connectivity_min_patch
            da, thematic_transform, clip_mask_da = raster_to_dataarray(
                path,
                band_index=raster_band,
                max_size=max_size_use,
                target_res_m=(args.predictions_resample_m if label == "predictions" and args.predictions_resample_m > 0 else None),
                clip_geom=clip_geom,
                region_bounds=region_bounds,
                connectivity_threshold=args.connectivity_threshold,
                min_patch_size=direct_min_patch,
                pre_filter_res_m=args.pre_filter_res,
                pre_filter_threshold=args.pre_filter_threshold,
                pre_filter_min_patch=args.pre_filter_min_patch,
                fill_interior_nodata=args.fill_interior_nodata,
            )
        ny, nx = da.sizes["y"], da.sizes["x"]
        out_w, out_h = resolve_output_dims(
            ny=ny,
            nx=nx,
            mode=args.mode,
            req_width=args.width,
            req_height=args.height,
            req_size=args.size,
            square_size=args.square_size,
        )
        print(f"  Shape {ny} x {nx} -> output {out_h} x {out_w}")

        if args.mode == "direct":
            # Build an unfiltered validity mask for coast/exterior nodata detection.
            da_raw_valid, _, _ = raster_to_dataarray(
                path,
                band_index=raster_band,
                max_size=max_size_use,
                target_res_m=(args.predictions_resample_m if label == "predictions" and args.predictions_resample_m > 0 else None),
                clip_geom=clip_geom,
                region_bounds=region_bounds,
                connectivity_threshold=0.0,
                min_patch_size=0,
                pre_filter_res_m=0.0,
                pre_filter_threshold=0.0,
                pre_filter_min_patch=0,
            )
            if out_w is not None and out_h is not None and (ny, nx) != (out_h, out_w):
                cvs = ds.Canvas(plot_width=out_w, plot_height=out_h)
                agg = cvs.raster(da, interpolate="linear")
                agg_clip = cvs.raster(clip_mask_da, interpolate="linear")
                agg_raw_valid = cvs.raster(da_raw_valid, interpolate="linear")
                raw_valid_mask_img = np.isfinite(agg_raw_valid.values[::-1, :])
            else:
                agg = da
                agg_clip = clip_mask_da
                raw_valid_mask_img = np.isfinite(da_raw_valid.values[::-1, :])
        else:
            cvs = ds.Canvas(plot_width=out_w, plot_height=out_h)
            agg = cvs.raster(da, interpolate=args.raster_interpolate)
            # Keep nearest for mask edges so clipping remains crisp.
            agg_clip = cvs.raster(clip_mask_da, interpolate="nearest")

        value_mask = args.value_mask_below
        if label == "predictions" and args.predictions_value_mask_below is not None:
            value_mask = args.predictions_value_mask_below
        elif label == "suitability" and args.suitability_value_mask_below is not None:
            value_mask = args.suitability_value_mask_below

        value_gamma = args.value_gamma
        if label == "predictions" and args.predictions_value_gamma is not None:
            value_gamma = args.predictions_value_gamma
        elif label == "suitability" and args.suitability_value_gamma is not None:
            value_gamma = args.suitability_value_gamma

        agg_color = style_agg_for_color(
            agg,
            value_mask_below=value_mask,
            value_gamma=value_gamma,
        )
        img = None
        if not args.terrain_only:
            img = tf.shade(agg_color, cmap=cmap, how=args.shade_how)
            img = tf.set_background(img, "white")

        if dem_path is not None:
            # Direct mode without resize doesn't create cvs; we need it to rasterize hillshade
            if args.mode == "direct" and (out_w is None or out_h is None or ((ny, nx) == (out_h, out_w))):
                cvs = ds.Canvas(plot_width=nx, plot_height=ny)

            # For density mode, compute hillshade at output resolution for sharper terrain
            if args.mode == "density" and out_h is not None and out_w is not None and (out_h, out_w) != (ny, nx):
                # Create a higher-res reference grid for hillshade
                ys = da.coords["y"].values
                xs = da.coords["x"].values
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                hires_ys = np.linspace(y_min, y_max, out_h)
                hires_xs = np.linspace(x_min, x_max, out_w)
                hires_da = xr.DataArray(
                    np.zeros((out_h, out_w), dtype=np.float32),
                    coords=[("y", hires_ys), ("x", hires_xs)],
                    dims=["y", "x"],
                )
                # Compute transform for hires grid
                cell_x = (x_max - x_min) / out_w
                cell_y = (y_max - y_min) / out_h
                hires_transform = rasterio.Affine(cell_x, 0, x_min, 0, -cell_y, y_max)
                print(f"  Hillshade at output resolution ({out_h} x {out_w}) for sharper terrain...")
                hillshade_ref_da = hires_da
                hillshade_ref_transform = hires_transform
            else:
                print(f"  Hillshade on same grid as thematic ({ny} x {nx})...")
                hillshade_ref_da = da
                hillshade_ref_transform = thematic_transform

            slope_b = args.dem_slope_band if args.dem_slope_band else None
            aspect_b = args.dem_aspect_band if args.dem_aspect_band else None
            hillshade_da = hillshade_dataarray_on_da_grid(
                dem_path,
                hillshade_ref_da,
                extent_crs,
                hillshade_ref_transform,
                azimuth=args.hillshade_azimuth,
                altitude=args.hillshade_altitude,
                dem_band=args.dem_band,
                slope_band=slope_b,
                aspect_band=aspect_b,
                use_multidirectional=args.hillshade_multidirectional,
                md_pairs=md_pairs,
                md_strength=args.hillshade_md_strength,
                md_mode=args.hillshade_md_mode,
                z_factor=args.hillshade_z_factor,
            )
            # Rasterize hillshade with the same canvas as the thematic → identical extent and alignment
            agg_hillshade = cvs.raster(hillshade_da, interpolate=args.raster_interpolate)
            top_rgb = None
            if not args.terrain_only:
                top_pil = img.to_pil()
                top_rgb = np.array(top_pil)
                if top_rgb.ndim == 2:
                    top_rgb = np.stack([top_rgb] * 3, axis=-1)
                elif top_rgb.shape[-1] == 4:
                    top_rgb = top_rgb[..., :3]
                top_rgb = top_rgb.astype(np.float64)
            # Datashader image rows are vertically flipped relative to raw agg.values.
            # Flip agg-backed arrays to image-row order before compositing.
            hillshade_vals = np.clip(agg_hillshade.values[::-1, :], 0, 255).astype(np.float64)

            # Load DEM for outline/pro modes
            dem_da = None
            dem_vals = None
            x_res_m, y_res_m = _grid_cellsize_from_agg(agg_hillshade, thematic_transform)
            if args.outline_mode != "off" or args.terrain_composite == "pro":
                dem_da = dem_dataarray_on_da_grid(
                    dem_path,
                    da,
                    extent_crs,
                    thematic_transform,
                    dem_band=args.dem_band,
                )
                agg_dem = cvs.raster(dem_da, interpolate=args.raster_interpolate)
                dem_vals = agg_dem.values[::-1, :].astype(np.float64)

            if args.outline_mode == "multidirectional":
                # Use multidirectional hillshade terrain
                print("  Using multidirectional terrain mode")
                base_vals = build_multidirectional_terrain(
                    dem=dem_vals,
                    x_res=x_res_m,
                    y_res=y_res_m,
                    gamma=args.outline_gamma,
                    z_factor=args.hillshade_z_factor,
                )
            elif args.terrain_composite == "pro":
                base_vals = build_pro_terrain_luminance(
                    hillshade_vals=hillshade_vals,
                    dem_vals=dem_vals,
                    x_res_m=x_res_m,
                    y_res_m=y_res_m,
                    clip_low=args.hillshade_clip_low,
                    clip_high=args.hillshade_clip_high,
                    final_gamma=args.pro_terrain_gamma,
                    lrm_radius_px=args.pro_lrm_radius_px,
                    lrm_weight=args.pro_lrm_weight,
                    slope_weight=args.pro_slope_weight,
                    curvature_weight=args.pro_curvature_weight,
                )
            else:
                base_vals = stretch_hillshade(
                    hillshade_vals,
                    low_pct=args.hillshade_clip_low,
                    high_pct=args.hillshade_clip_high,
                    gamma=args.hillshade_gamma,
                )
            agg_vals = agg_color.values[::-1, :]
            clip_inside_img = agg_clip.values[::-1, :] >= 0.5
            if raw_valid_mask_img is not None:
                raw_valid = raw_valid_mask_img
            else:
                raw_vals = agg.values[::-1, :]
                raw_valid = (
                    ~np.isnan(raw_vals)
                    if np.issubdtype(raw_vals.dtype, np.floating)
                    else np.ones_like(raw_vals, dtype=bool)
                )
            if args.mode == "hexbin":
                # For hexbin the canvas extends to DEM/output_bounds, so the data-extent
                # NaN region in the north is connected to the image edge and would be
                # incorrectly flood-filled as "outside". Use the clip boundary directly.
                outside_nodata = ~clip_inside_img
            elif raw_valid.any():
                outside_nodata = exterior_nodata_mask(raw_valid)
            else:
                outside_nodata = np.zeros_like(raw_valid, dtype=bool)
            # Track where DEM has valid data (not NaN) - these areas get terrain, others get white
            dem_valid = np.isfinite(base_vals)
            base_vals_safe = np.nan_to_num(base_vals, nan=0.0, posinf=1.0, neginf=0.0)
            if args.terrain_palette == "white_black":
                base_gray = ((1.0 - base_vals_safe) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                base_gray = (base_vals_safe * 255.0).clip(0, 255).astype(np.uint8)
            # Make DEM nodata areas white instead of black
            base_gray = np.where(dem_valid, base_gray, 255)
            base_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1).astype(np.float64)
            white_bg = np.full(base_rgb.shape, 255, dtype=np.uint8)
            valid = ~np.isnan(agg_vals) if np.issubdtype(agg_vals.dtype, np.floating) else np.ones_like(agg_vals, dtype=bool)
            if args.terrain_only:
                terrain_mask = clip_inside_img & ~outside_nodata
                composite = np.where(terrain_mask[:, :, np.newaxis], base_rgb.astype(np.uint8), white_bg)
            else:
                # Terrain drape blend:
                # 1) use contrast-stretched hillshade as luminance base
                # 2) modulate thematic by hillshade to preserve relief in colored regions
                # 3) alpha-ramp thematic by value so low values don't drown the terrain
                s = np.clip(args.hillshade_strength, 0.0, 1.0)
                if args.terrain_blend_mode == "alpha_only":
                    themed = top_rgb
                else:
                    shade_factor = ((1.0 - s) + s * base_vals_safe)[:, :, np.newaxis]
                    themed = (top_rgb * shade_factor).clip(0, 255)
                v = np.nan_to_num(agg_vals, nan=0.0).clip(0.0, 1.0)
                alpha_ramp = np.power(v, args.thematic_alpha_gamma)
                alpha = np.where(
                    valid,
                    np.clip(args.thematic_alpha_min + (args.overlay_alpha - args.thematic_alpha_min) * alpha_ramp, 0.0, 1.0),
                    0.0,
                )[:, :, np.newaxis]
                modulated = (base_rgb * (1.0 - alpha) + themed * alpha).clip(0, 255).astype(np.uint8)
                # For nodata/transparent cells inside the clip, show hillshade instead of white holes.
                interior_mask = clip_inside_img & ~outside_nodata
                nodata_bg = np.where(interior_mask[:, :, np.newaxis], base_rgb.astype(np.uint8), white_bg)
                composite = np.where(valid[:, :, np.newaxis], modulated, nodata_bg)
            outside_white_mask = (~clip_inside_img) | outside_nodata
            out_pil = Image.fromarray(composite)
        else:
            if args.terrain_only:
                raise ValueError("--terrain-only requires a valid --dem path")
            out_pil = img.to_pil()

        out_pil = apply_pillow_enhancements(
            out_pil,
            contrast=args.pillow_contrast,
            color=args.pillow_color,
            sharpness=args.pillow_sharpness,
            autocontrast_cutoff=args.pillow_autocontrast_cutoff,
            aa_scale=args.pillow_aa_scale,
            smooth_radius=args.pillow_smooth_radius,
            smooth_restore=args.pillow_smooth_restore,
            clarity_radius=args.pillow_clarity_radius,
            clarity_percent=args.pillow_clarity_percent,
            tone_curve_strength=args.pillow_tone_curve,
            grain_amount=args.pillow_grain,
        )
        if outside_white_mask is not None:
            out_arr = np.asarray(out_pil.convert("RGB"), dtype=np.uint8).copy()
            out_arr[outside_white_mask] = 255
            out_pil = Image.fromarray(out_arr, mode="RGB")
        map_width_m = None
        if extent_crs is not None:
            linear_units = getattr(extent_crs, "linear_units", None)
            linear_factor = getattr(extent_crs, "linear_units_factor", None)
            if linear_units in ("metre", "meter"):
                # Use transform-based width so it matches the full raster extent.
                map_width_m = abs(float(thematic_transform.a)) * float(nx)
            elif linear_factor and isinstance(linear_factor, tuple) and linear_factor[0] in ("metre", "meter"):
                map_width_m = abs(float(thematic_transform.a) * float(linear_factor[1])) * float(nx)
        if args.legend_title:
            map_legend_title = args.legend_title
        elif label == "predictions":
            map_legend_title = args.predictions_legend_title
        elif label == "suitability":
            map_legend_title = args.suitability_legend_title
        else:
            map_legend_title = "Value"
        out_pil = add_map_decorations(
            out_pil,
            cmap=cmap,
            map_width_m=map_width_m,
            add_scale_bar=args.add_scale_bar,
            add_legend=args.add_legend,
            scale_bar_km=args.scale_bar_km,
            legend_title=map_legend_title,
        )

        out_file = out_dir / out_name
        out_pil.save(out_file, dpi=(args.output_dpi, args.output_dpi))
        print(f"  Saved {out_file}")

    if args.terrain_only:
        if dem_path is None:
            raise ValueError("--terrain-only requires a valid --dem path")
        render_one("terrain", dem_path, "terrain_multidirectional.png")
    else:
        render_one("predictions", pred_path, "predictions.png")
        render_one("suitability", suit_path, "suitability.png")
    print("Done.")


if __name__ == "__main__":
    main()
