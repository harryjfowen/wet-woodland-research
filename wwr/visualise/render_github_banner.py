#!/usr/bin/env python3
"""
render_github_banner.py

Wide, short double-band banner for the GitHub repo header.

  Top band    – Hysteresis extent probability  (band 2 of wet_woodland_extent.tif)
  Bottom band – MaxEnt habitat suitability     (wet_woodland_potential.tif)

Palette: exact PLASMA_LUT from wetwoodland-map/docs/index.html
  pale sage-green → teal → deep navy
England boundary: clipped to shape + thin coastline outline.
Background: white.

Centred on Okehampton, Devon (BNG: 259 000 E, 95 000 N).
Output: <repo-root>/outputs/images/github_banner.png
"""

import os
import numpy as np
from PIL import Image, ImageFilter
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.features import rasterize
import geopandas as gpd
from scipy.ndimage import binary_dilation

# ── paths ─────────────────────────────────────────────────────────────────────
WWR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO    = os.path.dirname(WWR)

EXTENT_TIF  = os.path.join(WWR, 'data/output/postprocess/wet_woodland_extent.tif')
SUIT_TIF    = os.path.join(WWR, 'data/output/potential/maxent/wet_woodland_potential.tif')
ENGLAND_SHP = os.path.join(WWR, 'data/input/boundaries/england.shp')
OUT_DIR     = os.path.join(WWR, 'visualise', 'output')
OUT_FILE    = os.path.join(OUT_DIR, 'github_banner.png')

os.makedirs(OUT_DIR, exist_ok=True)

# ── strip window centred on Okehampton (OSGB36 BNG metres) ───────────────────
OKE_E, OKE_N = 259_000, 95_000
HALF_H = 38_000

# Shift window ~110 km east so Cornwall sits near the left edge
# and East Anglia fills the right — same total width, less empty sea
west  = 90_000
east  = 650_000
south = OKE_N - HALF_H
north = OKE_N + HALF_H

IMG_W      = 1600
IMG_H_BAND =  200

# ── exact PLASMA_LUT from wetwoodland-map/docs/index.html ────────────────────
_STOPS = [
    (0.00, [236, 245, 232,   0]),
    (0.08, [236, 245, 232,  40]),
    (0.22, [204, 227, 205, 110]),
    (0.40, [150, 199, 186, 150]),
    (0.58, [ 92, 160, 168, 185]),
    (0.78, [ 46, 112, 141, 220]),
    (1.00, [  9,  35,  69, 245]),
]

def _build_lut(stops):
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        s0, s1 = stops[0], stops[1]
        for j in range(1, len(stops)):
            if t <= stops[j][0]:
                s0, s1 = stops[j - 1], stops[j]
                break
            s0, s1 = stops[j], stops[j]
        span = s1[0] - s0[0]
        f = max(0.0, min(1.0, 0.0 if span == 0 else (t - s0[0]) / span))
        for c in range(4):
            lut[i, c] = int(round(s0[1][c] + f * (s1[1][c] - s0[1][c])))
    return lut

LUT = _build_lut(_STOPS)

# ── rasterise England boundary to strip window ───────────────────────────────
print('Rasterising England boundary …')
england = gpd.read_file(ENGLAND_SHP)
# ensure BNG
england = england.to_crs('EPSG:27700')

strip_transform = transform_from_bounds(west, south, east, north, IMG_W, IMG_H_BAND)

eng_mask = rasterize(
    [(geom, 1) for geom in england.geometry],
    out_shape=(IMG_H_BAND, IMG_W),
    transform=strip_transform,
    fill=0,
    dtype=np.uint8,
)  # 1 = inside England, 0 = outside

# coastline outline: dilate the mask edge by 3px, 50% alpha
outline = binary_dilation(eng_mask, iterations=2).astype(np.uint8) - eng_mask

# ── windowed raster read ──────────────────────────────────────────────────────
def read_band(path, band_index):
    with rasterio.open(path) as src:
        win = from_bounds(west, south, east, north, src.transform)
        arr = src.read(
            band_index,
            window=win,
            out_shape=(IMG_H_BAND, IMG_W),
            resampling=Resampling.average,
            boundless=True,
            fill_value=np.nan,
        ).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

def stretch(arr, lo_pct=2, hi_pct=98):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return arr
    lo, hi = np.nanpercentile(valid, lo_pct), np.nanpercentile(valid, hi_pct)
    return np.where(np.isfinite(arr),
                    np.clip((arr - lo) / max(hi - lo, 1e-9), 0.0, 1.0),
                    np.nan)

def apply_lut(arr):
    mask = ~np.isfinite(arr)
    safe = np.where(mask, 0.0, arr)
    idx  = np.clip((safe * 255).astype(np.int32), 0, 255)
    rgba = LUT[idx].copy()
    rgba[mask] = [0, 0, 0, 0]
    return rgba.astype(np.uint8)

def composite_band(arr):
    """Apply LUT, clip to England, draw outline, composite onto white."""
    rgba = apply_lut(arr)

    # outside England → fully transparent
    rgba[eng_mask == 0] = [0, 0, 0, 0]

    # coastline outline: darkest palette colour at 20% alpha
    rgba[outline == 1] = [9, 35, 69, 26]

    # composite onto white
    img_rgba  = Image.fromarray(rgba, mode='RGBA')
    white     = Image.new('RGBA', img_rgba.size, (255, 255, 255, 255))
    white.paste(img_rgba, mask=img_rgba.split()[3])
    return np.array(white.convert('RGB'))

# ── read & render ─────────────────────────────────────────────────────────────
print('Reading hysteresis extent probabilities …')
hysteresis  = composite_band(stretch(read_band(EXTENT_TIF, 2)))

print('Reading MaxEnt suitability …')
suitability = composite_band(stretch(read_band(SUIT_TIF, 1)))

canvas = np.concatenate([hysteresis, suitability], axis=0)
Image.fromarray(canvas, mode='RGB').save(OUT_FILE, optimize=True)
print(f'Saved → {OUT_FILE}  ({IMG_W}×{canvas.shape[0]})')
