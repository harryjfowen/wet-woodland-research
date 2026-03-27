#!/usr/bin/env python3
"""
Render a bivariate map combining restoration suitability with land-value classes.

Default setup assumes:
- suitability raster in 0..1 (data/wet_woodland_potential.tif)
- land-value raster coded 0=Grade 1-2, 1=Grade 3, 2=Grade 4-5

The output uses:
- a cubehelix suitability ramp (low = very pale, high = darker/saturated)
- a land-value ramp where higher-grade land is cooler/greyer and lower-grade land
  is greener/earthier
- a square bivariate legend with suitability on columns and land on rows
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
for env_name, subdir in (
    ("MPLCONFIGDIR", "mpl"),
    ("XDG_CACHE_HOME", "xdg_cache"),
):
    os.environ.setdefault(env_name, str(TMP_ROOT / subdir))
    Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

try:
    import fiona
    from pyproj import Transformer
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform
    from shapely.ops import unary_union
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def _rgb_to_hex(rgb: np.ndarray) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _interpolate_hex_ramp(anchors: list[str], n: int) -> np.ndarray:
    rgb = np.array([_hex_to_rgb(c) for c in anchors], dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(anchors))
    dst = np.linspace(0.0, 1.0, n)
    out = np.empty((n, 3), dtype=np.float64)
    for ch in range(3):
        out[:, ch] = np.interp(dst, src, rgb[:, ch])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _interpolate_palette_grid(anchor_rows: list[list[str]], n: int) -> np.ndarray:
    anchor = np.array(
        [[_hex_to_rgb(cell) for cell in row] for row in anchor_rows],
        dtype=np.float64,
    )
    src = np.linspace(0.0, 1.0, anchor.shape[0])
    dst = np.linspace(0.0, 1.0, n)

    row_interp = np.empty((anchor.shape[0], n, 3), dtype=np.float64)
    for row in range(anchor.shape[0]):
        for ch in range(3):
            row_interp[row, :, ch] = np.interp(dst, src, anchor[row, :, ch])

    out = np.empty((n, n, 3), dtype=np.float64)
    for col in range(n):
        for ch in range(3):
            out[:, col, ch] = np.interp(dst, src, row_interp[:, col, ch])
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _cubehelix_rgb(
    n: int,
    *,
    start: float,
    rot: float,
    light: float,
    dark: float,
    hue: float = 1.0,
    reverse: bool = False,
    fallback: list[str],
) -> np.ndarray:
    if SEABORN_AVAILABLE:
        pal = sns.cubehelix_palette(
            n,
            start=start,
            rot=rot,
            light=light,
            dark=dark,
            hue=hue,
            reverse=reverse,
        )
        return np.clip(np.round(np.asarray(pal) * 255.0), 0, 255).astype(np.uint8)
    return _interpolate_hex_ramp(fallback, n)


def build_suitability_ramp(n: int) -> np.ndarray:
    return _cubehelix_rgb(
        n,
        start=2.0,
        rot=-1.0,
        light=1.0,
        dark=0.05,
        hue=1.05,
        fallback=[
            "#FFFFFF",
            "#FFF8E8",
            "#D7EDDB",
            "#7EC8C9",
            "#2E88BD",
            "#163E82",
            "#081D58",
        ],
    )


def build_land_ramp(n: int) -> np.ndarray:
    base = _cubehelix_rgb(
        n,
        start=0.45,
        rot=-0.55,
        light=0.84,
        dark=0.36,
        hue=0.85,
        reverse=True,
        fallback=[
            "#C6D1D4",
            "#A8B9B0",
            "#8CA588",
            "#6E8C63",
            "#57704C",
        ],
    ).astype(np.float64)
    cool = np.array([198.0, 205.0, 210.0], dtype=np.float64)
    earthy = np.array([92.0, 121.0, 82.0], dtype=np.float64)
    out = np.empty_like(base)
    for i in range(n):
        t = 0.0 if n == 1 else i / (n - 1)
        tint = ((1.0 - t) * cool) + (t * earthy)
        out[i] = (0.55 * base[i]) + (0.45 * tint)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def build_bivariate_palette(
    n: int,
    blend_weight_suitability: float = 0.68,
    preset: str = "cubehelix_blend",
) -> np.ndarray:
    if preset == "slate_olive_green":
        return _interpolate_palette_grid(
            [
                ["#e6ece8", "#94a49a", "#263c33"],
                ["#edf1df", "#a3a96a", "#4e5520"],
                ["#e7f1e7", "#7aa67a", "#1e4a2f"],
            ],
            n,
        )
    if preset == "green_blue_black":
        return _interpolate_palette_grid(
            [
                ["#e5edef", "#76939d", "#0d1a21"],
                ["#edf1df", "#94a55f", "#202712"],
                ["#e6f1e8", "#5f9d79", "#0a2017"],
            ],
            n,
        )
    if preset == "stevens_green_blue":
        # Bivariate palette: green (land value) × blue (suitability).
        # Rows: land value class (0=Grade 1-2 high ag, 2=Grade 4-5 low ag → green axis).
        # Cols: suitability (0=low, 2=high → blue axis).
        # Diamond corners:
        #   TOP    palette[2,2] = near-black  (both high — the hotspot)
        #   LEFT   palette[2,0] = strong green (high land value, low suit)
        #   RIGHT  palette[0,2] = strong blue  (high suit, Grade 1-2 land)
        #   BOTTOM palette[0,0] = near-white   (both low)
        return _interpolate_palette_grid(
            [
                ["#CABED0", "#89A1C8", "#4885C1"],
                ["#BC7C8F", "#806A8A", "#435786"],
                ["#AE3A4E", "#77324C", "#3F2949"],
            ],
            n,
        )

    suit = build_suitability_ramp(n).astype(np.float64)
    land = build_land_ramp(n).astype(np.float64)
    w_s = float(np.clip(blend_weight_suitability, 0.0, 1.0))
    w_l = 1.0 - w_s
    palette = np.empty((n, n, 3), dtype=np.uint8)
    for land_idx in range(n):
        for suit_idx in range(n):
            rgb = (w_s * suit[suit_idx]) + (w_l * land[land_idx])
            palette[land_idx, suit_idx] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return palette


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: list[str]
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
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _nice_scale_km(target_km: float) -> float:
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


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    width, _ = _text_size(draw, text, font)
    if width <= max_width or " " not in text:
        return [text]

    words = text.split()
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_width, _ = _text_size(draw, candidate, font)
        if candidate_width <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def _lookup_property(properties, field_name):
    if field_name in properties:
        return properties[field_name]
    field_name_lower = str(field_name).lower()
    for key, value in properties.items():
        if str(key).lower() == field_name_lower:
            return value
    raise KeyError(field_name)


def _map_alc_grade_group(raw_value):
    value = str(raw_value).strip().lower()
    if value in {"grade 1", "grade 2"}:
        return 0
    if value == "grade 3":
        return 1
    if value in {"grade 4", "grade 5", "non agricultural"}:
        return 2
    return None


def load_clip_geometry(
    boundary_path: Path,
    target_crs,
    buffer_m: float = 0.0,
    *,
    property_name: str | None = None,
    property_values: set[str] | None = None,
) -> dict | None:
    if boundary_path is None:
        return None
    if not GEO_AVAILABLE:
        raise ValueError("Clip boundary requested but fiona/shapely/pyproj are not installed.")
    with fiona.open(boundary_path) as src:
        src_crs_raw = src.crs_wkt or src.crs
        geoms = []
        for feat in src:
            if feat.get("geometry") is None:
                continue
            if property_name is not None and property_values is not None:
                raw_value = feat["properties"].get(property_name)
                if raw_value is None or str(raw_value) not in property_values:
                    continue
            geoms.append(shape(feat["geometry"]))
    if not geoms:
        raise ValueError(f"No geometries found in clip boundary: {boundary_path}")

    merged = unary_union(geoms)
    src_crs = rasterio.crs.CRS.from_user_input(src_crs_raw) if src_crs_raw else None
    if src_crs is not None and target_crs is not None and src_crs != target_crs:
        transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
        merged = shapely_transform(transformer.transform, merged)
    if buffer_m != 0:
        merged = merged.buffer(buffer_m)
    if not merged.is_valid:
        merged = merged.buffer(0)
    return mapping(merged)


def apply_hatched_overlay(
    image: Image.Image,
    mask: np.ndarray,
    *,
    spacing: int = 18,
    line_width: int = 2,
    fill_rgb: tuple[int, int, int] = (255, 255, 255),
    line_rgb: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    if image.mode != "RGB":
        base = image.convert("RGB")
    else:
        base = image.copy()

    if mask.size == 0 or not np.any(mask):
        return base

    arr = np.asarray(base).copy()
    arr[mask] = np.array(fill_rgb, dtype=np.uint8)

    yy, xx = np.indices(mask.shape)
    spacing = max(6, int(spacing))
    line_width = max(1, int(line_width))
    hatch_a = np.mod(xx + yy, spacing) < line_width
    hatch_b = np.mod(xx - yy, spacing) < line_width
    hatch = (hatch_a | hatch_b) & mask
    arr[hatch] = np.array(line_rgb, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def build_multidirectional_terrain(
    dem: np.ndarray,
    x_res: float,
    y_res: float,
    azimuths: list[float] | None = None,
    altitude: float = 45.0,
    gamma: float = 0.7,
    z_factor: float = 1.5,
) -> np.ndarray:
    if azimuths is None:
        azimuths = [45.0, 135.0, 225.0, 315.0]
    dem_f = dem.astype(np.float64)
    valid = np.isfinite(dem_f)
    if not valid.any():
        return np.ones_like(dem_f)

    dem_fill = np.where(valid, dem_f, np.nanmedian(dem_f[valid]))
    gy, gx = np.gradient(dem_fill, y_res, x_res)
    slope_rad = np.arctan(z_factor * np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)
    altitude_rad = np.deg2rad(altitude)

    hillshades = []
    for az in azimuths:
        az_rad = np.deg2rad(az)
        shade = (
            np.cos(altitude_rad) * np.cos(slope_rad)
            + np.sin(altitude_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect)
        )
        hillshades.append(np.clip(shade, 0, 1))

    md_mean = np.mean(np.stack(hillshades, axis=0), axis=0)
    md_lo = np.nanpercentile(md_mean[valid], 2)
    md_hi = np.nanpercentile(md_mean[valid], 98)
    luminance = np.clip((md_mean - md_lo) / max(md_hi - md_lo, 1e-6), 0, 1)
    return np.where(valid, np.power(luminance, gamma), np.nan)


def resolve_output_dims(width_units: float, height_units: float, size: int | None) -> tuple[int, int]:
    if size is None:
        return max(1, int(round(width_units))), max(1, int(round(height_units)))
    if width_units >= height_units:
        out_w = size
        out_h = max(1, int(round(size * height_units / width_units)))
    else:
        out_h = size
        out_w = max(1, int(round(size * width_units / height_units)))
    return out_w, out_h


def reproject_to_grid(
    raster_path: Path,
    band: int,
    out_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    *,
    resampling: Resampling,
    dst_nodata,
) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        dest = np.full(out_shape, dst_nodata, dtype=np.float32 if np.isnan(dst_nodata) else np.float32)
        reproject(
            source=rasterio.band(src, band),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
        return dest


def load_landvalue_classes(
    landvalue_path: Path,
    *,
    out_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    band: int = 1,
    grade_field: str = "alc_grade",
) -> np.ndarray:
    if landvalue_path.suffix.lower() == ".shp":
        if not GEO_AVAILABLE:
            raise RuntimeError("fiona, shapely, and pyproj are required to rasterize ALC polygons.")
        shapes = []
        with fiona.open(landvalue_path) as src:
            src_crs_raw = src.crs_wkt or src.crs
            src_crs = rasterio.crs.CRS.from_user_input(src_crs_raw) if src_crs_raw else None
            transformer = None
            if src_crs is not None and dst_crs is not None and src_crs != dst_crs:
                transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

            for feat in src:
                if feat.get("geometry") is None:
                    continue
                properties = feat.get("properties") or {}
                try:
                    raw_grade = _lookup_property(properties, grade_field)
                except KeyError:
                    continue
                grade_group = _map_alc_grade_group(raw_grade)
                if grade_group is None:
                    continue
                geom = shape(feat["geometry"])
                if transformer is not None:
                    geom = shapely_transform(transformer.transform, geom)
                if geom.is_empty:
                    continue
                shapes.append((mapping(geom), grade_group))

        if not shapes:
            raise ValueError(
                f"No ALC polygons could be mapped from {landvalue_path} using field '{grade_field}'."
            )

        return rasterize(
            shapes,
            out_shape=out_shape,
            transform=dst_transform,
            fill=255,
            dtype=np.uint8,
        ).astype(np.int16)

    landvalue = reproject_to_grid(
        landvalue_path,
        band,
        out_shape,
        dst_transform,
        dst_crs,
        resampling=Resampling.nearest,
        dst_nodata=255,
    )
    return np.round(landvalue).astype(np.int16)


def classify_suitability(values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(values.shape, -1, dtype=np.int16)
    valid = np.isfinite(values)
    if not valid.any():
        return out

    vals = values.copy()
    lo = float(np.nanmin(vals[valid]))
    hi = float(np.nanmax(vals[valid]))
    if lo < 0.0 or hi > 1.0:
        vals = np.where(valid, np.clip((vals - lo) / max(hi - lo, 1e-9), 0, 1), np.nan)
    else:
        vals = np.where(valid, np.clip(vals, 0, 1), np.nan)

    edges = np.linspace(0.0, 1.0, n + 1)
    cls = np.digitize(vals, edges[1:-1], right=False)
    out[valid] = np.clip(cls[valid], 0, n - 1).astype(np.int16)
    return out


def add_bivariate_decorations(
    image: Image.Image,
    palette: np.ndarray,
    land_labels: list[str],
    suit_labels: list[str],
    legend_title: str,
    map_width_m: float | None,
    add_scale_bar: bool,
    legend_scale: float = 1.0,
) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    margin = max(20, int(0.025 * min(w, h)))
    n = palette.shape[0]
    base = min(w, h)
    legend_scale = max(0.5, float(legend_scale))
    base_title_size = max(22, min(46, int(base * 0.0155)))
    base_label_size = max(16, min(28, int(base * 0.0092)))
    base_axis_size = max(14, min(21, int(base * 0.0068)))
    title_size = int(round(base_title_size * legend_scale))
    label_size = int(round(base_label_size * legend_scale))
    axis_size = int(round(base_axis_size * legend_scale))
    title_font = _load_font(title_size, bold=True)
    label_font = _load_font(label_size)
    axis_font = _load_font(axis_size, bold=True)
    small_font = _load_font(max(11, int(round((base_axis_size - 1) * legend_scale))))

    cell_max = 108 if n <= 3 else 72
    base_cell = max(32, min(cell_max, int(0.036 * base)))
    cell = int(round(base_cell * legend_scale))
    grid_w = cell * n
    grid_h = cell * n

    title_w, title_h = _text_size(draw, legend_title, title_font)
    land_axis = "Land value"
    suit_axis = "Suitability"
    land_axis_w, land_axis_h = _text_size(draw, land_axis, axis_font)
    suit_axis_w, suit_axis_h = _text_size(draw, suit_axis, axis_font)

    row_label_widths = [_text_size(draw, label, label_font)[0] for label in land_labels[:n]]
    row_label_heights = [_text_size(draw, label, label_font)[1] for label in land_labels[:n]]
    widest_row_label = max(row_label_widths, default=0)
    row_block_w = max(widest_row_label, land_axis_w)

    col_lines = [_wrap_label(draw, label, small_font, cell + 18) for label in suit_labels[:n]]
    col_line_h = _text_size(draw, "Ag", small_font)[1]
    col_block_h = max((len(lines) * col_line_h) + ((len(lines) - 1) * 2) for lines in col_lines)

    panel_pad = int(round(max(12, base_cell // 5) * legend_scale))
    row_gap = int(round(max(16, base_cell // 4) * legend_scale))
    title_gap = int(round(max(14, base_cell // 4) * legend_scale))
    axis_gap = int(round(max(12, base_cell // 5) * legend_scale))
    col_gap = int(round(max(12, base_cell // 5) * legend_scale))

    legend_w = panel_pad + row_block_w + row_gap + grid_w + panel_pad
    legend_h = (
        panel_pad
        + title_h
        + title_gap
        + max(land_axis_h, suit_axis_h)
        + axis_gap
        + grid_h
        + col_gap
        + col_block_h
        + panel_pad
    )

    panel_x0 = margin
    panel_y0 = margin

    title_x = panel_x0 + panel_pad
    title_y = panel_y0 + panel_pad
    draw.text((title_x, title_y), legend_title, fill=(0, 0, 0), font=title_font)

    axis_y = title_y + title_h + title_gap
    row_label_x = panel_x0 + panel_pad
    grid_x = row_label_x + row_block_w + row_gap
    grid_y = axis_y + max(land_axis_h, suit_axis_h) + axis_gap

    draw.text(
        (row_label_x + max(0, (row_block_w - land_axis_w) // 2), axis_y),
        land_axis,
        fill=(35, 35, 35),
        font=axis_font,
    )
    draw.text(
        (grid_x + max(0, (grid_w - suit_axis_w) // 2), axis_y),
        suit_axis,
        fill=(35, 35, 35),
        font=axis_font,
    )

    for row in range(n):
        for col in range(n):
            x0 = grid_x + (col * cell)
            y0 = grid_y + (row * cell)
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=tuple(int(v) for v in palette[row, col]), outline=(255, 255, 255), width=1)

    for idx, label in enumerate(land_labels[:n]):
        label_w = row_label_widths[idx]
        label_h = row_label_heights[idx]
        ty = grid_y + idx * cell + (cell - label_h) // 2
        tx = row_label_x + row_block_w - label_w
        draw.text((tx, ty), label, fill=(0, 0, 0), font=label_font)

    col_y = grid_y + grid_h + col_gap
    for idx, lines in enumerate(col_lines):
        total_h = (len(lines) * col_line_h) + ((len(lines) - 1) * 2)
        text_y = col_y + max(0, (col_block_h - total_h) // 2)
        for line_idx, line in enumerate(lines):
            line_w, _ = _text_size(draw, line, small_font)
            tx = grid_x + idx * cell + max(0, (cell - line_w) // 2)
            ty = text_y + line_idx * (col_line_h + 2)
            draw.text((tx, ty), line, fill=(0, 0, 0), font=small_font)

    if add_scale_bar and map_width_m and map_width_m > 0:
        map_width_km = map_width_m / 1000.0
        scale_km = _nice_scale_km(max(1.0, map_width_km * 0.14))
        bar_px = max(40, int(round((scale_km / map_width_km) * w)))
        x0 = max(margin, (w - bar_px) // 2)
        y0 = h - margin - 40
        x1 = min(w - margin, x0 + bar_px)
        draw.line((x0, y0, x1, y0), fill=(255, 255, 255), width=10)
        draw.line((x0, y0, x1, y0), fill=(0, 0, 0), width=6)
        for x in (x0, x1):
            draw.line((x, y0 - 9, x, y0 + 9), fill=(255, 255, 255), width=8)
            draw.line((x, y0 - 9, x, y0 + 9), fill=(0, 0, 0), width=4)
        label = f"{scale_km:g} km"
        draw.text((x0 + (bar_px // 2) - 24, y0 + 14), label, fill=(0, 0, 0), font=axis_font)

    return out


def draw_diamond_legend(
    image: Image.Image,
    palette: np.ndarray,
    land_labels: list[str],
    suit_labels: list[str],
    legend_title: str,
    legend_scale: float = 1.0,
) -> Image.Image:
    """Overlay a rotated-diamond bivariate legend.

    Diamond orientation:
      TOP    = palette[n-1, n-1]  (high suitability + Grade 4-5 land) — the peak
      BOTTOM = palette[0,   0  ]  (low suitability  + Grade 1-2 land) — both low
      RIGHT  = palette[0,   n-1]  (high suitability + Grade 1-2 land) — suit only
      LEFT   = palette[n-1, 0  ]  (low suitability  + Grade 4-5 land) — land only
    """
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    margin = max(20, int(0.025 * min(w, h)))
    n = palette.shape[0]
    base = min(w, h)
    legend_scale = max(0.5, float(legend_scale))

    title_size = int(round(max(22, min(46, int(base * 0.0155))) * legend_scale))
    label_size = int(round(max(16, min(28, int(base * 0.0092))) * legend_scale))
    axis_size  = int(round(max(14, min(21, int(base * 0.0068))) * legend_scale))

    title_font = _load_font(title_size, bold=True)
    label_font = _load_font(label_size)
    axis_font  = _load_font(axis_size, bold=True)

    cell_max  = 108 if n <= 3 else 72
    half_cell = int(round(max(32, min(cell_max, int(0.036 * base))) * legend_scale))
    # half_cell is the half-diagonal of every small cell diamond.
    # The full legend diamond spans ±n*half_cell from its centre.
    diamond_r = n * half_cell

    gap       = max(8, half_cell // 4)
    panel_pad = max(12, half_cell // 4)

    title_w, title_h = _text_size(draw, legend_title, title_font)

    # Corner labels: right = high suitability, left = high land value.
    right_label = suit_labels[-1] if suit_labels else "High"
    left_label  = land_labels[-1] if land_labels else "Grade 4-5"
    rl_w, rl_h  = _text_size(draw, right_label, label_font)
    ll_w, ll_h  = _text_size(draw, left_label, label_font)

    # Diamond centre: budget space for title above, corner labels on sides.
    cx = margin + panel_pad + ll_w + gap + diamond_r
    cy = margin + panel_pad + title_h + gap + diamond_r

    # Title (centred above the top vertex).
    draw.text((cx - title_w // 2, margin + panel_pad), legend_title, fill=(0, 0, 0), font=title_font)

    # Draw each palette cell as a rotated-square (diamond) polygon.
    # Cell (row, col) centre in image coords:
    #   dx =  (col - row) * half_cell   → right = high suit & low land
    #   dy = -(row + col - (n-1)) * half_cell  → up = high row+col = both high at top
    for row in range(n):
        for col in range(n):
            dx  = (col - row) * half_cell
            dy  = -(row + col - (n - 1)) * half_cell
            ccx = cx + dx
            ccy = cy + dy
            pts = [
                (ccx,            ccy - half_cell),  # top
                (ccx + half_cell, ccy),              # right
                (ccx,            ccy + half_cell),   # bottom
                (ccx - half_cell, ccy),              # left
            ]
            draw.polygon(pts, fill=tuple(int(v) for v in palette[row, col]), outline=(255, 255, 255), width=1)

    # Corner labels.
    draw.text((cx + diamond_r + gap, cy - rl_h // 2), right_label, fill=(40, 40, 40), font=label_font)
    draw.text((cx - diamond_r - ll_w - gap, cy - ll_h // 2), left_label, fill=(40, 40, 40), font=label_font)

    # Axis labels centred under the bottom-right and bottom-left halves.
    suit_ax = "Suitability"
    land_ax = "Land value"
    sa_w, sa_h = _text_size(draw, suit_ax, axis_font)
    la_w, la_h = _text_size(draw, land_ax, axis_font)
    ax_y = cy + diamond_r + gap
    draw.text((cx + diamond_r // 2 - sa_w // 2, ax_y), suit_ax, fill=(60, 60, 60), font=axis_font)
    draw.text((cx - diamond_r // 2 - la_w // 2, ax_y), land_ax, fill=(60, 60, 60), font=axis_font)

    return out


def default_labels(n: int) -> tuple[list[str], list[str]]:
    if n == 3:
        return ["Grade 1-2", "Grade 3", "Grade 4-5"], ["Low", "Medium", "High"]
    if n == 5:
        return ["1", "2", "3", "4", "5"], ["Very low", "Low", "Medium", "High", "Very high"]
    return [str(i + 1) for i in range(n)], [str(i + 1) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a bivariate suitability x land-value map.")
    parser.add_argument("--suitability", default="data/output/potential/maxent/wet_woodland_potential.tif", help="Suitability raster (0-1).")
    parser.add_argument("--suitability-band", type=int, default=1, help="Suitability band index (1-based).")
    parser.add_argument("--landvalue", default="data/input/boundaries/agricultural_land_classification.shp", help="Land-value raster or ALC shapefile.")
    parser.add_argument("--landvalue-band", type=int, default=1, help="Land-value band index (1-based).")
    parser.add_argument("--landvalue-field", default="alc_grade", help="ALC field name when --landvalue is a shapefile.")
    parser.add_argument("--n-classes", type=int, default=3, choices=[3, 5], help="Bivariate grid size (3 or 5).")
    parser.add_argument("--size", type=int, default=3500, help="Longest output side in pixels.")
    parser.add_argument("--clip-boundary", default="data/input/boundaries/england.shp", help="Boundary to clip outputs.")
    parser.add_argument("--clip-buffer-m", type=float, default=-100.0, help="Buffer for clip boundary in metres.")
    parser.add_argument("--dem", default="data/output/potential/potential_predictors_100m.tif", help="Optional DEM for terrain underlay.")
    parser.add_argument("--dem-band", type=int, default=1, help="DEM band index (1-based).")
    parser.add_argument("--terrain-strength", type=float, default=0.22, help="Terrain modulation strength (0-1).")
    parser.add_argument("--terrain-gamma", type=float, default=0.72, help="Gamma for multidirectional terrain.")
    parser.add_argument("--terrain-z-factor", type=float, default=1.5, help="Terrain exaggeration factor.")
    parser.add_argument("--urban-overlay", default="data/input/boundaries/agricultural_land_classification.shp", help="Polygon layer used for urban hatch overlay.")
    parser.add_argument("--urban-field", default="alc_grade", help="Attribute field used to select urban polygons.")
    parser.add_argument("--urban-value", default="Urban", help="Attribute value used to select urban polygons.")
    parser.add_argument("--urban-hatch-spacing", type=int, default=18, help="Urban hatch spacing in output pixels.")
    parser.add_argument("--urban-hatch-width", type=int, default=2, help="Urban hatch line width in output pixels.")
    parser.add_argument("--no-urban-overlay", action="store_true", help="Disable the urban hatch overlay.")
    parser.add_argument("--legend-scale", type=float, default=1.5, help="Scale factor for the bivariate legend.")
    parser.add_argument(
        "--palette-preset",
        choices=["cubehelix_blend", "slate_olive_green", "green_blue_black", "stevens_green_blue"],
        default="cubehelix_blend",
        help="Bivariate palette preset. cubehelix_blend is the original generated blend; alternate presets use fixed 3x3 editorial matrices.",
    )
    parser.add_argument("--blend-weight-suitability", type=float, default=0.68, help="Weight given to suitability ramp when blending the bivariate palette.")
    parser.add_argument("--legend-diamond", action="store_true", help="Render the legend as a rotated diamond (peak combination at top) instead of a square grid.")
    parser.add_argument("--output", default="visualise/output/bivariate_suitability_landvalue.png", help="Output PNG path.")
    parser.add_argument("--palette-json", default="visualise/output/bivariate_suitability_landvalue_palette.json", help="Optional palette matrix JSON path.")
    parser.add_argument("--legend-title", default="Suitability × land value", help="Legend title.")
    parser.add_argument("--pillow-contrast", type=float, default=1.06, help="Final contrast enhancement factor.")
    parser.add_argument("--pillow-sharpness", type=float, default=1.0, help="Final sharpness enhancement factor.")
    parser.add_argument("--no-scale-bar", action="store_true", help="Disable scale bar.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    def _resolve_path(raw: str | None) -> Path | None:
        if not raw:
            return None
        path = Path(raw)
        if path.is_absolute():
            return path
        return repo_root / path

    suit_path = _resolve_path(args.suitability)
    land_path = _resolve_path(args.landvalue)
    dem_path = _resolve_path(args.dem)
    urban_overlay_path = _resolve_path(args.urban_overlay)
    out_path = _resolve_path(args.output)
    palette_json_path = _resolve_path(args.palette_json) if args.palette_json else None
    clip_boundary_path = _resolve_path(args.clip_boundary)

    if not suit_path.exists():
        raise FileNotFoundError(f"Suitability raster not found: {suit_path}")
    if not land_path.exists():
        raise FileNotFoundError(f"Land-value raster not found: {land_path}")

    with rasterio.open(suit_path) as suit_src:
        crs = suit_src.crs
        src_bounds = suit_src.bounds

    clip_geom = None
    if args.clip_boundary:
        clip_geom = load_clip_geometry(clip_boundary_path, crs, buffer_m=args.clip_buffer_m)

    if clip_geom is not None:
        clip_left, clip_bottom, clip_right, clip_top = shape(clip_geom).bounds
        left = max(src_bounds.left, clip_left)
        bottom = max(src_bounds.bottom, clip_bottom)
        right = min(src_bounds.right, clip_right)
        top = min(src_bounds.top, clip_top)
    else:
        left, bottom, right, top = src_bounds

    if not (right > left and top > bottom):
        raise ValueError("Derived output bounds are empty.")

    out_w, out_h = resolve_output_dims(right - left, top - bottom, args.size)
    dst_transform = from_bounds(left, bottom, right, top, out_w, out_h)

    suitability = reproject_to_grid(
        suit_path,
        args.suitability_band,
        (out_h, out_w),
        dst_transform,
        crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )
    landvalue = load_landvalue_classes(
        land_path,
        out_shape=(out_h, out_w),
        dst_transform=dst_transform,
        dst_crs=crs,
        band=args.landvalue_band,
        grade_field=args.landvalue_field,
    )

    clip_inside = np.ones((out_h, out_w), dtype=bool)
    if clip_geom is not None:
        clip_inside = geometry_mask([clip_geom], out_shape=(out_h, out_w), transform=dst_transform, invert=True)

    urban_mask = np.zeros((out_h, out_w), dtype=bool)
    if not args.no_urban_overlay and urban_overlay_path is not None and urban_overlay_path.exists():
        urban_geom = load_clip_geometry(
            urban_overlay_path,
            crs,
            property_name=args.urban_field,
            property_values={args.urban_value},
        )
        urban_mask = geometry_mask([urban_geom], out_shape=(out_h, out_w), transform=dst_transform, invert=True) & clip_inside

    suit_class = classify_suitability(suitability, args.n_classes)
    land_valid = (landvalue >= 0) & (landvalue < args.n_classes)
    valid = clip_inside & np.isfinite(suitability) & (suit_class >= 0) & land_valid

    palette = build_bivariate_palette(
        args.n_classes,
        blend_weight_suitability=args.blend_weight_suitability,
        preset=args.palette_preset,
    )

    terrain = np.ones((out_h, out_w), dtype=np.float64)
    if dem_path is not None and dem_path.exists():
        dem = reproject_to_grid(
            dem_path,
            args.dem_band,
            (out_h, out_w),
            dst_transform,
            crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        x_res = abs(dst_transform.a)
        y_res = abs(dst_transform.e)
        terrain = build_multidirectional_terrain(
            dem,
            x_res=x_res,
            y_res=y_res,
            gamma=args.terrain_gamma,
            z_factor=args.terrain_z_factor,
        )
        terrain = np.where(np.isfinite(terrain), terrain, 1.0)

    background = np.clip((0.88 + 0.12 * terrain) * 255.0, 0, 255).astype(np.uint8)
    composite = np.stack([background, background, background], axis=-1).astype(np.float64)

    thematic = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    thematic[valid] = palette[landvalue[valid], suit_class[valid]]
    shade_factor = ((1.0 - args.terrain_strength) + (args.terrain_strength * terrain))[:, :, np.newaxis]
    composite[valid] = np.clip(thematic[valid].astype(np.float64) * shade_factor[valid], 0, 255)

    outside = ~clip_inside
    composite[outside] = 255
    out_img = Image.fromarray(composite.astype(np.uint8), mode="RGB")
    if args.pillow_contrast != 1.0:
        out_img = ImageEnhance.Contrast(out_img).enhance(args.pillow_contrast)
    if args.pillow_sharpness != 1.0:
        out_img = ImageEnhance.Sharpness(out_img).enhance(args.pillow_sharpness)
    if np.any(urban_mask):
        out_img = apply_hatched_overlay(
            out_img,
            urban_mask,
            spacing=args.urban_hatch_spacing,
            line_width=args.urban_hatch_width,
        )

    land_labels, suit_labels = default_labels(args.n_classes)
    use_diamond = args.legend_diamond
    if use_diamond:
        out_img = draw_diamond_legend(
            out_img,
            palette,
            land_labels=land_labels,
            suit_labels=suit_labels,
            legend_title=args.legend_title,
            legend_scale=args.legend_scale,
        )
    else:
        out_img = add_bivariate_decorations(
            out_img,
            palette,
            land_labels=land_labels,
            suit_labels=suit_labels,
            legend_title=args.legend_title,
            map_width_m=right - left,
            add_scale_bar=not args.no_scale_bar,
            legend_scale=args.legend_scale,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path, dpi=(300, 300))
    print(f"Saved {out_path}")

    if palette_json_path is not None:
        palette_json_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = [[_rgb_to_hex(palette[r, c]) for c in range(args.n_classes)] for r in range(args.n_classes)]
        payload = {
            "legend_title": args.legend_title,
            "palette_preset": args.palette_preset,
            "land_labels": land_labels[: args.n_classes],
            "suitability_labels": suit_labels[: args.n_classes],
            "palette": matrix,
        }
        palette_json_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved {palette_json_path}")


if __name__ == "__main__":
    main()
