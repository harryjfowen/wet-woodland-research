#!/usr/bin/env python3
"""
Render a single national wet woodland predictions panel from the hysteresis raster.

The map uses the binary hysteresis band, averaged to a coarser grid so the national
pattern is readable at publication scale, then re-expanded with nearest-neighbour
to keep a crisp block structure.
"""

import argparse
import os
from pathlib import Path
from io import BytesIO

TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
for env_name, subdir in (
    ("MPLCONFIGDIR", "mpl"),
    ("XDG_CACHE_HOME", "xdg_cache"),
):
    os.environ.setdefault(env_name, str(TMP_ROOT / subdir))
    Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
from PIL import Image
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
import requests
import xyzservices.providers as xyz

try:
    import fiona
    from pyproj import Transformer
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform, unary_union
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

REPO = Path(__file__).resolve().parent.parent
CITY_LABEL_OFFSET_SCALE = 1.2


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO / path


def build_multidirectional_terrain(
    dem,
    x_res,
    y_res,
    azimuths=None,
    altitude=45.0,
    gamma=0.7,
    z_factor=1.5,
):
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
    alt_rad = np.deg2rad(altitude)
    hillshades = []
    for az in azimuths:
        az_rad = np.deg2rad(az)
        shade = (
            np.cos(alt_rad) * np.cos(slope_rad)
            + np.sin(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect)
        )
        hillshades.append(np.clip(shade, 0, 1))
    md = np.mean(np.stack(hillshades), axis=0)
    lo = np.nanpercentile(md[valid], 2)
    hi = np.nanpercentile(md[valid], 98)
    lum = np.clip((md - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.where(valid, np.power(lum, gamma), np.nan)


def load_clip_geometry(boundary_path, target_crs, buffer_m=0.0):
    if not GEO_AVAILABLE:
        return None
    with fiona.open(boundary_path) as src:
        src_crs_raw = src.crs_wkt or src.crs
        geoms = [shape(f["geometry"]) for f in src if f.get("geometry")]
    merged = unary_union(geoms)
    src_crs = rasterio.crs.CRS.from_user_input(src_crs_raw)
    if src_crs != target_crs:
        transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
        merged = shapely_transform(transformer.transform, merged)
    if buffer_m:
        merged = merged.buffer(buffer_m)
    return mapping(merged)


def build_predictions_colormap(palette_name):
    if palette_name == "viridis":
        cmap = plt.get_cmap("viridis").copy()
    elif palette_name == "viridis_r":
        cmap = plt.get_cmap("viridis_r").copy()
    elif palette_name == "turbo":
        cmap = plt.get_cmap("turbo").copy()
    elif palette_name == "cividis_r":
        cmap = plt.get_cmap("cividis_r").copy()
    elif palette_name == "editorial":
        try:
            import seaborn as sns

            colors = sns.cubehelix_palette(
                256, start=2.0, rot=-1.0, light=0.96, dark=0.05, hue=1.8
            )
            sandy = np.array([0.76, 0.72, 0.65])
            n_blend = 60
            for i in range(n_blend):
                t = i / n_blend
                colors[i] = (1 - t) * sandy + t * np.array(colors[i])
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "pred_ramp", colors, N=256
            )
        except ImportError:
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "pred_fallback",
                ["#FFFFFF", "#D7EDDB", "#7EC8C9", "#2E88BD", "#081D58"],
                N=256,
            )
    elif palette_name == "forest_teal":
        # Matches the PLASMA_LUT used in the deck.gl web viewer and GitHub banner:
        # pale sage-green → teal → deep navy
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "forest_teal",
            [
                (0.00, "#ECF5E8"),
                (0.08, "#ECF5E8"),
                (0.22, "#CCE3CD"),
                (0.40, "#96C7BA"),
                (0.58, "#5CA0A8"),
                (0.78, "#2E708D"),
                (1.00, "#092345"),
            ],
            N=256,
        )
    else:
        cmap = plt.get_cmap("cividis").copy()

    cmap.set_bad(color="white")
    return cmap


def build_quantile_scaled_values(values, valid_mask):
    """
    Map values to empirical rank so the palette uses equal visual space across
    the observed distribution rather than raw linear percentage.
    """
    scaled = np.full(values.shape, np.nan, dtype=np.float32)
    if not np.any(valid_mask):
        return scaled, np.array([0.0, 0.25, 0.5, 0.75, 1.0]), np.zeros(5, dtype=np.float32)

    # Use positive values when available so zero-heavy backgrounds do not eat
    # the entire lower part of the palette.
    scale_mask = valid_mask & (values > 0)
    if not np.any(scale_mask):
        scale_mask = valid_mask

    scale_values = values[scale_mask].astype(np.float64)
    unique_vals, inverse, counts = np.unique(
        scale_values, return_inverse=True, return_counts=True
    )
    if scale_values.size == 1:
        rank_values = np.array([1.0], dtype=np.float64)
    else:
        cum_counts = np.cumsum(counts).astype(np.float64)
        rank_values = (cum_counts - 1.0) / max(scale_values.size - 1.0, 1.0)

    scaled_values = rank_values[inverse]
    scaled[scale_mask] = scaled_values.astype(np.float32)
    scaled[valid_mask & ~scale_mask] = 0.0

    tick_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    tick_values = np.quantile(scale_values, tick_positions).astype(np.float32)
    if np.any(valid_mask & (values <= 0)):
        tick_values[0] = 0.0
    return scaled, tick_positions, tick_values


def compute_north_profile(values, valid_mask, smooth_px=11):
    """
    Mean wet woodland cover per north-south row, optionally smoothed with a
    short moving-average window for a cleaner marginal profile.
    """
    row_means = np.full(values.shape[0], np.nan, dtype=np.float32)
    valid_counts = valid_mask.sum(axis=1)
    has_values = valid_counts > 0
    if np.any(has_values):
        row_sums = np.where(valid_mask, values, 0.0).sum(axis=1)
        row_means[has_values] = (row_sums[has_values] / valid_counts[has_values]).astype(
            np.float32
        )
        row_means[~has_values] = 0.0

    window = max(1, int(round(smooth_px)))
    if window <= 1 or not np.any(np.isfinite(row_means)):
        return row_means

    if window % 2 == 0:
        window += 1

    weights = np.ones(window, dtype=np.float32) / float(window)
    valid_float = np.isfinite(row_means).astype(np.float32)
    filled = np.where(np.isfinite(row_means), row_means, 0.0).astype(np.float32)
    smooth_vals = np.convolve(filled, weights, mode="same")
    smooth_weights = np.convolve(valid_float, weights, mode="same")
    out = np.divide(
        smooth_vals,
        smooth_weights,
        out=np.zeros_like(smooth_vals),
        where=smooth_weights > 0,
    )
    return out.astype(np.float32)


def gaussian_smooth_density(values, support, sigma_cells):
    """
    Smooth a cover-fraction surface with Gaussian blur while normalizing by
    support so coastlines and clipped edges are not artificially depressed.
    """
    from scipy import ndimage

    sigma = float(max(sigma_cells, 0.0))
    if sigma <= 0:
        return values.astype(np.float32)

    finite = np.isfinite(values) & (support > 0)
    numer = np.where(finite, values * support, 0.0).astype(np.float32)
    denom = np.where(finite, support, 0.0).astype(np.float32)

    numer_s = ndimage.gaussian_filter(numer, sigma=sigma, mode="constant", cval=0.0)
    denom_s = ndimage.gaussian_filter(denom, sigma=sigma, mode="constant", cval=0.0)

    out = np.full(values.shape, np.nan, dtype=np.float32)
    np.divide(numer_s, denom_s, out=out, where=denom_s > 1e-6)
    return out


def load_inset_raster(raster_path, band, center_x, center_y, size_m):
    half = float(size_m) * 0.5
    xmin, ymin = center_x - half, center_y - half
    xmax, ymax = center_x + half, center_y + half
    with rasterio.open(raster_path) as src:
        window = window_from_bounds(xmin, ymin, xmax, ymax, src.transform)
        window = window.round_offsets().round_lengths()
        data = src.read(band, window=window, masked=True)
        transform = src.window_transform(window)
    return data, transform, (xmin, ymin, xmax, ymax)


def draw_probability_inset(
    fig,
    host_ax,
    pred_path,
    crs,
    left,
    bottom,
    right,
    top,
    out_w,
    out_h,
    center_x,
    center_y,
    size_m,
    band,
    label,
    inset_rect,
    basemap,
    basemap_alpha,
    basemap_zoom,
    overlay_alpha,
    render_dpi,
    basemap_max_px,
    basemap_contrast,
    locator_anchor="right",
):
    inset_data, _, inset_bounds = load_inset_raster(
        pred_path,
        band=band,
        center_x=float(center_x),
        center_y=float(center_y),
        size_m=float(size_m),
    )
    inset_ax = fig.add_axes(inset_rect)
    inset_vals = inset_data.filled(np.nan).astype(np.float32)
    inset_vals[inset_vals <= 0] = np.nan
    xmin, ymin, xmax, ymax = inset_bounds
    inset_extent = (xmin, xmax, ymin, ymax)

    if basemap != "none":
        display_w_px = max(1, int(np.ceil(inset_rect[2] * fig.get_figwidth() * float(render_dpi))))
        display_h_px = max(1, int(np.ceil(inset_rect[3] * fig.get_figheight() * float(render_dpi))))
        max_px = max(128, int(round(float(basemap_max_px))))
        basemap_shape = (
            min(max(display_h_px, int(inset_vals.shape[0])), max_px),
            min(max(display_w_px, int(inset_vals.shape[1])), max_px),
        )
        basemap_rgba = fetch_inset_basemap(
            inset_bounds,
            src_crs=crs,
            target_shape=basemap_shape,
            provider_name=basemap,
            zoom=basemap_zoom,
        )
        if basemap_rgba is not None:
            basemap_rgba = basemap_rgba.copy()
            basemap_rgba[..., :3] = np.clip(
                (basemap_rgba[..., :3] - 0.5) * float(basemap_contrast) + 0.5,
                0.0,
                1.0,
            )
            basemap_rgba[..., 3] *= float(np.clip(basemap_alpha, 0.0, 1.0))
            inset_ax.imshow(
                basemap_rgba,
                origin="upper",
                interpolation="bilinear",
                aspect="equal",
                extent=inset_extent,
            )

    inset_cmap = plt.get_cmap("plasma").copy()
    inset_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    inset_rgba = inset_cmap(
        mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)(inset_vals)
    )
    alpha = float(
        np.clip(
            overlay_alpha if overlay_alpha is not None else (0.82 if basemap != "none" else 1.0),
            0.0,
            1.0,
        )
    )
    inset_rgba[..., 3] = np.where(np.isfinite(inset_vals), alpha, 0.0)
    inset_ax.imshow(
        inset_rgba,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        extent=inset_extent,
    )
    inset_ax.set_xlim(xmin, xmax)
    inset_ax.set_ylim(ymin, ymax)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")
    inset_ax.set_title(
        f"{label}\n{int(round(float(size_m) / 1000.0))} km probability",
        fontsize=9,
        loc="left",
        pad=6,
    )

    rect_x0, rect_y1 = bng_to_panel_xy(xmin, ymax, left, bottom, right, top, out_w, out_h)
    rect_x1, rect_y0 = bng_to_panel_xy(xmax, ymin, left, bottom, right, top, out_w, out_h)
    locator = Rectangle(
        (rect_x0, rect_y0),
        rect_x1 - rect_x0,
        rect_y1 - rect_y0,
        fill=False,
        edgecolor="black",
        linewidth=0.9,
        zorder=8,
    )
    host_ax.add_patch(locator)

    if locator_anchor == "left":
        con_top = ConnectionPatch(
            xyA=(rect_x0, rect_y1),
            coordsA=host_ax.transData,
            xyB=(1.0, 0.88),
            coordsB=inset_ax.transAxes,
            color="black",
            linewidth=0.8,
            zorder=8,
        )
        con_bottom = ConnectionPatch(
            xyA=(rect_x0, rect_y0),
            coordsA=host_ax.transData,
            xyB=(1.0, 0.12),
            coordsB=inset_ax.transAxes,
            color="black",
            linewidth=0.8,
            zorder=8,
        )
    else:
        con_top = ConnectionPatch(
            xyA=(rect_x1, rect_y1),
            coordsA=host_ax.transData,
            xyB=(0.0, 0.88),
            coordsB=inset_ax.transAxes,
            color="black",
            linewidth=0.8,
            zorder=8,
        )
        con_bottom = ConnectionPatch(
            xyA=(rect_x1, rect_y0),
            coordsA=host_ax.transData,
            xyB=(0.0, 0.12),
            coordsB=inset_ax.transAxes,
            color="black",
            linewidth=0.8,
            zorder=8,
        )
    fig.add_artist(con_top)
    fig.add_artist(con_bottom)
    return inset_ax


def _lonlat_to_xyz_tile(lon, lat, zoom):
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    n = 2 ** int(zoom)
    x = (lon + 180.0) / 360.0 * n
    lat_rad = np.deg2rad(lat)
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) * 0.5 * n
    return x, y


def _xyz_tile_bounds_mercator(x, y, zoom):
    half_world = 20037508.342789244
    world = half_world * 2.0
    tile_span = world / (2 ** int(zoom))
    xmin = -half_world + x * tile_span
    xmax = xmin + tile_span
    ymax = half_world - y * tile_span
    ymin = ymax - tile_span
    return xmin, ymin, xmax, ymax


def choose_inset_basemap_zoom(bounds_native, src_crs, target_shape):
    xmin, ymin, xmax, ymax = bounds_native
    out_h, out_w = target_shape
    width_m = max(float(xmax - xmin), 1.0)
    height_m = max(float(ymax - ymin), 1.0)
    meters_per_px = max(width_m / max(out_w, 1), height_m / max(out_h, 1))
    lon_min, lat_min, lon_max, lat_max = transform_bounds(
        src_crs, "EPSG:4326", xmin, ymin, xmax, ymax, densify_pts=21
    )
    lat_center = 0.5 * (lat_min + lat_max)
    zoom_float = np.log2(
        (156543.03392804097 * max(np.cos(np.deg2rad(lat_center)), 1e-6))
        / max(meters_per_px, 1e-6)
    )
    return int(np.clip(np.round(zoom_float), 0, 18))


def fetch_inset_basemap(bounds_native, src_crs, target_shape, provider_name="cartodb_positron", zoom=None):
    if provider_name == "none":
        return None
    provider_lookup = {
        "cartodb_positron": xyz.CartoDB.Positron,
        "cartodb_voyager": xyz.CartoDB.Voyager,
        "cartodb_dark_matter": xyz.CartoDB.DarkMatter,
        "osm_mapnik": xyz.OpenStreetMap.Mapnik,
    }
    if provider_name not in provider_lookup:
        raise ValueError(f"Unsupported inset basemap provider: {provider_name}")

    provider = provider_lookup[provider_name]
    xmin, ymin, xmax, ymax = [float(v) for v in bounds_native]
    out_h, out_w = [int(v) for v in target_shape]
    zoom = choose_inset_basemap_zoom(bounds_native, src_crs, target_shape) if zoom is None else int(zoom)

    lon_min, lat_min, lon_max, lat_max = transform_bounds(
        src_crs, "EPSG:4326", xmin, ymin, xmax, ymax, densify_pts=21
    )
    x0f, y0f = _lonlat_to_xyz_tile(lon_min, lat_max, zoom)
    x1f, y1f = _lonlat_to_xyz_tile(lon_max, lat_min, zoom)
    x_min = int(np.floor(min(x0f, x1f)))
    x_max = int(np.floor(max(x0f, x1f)))
    y_min = int(np.floor(min(y0f, y1f)))
    y_max = int(np.floor(max(y0f, y1f)))

    tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
    if tile_count <= 0 or tile_count > 64:
        print(f"Warning: inset basemap needs {tile_count} tiles at zoom {zoom}; skipping basemap.")
        return None

    session = requests.Session()
    session.headers.update({"User-Agent": "wet-woodland-research/visualise"})
    tile_px = 256
    mosaic = np.zeros(
        ((y_max - y_min + 1) * tile_px, (x_max - x_min + 1) * tile_px, 4),
        dtype=np.uint8,
    )

    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            url = provider.build_url(x=tx, y=ty, z=zoom)
            try:
                resp = session.get(url, timeout=20)
                resp.raise_for_status()
                tile = np.asarray(Image.open(BytesIO(resp.content)).convert("RGBA"))
            except Exception as exc:
                print(f"Warning: failed to fetch inset basemap tile {zoom}/{tx}/{ty}: {exc}")
                return None
            y_off = (ty - y_min) * tile_px
            x_off = (tx - x_min) * tile_px
            mosaic[y_off:y_off + tile_px, x_off:x_off + tile_px, :] = tile

    src_bounds = _xyz_tile_bounds_mercator(x_min, y_max, zoom)
    dst_bounds = (xmin, ymin, xmax, ymax)
    src_xmin = src_bounds[0]
    src_ymin = _xyz_tile_bounds_mercator(x_min, y_max, zoom)[1]
    src_xmax = _xyz_tile_bounds_mercator(x_max, y_min, zoom)[2]
    src_ymax = _xyz_tile_bounds_mercator(x_min, y_min, zoom)[3]
    src_transform = from_bounds(
        src_xmin, src_ymin, src_xmax, src_ymax, mosaic.shape[1], mosaic.shape[0]
    )
    dst_transform = from_bounds(
        dst_bounds[0], dst_bounds[1], dst_bounds[2], dst_bounds[3], out_w, out_h
    )

    dest = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    for band_idx in range(4):
        reproject(
            source=mosaic[:, :, band_idx],
            destination=dest[:, :, band_idx],
            src_transform=src_transform,
            src_crs="EPSG:3857",
            dst_transform=dst_transform,
            dst_crs=src_crs,
            dst_nodata=0,
            resampling=Resampling.bilinear,
        )
    return dest.astype(np.float32) / 255.0


def bng_to_panel_xy(x, y, left, bottom, right, top, out_w, out_h):
    px = (x - left) / (right - left) * out_w
    py = (top - y) / (top - bottom) * out_h
    return px, py


def draw_outline(ax, geom, left, bottom, right, top, out_w, out_h, linewidth=0.7):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    verts, codes = [], []
    for poly in polys:
        for ring in [poly.exterior] + list(poly.interiors):
            xs, ys = ring.xy
            pxs = [(x - left) / (right - left) * out_w for x in xs]
            pys = [(top - y) / (top - bottom) * out_h for y in ys]
            verts += list(zip(pxs, pys))
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(pxs) - 2) + [MPath.CLOSEPOLY]
    path = MPath(verts, codes)
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor="black",
            linewidth=linewidth,
            zorder=7,
        )
    )


def add_city_labels(ax, crs, left, bottom, right, top, out_w, out_h):
    wgs_to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cities = [
        ("London", 51.505, -0.090, (58, 0)),
        ("Birmingham", 52.480, -1.900, (-70, 0)),
        ("Manchester", 53.480, -2.240, (-54, 0)),
        ("Leeds", 53.800, -1.550, (50, 0)),
        ("Newcastle", 54.970, -1.610, (38, 0)),
        ("RAF Fylingdales", 54.366, -0.674, (44, 0)),
        ("Plymouth", 50.375, -4.140, (38, 0)),
    ]
    for name, lat, lon, (dx, dy) in cities:
        dx *= CITY_LABEL_OFFSET_SCALE
        dy *= CITY_LABEL_OFFSET_SCALE
        mx, my = wgs_to_crs.transform(lon, lat)
        px = (mx - left) / (right - left) * out_w
        py = (top - my) / (top - bottom) * out_h
        ax.plot(px, py, "o", markersize=3.5, color="black", markeredgewidth=0, zorder=5)
        ax.annotate(
            name,
            xy=(px, py),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10.0,
            color="black",
            va="bottom" if dy < 0 else ("top" if dy > 0 else "center"),
            ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.35),
            zorder=6,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Render a single national wet woodland predictions panel."
    )
    parser.add_argument(
        "--predictions",
        default="data/output/rasters/wet_woodland_extent.tif",
        help="Hysteresis GeoTIFF. Default: data/output/rasters/wet_woodland_extent.tif",
    )
    parser.add_argument(
        "--predictions-band",
        type=int,
        default=1,
        help="Band index for predictions input. Default: 1 (binary hysteresis extent).",
    )
    parser.add_argument(
        "--dem",
        default="data/output/potential/potential_predictors_100m.tif",
        help="DEM / abiotic stack used for optional terrain underlay.",
    )
    parser.add_argument(
        "--boundary",
        default="data/input/boundaries/england.shp",
        help="Boundary shapefile used to clip the rendered figure.",
    )
    parser.add_argument(
        "--clip-buffer-m",
        type=float,
        default=-100.0,
        help="Buffer for clip boundary in metres. Default: -100.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2000,
        help="Longest output side in pixels. Default: 2000.",
    )
    parser.add_argument(
        "--aggregate-res-m",
        type=float,
        default=500.0,
        help="Map-space aggregation resolution in metres. Default: 500.",
    )
    parser.add_argument(
        "--surface-mode",
        choices=["block", "smooth_density"],
        default="block",
        help="block = crisp aggregated cells; smooth_density = Gaussian-smoothed cover surface. Default: block.",
    )
    parser.add_argument(
        "--smooth-sigma-m",
        type=float,
        default=0.0,
        help="Gaussian sigma in metres for smooth_density mode. Default: 0.",
    )
    parser.add_argument(
        "--display-smooth-sigma-px",
        type=float,
        default=0.0,
        help="Very light Gaussian smoothing in output pixels for display only. Default: 0.",
    )
    parser.add_argument(
        "--terrain-strength",
        type=float,
        default=0.55,
        help="Strength of terrain shadow overlay. Default: 0.55.",
    )
    parser.add_argument(
        "--include-terrain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include multidirectional terrain shadow beneath the predictions layer. Default: off.",
    )
    parser.add_argument(
        "--palette",
        choices=["editorial", "viridis", "viridis_r", "cividis", "cividis_r", "turbo", "forest_teal"],
        default="viridis_r",
        help="Predictions palette. Default: viridis_r.",
    )
    parser.add_argument(
        "--mask-below",
        type=float,
        default=0.0,
        help="Hide aggregated cells below this cover fraction. Default: 0.0",
    )
    parser.add_argument(
        "--min-wet-pixels",
        type=float,
        default=1.0,
        help=(
            "Require at least this many wet source pixels in each aggregated cell; "
            "cells below the threshold are set to nodata. Default: 1."
        ),
    )
    parser.add_argument(
        "--color-scale",
        choices=["linear", "quantile"],
        default="quantile",
        help="How values are mapped into the palette. Default: quantile.",
    )
    parser.add_argument(
        "--color-max",
        type=float,
        default=1.0,
        help="Cap the color scale at this cover fraction; higher values use the top color. Default: 1.0",
    )
    parser.add_argument(
        "--color-max-percentile",
        type=float,
        default=None,
        help="Optional percentile cap for the color scale, computed from displayed values (e.g. 95).",
    )
    parser.add_argument(
        "--value-gamma",
        type=float,
        default=1.0,
        help="Power-law stretch for the color scale. <1 emphasises low-mid values. Default: 1.0",
    )
    parser.add_argument(
        "--include-north-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a left-side north-south marginal profile. Default: on.",
    )
    parser.add_argument(
        "--north-profile-smooth-px",
        type=int,
        default=31,
        help="Moving-average window for the north profile, in rendered pixels. Default: 31.",
    )
    parser.add_argument(
        "--include-site-inset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a right-side site inset with connector lines. Default: off.",
    )
    parser.add_argument(
        "--inset-center-x",
        type=float,
        default=632600.0,
        help="Inset centre easting in EPSG:27700. Default: Wheatfen NNR approx 632600.",
    )
    parser.add_argument(
        "--inset-center-y",
        type=float,
        default=306100.0,
        help="Inset centre northing in EPSG:27700. Default: Wheatfen NNR approx 306100.",
    )
    parser.add_argument(
        "--inset-size-m",
        type=float,
        default=5000.0,
        help="Inset square size in metres. Default: 5000.",
    )
    parser.add_argument(
        "--inset-band",
        type=int,
        default=2,
        help="Raster band to use for the site inset. Default: 2 (hysteresis-masked probability).",
    )
    parser.add_argument(
        "--inset-label",
        default="Wheatfen NNR",
        help="Label shown above the site inset.",
    )
    parser.add_argument(
        "--include-left-inset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a second inset on the left side. Default: off.",
    )
    # Previous left inset target retained for reference:
    # Arlington Court, North Devon: x=260503.0, y=140419.0, size=1000.0
    parser.add_argument(
        "--left-inset-center-x",
        type=float,
        default=310283.0,
        help="Left inset centre easting in EPSG:27700. Default: Dobbs Moss Peat Nature Reserve.",
    )
    parser.add_argument(
        "--left-inset-center-y",
        type=float,
        default=529052.0,
        help="Left inset centre northing in EPSG:27700. Default: Dobbs Moss Peat Nature Reserve.",
    )
    parser.add_argument(
        "--left-inset-size-m",
        type=float,
        default=1000.0,
        help="Left inset square size in metres. Default: 1000.",
    )
    parser.add_argument(
        "--left-inset-band",
        type=int,
        default=2,
        help="Raster band to use for the left inset. Default: 2 (hysteresis-masked probability).",
    )
    parser.add_argument(
        "--left-inset-label",
        default="Dobbs Moss Peat NR",
        help="Label shown above the left inset.",
    )
    parser.add_argument(
        "--include-bottom-right-inset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a third inset at the bottom right. Default: off.",
    )
    parser.add_argument(
        "--bottom-right-inset-center-x",
        type=float,
        default=620822.0,
        help="Bottom-right inset centre easting in EPSG:27700. Default: Stodmarsh NNR.",
    )
    parser.add_argument(
        "--bottom-right-inset-center-y",
        type=float,
        default=161068.0,
        help="Bottom-right inset centre northing in EPSG:27700. Default: Stodmarsh NNR.",
    )
    parser.add_argument(
        "--bottom-right-inset-size-m",
        type=float,
        default=5000.0,
        help="Bottom-right inset square size in metres. Default: 5000.",
    )
    parser.add_argument(
        "--bottom-right-inset-band",
        type=int,
        default=2,
        help="Raster band to use for the bottom-right inset. Default: 2 (hysteresis-masked probability).",
    )
    parser.add_argument(
        "--bottom-right-inset-label",
        default="Stodmarsh NNR",
        help="Label shown above the bottom-right inset.",
    )
    parser.add_argument(
        "--inset-basemap",
        choices=["none", "cartodb_positron", "cartodb_voyager", "cartodb_dark_matter", "osm_mapnik"],
        default="none",
        help="Optional basemap underlay for the site inset. Default: none.",
    )
    parser.add_argument(
        "--inset-basemap-alpha",
        type=float,
        default=0.9,
        help="Opacity for the inset basemap underlay. Default: 0.9.",
    )
    parser.add_argument(
        "--inset-basemap-zoom",
        type=int,
        default=None,
        help="Optional explicit XYZ zoom for the inset basemap. Default: auto.",
    )
    parser.add_argument(
        "--inset-overlay-alpha",
        type=float,
        default=None,
        help="Opacity for the inset probability overlay. Default: 0.82 with a basemap, otherwise 1.0.",
    )
    parser.add_argument(
        "--inset-basemap-max-px",
        type=int,
        default=1200,
        help="Maximum fetched pixel size per inset basemap side. Default: 1200.",
    )
    parser.add_argument(
        "--inset-basemap-contrast",
        type=float,
        default=1.0,
        help="Contrast multiplier for inset basemap RGB values. Default: 1.0.",
    )
    parser.add_argument(
        "--output",
        default="visualise/output/predictions_panel.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output DPI. Default: 600.",
    )
    args = parser.parse_args()

    pred_path = _resolve_path(args.predictions)
    dem_path = _resolve_path(args.dem)
    boundary = _resolve_path(args.boundary)
    out_path = _resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    terrain_strength = args.terrain_strength
    size = args.size

    with rasterio.open(pred_path) as src:
        crs = src.crs
        bounds = src.bounds

    clip_geom = load_clip_geometry(boundary, crs, buffer_m=args.clip_buffer_m)

    if clip_geom is not None:
        cb = shape(clip_geom).bounds
        left = max(bounds.left, cb[0])
        bottom = max(bounds.bottom, cb[1])
        right = min(bounds.right, cb[2])
        top = min(bounds.top, cb[3])
    else:
        left, bottom, right, top = bounds

    w_u, h_u = right - left, top - bottom
    if w_u >= h_u:
        out_w, out_h = size, max(1, int(round(size * h_u / w_u)))
    else:
        out_h, out_w = size, max(1, int(round(size * w_u / h_u)))

    dst_transform = from_bounds(left, bottom, right, top, out_w, out_h)

    # Aggregate the binary predictions to a coarser grid. In block mode this is
    # later re-expanded with nearest-neighbour for a crisp national map. In
    # smooth_density mode the coarse grid is Gaussian-smoothed directly.
    coarse_res = float(max(args.aggregate_res_m, 10.0))
    coarse_w = max(1, int(round((right - left) / coarse_res)))
    coarse_h = max(1, int(round((top - bottom) / coarse_res)))
    dst_transform_c = from_bounds(left, bottom, right, top, coarse_w, coarse_h)
    with rasterio.open(pred_path) as src:
        pred_coarse = np.full((coarse_h, coarse_w), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, args.predictions_band),
            destination=pred_coarse,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform_c,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        wet_count_coarse = np.zeros((coarse_h, coarse_w), dtype=np.float32)
        reproject(
            source=rasterio.band(src, args.predictions_band),
            destination=wet_count_coarse,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform_c,
            dst_crs=crs,
            dst_nodata=0.0,
            resampling=Resampling.sum,
        )

    pred_coarse = np.clip(pred_coarse, 0.0, 1.0)
    if float(args.min_wet_pixels) > 0:
        pred_coarse = np.where(wet_count_coarse >= float(args.min_wet_pixels), pred_coarse, np.nan)

    if clip_geom is not None:
        coarse_clip_mask = geometry_mask(
            [clip_geom], out_shape=(coarse_h, coarse_w), transform=dst_transform_c, invert=True
        )
    else:
        coarse_clip_mask = np.ones((coarse_h, coarse_w), dtype=bool)

    if args.surface_mode == "smooth_density":
        support_coarse = np.where(coarse_clip_mask & np.isfinite(pred_coarse), 1.0, 0.0).astype(np.float32)
        pred_coarse = gaussian_smooth_density(
            pred_coarse,
            support=support_coarse,
            sigma_cells=float(args.smooth_sigma_m) / coarse_res,
        )
        pred_coarse = np.where(coarse_clip_mask, pred_coarse, np.nan)

    ri = np.clip(np.floor(np.arange(out_h) * coarse_h / out_h).astype(int), 0, coarse_h - 1)
    ci = np.clip(np.floor(np.arange(out_w) * coarse_w / out_w).astype(int), 0, coarse_w - 1)
    pred = pred_coarse[ri[:, None], ci[None, :]]

    terrain = None
    if args.include_terrain and dem_path.exists():
        dem = np.full((out_h, out_w), np.nan, dtype=np.float32)
        with rasterio.open(dem_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dem,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        x_res = abs(dst_transform.a)
        y_res = abs(dst_transform.e)
        t = build_multidirectional_terrain(
            dem, x_res=x_res, y_res=y_res, gamma=0.55, z_factor=2.5
        )
        terrain = np.where(np.isfinite(t), t, 1.0)
    elif args.include_terrain:
        print(f"Warning: DEM not found at {dem_path}; rendering without terrain.")

    if clip_geom is not None:
        clip_mask = geometry_mask(
            [clip_geom], out_shape=(out_h, out_w), transform=dst_transform, invert=True
        )
    else:
        clip_mask = np.ones((out_h, out_w), dtype=bool)

    cmap = build_predictions_colormap(args.palette)
    color_max = float(max(args.color_max, 1e-6))

    shadow_rgba = None
    if terrain is not None:
        all_px = clip_mask & np.isfinite(terrain)
        t_lo = np.nanpercentile(terrain[all_px], 2)
        t_hi = np.nanpercentile(terrain[all_px], 98)
        terrain_norm = np.clip((terrain - t_lo) / max(t_hi - t_lo, 1e-6), 0, 1)
        shadow_alpha = np.where(
            clip_mask, (1.0 - terrain_norm) * terrain_strength, 0.0
        ).astype(np.float32)
        shadow_rgba = np.zeros((out_h, out_w, 4), dtype=np.float32)
        shadow_rgba[:, :, 3] = shadow_alpha

    include_right_inset = bool(args.include_site_inset)
    include_left_inset = bool(args.include_left_inset)
    include_bottom_right_inset = bool(args.include_bottom_right_inset)
    include_any_right_inset = include_right_inset or include_bottom_right_inset
    include_any_side_inset = include_any_right_inset or include_left_inset

    show_north_profile = bool(args.include_north_profile and not include_any_side_inset)

    if show_north_profile:
        fig, (profile_ax, ax) = plt.subplots(
            1,
            2,
            figsize=((11.2 if args.include_site_inset else 9.5), 9.5),
            sharey=True,
            gridspec_kw={
                "width_ratios": [1.55, (6.6 if args.include_site_inset else 8.0)],
                "wspace": 0.035,
            },
        )
    else:
        if include_left_inset and include_any_right_inset:
            fig_w = 13.2
        elif include_any_side_inset:
            fig_w = 10.6
        else:
            fig_w = 8.5
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, 9.5))
        profile_ax = None
    fig.patch.set_facecolor("white")
    ax.patch.set_visible(False)

    if float(args.display_smooth_sigma_px) > 0:
        pred = gaussian_smooth_density(
            pred,
            support=np.where(clip_mask & np.isfinite(pred), 1.0, 0.0).astype(np.float32),
            sigma_cells=float(args.display_smooth_sigma_px),
        )
        pred = np.where(clip_mask, pred, np.nan)

    valid_px = clip_mask & np.isfinite(pred) & (pred >= float(args.mask_below))
    if args.color_max_percentile is not None and np.any(valid_px):
        pct = float(args.color_max_percentile)
        pct = min(max(pct, 0.0), 100.0)
        color_max = float(np.nanpercentile(pred[valid_px], pct))
        color_max = max(color_max, 1e-6)
    colour_img = np.zeros((out_h, out_w, 4), dtype=np.float32)
    if args.color_scale == "quantile":
        pred_scaled, colorbar_ticks, colorbar_tick_values = build_quantile_scaled_values(
            pred, valid_px
        )
        pred_rgba = cmap(pred_scaled)
        color_norm = mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)
        if args.surface_mode == "smooth_density":
            colorbar_label = (
                f"Smoothed wet woodland cover ({int(round(coarse_res))} m grid, "
                f"{int(round(args.smooth_sigma_m / 1000.0))} km Gaussian; quantile-ranked colours)"
            )
        else:
            colorbar_label = (
                f"Wet woodland cover within {int(round(coarse_res))} m grid cell "
                "(quantile-ranked colours)"
            )
    else:
        color_norm = (
            mcolors.Normalize(vmin=0.0, vmax=color_max, clip=True)
            if abs(float(args.value_gamma) - 1.0) < 1e-6 else
            mcolors.PowerNorm(
                gamma=float(args.value_gamma),
                vmin=0.0,
                vmax=color_max,
                clip=True,
            )
        )
        pred_rgba = cmap(color_norm(pred))
        colorbar_ticks = np.array([0.0, color_max / 2.0, color_max], dtype=np.float64)
        colorbar_tick_values = colorbar_ticks.astype(np.float32)
        if args.surface_mode == "smooth_density":
            colorbar_label = (
                f"Smoothed wet woodland cover ({int(round(coarse_res))} m grid, "
                f"{args.smooth_sigma_m / 1000.0:.1f} km Gaussian)"
            )
        else:
            colorbar_label = f"Wet woodland cover within {int(round(coarse_res))} m grid cell"

    for c in range(3):
        colour_img[:, :, c] = np.where(valid_px, pred_rgba[:, :, c], 1.0)
    colour_img[:, :, 3] = np.where(valid_px, 1.0, 0.0)

    ax.imshow(
        colour_img,
        origin="upper",
        interpolation=("bilinear" if args.surface_mode == "smooth_density" else "nearest"),
        aspect="equal",
    )
    if shadow_rgba is not None:
        ax.imshow(shadow_rgba, origin="upper", interpolation="nearest", aspect="equal")

    if clip_geom is not None:
        draw_outline(ax, shape(clip_geom), left, bottom, right, top, out_w, out_h)

    add_city_labels(ax, crs, left, bottom, right, top, out_w, out_h)
    ax.axis("off")

    if profile_ax is not None:
        profile = compute_north_profile(
            pred, valid_px, smooth_px=args.north_profile_smooth_px
        )
        y = np.arange(out_h, dtype=np.float32)
        valid_profile = np.isfinite(profile)
        x_max = float(np.nanmax(profile[valid_profile])) if np.any(valid_profile) else 0.0
        x_max = max(x_max, 1e-3)

        plot_x = np.where(valid_profile, profile, np.nan).astype(np.float32)
        valid_idx = np.flatnonzero(np.isfinite(plot_x))
        if valid_idx.size >= 2:
            points = np.column_stack([plot_x[valid_idx], y[valid_idx]]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            seg_values = ((plot_x[valid_idx[:-1]] + plot_x[valid_idx[1:]]) * 0.5).astype(
                np.float32
            )
            line = LineCollection(
                segments,
                cmap=cmap,
                norm=mcolors.Normalize(vmin=0.0, vmax=x_max, clip=True),
                linewidth=1.05,
                capstyle="round",
                joinstyle="round",
                zorder=3,
            )
            line.set_array(seg_values)
            profile_ax.add_collection(line)
        elif valid_idx.size == 1:
            profile_ax.plot(plot_x[valid_idx], y[valid_idx], color="black", linewidth=1.05, zorder=3)

        profile_ax.set_ylim(out_h - 0.5, -0.5)
        profile_ax.set_xlim(x_max * 1.08, 0.0)
        profile_ax.set_xticks([0.0, x_max])
        profile_ax.set_xticklabels(["0%", f"{int(round(x_max * 100))}%"], fontsize=9)
        profile_ax.tick_params(axis="x", length=0, pad=4)
        profile_ax.tick_params(axis="y", left=False, labelleft=False)
        for side in ("top", "right", "left"):
            profile_ax.spines[side].set_visible(False)
        profile_ax.spines["bottom"].set_color("#666666")
        profile_ax.spines["bottom"].set_linewidth(0.6)
        profile_ax.set_xlabel("Mean by northing", fontsize=10, labelpad=6)
        profile_ax.xaxis.set_label_coords(0.58, -0.07)
        profile_ax.patch.set_visible(False)

    left_margin = 0.045
    right_margin = 0.97
    if include_left_inset and include_any_right_inset:
        left_margin = 0.22
        right_margin = 0.78
    elif include_left_inset:
        left_margin = 0.22
        right_margin = 0.97
    elif include_any_right_inset:
        left_margin = 0.045
        right_margin = 0.75
    fig.subplots_adjust(bottom=0.125, top=0.98, left=left_margin, right=right_margin)

    inset_ax = None
    left_inset_ax = None
    bottom_right_inset_ax = None
    map_pos = ax.get_position()
    inset_size = min(map_pos.width * 0.38, 0.19)
    if include_right_inset:
        inset_left = min(0.985 - inset_size, map_pos.x1 + 0.045)
        inset_bottom = map_pos.y0 + map_pos.height * 0.50
        inset_ax = draw_probability_inset(
            fig=fig,
            host_ax=ax,
            pred_path=pred_path,
            crs=crs,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            out_w=out_w,
            out_h=out_h,
            center_x=float(args.inset_center_x),
            center_y=float(args.inset_center_y),
            size_m=float(args.inset_size_m),
            band=args.inset_band,
            label=args.inset_label,
            inset_rect=[inset_left, inset_bottom, inset_size, inset_size],
            basemap=args.inset_basemap,
            basemap_alpha=args.inset_basemap_alpha,
            basemap_zoom=args.inset_basemap_zoom,
            overlay_alpha=args.inset_overlay_alpha,
            render_dpi=args.dpi,
            basemap_max_px=args.inset_basemap_max_px,
            basemap_contrast=args.inset_basemap_contrast,
            locator_anchor="right",
        )
    if include_left_inset:
        inset_left = max(0.015, map_pos.x0 - inset_size + 0.08)
        inset_bottom = map_pos.y0 + map_pos.height * 0.50
        left_inset_ax = draw_probability_inset(
            fig=fig,
            host_ax=ax,
            pred_path=pred_path,
            crs=crs,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            out_w=out_w,
            out_h=out_h,
            center_x=float(args.left_inset_center_x),
            center_y=float(args.left_inset_center_y),
            size_m=float(args.left_inset_size_m),
            band=args.left_inset_band,
            label=args.left_inset_label,
            inset_rect=[inset_left, inset_bottom, inset_size, inset_size],
            basemap=args.inset_basemap,
            basemap_alpha=args.inset_basemap_alpha,
            basemap_zoom=args.inset_basemap_zoom,
            overlay_alpha=args.inset_overlay_alpha,
            render_dpi=args.dpi,
            basemap_max_px=args.inset_basemap_max_px,
            basemap_contrast=args.inset_basemap_contrast,
            locator_anchor="left",
        )
    if include_bottom_right_inset:
        inset_left = min(0.985 - inset_size, map_pos.x1 + 0.045)
        inset_bottom = max(0.14, map_pos.y0 + 0.02)
        bottom_right_inset_ax = draw_probability_inset(
            fig=fig,
            host_ax=ax,
            pred_path=pred_path,
            crs=crs,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            out_w=out_w,
            out_h=out_h,
            center_x=float(args.bottom_right_inset_center_x),
            center_y=float(args.bottom_right_inset_center_y),
            size_m=float(args.bottom_right_inset_size_m),
            band=args.bottom_right_inset_band,
            label=args.bottom_right_inset_label,
            inset_rect=[inset_left, inset_bottom, inset_size, inset_size],
            basemap=args.inset_basemap,
            basemap_alpha=args.inset_basemap_alpha,
            basemap_zoom=args.inset_basemap_zoom,
            overlay_alpha=args.inset_overlay_alpha,
            render_dpi=args.dpi,
            basemap_max_px=args.inset_basemap_max_px,
            basemap_contrast=args.inset_basemap_contrast,
            locator_anchor="right",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
    sm.set_array([])
    map_pos = ax.get_position()
    cbar_width = map_pos.width * 0.56
    cbar_left = map_pos.x0 + map_pos.width * 0.26
    cbar_bottom = 0.058
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, 0.022])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks(colorbar_ticks.tolist())
    if args.color_scale == "quantile":
        cb.set_ticklabels([f"{int(round(v * 100))}%" for v in colorbar_tick_values])
    else:
        upper_label = (
            f"{int(round(color_max * 100))}%"
            if color_max >= 0.999 else
            f">={int(round(color_max * 100))}%"
        )
        cb.set_ticklabels(
            [
                "0%",
                f"{int(round((color_max / 2.0) * 100))}%",
                upper_label,
            ]
        )
    cb.ax.tick_params(labelsize=12)
    cb.set_label(colorbar_label, fontsize=13, labelpad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
