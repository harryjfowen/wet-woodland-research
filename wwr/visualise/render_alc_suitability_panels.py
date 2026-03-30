#!/usr/bin/env python3
"""
Render a three-panel figure showing predicted wet woodland suitability
masked by Agricultural Land Classification grade.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

try:
    import fiona
    from pyproj import Transformer
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform, unary_union
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

REPO = Path(__file__).resolve().parent.parent


def build_multidirectional_terrain(
    dem, x_res, y_res,
    azimuths=None, altitude=45.0, gamma=0.7, z_factor=1.5,
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


def main():
    suit_path  = REPO / "data/wet_woodland_potential.tif"
    land_path  = REPO / "data/landvalue_classes.tif"
    dem_path   = REPO / "data/potential_predictors_100m.tif"
    boundary   = REPO / "data/england.shp"
    out_path   = REPO / "static/output/alc_suitability_panels.png"
    terrain_strength = 0.55

    SIZE = 2000

    with rasterio.open(suit_path) as src:
        crs    = src.crs
        bounds = src.bounds

    clip_geom = load_clip_geometry(boundary, crs, buffer_m=-100.0)

    if clip_geom is not None:
        cb = shape(clip_geom).bounds
        left   = max(bounds.left,   cb[0])
        bottom = max(bounds.bottom, cb[1])
        right  = min(bounds.right,  cb[2])
        top    = min(bounds.top,    cb[3])
    else:
        left, bottom, right, top = bounds

    w_u, h_u = right - left, top - bottom
    if w_u >= h_u:
        out_w, out_h = SIZE, max(1, int(round(SIZE * h_u / w_u)))
    else:
        out_h, out_w = SIZE, max(1, int(round(SIZE * w_u / h_u)))

    dst_transform = from_bounds(left, bottom, right, top, out_w, out_h)

    # Load suitability at ~250 m then upsample with nearest-neighbour for bolder pixels.
    suit_native_m = 100.0
    suit_target_m = 250.0
    scale = suit_native_m / suit_target_m
    suit_w_c = max(1, int(round(out_w * scale)))
    suit_h_c = max(1, int(round(out_h * scale)))
    dst_transform_c = from_bounds(left, bottom, right, top, suit_w_c, suit_h_c)
    suit_coarse = np.full((suit_h_c, suit_w_c), np.nan, dtype=np.float32)
    with rasterio.open(suit_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=suit_coarse,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=dst_transform_c, dst_crs=crs, dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    suit_coarse = np.clip(suit_coarse, 0.0, 1.0)
    # Upsample to full grid with nearest-neighbour (preserves blocky 250 m pixel edges)
    ri = np.clip(np.floor(np.arange(out_h) * suit_h_c / out_h).astype(int), 0, suit_h_c - 1)
    ci = np.clip(np.floor(np.arange(out_w) * suit_w_c / out_w).astype(int), 0, suit_w_c - 1)
    suit = suit_coarse[ri[:, None], ci[None, :]]

    # Load land value classes (0=Grade 1-2, 1=Grade 3, 2=Grade 4-5)
    land = np.full((out_h, out_w), 255, dtype=np.float32)
    with rasterio.open(land_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=land,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=dst_transform, dst_crs=crs, dst_nodata=255,
            resampling=Resampling.nearest,
        )
    land = np.round(land).astype(np.int16)

    # Load DEM and build terrain
    terrain = np.ones((out_h, out_w), dtype=np.float64)
    if dem_path.exists():
        dem = np.full((out_h, out_w), np.nan, dtype=np.float32)
        with rasterio.open(dem_path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=dem,
                src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                dst_transform=dst_transform, dst_crs=crs, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        x_res = abs(dst_transform.a)
        y_res = abs(dst_transform.e)
        t = build_multidirectional_terrain(dem, x_res=x_res, y_res=y_res, gamma=0.55, z_factor=2.5)
        terrain = np.where(np.isfinite(t), t, 1.0)

    # Clip mask
    if clip_geom is not None:
        clip_mask = geometry_mask(
            [clip_geom], out_shape=(out_h, out_w),
            transform=dst_transform, invert=True,
        )
    else:
        clip_mask = np.ones((out_h, out_w), dtype=bool)

    # Cubehelix ramp: light grey → muted green → blue (matches bivariate suitability ramp)
    try:
        import seaborn as sns
        colors = sns.cubehelix_palette(
            256, start=2.0, rot=-1.0, light=0.96, dark=0.05, hue=1.8,
        )
        # Blend the low end toward a faint sandy grey (#D4CBBF)
        sandy = np.array([0.76, 0.72, 0.65])
        n_blend = 60
        for i in range(n_blend):
            t = i / n_blend  # 0 = pure sandy, 1 = original cubehelix
            colors[i] = (1 - t) * sandy + t * np.array(colors[i])
        cmap = mcolors.LinearSegmentedColormap.from_list("suit_ramp", colors, N=256)
    except ImportError:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "fallback", ["#FFFFFF", "#D7EDDB", "#7EC8C9", "#2E88BD", "#081D58"], N=256
        )
    cmap.set_bad(color="white")

    panels = [
        (0, "a)  Grades 1–2\nexcellent / very good"),
        (1, "b)  Grade 3\ngood / moderate"),
        (2, "c)  Grades 4–5\npoor / very poor"),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 9),
        gridspec_kw={"wspace": -0.22},
    )
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.patch.set_visible(False)

    # Compute terrain shadow once — same for all panels.
    all_px = clip_mask & np.isfinite(terrain)
    t_lo = np.nanpercentile(terrain[all_px], 2)
    t_hi = np.nanpercentile(terrain[all_px], 98)
    terrain_norm = np.clip((terrain - t_lo) / max(t_hi - t_lo, 1e-6), 0, 1)
    # Shadow layer: steep slopes get dark alpha, flat areas transparent.
    # Applied on top of everything identically in every panel.
    shadow_alpha = np.where(clip_mask, (1.0 - terrain_norm) * terrain_strength, 0.0).astype(np.float32)
    shadow_rgba = np.zeros((out_h, out_w, 4), dtype=np.float32)
    shadow_rgba[:, :, 3] = shadow_alpha  # black shadow, variable transparency

    im = None
    for ax, (grade_code, title) in zip(axes, panels):
        valid_px = (land == grade_code) & clip_mask & np.isfinite(suit)

        # Layer 1: RGBA — England pixels opaque, outside transparent.
        colour_img = np.zeros((out_h, out_w, 4), dtype=np.float32)
        suit_rgba = cmap(suit)
        for c in range(3):
            colour_img[:, :, c] = np.where(valid_px, suit_rgba[:, :, c], 1.0)  # white for non-grade
        colour_img[:, :, 3] = np.where(clip_mask, 1.0, 0.0)  # transparent outside England

        ax.imshow(colour_img, origin="upper", interpolation="nearest", aspect="equal")

        # Layer 2: identical terrain shadow overlay on top — completely independent of colours.
        ax.imshow(shadow_rgba, origin="upper", interpolation="nearest", aspect="equal")
        # England outline
        if clip_geom is not None:
            from shapely.geometry import shape as sg
            from matplotlib.patches import PathPatch
            from matplotlib.path import Path as MPath
            import shapely

            def _geom_to_patch(geom, lw=0.6):
                polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
                verts, codes = [], []
                for poly in polys:
                    for ring in [poly.exterior] + list(poly.interiors):
                        xs, ys = ring.xy
                        pxs = [(x - left) / (right - left) * out_w for x in xs]
                        pys = [(top  - y) / (top - bottom) * out_h for y in ys]
                        verts += list(zip(pxs, pys))
                        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(pxs) - 2) + [MPath.CLOSEPOLY]
                path = MPath(verts, codes)
                return PathPatch(path, facecolor="none", edgecolor="black", linewidth=lw, zorder=7)
            eng = sg(clip_geom)
            ax.add_patch(_geom_to_patch(eng))

        ax.set_title(title, fontsize=14, fontweight="bold", pad=8, loc="left")
        ax.axis("off")

        # City labels on middle panel only
        if grade_code == 1:
            from pyproj import Transformer
            wgs_to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            # (name, lat, lon, text_offset_pixels)
            cities = [
                ("London",     51.505, -0.090, ( 270,  0)),
                ("Birmingham", 52.480, -1.900, (-324,  0)),
                ("Manchester", 53.480, -2.240, (-270,  0)),
                ("Leeds",      53.800, -1.550, ( 324,  0)),
                ("Newcastle",  54.970, -1.610, ( 270,  0)),
                ("Plymouth",   50.375, -4.140, ( 270,  0)),
            ]
            for name, lat, lon, (dx, dy) in cities:
                mx, my = wgs_to_crs.transform(lon, lat)
                px = (mx - left) / (right - left) * out_w
                py = (top  - my) / (top - bottom) * out_h
                # dot
                ax.plot(px, py, "o", markersize=3.5, color="black",
                        markeredgewidth=0, zorder=5)
                # leader line + label
                ax.annotate(
                    name,
                    xy=(px, py),
                    xytext=(px + dx, py + dy),
                    fontsize=10.0,
                    color="black",
                    va="bottom" if dy < 0 else ("top" if dy > 0 else "center"),
                    ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.35),
                    zorder=6,
                )

    # Shared colorbar via explicit ScalarMappable (avoids broken-imshow hack)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.subplots_adjust(bottom=0.18, top=0.93, left=0.02, right=0.98)
    cbar_ax = fig.add_axes([0.25, 0.10, 0.50, 0.025])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["0  (low)", "0.5", "1  (high)"])
    cb.ax.tick_params(labelsize=13)
    cb.set_label("Suitability", fontsize=14, labelpad=6)


    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")

    # PIL saturation + contrast boost to make mid-range colours pop
    from PIL import Image, ImageEnhance
    img = Image.open(out_path).convert("RGB")
    img = ImageEnhance.Color(img).enhance(1.4)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img.save(out_path, dpi=(600, 600))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
