#!/usr/bin/env python3
"""
Hysteresis thresholding for wet woodland probability rasters.

Algorithm (raster-native, no vectorization):
1) low_mask  = prob >= low_threshold
2) high_mask = prob >= high_threshold
3) Label connected components in low_mask
4) Keep only components that contain at least one high_mask pixel
5) Optional: remove kept components smaller than min_size pixels

This removes isolated low-confidence speckle while preserving coherent patches
anchored by high-confidence cores.

Default output is 2-band:
- Band 1: cleaned binary labels (0/1, nodata)
- Band 2: probability values masked by the same hysteresis keep-mask
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Compression
from rasterio.merge import merge as rio_merge
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    tow_root = Path(__file__).resolve().parents[2]
    default_input = tow_root / "data" / "output" / "predictions" / "tiles"
    default_output = tow_root / "data" / "output" / "postprocess" / "wet_woodland_mosaic_hysteresis.tif"
    p = argparse.ArgumentParser(
        description="Apply two-threshold hysteresis to a probability GeoTIFF"
    )
    p.add_argument(
        "--input",
        default=str(default_input),
        help=(
            "Input probability GeoTIFF OR directory of tiles. Directory input is always "
            f"mosaicked first. Default: {default_input}"
        ),
    )
    p.add_argument(
        "--output",
        default=str(default_output),
        help=(
            f"Output GeoTIFF (default: {default_output}). 2-band with binary in band 1 "
            "and probabilities in band 2."
        ),
    )
    p.add_argument(
        "--band",
        type=int,
        default=None,
        help="Probability band index (1-based). Default: auto-select band 2 if present, else band 1",
    )
    p.add_argument("--low", type=float, default=0.26, help="Low/expansion threshold (default: 0.26)")
    p.add_argument("--high", type=float, default=0.52, help="High/seed threshold (default: 0.52)")
    p.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Pixel connectivity (4 or 8, default: 8)",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=3,
        help="Optional minimum kept patch size in pixels after hysteresis (default: 3)",
    )
    p.add_argument(
        "--method",
        choices=["auto", "propagate", "label"],
        default="auto",
        help=(
            "Hysteresis engine: "
            "propagate=lower RAM, "
            "label=single-pass component selection (often fastest if RAM allows), "
            "auto=propagate when --min-size<=1 else label"
        ),
    )
    p.add_argument(
        "--prob-scale",
        choices=["auto", "1", "255"],
        default="auto",
        help=(
            "How probabilities are stored: 0-1, 0-255, or auto detect. "
            "If auto and max(valid) > 1.0, values are divided by 255."
        ),
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Print summary stats without writing output",
    )
    p.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only report pixel count and estimated peak RAM; skip full raster read",
    )
    p.add_argument(
        "--pattern",
        default="*.tif",
        help="Tile glob pattern when --input is a directory (default: *.tif)",
    )
    p.add_argument(
        "--balanced-output",
        type=str,
        default=None,
        help=(
            "Optional path for a 'balanced' output using flat --low threshold "
            "(no seed requirement). Same MMU/connectivity filtering. "
            "Higher recall, lower precision than hysteresis."
        ),
    )
    p.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Optional text report path. Default: <repo>/wwr/data/output/reports/<output>.report.txt",
    )
    return p.parse_args()


def _write_hysteresis_report(
    report_path: Path,
    *,
    mode: str,
    input_label: str,
    output_path: Path | None,
    balanced_output_path: Path | None,
    prob_band: int,
    low: float,
    high: float,
    connectivity: int,
    method: str,
    min_size: int,
    est_ram: float,
    num_pixels: int,
    valid_pixels: int | None,
    low_pixels: int | None,
    high_pixels: int | None,
    kept_before_min: int | None,
    kept_after_min: int | None,
    kept_area_ha: float | None,
    balanced_before_min: int | None = None,
    balanced_after_min: int | None = None,
    balanced_area_ha: float | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 70,
        "WET WOODLAND HYSTERESIS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "1. INPUTS",
        "-" * 70,
        f"Input: {input_label}",
        f"Mode: {mode}",
        f"Band: {prob_band}",
        f"Output raster: {output_path if output_path is not None else 'Not written'}",
        f"Balanced output: {balanced_output_path if balanced_output_path is not None else 'Not written'}",
        "",
        "2. SETTINGS",
        "-" * 70,
        f"Low threshold: {low}",
        f"High threshold: {high}",
        f"Connectivity: {connectivity}",
        f"Method: {method}",
        f"Minimum patch size: {min_size} px",
        f"Estimated peak RAM (GiB): {est_ram:.2f}",
        "",
        "3. SUMMARY",
        "-" * 70,
        f"Total pixels: {num_pixels:,}",
    ]
    if valid_pixels is not None:
        lines.append(f"Valid pixels: {valid_pixels:,}")
    if low_pixels is not None:
        lines.append(f"Low-mask pixels: {_format_count_share(low_pixels, valid_pixels)}")
    if high_pixels is not None:
        lines.append(f"High-seed pixels: {_format_count_share(high_pixels, valid_pixels)}")
    if kept_before_min is not None:
        lines.append(f"Kept before MMU: {_format_count_share(kept_before_min, valid_pixels)}")
    if kept_after_min is not None:
        lines.append(f"Kept final: {_format_count_share(kept_after_min, valid_pixels)}")
    if kept_area_ha is not None and np.isfinite(kept_area_ha):
        lines.append(
            f"Kept area (ha): {kept_area_ha:.2f}{_format_area_share_suffix(kept_after_min, valid_pixels)}"
        )

    if balanced_after_min is not None:
        lines.extend(
            [
                "",
                "4. BALANCED OUTPUT",
                "-" * 70,
                f"Kept before MMU: {_format_count_share(balanced_before_min, valid_pixels)}",
                f"Kept final: {_format_count_share(balanced_after_min, valid_pixels)}",
            ]
        )
        if balanced_area_ha is not None and np.isfinite(balanced_area_ha):
            lines.append(
                f"Kept area (ha): {balanced_area_ha:.2f}{_format_area_share_suffix(balanced_after_min, valid_pixels)}"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _estimate_peak_ram_gb(num_pixels: int, method: str, min_size: int) -> float:
    """
    Rough working-set estimate (GiB) excluding Python overhead.
    """
    # Common arrays: probs(float32) + valid(bool) + low(bool) + high(bool) + output(uint8)
    bytes_common = num_pixels * (4 + 1 + 1 + 1 + 1)

    if method == "label":
        # Labels(int32) + keep(bool)
        bytes_extra = num_pixels * (4 + 1)
    else:
        # keep(bool)
        bytes_extra = num_pixels * 1
        if min_size > 1:
            # Additional label pass for min-size filtering
            bytes_extra += num_pixels * 4

    return (bytes_common + bytes_extra) / (1024**3)


def _format_count_share(count: int | None, valid_pixels: int | None) -> str:
    if count is None:
        return "NA"
    if valid_pixels is None or valid_pixels <= 0:
        return f"{count:,}"
    return f"{count:,} ({count / valid_pixels:.2%} of valid)"


def _format_area_share_suffix(count: int | None, valid_pixels: int | None) -> str:
    if count is None or valid_pixels is None or valid_pixels <= 0:
        return ""
    return f" ({count / valid_pixels:.2%} of valid area)"


def _normalize_probabilities(arr: np.ndarray, valid_mask: np.ndarray, prob_scale: str) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)

    if prob_scale == "1":
        return arr
    if prob_scale == "255":
        return arr / 255.0

    # auto
    valid_vals = arr[valid_mask]
    if valid_vals.size == 0:
        return arr
    vmax = float(np.nanmax(valid_vals))
    if vmax > 1.0 + 1e-6:
        return arr / 255.0
    return arr


def _hysteresis_mask(
    probs: np.ndarray,
    valid_mask: np.ndarray,
    low: float,
    high: float,
    connectivity: int,
    min_size: int,
    method: str,
) -> tuple[np.ndarray, int, int]:
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)

    low_mask = valid_mask & (probs >= low)
    high_mask = valid_mask & (probs >= high)

    if method == "propagate":
        # Morphological reconstruction: grow high seeds inside low mask.
        keep = ndimage.binary_propagation(high_mask, structure=structure, mask=low_mask)
        kept_before_min = int(np.count_nonzero(keep))

        if min_size > 1 and kept_before_min > 0:
            labels, n_labels = ndimage.label(keep, structure=structure)
            component_sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
            large_enough = component_sizes >= int(min_size)
            large_enough[0] = False
            keep &= large_enough[labels]

        kept_after_min = int(np.count_nonzero(keep))
        return keep, kept_before_min, kept_after_min

    # method == "label"
    # Label components in low mask (fast C implementation)
    labels, n_labels = ndimage.label(low_mask, structure=structure)

    if n_labels == 0:
        return np.zeros_like(low_mask, dtype=bool), 0, 0

    # Keep components that touch at least one high-threshold seed pixel
    touched_counts = np.bincount(labels[high_mask].ravel(), minlength=n_labels + 1)
    keep_component = touched_counts > 0
    keep_component[0] = False
    keep = keep_component[labels] & low_mask

    kept_before_min = int(np.count_nonzero(keep))

    if min_size > 1:
        component_sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
        large_enough = component_sizes >= int(min_size)
        large_enough[0] = False
        keep &= large_enough[labels]

    kept_after_min = int(np.count_nonzero(keep))
    return keep, kept_before_min, kept_after_min


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


def _collect_tiles(input_dir: Path, pattern: str) -> list[Path]:
    tiles = sorted(input_dir.glob(pattern))
    return [p for p in tiles if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]


def _build_mosaic_from_tiles(
    tile_paths: list[Path],
    tile_prob_bands: dict[Path, int],
    nodata_value: float | None,
    temp_dir: Path,
) -> Path:
    """
    Build a single-band probability mosaic from tiles.
    Prefers GDAL VRT for scalability; falls back to rasterio.merge if needed.
    Returns path to mosaic source (VRT or GeoTIFF) with probability in band 1.
    """
    unique_bands = sorted({int(tile_prob_bands[p]) for p in tile_paths})
    uniform_band = len(unique_bands) == 1

    # Try low-overhead VRT first
    vrt_path = temp_dir / "probability_mosaic.vrt"
    file_list = temp_dir / "tile_list.txt"
    nd = str(float(nodata_value)) if nodata_value is not None else None

    if uniform_band:
        with open(file_list, "w", encoding="utf-8") as f:
            for p in tile_paths:
                f.write(f"{p}\n")

        cmd = ["gdalbuildvrt", "-overwrite", "-b", str(unique_bands[0])]
        if nd is not None:
            cmd.extend(["-srcnodata", nd, "-vrtnodata", nd])
        cmd.extend(["-input_file_list", str(file_list), str(vrt_path)])

        proc = _run_command(cmd)
        if proc.returncode == 0 and vrt_path.exists():
            return vrt_path
    else:
        # Mixed band indices across tiles (e.g., some 1-band probs, some 2-band binary+prob).
        # Build 1-band per-tile VRTs selecting each tile's probability band, then mosaic those.
        tile_vrt_dir = temp_dir / "tile_prob_vrts"
        tile_vrt_dir.mkdir(parents=True, exist_ok=True)
        per_tile_vrts: list[Path] = []
        gdal_translate_ok = True
        for i, p in enumerate(tile_paths):
            b = int(tile_prob_bands[p])
            out_vrt = tile_vrt_dir / f"tile_prob_{i:06d}.vrt"
            cmd = ["gdal_translate", "-of", "VRT", "-b", str(b), str(p), str(out_vrt)]
            if nd is not None:
                cmd.extend(["-a_nodata", nd])
            proc = _run_command(cmd)
            if proc.returncode != 0 or not out_vrt.exists():
                gdal_translate_ok = False
                break
            per_tile_vrts.append(out_vrt)

        if gdal_translate_ok and per_tile_vrts:
            with open(file_list, "w", encoding="utf-8") as f:
                for p in per_tile_vrts:
                    f.write(f"{p}\n")

            cmd = ["gdalbuildvrt", "-overwrite"]
            if nd is not None:
                cmd.extend(["-srcnodata", nd, "-vrtnodata", nd])
            cmd.extend(["-input_file_list", str(file_list), str(vrt_path)])

            proc = _run_command(cmd)
            if proc.returncode == 0 and vrt_path.exists():
                return vrt_path

    # Fallback: explicit merged temporary GeoTIFF
    print("⚠️ gdalbuildvrt unavailable/failed; falling back to rasterio.merge (higher memory).")
    tiff_path = temp_dir / "probability_mosaic.tif"

    if uniform_band:
        source_paths = tile_paths
        merge_band = unique_bands[0]
    else:
        # Slow fallback path for mixed-band tiles when GDAL VRT tools are unavailable.
        temp_tiles_dir = temp_dir / "single_band_tiles"
        temp_tiles_dir.mkdir(parents=True, exist_ok=True)
        source_paths: list[Path] = []
        for i, p in enumerate(tile_paths):
            selected_band = int(tile_prob_bands[p])
            single_path = temp_tiles_dir / f"tile_prob_{i:06d}.tif"
            with rasterio.open(p) as src:
                arr = src.read(selected_band, masked=False)
                tile_profile = src.profile.copy()
                tile_profile.update(
                    {
                        "driver": "GTiff",
                        "count": 1,
                        "dtype": arr.dtype,
                        "compress": "LZW",
                        "tiled": True,
                        "BIGTIFF": "IF_SAFER",
                    }
                )
                if nodata_value is not None:
                    tile_profile["nodata"] = float(nodata_value)
                with rasterio.open(single_path, "w", **tile_profile) as dst:
                    dst.write(arr, 1)
            source_paths.append(single_path)
        merge_band = 1

    srcs = [rasterio.open(p) for p in source_paths]
    try:
        mosaic, out_transform = rio_merge(
            srcs,
            indexes=[merge_band],
            nodata=nodata_value,
            method="first",
        )

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
            }
        )
        if nodata_value is not None:
            profile["nodata"] = float(nodata_value)

        with rasterio.open(tiff_path, "w", **profile) as dst:
            dst.write(mosaic.astype(np.float32))
    finally:
        for s in srcs:
            s.close()

    return tiff_path


def main() -> int:
    args = parse_args()

    if args.high < args.low:
        raise SystemExit("--high must be >= --low")

    in_path = Path(args.input)
    out_path = Path(args.output)
    reports_root = Path(__file__).resolve().parents[2] / "data" / "output" / "reports"
    report_path = Path(args.report_file) if args.report_file else reports_root / f"{out_path.stem}.report.txt"
    balanced_output_path = Path(args.balanced_output) if args.balanced_output else None

    chosen_method = args.method
    if chosen_method == "auto":
        chosen_method = "propagate" if args.min_size <= 1 else "label"
    temp_ctx: tempfile.TemporaryDirectory[str] | None = None
    process_path = in_path
    requested_band = args.band
    input_label = str(in_path)

    try:
        if in_path.is_dir():
            tiles = _collect_tiles(in_path, args.pattern)
            if not tiles:
                raise SystemExit(f"No tiles found in {in_path} matching pattern '{args.pattern}'")

            tile_prob_bands: dict[Path, int] = {}
            tile_nodata: float | None = None
            selected_band_counts: Counter[int] = Counter()
            bad_tiles: list[str] = []

            for i, t in enumerate(tiles):
                with rasterio.open(t) as ds:
                    if i == 0:
                        tile_nodata = ds.nodata

                    if args.band is None:
                        b = 2 if ds.count >= 2 else 1
                    else:
                        if args.band < 1 or args.band > ds.count:
                            bad_tiles.append(t.name)
                            if len(bad_tiles) >= 5:
                                break
                            continue
                        b = int(args.band)

                    tile_prob_bands[t] = b
                    selected_band_counts[b] += 1

            if bad_tiles:
                raise SystemExit(
                    f"Requested probability band {int(args.band)} not found in some tiles (examples: {bad_tiles})"
                )

            print("Mosaic-first mode")
            print("=" * 60)
            print(f"Input dir:         {in_path}")
            print(f"Tiles:             {len(tiles):,}")
            if len(selected_band_counts) > 1:
                mix = ", ".join(
                    f"band{b}: {n:,}" for b, n in sorted(selected_band_counts.items())
                )
                print(f"Tile prob band:    mixed auto-selection ({mix})")
            else:
                only_band = next(iter(selected_band_counts))
                print(f"Tile prob band:    {only_band}")

            temp_ctx = tempfile.TemporaryDirectory(prefix="hysteresis_mosaic_")
            temp_dir = Path(temp_ctx.name)
            process_path = _build_mosaic_from_tiles(
                tile_paths=tiles,
                tile_prob_bands=tile_prob_bands,
                nodata_value=(float(tile_nodata) if tile_nodata is not None else None),
                temp_dir=temp_dir,
            )
            requested_band = 1
            input_label = f"{in_path} (mosaic of {len(tiles):,} tiles)"
            print(f"Mosaic source:     {process_path}")
        elif not in_path.exists():
            raise SystemExit(f"Input path not found: {in_path}")

        print(f"Output raster:     {out_path}")
        print(f"Report file:       {report_path}")
        if balanced_output_path is not None:
            print(f"Balanced output:   {balanced_output_path}")

        with rasterio.open(process_path) as src:
            if requested_band is None:
                prob_band = 2 if src.count >= 2 else 1
            else:
                if requested_band < 1 or requested_band > src.count:
                    raise SystemExit(f"--band must be between 1 and {src.count}")
                prob_band = int(requested_band)

            num_pixels = int(src.width * src.height)
            est_ram = _estimate_peak_ram_gb(num_pixels, chosen_method, args.min_size)
            if args.estimate_only:
                print("Hysteresis estimate")
                print("=" * 60)
                print(f"Input:            {input_label}")
                print(f"Band:             {prob_band}")
                print(f"Pixels:           {num_pixels:,}")
                print(f"Low threshold:    {args.low}  (expansion — balanced-accuracy optimised)")
                print(f"High threshold:   {args.high}  (seed — F-beta 0.5 optimised)")
                print(f"Connectivity:     {args.connectivity}")
                print(f"Method:           {chosen_method}")
                print(f"Min patch size:   {args.min_size} px")
                print(f"Est. peak RAM:    ~{est_ram:.2f} GiB")
                _write_hysteresis_report(
                    report_path,
                    mode="estimate_only",
                    input_label=input_label,
                    output_path=None,
                    balanced_output_path=balanced_output_path,
                    prob_band=prob_band,
                    low=args.low,
                    high=args.high,
                    connectivity=args.connectivity,
                    method=chosen_method,
                    min_size=args.min_size,
                    est_ram=est_ram,
                    num_pixels=num_pixels,
                    valid_pixels=None,
                    low_pixels=None,
                    high_pixels=None,
                    kept_before_min=None,
                    kept_after_min=None,
                    kept_area_ha=None,
                )
                print(f"Report written:   {report_path}")
                return 0

            probs_raw = src.read(prob_band, masked=False)
            profile = src.profile.copy()
            num_pixels = int(probs_raw.size)

            nodata = src.nodata
            if nodata is None:
                # Conservative valid mask when nodata is unspecified.
                valid_mask = np.isfinite(probs_raw)
            else:
                valid_mask = np.isfinite(probs_raw) & (probs_raw != nodata)

            probs = _normalize_probabilities(probs_raw, valid_mask, args.prob_scale)
            valid_pixels = int(np.count_nonzero(valid_mask))

            low_pixels = int(np.count_nonzero(valid_mask & (probs >= args.low)))
            high_pixels = int(np.count_nonzero(valid_mask & (probs >= args.high)))

            keep, kept_before_min, kept_after_min = _hysteresis_mask(
                probs=probs,
                valid_mask=valid_mask,
                low=args.low,
                high=args.high,
                connectivity=args.connectivity,
                min_size=args.min_size,
                method=chosen_method,
            )

            px_area_m2 = abs(src.transform.a * src.transform.e)
            if not np.isfinite(px_area_m2) or px_area_m2 <= 0:
                px_area_m2 = math.nan

            print("Hysteresis summary")
            print("=" * 60)
            print(f"Input:            {input_label}")
            print(f"Band:             {prob_band}")
            print(f"Pixels:           {num_pixels:,}")
            print(f"Low threshold:    {args.low}  (expansion — balanced-accuracy optimised)")
            print(f"High threshold:   {args.high}  (seed — F-beta 0.5 optimised)")
            print(f"Connectivity:     {args.connectivity}")
            print(f"Method:           {chosen_method}")
            print(f"Min patch size:   {args.min_size} px")
            print(f"Est. peak RAM:    ~{est_ram:.2f} GiB")
            print(f"Valid pixels:     {valid_pixels:,}")
            print(f"Low-mask pixels:  {_format_count_share(low_pixels, valid_pixels)}")
            print(f"High-seed pixels: {_format_count_share(high_pixels, valid_pixels)}")
            print(f"Kept (pre-min):   {_format_count_share(kept_before_min, valid_pixels)}")
            print(f"Kept (final):     {_format_count_share(kept_after_min, valid_pixels)}")
            if np.isfinite(px_area_m2):
                print(
                    "Kept area:        "
                    f"{kept_after_min * px_area_m2 / 10_000:.2f} ha"
                    f"{_format_area_share_suffix(kept_after_min, valid_pixels)}"
                )

            if args.report_only:
                _write_hysteresis_report(
                    report_path,
                    mode="report_only",
                    input_label=input_label,
                    output_path=None,
                    balanced_output_path=balanced_output_path,
                    prob_band=prob_band,
                    low=args.low,
                    high=args.high,
                    connectivity=args.connectivity,
                    method=chosen_method,
                    min_size=args.min_size,
                    est_ram=est_ram,
                    num_pixels=num_pixels,
                    valid_pixels=valid_pixels,
                    low_pixels=low_pixels,
                    high_pixels=high_pixels,
                    kept_before_min=kept_before_min,
                    kept_after_min=kept_after_min,
                    kept_area_ha=(kept_after_min * px_area_m2 / 10_000 if np.isfinite(px_area_m2) else None),
                )
                print(f"Report written:   {report_path}")
                return 0

            out_path.parent.mkdir(parents=True, exist_ok=True)

            nodata_out = float(nodata) if nodata is not None else -9999.0

            # Band 1: cleaned binary labels
            bin_out = np.full(probs.shape, nodata_out, dtype=np.float32)
            bin_out[valid_mask] = 0.0
            bin_out[keep] = 1.0

            # Band 2: probabilities masked by hysteresis keep-mask
            prob_out = np.full(probs_raw.shape, nodata_out, dtype=np.float32)
            prob_out[valid_mask] = 0.0
            prob_out[keep] = probs_raw[keep].astype(np.float32)

            profile.update(
                {
                    "driver": "GTiff",
                    "count": 2,
                    "dtype": rasterio.float32,
                    "nodata": nodata_out,
                    "compress": Compression.deflate.value,
                    "predictor": 3,
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                    "BIGTIFF": "IF_SAFER",
                }
            )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(bin_out, 1)
                dst.write(prob_out, 2)
                dst.set_band_description(1, "Wet Woodland Hysteresis Binary")
                dst.set_band_description(2, "Hysteresis-Masked Probability")

        print(f"Output written: {out_path}")

        # Optional: balanced output (flat threshold, no seed requirement)
        balanced_before_min = None
        balanced_after_min = None
        balanced_area_ha = None
        if args.balanced_output:
            balanced_path = Path(args.balanced_output)
            balanced_path.parent.mkdir(parents=True, exist_ok=True)

            # Use low threshold for both (every pixel above low is its own seed)
            keep_bal, kept_bal_before, kept_bal_after = _hysteresis_mask(
                probs=probs,
                valid_mask=valid_mask,
                low=args.low,
                high=args.low,  # Same as low = flat threshold
                connectivity=args.connectivity,
                min_size=args.min_size,
                method=chosen_method,
            )

            print()
            print("Balanced output summary")
            print("=" * 60)
            print(f"Threshold:        {args.low} (flat, no seeds)")
            print(f"Kept (pre-min):   {_format_count_share(kept_bal_before, valid_pixels)}")
            print(f"Kept (final):     {_format_count_share(kept_bal_after, valid_pixels)}")
            if np.isfinite(px_area_m2):
                print(
                    "Kept area:        "
                    f"{kept_bal_after * px_area_m2 / 10_000:.2f} ha"
                    f"{_format_area_share_suffix(kept_bal_after, valid_pixels)}"
                )

            # Band 1: cleaned binary labels
            bin_bal = np.full(probs.shape, nodata_out, dtype=np.float32)
            bin_bal[valid_mask] = 0.0
            bin_bal[keep_bal] = 1.0

            # Band 2: probabilities masked by balanced keep-mask
            prob_bal = np.full(probs_raw.shape, nodata_out, dtype=np.float32)
            prob_bal[valid_mask] = 0.0
            prob_bal[keep_bal] = probs_raw[keep_bal].astype(np.float32)

            with rasterio.open(balanced_path, "w", **profile) as dst:
                dst.write(bin_bal, 1)
                dst.write(prob_bal, 2)
                dst.set_band_description(1, "Wet Woodland Balanced Binary")
                dst.set_band_description(2, "Balanced-Masked Probability")

            print(f"Balanced output written: {balanced_path}")
            balanced_before_min = kept_bal_before
            balanced_after_min = kept_bal_after
            balanced_area_ha = kept_bal_after * px_area_m2 / 10_000 if np.isfinite(px_area_m2) else None

        _write_hysteresis_report(
            report_path,
            mode="full",
            input_label=input_label,
            output_path=out_path,
            balanced_output_path=balanced_output_path,
            prob_band=prob_band,
            low=args.low,
            high=args.high,
            connectivity=args.connectivity,
            method=chosen_method,
            min_size=args.min_size,
            est_ram=est_ram,
            num_pixels=num_pixels,
            valid_pixels=int(np.count_nonzero(valid_mask)),
            low_pixels=low_pixels,
            high_pixels=high_pixels,
            kept_before_min=kept_before_min,
            kept_after_min=kept_after_min,
            kept_area_ha=(kept_after_min * px_area_m2 / 10_000 if np.isfinite(px_area_m2) else None),
            balanced_before_min=balanced_before_min,
            balanced_after_min=balanced_after_min,
            balanced_area_ha=balanced_area_ha,
        )
        print(f"Report written: {report_path}")

        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
