#!/usr/bin/env python3
"""
Write a compact report of the largest mapped wet woodland patches.

This utility is intentionally lightweight. It can:
1. Read a patch polygon layer, compute area and representative BNG points,
   join each patch to an LNRS polygon, and export the top-N table.
2. Rebuild the text report from a previously saved CSV.

For the March 2026 quick extraction, the input patch layer was derived from the
final binary wet woodland raster using 8-neighbour connectivity and a 10 ha
large-patch sieve before polygonisation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_lnrs = repo_root / "data" / "input" / "boundaries" / "lnrs_areas.shp"
    default_reports = repo_root / "data" / "output" / "reports"
    parser = argparse.ArgumentParser(description="Report the largest mapped wet woodland patches.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--patches", help="Input patch polygon layer (e.g. GPKG).")
    source.add_argument("--csv-in", help="Previously saved CSV of top patches.")
    parser.add_argument("--lnrs", default=str(default_lnrs), help=f"LNRS polygon layer (default: {default_lnrs})")
    parser.add_argument("--top-n", type=int, default=10, help="Number of patches to retain (default: 10)")
    parser.add_argument(
        "--method-note",
        default=(
            "Quick large-patch extraction from the final binary wet woodland raster "
            "using 8-neighbour connectivity, retaining patches >= 10 ha before polygonisation."
        ),
        help="Short note describing the extraction method.",
    )
    parser.add_argument(
        "--csv-out",
        default=str(default_reports / "top_large_wet_woodland_patches.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--report-out",
        default=str(default_reports / "top_large_wet_woodland_patches.txt"),
        help="Output text report path.",
    )
    return parser.parse_args()


def build_top_table_from_patches(patches_path: str, lnrs_path: str, top_n: int) -> pd.DataFrame:
    patches = gpd.read_file(patches_path)
    if "DN" in patches.columns:
        patches = patches.loc[patches["DN"] == 1].copy()
    patches = patches.loc[patches.geometry.notnull()].copy()
    if patches.empty:
        raise SystemExit(f"No wet patches found in {patches_path}")

    patches["area_ha"] = patches.geometry.area.astype(float) / 10000.0
    rep_points = patches.geometry.representative_point()
    top = patches.loc[:, ["area_ha"]].copy()
    top["bng_easting"] = rep_points.x.round(0).astype(int)
    top["bng_northing"] = rep_points.y.round(0).astype(int)

    lnrs = gpd.read_file(lnrs_path)
    name_field = "Name" if "Name" in lnrs.columns else ("NAME" if "NAME" in lnrs.columns else None)
    if name_field is None:
        raise SystemExit(f"Could not find an LNRS name field in {lnrs_path}")
    lnrs = lnrs[[name_field, "geometry"]].to_crs(patches.crs)
    rep_gdf = gpd.GeoDataFrame(top.copy(), geometry=rep_points, crs=patches.crs)
    joined = gpd.sjoin(rep_gdf, lnrs, how="left", predicate="within")
    top["lnrs_name"] = joined[name_field].fillna("Unknown").astype(str).values
    top = top.sort_values("area_ha", ascending=False).head(top_n).reset_index(drop=True)
    top.insert(0, "rank", range(1, len(top) + 1))
    return top


def read_top_table(csv_path: str, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"rank", "area_ha", "bng_easting", "bng_northing", "lnrs_name"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing required columns: {sorted(missing)}")
    df = df.sort_values("rank").head(top_n).copy()
    return df


def write_report(df: pd.DataFrame, report_path: str, source_label: str, method_note: str) -> None:
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    lnrs_counts = Counter(df["lnrs_name"].astype(str))
    with report_file.open("w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TOP LARGE WET WOODLAND PATCHES\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {source_label}\n")
        f.write(f"Method: {method_note}\n\n")
        f.write("Rank  Area_ha  BNG_E  BNG_N  LNRS\n")
        f.write("-" * 70 + "\n")
        for row in df.itertuples(index=False):
            f.write(
                f"{int(row.rank):>4}  "
                f"{float(row.area_ha):>7.2f}  "
                f"{int(row.bng_easting):>5}  "
                f"{int(row.bng_northing):>5}  "
                f"{row.lnrs_name}\n"
            )
        f.write("\nLNRS frequency among top patches\n")
        f.write("-" * 70 + "\n")
        for name, count in lnrs_counts.most_common():
            f.write(f"{name}: {count}\n")


def main() -> int:
    args = parse_args()
    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    if args.patches:
        top = build_top_table_from_patches(args.patches, args.lnrs, args.top_n)
        top.to_csv(csv_out, index=False)
        source_label = args.patches
    else:
        top = read_top_table(args.csv_in, args.top_n)
        top.to_csv(csv_out, index=False)
        source_label = args.csv_in

    write_report(top, args.report_out, source_label=source_label, method_note=args.method_note)
    print(f"Wrote CSV: {csv_out}")
    print(f"Wrote report: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
