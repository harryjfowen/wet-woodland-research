#!/usr/bin/env python3
"""
Legacy wrapper for running the two-stage abiotic preprocessing pipeline.

Stage 1:
  build_dtm_metrics.py
  - computes DTM metrics (elevation/slope/aspect/CTI)
  - writes mosaic to data/output/preprocess/dtm_metrics.tif

Stage 2:
  build_abiotic_stack.py
  - aligns abiotic predictors to the template grid
  - writes predictor stack to data/output/potential/potential_predictors.tif
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def build_parser() -> argparse.ArgumentParser:
    tow_root = Path(__file__).resolve().parents[2]

    default_dtm_dir = tow_root / "data" / "input" / "dtm"
    default_dtm_outdir = tow_root / "data" / "output" / "preprocess"
    default_template = default_dtm_outdir / "dtm_metrics.tif"

    default_outdir = tow_root / "data" / "output" / "potential"
    default_output_stack = default_outdir / "potential_predictors.tif"
    default_peat = tow_root / "data" / "input" / "peat" / "peat_probability.tif"
    default_smuk = tow_root / "data" / "input" / "smuk"
    default_soils = tow_root / "data" / "input" / "soils" / "soils_parent_material.shp"
    default_rivers = tow_root / "data" / "input" / "hydro" / "rivers.shp"
    default_lakes = tow_root / "data" / "input" / "hydro" / "lakes.gpkg"
    default_mask = tow_root / "data" / "input" / "boundaries" / "england.shp"

    p = argparse.ArgumentParser(
        description=(
            "Run DTM + abiotic preprocessing end-to-end with canonical defaults "
            "under tow/data/output."
        )
    )
    stage_group = p.add_mutually_exclusive_group()
    stage_group.add_argument("--dtm-only", action="store_true", help="Run only DTM stage (stage 1).")
    stage_group.add_argument("--abiotic-only", action="store_true", help="Run only abiotic stage (stage 2).")

    # Stage 1 (DTM metrics)
    p.add_argument("--dtm-dir", type=Path, default=default_dtm_dir, help=f"DTM tile directory (default: {default_dtm_dir})")
    p.add_argument("--dtm-outdir", type=Path, default=default_dtm_outdir, help=f"DTM metrics output directory (default: {default_dtm_outdir})")
    p.add_argument("--tile-size", type=int, default=8192, help="DTM processing tile size in pixels (default: 8192)")
    p.add_argument(
        "--buffer",
        type=int,
        default=512,
        help="DTM processing buffer in pixels for CTI/flow context (default: 512)",
    )
    p.add_argument("--pixel-size", type=float, default=10.0, help="Native DTM pixel size in meters (default: 10)")
    p.add_argument("--output-resolution", type=float, default=250.0, help="Stage-1 output resolution in meters (default: 250)")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers for stage 1 (default: 4)")
    p.add_argument("--pattern", default="*.tif", help="Glob for DTM tiles (default: *.tif)")
    p.add_argument("--no-mosaic", action="store_true", help="Do not build the DTM mosaic in stage 1")

    # Stage 2 (abiotic stack)
    p.add_argument(
        "--template",
        type=Path,
        default=None,
        help=f"Template raster path for stage 2 (default: <dtm-outdir>/dtm_metrics.tif, i.e. {default_template})",
    )
    p.add_argument(
        "--dtm-metrics",
        type=Path,
        default=None,
        help=f"DTM metrics raster for stage 2 (default: <dtm-outdir>/dtm_metrics.tif, i.e. {default_template})",
    )
    p.add_argument("--resolution", type=float, default=None, help="Optional stage-2 output resolution override (meters)")
    p.add_argument("--output-stack", type=Path, default=default_output_stack, help=f"Output predictor stack path (default: {default_output_stack})")
    p.add_argument("--outdir", type=Path, default=default_outdir, help=f"Output directory hint for stage 2 (default: {default_outdir})")
    p.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Copy stage-2 abiotic intermediate files into <output-stack-dir>/intermediates before cleanup.",
    )
    p.add_argument("--peat-prob", type=Path, default=default_peat, help=f"Peat probability raster (default: {default_peat})")
    p.add_argument("--smuk", type=Path, default=default_smuk, help=f"SMUK raster or directory (default: {default_smuk})")
    p.add_argument("--soils-shp", type=Path, default=default_soils, help=f"Soils shapefile (default: {default_soils})")
    p.add_argument("--rivers", type=Path, default=default_rivers, help=f"Rivers shapefile (default: {default_rivers})")
    p.add_argument("--lakes", type=Path, default=default_lakes, help=f"Lakes vector (default: {default_lakes})")
    p.add_argument("--mask-shp", type=Path, default=default_mask, help=f"Mask polygon shapefile (default: {default_mask})")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    tow_root = Path(__file__).resolve().parents[2]
    stage1_script = Path(__file__).with_name("build_dtm_metrics.py")
    stage2_script = Path(__file__).with_name("build_abiotic_stack.py")

    run_stage1 = not args.abiotic_only
    run_stage2 = not args.dtm_only

    if run_stage1:
        cmd_stage1 = [
            sys.executable,
            str(stage1_script),
            "--dtm-dir",
            str(args.dtm_dir),
            "--outdir",
            str(args.dtm_outdir),
            "--tile-size",
            str(args.tile_size),
            "--buffer",
            str(args.buffer),
            "--pixel-size",
            str(args.pixel_size),
            "--output-resolution",
            str(args.output_resolution),
            "--workers",
            str(args.workers),
            "--pattern",
            str(args.pattern),
        ]
        if not args.no_mosaic:
            cmd_stage1.append("--mosaic")
        _run(cmd_stage1, cwd=tow_root)

    if run_stage2:
        template = args.template if args.template is not None else (args.dtm_outdir / "dtm_metrics.tif")
        dtm_metrics = args.dtm_metrics if args.dtm_metrics is not None else template

        if not template.exists():
            print(f"ERROR: Stage-2 template not found: {template}")
            print("Hint: run stage 1 with mosaic enabled, or pass --template explicitly.")
            return 1
        if not dtm_metrics.exists():
            print(f"ERROR: Stage-2 DTM metrics raster not found: {dtm_metrics}")
            print("Hint: run stage 1 first, or pass --dtm-metrics explicitly.")
            return 1

        cmd_stage2 = [
            sys.executable,
            str(stage2_script),
            "--template",
            str(template),
            "--dtm-metrics",
            str(dtm_metrics),
            "--output-stack",
            str(args.output_stack),
            "--outdir",
            str(args.outdir),
            "--peat-prob",
            str(args.peat_prob),
            "--smuk",
            str(args.smuk),
            "--soils-shp",
            str(args.soils_shp),
            "--rivers",
            str(args.rivers),
            "--lakes",
            str(args.lakes),
            "--mask-shp",
            str(args.mask_shp),
        ]
        if args.resolution is not None:
            cmd_stage2.extend(["--resolution", str(args.resolution)])
        if args.save_intermediates:
            cmd_stage2.append("--save-intermediates")

        _run(cmd_stage2, cwd=tow_root)

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
