#!/usr/bin/env python3
"""
Compartments Processor for Wet Woodland Species Filtering

Wet-positive labels require a wet-indicator primary species (alder, willow, or
poplar). Birch is treated conservatively for pixel-level modelling: only
explicit downy birch records (species code ``PBI``) are allowed to support
wet-positive labels. On peat, ``PBI`` may be the primary species; off peat it
may also be primary, but alder must still be present somewhere in the recorded
species. Generic birch and silver birch do not support wet-positive label
assignment unless explicitly excluded via ``--exclude-birch``.
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
import gc
import os
import pandas as pd
import re
import sys
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# Define target species for filtering
TARGET_SPECIES = ['alder', 'willow', 'poplar']
REPORT_TARGET_SPECIES = ['alder', 'downy birch (PBI only)', 'willow', 'poplar']
WET_SPECIES = ['alder', 'willow', 'poplar']  # Species indicating wet conditions
PRIMARY_SPECIES_COLS = ['PRISPECIES', 'Primary_Species', 'Species', 'SPECIES']
SECONDARY_SPECIES_COLS = ['SECSPECIES', 'Secondary_Species', 'SEC_SPECIES']
TERTIARY_SPECIES_COLS = ['TERSPECIES', 'Tertiary_Species', 'TER_SPECIES']
SPECIES_CODE_BY_COLUMN = {
    'PRISPECIES': 'PRI_SPCODE',
    'Primary_Species': 'PRI_SPCODE',
    'Species': 'PRI_SPCODE',
    'SPECIES': 'PRI_SPCODE',
    'SECSPECIES': 'SEC_SPCODE',
    'Secondary_Species': 'SEC_SPCODE',
    'SEC_SPECIES': 'SEC_SPCODE',
    'TERSPECIES': 'TER_SPCODE',
    'Tertiary_Species': 'TER_SPCODE',
    'TER_SPECIES': 'TER_SPCODE',
}
DOWNY_BIRCH_CODE = 'PBI'
ANY_BIRCH_CODES = {'BI', 'PBI', 'SBI', 'XBI'}
DOWNY_BIRCH_LABEL = 'downy birch'

# Functional type split for non-wet background classes
DECIDUOUS_SPECIES = [
    'oak', 'beech', 'ash', 'birch', 'alder', 'willow', 'poplar',
    'sycamore', 'maple', 'cherry', 'hornbeam', 'lime', 'hazel',
    'sweet chestnut', 'walnut', 'rowan', 'aspen', 'elm',
    'larch'  # Larch is a deciduous conifer (loses needles in autumn)
]

EVERGREEN_SPECIES = [
    'pine', 'spruce', 'fir', 'cedar', 'douglas',
    'hemlock', 'redwood', 'yew', 'juniper', 'cypress',
    'western red cedar', 'lawson', 'thuja', 'sequoia',
    'sitka', 'norway spruce', 'scots pine', 'corsican pine'
]

DECIDUOUS_HINTS = ['broadleaf', 'broad leaf', 'deciduous', 'hardwood']
EVERGREEN_HINTS = ['conifer', 'coniferous', 'evergreen', 'softwood']
EMPTY_SPECIES_VALUES = {"", "nan", "none", "n/a"}
KNOWN_TREE_SPECIES = sorted(set(TARGET_SPECIES + DECIDUOUS_SPECIES + EVERGREEN_SPECIES), key=len, reverse=True)


def _normalise_species_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _normalise_species_code(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _mentions_species(text, species):
    if not text:
        return False
    return re.search(rf"\b{re.escape(species)}\b", text) is not None


def _species_hits(text, species_list):
    return {species for species in species_list if _mentions_species(text, species)}


def _resolve_code_column(species_column, available_columns):
    code_col = SPECIES_CODE_BY_COLUMN.get(species_column)
    if code_col and code_col in available_columns:
        return code_col
    return None


def _contains_any_species(series, species_list):
    text_series = series.astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for species in species_list:
        mask |= text_series.str.contains(rf"\b{re.escape(species)}\b", na=False, regex=True)
    return mask


def _series_has_explicit_downy_birch(species_series, code_series=None):
    text_mask = species_series.astype(str).str.lower().str.contains(rf"\b{re.escape(DOWNY_BIRCH_LABEL)}\b", na=False, regex=True)
    if code_series is None:
        return text_mask
    code_mask = code_series.astype(str).str.strip().str.upper().eq(DOWNY_BIRCH_CODE)
    return text_mask | code_mask


def _cell_has_any_birch(value, code=None):
    text = _normalise_species_value(value)
    species_code = _normalise_species_code(code)
    return species_code in ANY_BIRCH_CODES or _mentions_species(text, 'birch')


def _cell_has_downy_birch(value, code=None):
    text = _normalise_species_value(value)
    species_code = _normalise_species_code(code)
    return species_code == DOWNY_BIRCH_CODE or _mentions_species(text, DOWNY_BIRCH_LABEL)


def _row_has_any_birch(row, species_columns, code_columns):
    for species_col, code_col in zip(species_columns, code_columns):
        if species_col in row.index:
            code_val = row[code_col] if code_col and code_col in row.index else None
            if _cell_has_any_birch(row[species_col], code_val):
                return True
    return False


def _row_species_hits(row, columns, species_list):
    hits = set()
    for col in columns:
        if col in row.index:
            hits.update(_species_hits(_normalise_species_value(row[col]), species_list))
    return hits


def _contains_only_allowed_species(value, allowed_species, code=None):
    text = _normalise_species_value(value)
    if text in EMPTY_SPECIES_VALUES:
        return True

    if _cell_has_any_birch(text, code):
        return _cell_has_downy_birch(text, code)

    hits = _species_hits(text, KNOWN_TREE_SPECIES)
    if not hits:
        return False

    return hits.issubset(set(allowed_species))

def classify_nonwet_functional_type(row, primary_col, existing_secondary_cols, existing_tertiary_cols):
    """
    Classify non-wet background into:
      0 = evergreen
      1 = deciduous
    """
    def species_hits(text):
        text_lower = str(text).lower()
        deciduous_hits = [sp for sp in DECIDUOUS_SPECIES if sp in text_lower]
        evergreen_hits = [sp for sp in EVERGREEN_SPECIES if sp in text_lower]
        return deciduous_hits, evergreen_hits

    texts = [row[primary_col]]
    texts.extend([row[col] for col in existing_secondary_cols if col in row.index])
    texts.extend([row[col] for col in existing_tertiary_cols if col in row.index])

    primary_deciduous, primary_evergreen = species_hits(row[primary_col])

    all_deciduous = set(primary_deciduous)
    all_evergreen = set(primary_evergreen)
    merged_text = ""

    for val in texts:
        if pd.isna(val):
            continue
        txt = str(val).lower()
        merged_text += " " + txt
        dec_hits, evg_hits = species_hits(txt)
        all_deciduous.update(dec_hits)
        all_evergreen.update(evg_hits)

    has_deciduous = len(all_deciduous) > 0
    has_evergreen = len(all_evergreen) > 0
    primary_is_deciduous = len(primary_deciduous) > 0
    primary_is_evergreen = len(primary_evergreen) > 0

    if has_evergreen and not has_deciduous:
        return 0
    if has_deciduous and not has_evergreen:
        return 1

    # Mixed species: use primary species as tie-breaker when possible
    if has_deciduous and has_evergreen:
        if primary_is_evergreen and not primary_is_deciduous:
            return 0
        if primary_is_deciduous and not primary_is_evergreen:
            return 1
        return 0 if len(all_evergreen) > len(all_deciduous) else 1

    # No direct species match: use broad type hints
    if any(hint in merged_text for hint in EVERGREEN_HINTS) and not any(hint in merged_text for hint in DECIDUOUS_HINTS):
        return 0
    if any(hint in merged_text for hint in DECIDUOUS_HINTS):
        return 1

    # Conservative fallback: deciduous non-wet
    return 1


def class_name(code, binary=False, legacy_classes=False):
    """Human-readable class names for reporting."""
    if binary:
        return "wet woodland (combined)" if code == 1 else "not wet woodland"

    if legacy_classes:
        mapping = {
            0: "not wet woodland",
            1: "wet woodland on peat",
            2: "wet woodland not on peat",
        }
        return mapping.get(code, "unknown")

    mapping = {
        0: "non-wet evergreen",
        1: "non-wet deciduous",
        2: "wet woodland on peat",
        3: "wet woodland not on peat",
    }
    return mapping.get(code, "unknown")


def write_labels_report(
    report_path,
    *,
    args,
    shapefile_path,
    peat_extent_path,
    use_peat,
    output_path,
    output_raster_path,
    species_info,
    effective_target_species,
    original_count,
    target_count,
    non_target_count,
    result_count,
    dissolved_count,
    class_counts,
    raster_summary,
):
    """Write a concise single-file report for paper drafting and QA."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "WET WOODLAND LABEL GENERATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "1. INPUTS",
        "-" * 70,
        f"Subcompartments: {shapefile_path}",
        f"Peat shapefile: {peat_extent_path if use_peat else 'Not used'}",
        f"Output shapefile: {output_path}",
        f"Output raster: {output_raster_path if output_raster_path is not None else 'Not generated'}",
        "",
        "2. SETTINGS",
        "-" * 70,
        f"Binary mode: {bool(args.binary)}",
        f"Legacy classes: {bool(args.legacy_classes)}",
        f"Split background classes: {bool((not args.binary) and (not args.legacy_classes))}",
        f"Exclude birch: {bool(args.exclude_birch)}",
        "Birch rule: only explicit downy birch (PBI) counts; on peat it may be primary, off peat it may be primary only with alder present",
        f"Consolidate: {bool(args.consolidate)}",
        f"Consolidate by: {args.consolidate_by if args.consolidate_by else 'woodland_type'}",
        f"Target species: {', '.join(effective_target_species)}",
        f"Species columns detected: {', '.join(species_info.get('columns', [])) if species_info.get('found') else 'None'}",
        "",
        "3. SOURCE COUNTS",
        "-" * 70,
        f"Original subcompartments: {original_count:,}",
        f"Compartments with target species: {target_count:,}",
        f"Compartments without target species: {non_target_count:,}",
        f"Classified polygon features retained: {result_count:,}",
        f"Dissolved output polygons: {dissolved_count:,}",
        "",
        "4. CLASS COUNTS",
        "-" * 70,
    ]

    if class_counts:
        for woodland_type in sorted(class_counts):
            type_name = class_name(int(woodland_type), binary=args.binary, legacy_classes=args.legacy_classes)
            lines.append(f"Class {int(woodland_type)} ({type_name}): {int(class_counts[woodland_type]):,} features")
    else:
        lines.append("No classified features written.")

    if raster_summary is not None:
        lines.extend(
            [
                "",
                "5. RASTER OUTPUT",
                "-" * 70,
                f"Raster path: {output_raster_path}",
                f"Dimensions: {raster_summary['width']} x {raster_summary['height']} pixels",
                f"Pixel size: {raster_summary['pixel_size']} map units",
            ]
        )
        for woodland_type in sorted(raster_summary["value_counts"]):
            count = int(raster_summary["value_counts"][woodland_type])
            if woodland_type == 255:
                lines.append(f"Value 255 (nodata): {count:,} pixels")
            else:
                type_name = class_name(int(woodland_type), binary=args.binary, legacy_classes=args.legacy_classes)
                lines.append(f"Value {int(woodland_type)} ({type_name}): {count:,} pixels")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def check_for_species(gdf, species_list):
    """
    Check if any of the target species are mentioned in the primary species column.
    Returns a dictionary with 'found' boolean and 'columns' list.
    """
    species_columns = []
    species_found = False
    
    # Look specifically for primary species column
    primary_species_cols = PRIMARY_SPECIES_COLS
    
    for col in primary_species_cols:
        if col in gdf.columns:
            string_values = gdf[col]
            code_col = _resolve_code_column(col, gdf.columns)
            if _contains_any_species(string_values, WET_SPECIES).any() or _series_has_explicit_downy_birch(
                string_values,
                gdf[code_col] if code_col else None,
            ).any():
                species_columns.append(col)
                species_found = True
                break  # Found at least one relevant species in this column
    
    # If no primary species column found, fall back to checking all string columns
    if not species_found:
        print("⚠️  No primary species column found, checking all string columns...")
        for col in gdf.columns:
            if gdf[col].dtype == 'object':  # String columns
                # Convert to string and check for any species (case insensitive)
                string_values = gdf[col]
                if _contains_any_species(string_values, WET_SPECIES).any() or string_values.astype(str).str.lower().str.contains(
                    rf"\b{re.escape(DOWNY_BIRCH_LABEL)}\b", na=False, regex=True
                ).any():
                    species_columns.append(col)
                    species_found = True
                    break  # Found at least one species in this column
    
    return {
        'found': species_found,
        'columns': list(set(species_columns))  # Remove duplicates
    }

def classify_wet_woodland_not_on_peat(
    row,
    primary_col,
    existing_secondary_cols,
    existing_tertiary_cols,
    primary_code_col=None,
    existing_secondary_code_cols=None,
    existing_tertiary_code_cols=None,
    exclude_birch=False,
):
    """Apply rules for wet woodland NOT on peat.

    Primary must be a wet-indicator species (alder, willow, poplar) or
    explicit downy birch (PBI). Alder must also be present somewhere in the
    recorded species to strengthen the wet-signal. With --exclude-birch, any
    compartment containing birch is excluded from wet-positive label
    assignment.
    """
    primary_val = _normalise_species_value(row[primary_col])

    primary_code_val = row[primary_code_col] if primary_code_col and primary_code_col in row.index else None
    if not (_species_hits(primary_val, WET_SPECIES) or _cell_has_downy_birch(primary_val, primary_code_val)):
        return False

    species_present = set()
    species_present.update(_species_hits(primary_val, WET_SPECIES))
    species_present.update(_row_species_hits(row, existing_secondary_cols, WET_SPECIES))
    species_present.update(_row_species_hits(row, existing_tertiary_cols, WET_SPECIES))

    if 'alder' not in species_present:
        return False

    if exclude_birch and _row_has_any_birch(
        row,
        [primary_col] + existing_secondary_cols + existing_tertiary_cols,
        [primary_code_col] + list(existing_secondary_code_cols or []) + list(existing_tertiary_code_cols or []),
    ):
        return False

    return True

def classify_wet_woodland_on_peat(
    row,
    primary_col,
    existing_secondary_cols,
    existing_tertiary_cols,
    primary_code_col=None,
    existing_secondary_code_cols=None,
    existing_tertiary_code_cols=None,
    exclude_birch=False,
):
    """Apply rules for wet woodland ON peat.

    Primary must be a wet-indicator species (alder, willow, poplar) or
    explicit downy birch (PBI). Secondary and tertiary values must be absent or
    restricted to wet-woodland taxa (alder, willow, poplar) plus explicit
    downy birch. Generic and silver birch do not support wet-positive label
    assignment. With --exclude-birch, any compartment containing birch is
    excluded from wet-positive label assignment.
    """
    primary_val = _normalise_species_value(row[primary_col])
    primary_code_val = row[primary_code_col] if primary_code_col and primary_code_col in row.index else None
    if not (_species_hits(primary_val, WET_SPECIES) or _cell_has_downy_birch(primary_val, primary_code_val)):
        return False

    associated_cells = []
    secondary_code_cols = list(existing_secondary_code_cols or [None] * len(existing_secondary_cols))
    tertiary_code_cols = list(existing_tertiary_code_cols or [None] * len(existing_tertiary_cols))

    for col, code_col in zip(existing_secondary_cols, secondary_code_cols):
        if col in row.index:
            col_val = _normalise_species_value(row[col])
            if col_val not in EMPTY_SPECIES_VALUES:
                code_val = row[code_col] if code_col and code_col in row.index else None
                if not _contains_only_allowed_species(col_val, WET_SPECIES, code=code_val):
                    return False
                associated_cells.append((col_val, code_val))

    for col, code_col in zip(existing_tertiary_cols, tertiary_code_cols):
        if col in row.index:
            col_val = _normalise_species_value(row[col])
            if col_val not in EMPTY_SPECIES_VALUES:
                code_val = row[code_col] if code_col and code_col in row.index else None
                if not _contains_only_allowed_species(col_val, WET_SPECIES, code=code_val):
                    return False

                associated_cells.append((col_val, code_val))

    if exclude_birch:
        if _cell_has_any_birch(primary_val, primary_code_val) or any(_cell_has_any_birch(v, code) for v, code in associated_cells):
            return False

    return True

def filter_by_species(gdf, species_list, species_columns):
    """
    Filter the GeoDataFrame to only include rows where ALL species columns contain target species or are empty/NA.
    Returns filtered dataframe and found species info.
    """
    # Use only the first (primary) species column
    primary_col = species_columns[0]
    print(f"🔍 Filtering by primary species column: {primary_col}")
    
    # Create a mask for rows containing any of the target species in primary column
    primary_mask = gdf[primary_col].astype(str).str.lower().str.contains('|'.join(species_list), na=False)
    
    # Also check secondary and tertiary species columns if they exist
    secondary_cols = ['SECSPECIES', 'Secondary_Species', 'SEC_SPECIES']
    tertiary_cols = ['TERSPECIES', 'Tertiary_Species', 'TER_SPECIES']
    
    # Find which secondary and tertiary columns exist in the data
    existing_secondary_cols = [col for col in secondary_cols if col in gdf.columns]
    existing_tertiary_cols = [col for col in tertiary_cols if col in gdf.columns]
    
    print(f"🔍 Checking secondary species columns: {existing_secondary_cols}")
    print(f"🔍 Checking tertiary species columns: {existing_tertiary_cols}")
    
    # Create masks for secondary and tertiary columns
    secondary_mask = pd.Series([True] * len(gdf), index=gdf.index)  # Default to True
    tertiary_mask = pd.Series([True] * len(gdf), index=gdf.index)   # Default to True
    
    # Check secondary columns - must contain target species OR be empty/NA
    for col in existing_secondary_cols:
        # Check if column contains target species OR is empty/NA
        col_mask = (
            gdf[col].astype(str).str.lower().str.contains('|'.join(species_list), na=False) |  # Contains target species
            gdf[col].isna() |  # Is NA
            (gdf[col].astype(str).str.strip() == '') |  # Is empty string
            (gdf[col].astype(str).str.lower() == 'nan') |  # Is 'nan' string
            (gdf[col].astype(str).str.lower() == 'none') |  # Is 'none' string
            (gdf[col].astype(str).str.lower() == 'n/a')  # Is 'n/a' string
        )
        secondary_mask = secondary_mask & col_mask
    
    # Check tertiary columns - must contain target species OR be empty/NA
    for col in existing_tertiary_cols:
        # Check if column contains target species OR is empty/NA
        col_mask = (
            gdf[col].astype(str).str.lower().str.contains('|'.join(species_list), na=False) |  # Contains target species
            gdf[col].isna() |  # Is NA
            (gdf[col].astype(str).str.strip() == '') |  # Is empty string
            (gdf[col].astype(str).str.lower() == 'nan') |  # Is 'nan' string
            (gdf[col].astype(str).str.lower() == 'none') |  # Is 'none' string
            (gdf[col].astype(str).str.lower() == 'n/a')  # Is 'n/a' string
        )
        tertiary_mask = tertiary_mask & col_mask
    
    # Combine all masks
    final_mask = primary_mask & secondary_mask & tertiary_mask
    
    filtered_gdf = gdf[final_mask].copy()
    
    # Check which species were actually found
    found_species = []
    for species in species_list:
        if filtered_gdf[primary_col].astype(str).str.lower().str.contains(species, na=False).any():
            found_species.append(species)
    
    print(f"📊 Filtering results:")
    print(f"  • Primary species filter: {primary_mask.sum():,} features")
    print(f"  • Secondary species filter: {secondary_mask.sum():,} features")
    print(f"  • Tertiary species filter: {tertiary_mask.sum():,} features")
    print(f"  • Combined filter: {final_mask.sum():,} features")
    
    return filtered_gdf, found_species


def _default_label_paths():
    data_root = Path(__file__).resolve().parents[2] / "data"
    forestry_root = data_root / "input" / "forestry"
    preferred_subcompartments = forestry_root / "forestry_england_subcompartments.shp"
    subcompartment_matches = sorted(forestry_root.rglob("forestry_england_subcompartments.shp"))
    default_subcompartments = preferred_subcompartments if preferred_subcompartments.exists() else (
        subcompartment_matches[0] if subcompartment_matches else preferred_subcompartments
    )
    return {
        "shapefile": default_subcompartments,
        "peat_shapefile": data_root / "input" / "peat" / "peat_extent.shp",
        "output": data_root / "output" / "labels" / "wet_woodland_labelled.shp",
        "output_raster": data_root / "output" / "labels" / "wetwoodland.tif",
        "reports_root": data_root / "output" / "reports",
    }


def _paths_match(path_str, default_path):
    return os.path.normpath(str(Path(path_str).expanduser().resolve(strict=False))) == os.path.normpath(
        str(default_path.resolve(strict=False))
    )


def _resolve_source(value, label, default_path=None, required=False):
    candidate = Path(value).expanduser() if value else default_path
    if candidate is None:
        if required:
            raise SystemExit(f"❌ Error: {label} is required")
        return None
    if candidate.exists():
        return candidate
    if not required and default_path is not None and _paths_match(str(candidate), default_path):
        print(f"ℹ️  {label}: not used (default missing at {candidate})")
        return None
    raise SystemExit(f"❌ Error: {label} not found at {candidate}")


def main():
    """Main function with clean argument parsing."""
    defaults = _default_label_paths()

    parser = argparse.ArgumentParser(description="Vector-based wet woodland species filtering and peat intersection")
    core = parser.add_argument_group("Core inputs")
    core.add_argument(
        "--shapefile",
        default=str(defaults["shapefile"]),
        help=f"Forestry England subcompartments shapefile (default: {defaults['shapefile']})",
    )
    core.add_argument(
        "--peat-shapefile",
        default=str(defaults["peat_shapefile"]),
        help=f"Optional peat extent shapefile (default: {defaults['peat_shapefile']}; skipped if missing)",
    )

    outputs = parser.add_argument_group("Outputs")
    outputs.add_argument(
        "--output",
        default=str(defaults["output"]),
        help=f"Output shapefile path (default: {defaults['output']})",
    )
    outputs.add_argument(
        "--output-raster",
        default=str(defaults["output_raster"]),
        help=(
            "Output raster path. Default classes: 0=non-wet evergreen, 1=non-wet deciduous, "
            "2=wet on peat, 3=wet not on peat (or 0/1 with --binary). "
            f"Default: {defaults['output_raster']}"
        ),
    )
    outputs.add_argument("--report-file", help="Optional text report path. Default: data/output/reports/<output>.report.txt")
    outputs.add_argument("--pixel-size", type=float, default=10.0, help="Pixel size for raster output in meters (default: 10.0)")

    classification = parser.add_argument_group("Classification")
    classification.add_argument("--binary", action="store_true", help="Binary classification: combine all wet woodland types into class 1")
    classification.add_argument(
        "--exclude-birch",
        action="store_true",
        help="Exclude any compartment containing birch from wet woodland labels (ablation mode).",
    )
    classification.add_argument(
        "--legacy-classes",
        action="store_true",
        help="Use legacy non-binary classes (0=not wet, 1=wet on peat, 2=wet not on peat). Default uses split background classes.",
    )
    classification.add_argument("--consolidate", action="store_true", help="Consolidate touching polygons with same attributes")
    classification.add_argument("--consolidate-by", help="Attribute column to use for consolidation (default: all attributes)")
    classification.add_argument("--verbose", action="store_true", help="Show detailed processing output.")
    args = parser.parse_args()
    split_background_classes = (not args.binary) and (not args.legacy_classes)
    effective_target_species = [
        species for species in REPORT_TARGET_SPECIES if not (args.exclude_birch and 'birch' in species.lower())
    ]

    def vprint(*values, **kwargs):
        if args.verbose:
            print(*values, **kwargs)

    if args.binary:
        wet_on_peat_class = 1
        wet_not_on_peat_class = 1
        schema_desc = "binary (0=not wet woodland, 1=wet woodland)"
    elif args.legacy_classes:
        wet_on_peat_class = 1
        wet_not_on_peat_class = 2
        schema_desc = "legacy (0=not wet, 1=wet on peat, 2=wet not on peat)"
    else:
        wet_on_peat_class = 2
        wet_not_on_peat_class = 3
        schema_desc = "split background (0=non-wet evergreen, 1=non-wet deciduous, 2=wet on peat, 3=wet not on peat)"
    if args.exclude_birch:
        birch_desc = "excluded from all wet woodland labels"
    else:
        birch_desc = "PBI downy birch only; may be primary on peat, and off peat only where alder is also present"

    shapefile_path = _resolve_source(args.shapefile, "Subcompartments shapefile", defaults["shapefile"], required=True)
    peat_extent_path = _resolve_source(args.peat_shapefile, "Peat extent shapefile", defaults["peat_shapefile"], required=False)
    use_peat = peat_extent_path is not None

    output_path = Path(args.output) if args.output else defaults["output"]
    if output_path.suffix.lower() != ".shp":
        output_path = output_path.with_suffix(".shp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports_root = defaults["reports_root"]
    report_path = Path(args.report_file) if args.report_file else reports_root / f"{output_path.stem}.report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_raster_path = Path(args.output_raster) if args.output_raster else None
    if output_raster_path is not None:
        output_raster_path.parent.mkdir(parents=True, exist_ok=True)

    print("Wet woodland labels")
    print("=" * 60)
    print(f"Schema:           {schema_desc}")
    print(f"Birch rule:       {birch_desc}")
    print(f"Subcompartments:  {shapefile_path}")
    print(f"Peat extent:      {peat_extent_path if use_peat else 'Skipped'}")
    print(f"Output shapefile: {output_path}")
    if output_raster_path is not None:
        print(f"Output raster:    {output_raster_path} ({args.pixel_size}m pixels)")
    print(f"Report file:      {report_path}")
    
    # --- Load Vector Data ---
    vprint("Loading subcompartments data...")
    gdf = gpd.read_file(shapefile_path)
    vprint(f"Subcompartments CRS: {gdf.crs}")
    if gdf.crs and '27700' in str(gdf.crs).upper():
        vprint("British National Grid detected for subcompartments")
    elif gdf.crs and gdf.crs.is_geographic:
        print("Warning: subcompartments use a geographic CRS; pixel size will be interpreted in degrees, not meters.")

    # Load peat data only if available
    if use_peat:
        vprint("Loading peat extent data...")
        peat_gdf = gpd.read_file(peat_extent_path)
        vprint(f"Peat CRS: {peat_gdf.crs}")
        if peat_gdf.crs and '27700' in str(peat_gdf.crs).upper():
            vprint("British National Grid detected for peat extent")
        elif peat_gdf.crs and peat_gdf.crs.is_geographic:
            print("Warning: peat extent uses a geographic CRS.")
    else:
        peat_gdf = None
    
    primary_col = next((col for col in PRIMARY_SPECIES_COLS if col in gdf.columns), None)
    primary_code_col = _resolve_code_column(primary_col, gdf.columns) if primary_col else None
    existing_secondary_cols = [col for col in SECONDARY_SPECIES_COLS if col in gdf.columns]
    existing_tertiary_cols = [col for col in TERTIARY_SPECIES_COLS if col in gdf.columns]
    existing_secondary_code_cols = [_resolve_code_column(col, gdf.columns) for col in existing_secondary_cols]
    existing_tertiary_code_cols = [_resolve_code_column(col, gdf.columns) for col in existing_tertiary_cols]

    species_info = {
        'found': False,
        'columns': [col for col in [primary_col] + existing_secondary_cols + existing_tertiary_cols if col],
    }
    original_count = len(gdf)

    if primary_col is not None:
        vprint(
            f"Species columns - primary: {primary_col}, secondary: {existing_secondary_cols}, tertiary: {existing_tertiary_cols}"
        )

        target_mask = pd.Series(False, index=gdf.index)
        species_cols = [primary_col] + existing_secondary_cols + existing_tertiary_cols
        code_cols = [primary_code_col] + existing_secondary_code_cols + existing_tertiary_code_cols

        for species_col, code_col in zip(species_cols, code_cols):
            species_series = gdf[species_col]
            code_series = gdf[code_col] if code_col else None
            target_mask |= _contains_any_species(species_series, WET_SPECIES)
            if not args.exclude_birch:
                target_mask |= _series_has_explicit_downy_birch(species_series, code_series)

        species_info['found'] = bool(target_mask.any())
        target_gdf = gdf[target_mask].copy()
        non_target_gdf = gdf[~target_mask].copy()
    else:
        print("No target species found in the input shapefile; all compartments will be treated as non-target background.")
        target_gdf = gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)  # Empty
        non_target_gdf = gdf.copy()  # All compartments
        primary_col = gdf.columns[0] if len(gdf.columns) > 0 else 'dummy'
        primary_code_col = None
        existing_secondary_cols = []
        existing_tertiary_cols = []
        existing_secondary_code_cols = []
        existing_tertiary_code_cols = []

    print(
        f"Input summary:    {len(gdf):,} compartments | "
        f"{len(target_gdf):,} target candidates | "
        f"{len(non_target_gdf):,} non-target"
    )
    
    # 1. Fix invalid geometries in all datasets
    vprint("Fixing invalid geometries...")
    if len(target_gdf) > 0:
        target_gdf["geometry"] = target_gdf["geometry"].apply(lambda geom: make_valid(geom))
        target_gdf = target_gdf[~target_gdf["geometry"].is_empty & target_gdf["geometry"].notna()]

    if len(non_target_gdf) > 0:
        non_target_gdf["geometry"] = non_target_gdf["geometry"].apply(lambda geom: make_valid(geom))
        non_target_gdf = non_target_gdf[~non_target_gdf["geometry"].is_empty & non_target_gdf["geometry"].notna()]

    if use_peat and peat_gdf is not None:
        peat_gdf["geometry"] = peat_gdf["geometry"].apply(lambda geom: make_valid(geom))
        peat_gdf = peat_gdf[~peat_gdf["geometry"].is_empty & peat_gdf["geometry"].notna()]
    
    # 2. Ensure all datasets have the same CRS
    vprint("Checking coordinate reference systems...")
    base_crs = target_gdf.crs if len(target_gdf) > 0 else non_target_gdf.crs

    if use_peat and peat_gdf is not None:
        if len(target_gdf) > 0 and target_gdf.crs != peat_gdf.crs:
            vprint(f"Reprojecting peat extent to match subcompartments CRS: {base_crs}")
            peat_gdf = peat_gdf.to_crs(base_crs)

        if len(non_target_gdf) > 0 and non_target_gdf.crs != peat_gdf.crs:
            peat_gdf = peat_gdf.to_crs(non_target_gdf.crs)

    # 3. Explode overlapping polygons into non-overlapping parts
    vprint("Exploding overlapping polygons...")
    if len(target_gdf) > 0:
        target_gdf = target_gdf.explode(index_parts=False).reset_index(drop=True)
    if len(non_target_gdf) > 0:
        non_target_gdf = non_target_gdf.explode(index_parts=False).reset_index(drop=True)
    if use_peat and peat_gdf is not None:
        peat_gdf = peat_gdf.explode(index_parts=False).reset_index(drop=True)

    # 4. Process each group efficiently
    vprint("Processing compartments...")

    # Create a spatial index for better performance (only if using peat)
    if use_peat and peat_gdf is not None:
        from rtree import index
        idx = index.Index()
        for i, geom in enumerate(peat_gdf.geometry):
            idx.insert(i, geom.bounds)
    else:
        idx = None

    def assign_non_wet_class(row):
        if args.binary or args.legacy_classes:
            return 0
        return classify_nonwet_functional_type(row, primary_col, existing_secondary_cols, existing_tertiary_cols)

    all_results = []

    # Process compartments with target species
    if len(target_gdf) > 0:
        if use_peat:
            vprint(f"Classifying {len(target_gdf):,} target compartments with peat support...")

            # Vectorized peat intersection check
            vprint("Finding peat intersections...")
            intersects_peat = gpd.sjoin(target_gdf, peat_gdf, how='left', predicate='intersects')
            peat_mask = ~intersects_peat.index_right.isna()

            # Split into peat and non-peat groups
            target_on_peat = target_gdf[target_gdf.index.isin(intersects_peat[peat_mask].index)]
            target_not_on_peat = target_gdf[~target_gdf.index.isin(intersects_peat[peat_mask].index)]

            print(
                f"Peat split:       {len(target_on_peat):,} on peat | "
                f"{len(target_not_on_peat):,} off peat"
            )

            # Process ON PEAT compartments
            for i, (idx_val, row) in enumerate(target_on_peat.iterrows()):
                new_row = row.copy()
                if classify_wet_woodland_on_peat(
                    row,
                    primary_col,
                    existing_secondary_cols,
                    existing_tertiary_cols,
                    primary_code_col=primary_code_col,
                    existing_secondary_code_cols=existing_secondary_code_cols,
                    existing_tertiary_code_cols=existing_tertiary_code_cols,
                    exclude_birch=args.exclude_birch,
                ):
                    new_row['woodland_type'] = wet_on_peat_class
                else:
                    new_row['woodland_type'] = assign_non_wet_class(row)
                all_results.append(new_row)

            # Process NOT ON PEAT compartments
            for i, (idx_val, row) in enumerate(target_not_on_peat.iterrows()):
                new_row = row.copy()
                if classify_wet_woodland_not_on_peat(
                    row,
                    primary_col,
                    existing_secondary_cols,
                    existing_tertiary_cols,
                    primary_code_col=primary_code_col,
                    existing_secondary_code_cols=existing_secondary_code_cols,
                    existing_tertiary_code_cols=existing_tertiary_code_cols,
                    exclude_birch=args.exclude_birch,
                ):
                    new_row['woodland_type'] = wet_not_on_peat_class
                else:
                    new_row['woodland_type'] = assign_non_wet_class(row)
                all_results.append(new_row)
        else:
            print(f"Peat split:       skipped ({len(target_gdf):,} target compartments classified without peat)")

            # Without peat data, classify all as "wet woodland" if they pass species criteria
            for i, (idx_val, row) in enumerate(target_gdf.iterrows()):
                new_row = row.copy()
                if classify_wet_woodland_not_on_peat(
                    row,
                    primary_col,
                    existing_secondary_cols,
                    existing_tertiary_cols,
                    primary_code_col=primary_code_col,
                    existing_secondary_code_cols=existing_secondary_code_cols,
                    existing_tertiary_code_cols=existing_tertiary_code_cols,
                    exclude_birch=args.exclude_birch,
                ):
                    new_row['woodland_type'] = wet_not_on_peat_class
                else:
                    new_row['woodland_type'] = assign_non_wet_class(row)
                all_results.append(new_row)

    # Process compartments without target species (split by functional type if enabled)
    if len(non_target_gdf) > 0:
        vprint(f"Classifying {len(non_target_gdf):,} non-target compartments...")
        for i, (idx_val, row) in enumerate(non_target_gdf.iterrows()):
            new_row = row.copy()
            new_row['woodland_type'] = assign_non_wet_class(row)
            all_results.append(new_row)

    if all_results:
        vprint(f"Processing {len(all_results)} total classified features...")
        result_crs = target_gdf.crs if len(target_gdf) > 0 else non_target_gdf.crs
        result_gdf = gpd.GeoDataFrame(all_results, crs=result_crs)

        # Filter out non-polygon geometries
        vprint("Filtering out non-polygon geometries...")
        polygon_mask = result_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        result_gdf = result_gdf[polygon_mask].copy()

        if len(result_gdf) > 0:
            # Optional consolidation
            if args.consolidate:
                vprint("Consolidating touching polygons...")
                if args.consolidate_by:
                    consolidate_cols = [args.consolidate_by, 'woodland_type']
                    result_gdf = result_gdf.dissolve(by=consolidate_cols, as_index=False)
                else:
                    result_gdf = result_gdf.dissolve(by='woodland_type', as_index=False)

            counts = result_gdf['woodland_type'].value_counts().sort_index()

            # Keep only geometry and woodland_type columns
            output_gdf = result_gdf[['woodland_type', 'geometry']].copy()

            # Dissolve geometries by woodland_type to create simple polygon masks
            vprint("Dissolving geometries by woodland type...")
            dissolved_gdf = output_gdf.dissolve(by='woodland_type', as_index=False)
            dissolved_write_gdf = dissolved_gdf.rename(
                columns={'woodland_type': 'ww_type'}
            ) if output_path.suffix.lower() == ".shp" else dissolved_gdf
            dissolved_write_gdf.to_file(output_path)

            # Optional raster output
            if output_raster_path is not None:
                vprint(f"Creating raster output: {output_raster_path}")

                # Get bounds of all data
                bounds = result_gdf.total_bounds
                vprint(f"Raster bounds: {bounds}")

                # Check if CRS is geographic (degrees) vs projected (meters)
                if result_gdf.crs.is_geographic:
                    print("Warning: raster output uses a geographic CRS; pixel size will be interpreted in degrees, not meters.")

                    # Estimate actual pixel size in meters at center of bounds
                    center_lat = (bounds[1] + bounds[3]) / 2
                    meters_per_degree_lat = 111320 * np.cos(np.radians(center_lat))
                    meters_per_degree_lon = 111320
                    actual_pixel_size_m = args.pixel_size * np.sqrt(meters_per_degree_lat * meters_per_degree_lon)

                    print(
                        f"Approximate raster pixel size at mid-latitude: "
                        f"{actual_pixel_size_m:.0f}m x {actual_pixel_size_m:.0f}m"
                    )
                else:
                    vprint(f"Raster CRS: {result_gdf.crs}")

                width = int((bounds[2] - bounds[0]) / args.pixel_size)
                height = int((bounds[3] - bounds[1]) / args.pixel_size)
                vprint(f"Raster dimensions: {width} x {height}")

                # Create transform
                transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

                # Create raster array filled with 255 (NoData)
                raster_array = np.full((height, width), 255, dtype=np.uint8)

                # Rasterize each class present in output
                for woodland_type in sorted(result_gdf['woodland_type'].unique()):
                    type_gdf = result_gdf[result_gdf['woodland_type'] == woodland_type]
                    if len(type_gdf) > 0:
                        # Create list of (geometry, value) tuples
                        shapes = [(geom, woodland_type) for geom in type_gdf.geometry]

                        # Rasterize this woodland type
                        type_raster = rasterize(
                            shapes=shapes,
                            out_shape=(height, width),
                            transform=transform,
                            fill=255,  # NoData value
                            dtype=np.uint8
                        )

                        # Update main raster where this type occurs
                        mask = type_raster != 255
                        raster_array[mask] = woodland_type

                # Write raster
                with rasterio.open(
                    output_raster_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=np.uint8,
                    crs=result_gdf.crs,
                    transform=transform,
                    nodata=255,
                    compress='lzw'
                ) as dst:
                    dst.write(raster_array, 1)

            # Summary by type
            wet_total = counts.get(wet_on_peat_class, 0) + counts.get(wet_not_on_peat_class, 0)
            raster_summary = None
            if output_raster_path is not None:
                raster_values, raster_counts = np.unique(raster_array, return_counts=True)
                raster_summary = {
                    "width": width,
                    "height": height,
                    "pixel_size": args.pixel_size,
                    "value_counts": {int(v): int(c) for v, c in zip(raster_values, raster_counts)},
                }
            write_labels_report(
                report_path,
                args=args,
                shapefile_path=shapefile_path,
                peat_extent_path=peat_extent_path,
                use_peat=use_peat,
                output_path=output_path,
                output_raster_path=output_raster_path,
                species_info=species_info,
                effective_target_species=effective_target_species,
                original_count=len(gdf),
                target_count=len(target_gdf),
                non_target_count=len(non_target_gdf),
                result_count=len(result_gdf),
                dissolved_count=len(dissolved_gdf),
                class_counts={int(k): int(v) for k, v in counts.items()},
                raster_summary=raster_summary,
            )
            print("Outputs")
            print("=" * 60)
            print(f"Classified features: {len(result_gdf):,}")
            print(f"Wet woodland:      {wet_total:,}")
            if args.binary or args.legacy_classes:
                print(f"Non-wet:           {counts.get(0, 0):,}")
            else:
                print(f"Non-wet evergreen: {counts.get(0, 0):,}")
                print(f"Non-wet deciduous: {counts.get(1, 0):,}")
            for woodland_type, count in counts.items():
                type_name = class_name(int(woodland_type), binary=args.binary, legacy_classes=args.legacy_classes)
                print(f"Class {int(woodland_type)} ({type_name}): {count:,}")
            print(f"Saved shapefile:   {output_path}")
            if output_raster_path is not None:
                print(f"Saved raster:      {output_raster_path}")
            print(f"Saved report:      {report_path}")

        else:
            print("No valid polygon features found after filtering.")
            write_labels_report(
                report_path,
                args=args,
                shapefile_path=shapefile_path,
                peat_extent_path=peat_extent_path,
                use_peat=use_peat,
                output_path=output_path,
                output_raster_path=output_raster_path,
                species_info=species_info,
                effective_target_species=effective_target_species,
                original_count=len(gdf),
                target_count=len(target_gdf),
                non_target_count=len(non_target_gdf),
                result_count=0,
                dissolved_count=0,
                class_counts={},
                raster_summary=None,
            )
            print(f"Saved report:      {report_path}")
    else:
        print("No wet woodland features to process.")
        write_labels_report(
            report_path,
            args=args,
            shapefile_path=shapefile_path,
            peat_extent_path=peat_extent_path,
            use_peat=use_peat,
            output_path=output_path,
            output_raster_path=output_raster_path,
            species_info=species_info,
            effective_target_species=effective_target_species,
            original_count=len(gdf),
            target_count=len(target_gdf),
            non_target_count=len(non_target_gdf),
            result_count=0,
            dissolved_count=0,
            class_counts={},
            raster_summary=None,
        )
        print(f"Saved report:      {report_path}")

if __name__ == "__main__":
    main()
