#!/usr/bin/env python3
"""
Compartments Processor for Functional Type Classification

DESCRIPTION: Processes Forestry England compartment data to classify by functional type (deciduous vs evergreen).
Checks primary, secondary, and tertiary species columns, handles various data formats, and creates labeled
shapefiles for further processing. Optionally filters to peat-only areas if peat extent data is provided.
Used in the data pipeline to identify deciduous vs evergreen woodland areas.
"""

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.validation import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
import gc
import os
import pandas as pd
import sys
import argparse
import numpy as np
from pathlib import Path

# Define species by functional type
DECIDUOUS_SPECIES = [
    'oak', 'beech', 'ash', 'birch', 'alder', 'willow', 'poplar',
    'sycamore', 'maple', 'cherry', 'hornbeam', 'lime', 'hazel',
    'sweet chestnut', 'walnut', 'rowan', 'aspen', 'elm'
]

EVERGREEN_SPECIES = [
    'pine', 'spruce', 'fir', 'larch', 'cedar', 'douglas',
    'hemlock', 'redwood', 'yew', 'juniper', 'cypress',
    'western red cedar', 'lawson', 'thuja', 'sequoia',
    'sitka', 'norway spruce', 'scots pine', 'corsican pine'
]

def check_for_species(gdf, species_list):
    """
    Check if any of the target species are mentioned in the primary species column.
    Returns a dictionary with 'found' boolean and 'columns' list.
    """
    species_columns = []
    species_found = False

    # Look specifically for primary species column
    primary_species_cols = ['PRISPECIES', 'Primary_Species', 'Species', 'SPECIES']

    for col in primary_species_cols:
        if col in gdf.columns:
            # Convert to string and check for any species (case insensitive)
            string_values = gdf[col].astype(str).str.lower()
            for species in species_list:
                if string_values.str.contains(species, na=False).any():
                    species_columns.append(col)
                    species_found = True
                    break  # Found at least one species in this column

    # If no primary species column found, fall back to checking all string columns
    if not species_found:
        print("⚠️  No primary species column found, checking all string columns...")
        for col in gdf.columns:
            if gdf[col].dtype == 'object':  # String columns
                # Convert to string and check for any species (case insensitive)
                string_values = gdf[col].astype(str).str.lower()
                for species in species_list:
                    if string_values.str.contains(species, na=False).any():
                        species_columns.append(col)
                        species_found = True
                        break  # Found at least one species in this column

    return {
        'found': species_found,
        'columns': list(set(species_columns))  # Remove duplicates
    }

def classify_functional_type(row, primary_col, existing_secondary_cols, existing_tertiary_cols):
    """
    Classify compartment by functional type based on dominant species.
    Returns: 1 for deciduous, 2 for evergreen, 0 for mixed/other
    """
    def get_species_in_text(text):
        """Extract deciduous and evergreen species from text."""
        text_lower = str(text).lower()
        deciduous_found = [sp for sp in DECIDUOUS_SPECIES if sp in text_lower]
        evergreen_found = [sp for sp in EVERGREEN_SPECIES if sp in text_lower]
        return deciduous_found, evergreen_found

    # Check primary species
    primary_val = str(row[primary_col]).lower()
    primary_deciduous, primary_evergreen = get_species_in_text(primary_val)

    # Check secondary and tertiary species
    all_deciduous = set(primary_deciduous)
    all_evergreen = set(primary_evergreen)

    for col in existing_secondary_cols:
        if pd.notna(row[col]):
            dec, evg = get_species_in_text(row[col])
            all_deciduous.update(dec)
            all_evergreen.update(evg)

    for col in existing_tertiary_cols:
        if pd.notna(row[col]):
            dec, evg = get_species_in_text(row[col])
            all_deciduous.update(dec)
            all_evergreen.update(evg)

    # Classification logic:
    # - If primary species is deciduous and no evergreens anywhere: deciduous (1)
    # - If primary species is evergreen and no deciduous anywhere: evergreen (2)
    # - Otherwise: mixed/other (0)

    has_deciduous = len(all_deciduous) > 0
    has_evergreen = len(all_evergreen) > 0
    primary_is_deciduous = len(primary_deciduous) > 0
    primary_is_evergreen = len(primary_evergreen) > 0

    # Pure deciduous: primary is deciduous AND no evergreens present
    if primary_is_deciduous and not has_evergreen:
        return 1  # Deciduous

    # Pure evergreen: primary is evergreen AND no deciduous present
    elif primary_is_evergreen and not has_deciduous:
        return 2  # Evergreen

    # Mixed or other
    else:
        return 0  # Mixed/Other

def main():
    """Main function with clean argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Classify compartments by functional type (deciduous vs evergreen)")
    parser.add_argument("--data-dir", help="Directory containing forestry_england_subcompartments.shp")
    parser.add_argument("--shapefile", help="Direct path to compartments shapefile (alternative to --data-dir)")
    parser.add_argument(
        "--output",
        help="Output file path (default: tow/data/output/labels/functional_types_labelled.shp)",
    )
    parser.add_argument("--peat-extent", help="Optional: Shapefile or raster of peat extent to filter compartments to peat-only areas")
    parser.add_argument("--consolidate", action="store_true", help="Consolidate touching polygons with same functional type")
    parser.add_argument("--output-raster", help="Output path for raster TIFF with cell values 0, 1, 2")
    parser.add_argument("--pixel-size", type=float, default=10.0, help="Pixel size for raster output in map units (default: 10.0)")
    args = parser.parse_args()

    # Construct file paths
    if args.shapefile:
        shapefile_path = args.shapefile
        data_dir = str(Path(args.shapefile).parent)
    elif args.data_dir:
        data_dir = args.data_dir.rstrip('/')  # Remove trailing slash if present
        shapefile_path = os.path.join(data_dir, "forestry_england_subcompartments.shp")
    else:
        print("❌ Error: Must provide either --data-dir or --shapefile")
        sys.exit(1)

    # Peat extent path (if provided)
    peat_extent_path = None
    if args.peat_extent:
        peat_extent_path = args.peat_extent
        if not os.path.exists(peat_extent_path):
            print(f"❌ Error: Peat extent file not found at {peat_extent_path}")
            sys.exit(1)

    # Set output path under canonical data/output tree.
    data_root = Path(__file__).resolve().parents[2] / "data"
    default_labels_dir = data_root / "output" / "labels"
    default_output_path = default_labels_dir / "functional_types_labelled.shp"

    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() != ".shp":
            output_path = output_path.with_suffix(".shp")
    else:
        output_path = default_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_raster_path = Path(args.output_raster) if args.output_raster else None
    if output_raster_path is not None:
        output_raster_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if files exist
    if not os.path.exists(shapefile_path):
        print(f"❌ Error: Shapefile not found at {shapefile_path}")
        sys.exit(1)

    print(f"📁 Using data directory: {data_dir}")
    print(f"🗺️  Subcompartments: {shapefile_path}")
    if peat_extent_path:
        print(f"🌱 Peat extent filter: {peat_extent_path}")
    print(f"💾 Output: {output_path}")

    # --- Load Vector Data ---
    print("Loading subcompartments data...")
    gdf = gpd.read_file(shapefile_path)

    # Load peat extent if provided
    peat_gdf = None
    if peat_extent_path:
        print("Loading peat extent data...")
        if peat_extent_path.endswith('.shp') or peat_extent_path.endswith('.gpkg') or peat_extent_path.endswith('.geojson'):
            # Vector peat extent
            peat_gdf = gpd.read_file(peat_extent_path)
            print(f"  Loaded {len(peat_gdf):,} peat polygons")
        elif peat_extent_path.endswith('.tif') or peat_extent_path.endswith('.tiff'):
            # Raster peat extent - convert to vector
            print("  Converting raster peat extent to vector...")
            from rasterio.features import shapes
            with rasterio.open(peat_extent_path) as src:
                peat_data = src.read(1)
                peat_transform = src.transform
                peat_crs = src.crs

                # Extract shapes where value is not nodata (assuming peat pixels have value 1 or similar)
                peat_shapes = []
                for geom, value in shapes(peat_data, transform=peat_transform):
                    if value != 0 and value != 255:  # Assuming 0 or 255 is nodata
                        peat_shapes.append(geom)

                if peat_shapes:
                    peat_gdf = gpd.GeoDataFrame({'geometry': peat_shapes}, crs=peat_crs)
                    print(f"  Extracted {len(peat_gdf):,} peat polygons from raster")
                else:
                    print("  ⚠️  Warning: No peat areas found in raster")
                    peat_gdf = None
        else:
            print(f"  ⚠️  Warning: Unrecognized peat extent file format, ignoring")
            peat_gdf = None

    # Combine all species for checking
    all_species = DECIDUOUS_SPECIES + EVERGREEN_SPECIES

    # Check for species mentions
    species_info = check_for_species(gdf, all_species)
    if species_info['found']:
        print(f"🌳 TREE SPECIES FOUND in shapefile!")
        print(f"Species columns: {species_info['columns']}")
        print(f"Deciduous species to check: {len(DECIDUOUS_SPECIES)}")
        print(f"Evergreen species to check: {len(EVERGREEN_SPECIES)}")

        # Get column information for classification
        primary_col = species_info['columns'][0]
        secondary_cols = ['SECSPECIES', 'Secondary_Species', 'SEC_SPECIES']
        tertiary_cols = ['TERSPECIES', 'Tertiary_Species', 'TER_SPECIES']
        existing_secondary_cols = [col for col in secondary_cols if col in gdf.columns]
        existing_tertiary_cols = [col for col in tertiary_cols if col in gdf.columns]

        print(f"Using species columns - Primary: {primary_col}, Secondary: {existing_secondary_cols}, Tertiary: {existing_tertiary_cols}")
    else:
        print(f"❌ No tree species found in shapefile")
        print(f"Checked species: {len(all_species)} total")
        sys.exit(1)

    # Fix invalid geometries
    print("Fixing invalid geometries...")
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: make_valid(geom) if geom else None)
    gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()]

    if peat_gdf is not None:
        peat_gdf["geometry"] = peat_gdf["geometry"].apply(lambda geom: make_valid(geom) if geom else None)
        peat_gdf = peat_gdf[~peat_gdf["geometry"].is_empty & peat_gdf["geometry"].notna()]

    # Explode overlapping polygons into non-overlapping parts
    print("Exploding overlapping polygons...")
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    if peat_gdf is not None:
        peat_gdf = peat_gdf.explode(index_parts=False).reset_index(drop=True)

    # Filter to peat areas if peat extent provided
    if peat_gdf is not None:
        print("Filtering compartments to peat-only areas...")
        original_count = len(gdf)

        # Ensure same CRS
        if gdf.crs != peat_gdf.crs:
            print(f"  Reprojecting peat extent from {peat_gdf.crs} to {gdf.crs}")
            peat_gdf = peat_gdf.to_crs(gdf.crs)

        # Spatial join to find compartments that intersect with peat
        intersects_peat = gpd.sjoin(gdf, peat_gdf, how='inner', predicate='intersects')

        # Get unique compartment indices that intersect peat
        peat_compartment_indices = intersects_peat.index.unique()

        # Filter to only compartments on peat
        gdf = gdf.loc[peat_compartment_indices].copy()

        filtered_count = len(gdf)
        print(f"  Filtered from {original_count:,} to {filtered_count:,} compartments ({100*filtered_count/original_count:.1f}% on peat)")

        if filtered_count == 0:
            print("❌ No compartments intersect with peat extent")
            sys.exit(1)

    # Classify each compartment
    print(f"Classifying {len(gdf):,} compartments by functional type...")

    all_results = []
    for idx_val, row in gdf.iterrows():
        new_row = row.copy()
        functional_type = classify_functional_type(row, primary_col, existing_secondary_cols, existing_tertiary_cols)
        new_row['functional_type'] = functional_type
        all_results.append(new_row)

    if all_results:
        print(f"Processing {len(all_results)} total classified features...")
        result_gdf = gpd.GeoDataFrame(all_results, crs=gdf.crs)

        # Filter out non-polygon geometries
        print("Filtering out non-polygon geometries...")
        polygon_mask = result_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        result_gdf = result_gdf[polygon_mask].copy()

        if len(result_gdf) > 0:
            # Optional consolidation
            if args.consolidate:
                print("Consolidating touching polygons...")
                result_gdf = result_gdf.dissolve(by='functional_type', as_index=False)

            # Show classification counts
            print("Classification results:")
            counts = result_gdf['functional_type'].value_counts().sort_index()
            for functional_type, count in counts.items():
                if functional_type == 0:
                    type_name = "mixed/other"
                elif functional_type == 1:
                    type_name = "deciduous"
                elif functional_type == 2:
                    type_name = "evergreen"
                else:
                    type_name = "unknown"
                print(f"  • Type {functional_type} ({type_name}): {count:,} features")

            # Keep only geometry and functional_type columns
            output_gdf = result_gdf[['functional_type', 'geometry']].copy()

            # Dissolve geometries by functional_type to create simple polygon masks
            print("Dissolving geometries by functional type to create simple polygon masks...")
            dissolved_gdf = output_gdf.dissolve(by='functional_type', as_index=False)

            print(f"Simplified from {len(output_gdf):,} features to {len(dissolved_gdf)} polygon masks")
            for functional_type in dissolved_gdf['functional_type'].sort_values():
                if functional_type == 0:
                    type_name = "mixed/other"
                elif functional_type == 1:
                    type_name = "deciduous"
                elif functional_type == 2:
                    type_name = "evergreen"
                print(f"  • Type {functional_type} ({type_name}): 1 simplified polygon mask")

            print(f"Saving {len(dissolved_gdf)} simplified polygon masks to {output_path}")
            dissolved_gdf.to_file(output_path)

            # Optional raster output
            if output_raster_path is not None:
                print(f"Creating raster output: {output_raster_path}")

                # Get bounds of all data
                bounds = result_gdf.total_bounds
                width = int((bounds[2] - bounds[0]) / args.pixel_size)
                height = int((bounds[3] - bounds[1]) / args.pixel_size)

                # Create transform
                transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

                # Create raster array filled with 255 (NoData)
                raster_array = np.full((height, width), 255, dtype=np.uint8)

                # Rasterize each functional type
                for functional_type in [0, 1, 2]:
                    type_gdf = result_gdf[result_gdf['functional_type'] == functional_type]
                    if len(type_gdf) > 0:
                        # Create list of (geometry, value) tuples
                        shapes = [(geom, functional_type) for geom in type_gdf.geometry]

                        # Rasterize this functional type
                        type_raster = rasterize(
                            shapes=shapes,
                            out_shape=(height, width),
                            transform=transform,
                            fill=255,  # NoData value
                            dtype=np.uint8
                        )

                        # Update main raster where this type occurs
                        mask = type_raster != 255
                        raster_array[mask] = functional_type

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

                print(f"✅ Raster saved: {output_raster_path}")
                print(f"   Dimensions: {width} x {height}")
                print(f"   Pixel size: {args.pixel_size} map units")
                print(f"   Cell values: 0=mixed/other, 1=deciduous, 2=evergreen, 255=NoData")

            print(f"✅ Processing complete!")
            print(f"📊 Original subcompartments: {len(gdf)}")
            print(f"📊 Final classified features: {len(result_gdf)}")

            # Summary by type
            deciduous = counts.get(1, 0)
            evergreen = counts.get(2, 0)
            mixed = counts.get(0, 0)
            print(f"📊 Summary: {deciduous:,} deciduous, {evergreen:,} evergreen, {mixed:,} mixed/other")

        else:
            print("❌ No valid polygon features found after filtering")
    else:
        print("❌ No features to process")

if __name__ == "__main__":
    main()
