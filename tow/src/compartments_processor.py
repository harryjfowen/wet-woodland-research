import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
import gc
import os
import pandas as pd
import sys
import argparse

# Define target species for filtering
TARGET_SPECIES = ['alder', 'birch', 'willow']

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

def main():
    """Main function with clean argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector-based wet woodland species filtering and peat intersection")
    parser.add_argument("--data-dir", required=True, help="Directory containing the input data files")
    parser.add_argument("--output", help="Output file path (default: data_dir/wet_woodland_peat_intersection.shp)")
    parser.add_argument("--consolidate", action="store_true", help="Consolidate touching polygons with same attributes")
    parser.add_argument("--consolidate-by", help="Attribute column to use for consolidation (default: all attributes)")
    args = parser.parse_args()
    
    # Construct file paths
    data_dir = args.data_dir.rstrip('/')  # Remove trailing slash if present
    shapefile_path = os.path.join(data_dir, "Forestry_England_Subcompartments.shp")
    peat_extent_path = os.path.join(data_dir, "peaty_soil_extent_v1.shp")
    
    # Set output path
    if args.output:
        output_path = args.output
        if not output_path.endswith('.shp'):
            output_path = output_path + '.shp'
    else:
        output_path = os.path.join(data_dir, "wet_woodland_peat_intersection.shp")
    
    # Check if files exist
    if not os.path.exists(shapefile_path):
        print(f"❌ Error: Shapefile not found at {shapefile_path}")
        sys.exit(1)
    
    if not os.path.exists(peat_extent_path):
        print(f"❌ Error: Peat extent shapefile not found at {peat_extent_path}")
        sys.exit(1)
    
    print(f"📁 Using data directory: {data_dir}")
    print(f"🗺️  Subcompartments: {shapefile_path}")
    print(f"🌱 Peat extent: {peat_extent_path}")
    print(f"💾 Output: {output_path}")
    
    # --- Load Vector Data ---
    print("Loading subcompartments data...")
    gdf = gpd.read_file(shapefile_path)
    
    print("Loading peat extent data...")
    peat_gdf = gpd.read_file(peat_extent_path)
    
    # Check for target species mentions
    species_info = check_for_species(gdf, TARGET_SPECIES)
    if species_info['found']:
        print(f"🌳 TARGET SPECIES FOUND in shapefile!")
        print(f"Species columns: {species_info['columns']}")
        print(f"Target species: {TARGET_SPECIES}")
        
        # Filter the data to only include rows with target species
        gdf, found_species = filter_by_species(gdf, TARGET_SPECIES, species_info['columns'])
        print(f"Filtered to {len(gdf)} features containing target species")
        print(f"✅ Species found: {found_species}")
        print(f"❌ Species not found: {[s for s in TARGET_SPECIES if s not in found_species]}")
    else:
        print(f"❌ No target species found in shapefile")
        print(f"Target species: {TARGET_SPECIES}")
        print("Exiting as no target species found...")
        return
    
    # 1. Fix invalid geometries in both datasets
    print("Fixing invalid geometries...")
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: make_valid(geom))
    peat_gdf["geometry"] = peat_gdf["geometry"].apply(lambda geom: make_valid(geom))
    
    # Drop empty geometries
    gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()]
    peat_gdf = peat_gdf[~peat_gdf["geometry"].is_empty & peat_gdf["geometry"].notna()]
    
    # 2. Ensure both datasets have the same CRS
    print("Checking coordinate reference systems...")
    if gdf.crs != peat_gdf.crs:
        print(f"Reprojecting peat extent to match subcompartments CRS: {gdf.crs}")
        peat_gdf = peat_gdf.to_crs(gdf.crs)
    
    # 3. Explode overlapping polygons into non-overlapping parts
    print("Exploding overlapping polygons...")
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    peat_gdf = peat_gdf.explode(index_parts=False).reset_index(drop=True)
    
    # 4. Perform intersection between filtered subcompartments and peat extent
    print("Performing intersection between wet woodland species and peat extent...")
    
    # Create a spatial index for better performance
    from rtree import index
    
    # Build spatial index for peat polygons
    idx = index.Index()
    for i, geom in enumerate(peat_gdf.geometry):
        idx.insert(i, geom.bounds)
    
    # Perform intersection
    intersection_results = []
    
    for i, row in gdf.iterrows():
        subcomp_geom = row.geometry
        # Find potential intersections using spatial index
        potential_intersections = list(idx.intersection(subcomp_geom.bounds))
        
        for peat_idx in potential_intersections:
            peat_geom = peat_gdf.iloc[peat_idx].geometry
            if subcomp_geom.intersects(peat_geom):
                intersection = subcomp_geom.intersection(peat_geom)
                if not intersection.is_empty:
                    # Create a new row with intersection geometry and original attributes
                    new_row = row.copy()
                    new_row.geometry = intersection
                    intersection_results.append(new_row)
    
    if intersection_results:
        # Create GeoDataFrame from intersection results
        intersection_gdf = gpd.GeoDataFrame(intersection_results, crs=gdf.crs)
        
        # Filter out non-polygon geometries (points, lines) that can't be saved to shapefile
        print("Filtering out non-polygon geometries...")
        polygon_mask = intersection_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])
        intersection_gdf = intersection_gdf[polygon_mask].copy()
        
        if len(intersection_gdf) == 0:
            print("❌ No valid polygon intersections found after filtering")
            print("All intersections resulted in points or lines")
            return
        
        print(f"Kept {len(intersection_gdf)} polygon intersections out of {len(intersection_results)} total intersections")
        
        # Optional consolidation of touching polygons
        if args.consolidate:
            print("Consolidating touching polygons...")
            if args.consolidate_by:
                print(f"Consolidating by attribute: {args.consolidate_by}")
                intersection_gdf = intersection_gdf.dissolve(by=args.consolidate_by, as_index=False)
            else:
                print("Consolidating by all attributes")
                intersection_gdf = intersection_gdf.dissolve(by=None, as_index=False)
            print(f"Consolidated to {len(intersection_gdf)} features")
        
        # Save the result
        print(f"Saving intersection result to {output_path}")
        intersection_gdf.to_file(output_path)
        
        print(f"✅ Processing complete!")
        print(f"📊 Original subcompartments: {len(gdf)}")
        print(f"📊 Peat extent polygons: {len(peat_gdf)}")
        print(f"📊 Intersection polygons: {len(intersection_gdf)}")
        print(f"📊 Final result: {len(intersection_gdf)} individual subcompartments")
        
    else:
        print("❌ No intersection found between wet woodland species and peat extent")
        print("This could mean:")
        print("- The areas don't overlap geographically")
        print("- There are CRS issues")
        print("- Geometry issues in the data")

if __name__ == "__main__":
    main()