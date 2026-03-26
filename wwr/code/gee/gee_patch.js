// ============================================================================
// WET WOODLAND EARTH ENGINE PATCH EXPORT
// ============================================================================
//
// Purpose:
//   Export a small buffered patch around the actual EA terrain hole near
//   RAF Fylingdales so it can be spliced into the existing national inference.
//
// Outputs:
//   1) A 68-band inference patch:
//      64 x Google satellite embeddings
//      1 x dtm_elevation
//      1 x dtm_slope
//      1 x chm_canopy_height
//      1 x chm_canopy_gap   (legacy name; actually canopy cover fraction)
//   2) The actual EA-missing-hole mask, so the patch can be trimmed locally
//      against the true hole rather than by thresholding.

// ---------------------------
// 0) CONFIG
// ---------------------------
var CRS = 'EPSG:27700';
var SCALE = 10;
var START_DATE = '2022-01-01';
var END_DATE = '2023-01-01';

var CHM_HEIGHT_M = 3;
var MIN_COVER_FRAC = 0.15;

var PATCH_BUFFER_M = 2000;
var HOLE_SEARCH_BUFFER_M = 12000;

var TERRAIN_MODE = 'meta_chm_copdem';         // 'meta_chm_copdem' | 'ea_lidar'
var EXPORT_MASK_MODE = 'full_valid_footprint'; // 'full_valid_footprint' | 'woodland_only'

var MASK_ASSET = 'projects/desert-ecosystems/assets/tow_mask';
var META_CHM_ASSET = 'projects/sat-io/open-datasets/facebook/meta-canopy-height';
var COPDEM_ASSET = 'COPERNICUS/DEM/GLO30';

var RAF_FYLINGDALES = ee.Geometry.Point([-0.6697, 54.3589]);
var holeSearchRegion = RAF_FYLINGDALES.buffer(HOLE_SEARCH_BUFFER_M).bounds(1);

// ---------------------------
// 1) STUDY MASK + EA HOLE
// ---------------------------
var maskImage = ee.Image(MASK_ASSET);
var maskRaster = maskImage
  .rename('mask')
  .updateMask(maskImage.neq(255))
  .reproject({crs: CRS, scale: SCALE});

var validMask = maskRaster.mask();
var validFootprint = ee.Image.constant(1)
  .rename('valid_footprint')
  .updateMask(validMask);

var eaTerrain = ee.Image('UK/EA/ENGLAND_1M_TERRAIN/2022');
var eaTerrainCoverage = ee.Image.constant(1)
  .rename('ea_terrain_coverage')
  .updateMask(eaTerrain.select('dtm').mask())
  .reproject({crs: CRS, scale: SCALE});

var eaMissingTerrainSearch = validFootprint.unmask(0).eq(1)
  .and(eaTerrainCoverage.unmask(0).eq(0))
  .selfMask()
  .clip(holeSearchRegion)
  .rename('ea_missing_terrain');

var holeGeometry = eaMissingTerrainSearch.geometry();
var patchRegion = ee.Geometry(ee.Algorithms.If(
  holeGeometry.area(1).gt(0),
  holeGeometry.buffer(PATCH_BUFFER_M).bounds(1),
  holeSearchRegion
));

var eaMissingTerrain = validFootprint.unmask(0).eq(1)
  .and(eaTerrainCoverage.unmask(0).eq(0))
  .selfMask()
  .clip(patchRegion)
  .rename('ea_missing_terrain');

print('Terrain mode:', TERRAIN_MODE);
print('Export mask mode:', EXPORT_MASK_MODE);
print('Patch buffer (m):', PATCH_BUFFER_M);
print('Hole search buffer (m):', HOLE_SEARCH_BUFFER_M);
print('Patch region (km^2):', patchRegion.area().divide(1e6));
print('Actual EA hole in patch region (ha):',
  ee.Image.pixelArea().divide(1e4).updateMask(eaMissingTerrain).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: patchRegion,
    scale: SCALE,
    maxPixels: 1e12
  }).get('area'));

// ---------------------------
// 2) GOOGLE 64-DIM EMBEDDINGS (10m)
// ---------------------------
var embCol = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate(START_DATE, END_DATE)
  .filterBounds(patchRegion);

var embProj = ee.Image(embCol.first()).select(0).projection();
var embeddingNames = ee.List.sequence(0, 63).map(function(i) {
  return ee.String('embedding_').cat(ee.Number(i).format('%d'));
});

var emb10m = embCol
  .mosaic()
  .setDefaultProjection(embProj)
  .reduceResolution({
    reducer: ee.Reducer.mean(),
    maxPixels: 1024
  })
  .reproject({crs: CRS, scale: SCALE})
  .rename(embeddingNames)
  .updateMask(validMask)
  .clip(patchRegion);

// ---------------------------
// 3) TERRAIN / CHM SOURCES
// ---------------------------
function buildEaLidarTerrain(region) {
  var terrainProj = eaTerrain.select('dtm').projection();

  var dsm1m = eaTerrain.select('dsm_first')
    .setDefaultProjection(terrainProj)
    .updateMask(validMask)
    .clip(region);

  var dtm1m = eaTerrain.select('dtm')
    .setDefaultProjection(terrainProj)
    .updateMask(validMask)
    .clip(region);

  var chm1m = dsm1m.subtract(dtm1m)
    .max(0)
    .rename('chm_raw')
    .setDefaultProjection(terrainProj);

  var dtmElevation = dtm1m
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    })
    .reproject({crs: CRS, scale: SCALE})
    .rename('dtm_elevation');

  var dtmSlope = ee.Terrain.slope(dtmElevation)
    .rename('dtm_slope');

  var chmHeight = chm1m
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    })
    .reproject({crs: CRS, scale: SCALE})
    .rename('chm_canopy_height');

  var canopyCover = chm1m.gte(CHM_HEIGHT_M)
    .setDefaultProjection(terrainProj)
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    })
    .reproject({crs: CRS, scale: SCALE})
    .rename('chm_canopy_gap');

  return {
    dtmElevation: dtmElevation.updateMask(validMask).clip(region),
    dtmSlope: dtmSlope.updateMask(validMask).clip(region),
    chmHeight: chmHeight.updateMask(validMask).clip(region),
    canopyCover: canopyCover.updateMask(validMask).clip(region),
    sourceLabel: 'EA lidar terrain'
  };
}

function buildMetaGlobalTerrain(region) {
  var metaCol = ee.ImageCollection(META_CHM_ASSET).filterBounds(region);
  var metaFirst = ee.Image(metaCol.first());
  var metaProj = metaFirst.projection();

  var metaChm = metaCol
    .mosaic()
    .setDefaultProjection(metaProj)
    .rename('chm_raw')
    .updateMask(validMask)
    .clip(region);

  var chmHeight = metaChm
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    })
    .reproject({crs: CRS, scale: SCALE})
    .rename('chm_canopy_height');

  var canopyCover = metaChm.gte(CHM_HEIGHT_M)
    .setDefaultProjection(metaProj)
    .reduceResolution({
      reducer: ee.Reducer.mean(),
      maxPixels: 1024
    })
    .reproject({crs: CRS, scale: SCALE})
    .rename('chm_canopy_gap');

  var dem30m = ee.ImageCollection(COPDEM_ASSET)
    .select('DEM')
    .filterBounds(region)
    .mosaic()
    .rename('dem_30m');

  var dem30mBng = dem30m
    .resample('bilinear')
    .reproject({crs: CRS, scale: 30});

  var dtmElevation = dem30mBng
    .resample('bilinear')
    .reproject({crs: CRS, scale: SCALE})
    .rename('dtm_elevation')
    .updateMask(validMask);

  var dtmSlope = ee.Terrain.slope(dem30mBng)
    .resample('bilinear')
    .reproject({crs: CRS, scale: SCALE})
    .rename('dtm_slope')
    .updateMask(validMask);

  return {
    dtmElevation: dtmElevation.clip(region),
    dtmSlope: dtmSlope.clip(region),
    chmHeight: chmHeight.updateMask(validMask).clip(region),
    canopyCover: canopyCover.updateMask(validMask).clip(region),
    sourceLabel: 'Meta CHM + Copernicus DEM GLO-30 fallback'
  };
}

var terrain = TERRAIN_MODE === 'ea_lidar'
  ? buildEaLidarTerrain(patchRegion)
  : buildMetaGlobalTerrain(patchRegion);

print('Terrain source:', terrain.sourceLabel);

// ---------------------------
// 4) FEATURE STACK
// ---------------------------
var woodlandMask = terrain.canopyCover
  .gte(MIN_COVER_FRAC)
  .rename('woodland_mask');

var rawFeatures = emb10m
  .addBands(terrain.dtmElevation)
  .addBands(terrain.dtmSlope)
  .addBands(terrain.chmHeight)
  .addBands(terrain.canopyCover)
  .updateMask(validMask)
  .clip(patchRegion);

var exportImage = EXPORT_MASK_MODE === 'woodland_only'
  ? rawFeatures.updateMask(woodlandMask)
  : rawFeatures;

var exportMask = exportImage.mask()
  .reduce(ee.Reducer.min())
  .rename('export_mask')
  .selfMask()
  .clip(patchRegion);

var residualMissingMask = validFootprint.unmask(0).eq(1)
  .and(exportMask.unmask(0).eq(0))
  .selfMask()
  .clip(patchRegion)
  .rename('residual_missing_mask');

print('Output band count:', exportImage.bandNames().size());
print('Residual missing inside patch after export mask (ha):',
  ee.Image.pixelArea().divide(1e4).updateMask(residualMissingMask).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: patchRegion,
    scale: SCALE,
    maxPixels: 1e12
  }).get('area'));

// ---------------------------
// 5) EXPORTS
// ---------------------------
Export.image.toDrive({
  image: exportImage.toFloat(),
  description: 'wet_woodland_inference_patch_raf_fylingdales',
  folder: 'wet_woodland_inference_gap_fill',
  region: patchRegion,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true},
  skipEmptyTiles: false
});

Export.image.toDrive({
  image: eaMissingTerrain.unmask(0).toByte(),
  description: 'wet_woodland_inference_patch_raf_fylingdales_hole_mask',
  folder: 'wet_woodland_inference_gap_fill',
  region: patchRegion,
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true},
  skipEmptyTiles: false
});

// ---------------------------
// 6) VISUALIZATION
// ---------------------------
Map.centerObject(patchRegion, 12);

Map.addLayer(
  ee.Image().byte().paint(patchRegion, 1, 3),
  {palette: ['ffff00']},
  'Patch region (hole + 2km buffer)',
  true
);

Map.addLayer(
  validFootprint.clip(patchRegion),
  {min: 0, max: 1, palette: ['ffffff', '000000'], opacity: 0.4},
  'Valid footprint',
  true
);

Map.addLayer(
  eaMissingTerrain,
  {palette: ['ff0000']},
  'Actual EA hole',
  true
);

Map.addLayer(
  exportMask,
  {palette: ['00ff00']},
  'Patch export footprint',
  false
);

Map.addLayer(
  residualMissingMask,
  {palette: ['ff00ff']},
  'Residual missing after export mask',
  false
);

print('Script ready. Run both export tasks from the Tasks tab.');
print('Local trim should use the exported hole-mask TIFF, not hysteresis.');
