{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 // ============================================================================\
// ENGLAND EMBEDDINGS EXTRACTOR (CLEAN + WET-WOODLAND-SAFE MASK)\
// ============================================================================\
\
// ---------------------------\
// 0) CONFIG\
// ---------------------------\
var CRS = 'EPSG:27700';\
var SCALE = 10;\
var START_DATE = '2022-01-01';\
var END_DATE = '2023-01-01';\
\
// Woodland mask tuning\
var CHM_HEIGHT_M = 3;      // canopy threshold at 1m\
var MIN_COVER_FRAC = 0.15; // minimum fraction of 1m cells above CHM_HEIGHT_M within 10m cell\
\
// Visualization AOI\
var bboxVisualize = ee.Geometry.Rectangle([-1.13, 53.23, -0.96, 53.30]);\
\
// ---------------------------\
// 1) MASK + STUDY AREA\
// ---------------------------\
var maskImage = ee.Image('users/harryjfowen/compartment_duo_mask');\
var maskRaster = maskImage\
  .rename('mask')\
  .updateMask(maskImage.neq(255))\
  .reproject(\{crs: CRS, scale: SCALE\});\
\
// Valid footprint mask\
var validMask = maskRaster.mask();\
var studyArea = maskRaster.geometry();\
\
print('Study area (km\'b2):', studyArea.area().divide(1e6));\
\
// ---------------------------\
// 2) GOOGLE 64-DIM EMBEDDINGS (10m)\
// ---------------------------\
var englandBounds = ee.Geometry.Rectangle([-6, 49, 2, 56]);\
\
var embCol = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')\
  .filterDate(START_DATE, END_DATE)\
  .filterBounds(englandBounds);\
\
// Ensure a valid default projection for reduceResolution\
var embProj = ee.Image(embCol.first()).select(0).projection();\
\
var embMosaic = embCol.mosaic()\
  .setDefaultProjection(embProj)\
  .updateMask(validMask);\
\
// Downsample/resample to 10m mean\
var emb10m = embMosaic\
  .reduceResolution(\{\
    reducer: ee.Reducer.mean(),\
    maxPixels: 1024\
  \})\
  .reproject(\{crs: CRS, scale: SCALE\});\
\
// ---------------------------\
// 3) EA LIDAR -> CHM + DTM (10m)\
// ---------------------------\
var terrain2022 = ee.Image('UK/EA/ENGLAND_1M_TERRAIN/2022');\
var terrainProj = terrain2022.select('dtm').projection();\
\
var dsm1m = terrain2022.select('dsm_first')\
  .setDefaultProjection(terrainProj)\
  .updateMask(validMask);\
\
var dtm1m = terrain2022.select('dtm')\
  .setDefaultProjection(terrainProj)\
  .updateMask(validMask);\
\
var chm1m = dsm1m.subtract(dtm1m)\
  .rename('chm_1m')\
  .setDefaultProjection(terrainProj);\
\
var dtm10m = dtm1m\
  .reduceResolution(\{\
    reducer: ee.Reducer.mean(),\
    maxPixels: 1024\
  \})\
  .reproject(\{crs: CRS, scale: SCALE\})\
  .rename('dtm_mean');\
\
var chm10mMean = chm1m\
  .reduceResolution(\{\
    reducer: ee.Reducer.mean(),\
    maxPixels: 1024\
  \})\
  .reproject(\{crs: CRS, scale: SCALE\})\
  .rename('chm_mean');\
\
// ---------------------------\
// 4) WOODLAND MASK (cover-based)\
// ---------------------------\
var coverGe2 = chm1m.gte(CHM_HEIGHT_M)\
  .setDefaultProjection(terrainProj)\
  .reduceResolution(\{\
    reducer: ee.Reducer.mean(), // mean of 0/1 => fraction\
    maxPixels: 1024\
  \})\
  .reproject(\{crs: CRS, scale: SCALE\})\
  .rename('chm_cover_ge2');\
\
var woodlandMask = coverGe2.gte(MIN_COVER_FRAC).rename('woodland_mask');\
\
Map.addLayer(\
  woodlandMask,\
  \{min: 0, max: 1, palette: ['white', 'darkgreen']\},\
  'Woodland mask (cover >= ' + CHM_HEIGHT_M + 'm, >= ' + (MIN_COVER_FRAC * 100) + '%)'\
);\
\
// ---------------------------\
// 5) COMBINE FEATURES\
// ---------------------------\
var allFeatures = emb10m\
  .addBands(dtm10m)\
  .addBands(chm10mMean)\
  .addBands(coverGe2)\
  .updateMask(validMask)\
  .updateMask(woodlandMask);\
\
print('Output band count:', allFeatures.bandNames().size());\
print('Output band names:', allFeatures.bandNames());\
\
// ---------------------------\
// 6) EXPORT\
// ---------------------------\
Export.image.toDrive(\{\
  image: allFeatures.toFloat(),\
  description: 'wet_woodland_training',\
  folder: 'wet_woodland_train',\
  region: studyArea,\
  scale: SCALE,\
  crs: CRS,\
  maxPixels: 1e13,\
  fileFormat: 'GeoTIFF',\
  formatOptions: \{cloudOptimized: true\},\
  skipEmptyTiles: true\
\});\
\
// ---------------------------\
// 7) VISUALIZATION\
// ---------------------------\
Map.centerObject(bboxVisualize, 13);\
\
var embeddingsPreview = emb10m.clip(bboxVisualize).updateMask(woodlandMask);\
var chmPreview = chm10mMean.clip(bboxVisualize).updateMask(woodlandMask);\
var woodlandPreview = woodlandMask.clip(bboxVisualize);\
\
Map.addLayer(\
  embeddingsPreview,\
  \{min: -0.3, max: 0.3, bands: ['A01', 'A16', 'A09']\},\
  'Embeddings 10m (woodland-masked)'\
);\
\
Map.addLayer(\
  chmPreview,\
  \{min: 0, max: 30, palette: ['blue', 'green', 'yellow', 'red']\},\
  'CHM mean 10m (woodland-masked)'\
);\
\
Map.addLayer(\
  woodlandPreview,\
  \{min: 0, max: 1, palette: ['white', 'darkgreen']\},\
  'Woodland mask preview'\
);\
\
print('\uc0\u9989  Script ready. Start export from Tasks tab.');\
print('\uc0\u55356 \u57138  Woodland mask:', 'CHM >= ' + CHM_HEIGHT_M + 'm cover >= ' + (MIN_COVER_FRAC * 100) + '% per 10m pixel');\
}
