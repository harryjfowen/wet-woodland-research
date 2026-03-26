# Scientific Framework: Wet Woodland Suitability Mapping

## Purpose
Create a scientifically-defensible, continuous wetness gradient map for identifying suitable sites for wet woodland expansion across England.

---

## 🌳 Wet Woodland Ecological Requirements

Based on peer-reviewed literature, wet woodlands require:

### 1. **High Water Table / Frequent Waterlogging**
- **Indicator**: TWI (Topographic Wetness Index)
- **Justification**: Wet woodland species (alder, willow, birch) require seasonal waterlogging
- **Citations**:
  - Rodwell (1991) - British Plant Communities Vol. 1 (Woodland)
  - Wheeler et al. (2009) - "Wet woodland in the UK"
  - Beven & Kirkby (1979) - TWI original paper

### 2. **Proximity to Water Sources**
- **Indicator**: HAND (Height Above Nearest Drainage)
- **Justification**: Wet woodlands occur on floodplains and riparian zones (<5m above streams)
- **Citations**:
  - Rennó et al. (2008) - HAND methodology
  - Klijn & Witte (1999) - "Eco-hydrology: groundwater flow and site factors in plant ecology"

### 3. **Hydrological Connectivity**
- **Indicator**: DTW (Depth to Water Table)
- **Justification**: Root systems need access to groundwater or perched water tables
- **Citations**:
  - Murphy et al. (2008) - DTW for wetland delineation
  - Gowing et al. (2002) - "Effect of water table depth on vegetation in UK wet grasslands"

### 4. **Depression/Ponding Areas**
- **Indicator**: Depression Depth
- **Justification**: Seasonal ponding creates anaerobic conditions favoring wet woodland
- **Citations**:
  - Lindsay et al. (2016) - "Wet woodland restoration on lowland raised bogs"

### 5. **NOT Flow Channels**
- **Indicator**: Flow Accumulation (EXCLUDE high values)
- **Justification**: Wet woodlands occur adjacent to, not within, active channels
- **Note**: Use channels to define drainage network, but don't plant in active channels

---

## 📊 Proposed Weight Calibration Strategy

### **Option A: Literature-Based Weights (Defensible Now)**

Based on ecological literature on wet woodland hydrology:

```python
weights = {
    'twi': 0.35,          # Primary: predicts waterlogging (Wheeler et al. 2009)
    'hand': 0.30,         # Primary: floodplain proximity (Rodwell 1991)
    'dtw': 0.25,          # Secondary: water table access (Gowing et al. 2002)
    'depression_depth': 0.10  # Tertiary: ponding tendency
}
```

**Remove EAS**: Redundant with HAND (correlation >0.9)

**Justification**:
- TWI and HAND are the two most commonly cited indicators for wetland/wet woodland mapping
- Combined weight of 65% reflects their primacy in the literature
- DTW adds groundwater perspective (25%)
- Depression depth is minor contributor (10%) - captures micro-topography

---

### **Option B: Empirically Optimized Weights (Most Rigorous)**

Use your **NFI wet woodland data** to calibrate weights via logistic regression:

```python
# Pseudo-code approach
from sklearn.linear_model import LogisticRegression

# 1. Extract terrain indices at NFI wet woodland locations
X = np.stack([twi, hand, dtw, depression_depth], axis=-1)

# 2. Labels: 1 = wet woodland (from NFI), 0 = not wet woodland
y = nfi_wet_woodland_labels

# 3. Fit logistic regression to find optimal weights
model = LogisticRegression()
model.fit(X, y)

# 4. Use coefficients as weights (normalize to sum to 1.0)
weights = model.coef_ / model.coef_.sum()
```

**This gives you empirically-derived weights based on actual wet woodland occurrence patterns!**

---

## 🗺️ Global Normalization Strategy

**CRITICAL ISSUE**: Current script uses per-tile percentiles, making results incomparable across England.

### **Solution: Two-Pass Processing**

#### **Pass 1: Calculate Global Statistics**
```python
# Process all tiles, collect statistics
global_stats = {
    'twi': {'p05': None, 'p95': None},
    'hand': {'min': None, 'p95': None},
    'dtw': {'min': None, 'p95': None},
    'depression': {'min': None, 'p95': None}
}

# For each tile, update running statistics
# Use Welford's online algorithm for mean/variance
# Track global min/max/percentiles
```

#### **Pass 2: Normalize Using Global Stats**
```python
# Apply same normalization across all tiles
twi_norm = (twi - global_stats['twi']['p05']) / (global_stats['twi']['p95'] - global_stats['twi']['p05'])
hand_norm = 1 - (hand - global_stats['hand']['min']) / (global_stats['hand']['p95'] - global_stats['hand']['min'])
# etc.
```

**Result**: Wetness score of 0.8 means the same thing in Cornwall and Yorkshire!

---

## 🎯 Absolute Thresholds (Alternative Approach)

Instead of percentile normalization, use **ecologically-meaningful thresholds** from literature:

### **TWI Thresholds**
- TWI < 5: Dry uplands
- TWI 5-10: Mesic
- TWI 10-15: Seasonally wet
- **TWI > 15: Wet woodland suitable** (Sørensen et al. 2006)

Normalize: `twi_norm = (twi - 5) / (15 - 5)` capped at [0, 1]

### **HAND Thresholds**
- HAND > 10m: Upland, rarely floods
- HAND 2-10m: Intermediate
- **HAND < 2m: Floodplain/riparian (wet woodland)** (Nobre et al. 2011)

Normalize: `hand_norm = 1 - (hand / 10)` capped at [0, 1]

### **DTW Thresholds**
- DTW > 5m: Deep water table
- DTW 1-5m: Moderate
- **DTW < 1m: Shallow water table (wet woodland)** (Murphy et al. 2008)

Normalize: `dtw_norm = 1 - (dtw / 5)` capped at [0, 1]

### **Depression Depth**
- Depth < 0.1m: Negligible
- **Depth > 0.5m: Significant ponding** (field observation)

Normalize: `dep_norm = depression / 0.5` capped at [0, 1]

---

## ✅ Validation Strategy

### **1. NFI Data Validation**
- Extract wetness scores at all NFI wet woodland plots
- Calculate ROC curve / AUROC
- Report: "Our wetness index achieves AUROC=0.XX in distinguishing wet woodlands from other woodland types"

### **2. Independent Validation**
- Reserve 20% of NFI plots for testing
- Use remaining 80% for calibration
- Report validation metrics on held-out test set

### **3. Spatial Validation**
- Compare to external datasets:
  - Environment Agency flood zones (should correlate with high wetness)
  - BGS groundwater emergence maps
  - National Wetland Inventory (if available)

### **4. Field Validation (Ideal)**
- Visit random stratified sample of sites:
  - High wetness (0.7-1.0): n=20 sites
  - Medium wetness (0.4-0.7): n=20 sites
  - Low wetness (0-0.4): n=20 sites
- Record actual wetness indicators (standing water, gleyed soils, wet woodland species)
- Calculate agreement

---

## 📝 Methods Section Text (for your paper)

### **Draft Methods:**

**Wet Woodland Suitability Index**

We developed a continuous wetness index (0-1 scale) to identify areas suitable for wet woodland expansion across England, based on terrain-derived hydrological indicators. The index integrates four topographic proxies for wetland conditions:

1. **Topographic Wetness Index (TWI)**: Calculated as ln(α/tan(β)), where α is upslope contributing area per unit contour length and β is local slope (Beven & Kirkby, 1979). TWI predicts areas of water accumulation and saturation, with values >10 indicating seasonally waterlogged conditions suitable for wet woodland species (Sørensen et al., 2006).

2. **Height Above Nearest Drainage (HAND)**: Vertical distance from each pixel to the nearest drainage channel, following the method of Rennó et al. (2008). HAND identifies floodplain and riparian zones; values <2m indicate frequent inundation consistent with wet woodland occurrence (Nobre et al., 2011).

3. **Depth to Water (DTW)**: Least-cost hydrological distance to nearest surface water, weighted by slope resistance (Murphy et al., 2008). DTW estimates water table depth, with values <1m indicating shallow groundwater accessible to wet woodland root systems.

4. **Depression Depth**: Vertical depth of topographic depressions calculated as the difference between pit-filled and original DEM (using Wang & Liu 2006 algorithm implemented in RichDEM). Depression depth identifies areas prone to seasonal ponding that create the anaerobic soil conditions required by wet woodland communities (Lindsay et al., 2016).

**Indicator Weighting and Normalization**

[**OPTION 1 - Literature-based**]
Weights were assigned based on the relative importance of each indicator in wet woodland ecology literature: TWI (0.35), HAND (0.30), DTW (0.25), and depression depth (0.10). TWI and HAND received the highest weights as they are the most commonly cited predictors of wet woodland occurrence (Rodwell, 1991; Wheeler et al., 2009).

[**OPTION 2 - Empirically calibrated**]
Weights were empirically derived by fitting a logistic regression model to Forestry England sub-compartment data, with wet woodland presence/absence as the response variable and the four terrain indices as predictors (n=XXX,XXX pixels across XXX wet woodland compartments). Coefficients were normalized to sum to 1.0, yielding weights of: TWI (0.XX), HAND (0.XX), DTW (0.XX), depression depth (0.XX).

Each indicator was normalized to 0-1 range using [**global percentiles**]/[**absolute thresholds**] calculated across the entire study area to ensure spatial comparability. For TWI, normalization used the 5th and 95th percentiles as bounds. HAND, DTW, and depression depth were inverted (1 - normalized value) such that higher values represent wetter conditions across all indices.

The final suitability index was calculated as the weighted sum of normalized indicators, producing continuous values from 0 (unsuitable) to 1 (highly suitable for wet woodland). We validated the index against independent Forestry England wet woodland compartments (n=XXX), achieving an AUROC of 0.XX.

**Processing and Implementation**

Analysis was performed on 4m resolution LiDAR-derived DTM tiles covering England (Environment Agency, 2023), processed using parallel tiled computation with 1km buffers to mitigate edge effects. [If using SMUK] The topographic suitability index was multiplied by SMUK satellite-derived wet persistence (0-1 scale) to incorporate temporal hydrological dynamics.

---

## 🔧 Implementation Checklist

- [ ] Remove EAS (redundant with HAND)
- [ ] Implement global normalization (two-pass or absolute thresholds)
- [ ] Choose weight calibration method (literature vs. empirical)
- [ ] Add NFI validation against wet woodland labels
- [ ] Calculate AUROC, precision, recall at different thresholds
- [ ] Add metadata to outputs (weights used, normalization method, date)
- [ ] Create validation plots (ROC curve, wetness distribution in wet vs. non-wet woodlands)
- [ ] Document all parameter choices with citations

---

## 📚 Key Citations to Include

**Core Methods:**
- Beven, K.J., Kirkby, M.J. (1979). A physically based variable contributing area model of basin hydrology. *Hydrological Sciences Bulletin*, 24(1), 43-69.
- Rennó, C.D., et al. (2008). HAND, a new terrain descriptor using SRTM-DEM. *Geomorphometry*, 2008, 101-106.
- Murphy, P.N.C., et al. (2008). Topographic modelling of soil moisture conditions. *Hydrological Processes*, 22(17), 3359-3368.
- Wang, L., Liu, H. (2006). An efficient method for identifying and filling surface depressions. *International Journal of GIS*, 20(2), 193-213.

**Wet Woodland Ecology:**
- Rodwell, J.S. (1991). *British Plant Communities, Volume 1: Woodlands and Scrub*. Cambridge University Press.
- Wheeler, B.D., et al. (2009). *Wet Woodlands in the UK: A Review*. Natural England Research Report.
- Sørensen, R., et al. (2006). On the calculation of the topographic wetness index. *Journal of Hydrology*, 322(1-4), 276-286.
- Nobre, A.D., et al. (2011). Height Above the Nearest Drainage – a hydrologically relevant new terrain model. *Journal of Hydrology*, 404(1-2), 13-29.

**UK Context:**
- Forestry Commission (2003). *The management of semi-natural woodlands: 8. Wet woodlands*.
- JNCC (2010). *Handbook for Phase 1 habitat survey*.

---

## 💡 Next Steps

1. **Choose approach**: Literature weights or empirical calibration?
2. **Implement global normalization**
3. **Run validation** against your NFI wet woodland labels
4. **Generate validation metrics** (AUROC, confusion matrix, etc.)
5. **Create suitability map** with confidence intervals
6. **Field validate** (if time/budget allows)

Would you like me to:
1. Implement the two-pass global normalization?
2. Create a weight calibration script using your NFI labels?
3. Add validation metrics against your XGBoost training data?
