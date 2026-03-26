# Implementation Plan: Scientific Wet Woodland Suitability Mapping

## 🎯 Goal
Transform `wet_areas_tiled.py` from an exploratory tool into a scientifically-rigorous wet woodland suitability mapper.

---

## 🔧 Required Changes

### **1. Remove Redundant Indicator (EAS)**
**Why**: EAS and HAND are highly correlated (both measure elevation above drainage)
**Action**: Remove `calculate_elevation_above_stream()` and EAS from composite

### **2. Implement Global Normalization**
**Why**: Current per-tile percentile normalization makes results incomparable across England
**Action**: Two-pass processing:
- Pass 1: Calculate global statistics across all tiles
- Pass 2: Normalize using global stats

### **3. Scientifically-Justified Weights**
**Current**: Arbitrary weights with no justification
**Options**:
- **Option A (Quick)**: Literature-based weights with citations
- **Option B (Rigorous)**: Calibrate against your NFI wet woodland labels

### **4. Add Validation Module**
**Why**: Need to demonstrate the index actually predicts wet woodland
**Action**: Compare to your NFI/XGBoost training labels
- Calculate AUROC, precision, recall
- Generate ROC curve
- Report performance metrics

### **5. Add Metadata and Provenance**
**Action**: Include in output files:
- Weights used
- Normalization method
- Processing date
- Citations for methods
- Validation metrics

---

## 📊 Two Implementation Strategies

### **Strategy A: Fast, Defensible (1-2 days)**
1. Remove EAS
2. Use absolute threshold normalization (from literature)
3. Use literature-based weights (TWI=0.35, HAND=0.30, DTW=0.25, Depression=0.10)
4. Validate against NFI labels (calculate AUROC)
5. Add citations to outputs

**Pros**: Quick, defensible with literature citations
**Cons**: Not optimized for your specific study area

---

### **Strategy B: Optimal, Rigorous (3-5 days)**
1. Remove EAS
2. Implement two-pass global normalization
3. **Calibrate weights using your NFI wet woodland labels**
4. Cross-validate (80/20 split)
5. Generate validation report
6. Compare to XGBoost model predictions

**Pros**: Empirically optimized for England + your data
**Cons**: More complex, requires NFI label extraction

---

## 🚀 Recommended Approach: **Strategy B**

**Why?** You already have high-quality wet woodland labels from NFI! Use them!

### **Step-by-Step:**

#### **Step 1: Extract Terrain Indices at NFI Locations**
```python
# For each NFI wet woodland polygon:
# 1. Extract mean TWI, HAND, DTW, depression_depth
# 2. Label = 1 if wet woodland, 0 otherwise
# Result: training dataset for calibration
```

#### **Step 2: Optimize Weights via Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Fit model
X = np.column_stack([twi, hand, dtw, depression])
y = wet_woodland_labels  # 1 = wet woodland, 0 = not

model = LogisticRegression(penalty=None)  # No regularization
model.fit(X, y)

# Extract normalized weights
weights_raw = model.coef_[0]
weights_normalized = weights_raw / weights_raw.sum()

print(f"Empirically-optimized weights:")
print(f"  TWI: {weights_normalized[0]:.3f}")
print(f"  HAND: {weights_normalized[1]:.3f}")
print(f"  DTW: {weights_normalized[2]:.3f}")
print(f"  Depression: {weights_normalized[3]:.3f}")

# Validate
y_pred = model.predict_proba(X)[:, 1]
auroc = roc_auc_score(y, y_pred)
print(f"AUROC: {auroc:.3f}")
```

#### **Step 3: Implement Global Normalization**
```python
# Pass 1: Collect statistics
for tile in tiles:
    twi, hand, dtw, dep = calculate_indices(tile)

    # Update running statistics
    global_stats['twi']['values'].append(twi[~np.isnan(twi)])
    # ... etc for all indices

# Calculate global percentiles
global_twi_p05 = np.percentile(np.concatenate(global_stats['twi']['values']), 5)
global_twi_p95 = np.percentile(np.concatenate(global_stats['twi']['values']), 95)
# ... etc

# Pass 2: Normalize with global stats
twi_norm = (twi - global_twi_p05) / (global_twi_p95 - global_twi_p05)
```

#### **Step 4: Update Script**
- Modify `calculate_composite_wetness()` to use:
  - Empirically-derived weights
  - Global normalization stats
  - Only 4 indicators (remove EAS)

#### **Step 5: Generate Validation Report**
```python
# At end of processing, generate:
validation_report.txt:
  - Weights used (with justification)
  - AUROC on training data
  - AUROC on held-out test data (20%)
  - Confusion matrix at threshold = 0.5
  - Precision/Recall at different thresholds
  - Comparison to XGBoost predictions (correlation)
```

---

## 📝 Modified Methods Section

After implementing Strategy B, your methods text would be:

> **Wet Woodland Suitability Mapping**
>
> We developed a terrain-based wet woodland suitability index using four hydrological indicators derived from Environment Agency 4m LiDAR DTM: Topographic Wetness Index (TWI; Beven & Kirkby 1979), Height Above Nearest Drainage (HAND; Rennó et al. 2008), Depth to Water (DTW; Murphy et al. 2008), and topographic depression depth (Wang & Liu 2006).
>
> Indicator weights were empirically calibrated using logistic regression fitted to Forestry England sub-compartment wet woodland occurrence data (n=XXX wet woodland compartments, XXX non-wet woodland compartments). The calibrated model assigned weights of: TWI (0.XX), HAND (0.XX), DTW (0.XX), depression depth (0.XX). All indicators were normalized to 0-1 range using global 5th-95th percentiles calculated across England.
>
> The final suitability index was calculated as the weighted sum of normalized indicators. Cross-validation (80/20 split) yielded AUROC = 0.XX for discriminating wet woodland from other woodland types, demonstrating strong predictive ability. The suitability map provides a continuous 0-1 gradient identifying areas most suitable for wet woodland establishment or expansion.

---

## 🎯 Deliverables

After implementation, you'll have:

1. **✅ Scientifically-defensible suitability map**
   - Empirically-calibrated weights (not arbitrary!)
   - Global normalization (spatially comparable)
   - Validation metrics (proven accuracy)

2. **✅ Validation report**
   - AUROC, precision, recall
   - ROC curve figure
   - Confusion matrix
   - Weight justification

3. **✅ Publication-ready methods**
   - Clear citations
   - Reproducible workflow
   - Quantified accuracy

4. **✅ Comparison to XGBoost**
   - Are terrain indices alone sufficient?
   - Does Alpha Earth embeddings add value?
   - Which approach is better for policy/practice?

---

## ⏱️ Time Estimate

- **Strategy A (literature-based)**: 1-2 days
- **Strategy B (empirically-calibrated)**: 3-5 days

**Recommendation**: Go with Strategy B - you'll invest a bit more time now but get a much stronger scientific product that reviewers will love!

---

## 🤔 Discussion Point: Relationship to XGBoost Model

**Important question**: How does this terrain-based suitability map relate to your XGBoost wet woodland classifier?

**Two complementary uses:**

1. **XGBoost**: Identifies **existing** wet woodland (uses spectral + terrain + climate)
2. **Suitability Map**: Identifies **potential** sites for expansion (terrain only)

**Why terrain-only for suitability?**
- Spectral signatures reflect *current* vegetation (not potential)
- Terrain is permanent (doesn't change with land use)
- Suitability = "if we planted here, would hydrology support wet woodland?"

**Validation strategy:**
- Suitability should be HIGH at existing wet woodland locations (XGBoost = wet woodland)
- But suitability can also be HIGH at non-woodland locations (e.g., wet grassland on suitable terrain)

This is actually a really nice two-model approach:
- Model 1 (XGBoost): "Where is wet woodland now?"
- Model 2 (Suitability): "Where could it be in the future?"

Make sense?
