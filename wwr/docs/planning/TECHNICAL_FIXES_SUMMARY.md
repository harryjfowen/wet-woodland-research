# Technical Fixes Summary: Wet Areas Mapping

## 🎯 Core Technical Issues & Solutions

---

## **Issue 1: Local vs. Global Normalization**

### **Current Problem:**
```python
# wet_areas_tiled.py line 559-576
twi_norm = (twi - np.nanpercentile(twi, 5)) / (np.nanpercentile(twi, 95) - np.nanpercentile(twi, 5))
hand_norm = 1 - (hand - np.nanmin(hand)) / (np.nanpercentile(hand, 95) - np.nanmin(hand))
```

**Issue**: Uses per-tile percentiles → results not comparable across tiles
- Tile A wetness=0.8 ≠ Tile B wetness=0.8
- Creates artificial discontinuities at tile boundaries
- Can't meaningfully threshold entire map (e.g., "wetness > 0.7 = suitable")

### **Solution: Global Normalization**

#### **Two-Pass Approach:**

**Pass 1: Calculate global statistics**
```python
# Collect statistics from ALL tiles
global_twi_p05 = np.percentile(all_twi_values, 5)
global_twi_p95 = np.percentile(all_twi_values, 95)
global_hand_min = np.min(all_hand_values)
global_hand_p95 = np.percentile(all_hand_values, 95)
# ... etc for DTW and depression
```

**Pass 2: Normalize using global stats**
```python
# Apply same normalization everywhere
twi_norm = (twi - global_twi_p05) / (global_twi_p95 - global_twi_p05)
hand_norm = 1 - (hand - global_hand_min) / (global_hand_p95 - global_hand_min)
```

**Benefits:**
✅ Wetness scores are spatially comparable
✅ No tile boundary artifacts
✅ Can apply global thresholds (e.g., >0.7 = high suitability)

---

## **Issue 2: Redundant Indicators (EAS and HAND)**

### **Current Problem:**
```python
# wet_areas_tiled.py line 524
weights = {'twi': 0.25, 'hand': 0.2, 'eas': 0.2, 'dtw': 0.25, 'depression_depth': 0.1}
```

**Issue**: EAS and HAND are highly correlated (both measure elevation above drainage)
- Correlation likely >0.9
- Giving them combined weight of 0.4 (40%) is double-counting
- Violates statistical independence assumption

### **Solution: Remove EAS**

```python
# Use only 4 independent indicators
weights = {
    'twi': 0.30,           # +0.05 from removing EAS
    'hand': 0.30,          # +0.10 from removing EAS
    'dtw': 0.30,           # +0.05 from removing EAS
    'depression_depth': 0.10
}
```

**Technical justification:**
- HAND is more established in literature (Rennó et al. 2008)
- EAS is essentially HAND along flow paths (minor variation)
- Removing reduces computation time by ~20%

---

## **Issue 3: Hardcoded Arbitrary Weights**

### **Current Problem:**
```python
# wet_areas_tiled.py line 524
weights = {'twi': 0.25, 'hand': 0.2, 'eas': 0.2, 'dtw': 0.25, 'depression_depth': 0.1}
```

**Issues:**
❌ No justification for these specific values
❌ User can't customize without editing code
❌ No sensitivity analysis possible

### **Solution: Configurable Weights**

```python
def calculate_composite_wetness(..., weights: Optional[Dict[str, float]] = None):
    """
    Calculate composite wetness with configurable weights.

    Args:
        weights: Optional custom weights. If None, uses equal weighting.
                 Must sum to 1.0.
    """
    if weights is None:
        # Conservative default: equal weights
        n_indicators = 4
        weights = {
            'twi': 1.0/n_indicators,
            'hand': 1.0/n_indicators,
            'dtw': 1.0/n_indicators,
            'depression_depth': 1.0/n_indicators
        }

    # Validate
    assert abs(sum(weights.values()) - 1.0) < 0.001, "Weights must sum to 1.0"

    # Continue...
```

**Benefits:**
✅ User can specify weights via command line or config file
✅ Can run sensitivity analysis (try different weight combinations)
✅ Equal weights as default (most conservative, no bias)

---

## **Issue 4: No Validation Metrics**

### **Current Problem:**
- Script generates suitability map
- No way to know if it's accurate
- Can't compare different weight configurations
- No quantitative assessment

### **Solution: Add Validation Module**

```python
def validate_against_labels(suitability_map: np.ndarray,
                           ground_truth_labels: np.ndarray) -> Dict:
    """
    Validate suitability map against ground truth wet woodland labels.

    Returns:
        metrics: {
            'auroc': float,
            'auprc': float,
            'confusion_matrix': np.ndarray,
            'optimal_threshold': float
        }
    """
    from sklearn.metrics import roc_auc_score, roc_curve, auc

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth_labels, suitability_map)
    auroc = auc(fpr, tpr)

    # Find optimal threshold (Youden's index)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    return {
        'auroc': auroc,
        'optimal_threshold': optimal_threshold,
        'tpr_at_optimal': tpr[optimal_idx],
        'fpr_at_optimal': fpr[optimal_idx]
    }
```

**Usage:**
```bash
# Generate suitability map with validation
python wet_areas_scientific.py \
  --dtm-dir tiles/ \
  --validation-labels nfi_wet_woodland.tif \
  --output results/

# Output includes:
#   - suitability_map.tif
#   - validation_report.txt (AUROC, optimal threshold, confusion matrix)
```

---

## **Issue 5: Memory-Inefficient Statistics Collection**

### **Current Problem:**
If calculating global statistics, naively storing all pixel values would require:
- England @ 4m resolution ≈ 30 billion pixels
- 4 indicators × 4 bytes/pixel × 30B pixels = 480 GB RAM ❌

### **Solution: Online Statistics (Welford's Algorithm)**

```python
class OnlineStatistics:
    """Calculate running mean/std without storing all values."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0  # For variance

    def update(self, new_values: np.ndarray):
        """Update with new batch (Welford's online algorithm)."""
        for x in new_values[~np.isnan(new_values)]:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (x - self.mean)

    @property
    def variance(self):
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance)
```

**For percentiles:** Subsample to 1M values total (sufficient for accurate percentiles)

**Benefits:**
✅ Constant memory usage (~100MB instead of 480GB)
✅ Single pass through data
✅ Numerically stable

---

## **Issue 6: DTW Falls Back Silently**

### **Current Problem:**
```python
# wet_areas_tiled.py line 407-409
except Exception as e:
    print(f"DTW least-cost calculation failed, using fallback: {e}")
    return calculate_elevation_above_stream(dem, channels)
```

**Issues:**
- Fallback uses completely different algorithm
- Results not comparable between tiles (some use DTW, some use fallback)
- User doesn't know which tiles failed

### **Solution: Explicit Failure Tracking**

```python
def calculate_depth_to_water(...) -> Tuple[np.ndarray, bool]:
    """
    Returns:
        dtw: Depth to water array
        success: True if least-cost succeeded, False if fallback used
    """
    try:
        # ... DTW calculation ...
        return dtw, True
    except Exception as e:
        logging.warning(f"DTW calculation failed, using fallback: {e}")
        fallback = calculate_elevation_above_stream(dem, channels)
        return fallback, False

# Track failures
dtw, dtw_success = calculate_depth_to_water(...)
if not dtw_success:
    failed_tiles.append(tile_id)

# Report at end
if failed_tiles:
    print(f"⚠️  DTW fallback used for {len(failed_tiles)} tiles:")
    print(f"    {', '.join(failed_tiles[:10])}")
```

---

## 📋 Implementation Checklist

### **Critical (Must Fix):**
- [ ] Implement global normalization (two-pass processing)
- [ ] Remove EAS indicator (redundant with HAND)
- [ ] Make weights configurable (command-line arg or config file)

### **Important (Should Fix):**
- [ ] Add validation metrics module
- [ ] Use online statistics for memory efficiency
- [ ] Track DTW fallback failures

### **Nice to Have:**
- [ ] Sensitivity analysis script (test different weights)
- [ ] Comparison to XGBoost predictions
- [ ] Uncertainty quantification (bootstrap estimates)

---

## 🚀 Quick Implementation Guide

### **Step 1: Modify `calculate_composite_wetness()`**

**Current:**
```python
def calculate_composite_wetness(dtm, smuk_persistence=None, pixel_size=4.0, gaussian_sigma=0.0):
    weights = {'twi': 0.25, 'hand': 0.2, 'eas': 0.2, 'dtw': 0.25, 'depression_depth': 0.1}  # ❌ Hardcoded
    # ... calculate EAS ...  # ❌ Redundant
    # ... local normalization ...  # ❌ Per-tile percentiles
```

**Fixed:**
```python
def calculate_composite_wetness(dtm, global_stats, weights=None, smuk_persistence=None,
                               pixel_size=4.0, gaussian_sigma=0.0):
    # Default to equal weights if not provided
    if weights is None:
        weights = {'twi': 0.25, 'hand': 0.25, 'dtw': 0.25, 'depression_depth': 0.25}

    # ... DON'T calculate EAS ...  # ✅ Removed

    # Normalize using global statistics (not local percentiles)
    twi_norm = (twi - global_stats.twi_p05) / (global_stats.twi_p95 - global_stats.twi_p05)
    hand_norm = 1 - (hand - global_stats.hand_min) / (global_stats.hand_p95 - global_stats.hand_min)
    dtw_norm = 1 - (dtw - global_stats.dtw_min) / (global_stats.dtw_p95 - global_stats.dtw_min)
    dep_norm = (depression - global_stats.depression_min) / (global_stats.depression_p95 - global_stats.depression_min)

    # All normalized indicators clipped to [0, 1]
    indicators = {
        'twi': np.clip(twi_norm, 0, 1),
        'hand': np.clip(hand_norm, 0, 1),
        'dtw': np.clip(dtw_norm, 0, 1),
        'depression_depth': np.clip(dep_norm, 0, 1)
    }

    # Calculate weighted composite
    composite = sum(indicators[k] * weights[k] for k in weights.keys())
    return {'composite': composite, **indicators}
```

### **Step 2: Add Two-Pass Processing**

**Main script modification:**
```python
def main():
    parser = argparse.ArgumentParser()
    # ... existing args ...
    parser.add_argument('--pass', choices=['1', '2', 'both'], default='both',
                       help='Processing pass: 1=stats only, 2=map only, both=do both')
    parser.add_argument('--global-stats', help='Global stats JSON (for pass 2)')
    parser.add_argument('--weights-file', help='Custom weights JSON')

    args = parser.parse_args()

    # Pass 1: Calculate global statistics
    if args.pass in ['1', 'both']:
        global_stats = calculate_global_stats(dtm_tiles)
        global_stats.save(output_dir / 'global_stats.json')

    # Pass 2: Generate maps
    if args.pass in ['2', 'both']:
        if args.global_stats:
            global_stats = GlobalStatistics.load(args.global_stats)
        elif args.pass == '2':
            raise ValueError("Must provide --global-stats for pass 2")

        # Load custom weights if provided
        weights = None
        if args.weights_file:
            with open(args.weights_file) as f:
                weights = json.load(f)

        # Process tiles with global normalization
        process_tiles(dtm_tiles, global_stats, weights, output_dir)
```

---

## ⏱️ Time Estimate

- **Fixing normalization**: 4-6 hours
- **Removing EAS**: 1 hour
- **Making weights configurable**: 2 hours
- **Adding validation**: 3-4 hours
- **Testing**: 2-3 hours

**Total**: ~2 days of development + testing

---

## 💡 Bottom Line

**What makes it scientifically sound?**
1. ✅ **Spatially consistent** normalization (global, not local)
2. ✅ **No redundancy** (removed correlated EAS)
3. ✅ **Transparent** (configurable weights, not black box)
4. ✅ **Validated** (quantitative accuracy metrics)
5. ✅ **Reproducible** (documented parameters)

With these fixes, your wet woodland suitability map will be defensible in peer review!
