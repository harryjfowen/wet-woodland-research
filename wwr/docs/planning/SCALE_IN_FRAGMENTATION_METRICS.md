# Understanding Scale in Fragmentation Metrics

## Your Excellent Question

> "Maybe the fragmentation index needs to be more expansive than just a local radius. Like it should be fragmentation considered across the entirety of the UK right? As each local perspective could look connected so scale matters?"

**You're absolutely correct!** Scale matters enormously in landscape ecology, and different metrics measure fragmentation at different scales.

---

## Metrics by Scale

### 🔍 **Local Scale (Patch-level)**
**Proximity Index** - Only within specified radius (default: 1km, adjustable with `--prox-radius-m`)

```
Patch A ────1km radius────
   │     (looks connected)
   ├─ 100m to Patch B
   ├─ 500m to Patch C
   └─ 800m to Patch D

Proximity = HIGH (locally connected)
```

**Use for**:
- Species dispersal analysis
- Local habitat connectivity
- Ecological neighborhoods

### 🗺️ **Landscape Scale (Study area-wide)**
**Landscape Division Index (LDI)** and **Effective Mesh Size** - Consider ALL patches across ENTIRE study area

```
Entire UK Study Area
├─ Massive patch in Scotland (15 million ha) ─┐
├─ 150,000 tiny patches across England        │ → LDI = 0.006
└─ Few medium patches in Wales                ┘   (looks "connected"!)
```

**These already ARE landscape-scale!** They consider every single patch together.

---

## Why Your Metrics Look "Un-fragmented"

Your results:
```
LDI = 0.0063 (only 0.6% fragmented)
Mesh Size = 150,000 km²
```

This doesn't mean the landscape isn't fragmented. It means:

### ⚠️ **One Massive Patch Dominates Everything**

The formula: `LDI = 1 - Σ(area_i²) / total_area²`

If you have:
- 1 patch of 15 million ha: contributes `(15,000,000)² = 225 trillion`
- 150,000 patches of 0.02 ha each: contribute almost nothing

**The huge patch dominates the calculation**, making it appear "un-fragmented" at landscape scale, even though there are 150,000 isolated fragments!

---

## The Problem: Simpson's Paradox in Landscape Ecology

```
┌─────────────────────────────────────────┐
│  [HUGE PATCH: 99% of area]             │
│                                         │
│  · · · · · ·  ← 150,000 tiny isolated  │
│  · · · · · ·     patches (1% of area)  │
└─────────────────────────────────────────┘

Landscape-scale metrics say: "NOT FRAGMENTED!"
Reality: "Depends on perspective"
```

**For the huge patch**: It's contiguous and well-connected
**For the 150k small patches**: They're isolated and fragmented

Both perspectives are valid, but traditional metrics are area-weighted, so the big patch wins.

---

## New Metrics Added to Address This

The script now shows:

```
Landscape Division Index: 0.0063
Effective Mesh Size: 14,979,983.75 ha
Largest patch: 99.2% of total area  ← NEW!
Top 10% of patches: 99.8% of total area  ← NEW!

⚠ INTERPRETATION: Landscape dominated by 1 massive patch!
   Fragmentation metrics may be misleading. Most area is in one patch,
   but there may be many small isolated patches elsewhere.
```

This tells you:
1. The landscape-scale metrics are dominated by one huge patch
2. True fragmentation story is more complex
3. Need to look at patch size distribution separately

---

## Alternative Approaches for Multi-Scale Analysis

### 1. **Report Both Metrics**

**Area-weighted (current)**:
- LDI, Mesh Size → What dominates by area?
- Biased toward large patches

**Number-weighted**:
- % of patches <5ha: 99.3% → Most patches are tiny!
- Median patch size: 0.02 ha → Typical patch is tiny!

### 2. **Exclude Dominant Patches**

Calculate fragmentation excluding the largest 1-5 patches:

```python
# Remove massive outliers
df_small = df_patches[df_patches['area_ha'] < 1000]
# Recalculate LDI/mesh on just these
```

This shows fragmentation of the "typical" landscape.

### 3. **Multi-Scale Proximity** (what you suggested!)

Instead of just 1km radius, calculate at multiple scales:

```
Proximity at:
- 100m: Very connected (90% have neighbors)
- 1km: Moderately connected (60% have neighbors)
- 10km: Slightly connected (20% have neighbors)
- 100km: Isolated (5% have neighbors)
```

Would you like me to add multi-scale proximity analysis?

### 4. **Grid-Based Fragmentation**

Divide UK into 10km × 10km grid cells, calculate LDI per cell:

```
North Scotland: LDI = 0.01 (connected)
Central England: LDI = 0.95 (fragmented)
Wales: LDI = 0.60 (moderately fragmented)
```

This shows spatial variation in fragmentation.

---

## What Should You Report in Your Paper?

### Option A: Report the Paradox
```
"While landscape-scale metrics suggest low fragmentation (LDI = 0.006),
this reflects dominance by a few large patches (largest = 99% of area).
In contrast, 99% of patches were <5ha, indicating most wet woodland
exists as small isolated fragments."
```

### Option B: Use Number-Weighted Metrics
```
"The median patch size was 0.02ha, with 99.3% of patches smaller than
5ha, indicating high fragmentation despite the presence of a few large
contiguous areas."
```

### Option C: Multi-Scale Analysis
```
"Fragmentation was scale-dependent: locally (1km), patches showed high
connectivity (proximity index = 2.3), but at landscape scale (100km),
most patches were isolated (median nearest neighbor = 5.2km)."
```

---

## Summary

**You were right**: Scale matters!

**The good news**: LDI and Mesh Size ALREADY operate at landscape scale (entire study area)

**The issue**: They're area-weighted, so one huge patch can dominate

**The solution**: Now the script reports:
1. ✅ Landscape-scale metrics (LDI, Mesh)
2. ✅ Patch concentration metrics (largest %, top 10%)
3. ✅ Number-based metrics (median, % <5ha)
4. ✅ Local connectivity (proximity within 1km)
5. ✅ Warnings when large patches dominate

**Next step**: Check your largest patch size! I bet it's enormous and possibly an artifact from bridging or data processing.

---

## Recommended Actions

1. **Check largest patch**: Look at the output showing largest patch size
2. **Disable bridging** if enabled (might be artificially merging patches)
3. **Inspect visually** in QGIS: Load `patch_labels.tif` and see if that huge patch is real
4. **Consider multi-scale** if you want proximity at 10km, 50km, 100km scales (I can add this!)

Would you like me to implement grid-based fragmentation or multi-scale proximity analysis?
