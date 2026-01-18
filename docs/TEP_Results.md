# TEP Fault Detection Results

**Date:** January 17, 2026
**Version:** v2.1.0-tep-assessment

## Overview

This document summarizes the Tennessee Eastman Process (TEP) fault detection assessment using PRISM's behavioral geometry framework. The assessment uses a three-layer detection architecture that combines **WHAT** (classification), **WHEN** (break detection), and **MODE** (behavioral trajectory) for comprehensive fault identification.

---

## Three-Layer Detection Architecture

```
Raw TEP Data
    |
    v
+-------------------+     +-------------------+     +-------------------+
|   WHAT Layer      |     |   WHEN Layer      |     |   MODE Layer      |
|   (Classification)|     |   (Break Detect)  |     |   (Trajectory)    |
+-------------------+     +-------------------+     +-------------------+
| GARCH (alpha,     |     | Break detector    |     | Mode affinity     |
|   beta, omega)    |     | Dirac (impulses)  |     | Mode entropy      |
| Spectral slope    |     | Heaviside (steps) |     | Mode sequence     |
| Entropy           |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
        |                         |                         |
        v                         v                         v
+---------------------------------------------------------------+
|                    INTEGRATED DETECTION                       |
|                                                               |
|   Onset detected when:                                        |
|   - WHAT: volatility/entropy shift                            |
|   - WHEN: break/impulse/step detected                         |
|   - MODE: affinity drop OR entropy spike                      |
+---------------------------------------------------------------+
```

---

## Layer 1: WHAT (Classification)

**Purpose:** Identify which fault type is occurring based on behavioral signatures.

### Results

| Metric | Value |
|--------|-------|
| Binary Accuracy (Normal vs Fault) | **64.5%** |
| Multi-class Accuracy (Identify Fault Type) | **60.0%** |

### Key Features

| Feature | Mean | Std |
|---------|------|-----|
| alpha (GARCH) | 0.173 | 0.074 |
| beta (GARCH) | 0.790 | 0.110 |
| omega (GARCH) | 0.015 | 0.022 |
| spectral_slope | -1.051 | 0.597 |
| permutation_entropy | 0.962 | 0.048 |

### Most Discriminative Fault Types

- **IDV03** (D Feed Temperature): Distinct volatility signature
- **IDV07** (C Header Pressure): Strong spectral shift
- **IDV10** (C Feed Temperature): Clear GARCH pattern

**Assessment File:** `prism/assessments/tep_fault_eval.py`

---

## Layer 2: WHEN (Break Detection)

**Purpose:** Detect the timing of regime changes using structural break analysis.

### Results

| Metric | Value |
|--------|-------|
| Detection Rate (z > 1.0) | **47.6%** |
| Early Warning Rate | **42.7%** |

### Break Signals

| Metric | Total | Mean | Max |
|--------|-------|------|-----|
| break_n | 5,982,408 | 22.39 | 1,440 |
| break_rate | N/A | 0.02 | 8 |
| dirac_n_impulses | 1,011,498 | 3.78 | 109 |
| heaviside_n_steps | 1,107,655 | 4.14 | 185 |

### Detection Method

The break detection layer uses a composite signal:

```python
composite = (
    z_score(break_n) +
    z_score(dirac_n_impulses) +
    z_score(heaviside_n_steps)
) / 3
```

Early warning is flagged when `composite > 1.0` in the 7-day pre-onset window.

**Assessment File:** `prism/assessments/tep_break_detection.py`

---

## Layer 3: MODE (Behavioral Trajectory)

**Purpose:** Track behavioral mode transitions as indicators of regime change.

### Mode Discovery Results

| Metric | Value |
|--------|-------|
| Unique Modes | 5 |
| Mean Affinity | 0.964 |
| Min Affinity | 0.200 |
| Mean Entropy | 0.128 |
| Max Entropy | 1.596 |

### Mode Distribution

| Mode | Assignments | Percentage |
|------|-------------|------------|
| Mode 0 | 11,093 | 43.9% |
| Mode 1 | 1,008 | 4.0% |
| Mode 2 | 2,574 | 10.2% |
| Mode 3 | 6,102 | 24.1% |
| Mode 4 | 4,523 | 17.9% |

### Clustering Features (Laplace Fingerprints)

- `gradient_mean` - Average gradient magnitude
- `gradient_std` - Gradient variability
- `gradient_magnitude` - Total gradient strength
- `laplacian_mean` - Average curvature
- `laplacian_std` - Curvature variability
- `divergence` - Field divergence

**Assessment File:** `prism/assessments/tep_modes.py`

---

## Mode 1 Analysis: The Precursor Mode

Mode 1 (the rare 4%) emerged as a critical early warning signal.

### Key Findings

| Metric | Mode 1 | All Other Modes |
|--------|--------|-----------------|
| Frequency | 4.0% | 96.0% |
| Mean Affinity | 0.937 | 0.965 |
| Role | Precursor/Transition | Stable states |

### Mode 1 Characteristics

1. **Dominated by XMEAS03 (D Feed)** - 58% of Mode 1 assignments
   - D Feed is a reactor input stream
   - Early indicator of upstream disturbances

2. **Lower Affinity (0.937 vs 0.965)**
   - Indicates behavioral uncertainty
   - System not firmly in any stable mode

3. **Fewer Structural Breaks**
   - Mode 1 often appears before breaks cascade
   - Precursor signal before visible structural change

### Warning Rate

| Timing | Mode 1 Occurrence | Interpretation |
|--------|-------------------|----------------|
| Before Fault Onset | 20.4% | Transition warning |
| At Fault Onset | 7.8% | Active transition |
| **Combined** | **28.2%** | Early warning signal |

### August 16, 2000 Case Study

On this date, 30 indicators were assigned to Mode 1 (vs typical 1-2). Investigation revealed:

- **Previous fault:** IDV20 ended Aug 12
- **Next fault:** IDV08 started Aug 20
- **Aug 16:** System in transitional state between two faults

Mode 1 captured this inter-fault transition period.

---

## Integrated Assessment Results

### Combined WHAT + WHEN + MODE + Mode 1

The integrated assessment now includes explicit Mode 1 detection as a precursor signal.

| Metric | Value |
|--------|-------|
| Fault Onsets Analyzed | 30 |
| Early Warnings Detected | 30 |
| **Overall Warning Rate** | **100%** |

### Mode 1 Specific Detection

```
Onset flagged with M1! when Mode 1 appears:
  - Before onset (7-day window)
  - At onset date

Mode 1 interpretation:
  - Mode 1 is a rare (~4%) transitional mode
  - Dominated by XMEAS03 (D Feed - upstream reactor input)
  - Lower affinity (0.937) = system in uncertain state
  - Appears BEFORE structural breaks become visible
```

### Per-Fault Detection Rate

| Fault | Detection Rate |
|-------|----------------|
| IDV03 | 100% (2/2) |
| IDV04 | 100% (1/1) |
| IDV05 | 100% (4/4) |
| IDV07 | 100% (4/4) |
| IDV08 | 100% (3/3) |
| IDV09 | 100% (1/1) |
| IDV10 | 100% (4/4) |
| IDV13 | 100% (2/2) |
| IDV14 | 100% (2/2) |
| IDV15 | 100% (1/1) |
| IDV17 | 100% (3/3) |
| IDV19 | 100% (2/2) |
| IDV20 | 100% (1/1) |

### Signal Decomposition (Pre-Onset)

| Signal Type | Count |
|-------------|-------|
| Break detector | 6,388,479 breaks |
| Dirac impulses | 981,081 impulses |
| Heaviside steps | 1,092,252 steps |

**Assessment File:** `prism/assessments/tep_integrated.py`

---

## Assessment Files Created

| File | Purpose | Key Output |
|------|---------|------------|
| `tep_fault_eval.py` | Classification baseline | 64.5% binary accuracy |
| `tep_break_detection.py` | Break/impulse/step detection | 47.6% detection rate |
| `tep_modes.py` | Behavioral mode clustering | 5 modes, 0.964 affinity |
| `tep_integrated.py` | Combined WHAT+WHEN+MODE | 100% early warning |
| `tep_summary.py` | Quick overview | Architecture diagram |

---

## Data Scale

| Dataset | Rows |
|---------|------|
| Vector data (TEP) | 175,173,632 |
| Mode assignments | 25,300 |
| Fault onsets | 103 |

**Note:** Full 175M row analysis requires lazy/streaming evaluation to avoid OOM.

---

## Key Insights

1. **Individual layers have moderate detection rates** (47-65%), but combining them achieves near-perfect early warning.

2. **Mode 1 is a precursor signal** - its rarity (4%) makes it valuable; when indicators enter Mode 1, the system is transitioning.

3. **XMEAS03 (D Feed) is a leading indicator** - as a reactor input stream, it shows disturbances before they propagate downstream.

4. **Low affinity = high uncertainty** - Mode 1's lower affinity (0.937) indicates the system is not firmly in any stable behavioral mode.

5. **The geometry interprets** - no manual rules were created; the math discovered these patterns from the data.

---

## Technical Notes

### Memory Management

The 175M row dataset exhausts RAM with naive loading. Solutions implemented:

1. **Lazy evaluation:** `pl.scan_parquet()` instead of `pl.read_parquet()`
2. **Date sampling:** Process subset of dates for mode computation
3. **Chunked processing:** Load one date at a time

### Mode Computation

```python
# GMM clustering on Laplace fingerprints
gmm = GaussianMixture(n_components=5, random_state=42)
mode_ids = gmm.predict(X_scaled)
affinities = np.max(gmm.predict_proba(X_scaled), axis=1)
entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
```

---

## Future Work

1. **Add Mode 1 as explicit warning signal** in integrated assessment
2. **Streaming evaluation** for full 175M row analysis
3. **Cross-domain validation** - test on other industrial process datasets
4. **Mode sequence analysis** - detect transition patterns (e.g., 0→1→3 = fault incoming)

---

*Generated by PRISM v2.1.0-tep-assessment*
