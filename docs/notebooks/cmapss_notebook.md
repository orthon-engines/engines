# C-MAPSS Turbofan Analysis

## Dataset Versions

| Domain | Datasets | Units | Indicators | Best CV RMSE | Model |
|--------|----------|-------|------------|--------------|-------|
| **C_MAPSS_v1** | FD001 only | 100 | 2,500 | **7.38** | Lasso |
| **C_MAPSS_v2** | FD001-FD004 | 709 | 17,725 | **12.25** | GBM |

**v1 (FD001):** Single fault mode, single operating condition → cleaner signal, easier prediction

**v2 (Full):** 2 fault modes × 1-6 operating conditions → more variance, harder prediction

---

## Benchmark Protocol

**Split Strategy:** Unit-level train/test split (engines seen in training never appear in test)

**RUL Capping:** None applied (raw RUL values used)

**Evaluation:** Cross-validation RMSE on training set; final test RMSE on held-out engines

**Feature Timing:** All PRISM features computed using rolling windows that do not include future cycles beyond the prediction timestamp. Laplace field gradients use only past observations.

**Note on Retrospective Metrics:** "Lead days" and phase-based analysis in Physics Validation sections use known failure times for explanatory/diagnostic purposes only. These are NOT used as predictive features.

---

## FD001 Benchmark Results (C_MAPSS_v1)

| Method | CV RMSE | Test RMSE | R² | Notes |
|--------|---------|-----------|-----|-------|
| **PRISM + Lasso** | **7.38** | **6.10** | 0.954 | 48 features, no tuning |
| PRISM + Ridge | 8.86 | 8.37 | 0.913 | |
| PRISM + ElasticNet | 9.32 | 8.24 | 0.915 | |
| PRISM + GBM | 11.75 | 7.54 | 0.929 | |
| LSTM (literature) | ~12-13 | — | ~0.85 | Hours to train |
| NASA Benchmark (2008) | — | 7.80 | — | Rule-based, hand-tuned |

---

## Full Dataset Results (C_MAPSS_v2)

| Method | CV RMSE | Test RMSE | R² | Notes |
|--------|---------|-----------|-----|-------|
| **PRISM + GBM** | **12.25** | **11.51** | 0.966 | 20 features |
| PRISM + RF | 12.44 | 12.09 | 0.963 | |
| PRISM + Ridge | 32.23 | 26.82 | 0.817 | |
| PRISM + Lasso | 32.26 | 26.96 | 0.815 | |

**Key Finding:** Linear models excel on clean single-condition data (v1). Tree models handle multi-condition variance better (v2).

---

# FD001 Physics Validation (C_MAPSS_v1)

*Note: This section uses the C_MAPSS_v1 domain (FD001 processed separately). Values differ from FD001 subset within v2 due to different window/stride configuration.*

## Leading Indicators (FD001)

*Retrospective analysis using known failure times — NOT predictive features.*

| Sensor | Avg Lead Days | Std Dev | Shift Magnitude | n_units |
|--------|---------------|---------|-----------------|---------|
| **P15** | **66.2** | 45.3 | 2.66 | 43 |
| T24 | 56.6 | 42.6 | 2.70 | 64 |
| BPR | 56.3 | 47.5 | 2.52 | 61 |
| op1 | 53.7 | 36.6 | 2.69 | 81 |
| Nc | 53.4 | 41.4 | 2.61 | 47 |
| P30 | 51.9 | 41.8 | 2.52 | 58 |
| Ps30 | 51.7 | 50.2 | 2.64 | 54 |
| W32 | 51.7 | 40.3 | 2.45 | 62 |

**Key Finding:** P15 leads by 66 days in FD001. These lead times are diagnostic — they show which sensors respond earliest to degradation.

---

## Degradation Signature (FD001)

*Retrospective phase labeling using known RUL — NOT predictive features.*

### Hurst Exponent by RUL Phase

| Sensor | Healthy | Degrading | Critical | Failing | Δ (H→F) |
|--------|---------|-----------|----------|---------|---------|
| **P15** | 0.682 | 0.726 | 0.834 | **0.902** | **+0.220** |
| NRc | 0.644 | 0.702 | 0.785 | **0.840** | +0.196 |
| Nc | 0.643 | 0.703 | 0.769 | **0.818** | +0.175 |
| Ps30 | 0.629 | 0.649 | 0.714 | **0.805** | +0.176 |
| P30 | 0.635 | 0.661 | 0.698 | **0.760** | +0.125 |
| NRf | 0.668 | 0.655 | 0.693 | **0.753** | +0.085 |
| Nf | 0.647 | 0.640 | 0.677 | **0.747** | +0.100 |
| BPR | 0.606 | 0.623 | 0.665 | **0.736** | +0.130 |

**Key Finding:** P15 shows the largest Hurst increase (+0.220) — becomes strongly persistent near failure.

### Sample Entropy by RUL Phase

| Sensor | Healthy | Degrading | Critical | Failing | Δ (H→F) |
|--------|---------|-----------|----------|---------|---------|
| NRc | 2.291 | 2.066 | 1.765 | **1.522** | -0.769 |
| Nc | 2.182 | 2.030 | 1.786 | **1.540** | -0.641 |
| NRf | 2.519 | 2.464 | 2.420 | **2.033** | -0.486 |
| P30 | 2.277 | 2.226 | 2.046 | **1.813** | -0.464 |
| Ps30 | 2.187 | 2.185 | 2.035 | **1.712** | -0.475 |
| Nf | 2.373 | 2.571 | 2.457 | **2.057** | -0.316 |
| BPR | 2.189 | 2.233 | 2.096 | **1.961** | -0.228 |

**Key Finding:** NRc shows the largest entropy drop (-0.769) — becomes highly deterministic near failure.

---

## Stress Accumulation (FD001 via C_MAPSS_v1)

| Metric | Value |
|--------|-------|
| Total Field Rows | 1,052,195 |
| Avg Gradient | 0.53 |
| Peak Gradient | 1,051 |
| Total Sources | 359,530 |
| Total Sinks | 362,375 |

**Key Finding:** Sources ≈ Sinks in FD001 (balanced). This reflects stable single-condition operation.

---

## Sensor Importance (FD001)

**Importance Score:** 0.4×Memory + 0.3×(1-Stationarity) + 0.2×Complexity + 0.1×Determinism

*Note: RUL excluded from ranking as it is the prediction target, not a feature.*

| Rank | Sensor | Score | Memory | Stationarity |
|------|--------|-------|--------|--------------|
| 1 | **Ps30** | **0.667** | 0.827 | 0.453 |
| 2 | **P15** | **0.661** | 0.769 | 0.362 |
| 3 | phi | 0.656 | 0.819 | 0.466 |
| 4 | T50 | 0.656 | 0.812 | 0.449 |
| 5 | NRf | 0.655 | 0.803 | 0.457 |
| 6 | Nf | 0.651 | 0.797 | 0.476 |
| 7 | T30 | 0.649 | 0.795 | 0.464 |
| 8 | NRc | 0.648 | 0.819 | 0.499 |
| 9 | P30 | 0.647 | 0.808 | 0.488 |
| 10 | farB | 0.643 | 0.791 | 0.510 |

**Top Degradation Sensors (FD001):** Ps30, P15, phi, T50, NRf

---

## Dynamical Class Distribution (FD001)

| Class | Count | % |
|-------|-------|---|
| PERSISTENT_APERIODIC_COMPLEX_STOCHASTIC | 588 | 23.5% |
| OSCILLATORY | 225 | 9.0% |
| STATIONARY_PERSISTENT_APERIODIC_COMPLEX_STOCHASTIC | 218 | 8.7% |
| STATIONARY_APERIODIC_COMPLEX_STOCHASTIC | 172 | 6.9% |
| STATIONARY_PERSISTENT_OSCILLATORY | 145 | 5.8% |

### Periodicity Clarification

**OSCILLATORY classification** requires periodicity > 0.5.

In FD001, 473 indicator time series (19%) have periodicity > 0.5. These are concentrated in:
- **op3** (operational setting): mean periodicity 0.64
- **T2** (inlet temperature): mean periodicity 0.64
- **PCNfR** (corrected fan speed): mean periodicity 0.64

**Degradation sensors** (P30, Ps30, NRf, T30, etc.) have mean periodicity < 0.05 — they show trending behavior, not oscillation.

The OSCILLATORY patterns in FD001 reflect environmental/operational sensors, not degradation dynamics.

---

## FD001 vs Full Dataset Comparison

| Metric | FD001 (v1) | Full (v2) | Interpretation |
|--------|------------|-----------|----------------|
| **Best CV RMSE** | **7.38** | 12.25 | Cleaner signal in single condition |
| Best Model | Lasso | GBM | Linear works better with clean data |
| Lead Time (P15) | 66 days | 71 days | Consistent across datasets |
| Hurst Δ (P15) | +0.220 | +0.086 | Stronger signal in FD001 |
| Entropy Δ (NRc) | -0.769 | -0.192 | Stronger signal in FD001 |
| Sources/Sinks | Balanced | Sources > Sinks | Multi-condition adds expansion |

**Key Insight:** FD001's single operating condition produces cleaner degradation signatures with larger metric deltas.

---

# Physics Validation Results (Full C_MAPSS_v2)

## Leading Indicators

*Retrospective analysis using known failure times — NOT predictive features.*

| Sensor | Avg Lead Days | Std Dev | Shift Magnitude | n_units |
|--------|---------------|---------|-----------------|---------|
| P15 | 71.3 | 58.1 | 2.72 | 518 |
| htBleed | 70.3 | 59.6 | 2.60 | 531 |
| BPR | 70.1 | 55.2 | 2.63 | 525 |
| T30 | 69.3 | 56.7 | 2.63 | 505 |
| T50 | 68.6 | 56.2 | 2.62 | 527 |
| phi | 68.6 | 57.7 | 2.64 | 506 |
| farB | 68.5 | 56.7 | 2.61 | 524 |
| Nf | 68.5 | 57.7 | 2.65 | 534 |

**Key Finding:** P15, htBleed, and BPR show behavioral shifts ~70 days before failure.

---

## Degradation Signature by RUL Phase

*Retrospective phase labeling using known RUL — NOT predictive features.*

### Hurst Exponent (Memory/Persistence)

| Sensor | Healthy | Degrading | Critical | Failing | Δ (H→F) |
|--------|---------|-----------|----------|---------|---------|
| NRc | 0.630 | 0.651 | 0.679 | **0.696** | +0.066 |
| Nc | 0.632 | 0.652 | 0.678 | **0.693** | +0.061 |
| Ps30 | 0.628 | 0.640 | 0.663 | **0.687** | +0.059 |
| P30 | 0.639 | 0.657 | 0.671 | **0.687** | +0.048 |
| P15 | 0.591 | 0.627 | 0.660 | **0.677** | +0.086 |

**Key Finding:** Hurst exponent increases monotonically toward failure. System becomes more persistent as it degrades.

### Sample Entropy (Predictability)

| Sensor | Healthy | Degrading | Critical | Failing | Δ (H→F) |
|--------|---------|-----------|----------|---------|---------|
| P30 | 1.691 | 1.645 | 1.593 | **1.532** | -0.159 |
| Ps30 | 1.643 | 1.598 | 1.563 | **1.487** | -0.156 |
| Nc | 1.517 | 1.452 | 1.393 | **1.329** | -0.188 |
| NRc | 1.491 | 1.417 | 1.363 | **1.299** | -0.192 |

**Key Finding:** Sample entropy decreases toward failure — system becomes more deterministic.

---

## Stress Accumulation (Laplace Field — Full v2)

| Dataset | Rows | Avg Gradient | Peak Gradient | Sources | Sinks |
|---------|------|--------------|---------------|---------|-------|
| FD001 | 2.1M | 0.78 | 4,082 | 727,102 | 684,706 |
| FD003 | 1.3M | 1.15 | 2,689 | 1,339,905 | 1,174,646 |
| FD002 | 7.4M | 716 | 8.5M | 2,388,219 | 2,253,611 |
| FD004 | 8.9M | 1,093 | 8.5M | 3,810,322 | 3,398,836 |

*Note: FD001 values here differ from C_MAPSS_v1 section above because v2 uses different window/stride configuration and processes all datasets together.*

**Key Findings:**
1. Single-condition datasets (FD001/FD003): low, stable gradients
2. Multi-condition datasets (FD002/FD004): 1000x higher gradients from regime switching
3. Sources > Sinks: system expanding in behavioral space toward failure

---

## Sensor Importance Ranking (Full v2)

**Importance Score:** 0.4×Memory + 0.3×(1-Stationarity) + 0.2×Complexity + 0.1×Determinism

*Note: RUL excluded from ranking as it is the prediction target.*

| Rank | Sensor | Score | Memory | Stationarity | Complexity | Determinism |
|------|--------|-------|--------|--------------|------------|-------------|
| 1 | NRf | 0.565 | 0.659 | 0.553 | 0.709 | 0.259 |
| 2 | epr | 0.565 | 0.714 | 0.656 | 0.625 | 0.510 |
| 3 | NRc | 0.557 | 0.675 | 0.580 | 0.697 | 0.214 |
| 4 | Ps30 | 0.556 | 0.677 | 0.594 | 0.703 | 0.230 |
| 5 | P30 | 0.554 | 0.683 | 0.605 | 0.695 | 0.222 |

**Top Degradation Sensors:** NRf, epr, NRc, Ps30, P30

---

## Physics Interpretation

### Degradation Dynamics

1. **Increasing Persistence (Hurst ↑)**
   - Healthy: H ≈ 0.63; Failing: H ≈ 0.69
   - Physics: Degradation creates correlated trends that compound

2. **Decreasing Entropy (Sample Entropy ↓)**
   - Healthy: high randomness; Failing: more deterministic
   - Physics: Failure modes constrain system dynamics

3. **Gradient Accumulation**
   - Sources > Sinks throughout degradation
   - Physics: Degradation increases dimensionality of failure modes

### Subsystem Mapping

| Subsystem | Key Sensors | Degradation Signature |
|-----------|-------------|----------------------|
| HPC | P30, Ps30, T30, NRc | Memory ↑, Entropy ↓ |
| Fan | NRf, BPR, P15 | Early warning, Memory ↑ |
| Combustor | T50, epr, phi | Temperature drift |
| Turbine | Nc, Nf | Speed ratio changes |

---

# PRISM Hybrid: Feature Importance

## Top Features (GBM on v2)

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | source_ratio | 13.6% | % windows with expanding dynamics |
| 2 | gradient_magnitude_std | 12.4% | Variability in rate of change |
| 3 | gradient_magnitude_max | 12.2% | Peak rate of behavioral change |
| 4 | divergence_std | 12.2% | Variability in source/sink behavior |
| 5 | divergence_range | 10.2% | Spread of divergence values |
| 6 | bridge_ratio | 10.0% | % windows near equilibrium |
| 7 | sink_ratio | 6.7% | % windows with contracting dynamics |
| 8 | divergence_min | 6.0% | Minimum divergence (max contraction) |
| 9 | gradient_magnitude_mean | 5.5% | Average rate of change |
| 10 | gradient_magnitude_range | 4.6% | Spread of gradient values |

**Key Insight:** All top features are Laplace field derivatives (gradient, divergence), not raw sensor statistics.

### Why Laplace Features Work

1. **source_ratio** — Engines approaching failure have more "expanding" dynamics
2. **gradient_magnitude** — Higher gradients indicate faster behavioral change
3. **divergence_std** — High variability indicates unstable dynamics
4. **bridge_ratio** — Low bridge ratio indicates engine consistently in source/sink mode

### Feature vs Sensor Space

| Approach | Features | CV RMSE | Interpretability |
|----------|----------|---------|------------------|
| Raw sensors | 21 × 4 stats = 84 | ~15-20 | Low |
| PRISM Laplace | 20 field features | 12.25 | High |
| LSTM | 21 sensors | ~12-13 | None |

---

# Cohort Geometry Results

## Overview

Cohort geometry computes structural metrics across indicator vectors within each cohort-window.

**Processing:**
- Input: 19.7M indicator field rows
- Windows: 8,345
- Cohorts: 98
- Output: 1,696 cohort-window geometry snapshots

## Geometry Metrics

| Metric | Description | Mean | Std |
|--------|-------------|------|-----|
| cohesion | 1/(1 + mean_distance) | 0.454 | 0.056 |
| pca_effective_dim | Intrinsic dimensionality | 1.49 | 0.37 |
| pca_var_pc1 | Variance explained by PC1 | 78.2% | 16.8% |
| distance_mean | Avg pairwise distance | 1.23 | 0.26 |
| silhouette_score | Cluster separation | 0.79 | 0.09 |
| mst_total_weight | MST edge weight | 9.71 | 2.34 |

## Top Cohorts by Cohesion

| Rank | Cohort | Avg Cohesion |
|------|--------|--------------|
| 1 | FD001_U070 | 0.557 |
| 2 | FD001_U035 | 0.496 |
| 3 | FD001_U093 | 0.492 |
| 4 | FD001_U065 | 0.484 |
| 5 | FD001_U013 | 0.481 |

**Key Finding:** FD001 units dominate top cohesion — single-condition operation produces tighter behavioral clusters.

## Cohort Field (Laplace on Geometry)

**Output:** 32,224 cohort field rows

| Field State | Count | % |
|-------------|-------|---|
| Sources | 15,200 | 47.2% |
| Sinks | 13,300 | 41.3% |
| Bridges | 3,724 | 11.6% |

**Key Finding:** More sources than sinks — cohort structure becomes more dispersed as engines degrade.

---

# PRISM 7-Output Characterization

Each time series is characterized with **6 continuous axes** + **1 discontinuity flag**.

| Output | Type | Description |
|--------|------|-------------|
| **ax_stationarity** | [0, 1] | Non-stationary → Stationary |
| **ax_memory** | [0, 1] | Anti-persistent → Persistent |
| **ax_periodicity** | [0, 1] | Aperiodic → Periodic |
| **ax_complexity** | [0, 1] | Simple → Complex |
| **ax_determinism** | [0, 1] | Stochastic → Deterministic |
| **ax_volatility** | [0, 1] | Homoscedastic → Clustered variance |
| **has_discontinuities** | bool | Regime breaks detected |

## Dynamical Class Labels

| Axis | Threshold | Label |
|------|-----------|-------|
| Stationarity ≥ 0.6 | High | `STATIONARY` |
| Memory > 0.65 | High | `PERSISTENT` |
| Periodicity > 0.5 | High | `OSCILLATORY` |
| Complexity > 0.6 | High | `COMPLEX` |
| Determinism > 0.7 | High | `DETERMINISTIC` |
| Volatility > 0.7 | High | `CLUSTERED_VOL` |

---

## Output Files

```
data/C_MAPSS_v2/
├── vector/
│   ├── indicator.parquet         # 19.8M rows (51 metrics per indicator)
│   ├── indicator_field.parquet   # 19.7M rows (Laplace field)
│   └── cohort_field.parquet      # 32,224 rows
├── geometry/
│   └── cohort.parquet            # 1,696 rows
├── summary/
│   ├── hybrid_results.parquet    # Model comparison
│   ├── feature_importance.parquet
│   └── prism_features.parquet    # Feature matrix (700 × 24)
└── raw/
    └── characterization.parquet  # 17,725 indicators

data/C_MAPSS_v1/
├── vector/
│   ├── indicator.parquet
│   └── indicator_field.parquet   # 1.05M rows
├── summary/
│   ├── hybrid_results.parquet
│   └── prism_features.parquet
└── raw/
    └── characterization.parquet  # 2,500 indicators
```
