# Z-Score Audit: Manifold

## Summary
- **Total z-score instances found: 8**
- **Engine primitives affected: 3** (normalization.py, eigendecomp.py, lof.py)
- **Stage runners affected: 2** (sensor_eigendecomp, information_flow/cohort_information_flow via cointegration)
- **Output schemas affected: 4** (state_geometry, signal_geometry, sensor_eigendecomp, information_flow)
- **Config files affected: 1** (sensor_eigendecomp `norm_method` parameter)
- **Rust primitives with normalize parameters: 6** (spectral_entropy, permutation_entropy, cross_correlation, euclidean/manhattan_distance, centrality functions, persistence_entropy, mutual_information) — all are algorithm-intrinsic normalizations, not z-score

---

## Findings

### 1. `manifold/core/state_geometry.py:134` — CRITICAL

**Code:**
```python
from manifold.primitives.individual.normalization import zscore_normalize  # line 30

# line 130-134:
# Z-SCORE NORMALIZE BEFORE SVD
# Without this, features with large variance dominate
# eigenvalues, making them incomparable across time
normalized, _ = zscore_normalize(centered, axis=0)
```

**What it does:** Z-score normalizes the centered signal matrix (signals x features) per-column before computing the covariance matrix and eigendecomposition. Each feature (kurtosis, skewness, spectral_entropy, etc.) is scaled to mean=0, std=1 within the current (cohort, I) group.

**Computation stage:** Stage 03 — state_geometry

**Input data:** `centered = signal_matrix - centroid` — signal feature vectors centered around the state centroid. Features come from signal_vector.parquet (shape: kurtosis/skewness/crest_factor; complexity: permutation_entropy/hurst/acf_lag1; spectral: spectral_entropy/spectral_centroid/band_ratios).

**Output effect:** All eigenvalue-derived columns in `state_geometry.parquet` are computed on the z-scored covariance matrix:
- `eigenvalue_1` through `eigenvalue_5`
- `explained_1` through `explained_5`
- `effective_dim`
- `total_variance` (this is the sum of eigenvalues from the z-scored covariance — **always equals the number of features** since z-scoring makes each feature have variance=1, so the covariance diagonal is all 1s, and total_variance = trace = D)
- `eigenvalue_entropy`, `eigenvalue_entropy_norm`
- `condition_number`
- `ratio_2_1`, `ratio_3_1`

**Downstream consumers:**
- Stage 07 (geometry_dynamics): Computes velocity/acceleration/jerk of `effective_dim`, `eigenvalue_1`, `total_variance` — all derived from z-scored eigenvalues
- Stage 09a (cohort_thermodynamics): Computes temperature from `var(diff(effective_dim))` and entropy from eigenvalue spectrum — effective_dim is derived from z-scored eigenvalues
- Stage 34 (cohort_baseline): Reads state_geometry for baseline comparison
- Stage 35 (observation_geometry): Reads state_geometry for health scoring
- Stage 25 (cohort_vector): Aggregates state_geometry metrics across cohorts
- Stage 26 (system_geometry): Performs eigendecomposition at fleet level on state_geometry outputs

**Why this matters:** Z-scoring before SVD makes total_variance constant (= number of features), which means total_variance cannot track genuine variance changes over time. This directly masks real physics: if a system's eigenvalue spectrum genuinely expands (more variance as degradation progresses), the z-scoring hides this. The `effective_dim` and eigenvalue ratios still vary meaningfully (they capture shape, not magnitude), but absolute eigenvalue magnitudes and total_variance are destroyed.

---

### 2. `manifold/core/signal_geometry.py:102` — CRITICAL

**Code:**
```python
from manifold.primitives.individual.normalization import zscore_normalize  # line 29

# line 75-104:
def normalize_for_svd(matrix: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Z-score normalize features before SVD."""
    svd_exclude = _get_svd_exclude_features()  # ['cv', 'range_ratio', 'window_size']
    keep_idx = [i for i, f in enumerate(feature_names) if f not in svd_exclude]
    if not keep_idx:
        keep_idx = list(range(len(feature_names)))
    matrix = matrix[:, keep_idx]

    normalized, _ = zscore_normalize(matrix, axis=0)  # line 102
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
```

**What it does:** Z-score normalizes the signal feature matrix per feature before computing eigendecomposition for signal-level geometry. Excludes unbounded features (cv, range_ratio) that can explode.

**Computation stage:** Stage 05 — signal_geometry

**Input data:** Signal feature matrix from signal_vector.parquet, grouped by (cohort, I). Each row is a signal's feature vector.

**Output effect:** The `coherence` column in `signal_geometry.parquet` is affected. Coherence measures alignment of each signal to PC1 from the z-scored eigendecomposition. Other output columns (`distance`, `contribution`, `residual`, `magnitude`) are computed on raw data.

**Downstream consumers:**
- Stage 07 (geometry_dynamics): Can optionally compute signal_dynamics from signal_geometry (distance/coherence derivatives)

**Why this matters:** Same concern as state_geometry — the z-scoring ensures features are comparable but masks absolute variance information in the eigendecomposition.

---

### 3. `manifold/core/state/eigendecomp.py:78-81` — CRITICAL

**Code:**
```python
def compute(
    signal_matrix: np.ndarray,
    centroid: np.ndarray = None,
    norm_method: Literal["zscore", "robust", "mad", "none"] = "zscore",  # line 23
    min_signals: int = 3,
) -> dict:
    # line 60-81:
    centroid = np.mean(signal_matrix, axis=0)
    centered = signal_matrix - centroid

    if norm_method == "none":
        normalized = centered
    elif norm_method == "robust":
        q75, q25 = np.percentile(centered, [75, 25], axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr < 1e-10, 1.0, iqr)
        normalized = centered / iqr
    elif norm_method == "mad":
        median = np.median(centered, axis=0)
        mad = np.median(np.abs(centered - median), axis=0)
        mad = np.where(mad < 1e-10, 1.0, mad)
        normalized = (centered - median) / mad
    else:  # zscore (default)
        std = np.std(centered, axis=0)
        std = np.where(std < 1e-10, 1.0, std)
        normalized = centered / std  # line 81

    U, S, Vt = np.linalg.svd(normalized, full_matrices=False)  # line 85
```

**What it does:** The shared eigendecomp primitive used by sensor_eigendecomp. Defaults to z-score normalization before SVD. Supports configurable normalization methods.

**Computation stage:** Stage 20 — sensor_eigendecomp

**Input data:** Rolling window of sensor observation values (rows=time, cols=sensors). This operates on raw sensor values, not derived features.

**Output effect:** All columns in `sensor_eigendecomp.parquet`:
- `effective_dim`, `eff_dim_entropy`
- `eigenvalue_entropy`, `eigenvalue_entropy_normalized`
- `total_variance`, `condition_number`
- `ratio_2_1`, `ratio_3_1`
- `eigenvalue_0` through `eigenvalue_4`
- `explained_ratio_0` through `explained_ratio_2`
- `energy_concentration`

**Downstream consumers:** No downstream stages directly consume sensor_eigendecomp.parquet. It is a terminal output for analysis.

**Why this matters:** The `norm_method` parameter is configurable (default "zscore"), which is good. However, the default behavior z-scores raw sensor values before eigendecomposition, which makes total_variance = number of sensors (constant). The same total-variance-masking concern applies.

---

### 4. `manifold/core/signal/lof.py:63-66` — MODERATE

**Code:**
```python
# line 62-66:
# Normalize embedding for better LOF performance
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std < 1e-10] = 1.0
X_norm = (X - X_mean) / X_std
```

**What it does:** Z-score normalizes the time-delay embedding matrix before computing Local Outlier Factor scores. This is an inline z-score (not using the normalization primitive).

**Computation stage:** Stage 01 — signal_vector (LOF engine, called per signal)

**Input data:** Time-delay embedding of a single signal (n_samples x embedding_dim).

**Output effect:** LOF-derived columns in `signal_vector.parquet`:
- `lof_max`, `lof_mean`, `lof_std`
- `outlier_fraction`, `n_outliers`

**Downstream consumers:** LOF features feed into state_geometry and signal_geometry (as part of the feature vector). The z-scoring is local to the embedding computation and does not propagate the normalized values — only the LOF scores (which are relative density measures) are output.

**Why this matters:** LOW concern. Z-scoring the embedding before LOF is standard ML practice — it ensures dimensions of the embedding are comparable for distance-based outlier detection. The output is LOF scores, not the normalized embedding values.

---

### 5. `manifold/core/pairwise/cointegration.py:88` — MODERATE

**Code:**
```python
# line 86-88:
spread_current = float(residuals[-1])
spread_zscore = spread_current / residual_std if residual_std > 1e-10 else 0.0
```

**What it does:** Computes the current spread (residual from cointegration regression) expressed in standard deviation units. This is a z-score of the most recent residual relative to the full residual distribution.

**Computation stage:** Stage 10 — information_flow; Stage 28 — cohort_information_flow

**Input data:** OLS regression residuals between two signal pairs (y1 - (alpha + beta * y2)).

**Output effect:** `spread_zscore` column in `information_flow.parquet` and `cohort_information_flow.parquet`.

**Downstream consumers:** No downstream stages consume spread_zscore. It is a terminal metric for interpretation.

**Why this matters:** LOW concern. This is a legitimate statistical quantity — expressing the current spread in units of its own standard deviation is the standard way to measure cointegration spread deviation. This is not a preprocessing normalization; it's the output metric itself. The z-score here is mathematically necessary for the cointegration interpretation (how many sigma away from equilibrium).

---

### 6. `manifold/primitives/individual/normalization.py:17-55` — LIBRARY (not directly invoked)

**Code:**
```python
def zscore_normalize(values, axis=0, ddof=0):
    """Z-score normalization: (x - mean) / std"""
    mean = np.nanmean(values, axis=axis, keepdims=True)
    std = np.nanstd(values, axis=axis, ddof=ddof, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)
    normalized = (values - mean) / std
    return normalized, {'method': 'zscore', 'mean': mean, 'std': std}
```

**What it does:** The primitive z-score function. Called by state_geometry (Finding 1) and signal_geometry (Finding 2).

**Computation stage:** Library function — not a stage.

**Additional functions in this module:** `robust_normalize`, `mad_normalize`, `minmax_normalize`, `quantile_normalize`, `normalize` (dispatcher), `inverse_normalize`, `recommend_method`. Only `zscore_normalize` is currently called by production stages.

---

### 7. `manifold/core/normalization.py:36-79` — LIBRARY (not directly invoked by stages)

**Code:**
```python
def compute_zscore(data, axis=0, ddof=0):
    """Z-score normalization (engine layer wrapper)."""
    # Same implementation as primitive zscore_normalize
```

**What it does:** Engine-layer wrapper around the z-score primitive. Exported via `manifold.core.__init__` as `compute_zscore`. **Not currently called** by any production stage.

**Computation stage:** Library function — available but unused.

---

### 8. `manifold/primitives/tests/normalization.py:12-48` — TEST UTILITY

**Code:**
```python
def z_score(data, axis=None, ddof=0):
    """Z-score normalize data."""
    mean = np.nanmean(data, axis=axis, keepdims=True)
    std = np.nanstd(data, axis=axis, ddof=ddof, keepdims=True)
    std = np.where(std == 0, 1, std)
    return (data - mean) / std

def z_score_significance(value, mean, std):
    """Z-score and p-value for a single observation."""
    z = (value - mean) / std
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p
```

**What it does:** Test utility functions for normalization. Exported via `manifold.primitives.__init__` as `z_score` and `z_score_significance`.

**Computation stage:** Test/validation utilities — not used in production pipeline.

---

## Categorization

### Z-scores inside engine primitives (CRITICAL)
1. `manifold/primitives/individual/normalization.py:17` — zscore_normalize function (called by state_geometry and signal_geometry)
2. `manifold/core/state/eigendecomp.py:78-81` — inline zscore in eigendecomp compute (called by sensor_eigendecomp)
3. `manifold/core/signal/lof.py:63-66` — inline zscore of embedding matrix

### Z-scores in stage runner pre-processing (NONE)
No stage runners apply z-score normalization before calling engines. All z-scoring happens inside the engines themselves.

### Z-scores in inter-stage data flow (CRITICAL)
1. **state_geometry → geometry_dynamics → cohort_thermodynamics**: Eigenvalues, effective_dim, total_variance are all computed on z-scored covariance matrices. These z-scored-derived values flow to geometry_dynamics (velocity/acceleration) and cohort_thermodynamics (temperature/entropy/energy).
2. **state_geometry → cohort_vector → system_geometry**: Z-scored eigenvalue metrics from state_geometry are aggregated at the fleet level by cohort_vector and then used as input to system_geometry for fleet-level eigendecomposition.
3. **signal_geometry → geometry_dynamics**: Coherence values (derived from z-scored principal components) can flow to signal dynamics computations.

### Z-scores in aggregation (state_vector, cohort_vector) (NONE)
- `state_vector` (stage 02): Computes centroids via `np.mean()` — no z-scoring.
- `cohort_vector` (stage 25): Aggregates state_geometry metrics — no additional z-scoring applied during aggregation.

### Z-scores in output column naming/labeling only (LOW)
- `eigenvalue_entropy_normalized`: This column name contains "normalized" but it is **entropy normalization** (entropy / max_entropy for [0,1] scale), not z-score normalization. Appears in state_geometry.parquet, sensor_eigendecomp.parquet.
- `spread_zscore`: Legitimate z-score metric in information_flow.parquet.

---

## Data Flow Diagram

### Primary Z-Score Cascade (CRITICAL)

```
observations.parquet (raw sensor values)
    │
    ▼
Stage 01: signal_vector
    ├── LOF engine: z-scores embedding internally (lof.py:63-66)
    │   └── Output: lof_max, lof_mean, etc. (LOF scores, not z-scored values)
    └── Other engines: NO z-scoring
    │
    ▼
signal_vector.parquet (RAW feature values per signal per I)
    │
    ├────────────────────────────────────────────┐
    ▼                                            ▼
Stage 02: state_vector                    Stage 05: signal_geometry
    │ (centroid = mean, NO z-score)           │ normalize_for_svd():
    │                                         │   ★ zscore_normalize(matrix, axis=0)
    ▼                                         │   → cov_matrix → eigendecomp
state_vector.parquet (RAW centroids)          │   → coherence column AFFECTED
    │                                         ▼
    ▼                                    signal_geometry.parquet
Stage 03: state_geometry                      (coherence: z-score derived)
    │ compute_eigenvalues():                  (distance, contribution: RAW)
    │   centered = signal_matrix - centroid
    │   ★ normalized = zscore_normalize(centered, axis=0)
    │   → covariance_matrix(normalized)
    │   → eigendecomposition(cov_matrix)
    │   → ALL eigenvalue metrics derived from z-scored cov
    ▼
state_geometry.parquet
    │ (eigenvalue_1..5: z-score derived)
    │ (effective_dim: z-score derived)
    │ (total_variance: ≈ CONSTANT due to z-score)
    │ (condition_number: z-score derived)
    │
    ├─────────────────────┬──────────────────────┬──────────────────┐
    ▼                     ▼                      ▼                  ▼
Stage 07:            Stage 09a:            Stage 25:          Stage 34/35:
geometry_dynamics    cohort_thermo         cohort_vector      baseline/scoring
    │                    │                      │
    │ d/dt(eff_dim)      │ temperature =        │ aggregate
    │ d/dt(eigenvalue_1) │   var(diff(eff_dim)) │ across cohorts
    │ d/dt(total_var)    │ entropy = Shannon(λ) │
    ▼                    │ energy = mean(eff_dim)▼
geometry_dynamics.pq     ▼                 cohort_vector.pq
                    cohort_thermo.pq            │
                                                ▼
                                           Stage 26:
                                           system_geometry
                                                │ (fleet-level eigendecomp)
                                                ▼
                                           system_geometry.pq
```

### Sensor Eigendecomp (Separate Path)

```
observations.parquet (raw sensor values)
    │
    ▼
Stage 20: sensor_eigendecomp
    │ eigendecomp.compute():
    │   ★ norm_method="zscore" (default, configurable)
    │   → centered / std → SVD
    │   → eigenvalue metrics
    ▼
sensor_eigendecomp.parquet (z-score derived eigenvalues)
    (NO downstream consumers — terminal output)
```

### Cointegration Z-Score (Separate Path)

```
signal values from observations.parquet
    │
    ▼
Stage 10: information_flow / Stage 28: cohort_information_flow
    │ cointegration.compute():
    │   residuals = y1 - (alpha + beta * y2)
    │   ★ spread_zscore = residuals[-1] / std(residuals)
    ▼
information_flow.parquet / cohort_information_flow.parquet
    │ spread_zscore column (terminal metric)
    (NO downstream consumers)
```

---

## Rust Primitives — Normalize Parameters (NOT z-score)

The following Rust primitives have `normalize` parameters, but these are **algorithm-intrinsic normalizations**, not mean/std z-scoring:

| Function | File | What `normalize` does | Z-score? |
|----------|------|-----------------------|----------|
| `spectral_entropy` | `src/individual/spectral.rs:223` | Divides entropy by log2(N) to get [0,1] range | NO |
| `permutation_entropy` | `src/individual/entropy.rs:63` | Divides entropy by log2(order!) to get [0,1] range | NO |
| `cross_correlation` | `src/pairwise/correlation.rs:70` | Divides by sqrt(var_a * var_b) — standard Pearson normalization | NO (inherent to correlation) |
| `euclidean_distance` | `src/pairwise/distance.rs:50` | Divides by sqrt(n) | NO |
| `manhattan_distance` | `src/pairwise/distance.rs:108` | Divides by n | NO |
| `mutual_information` | `src/information/mutual_info.rs:7` | MI / sqrt(H(X)*H(Y)) — normalized MI | NO |
| `centrality_degree` | `src/network/centrality.rs:7` | Divides by (n-1) — standard graph normalization | NO |
| `centrality_betweenness` | `src/network/centrality.rs:70` | Divides by (n-1)(n-2) | NO |
| `centrality_closeness` | `src/network/centrality.rs:192` | Multiplied by reachable/(n-1) | NO |
| `persistence_entropy` | `src/topology/persistence.rs:152` | Divides by ln(n_lifetimes) | NO |
| `laplacian_matrix` | `src/matrix/graph.rs:13` | I - D^{-1/2} A D^{-1/2} (normalized Laplacian) | NO |

**Rust `lyapunov.rs:274-278`:** Uses `std_dev` to compute default epsilon for neighbor search (`epsilon = std_dev * 0.1`). This is a scale-adaptive threshold, not z-score normalization.

---

## Python Primitives — Mean/Std Usage (NOT z-score)

Several Python primitives compute mean and/or std as part of their algorithm, not as preprocessing z-scores:

| Function | File | What mean/std does | Z-score? |
|----------|------|--------------------|----------|
| `correlation_coefficient` | `primitives/individual/similarity.py:92` | `r = cov(x,y)/(std(x)*std(y))` via np.corrcoef | NO (inherent to Pearson) |
| `cross_correlation` | `primitives/individual/similarity.py:235` | Mean-center + divide by std*std*n | NO (inherent to XCF) |
| `phase_coherence` | `core/signal/phase_coherence.py:39` | `y = y - np.mean(y)` — DC removal for Hilbert transform | NO (centering only) |
| `rate_of_change` | `core/signal/rate_of_change.py:74` | `np.std(dy)` — computes rate_std as output metric | NO (descriptive statistic) |
| `trend` | `core/signal/trend.py:65` | `cusum_range / std_y` — scale normalization only | NO (no centering) |
| `eigenvalue_spread` | `primitives/individual/geometry.py:293` | `std(eigenvals) / mean(eigenvals)` — coefficient of variation | NO (CV is a ratio) |
| `compute_effective_temperature` | `stages/dynamics/cohort_thermodynamics.py:62` | `np.var(velocities)` — temperature proxy | NO (descriptive stat) |

---

## Configuration Files

### `manifold/stages/geometry/sensor_eigendecomp.py:87`
```python
norm_method: str = "zscore",  # default parameter
```
And CLI argument at line 330-332:
```python
parser.add_argument('--norm', default='zscore',
                    choices=['zscore', 'robust', 'mad', 'none'],
                    help='Normalization method (default: zscore)')
```

**This is the only configurable z-score parameter in the pipeline.** All other z-score instances are hardcoded.

### `manifold/core/signal/hurst.yaml`
No z-score related parameters found. This file contains hurst exponent computation parameters.

### `config/defaults/pairwise.yaml`
```yaml
distance:
  normalize_by_features: false
```
Distance normalization is disabled. Not z-score related.

---

## Critical Cascade Analysis

### The total_variance Problem

When state_geometry z-scores before computing the covariance matrix:
1. Each feature is scaled to std=1
2. The covariance matrix diagonal becomes ~1 for each feature
3. `total_variance = sum(eigenvalues) = trace(covariance) ≈ D` (number of features)
4. Total variance is approximately **constant** regardless of the actual signal variance

**This means:**
- `total_variance` in state_geometry.parquet does not track genuine variance changes
- `variance_velocity` in geometry_dynamics.parquet is approximately zero
- Any downstream computation that depends on total_variance magnitude is compromised

### What IS preserved by z-scoring:
- `effective_dim` (ratio of eigenvalues — shape, not magnitude)
- `ratio_2_1`, `ratio_3_1` (eigenvalue ratios)
- `eigenvalue_entropy` (distribution shape)
- `condition_number` (ratio of max/min eigenvalue)

### What is LOST by z-scoring:
- Absolute eigenvalue magnitudes
- Total variance trends over time
- Variance velocity/acceleration
- Genuine eigenvalue spikes from system dynamics

---

## Intentional vs Accidental Normalization

### Intentional (mathematically necessary):
- **Pearson correlation** (similarity.py): r = cov/(std*std) — this IS the definition
- **Cross-correlation** (similarity.py): normalization to [-1,1] is standard
- **Spectral/permutation entropy normalization** (Rust): Division by max_entropy for [0,1] range is standard
- **Cointegration spread_zscore**: Expressing spread in sigma units is the standard cointegration metric
- **LOF z-scoring** (lof.py): Standard preprocessing for distance-based outlier detection

### Preprocessing choice (could be different):
- **state_geometry zscore_normalize** (state_geometry.py:134): Makes features comparable before SVD. Could alternatively use robust_normalize, mad_normalize, or correlation matrix instead of covariance matrix
- **signal_geometry normalize_for_svd** (signal_geometry.py:102): Same rationale as state_geometry
- **sensor_eigendecomp norm_method** (eigendecomp.py:78): Default "zscore" but configurable to "robust", "mad", or "none"
