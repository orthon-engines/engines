# ORTHON Quick Reference Cheatsheet

## Installation

```bash
# Development install
pip install -e .

# With ML frameworks
pip install -e ".[ml]"

# Verify
python -c "import orthon; print(orthon.__version__)"
orthon --version
```

---

## Core Imports

```python
import orthon
import numpy as np
import polars as pl

# Check version
print(orthon.__version__)  # 0.1.0

# List engines
orthon.list_vector_engines()    # ['entropy', 'garch', 'hilbert_amplitude', ...]
orthon.list_geometry_engines()  # ['barycenter', 'clustering', 'copula', ...]
orthon.list_state_engines()     # ['cointegration', 'cross_correlation', ...]
```

---

## Vector Engines (Single Signal)

| Engine | Function | Use Case |
|--------|----------|----------|
| **hurst** | `compute_hurst(signal)` | Long-range dependence |
| **entropy** | `compute_entropy(signal)` | Information content |
| **garch** | `compute_garch(signal)` | Volatility clustering |
| **wavelet** | `compute_wavelets(signal)` | Multi-scale analysis |
| **spectral** | `compute_spectral(signal)` | Frequency content |
| **rqa** | `compute_rqa(signal)` | Recurrence patterns |
| **lyapunov** | `compute_lyapunov(signal)` | Chaos detection |
| **realized_vol** | `compute_realized_vol(signal)` | Volatility metrics |
| **hilbert_**** | `compute_hilbert_*(signal)` | Instantaneous properties |

### Example

```python
signal = np.random.randn(1000)

# Hurst exponent
h = orthon.compute_hurst(signal)
print(f"Hurst: {h['hurst_exp']:.3f}")

# Entropy
e = orthon.compute_entropy(signal)
print(f"Sample entropy: {e['sample_entropy']:.3f}")

# GARCH
g = orthon.compute_garch(signal)
print(f"GARCH alpha: {g['garch_alpha']:.4f}")
```

---

## Geometry Engines (Multi-Signal)

| Engine | Class | Use Case |
|--------|-------|----------|
| **pca** | `PCAEngine()` | Dimensionality, variance |
| **clustering** | `ClusteringEngine(n_clusters=3)` | Signal grouping |
| **distance** | `DistanceEngine()` | Pairwise distances |
| **mutual_information** | `MutualInformationEngine()` | Information coupling |
| **copula** | `CopulaEngine()` | Dependency structure |
| **mst** | `MSTEngine()` | Network topology |
| **lof** | `LOFEngine()` | Outlier detection |
| **convex_hull** | `ConvexHullEngine()` | Boundary geometry |
| **barycenter** | `BarycenterEngine()` | Geometric center |

### Example

```python
# Multi-signal data: (samples, signals)
data = np.random.randn(100, 10)

# PCA
pca = orthon.PCAEngine(n_components=3)
result = pca.compute(data)
print(f"Explained variance: {result['explained_variance_ratio']}")

# Clustering
cluster = orthon.ClusteringEngine(n_clusters=3)
result = cluster.compute(data.T)  # signals as rows
print(f"Labels: {result['labels']}")
```

---

## State Engines (Temporal)

| Engine | Class | Use Case |
|--------|-------|----------|
| **granger** | `GrangerEngine()` | Causal relationships |
| **cointegration** | `CointegrationEngine()` | Long-run equilibrium |
| **cross_correlation** | `CrossCorrelationEngine()` | Lag correlations |
| **dtw** | `DTWEngine()` | Time series similarity |
| **dmd** | `DMDEngine()` | Dynamic modes |
| **transfer_entropy** | `TransferEntropyEngine()` | Information flow |
| **coupled_inertia** | `CoupledInertiaEngine()` | Coupled dynamics |

### Example

```python
ts = np.random.randn(100, 5)

# Granger causality
granger = orthon.GrangerEngine(max_lag=5)
result = granger.compute(ts)
print(f"Causality matrix:\n{result['causality_matrix']}")

# DTW distance
dtw = orthon.DTWEngine()
result = dtw.compute(ts[:, 0], ts[:, 1])
print(f"DTW distance: {result['distance']:.3f}")
```

---

## I/O Operations

```python
# File constants
orthon.OBSERVATIONS  # "observations"
orthon.VECTOR        # "vector"
orthon.GEOMETRY      # "geometry"
orthon.STATE         # "state"

# Path management
path = orthon.get_path(orthon.OBSERVATIONS)  # data/observations.parquet

# Read
df = orthon.read_parquet("data/observations.parquet")

# Write (atomic - safe for concurrent access)
orthon.write_parquet_atomic(df, "data/output.parquet")

# Upsert (update existing, insert new)
orthon.upsert_parquet(df, "data/vector.parquet", key_cols=["signal_id", "timestamp"])

# Append
orthon.append_parquet(new_df, "data/observations.parquet")
```

---

## C-MAPSS RUL Best Configuration

### Feature Set (29 features)

```python
FEATURES = [
    # Base
    "cycle", "dist_top1", "dist_top2", "dist_top3",

    # HD temporal (windows: 10, 20, 30)
    "hd_slope_10", "hd_slope_20", "hd_slope_30",
    "hd_delta_10", "hd_delta_20", "hd_delta_30",
    "hd_std_10", "hd_std_20",
    "hd_curv_20", "hd_curv_30",
    "hd_max_20", "hd_min_20",

    # Sensor slopes
    "s11_slope_20", "s11_delta_20", "s11_std_20",
    "s12_slope_20", "s12_delta_20",
    "s15_slope_20", "s15_delta_20",
    "s11_s12_corr",

    # Transforms
    "cycle_sq", "cycle_log",
    "s11", "s12", "s15",
]
```

### XGBoost Parameters

```python
XGB_PARAMS = {
    'n_estimators': 800,
    'max_depth': 7,
    'learning_rate': 0.015,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 2,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}
```

### Results

| Dataset | RMSE | SOTA | Gap |
|---------|------|------|-----|
| FD001 | 13.36 | 10.82 | +23% |
| FD002 | 15.04 | 11.46 | +31% |

---

## Key Formulas

### Healthy Distance (z-score)

```
hd[signal] = |value - baseline_mean| / baseline_std
```

### Regime-Aware Baseline

```python
for regime in range(n_regimes):
    healthy_data = data[(regime == r) & (life_pct < 0.20)]
    baseline[regime][signal] = {
        'mean': healthy_data[signal].mean(),
        'std': healthy_data[signal].std()
    }
```

### Temporal Features

```python
# Slope (degradation rate)
slope = polyfit(arange(W), hd_history[-W:], 1)[0]

# Delta (change)
delta = hd_current - hd_history[-W]

# Curvature (acceleration)
curv = slope_second_half - slope_first_half
```

---

## Common Patterns

### Load C-MAPSS Data

```python
import pandas as pd
import numpy as np

COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

train = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=COLS)
test = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=COLS)
rul = np.loadtxt('RUL_FD001.txt')
```

### Add RUL Column

```python
max_cycles = train.groupby('unit')['cycle'].max()
train = train.merge(max_cycles.rename('max_cycle'), on='unit')
train['RUL'] = (train['max_cycle'] - train['cycle']).clip(upper=125)
```

### Cluster Operating Conditions

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
op_scaled = scaler.fit_transform(train[['op1', 'op2']])
kmeans = KMeans(n_clusters=6, random_state=42)
train['regime'] = kmeans.fit_predict(op_scaled)
```

---

## File Locations

```
prism-mac/
├── orthon/                    # Package root
│   ├── __init__.py            # Public API
│   └── _internal/             # Implementation
│
├── proof/                     # Validation scripts
│   ├── best_rul_model.py      # Best configuration
│   └── fd002_regime_test.py   # Standalone test
│
├── notebooks/                 # Documentation
│   ├── orthon_complete_documentation.ipynb
│   └── ORTHON_CHEATSHEET.md
│
└── pyproject.toml             # Package config
```

---

## CLI Commands

```bash
# Version
orthon --version

# List engines
orthon --list-engines

# Help
orthon --help
```

---

*Orthon v0.1.0 - Behavioral Geometry Engine*
