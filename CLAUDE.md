# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PRISM Diagnostics is a behavioral geometry engine for industrial signal topology analysis. It computes intrinsic properties, relational structure, and temporal dynamics of sensor data from turbofans, bearings, hydraulic systems, and chemical processes.

**Repository:** `prism-engines/diagnostics`

**Architecture: Pure Polars + Parquet**
- All storage via Parquet files (no database)
- All I/O via Polars DataFrames
- Pandas only at engine boundaries (scipy/sklearn compatibility)
- Data stays local (gitignored), only code goes to GitHub

**Core Philosophy: Do It Right, Not Quick**
- Correctness over speed - a wrong answer fast is still wrong
- Complete data, not samples - academic-grade analysis requires full datasets
- Verify before proceeding - check results match expectations
- Run the full pipeline - Vector → Geometry → State → ML

**Design Principles:**
- Record observations faithfully
- Persist all measurements to Parquet
- Explicit time (nothing inferred between steps)
- No implicit execution (importing does nothing)

**Academic Research Standards:**
- **NO SHORTCUTS** - All engines use complete data (no subsampling)
- **NO APPROXIMATIONS** - Peer-reviewed algorithms (antropy, pyrqa)
- **NO SPEED HACKS** - 2-3 hour runs acceptable, 2-3 week runs expected
- **VERIFIED QUALITY** - All engines audited for data integrity
- **Publication-grade** - Suitable for peer-reviewed research

## Directory Structure

```
prism-engines/diagnostics/
├── prism/                      # Core package
│   ├── core/                   # Types and utilities
│   │   ├── domain_clock.py     # DomainClock, DomainInfo, auto_detect_window
│   │   └── signals/            # Signal types (DenseSignal, SparseSignal, LaplaceField)
│   │
│   ├── db/                     # Parquet I/O layer
│   │   └── parquet_store.py    # 5 core files + ML files
│   │
│   ├── engines/                # 33 computation engines
│   │   ├── vector/             # Intrinsic metrics (hurst, entropy, garch, etc.)
│   │   ├── geometry/           # Structural (pca, mst, clustering, coupling, modes)
│   │   ├── state/              # Temporal dynamics (granger, dtw, trajectory, etc.)
│   │   ├── laplace/            # Laplace transform and pairwise
│   │   ├── spectral/           # Wavelet microscope
│   │   ├── pointwise/          # Derivatives, hilbert, statistical
│   │   └── observation/        # Break detector, heaviside, dirac
│   │
│   ├── entry_points/           # CLI entrypoints (python -m prism.entry_points.*)
│   │   ├── fetch.py            # Data fetching
│   │   ├── cohort.py           # Cohort discovery
│   │   ├── signal_vector.py    # Vector computation
│   │   ├── geometry.py         # Geometry computation
│   │   ├── state.py            # State computation
│   │   ├── ml_features.py      # ML feature generation
│   │   └── ml_train.py         # ML model training
│   │
│   └── utils/                  # Utilities (including monitor.py)
│
├── fetchers/                   # Data fetchers
│   ├── cmapss_fetcher.py       # NASA C-MAPSS turbofan
│   ├── femto_fetcher.py        # FEMTO bearing degradation
│   ├── hydraulic_fetcher.py    # UCI hydraulic system
│   ├── cwru_bearing_fetcher.py # CWRU bearing faults
│   ├── tep_fetcher.py          # Tennessee Eastman Process
│   └── yaml/                   # Fetch configurations
│
├── config/                     # YAML configurations
│   ├── engine.yaml             # Engine settings
│   ├── window.yaml             # Window/stride settings
│   ├── stride.yaml             # Legacy stride config
│   └── domain.yaml             # Active domain metadata
│
└── data/                       # LOCAL ONLY (gitignored)
    ├── observations.parquet    # Raw sensor data
    ├── vector.parquet          # Behavioral metrics
    ├── geometry.parquet        # Structural snapshots
    ├── state.parquet           # Temporal dynamics
    ├── cohorts.parquet         # Entity groupings
    ├── ml_features.parquet     # ML-ready features
    ├── ml_results.parquet      # Model predictions
    └── ml_model.pkl            # Trained model
```

## Essential Commands

### Full Pipeline
```bash
# 1. Fetch data (interactive picker or specify source)
python -m prism.entry_points.fetch
python -m prism.entry_points.fetch cmapss

# 2. Compute vector metrics
python -m prism.entry_points.signal_vector

# 3. Compute geometry
python -m prism.entry_points.geometry

# 4. Compute state
python -m prism.entry_points.state

# 5. Generate ML features
python -m prism.entry_points.ml_features --target RUL

# 6. Train ML model
python -m prism.entry_points.ml_train --model xgboost
```

### Testing Mode
All entry points support `--testing` flag for quick iteration:
```bash
python -m prism.entry_points.signal_vector --testing --limit 100
python -m prism.entry_points.geometry --testing
python -m prism.entry_points.state --testing
```

### Common Flags
| Flag | Description |
|------|-------------|
| `--adaptive` | Auto-detect window size from data |
| `--force` | Clear progress and recompute all |
| `--testing` | Enable testing mode (required for --limit, --signal) |
| `--limit N` | [TESTING] Max observations per signal |
| `--signal x,y` | [TESTING] Only process specific signals |

## Pipeline Architecture

```
Layer 0: OBSERVATIONS
         Raw sensor data
         Output: data/observations.parquet

Layer 1: VECTOR
         Raw observations → 51 behavioral metrics per signal
         Output: data/vector.parquet

Layer 2: GEOMETRY
         Vector signals → Laplace fields → structural geometry
         Output: data/geometry.parquet

Layer 3: STATE
         Geometry evolution → temporal dynamics
         Output: data/state.parquet

Layer 4: ML ACCELERATOR
         All layers → denormalized features → trained model
         Output: data/ml_features.parquet, data/ml_model.pkl
```

## ML Accelerator

The ML Accelerator provides end-to-end ML workflow on PRISM features:

### Generate Features
```bash
# Basic feature generation
python -m prism.entry_points.ml_features

# With target variable for supervised learning
python -m prism.entry_points.ml_features --target RUL
python -m prism.entry_points.ml_features --target fault_type
```

### Train Models
```bash
# Train with XGBoost (default)
python -m prism.entry_points.ml_train

# Choose framework
python -m prism.entry_points.ml_train --model catboost
python -m prism.entry_points.ml_train --model lightgbm
python -m prism.entry_points.ml_train --model randomforest

# Hyperparameter tuning
python -m prism.entry_points.ml_train --tune

# Cross-validation
python -m prism.entry_points.ml_train --cv 5
```

### Supported Models
- **xgboost**: XGBoost (default, fast, robust)
- **catboost**: CatBoost (handles categoricals well)
- **lightgbm**: LightGBM (fastest for large data)
- **randomforest**: Scikit-learn Random Forest
- **gradientboosting**: Scikit-learn Gradient Boosting

### Outputs
- `ml_features.parquet`: Denormalized feature table (one row per entity)
- `ml_results.parquet`: Predictions vs actuals for test set
- `ml_importance.parquet`: Feature importance rankings
- `ml_model.pkl`: Serialized trained model

## Engine Categories

**Vector Engines (9)** - Intrinsic properties of single series
- Hurst, Entropy, GARCH, Wavelet, Spectral, Lyapunov, RQA, Realized Vol, Hilbert

**Geometry Engines (9)** - Structural relationships
- PCA, MST, Clustering, LOF, Distance, Convex Hull, Copula, Mutual Information, Barycenter

**State Engines (7)** - Temporal dynamics
- Granger, Cross-Correlation, Cointegration, DTW, DMD, Transfer Entropy, Coupled Inertia

**Temporal Dynamics (5)** - Geometry evolution
- Energy Dynamics, Tension Dynamics, Phase Detector, Cohort Aggregator, Transfer Detector

**Observation Engines (3)** - Discontinuity detection
- Break Detector, Heaviside, Dirac

## Key Patterns

### Reading Data
```python
import polars as pl
from prism.db.parquet_store import get_path, OBSERVATIONS, VECTOR

observations = pl.read_parquet(get_path(OBSERVATIONS))
filtered = observations.filter(pl.col('signal_id') == 'sensor_1')

vector = pl.read_parquet(get_path(VECTOR))
```

### Writing Data
```python
from prism.db.polars_io import upsert_parquet, write_parquet_atomic
from prism.db.parquet_store import get_path, VECTOR

# Upsert (preserves existing rows, updates by key)
upsert_parquet(df, get_path(VECTOR), key_cols=['signal_id', 'timestamp'])

# Atomic write (replaces entire file)
write_parquet_atomic(df, get_path(VECTOR))
```

## Validated Domains

| Domain | Source | Use Case |
|--------|--------|----------|
| **C-MAPSS** | NASA | Turbofan engine degradation (FD001-FD004) |
| **FEMTO** | PHM Society | Bearing degradation (PRONOSTIA) |
| **Hydraulic** | UCI | Hydraulic system condition monitoring |
| **CWRU** | Case Western | Bearing fault classification |
| **TEP** | Tennessee Eastman | Chemical process fault detection |
| **MetroPT** | Metro do Porto | Train compressor failures |

## Technical Stack

- **Language:** Python 3.10+
- **Storage:** Parquet files (columnar, compressed)
- **DataFrame:** Polars (primary), Pandas (engine compatibility)
- **Core:** NumPy, SciPy, scikit-learn
- **ML:** XGBoost, CatBoost, LightGBM (optional)
- **Specialized:** antropy, nolds, pyrqa, arch, PyWavelets, networkx
