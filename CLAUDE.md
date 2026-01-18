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
- Run the full pipeline - Vector → Geometry → Mode → State

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

**ML Accelerator Benchmarks:**
- Always train on train set, test on test set
- Compare predictions with ground truth RUL
- Use full PRISM pipeline (Vector + Geometry + Laplace + Mode)
- Report RMSE against published benchmarks

## Directory Structure

```
prism-engines/diagnostics/
├── prism/                      # Core package
│   ├── db/                     # Parquet I/O layer
│   ├── engines/                # 33 computation engines
│   ├── entry_points/           # CLI entrypoints (python -m prism.entry_points.*)
│   ├── modules/                # Reusable computation modules
│   ├── cohorts/                # Cohort definitions
│   ├── state/                  # State tracking
│   └── utils/                  # Utilities (including monitor.py)
│
├── fetchers/                   # Data fetchers (16 total)
│   ├── cmapss_fetcher.py       # NASA C-MAPSS turbofan
│   ├── femto_fetcher.py        # FEMTO bearing degradation
│   ├── hydraulic_fetcher.py    # UCI hydraulic system
│   ├── cwru_bearing_fetcher.py # CWRU bearing faults
│   ├── tep_fetcher.py          # Tennessee Eastman Process
│   └── yaml/                   # Fetch configurations
│
├── config/                     # YAML configurations
│   ├── stride.yaml             # Window/stride settings
│   ├── normalization.yaml      # Normalization per domain
│   └── cohorts/                # Cohort definitions
│
├── scripts/                    # Evaluation/testing scripts
│
├── docs/                       # Documentation
│   ├── notebooks/              # Jupyter notebooks & analysis
│   │   ├── ml_accelerator/     # ML benchmark scripts
│   │   └── cmapss/             # C-MAPSS analysis
│   └── validation/             # Validation studies
│
└── data/                       # LOCAL ONLY (gitignored)
    ├── raw/                    # Raw observations
    ├── vector/                 # Computed metrics
    ├── geometry/               # Structural snapshots
    ├── state/                  # Temporal dynamics
    └── [domain]/               # Domain-specific data (cmapss, femto, etc.)
```

## Essential Commands

### Data Fetching
```bash
# Fetch C-MAPSS turbofan data
python -m prism.db.fetch --cmapss

# Fetch FEMTO bearing data
python -m prism.db.fetch --femto

# Fetch hydraulic system data
python -m prism.db.fetch --hydraulic
```

### Vector Computation
```bash
# Run vector engines on all signals
python -m prism.entry_points.signal_vector

# Specific domain
python -m prism.entry_points.signal_vector --domain cmapss

# Parallel execution
python -m prism.entry_points.signal_vector --workers 4
```

### Geometry & State
```bash
# Compute geometry
python -m prism.entry_points.geometry --domain cheme

# Compute Laplace field
python -m prism.entry_points.laplace --domain cheme

# Cohort state
python -m prism.entry_points.cohort_state --domain cheme
```

### Monitoring
```bash
# Monitor long-running jobs
python -m prism.utils.monitor
```

## Pipeline Architecture

```
Layer 0: OBSERVATIONS
         Raw sensor data → signal topology
         Output: data/raw/observations.parquet

Layer 1: SIGNAL VECTOR
         Raw observations → 51 behavioral metrics per signal
         Output: data/vector/signal.parquet

Layer 2: COHORT GEOMETRY
         Signal vectors → pairwise relationships + cohort structure
         Output: data/geometry/cohort.parquet

Layer 3: STATE
         Temporal dynamics, transitions, regime tracking
         Output: data/state/cohort.parquet

REGIME CHANGE = geometric deformation at any layer
```

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
from prism.db.parquet_store import get_parquet_path

observations = pl.read_parquet(get_parquet_path('raw', 'observations'))
filtered = observations.filter(pl.col('signal_id') == 'sensor_1')
```

### Writing Data
```python
from prism.db.polars_io import upsert_parquet, write_parquet_atomic

# Upsert (preserves existing rows, updates by key)
upsert_parquet(df, target_path, key_cols=['signal_id', 'obs_date'])

# Atomic write (replaces entire file)
write_parquet_atomic(df, target_path)
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
- **Specialized:** antropy, nolds, pyrqa, arch, PyWavelets, networkx
