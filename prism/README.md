# PRISM Package

**`pip install prism-engine`**

Core Python package for PRISM behavioral signal topology analysis.

---

## Package Structure

```
prism/
├── db/                  # Database utilities (Parquet-based)
│   ├── parquet_store.py # Path management, directory structure
│   ├── polars_io.py     # Atomic writes, upsert operations
│   ├── query.py         # Table introspection utilities
│   └── scratch.py       # TempParquet for parallel workers
│
├── config/              # Configuration
│   └── windows.py       # Domain-aware window configs
│
├── engines/             # 33 measurement engines
│   ├── hurst.py         # Hurst exponent (memory)
│   ├── entropy.py       # Sample/permutation entropy
│   ├── garch.py         # Volatility modeling
│   ├── rqa.py           # Recurrence quantification
│   ├── lyapunov.py      # Chaos indicator
│   ├── pca.py           # Principal components
│   ├── granger.py       # Granger causality
│   └── ...              # 14 more engines
│
├── entry_points/        # CLI entry points (15 total)
│   ├── fetch.py         # Data fetching to Parquet
│   ├── characterize.py  # 6-axis indicator classification
│   ├── indicator_vector.py # Layer 1: Vector metrics
│   ├── laplace.py       # Layer 2: Laplace field
│   ├── geometry.py      # Layer 3: Cohort geometry
│   ├── state.py         # Layer 4: State derivation
│   └── ...
│
├── modules/             # Reusable computation (NOT entry points)
│   ├── characterize.py  # Inline characterization
│   ├── laplace.py       # Laplace field computation
│   ├── modes.py         # Mode discovery from signatures
│   └── wavelet_microscope.py # Frequency-band degradation
│
├── cohorts/             # Cohort definitions
│   ├── climate.py       # Climate cohorts
│   └── cheme.py         # Chemical engineering cohorts
│
└── utils/               # Shared utilities
    ├── stride.py        # Window/stride configuration
    └── monitor.py       # Progress monitoring
```

---

## Usage

### As a Library

```python
# Import engines
from prism.engines import (
    compute_hurst, compute_entropy, compute_rqa,
    PCAEngine, GrangerEngine,
    VECTOR_ENGINES, list_engines
)

# Compute metrics on sensor data
import numpy as np
values = np.random.randn(500)
hurst = compute_hurst(values)
print(hurst['hurst_exponent'])
```

### Database Access

```python
import polars as pl
from prism.db.parquet_store import get_parquet_path

# Read observations
observations = pl.read_parquet(get_parquet_path('raw', 'observations'))

# Query data
df = observations.filter(pl.col('indicator_id') == 'sensor_T30')
```

### CLI Entry Points

```bash
# Fetch industrial data
python -m prism.entry_points.fetch --cmapss
python -m prism.entry_points.fetch --femto

# Run pipeline
python -m prism.entry_points.indicator_vector --domain cmapss
python -m prism.entry_points.laplace --domain cmapss
python -m prism.entry_points.geometry --domain cmapss
```

---

## Pipeline Architecture

```
fetch → characterize
  → indicator_vector → laplace → geometry → state
```

**Key:** Observations → Vector Metrics → Laplace Field → Geometry → State

---

## Engine Categories

| Category | Count | Purpose | Interface |
|----------|-------|---------|-----------|
| **Vector** | 9 | Single-indicator intrinsic properties | `f(array) -> dict` |
| **Geometry** | 9 | Multi-indicator relational structure | `Engine.compute(matrix)` |
| **State** | 7 | Temporal dynamics and causality | `Engine.compute(x, y)` |
| **Temporal Dynamics** | 5 | Geometry evolution over time | `Engine.compute(snapshots)` |
| **Observation** | 3 | Discontinuity detection | `f(array) -> dict` |

See [engines/README.md](engines/README.md) for detailed documentation.

---

## Validated Domains

| Domain | Source | Sensors | Use Case |
|--------|--------|---------|----------|
| **C-MAPSS** | NASA | 25 | Turbofan run-to-failure |
| **FEMTO** | PHM Society | 2 | Bearing degradation |
| **Hydraulic** | UCI | 17 | System condition monitoring |
| **CWRU** | CWRU | 4 | Bearing fault diagnosis |
| **TEP** | Tennessee Eastman | 52 | Chemical process faults |

---

## Storage Schema (v2.0 - Parquet)

```
data/{domain}/
├── raw/                # Layer 0: Raw observations
│   ├── observations.parquet
│   └── characterization.parquet
├── vector/             # Layer 1-2: Vector metrics + Laplace field
│   ├── indicator.parquet
│   └── indicator_field.parquet
├── geometry/           # Layer 3: Structural snapshots
│   ├── cohort.parquet
│   └── indicator_pair.parquet
└── state/              # Layer 4: Temporal dynamics
    └── indicator.parquet
```

---

## Dependencies

**Required:**
- polars >= 0.20.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- pyyaml >= 6.0
- antropy >= 0.1.6
- arch >= 6.0
- nolds >= 0.5.2
- pyrqa >= 8.0.0
- PyWavelets >= 1.4.0

**Optional:**
- requests (for fetching)

---

## Version

**2.0.0** — Pure Polars + Parquet (no DuckDB)
