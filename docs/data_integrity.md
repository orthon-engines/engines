# PRISM Core: Data Pipeline Reference

**Version:** 4.0 | **Last Updated:** December 2024

---

## Overview

**prism-core** is a data-only repository. It fetches, measures, and stores signal topology data.

```
FETCH → MEASURE → STORE
```

That's it. No interpretation. No state inference. No dynamics modeling.

> State-dynamics and interpretation live in **prism-dynamics** (separate repo).

---

## Architecture

### Data Flow

```
External APIs          Parquet Layers             Output

  NASA    ───┐         ┌─ raw/observations        Per-signal
  USGS    ───┼─ fetch ─┼─ vector/signal        signal topology
  NOAA    ───┤         ├─ geometry/cohort         measurements
  TEP     ───┘         └─ state/signal         and relationships
```

### Core Principles

1. **Database is truth** - All measurements persist to DuckDB
2. **Immutable raw data** - `raw.observations` is append-only
3. **Explicit execution** - Importing doesn't run anything
4. **No interpretation** - PRISM measures; you assign meaning

---

## Directory Structure

```
prism-core/
├── prism/                    # Main package
│   ├── fetch/                # Data acquisition
│   │   ├── sources/          # NASA, USGS, NOAA, TEP, etc.
│   │   └── fetch_runner.py   # Orchestration
│   │
│   ├── vector_engines/       # Per-signal measurements
│   │   ├── hurst.py          # Persistence/memory
│   │   ├── entropy.py        # Complexity
│   │   ├── garch.py          # Volatility clustering
│   │   ├── wavelet.py        # Multi-scale decomposition
│   │   ├── spectral.py       # Frequency analysis
│   │   ├── lyapunov.py       # Chaos/sensitivity
│   │   └── rqa.py            # Recurrence patterns
│   │
│   ├── geometry_engines/     # Cross-signal relationships
│   │   ├── pca.py            # Variance structure
│   │   ├── clustering.py     # Grouping
│   │   ├── granger.py        # Lead-lag causality
│   │   ├── cointegration.py  # Long-run equilibrium
│   │   ├── copula.py         # Tail dependencies
│   │   ├── dtw.py            # Shape similarity
│   │   ├── dmd.py            # Dynamic modes
│   │   └── mutual_information.py, transfer_entropy.py, etc.
│   │
│   ├── runners/              # Execution orchestration
│   │   ├── vector_runner.py
│   │   └── geometry_runner.py
│   │
│   ├── audit/                # Health monitoring
│   │   ├── health.py         # Operational checks (GREEN/YELLOW/RED)
│   │   ├── measurements.py   # Analytical measurements
│   │   └── runner.py         # Audit orchestration
│   │
│   ├── db/                   # Database layer
│   │   ├── open.py           # Connection management
│   │   └── migrations/       # Schema definitions
│   │
│   └── registry/             # Signal definitions
│       └── loader.py
│
├── scripts/                  # Entry points
│   ├── build_database.py     # Create fresh DB
│   ├── fetchers/fetch.py     # Fetch data from sources
│   ├── vector.py             # Run vector engines
│   ├── geometry.py           # Run geometry engines
│   ├── promote.py            # Push to MotherDuck
│   ├── sync_motherduck.py
│   └── test_engines.py       # pytest suite
│
├── config/                   # YAML configuration
│   ├── fetch_sources.yaml
│   ├── geometry.yaml
│   ├── registry_*.yaml       # Signal registries by domain
│   └── ...
│
└── data/                     # Local database
    └── prism.duckdb
```

---

## Database Schema

### Layers

| Schema | Purpose | Mutability |
|--------|---------|------------|
| `raw` | Source observations | Append-only |
| `vector` | Per-signal measurements | Write per run |
| `geometry` | Cross-signal relationships | Write per run |
| `results` | Immutable ledgers | Append-only |
| `meta` | Run tracking, schema version | System-managed |
| `audit` | Health checks, findings | Append-only |

### Core Tables

```sql
raw.observations        -- What sources provided (immutable)
vector.measurements     -- Per-signal engine outputs
geometry.results        -- Cross-signal engine outputs
results.vector          -- Immutable vector ledger
results.geometry        -- Immutable geometry ledger
meta.schema_version     -- Schema tracking
meta.fetch_runs         -- Fetch run history
audit.health            -- Health check results
audit.findings          -- Agent findings
```

---

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `build_database.py` | Create/rebuild local DB | `python scripts/build_database.py --force` |
| `fetchers/fetch.py` | Fetch data into raw layer | `python -m prism.db.fetch --cmapss` |
| `vector.py` | Run vector engines | `python scripts/vector.py` |
| `geometry.py` | Run geometry engines | `python scripts/geometry.py` |
| `promote.py` | Push local → MotherDuck | `python scripts/promote.py` |
| `sync_motherduck.py` | Sync with MotherDuck | `python scripts/sync_motherduck.py` |
| `test_engines.py` | Run pytest suite | `pytest scripts/test_engines.py -v` |

---

## Audit System

### Health Agents (Operational)

Traffic light status: GREEN / YELLOW / RED

| Agent | Checks |
|-------|--------|
| `data_integrity` | Gaps, duplicates, null values |
| `staleness` | Data freshness per signal |
| `engine_health` | Engine failure rates |

### Measurement Agents (Analytical)

Pure numbers, no status judgment.

| Agent | Measures |
|-------|----------|
| `displacement` | Distance from historical baseline |
| `clustering` | Cohort structure |
| `variance` | PCA variance concentration |
| `geometry` | System geometry classification |
| `agreement` | Cross-engine agreement |

### Running Audits

```bash
# All agents
python -m prism.audit

# Health only
python -m prism.audit --health

# Measurements only
python -m prism.audit --measure

# Specific agents
python -m prism.audit --agents data_integrity,staleness

# List available agents
python -m prism.audit --list
```

---

## Vector Engines

Per-signal measurements. Each produces a signal topology of measurements.

| Engine | What it measures |
|--------|------------------|
| `hurst` | Long-term memory (H > 0.5 = trending) |
| `entropy` | Complexity/disorder |
| `garch` | Volatility clustering |
| `wavelet` | Multi-scale energy distribution |
| `spectral` | Dominant frequencies |
| `lyapunov` | Sensitivity to initial conditions |
| `rqa` | Recurrence structure |

---

## Geometry Engines

Cross-signal relationships. Each compares signals to each other.

| Engine | What it measures |
|--------|------------------|
| `pca` | Shared variance structure |
| `clustering` | Natural groupings |
| `granger` | Lead-lag causality |
| `cointegration` | Long-run equilibrium |
| `copula` | Tail dependencies |
| `dtw` | Shape similarity |
| `dmd` | Dynamic modes |
| `cross_correlation` | Linear synchronization |
| `mutual_information` | Non-linear dependencies |
| `transfer_entropy` | Information flow |

---

## Configuration

All configuration lives in `config/`.

### Signal Registries

```yaml
# config/registry_climate.yaml
signals:
  - id: CO2_MONTHLY
    source: noaa
    name: "Monthly CO2 Mauna Loa"

  - id: GISS_TEMP_GLOBAL
    source: nasa
    name: "Global Temperature Anomaly"
```

### Geometry Configuration

```yaml
# config/geometry.yaml
engines:
  pca:
    enabled: true
    n_components: 5

  clustering:
    enabled: true
    method: kmeans
    n_clusters: 3
```

---

## Common Operations

### Fresh Start

```bash
# Rebuild database from scratch
python scripts/build_database.py --force

# Fetch data
python -m prism.db.fetch --cmapss

# Run measurements
python scripts/vector.py
python scripts/geometry.py
```

### Daily Update

```bash
# Fetch latest data
python -m prism.db.fetch --cmapss

# Run audit
python -m prism.audit --health
```

### Push to MotherDuck

```bash
python scripts/promote.py
```

---

## What Belongs Where

| This repo (prism-core) | Other repo (prism-dynamics) |
|------------------------|----------------------------|
| Data fetching | State inference |
| Vector measurements | Regime detection |
| Geometry calculations | Dynamics modeling |
| Health monitoring | Interpretation |
| Immutable ledgers | Predictions |

**prism-core measures. prism-dynamics interprets.**

---

## Database Connection

```python
from prism.db.open import get_connection, LOCAL_DB_PATH

# Read-only
conn = get_connection(read_only=True)

# Read-write
conn = get_connection()
```

Never construct paths manually. Always use `get_connection()`.

---

## Quick Queries

```sql
-- Count by signal
SELECT signal_id, COUNT(*)
FROM raw.observations
GROUP BY 1;

-- Date range per signal
SELECT signal_id, MIN(date), MAX(date)
FROM raw.observations
GROUP BY 1;

-- Latest vector measurements
SELECT * FROM vector.measurements
ORDER BY computed_at DESC
LIMIT 10;

-- Health check status
SELECT status, COUNT(*)
FROM audit.health
GROUP BY 1;
```

---

*PRISM Architect: Jason Rudder*
