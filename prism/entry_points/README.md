# PRISM Entry Points

**CLI entry points for the PRISM pipeline**

Entry points are the execution layer of PRISM. Each entry point computes measurements that persist to Parquet files.

---

## Architecture

```
prism/entry_points/           # CLI entry points (python -m prism.entry_points.*)
├── fetch.py                 # Data fetching to Parquet
├── characterize.py          # 6-axis indicator classification
├── indicator_vector.py      # Layer 1: Vector metrics (51 per indicator)
├── laplace.py               # Layer 2: Laplace field computation
├── laplace_pairwise.py      # Layer 2: Pairwise geometry (vectorized)
├── geometry.py              # Layer 3: Cohort geometry + modes + wavelet
├── state.py                 # Layer 4: Query-time state derivation
├── cohort_state.py          # Layer 4: Cohort-level state
├── mode_geometry.py         # Behavioral mode geometry analysis
├── dynamic_vector.py        # Dynamical systems vector computation
├── dynamic_state.py         # Dynamical systems state computation
├── generate_dynamical.py    # Generate test data (Lorenz, Rossler, etc.)
├── generate_pendulum_regime.py  # Generate pendulum regime data
├── physics.py               # Physics validation (conservation laws)
└── hybrid.py                # PRISM + ML supervised bridge

prism/modules/               # Reusable computation (NOT entry points)
├── characterize.py          # Inline characterization
├── laplace.py               # Laplace field computation
├── modes.py                 # Mode discovery from signatures
├── wavelet_microscope.py    # Frequency-band degradation
└── prefilter.py             # O(n) Laplacian pre-filter
```

**Key Principle:** Modules are building blocks imported by entry points, not run directly.

---

## Pipeline Execution

### Core Pipeline

```bash
# Layer 0: Data Ingestion
python -m prism.entry_points.fetch --cmapss
python -m prism.entry_points.fetch --femto
python -m prism.entry_points.fetch --hydraulic

# Layer 1: Vector Computation
python -m prism.entry_points.indicator_vector --domain cmapss

# Layer 2: Laplace Field
python -m prism.entry_points.laplace --domain cmapss

# Layer 2: Pairwise Geometry
python -m prism.entry_points.laplace_pairwise --domain cmapss

# Layer 3: Cohort Geometry (includes modes + wavelet)
python -m prism.entry_points.geometry --domain cmapss

# Layer 4: State
python -m prism.entry_points.state --domain cmapss
```

### Dynamical Systems Validation

```bash
# Generate test data
python -m prism.entry_points.generate_dynamical --system lorenz
python -m prism.entry_points.generate_pendulum_regime

# Compute vectors
python -m prism.entry_points.dynamic_vector

# Validate physics
python -m prism.entry_points.physics
```

### Supervised Learning Bridge

```bash
# PRISM features + ML models
python -m prism.entry_points.hybrid --domain cmapss --model xgboost
```

---

## Entry Points Summary (15)

| Entry Point | Layer | Purpose |
|-------------|-------|---------|
| `fetch.py` | 0 | Data ingestion from fetchers |
| `characterize.py` | 0.5 | 6-axis dynamical classification |
| `indicator_vector.py` | 1 | 51 behavioral metrics per indicator |
| `laplace.py` | 2 | Laplace field computation |
| `laplace_pairwise.py` | 2 | Pairwise geometry (vectorized) |
| `geometry.py` | 3 | Cohort geometry + modes + wavelet |
| `state.py` | 4 | Query-time state derivation |
| `cohort_state.py` | 4 | Cohort-level state computation |
| `mode_geometry.py` | 3 | Behavioral mode geometry |
| `dynamic_vector.py` | 1 | Dynamical systems vectors |
| `dynamic_state.py` | 4 | Dynamical systems state |
| `generate_dynamical.py` | — | Generate Lorenz/Rossler test data |
| `generate_pendulum_regime.py` | — | Generate pendulum regime data |
| `physics.py` | 5 | Conservation law validation |
| `hybrid.py` | — | PRISM + ML supervised bridge |

---

## Common Options

| Option | Description |
|--------|-------------|
| `--domain NAME` | Target domain (cmapss, femto, hydraulic, tep, etc.) |
| `--testing` | Enable testing mode (allows limiting flags) |
| `--force` | Recompute all (clear progress) |
| `--cohort NAME` | Filter to specific cohort |
| `--workers N` | Parallel workers (Parquet has no lock issues) |

---

## Storage

All storage uses Parquet files (no database):

```
data/{domain}/
├── raw/                    # Layer 0: Raw observations
├── vector/                 # Layer 1-2: Vector metrics + Laplace field
├── geometry/               # Layer 3: Structural snapshots
├── state/                  # Layer 4: Temporal dynamics
├── physics/                # Physics validation results
└── filter/                 # Redundancy analysis
```
