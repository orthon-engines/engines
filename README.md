# Engines

**Dynamical systems computation engines.** Part of the [Orthon Engines](https://github.com/orthon-engines) platform. Drop a CSV, get eigendecomposition, Lyapunov exponents, velocity fields, and urgency metrics.

```bash
pip install orthon-engines
engines run sensor_data.csv
```

---

## Quickstart

```bash
# Analyze any CSV (wide format: columns = signals, rows = timepoints)
engines run data.csv

# Inspect data before running
engines inspect data.csv

# Run with full dynamical atlas (velocity fields, FTLE ridges, urgency)
engines run data.csv --atlas

# Explore results in browser
engines explore output/
```

Engines auto-detects your CSV format, generates a manifest, and runs the full pipeline. No configuration needed.

### Input Formats

Engines accepts:

- **Wide CSV** — columns are signals, rows are timepoints (most common)
- **Long CSV** — columns: `signal_id`, `value`, and optionally `time`/`index`
- **Parquet** — `observations.parquet` with `signal_id`, `I`, `value` columns
- **Directory** — containing `observations.parquet`

Column names are auto-detected. `time`, `timestamp`, `t`, `step`, `index` all map to the time axis. `sensor`, `channel`, `variable`, `feature` all map to `signal_id`.

### Example: Wide CSV

```csv
time,temperature,pressure,vibration,flow_rate
0,300.1,101.3,0.02,50.1
1,300.3,101.2,0.03,50.0
2,300.5,101.4,0.02,49.9
...
```

```bash
engines run sensors.csv
# Output: 15 parquet files in output/
```

---

## What Engines Computes

Engines runs a 15-stage pipeline that transforms raw observations into a complete dynamical characterization:

### Core Pipeline (stages 0-14)

| Stage | Output | What It Computes |
|-------|--------|-----------------|
| 00 | `breaks.parquet` | Regime changes (steps + impulses) per signal |
| 01 | `signal_vector.parquet` | Per-signal features: kurtosis, spectral entropy, Hurst, ACF, etc. |
| 02 | `state_vector.parquet` | System centroid (position in feature space) |
| 03 | `state_geometry.parquet` | Eigendecomposition: eigenvalues, effective dimension, eigenvectors |
| 04 | `cohorts.parquet` | Cohort-level aggregates |
| 05 | `signal_geometry.parquet` | Per-signal distance/coherence to system state |
| 06 | `signal_pairwise.parquet` | Pairwise signal correlations with eigenvector gating |
| 07 | `geometry_dynamics.parquet` | Derivatives: velocity, acceleration, jerk of eigenstructure |
| 08 | `ftle.parquet` | Finite-Time Lyapunov Exponents (chaos/stability per signal) |
| 09 | `dynamics.parquet` | Per-signal stability classification |
| 10 | `information_flow.parquet` | Granger causality between signal pairs |
| 11 | `topology.parquet` | Topological features of signal manifold |
| 12 | `zscore.parquet` | Z-score normalization of all metrics |
| 13 | `statistics.parquet` | Summary statistics per signal |
| 14 | `correlation.parquet` | Feature correlation matrix |

### Atlas Pipeline (stages 15-23, `--atlas` flag)

The atlas adds system-level dynamics that reveal **where your system is going** and **how fast**:

| Stage | Output | What It Computes |
|-------|--------|-----------------|
| 15 | `ftle_field.parquet` | Spatiotemporal FTLE field |
| 16 | `break_sequence.parquet` | Break propagation order across signals |
| 17 | `ftle_backward.parquet` | Backward FTLE (attracting structures) |
| 18 | `segment_comparison.parquet` | Pre/post segment geometry deltas |
| 19 | `info_flow_delta.parquet` | Causality changes across segments |
| 20 | `geometry_full.parquet` | Expanding-window eigendecomposition trajectory |
| 21 | `velocity_field.parquet` | State-space velocity, acceleration, curvature |
| 22 | `ftle_rolling.parquet` | Rolling FTLE stability evolution |
| 23 | `ridge_proximity.parquet` | Urgency: velocity toward FTLE ridges |

---

## Key Metrics

### effective_dim (Participation Ratio)

The number of directions that matter. Computed from eigenvalues of the signal covariance matrix.

- `effective_dim = 1` — all variance in one direction (1D behavior)
- `effective_dim = N` — variance spread equally across N signals
- **Dropping effective_dim** = dimensional collapse = system losing degrees of freedom

Found in: `state_geometry.parquet`, `geometry_full.parquet`, `geometry_dynamics.parquet`

### FTLE (Finite-Time Lyapunov Exponent)

Measures sensitivity to initial conditions over finite time windows.

- `FTLE > 0` — trajectories diverge (instability, chaos)
- `FTLE ~ 0` — trajectories parallel (quasi-periodic)
- `FTLE < 0` — trajectories converge (stable)

Found in: `ftle.parquet`, `ftle_rolling.parquet`, `ftle_backward.parquet`

### Urgency

How fast you're approaching an FTLE ridge (regime boundary). Combines FTLE gradient with velocity.

- `nominal` — stable, not approaching any boundary
- `warning` — moving toward a ridge (early warning)
- `elevated` — near ridge, moving away
- `critical` — near ridge, heading in

Found in: `ridge_proximity.parquet`

### Eigenvector Loadings

Which signals contribute most to each principal component. Stored as `ev1_{signal}`, `ev2_{signal}`, `ev3_{signal}` columns.

- Eigenvector continuity is enforced across sequential windows (sign flips corrected)
- Bootstrap confidence intervals provided: `eff_dim_std`, `eff_dim_ci_low`, `eff_dim_ci_high`

Found in: `state_geometry.parquet`

### Embedding Parameters (Cao's Method)

FTLE uses delay embedding with parameters chosen automatically:

- `embedding_dim` — Cao's method (E1 saturation)
- `embedding_tau` — Average Mutual Information (first minimum)
- `is_deterministic` — Cao's E2 test (E2 != 1 for deterministic dynamics)

Found in: `ftle.parquet`

---

## Python API

```python
# Full pipeline from code
from engines.input_loader import load_input, detect_data_characteristics, generate_auto_manifest
from engines.entry_points.run_pipeline import run as run_pipeline

# Load any format
observations = load_input("data.csv")

# Analyze
chars = detect_data_characteristics(observations)
print(f"{chars['n_signals']} signals, {chars['min_samples']} samples each")

# Generate manifest and run
manifest = generate_auto_manifest(chars)
```

### Individual Stages

```python
# Signal features
from engines.entry_points.stage_01_signal_vector import run
signal_vector = run("observations.parquet", "signal_vector.parquet", manifest=manifest)

# Eigendecomposition
from engines.entry_points.stage_03_state_geometry import run
state_geometry = run("signal_vector.parquet", "state_vector.parquet", "state_geometry.parquet")

# FTLE
from engines.entry_points.stage_08_ftle import run
ftle = run("observations.parquet", "ftle.parquet")

# Velocity field (atlas)
from engines.entry_points.stage_21_velocity_field import run
velocity = run("observations.parquet", "velocity_field.parquet")
```

### Engine-Level Access

```python
# Direct eigendecomposition
from engines.manifold.state.eigendecomp import compute
result = compute(signal_matrix, centroid=centroid)
# result: eigenvalues, explained_ratio, effective_dim, principal_components, signal_loadings

# Direct FTLE
from engines.manifold.dynamics.ftle import compute as compute_ftle
result = compute_ftle(time_series, min_samples=200)
# result: ftle, ftle_std, embedding_dim, embedding_tau, is_deterministic

# Cao's embedding analysis
from engines.primitives.embedding import cao_embedding_analysis
cao = cao_embedding_analysis(signal, delay=10)
# cao: dimension, E1_values, E2_values, is_deterministic
```

---

## Architecture

```
engines/
├── cli.py                    CLI: run, inspect, explore, atlas
├── input_loader.py           Auto-detect CSV/parquet, generate manifest
│
├── entry_points/             Orchestration (parquet -> parquet)
│   ├── stage_00_breaks.py        Regime change detection
│   ├── stage_01_signal_vector.py Per-signal features
│   ├── stage_02_state_vector.py  System centroid
│   ├── stage_03_state_geometry.py Eigendecomposition
│   ├── stage_08_ftle.py          Lyapunov exponents
│   ├── stage_16-23_*.py          Atlas engines
│   └── run_pipeline.py          Full pipeline orchestrator
│
├── engines/                  Computation (numpy arrays -> dicts)
│   ├── signal/               Per-signal feature engines
│   ├── state/                Eigendecomposition, centroids
│   ├── dynamics/             FTLE, RQA, stability
│   ├── geometry/             Signal-to-state relationships
│   └── pairwise/             Signal-signal relationships
│
└── primitives/               Pure math (numpy -> float)
    ├── individual/           Statistics, spectral, entropy
    ├── embedding/            Delay embedding, Cao, AMI
    └── pairwise/             Correlation, Granger, transfer entropy
```

**Design principle:** Entry points orchestrate. Engines compute. Primitives do math. Each layer only calls the layer below it.

---

## CLI Reference

```bash
# Run full pipeline on any input
engines run <input> [--output DIR] [--atlas] [--manifest FILE] [--segments name:start:end]

# Inspect data without running
engines inspect <input>

# Launch browser-based explorer
engines explore <output_dir> [--port 8080]

# Run full atlas pipeline on existing output
engines atlas <data_dir> [--output DIR]

# Individual atlas stages
engines break-sequence <data_dir>
engines ftle-backward <data_dir>
engines geometry-full <data_dir>
engines velocity-field <data_dir>
engines ftle-rolling <data_dir>
engines ridge-proximity <data_dir>
engines segment-comparison <data_dir>
engines info-flow-delta <data_dir>
```

---

## Requirements

- Python 3.9+
- numpy, scipy, polars, joblib, pyyaml, pyarrow

Optional:
- `nolds` — enhanced Lyapunov estimation
- `scikit-learn` — ML features
- `ripser`, `persim` — topological data analysis

```bash
pip install orthon-engines          # core
pip install orthon-engines[all]     # everything
```

---

## How It Works

1. **Input loading** — auto-detects CSV format (wide/long), maps column aliases, converts to observations format
2. **Manifest generation** — analyzes signal characteristics (sample count, stationarity), selects window sizes and engines
3. **Signal features** — computes scale-invariant features per signal per window (kurtosis, spectral entropy, Hurst, etc.)
4. **State geometry** — eigendecomposes the signal feature matrix via SVD. Tracks eigenvalue evolution, effective dimension, and eigenvector continuity across windows
5. **Dynamics** — computes FTLE per signal using delay embedding (Cao's method for dimension, AMI for delay)
6. **Atlas** (optional) — velocity fields, rolling FTLE, ridge proximity = "where is the system going and how urgent is it?"
7. **Output** — all results as Parquet files, queryable with DuckDB, Polars, or pandas

---

## Citation

```bibtex
@software{orthon2026,
  title = {Orthon Engines: Domain-Agnostic Dynamical Systems Analysis Platform},
  author = {Rudder, Jason},
  year = {2026},
  url = {https://github.com/orthon-engines/engines}
}
```

---

## License

Free for academic and noncommercial use.
Citation required for publications.
Commercial license: [orthon.io/licensing](https://orthon.io/licensing)
