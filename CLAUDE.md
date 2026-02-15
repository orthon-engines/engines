# CLAUDE.md — Manifold

Manifold is a domain-agnostic dynamical systems computation engine.
It reads `observations.parquet` and produces 29 parquet files in 6 directories.

**Design principle:** Entry points orchestrate. Manifold computes. Primitives do math.
Each layer only calls the layer below it.

**Manifold computes. It never classifies, labels, or interprets.**
No `trajectory_type`. No `stability_class`. No `is_chaotic`.
If you find yourself writing `if signal_type == 'PERIODIC'`, STOP. That belongs in Prime.

---

## Repositories

GitHub org: `rudder-framework`

```
rudder-framework/prime       → interpreter, classification, SQL analysis
rudder-framework/manifold    → THIS REPO — blind compute engine
rudder-framework/primitives  → Rust+Python math functions (shared dependency)
```

Prime depends on Manifold's output. Manifold never depends on Prime.
Both depend on primitives. Primitives depends on nothing.

```
primitives   (leaf — no dependencies)
   ↑    ↑
   |    |
Prime  Manifold
   |    ↑
   └────┘    (Prime calls Manifold's pipeline)
```

**Note:** `manifold/primitives/` still exists in-tree as the Python implementation.
`~/primitives/` is the shared Rust+Python package. Both are active.

---

## How to Run

```bash
cd ~/manifold
python -m manifold ~/domains/rossler           # Full pipeline on rossler
python -m manifold ~/domains/cmapss/FD_004/train  # Full pipeline on FD004
python -m manifold ~/domains/calce             # Full pipeline on calce
```

One command. All 29 stages run. No flags needed.

**Virtual environment:** `./venv/` — always use it. Never create a new one.

**Rust backend:** Controlled by `USE_RUST` env var (default: on).
Set `USE_RUST=0` to force Python fallback.

**Parallel workers:** `MANIFOLD_WORKERS` env var (default: `0` = auto-detect).
FD004 with 249 cohorts across 18 stages: ~3 minutes with Rust + 9 workers.

---

## Architecture: Three Layers

```
manifold/stages/       RUNNERS — orchestrate I/O. Read parquet, call engines, write parquet.
manifold/core/         ENGINES — compute. DataFrames in, DataFrames out. No file I/O.
manifold/primitives/   MATH — pure functions. numpy in, numpy out. No DataFrames, no I/O.
```

Each layer ONLY calls the layer below it.

| Layer | Input | Returns | Does I/O |
|-------|-------|---------|----------|
| `stages/group/run.py` (runner) | manifest, file paths | writes parquet files | YES |
| `core/*.py` (engine) | config dict, DataFrames | DataFrames | NO |
| `primitives/*.py` | numpy arrays | numpy arrays | NO |

Also:
- `manifold/io/` — reader.py, writer.py, manifest.py (all parquet I/O)
- `manifold/validation/` — input checks (sequential I, no nulls, signal_id present)
- `manifold/run.py` — top-level sequencer (dependency-ordered, not strictly group-sequential)

### Layer enforcement

```bash
# These should return NOTHING. If they return hits, someone broke the contract:
grep -rn "from manifold.stages" manifold/core/ manifold/primitives/
grep -rn "from manifold.core" manifold/primitives/
```

Nobody reaches up. Nobody skips a layer.

---

## Import Conventions

```python
# Primitives: direct function imports
from manifold.primitives.individual.spectral import psd
from manifold.primitives.dynamical.ftle import compute_ftle

# Core engines: module imports
from manifold.core.state_geometry import compute_state_geometry

# Stages import from core, never from other stages
from manifold.core.geometry_dynamics import compute_geometry_dynamics
```

---

## Data Model

### Input: observations.parquet (READ ONLY)

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| cohort | String | Optional | Grouping key (engine_1, pump_A). User-defined, never used inside engines. |
| signal_id | String | Required | Which signal (temp, pressure, vibration) |
| I | UInt32 | Required | Canonical sequential index: 0, 1, 2, 3... per signal_id. **NOT a timestamp.** |
| value | Float64 | Required | The measurement |

**I is sacred.** Manifold groups by `(signal_id, I)` where I is always sequential integers starting at 0.
Never reindex. Never use timestamps. Never skip values.

**observations.parquet is sacred.** Never modify it. Never normalize it in place.
Never add columns. It is READ ONLY.

### Output: 29 parquet files in 6 directories

All output goes to `<data_dir>/output/`:

#### `1_signal_features/` — "What does each signal look like?"

| File | Stage | Group | Description |
|------|-------|-------|-------------|
| signal_vector.parquet | 01 | vector | Per-signal windowed features (FFT, entropy, hurst, kurtosis...) |
| signal_geometry.parquet | 05 | geometry | Distance-to-centroid, coherence, projections |
| signal_stability.parquet | 33 | vector | Hilbert + Wavelet analysis |

#### `2_system_state/` — "What is the system's geometric structure?"

| File | Stage | Group | Description |
|------|-------|-------|-------------|
| state_vector.parquet | 02 | geometry | Cross-signal centroid per window |
| state_geometry.parquet | 03 | geometry | Eigenvalues, effective_dim, condition_number — **THE core output** |
| geometry_dynamics.parquet | 07 | geometry | Velocity/jerk of geometry evolution |
| sensor_eigendecomp.parquet | 20 | geometry | Rolling 2-level SVD |

#### `3_health_scoring/` — "How healthy is the system?"

| File | Stage | Group | Description |
|------|-------|-------|-------------|
| breaks.parquet | 00 | vector | Change-point detection (Heaviside/Dirac) |
| cohort_baseline.parquet | 34 | geometry | SVD on early-life observations — healthy reference frame |
| observation_geometry.parquet | 35 | geometry | Per-observation scoring against baseline |

#### `4_signal_relationships/` — "How do signals relate to each other?"

| File | Stage | Group | Description |
|------|-------|-------|-------------|
| signal_pairwise.parquet | 06 | information | Covariance matrix within each window |
| information_flow.parquet | 10 | information | Granger causality, transfer entropy, copula |
| segment_comparison.parquet | 18 | information | Segment-level statistical comparison |
| info_flow_delta.parquet | 19 | information | Change in information flow over time |

#### `5_evolution/` — "How is the system changing over time?"

| File | Stage | Group | Description |
|------|-------|-------|-------------|
| ftle.parquet | 08 | dynamics | Finite-Time Lyapunov Exponents |
| lyapunov.parquet | 08_lyapunov | dynamics | Largest Lyapunov exponent |
| cohort_thermodynamics.parquet | 09a | dynamics | Temperature, entropy from eigenvalue spectra |
| ftle_field.parquet | 15 | dynamics | Local FTLE grid (LCS/astrodynamics) |
| ftle_backward.parquet | 17 | dynamics | Backward FTLE (attracting structures) |
| velocity_field.parquet | 21 | dynamics | State-space velocity field |
| ftle_rolling.parquet | 22 | dynamics | Time-varying FTLE |
| ridge_proximity.parquet | 23 | dynamics | Urgency = v . grad(FTLE) |
| persistent_homology.parquet | 36 | dynamics | Topological persistence (Betti numbers) |

#### `6_fleet/` — "How do cohorts compare across the fleet?"

| File | Stage | Group | Description |
|------|-------|-------|-------------|
| cohort_vector.parquet | 25 | energy | Cohort-level feature vector (Scale 2 input) |
| system_geometry.parquet | 26 | energy | SVD at cohort scale (Scale 2) |
| cohort_pairwise.parquet | 27 | energy | Pairwise cohort comparison |
| cohort_information_flow.parquet | 28 | energy | Granger at system scale |
| cohort_ftle.parquet | 30 | energy | FTLE on cohort trajectories |
| cohort_velocity_field.parquet | 31 | energy | Velocity field at fleet scale |

---

## 29 Stages, 5 Source Groups

Stages live in `manifold/stages/<group>/` and run in dependency order (not strictly by group):

| Group | Stages | Count |
|-------|--------|-------|
| **vector/** | breaks(00), signal_vector(01), signal_stability(33) | 3 |
| **geometry/** | state_vector(02), state_geometry(03), signal_geometry(05), geometry_dynamics(07), sensor_eigendecomp(20), cohort_baseline(34), observation_geometry(35) | 7 |
| **dynamics/** | ftle(08), lyapunov(08_lyapunov), cohort_thermodynamics(09a), ftle_field(15), ftle_backward(17), velocity_field(21), ftle_rolling(22), ridge_proximity(23), persistent_homology(36) | 9 |
| **information/** | signal_pairwise(06), information_flow(10), segment_comparison(18), info_flow_delta(19) | 4 |
| **energy/** | cohort_vector(25), system_geometry(26), cohort_pairwise(27), cohort_information_flow(28), cohort_ftle(30), cohort_velocity_field(31) | 6 |

---

## Two-Scale Architecture

Same engines run at two scales:

```
Scale 1:  signals → signal_vector → state_geometry     (per cohort)
Scale 2:  cohort_vector → system_geometry               (across cohorts)
```

Eigendecomposition at Scale 1 tells you about individual system health.
Eigendecomposition at Scale 2 tells you about fleet-wide patterns.

---

## Engine Minimum Sample Requirements

FFT-based engines require larger windows. This is physics, not a bug.

| Engine | Min Samples | Reason |
|--------|-------------|--------|
| spectral | 64 | FFT resolution |
| hurst | 128 | Long-range dependence |
| sample_entropy | 64 | Statistical validity |
| perm_entropy | 8 | Permutation patterns |
| acf_decay | 16 | Lag structure |
| kurtosis | 4 | 4th moment |

Do NOT lower minimums to fit small windows. The math doesn't work.
Insufficient data → return NaN, never skip.

---

## Rules

### 1. THREE LAYERS, ONE DIRECTION
Runners call engines. Engines call primitives. Nobody reaches up.

### 2. NO CLASSIFICATION IN MANIFOLD
Manifold computes numbers. It never decides what those numbers mean.
No trajectory_type, stability_class, is_chaotic, regime labels.
Classification lives in Prime via SQL.

### 3. OBSERVATIONS ARE SACRED
Never modify `observations.parquet`. Never normalize it in place.
Never add columns. It is READ ONLY.

### 4. I IS SEQUENTIAL
I = 0, 1, 2, 3... per signal_id. Always. No timestamps. No gaps.

### 5. COHORT IS USER-OPTIONAL
`cohort` is a grouping key the user provides for their own bookkeeping.
Engines never filter or classify by cohort. The math is blind to it.

### 6. COMPUTE ONCE, QUERY FOREVER
Manifold runs once and writes parquet files. Every question gets answered
by Prime running SQL against those files. If Manifold needs to run twice
to answer a question, the architecture is wrong.

### 7. NO HACKS
- No `sys.modules` aliases
- No `_PrimitivesNamespace` shim classes
- No compatibility layers
- If imports are broken, fix them properly

### 8. SHOW YOUR WORK
Before modifying any file:
1. Show the existing file you're modifying
2. Show the existing pattern you're following
3. Get approval before creating NEW files

---

## Where New Code Goes

| Type of Code | Location | Pattern |
|--------------|----------|---------|
| New primitive | `manifold/primitives/` | Pure function, numpy in/out |
| New engine | `manifold/core/` | DataFrame in, DataFrame out, no I/O |
| New stage runner | `manifold/stages/<group>/` | Follow existing runner pattern |
| New I/O helper | `manifold/io/` | Reader or writer only |

---

## Sibling Repos

- **Prime** (`~/prime/`) — The interpreter. Reads Manifold's 29 parquet files via DuckDB SQL. Handles typology, classification, analysis, canary detection, brittleness scoring, ML features. Static HTML + DuckDB-WASM explorer for browser-based analysis.
- **Primitives** (`~/primitives/`) — Rust+Python math functions. Hurst, Lyapunov, FTLE, perm_entropy, ADF, spectral analysis. Shared by both Prime and Manifold.
- **Manifold** (this repo) — The compute engine. `observations.parquet` in, 29 parquet files out.

---

## Known Domains

```
~/domains/
├── cmapss/           # C-MAPSS turbofan (FD001-FD004)
├── calce/            # Battery calendar aging
├── rossler/          # Chaotic system (Rossler attractor)
├── battery/          # NASA battery degradation
├── electrochemistry/ # Electrochemical data
├── cwru/             # CWRU bearing fault
├── hydraulic/        # Hydraulic condition monitoring
├── industrial/       # SKAB, MetroPT
└── secom/            # Semiconductor manufacturing
```

---

## What NOT to Touch

- Mathematical logic inside engines (only change with explicit approval)
- Output file schemas (downstream SQL depends on column names)
- The `main` branch (work on feature branches)
- Stage execution order in `run.py` (dependencies are real)
- observations.parquet (READ ONLY, always)
