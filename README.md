# PRISM

**Pure Relational Inference & Structural Measurement**

A behavioral geometry engine for signal topology analysis. PRISM transforms raw observations into eigenvalue-based state representations that capture the SHAPE of signal distributions.

**PRISM computes numbers. ORTHON classifies.**

---

## Quick Start

```bash
# Run full pipeline
python -m prism data/cmapss

# Run individual stages
python -m prism signal-vector-temporal data/cmapss
python -m prism state-vector data/cmapss
python -m prism geometry data/cmapss
python -m prism geometry-dynamics data/cmapss
python -m prism lyapunov data/cmapss
python -m prism dynamics data/cmapss
python -m prism sql data/cmapss
```

**Note:** PRISM expects `typology.parquet` to exist (created by ORTHON).

---

## Architecture

```
observations.parquet (raw signals)
        │
        ▼
┌─────────────────────────────────────────────┐
│            TYPOLOGY (ORTHON)                │
│  Signal classification from raw observations │
│  Creates: typology.parquet + manifest.yaml  │
│  ORTHON's only computation                  │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│        SIGNAL_VECTOR (PRISM)                │
│  Per-signal features at each I              │
│  (kurtosis, skewness, entropy, hurst, etc.) │
│  Scale-invariant only                       │
│  Output: signal_vector.parquet              │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│         STATE_VECTOR (PRISM)                │
│  Centroid = mean position in feature space  │
│  WHERE the system is                        │
│  NO eigenvalues here                        │
│  Output: state_vector.parquet               │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│        STATE_GEOMETRY (PRISM)               │
│  SVD → eigenvalues, effective_dim           │
│  SHAPE of signal cloud                      │
│  This is where eigenvalues live             │
│  Output: state_geometry.parquet             │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  SIGNAL_GEOMETRY + PAIRWISE (PRISM)         │
│  Signal-to-centroid distances               │
│  Signal-to-signal relationships             │
│  Output: signal_geometry.parquet            │
│          signal_pairwise.parquet            │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│      GEOMETRY_DYNAMICS (PRISM)              │
│  Derivatives: velocity, acceleration, jerk  │
│  Output: geometry_dynamics.parquet          │
│          signal_dynamics.parquet            │
│          pairwise_dynamics.parquet          │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│         DYNAMICS LAYER (PRISM)              │
│  Lyapunov exponents, RQA, transfer entropy  │
│  Output: lyapunov.parquet                   │
│          dynamics.parquet                   │
│          information_flow.parquet           │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│          SQL LAYER (PRISM)                  │
│  Aggregations and normalization             │
│  Output: zscore.parquet                     │
│          statistics.parquet                 │
│          correlation.parquet                │
└─────────────────────────────────────────────┘
```

---

## Input Schema (v2.4)

```
observations.parquet
├── cohort     (str)     # Optional: grouping key (engine_1, pump_A) - cargo only
├── signal_id  (str)     # Required: signal name (temp, pressure, sensor_01)
├── I          (UInt32)  # Required: sequential index 0,1,2... per (cohort, signal_id)
├── value      (Float64) # Required: measurement
```

**I is canonical.** Sequential integers per (cohort, signal_id). Not timestamps.

**cohort is cargo.** Passes through, has ZERO effect on compute. Unique time series = (cohort, signal_id).

See [MANIFEST_CONTRACT.md](MANIFEST_CONTRACT.md) for complete manifest v2.4 specification.

---

## Output Files (14 total)

### From ORTHON (input to PRISM)
| File | Description |
|------|-------------|
| `typology.parquet` | Signal characterization |
| `manifest.yaml` | Engine selection per signal |

### Feature Layer (PRISM)
| File | Description |
|------|-------------|
| `signal_vector.parquet` | Per-signal features with I |

### State Layer (PRISM)
| File | Description |
|------|-------------|
| `state_vector.parquet` | Centroid (position) per I |

### Geometry Layer (PRISM)
| File | Description |
|------|-------------|
| `state_geometry.parquet` | Eigenvalues, effective_dim per I |
| `signal_geometry.parquet` | Signal-to-centroid distances |
| `signal_pairwise.parquet` | Pairwise signal relationships |

### Geometry Dynamics (PRISM)
| File | Description |
|------|-------------|
| `geometry_dynamics.parquet` | State trajectory derivatives |
| `signal_dynamics.parquet` | Per-signal trajectories |
| `pairwise_dynamics.parquet` | Pairwise trajectories |

### Dynamics Layer (PRISM)
| File | Description |
|------|-------------|
| `lyapunov.parquet` | Lyapunov exponents |
| `dynamics.parquet` | RQA, attractor metrics |
| `information_flow.parquet` | Transfer entropy, Granger |

### SQL Layer (PRISM)
| File | Description |
|------|-------------|
| `zscore.parquet` | Z-score normalized |
| `statistics.parquet` | Summary statistics |
| `correlation.parquet` | Correlation matrix |

---

## Key Concepts

### Typology (Lives in ORTHON)

**Typology is the ONLY computation in ORTHON.** It classifies signals and creates manifest.yaml specifying which engines PRISM should run.

### State Vector vs State Geometry

| File | Computes | Analogy |
|------|----------|---------|
| state_vector | Centroid (mean position) | WHERE the system is |
| state_geometry | Eigenvalues (SVD) | SHAPE of signal cloud |

### Scale-Invariant Features Only

All features are ratios, entropy, or shape metrics. Deprecated: rms, peak, mean, std.

### Eigenvalue-Based Geometry

```python
# In state_geometry.py:
U, S, Vt = svd(signal_matrix - centroid)
eigenvalues = S² / (N - 1)
effective_dim = (Σλ)² / Σλ²
```

### Geometry Dynamics

Differential geometry on state evolution:
- **velocity** = dx/dt
- **acceleration** = d²x/dt²
- **jerk** = d³x/dt³

**PRISM computes derivatives. ORTHON interprets trajectories.**

---

## Directory Structure

```
prism/
├── prism/
│   ├── cli.py
│   ├── signal_vector_temporal.py
│   ├── sql_runner.py
│   │
│   └── engines/
│       ├── state_vector.py       # Centroid only
│       ├── state_geometry.py     # Eigenvalues here
│       ├── signal_geometry.py
│       ├── signal_pairwise.py
│       ├── geometry_dynamics.py
│       ├── lyapunov_engine.py
│       ├── dynamics_runner.py
│       ├── information_flow_runner.py
│       ├── signal/               # Per-signal engines
│       ├── rolling/              # Rolling window engines
│       └── sql/                  # SQL engines
│
└── data/
```

---

## Rules

1. **PRISM computes, ORTHON classifies** - no labels, no thresholds in PRISM
2. **Typology lives in ORTHON** - PRISM receives manifest.yaml
3. **state_vector = centroid, state_geometry = eigenvalues** - separate concerns
4. **Scale-invariant features only** - no absolute values
5. **I is canonical** - sequential per (cohort, signal_id), not timestamps
6. **cohort is cargo** - never in groupby, unique series = (cohort, signal_id)

---

## Do NOT

- Put classification logic in PRISM
- Compute eigenvalues in state_vector (they belong in state_geometry)
- Use deprecated scale-dependent engines
- Include cohort in groupby operations
- Create typology in PRISM (ORTHON's job)

---

## Credits

- **Avery Rudder** - "Laplace transform IS the state engine" - eigenvalue insight
