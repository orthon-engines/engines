# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## STOP: READ THIS FIRST

### You are working on PRISM. Not ORTHON.

**PRISM computes numbers. ORTHON classifies.**

- Do NOT modify any files in the `orthon` repository.
- Do NOT create typology logic, classification rules, or signal labels in PRISM.
- Do NOT modify observations.parquet or typology.parquet.
- Do NOT second-guess manifest.yaml — execute what it says.
- If you need to understand what ORTHON sends, read `MANIFEST_CONTRACT.md`.
- If MANIFEST_CONTRACT.md doesn't answer your question, ASK THE USER.

**If you find yourself writing `if signal_type == 'PERIODIC'` in PRISM, STOP.
That is classification. Classification belongs in ORTHON.**

---

## DATA COMPLIANCE IS ORTHON'S RESPONSIBILITY

**ORTHON delivers data. PRISM computes. The contract is non-negotiable.**

If `observations.parquet` does not comply with PRISM's schema:

1. **STOP** — Do not proceed
2. **REJECT** — Return error explaining what's wrong
3. **DO NOT** adjust PRISM to accommodate bad data

**Bad data from ORTHON is ORTHON's problem.**

---

## Architecture

```
observations.parquet   (from ORTHON)
typology.parquet       (from ORTHON)
manifest.yaml          (from ORTHON)
        │
        ▼
┌─────────────────────────────────────────────┐
│        signal_vector/  (PRISM Stage 1)      │
│  Per-signal features at each I              │
│  signal/ = kurtosis, spectral, entropy...   │
│  rolling/ = rolling_kurtosis, rolling_vol...│
│  sql/ = zscore, statistics                  │
│  Output: signal_vector.parquet              │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│        state_vector/  (PRISM Stage 2)       │
│  state_vector.py = centroid (WHERE)         │
│  state_geometry.py = eigenvalues (SHAPE)    │
│  signal_geometry.py = signal-to-centroid    │
│  Output: state_vector.parquet               │
│          state_geometry.parquet              │
│          signal_geometry.parquet             │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│    geometry_pairwise/  (PRISM Stage 3)      │
│  signal_pairwise.py = between-signal        │
│  granger, transfer_entropy = directional    │
│  correlation, mutual_info = symmetric       │
│  Output: signal_pairwise.parquet            │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│    geometry_laplace/  (PRISM Stage 4)       │
│  geometry_dynamics.py = velocity, accel     │
│  lyapunov_engine.py = chaos measurement     │
│  dynamics_runner.py = RQA, attractor        │
│  information_flow_runner.py = TE, causal    │
│  Output: geometry_dynamics.parquet          │
│          lyapunov.parquet                   │
│          dynamics.parquet                   │
│          information_flow.parquet           │
└─────────────────────────────────────────────┘
```

---

## Directory Structure

```
prism/
├── signal_vector/             # Stage 1: per-signal computation
│   ├── __init__.py
│   ├── runner.py              # Orchestrates signal vector creation
│   ├── signal/                # Per-signal engines (one value per signal per window)
│   │   ├── kurtosis.py
│   │   ├── skewness.py
│   │   ├── crest_factor.py
│   │   ├── entropy.py
│   │   ├── hurst.py
│   │   ├── spectral.py
│   │   ├── harmonics.py
│   │   ├── frequency_bands.py
│   │   ├── lyapunov.py
│   │   ├── garch.py
│   │   ├── attractor.py
│   │   ├── dmd.py
│   │   ├── pulsation_index.py
│   │   ├── rate_of_change.py
│   │   ├── time_constant.py
│   │   ├── cycle_counting.py
│   │   ├── basin.py
│   │   └── lof.py
│   ├── rolling/               # Rolling window engines
│   │   ├── derivatives.py
│   │   ├── rolling_kurtosis.py
│   │   ├── rolling_skewness.py
│   │   ├── rolling_entropy.py
│   │   ├── rolling_crest_factor.py
│   │   ├── rolling_hurst.py
│   │   ├── rolling_lyapunov.py
│   │   ├── rolling_volatility.py
│   │   ├── rolling_pulsation.py
│   │   ├── manifold.py
│   │   └── stability.py
│   └── sql/                   # SQL-based engines
│       ├── zscore.py
│       ├── statistics.py
│       └── correlation.py
│
├── state_vector/              # Stage 2: system state + geometry
│   ├── __init__.py
│   ├── state_vector.py        # Centroid (WHERE the system is)
│   ├── state_geometry.py      # Eigenvalues (SHAPE of signal cloud)
│   └── signal_geometry.py     # Signal-to-centroid distances
│
├── geometry_pairwise/         # Stage 3: signal relationships
│   ├── __init__.py
│   ├── signal_pairwise.py     # Pairwise computation orchestrator
│   ├── granger.py             # Directional: A → B causality
│   ├── transfer_entropy.py    # Directional: information flow
│   ├── correlation.py         # Symmetric: linear relationship
│   ├── mutual_info.py         # Symmetric: nonlinear relationship
│   └── cointegration.py       # Symmetric: long-run equilibrium
│
├── geometry_laplace/          # Stage 4: dynamics on geometry
│   ├── __init__.py
│   ├── geometry_dynamics.py   # Derivatives: velocity, acceleration, jerk
│   ├── lyapunov_engine.py     # Chaos measurement
│   ├── dynamics_runner.py     # RQA, attractor reconstruction
│   └── information_flow_runner.py
│
├── db/                        # Database utilities
├── config/                    # Configuration
├── cli.py                     # CLI entry point
├── sql_runner.py              # SQL execution
├── MANIFEST_CONTRACT.md       # What ORTHON delivers (READ THIS)
│
└── _legacy/                   # Deprecated, do not use
    ├── runner.py
    └── python_runner.py
```

---

## Input: manifest.yaml (from ORTHON)

**Read MANIFEST_CONTRACT.md for the full specification.**

Quick reference — for each signal, the manifest tells PRISM:
- `engines`: which signal-level engines to run
- `rolling_engines`: which rolling engines to run
- `window_size`: samples per window
- `stride`: samples between windows
- `derivative_depth`: max derivative order (0, 1, or 2)
- `eigenvalue_budget`: max eigenvalues to compute
- `output_hints`: how to format engine output (per_bin vs summary, etc.)

PRISM executes what the manifest says. No more, no less.

---

## Input: observations.parquet Schema (v2.4)

### Required Columns
| Column | Type | Description |
|--------|------|-------------|
| signal_id | str | What signal (temp, pressure, etc.) |
| I | UInt32 | Sequential index 0,1,2,3... per signal_id |
| value | Float64 | The measurement |

### Optional Columns
| Column | Type | Description |
|--------|------|-------------|
| cohort | str | Grouping key (engine_1, pump_A) - cargo only |

### I is Canonical
```
CORRECT:
signal_id | I | value
----------|---|------
temp      | 0 | 45.2
temp      | 1 | 45.4
temp      | 2 | 45.6

WRONG (timestamps):
signal_id | I          | value
----------|------------|------
temp      | 1596760568 | 45.2
```

### unit_id is Cargo
**unit_id has ZERO effect on compute.**
- DO NOT include unit_id in groupby operations
- Group by I or signal_id only
- unit_id passes through for ORTHON reporting

---

## Key Rules

1. **PRISM computes, ORTHON classifies** — no labels, no thresholds in PRISM
2. **Typology lives in ORTHON** — PRISM receives manifest.yaml
3. **state_vector = centroid, state_geometry = eigenvalues** — separate concerns
4. **Scale-invariant features only** — no absolute values (deprecated: rms, peak, mean, std)
5. **I is canonical** — sequential per signal_id, not timestamps
6. **unit_id is cargo** — never in groupby
7. **Read MANIFEST_CONTRACT.md** — not ORTHON source code

---

## Output Files

### From PRISM
| File | Stage | Description |
|------|-------|-------------|
| signal_vector.parquet | 1 | Per-signal features |
| state_vector.parquet | 2 | Centroids (position) |
| state_geometry.parquet | 2 | Eigenvalues (shape) |
| signal_geometry.parquet | 2 | Signal-to-centroid |
| signal_pairwise.parquet | 3 | Pairwise relationships |
| geometry_dynamics.parquet | 4 | State derivatives |
| signal_dynamics.parquet | 4 | Signal derivatives |
| pairwise_dynamics.parquet | 4 | Pairwise derivatives |
| lyapunov.parquet | 4 | Lyapunov exponents |
| dynamics.parquet | 4 | RQA, attractor |
| information_flow.parquet | 4 | Transfer entropy |
| zscore.parquet | SQL | Normalized |
| statistics.parquet | SQL | Summary stats |
| correlation.parquet | SQL | Correlation matrix |

---

## Key Separation: state_vector vs state_geometry

| File | Computes | Contains |
|------|----------|----------|
| state_vector.py | Centroid (mean) | centroid_*, mean_distance |
| state_geometry.py | SVD on features | eigenvalue_*, effective_dim |

**Eigenvalues ONLY in state_geometry.py. Never in state_vector.py.**

---

## Do NOT

- Put classification logic in PRISM
- Modify ORTHON code or repository
- Compute eigenvalues in state_vector (they belong in state_geometry)
- Use deprecated scale-dependent engines (rms, peak, mean, std)
- Include unit_id in groupby operations
- Create typology in PRISM (ORTHON's job)
- Second-guess the manifest's engine selections or window sizes
- Write `if signal_type == ...` anywhere in PRISM

---

## Credits

- **Avery Rudder** — "Laplace transform IS the state engine" — eigenvalue insight
