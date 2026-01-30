# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## CRITICAL: PRISM ↔ ORTHON Architecture

**PRISM is an HTTP service ONLY. NOT a pip install. NO code sharing with ORTHON.**

```
┌─────────────────┐         HTTP          ┌─────────────────┐
│     ORTHON      │ ──────────────────▶   │      PRISM      │
│   (Frontend)    │   POST /compute       │  (Compute API)  │
│   Streamlit     │ ◀──────────────────   │  localhost:8100 │
│                 │   {status, parquets}  │                 │
└─────────────────┘                       └─────────────────┘
        │                                         │
        │ reads                                   │ writes
        ▼                                         ▼
   ~/prism/data/*.parquet                  ~/prism/data/*.parquet
```

**ORTHON creates observations.parquet and manifest.yaml. PRISM only reads them.**

---

## What PRISM Is

Pure computation. No decisions. No interpretation.
Read manifest → Run ALL engines → Write parquets.

## The One Command

```bash
python -m prism manifest.yaml
# or
python -m prism observations.parquet
```

That's it. Everything runs. 100%.

---

## Input: observations.parquet (PRISM Format)

PRISM expects THIS format. No exceptions.

| Column | Type | Description |
|--------|------|-------------|
| entity_id | str | Which entity (pump, bearing, industry) |
| I | UInt32 | Observation index within entity |
| signal_id | str | Which signal (temp, pressure, return) |
| value | Float64 | The measurement |

Example:
```
entity_id | I | signal_id | value
----------|---|-----------|------
pump_1    | 0 | temp      | 45.2
pump_1    | 0 | pressure  | 101.3
pump_1    | 1 | temp      | 45.4
pump_1    | 1 | pressure  | 101.5
```

**If data is not in this format, ORTHON must transform it first.**

---

## Output: 12 Parquet Files

### Geometry (structure)
- `primitives.parquet` - Signal-level metrics
- `primitives_pairs.parquet` - Directed pair metrics
- `geometry.parquet` - Symmetric pair metrics
- `topology.parquet` - Betti numbers, persistence
- `manifold.parquet` - Embedding metrics

### Dynamics (change)
- `dynamics.parquet` - Lyapunov, RQA, Hurst
- `information_flow.parquet` - Transfer entropy, Granger
- `observations_enriched.parquet` - Rolling window metrics

### Energy (physics)
- `physics.parquet` - Entropy, energy, free energy

### SQL Reconciliation
- `zscore.parquet` - Normalized metrics
- `statistics.parquet` - Summary statistics
- `correlation.parquet` - Correlation matrix
- `regime_assignment.parquet` - State labels

---

## Engine Execution Order

1. Signal engines (per signal)
2. Pair engines (directed: A→B)
3. Symmetric pair engines (undirected: A↔B)
4. Windowed engines (rolling computations)
5. Dynamics runner (Lyapunov, RQA)
6. Topology runner (Betti)
7. Information flow runner (transfer entropy)
8. Physics runner (energy, entropy)
9. SQL engines (zscore, statistics, etc.)

---

## Rules

1. **ALL engines run. Always. No exceptions.**
2. Insufficient data → return NaN, never skip
3. No domain-specific logic in PRISM
4. No interpretation in PRISM
5. RAM managed via batching (see ram_manager.py)

---

## Do NOT

- Skip engines based on domain
- Gate metrics by observation count
- Make decisions about what to compute
- Interpret results
- Add CLI flags for engine selection
- Create observations.parquet (ORTHON's job)
- Create manifest.yaml (ORTHON's job)

---

## Key Files

| File | Purpose |
|------|---------|
| `~/prism/prism/runner.py` | Main runner (Geometry→Dynamics→Energy→SQL) |
| `~/prism/prism/python_runner.py` | Signal/pair/windowed engines |
| `~/prism/prism/sql_runner.py` | SQL reconciliation engines |
| `~/prism/prism/ram_manager.py` | RAM-optimized batch processing |
| `~/prism/prism/cli.py` | CLI entry point |
| `~/prism/data/observations.parquet` | Input (ORTHON creates) |
| `~/prism/data/manifest.yaml` | Config (ORTHON creates) |

---

## Technical Stack

- **Language:** Python 3.10+
- **Storage:** Parquet files (columnar, compressed)
- **DataFrame:** Polars (primary), Pandas (engine compatibility)
- **Core:** NumPy, SciPy, scikit-learn
- **Specialized:** antropy, nolds, pyrqa, arch, PyWavelets, networkx

---

## Directory Structure

```
~/prism/
├── CLAUDE.md
├── venv/
├── data/
│   ├── observations.parquet   ← ORTHON creates
│   └── manifest.yaml          ← ORTHON creates
└── prism/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py
    ├── runner.py
    ├── python_runner.py
    ├── sql_runner.py
    ├── ram_manager.py
    └── engines/
```

## Session Recovery

```bash
# Start PRISM (from repo root, using venv)
cd ~/prism
./venv/bin/python -m prism data/manifest.yaml

# Or via API:
./venv/bin/python -m prism.entry_points.api --port 8100
curl http://localhost:8100/health
```

---

## DO NOT TOUCH

- ORTHON code lives in `~/orthon/` - let CC ORTHON handle it
- Never `pip install prism` - PRISM is HTTP only
- Never create observations.parquet or manifest.yaml - ORTHON's job
