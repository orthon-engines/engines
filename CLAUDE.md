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

## Input: observations.parquet (Schema v2.0)

### Required Columns
| Column | Type | Description |
|--------|------|-------------|
| signal_id | str | What signal (temp, pressure, return) |
| I | UInt32 | Index 0,1,2,3... per unit/signal. Sequential, no gaps. |
| value | Float64 | The measurement |

### Optional Columns
| Column | Type | Description |
|--------|------|-------------|
| unit_id | str | Label for grouping. Blank is fine. "bananas" is fine. |

### About unit_id

**unit_id is OPTIONAL.** It's just a sticky note for humans.

- PRISM passes unit_id through to output for SQL filtering
- unit_id has ZERO effect on compute
- unit_id can be blank, null, "pump_1", "friday_data", "bananas" - whatever
- unit_id is NOT an index (that's what I is for)
- If no unit_id provided, PRISM uses blank ""

**DO NOT validate unit_id contents. DO NOT require unit_id.**

### Example
```
unit_id | signal_id | I | value
--------|-----------|---|------
pump_1  | temp      | 0 | 45.2
pump_1  | temp      | 1 | 45.4
pump_1  | pressure  | 0 | 101.3
pump_1  | pressure  | 1 | 101.5
        | temp      | 0 | 30.1   ← blank unit_id is fine
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
6. **Slow compute is fine.** Rolling Lyapunov, Hurst, entropy are O(n²). Hours/days acceptable for publication-grade results.

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
