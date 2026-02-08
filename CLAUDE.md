# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## ⛔ STOP: MANDATORY RULES — READ BEFORE EVERY ACTION

### Rule 0: SEARCH BEFORE YOU CREATE

**Before writing ANY new code, you MUST search the repo for existing implementations.**

```bash
# Find existing files
find . -name "*.py" | xargs grep -l "function_name"

# Find existing patterns
grep -r "def compute" engines/manifold/

# Find how similar things are done
grep -r "sample_rate" engines/
```

**If you think something doesn't exist, ASK THE USER before creating it.**

### Rule 1: USE EXISTING CODE

If a function, engine, or pattern exists in the repo, **USE IT**. Do not recreate.

```
WRONG: "I'll write a quick FFT function..."
RIGHT: "I see engines/primitives/spectral.py has psd() — using that."

WRONG: "Let me create a runner to orchestrate this..."
RIGHT: "I see engines/signal_vector/runner.py handles this — adding to it."
```

### Rule 2: NO ROGUE FILE CREATION

| Location | Allowed? |
|----------|----------|
| `/tmp/` | ❌ NEVER |
| `~/` | ❌ NEVER |
| Random standalone scripts | ❌ NEVER |
| Inside existing repo structure | ✅ With approval |

**If you create something in /tmp, you are hiding evidence. This is forbidden.**

### Rule 3: SHOW YOUR WORK BEFORE CHANGES

Before modifying any file, show:
1. The existing file/function you're modifying
2. The existing pattern you're following
3. Get explicit approval before creating NEW files

```
WRONG: *silently creates new_runner.py*
RIGHT: "I found runner.py at engines/signal_vector/runner.py.
        I'll add the new function there following the existing pattern.
        Here's the current structure: [shows code]
        Okay to proceed?"
```

### Rule 4: ENGINES COMPUTES, ORTHON CLASSIFIES

- Do NOT create typology logic, classification rules, or signal labels in ENGINES
- Do NOT modify observations.parquet or typology.parquet
- Do NOT second-guess manifest.yaml — execute what it says
- If MANIFEST_CONTRACT.md doesn't answer your question, ASK THE USER

**If you find yourself writing `if signal_type == 'PERIODIC'` in ENGINES, STOP.
That is classification. Classification belongs in ORTHON.**

---

## 🚫 EXPLICITLY FORBIDDEN BEHAVIORS

| Behavior | Why It's Forbidden |
|----------|-------------------|
| Create scripts in /tmp | Hides work, not verifiable |
| Create "one-off" runners | Bypasses established patterns |
| Inline compute in orchestrators | Engines compute, orchestrators orchestrate |
| Duplicate existing functionality | Creates inconsistency |
| Create new venv | Use existing `./venv/` |
| Guess at implementations | Ask if unsure |
| Generate code without showing existing | Must show what exists first |

### The /tmp Rule (CRITICAL)

```
/tmp is where code goes to avoid accountability.

You will NEVER:
- Write scripts to /tmp
- Write data to /tmp
- Write anything to /tmp

EVERYTHING stays in the repository where it can be reviewed.
```

### The One-Off Runner Rule (CRITICAL)

```
WRONG:
  "I'll create a quick script to do this..."
  "Let me write a standalone runner..."
  "Here's a temporary solution..."

RIGHT:
  "I found the existing runner at [path]. Adding to it."
  "This matches the pattern in [existing file]. Following that."
  "The manifest contract says [X]. Implementing exactly that."
```

---

## ✅ ALLOWED BEHAVIORS

| Behavior | How To Do It |
|----------|--------------|
| Call existing engines | `engine_registry[name](window)` |
| Sequence operations | Chain existing entry points |
| Pass config from manifest | Read and forward, don't interpret |
| Add to existing files | Show the file first, get approval |
| Create new engine | Follow existing engine pattern, propose location first |

---

## Architecture

Orthon Engines is a domain-agnostic dynamical systems analysis platform.
- **orthon-engines/orthon** — dynamical systems analysis interpreter (brain)
- **orthon-engines/engines** — dynamical systems computation engines (muscle)

### Pipeline (24 stages)

```
observations.parquet + typology.parquet + manifest.yaml  (from Orthon)
        │
        ▼
  Core (stages 00-14):
    00 breaks           01 signal_vector     02 state_vector
    03 state_geometry    04 cohorts           05 signal_geometry
    06 signal_pairwise   07 geometry_dynamics 08 ftle
    09 dynamics          10 information_flow  11 topology
    12 zscore            13 statistics        14 correlation

  Atlas (stages 15-23, --atlas flag):
    15 ftle_field         16 break_sequence    17 ftle_backward
    18 segment_comparison 19 info_flow_delta   20 geometry_full
    21 velocity_field     22 ftle_rolling      23 ridge_proximity
```

---

## Engine Minimum Sample Requirements

**FFT-based engines require larger windows. This is physics, not a bug.**

| Engine | Minimum Samples | Reason |
|--------|-----------------|--------|
| spectral | 64 | FFT resolution |
| harmonics | 64 | FFT resolution |
| fundamental_freq | 64 | FFT resolution |
| thd | 64 | FFT resolution |
| sample_entropy | 64 | Statistical validity |
| hurst | 128 | Long-range dependence |
| crest_factor | 4 | Simple ratio |
| kurtosis | 4 | 4th moment |
| skewness | 4 | 3rd moment |
| perm_entropy | 8 | Permutation patterns |
| acf_decay | 16 | Lag structure |
| snr | 32 | Power estimation |
| phase_coherence | 32 | Phase estimation |

**When system window < engine minimum:**
- Manifest specifies `engine_window_overrides`
- ENGINES uses expanded window for that engine
- I (window end index) alignment is preserved

**Do NOT lower engine minimums to fit small windows. The math doesn't work.**

---

## Directory Structure

```
~/engines/
├── CLAUDE.md                     # This file
├── README.md                     # Project docs
├── LICENSE.md                    # PolyForm Noncommercial 1.0.0
├── pyproject.toml                # Package: orthon-engines
├── requirements.txt              # Dev dependencies
├── requirements.web.txt          # Web service dependencies (FastAPI)
├── Dockerfile                    # Web service container
├── fly.toml                      # Fly.io deployment config
├── .dockerignore
├── server.py                     # FastAPI web service (CSV upload → atlas)
├── static/index.html             # Web UI (registration + upload)
│
├── engines/                      # Main Python package
│   ├── __init__.py               # Public API (180+ functions)
│   ├── cli.py                    # CLI: run, inspect, explore, atlas
│   ├── input_loader.py           # Auto-detect CSV/parquet, generate manifest
│   │
│   ├── entry_points/             # Stage orchestrators (24 stages)
│   │   ├── run_pipeline.py       # Full pipeline orchestrator
│   │   ├── stage_00_breaks.py    # → breaks.parquet
│   │   ├── stage_01_signal_vector.py  # → signal_vector.parquet
│   │   ├── stage_02_state_vector.py   # → state_vector.parquet
│   │   ├── stage_03_state_geometry.py # → state_geometry.parquet
│   │   ├── stage_04_cohorts.py        # → cohorts.parquet
│   │   ├── stage_05-14_*.py           # geometry, dynamics, sql
│   │   └── stage_15-23_*.py           # Atlas engines
│   │
│   ├── manifold/                 # Compute engines (numpy → dicts)
│   │   ├── signal/               # 38 per-signal engines (spectral, hurst, entropy, etc.)
│   │   ├── state/                # Centroid, eigendecomposition (SVD)
│   │   ├── dynamics/             # FTLE, Lyapunov, attractor, saddle detection
│   │   ├── geometry/             # Signal-to-state relationships
│   │   ├── pairwise/             # Correlation, Granger causality
│   │   ├── parallel/             # Parallel runners (dynamics, info flow, topology)
│   │   ├── sql/                  # SQL engines (zscore, statistics, correlation)
│   │   ├── registry.py           # Engine discovery & loading
│   │   ├── base.py               # BaseEngine class
│   │   └── rolling.py            # Rolling window wrapper
│   │
│   ├── primitives/               # Pure math (numpy → float)
│   │   ├── individual/           # Statistics, spectral, entropy, derivatives, etc.
│   │   ├── embedding/            # Delay embedding, Cao's method, AMI
│   │   ├── dynamical/            # FTLE, Lyapunov, RQA, saddle
│   │   ├── pairwise/             # Correlation, causality, distance
│   │   ├── matrix/               # SVD, covariance, graph Laplacian
│   │   ├── information/          # Transfer entropy, mutual info, divergence
│   │   ├── network/              # Centrality, community, paths
│   │   ├── topology/             # Persistent homology, distance
│   │   └── tests/                # Bootstrap, hypothesis, null models
│   │
│   ├── server/                   # HTTP server (streaming protocol)
│   ├── stream/                   # Streaming I/O (parser, buffer, writer)
│   ├── validation/               # Input validation & prerequisites
│   ├── config/                   # Metric requirements
│   └── tests/                    # Internal tests
│
├── config/                       # Domain & environment YAML configs
├── tests/                        # Test suite
└── docs/                         # Documentation
```

### Where New Code Goes

| Type of Code | Location | Pattern to Follow |
|--------------|----------|-------------------|
| New signal engine | `engines/manifold/signal/` | Copy `kurtosis.py` pattern |
| New rolling engine | `engines/manifold/rolling/` | Copy existing rolling pattern |
| New primitive | `engines/primitives/individual/` | Pure function, no I/O |
| New stage | ASK FIRST | Probably doesn't need new stage |

---

## Input: manifest.yaml (from ORTHON)

**Read MANIFEST_CONTRACT.md for the full specification.**

Quick reference — for each signal, the manifest tells ENGINES:
- `engines`: which signal-level engines to run
- `rolling_engines`: which rolling engines to run
- `window_size`: samples per window (system default)
- `stride`: samples between windows
- `engine_window_overrides`: per-engine window sizes (when different from system)
- `derivative_depth`: max derivative order (0, 1, or 2)
- `eigenvalue_budget`: max eigenvalues to compute
- `output_hints`: how to format engine output (per_bin vs summary, etc.)

ENGINES executes what the manifest says. No more, no less.

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
- Sequential integers per signal_id
- NOT timestamps
- Starts at 0, no gaps

### cohort is Cargo
- ZERO effect on compute
- Never in groupby
- Passes through for reporting

---

## Key Rules Summary

1. **SEARCH BEFORE CREATE** — find existing code first
2. **USE EXISTING PATTERNS** — don't reinvent
3. **NO /tmp** — everything in repo
4. **NO ONE-OFF RUNNERS** — use established orchestrators
5. **ENGINES computes, ORTHON classifies** — no labels in ENGINES
6. **state_vector = centroid, state_geometry = eigenvalues** — separate concerns
7. **Scale-invariant features only** — no absolute values
8. **I is canonical** — sequential per signal_id
9. **ASK IF UNSURE** — don't guess

---

## Checklist Before Any Change

```
□ Did I search for existing implementations?
□ Am I using existing patterns/files?
□ Did I show the user what I'm modifying?
□ Is this going in the repo (not /tmp)?
□ Am I following MANIFEST_CONTRACT.md?
□ Did I get approval for new files?
```

**If any answer is NO, stop and fix it.**

---

## Error Handling

**Engines must fail loudly, not silently.**

```python
# WRONG - silent failure
def compute(y):
    try:
        result = complex_math(y)
    except:
        pass  # Silent! BAD!
    return result

# RIGHT - loud failure
def compute(y):
    if len(y) < MIN_SAMPLES:
        raise ValueError(f"Need {MIN_SAMPLES} samples, got {len(y)}")
    return complex_math(y)
```

If an engine can't run (insufficient samples, bad data), it should:
1. Raise an exception, OR
2. Return explicit NaN with logged warning

Never silently return garbage.

---

## Credits

- **Avery Rudder** — "Laplace transform IS the state engine" — eigenvalue insight
