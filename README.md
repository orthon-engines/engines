# PRISM

**PRISM**

```bash
pip install prism-signal
```

---

## What It Does

PRISM transforms raw observations into computed primitives and mathematical calculations. For use with Orthon Data Interpreter.

```
observations.parquet     Raw measurements
       ↓
signal_vector.parquet    Per-signal metrics
       ↓
state_vector.parquet     System centroid (position)
       ↓
state_geometry.parquet   Eigenstructure (shape)
```


---

## Quick Start

```bash
# Run the pipeline
python -m prism.entry_points.stage_01_signal_vector manifest.yaml
python -m prism.entry_points.stage_02_state_vector signal_vector.parquet typology.parquet
python -m prism.entry_points.stage_03_state_geometry signal_vector.parquet state_vector.parquet
```

Or use Python directly:

```python
from prism.entry_points.stage_01_signal_vector import run

df = run(
    observations_path="observations.parquet",
    output_path="signal_vector.parquet",
    manifest=manifest,
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ENTRY POINTS                           │
│              Orchestration: parquet → parquet               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        ENGINES                              │
│         Orchestrate primitives, return dicts                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      PRIMITIVES                             │
│              Pure math: numpy → float/array                 │
└─────────────────────────────────────────────────────────────┘
```

PRISM produces data. For classification and visualization, see [ORTHON](https://github.com/prism-engines/orthon).

---

## Features Computed

**Per-signal (stage_01):**
- Shape: kurtosis, skewness, crest factor
- Memory: ACF decay, Hurst exponent
- Spectral: entropy, centroid, dominant frequency, bandwidth
- Complexity: permutation entropy, sample entropy
- Trend: slope, R², CUSUM

**Cross-signal (stage_02, stage_03):**
- State vector: centroid position in feature space
- Eigenvalues: variance distribution across principal components
- Effective dimension: participation ratio
- Condition number: spectral spread

---

## Input Format

PRISM expects `observations.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| `signal_id` | string | Unique signal identifier |
| `I` | int | Sequential index (0, 1, 2, ...) |
| `value` | float | Measurement value |
| `cohort` | string | (optional) Grouping for multi-unit analysis |

Plus a `manifest.yaml` specifying window size, stride, and engines per signal.

---

## Requirements

- Python 3.10+
- numpy, polars, scipy
- joblib (parallel processing)

---

## Citation

```bibtex
@software{prism2026,
  title = {PRISM: Signal Computation},
  author = {Rudder, Jason and Rudder, Avery},
  year = {2026},
  url = {https://github.com/prism-engines/prism}
}
```

---

## License

MIT