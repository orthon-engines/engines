# PR: Add ML Ablation Study Entry Point (v2)

## Purpose

Add `ml_ablation.py` to validate each PRISM layer's contribution to prediction accuracy AND demonstrate PRISM's "blind discovery" capability. This is the **"give me your mystery data and I'll tell you what you have"** demo.

Two validation modes:
1. **Layer Ablation** — proves PRISM adds value (raw → vector → geometry → state)
2. **Cohort Discovery** — shows PRISM discovered physical system structure from unlabeled data

## Why This Matters

- **Proves PRISM value to customers** — shows concrete RMSE improvement
- **Identifies which layers matter** — Vector? Geometry? State?
- **Catches regressions** — if a refactor breaks something, RMSE will show it
- **Publication evidence** — quantitative validation for academic papers
- **The Jeff Dick Demo** — "Give me unlabeled sensor data, I'll tell you what physical systems you have"

## Expected Results

```
======================================================================
PRISM ABLATION STUDY v2
With Cohort Discovery Validation
======================================================================
Target: RUL

======================================================================
COHORT DISCOVERY: What PRISM Found
======================================================================
Number of cohorts discovered: 10

  raw_cohort_8: Operational/control parameters
     Signals: BPR, P15, P2, PCNfR_dmd, Ps30 ... (+7 more)
  raw_cohort_4: Temperature sensors
     Signals: P30, T2, T24, htBleed, phi
  raw_cohort_5: Fan spool speed
     Signals: NRf, Nf, Nf_dmd
  raw_cohort_6: Hot section temperatures
     Signals: T30, T50
  raw_cohort_0: Core spool speed
     Signals: NRc, Nc
  raw_cohort_1: Health/failure indicators
     Signals: RUL

======================================================================
LAYER ABLATION
======================================================================

Stage 0: Raw Observations Only
  Features: 63 (21 sensors × 3 stats)
  RMSE: 18.42
  
Stage 2: + Vector Metrics (51 behavioral)
  Features: 156 (+ mean/std/last per metric)
  RMSE: 10.23 (Δ -8.19) ← BIG JUMP
  Top features: ['v_hilbert_inst_freq_mean', 'v_hurst_mean', 'v_entropy_mean']

Stage 3: + Geometry (coupling, structure)
  Features: 189 (+ PCA, coupling)
  RMSE: 7.81 (Δ -2.42)

Stage 4: + State (velocity, acceleration)
  Features: 205
  RMSE: 6.43 (Δ -1.38)

======================================================================
SUMMARY
======================================================================
Raw → Full PRISM: 18.42 → 6.43 (65% reduction)
Biggest contributor: 2_vector (Δ -8.19 RMSE)

KEY INSIGHT:
  PRISM discovered physical system structure from unlabeled data.
  'Give me your mystery sensors — I'll tell you what you have.'
```

## File to Create

```
prism/entry_points/ml_ablation.py
```

## Usage

```bash
# Basic ablation
python -m prism.entry_points.ml_ablation --target RUL

# With per-cohort analysis (shows which discovered groupings matter)
python -m prism.entry_points.ml_ablation --target RUL --cohort-ablation

# Custom entity column
python -m prism.entry_points.ml_ablation --target RUL --entity unit_id

# Export results
python -m prism.entry_points.ml_ablation --target RUL --output results.json
```

## Data Sources (Read-Only)

The script reads from existing Parquet files — no new tables:

| Stage | Parquet Path | Schema |
|-------|--------------|--------|
| 0 | `raw/observations.parquet` | mean/std/last per signal |
| 1 | `config/cohorts.parquet` | cohort_id as categorical |
| 2 | `vector/signal.parquet` | 51 behavioral metrics |
| 3 | `geometry/signal_pair.parquet` | PCA, coupling, structure |
| 4 | `state/signal_pair.parquet` | velocity, acceleration |

## Key Implementation Notes

1. **Use `get_parquet_path()`** from `prism.db.parquet_store`
2. **Auto-detect entity column** — check for `entity_id`, `engine_id`, `unit_id`, `unit`
3. **Handle missing layers gracefully** — skip stages if Parquet doesn't exist
4. **No writes** — this is analysis only, no side effects

## Dependencies

```python
# Already in project
import polars as pl
from prism.db.parquet_store import get_parquet_path

# External (standard ML stack)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
```

## Interpretation Guide

| If... | Then... |
|-------|---------|
| Raw → Vector is big jump | Behavioral metrics matter (Hilbert, Hurst) |
| Vector → Geometry is small | Cohort structure adds marginal value |
| Geometry → State is big | Temporal dynamics (velocity) are key |
| Any stage increases RMSE | That layer is adding noise, fix it |

## Checklist

- [ ] Create `prism/entry_points/ml_ablation.py`
- [ ] Test on C-MAPSS data with `--target RUL`
- [ ] Verify all stages run without errors
- [ ] Confirm RMSE progression makes sense
- [ ] Optional: Add `--output` JSON export

## Notes

- This is the **ML Ablation** study (layer-by-layer validation)
- Different from the earlier **Agent Ablation** study (agent vs direct execution)
- Run after full pipeline has generated all Parquet files
