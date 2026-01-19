# PR: ML Acceleration Layer

**Branch:** `main`
**Type:** Feature Addition
**Priority:** After core pipeline is working

---

## Summary

Add ML acceleration layer to PRISM. One parquet file with all features, user picks their framework. PRISM provides the behavioral geometry, user brings the model.

**The Product Tiers:**
| Tier | Gets | Price |
|------|------|-------|
| API | Reports (JSON/PDF) | Per-run |
| Data | Reports + Parquet | Subscription |
| Enterprise | Full integration | Custom |

---

## New Entry Points

### ml_features.py

Combines all PRISM layers into one ML-ready parquet. One row per entity, all features denormalized.

```bash
# Generate features
python -m prism.entry_points.ml_features

# With target variable (for supervised learning)
python -m prism.entry_points.ml_features --target RUL

# Testing
python -m prism.entry_points.ml_features --testing --limit 10
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--target COL` | Target column for supervised learning (e.g., RUL, fault_type) |
| `--entity COL` | Entity column (auto-detected if not specified) |
| `--testing` | Enable test mode |
| `--limit N` | [TESTING] Max entities to process |

**Reads:** `data/vector.parquet`, `data/geometry.parquet`, `data/state.parquet`
**Writes:** `data/ml_features.parquet`

---

### ml_train.py

Train ML model on PRISM features. Supports multiple frameworks.

```bash
# Train with XGBoost (default)
python -m prism.entry_points.ml_train

# Train with other frameworks
python -m prism.entry_points.ml_train --model catboost
python -m prism.entry_points.ml_train --model lightgbm
python -m prism.entry_points.ml_train --model randomforest
python -m prism.entry_points.ml_train --model gradientboosting

# Hyperparameter tuning
python -m prism.entry_points.ml_train --tune

# Cross-validation
python -m prism.entry_points.ml_train --cv 5

# Testing
python -m prism.entry_points.ml_train --testing
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--model NAME` | ML framework: xgboost, catboost, lightgbm, randomforest, gradientboosting |
| `--tune` | Run hyperparameter tuning |
| `--cv N` | N-fold cross-validation |
| `--split RATIO` | Train/test split ratio (default: 0.8) |
| `--testing` | Enable test mode |

**Reads:** `data/ml_features.parquet`
**Writes:** `data/ml_model.pkl`, `data/ml_results.parquet`, `data/ml_importance.parquet`

---

## Data Files

### data/ml_features.parquet

One row per entity, all features from all layers.

```
entity_id | target | vector_* | geometry_* | state_*
----------+--------+----------+------------+---------
engine_001| 112    | ...      | ...        | ...
engine_002| 98     | ...      | ...        | ...
```

**Feature Groups:**

| Prefix | Source | Description |
|--------|--------|-------------|
| `vector_*` | vector.parquet | 51 behavioral metrics (mean, std, last per metric) |
| `geometry_*` | geometry.parquet | Cohort structure (PCA, clustering, MST, LOF) |
| `state_*` | state.parquet | Temporal dynamics (velocity, acceleration) |
| `target` | observations.parquet | Label for supervised learning |

**Total: ~100-150 features per entity**

---

### data/ml_results.parquet

Predictions vs actuals for test set.

```
entity_id | actual | predicted | error | abs_error
----------+--------+-----------+-------+----------
engine_001| 112    | 108.5     | 3.5   | 3.5
engine_002| 98     | 102.1     | -4.1  | 4.1
```

---

### data/ml_importance.parquet

Feature importance from trained model.

```
feature                    | importance | importance_pct | cumulative_pct
---------------------------+------------+----------------+----------------
vector_hilbert_freq_std    | 0.152      | 15.2%          | 15.2%
vector_hurst_mean          | 0.089      | 8.9%           | 24.1%
geometry_pca_var_pc1       | 0.067      | 6.7%           | 30.8%
...
```

---

## Updated Pipeline

```
fetch → cohort → signal_vector → geometry → state → ml_features → ml_train
                                                         ↓             ↓
                                                  ml_features     ml_model.pkl
                                                   .parquet       ml_results.parquet
                                                                  ml_importance.parquet
```

---

## Updated Data Directory

```
data/
├── observations.parquet   # Raw + metadata
├── vector.parquet         # 51 metrics per signal
├── geometry.parquet       # Pairwise + structure
├── state.parquet          # Temporal dynamics
├── ml_features.parquet    # ML-ready features (one row per entity)
├── ml_model.pkl           # Trained model
├── ml_model.json          # Model metadata (RMSE, params)
├── ml_results.parquet     # Predictions vs actuals
└── ml_importance.parquet  # Feature importance
```

---

## Implementation Checklist

### Phase 1: Add Files

- [ ] Copy `entry_points/ml_features.py` to `prism/entry_points/`
- [ ] Copy `entry_points/ml_train.py` to `prism/entry_points/`

### Phase 2: Update parquet_store.py

Add new file paths to `get_path()`:

```python
PATHS = {
    'observations': 'data/observations.parquet',
    'vector': 'data/vector.parquet',
    'geometry': 'data/geometry.parquet',
    'state': 'data/state.parquet',
    'ml_features': 'data/ml_features.parquet',
    'ml_model': 'data/ml_model',  # .pkl and .json
    'ml_results': 'data/ml_results.parquet',
    'ml_importance': 'data/ml_importance.parquet',
}
```

### Phase 3: Update entry_points/__init__.py

Add new entry points to the registry:

```python
ENTRY_POINTS = {
    # ... existing ...
    'ml_features': {
        'module': 'prism.entry_points.ml_features',
        'goal': 'Generate ML-ready feature parquet',
        'inputs': ['vector.parquet', 'geometry.parquet', 'state.parquet'],
        'outputs': ['ml_features.parquet'],
    },
    'ml_train': {
        'module': 'prism.entry_points.ml_train',
        'goal': 'Train ML model on PRISM features',
        'inputs': ['ml_features.parquet'],
        'outputs': ['ml_model.pkl', 'ml_results.parquet', 'ml_importance.parquet'],
    },
}
```

### Phase 4: Update requirements.txt

Add optional ML dependencies:

```
# ML Frameworks (optional - install as needed)
xgboost>=1.7.0
catboost>=1.2.0
lightgbm>=4.0.0
```

### Phase 5: Verify

```bash
# Generate features
python -m prism.entry_points.ml_features --target RUL

# Train with each framework
python -m prism.entry_points.ml_train --model xgboost
python -m prism.entry_points.ml_train --model lightgbm
python -m prism.entry_points.ml_train --model catboost

# Check outputs exist
ls -la data/ml_*.parquet
ls -la data/ml_model.*
```

---

## C-MAPSS Validation

After implementation, validate on C-MAPSS FD001:

```bash
# Full pipeline
python -m prism.entry_points.fetch
python -m prism.entry_points.cohort
python -m prism.entry_points.signal_vector
python -m prism.entry_points.geometry
python -m prism.entry_points.state
python -m prism.entry_points.ml_features --target RUL
python -m prism.entry_points.ml_train --model xgboost

# Expected output:
# Test RMSE: ~6-7 (should match previous 6.43 benchmark)
# Top features: hilbert_*, hurst_*, entropy_*
```

---

## The Math is the Moat

| What User Gets | What PRISM Provides |
|----------------|---------------------|
| ml_features.parquet | 100+ behavioral geometry features |
| Feature importance | Hilbert inst_freq dominates (47%) |
| RMSE 6.43 | Beats LSTM/CNN on C-MAPSS |
| Any ML framework | XGBoost, CatBoost, LightGBM, etc. |
| Mac Mini | No GPU required |
| Interpretable | Every feature has physical meaning |

**Reports are the interface. Parquet is the product. Math is the moat.** 🔔

---

## Future: API Endpoint

```python
# Future implementation
@app.post("/ml/features")
async def generate_features(
    observations: UploadFile,
    target: str = None
):
    """Upload observations, get ML-ready features."""
    pass

@app.post("/ml/train")  
async def train_model(
    model: str = "xgboost",
    tune: bool = False
):
    """Train model on uploaded features."""
    pass

@app.get("/ml/predict")
async def predict(entity_id: str):
    """Get prediction for entity."""
    pass
```

---

## Notes for Claude Code

1. **Dynamic imports** - ML frameworks are imported only when used (avoid requiring all deps)
2. **Feature naming** - Prefix with source layer: `vector_`, `geometry_`, `state_`
3. **Null handling** - Fill nulls with 0 (safe for tree models)
4. **Entity detection** - Auto-detect entity column (engine_id, unit_id, etc.)
5. **No --domain flag** - One domain at a time, hardcoded paths

---

## CLI Summary (Final)

```bash
# Core pipeline
python -m prism.entry_points.fetch
python -m prism.entry_points.cohort
python -m prism.entry_points.signal_vector
python -m prism.entry_points.geometry
python -m prism.entry_points.state

# ML acceleration
python -m prism.entry_points.ml_features --target RUL
python -m prism.entry_points.ml_train --model xgboost

# Compare models
python -m prism.entry_points.ml_train --model catboost
python -m prism.entry_points.ml_train --model lightgbm
python -m prism.entry_points.ml_train --model randomforest
```

**7 entry points. Clean pipeline. User picks the model.**

*Compute once, query forever. More cowbell.* 🔔
