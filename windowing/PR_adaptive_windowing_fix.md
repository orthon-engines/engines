# PR: Fix Adaptive Windowing + Ensure Data Recording

## Problem

Two issues discovered during C-MAPSS pipeline run:

1. **Hardcoded window sizes too large** — 252-sample anchor tier exceeds entity length for short-lived engines (~192 cycles), resulting in zero windows generated

2. **Data not being recorded** — Even when `--adaptive` increases min_obs, neither fast pathway nor adaptive pathway writes output data

## Root Cause

```
Entity lifecycle: ~150-300 cycles (varies by failure mode)
Anchor tier: 252 samples (hardcoded from financial domain)
Result: Entities with <252 observations → 0 windows → no geometry → state fails
```

## Solution

### 1. Dynamic Window Sizing (Observation-Count Based)

Replace hardcoded tiers with entity-relative windows:

```python
# prism/utils/adaptive_windows.py

def compute_adaptive_windows(
    entity_length: int,
    min_windows: int = 4,
    max_window_size: int = 100,
    min_window_size: int = 20,
    overlap_ratio: float = 0.5,
) -> dict:
    """
    Compute window parameters based on entity observation count.
    
    Guarantees at least `min_windows` windows per entity.
    
    Args:
        entity_length: Number of observations for this entity
        min_windows: Minimum number of windows to generate
        max_window_size: Cap on window size (even for long entities)
        min_window_size: Floor on window size (need enough for metrics)
        overlap_ratio: Fraction of overlap between windows (0.5 = 50%)
    
    Returns:
        dict with 'size', 'stride', 'n_windows'
    """
    # Target: at least min_windows windows
    # With overlap: n_windows ≈ (entity_length - size) / stride + 1
    # Solve for size given stride = size * (1 - overlap_ratio)
    
    # Start with entity_length / (min_windows + 1) as window size
    target_size = entity_length // (min_windows + 1)
    
    # Apply bounds
    size = max(min_window_size, min(max_window_size, target_size))
    
    # Compute stride for desired overlap
    stride = max(1, int(size * (1 - overlap_ratio)))
    
    # Verify we get enough windows
    n_windows = max(1, (entity_length - size) // stride + 1)
    
    # If still too few windows, reduce size
    while n_windows < min_windows and size > min_window_size:
        size = max(min_window_size, size - 10)
        stride = max(1, int(size * (1 - overlap_ratio)))
        n_windows = max(1, (entity_length - size) // stride + 1)
    
    return {
        'size': size,
        'stride': stride,
        'n_windows': n_windows,
        'entity_length': entity_length,
    }


def get_entity_window_configs(
    observations: pl.DataFrame,
    entity_col: str = 'entity_id',
    **kwargs,
) -> Dict[str, dict]:
    """
    Compute adaptive window config for each entity.
    
    Returns dict mapping entity_id → window config.
    """
    entity_lengths = (
        observations
        .group_by(entity_col)
        .agg(pl.count().alias('n_obs'))
    )
    
    configs = {}
    for row in entity_lengths.iter_rows(named=True):
        entity_id = row[entity_col]
        n_obs = row['n_obs']
        configs[entity_id] = compute_adaptive_windows(n_obs, **kwargs)
    
    return configs
```

### 2. Fix Data Recording in Both Pathways

```python
# In signal_vector.py — ensure writes happen

def process_entity_signal(
    observations: pl.DataFrame,
    entity_id: str,
    signal_id: str,
    window_config: dict,
) -> pl.DataFrame:
    """Process single entity-signal pair with adaptive windows."""
    
    # Filter to this entity-signal
    df = observations.filter(
        (pl.col('entity_id') == entity_id) & 
        (pl.col('signal_id') == signal_id)
    ).sort('timestamp')
    
    n_obs = len(df)
    size = window_config['size']
    stride = window_config['stride']
    
    # Skip if not enough observations for even one window
    if n_obs < size:
        logger.warning(f"Entity {entity_id} signal {signal_id}: {n_obs} obs < window size {size}, skipping")
        return pl.DataFrame()  # Return empty, don't fail silently
    
    # Generate windows
    results = []
    window_id = 0
    
    for start_idx in range(0, n_obs - size + 1, stride):
        end_idx = start_idx + size
        window_data = df.slice(start_idx, size)
        
        values = window_data['value'].to_numpy()
        timestamps = window_data['timestamp'].to_numpy()
        
        # Compute metrics
        metrics = compute_all_metrics(values)
        
        # Build result row
        result = {
            'entity_id': entity_id,
            'signal_id': signal_id,
            'window_id': window_id,
            'window_start': float(timestamps[0]),
            'window_end': float(timestamps[-1]),
            'n_obs': len(values),
            **metrics,
        }
        results.append(result)
        window_id += 1
    
    if not results:
        logger.warning(f"Entity {entity_id} signal {signal_id}: no windows generated")
        return pl.DataFrame()
    
    return pl.DataFrame(results)


def run_adaptive_pipeline(observations: pl.DataFrame) -> pl.DataFrame:
    """
    Run signal vector with adaptive per-entity windowing.
    
    CRITICAL: This must write output even if some entities fail.
    """
    # Get adaptive configs per entity
    entity_configs = get_entity_window_configs(observations)
    
    # Log config summary
    sizes = [c['size'] for c in entity_configs.values()]
    logger.info(f"Adaptive windows: size range {min(sizes)}-{max(sizes)}, "
                f"{len(entity_configs)} entities")
    
    # Process all entity-signal pairs
    all_results = []
    
    entity_signal_pairs = (
        observations
        .select(['entity_id', 'signal_id'])
        .unique()
        .to_dicts()
    )
    
    for pair in tqdm(entity_signal_pairs, desc="Processing signals"):
        entity_id = pair['entity_id']
        signal_id = pair['signal_id']
        
        window_config = entity_configs.get(entity_id, {
            'size': 50, 'stride': 25  # Fallback
        })
        
        result = process_entity_signal(
            observations, entity_id, signal_id, window_config
        )
        
        if len(result) > 0:
            all_results.append(result)
    
    # CRITICAL: Concatenate and write even if some failed
    if not all_results:
        raise RuntimeError("No results generated! Check window sizes vs entity lengths.")
    
    final_df = pl.concat(all_results)
    
    # ALWAYS WRITE OUTPUT
    output_path = get_parquet_path('vector', 'signal')
    write_parquet_atomic(final_df, output_path)
    
    logger.info(f"Wrote {len(final_df)} rows to {output_path}")
    
    return final_df
```

### 3. Update Geometry to Preserve Window Dimension

```python
# In geometry.py — keep window_id in output

def compute_geometry_per_window(
    vector: pl.DataFrame,
    entity_col: str = 'entity_id',
) -> pl.DataFrame:
    """
    Compute geometry features PER WINDOW, not per entity.
    
    State layer needs multiple geometry snapshots to compute velocity.
    """
    # Group by entity AND window
    results = []
    
    for (entity_id, window_id), group in vector.group_by([entity_col, 'window_id']):
        # Get all signals for this entity-window
        signals = group['signal_id'].unique().to_list()
        
        if len(signals) < 2:
            continue  # Need at least 2 signals for geometry
        
        # Build feature matrix: signals × metrics
        # ... PCA, coupling, etc.
        
        geometry_row = {
            'entity_id': entity_id,
            'window_id': window_id,
            'window_start': group['window_start'].min(),
            'window_end': group['window_end'].max(),
            'n_signals': len(signals),
            # ... geometry metrics
        }
        results.append(geometry_row)
    
    return pl.DataFrame(results)
```

### 4. Domain YAML Update

```yaml
# config/domains/cmapss.yaml

windowing:
  strategy: adaptive    # NEW: per-entity adaptive
  min_windows: 4        # Guarantee at least 4 windows per entity
  max_window_size: 100  # Cap even for long entities
  min_window_size: 20   # Floor (need enough obs for metrics)
  overlap_ratio: 0.5    # 50% overlap
  
  # Fallback for non-adaptive mode
  fallback_size: 50
  fallback_stride: 25
```

---

## Migration Checklist

### Code Changes

- [ ] Add `prism/utils/adaptive_windows.py` with dynamic sizing
- [ ] Update `signal_vector.py`:
  - [ ] Use adaptive window configs when `--adaptive`
  - [ ] Ensure writes happen even if some entities fail
  - [ ] Log warnings for skipped entities (don't fail silently)
- [ ] Update `geometry.py`:
  - [ ] Output per-window, not per-entity
  - [ ] Keep `window_id` in output schema
- [ ] Update `state.py`:
  - [ ] Expect multiple windows per entity in geometry input
  - [ ] Compute velocity as diff between consecutive windows

### Validation

```bash
# After fix, expect:
python -m prism.entry_points.signal_vector --adaptive

# Check output counts
python -c "
import polars as pl
from prism.db import get_parquet_path

v = pl.read_parquet(get_parquet_path('vector', 'signal'))
print(f'Vector: {len(v)} rows')
print(f'Entities: {v[\"entity_id\"].n_unique()}')
print(f'Windows per entity: {len(v) / v[\"entity_id\"].n_unique() / v[\"signal_id\"].n_unique():.1f}')
"

# Should see:
# Vector: ~1.2M rows
# Entities: 100
# Windows per entity: 4-10 (varies by entity length)
```

---

## Expected Results After Fix

```
┌──────────────────────┬───────────┬──────────────────────────────────────────────┐
│         File         │   Rows    │                 Description                  │
├──────────────────────┼───────────┼──────────────────────────────────────────────┤
│ observations.parquet │ 515,775   │ Raw observations (100 entities × 25 signals) │
├──────────────────────┼───────────┼──────────────────────────────────────────────┤
│ vector.parquet       │ 1,258,124 │ Windowed metrics (multiple per entity)       │
├──────────────────────┼───────────┼──────────────────────────────────────────────┤
│ geometry.parquet     │ ~600      │ Geometry per window (not per entity!)        │
├──────────────────────┼───────────┼──────────────────────────────────────────────┤
│ state.parquet        │ ~500      │ Velocity between consecutive windows         │
└──────────────────────┴───────────┴──────────────────────────────────────────────┘
```

---

## The Principle

**Window size is a function of data, not a hardcoded constant.**

```python
# BAD: Hardcoded from financial domain
WINDOW_TIERS = [252, 126, 63]  # Trading days

# GOOD: Derived from entity characteristics  
window_size = f(entity_length, min_windows=4, max_size=100)
```

This makes PRISM truly domain-agnostic — same code works whether entities have 100 observations (fast-failing machines) or 10,000 observations (long-running processes).
