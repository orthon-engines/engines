# PR: Simplified CLI & Config Alignment

**Branch:** `main`
**Type:** Cleanup / Simplification
**Priority:** Before RMSE validation

---

## Summary

Strip all CLI bloat. Each entry point does one thing. No backward compatibility. No legacy flags. Config aligns with new single-domain architecture.

**Philosophy:** Scripts compute. Config configures. CLI is for overrides only.

---

## CRITICAL RULES FOR CLAUDE CODE

1. **NO BACKWARD COMPATIBILITY** - Delete old flags, don't deprecate them
2. **NO --domain FLAG** - One domain loaded at a time, period
3. **NO DATE FILTERING** - Use `--limit` for observation count instead
4. **NO OUTPUT PATH FLAGS** - Hardcoded paths only
5. **NO METHOD FLAGS** - Hardcode the best algorithm
6. **TESTING FLAGS REQUIRE --testing** - Guard all subset flags behind --testing

---

## Data Directory (Simplified)

```
data/
├── observations.parquet   # Raw + metadata (signal_type, cohort_id after cohort)
├── vector.parquet         # 51 metrics per signal_type  
├── geometry.parquet       # Pairwise relationships + structure
├── state.parquet          # Temporal dynamics
├── cohorts_raw.parquet    # [OPTIONAL] Raw discovery results (--compare)
└── cohorts_vector.parquet # [OPTIONAL] Vector discovery results (--compare)
```

**One domain at a time. Switch domains by replacing data/.**

---

## Entry Points (Final)

### fetch.py

```bash
python -m prism.entry_points.fetch
```

**Flags:** NONE

```python
def main():
    """Fetch data. No options."""
    # Reads from fetchers/yaml config
    # Writes to data/observations.parquet
    pass
```

---

### cohort.py

```bash
# Normal - raw observations
python -m prism.entry_points.cohort

# From vector data
python -m prism.entry_points.cohort --source vector

# Compare raw vs vector
python -m prism.entry_points.cohort --compare

# Adaptive windowing
python -m prism.entry_points.cohort --adaptive

# Testing
python -m prism.entry_points.cohort --testing --limit 1000 --signal s1,s2
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--source {raw,vector}` | Data source for discovery (default: raw) |
| `--compare` | Run both sources, output accuracy comparison |
| `--adaptive` | Auto-detect window size from data frequency |
| `--testing` | Enable test mode (required for limit/signal) |
| `--limit N` | [TESTING] Max observations per signal |
| `--signal x,y,z` | [TESTING] Only process these signal_types |

```python
parser = argparse.ArgumentParser(description='Discover signal types and cohorts via Laplace')
parser.add_argument('--source', choices=['raw', 'vector'], default='raw', help='Data source')
parser.add_argument('--compare', action='store_true', help='Compare raw vs vector discovery')
parser.add_argument('--adaptive', action='store_true', help='Auto-detect window size from data frequency')
parser.add_argument('--testing', action='store_true', help='Enable test mode')
parser.add_argument('--limit', type=int, default=None, help='[TESTING] Max observations per signal')
parser.add_argument('--signal', type=str, default=None, help='[TESTING] Comma-separated signal_types')
```

**Reads:** `data/observations.parquet` (and `data/vector.parquet` if --source vector or --compare)
**Writes:** `data/observations.parquet` (adds signal_type, cohort_id columns)

---

### signal_vector.py

```bash
# Normal
python -m prism.entry_points.signal_vector

# Adaptive windowing  
python -m prism.entry_points.signal_vector --adaptive

# Force recompute
python -m prism.entry_points.signal_vector --force

# Testing
python -m prism.entry_points.signal_vector --testing --limit 500 --signal s1,s2,s3
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--adaptive` | Auto-detect window size from data frequency |
| `--force` | Clear cache, recompute everything |
| `--testing` | Enable test mode (required for limit/signal) |
| `--limit N` | [TESTING] Max observations per signal |
| `--signal x,y,z` | [TESTING] Only process these signal_types |

```python
parser = argparse.ArgumentParser(description='Compute 51 behavioral metrics per signal')
parser.add_argument('--adaptive', action='store_true', help='Auto-detect window size from data frequency')
parser.add_argument('--force', action='store_true', help='Clear cache, recompute everything')
parser.add_argument('--testing', action='store_true', help='Enable test mode')
parser.add_argument('--limit', type=int, default=None, help='[TESTING] Max observations per signal')
parser.add_argument('--signal', type=str, default=None, help='[TESTING] Comma-separated signal_types')
```

**Reads:** `data/observations.parquet`
**Writes:** `data/vector.parquet`

---

### geometry.py

```bash
# Normal
python -m prism.entry_points.geometry

# Adaptive
python -m prism.entry_points.geometry --adaptive

# Force recompute
python -m prism.entry_points.geometry --force

# Testing
python -m prism.entry_points.geometry --testing
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--adaptive` | Auto-detect window size from data frequency |
| `--force` | Clear cache, recompute everything |
| `--testing` | Enable test mode (uses subset from vector.parquet) |

```python
parser = argparse.ArgumentParser(description='Compute pairwise geometry and cohort structure')
parser.add_argument('--adaptive', action='store_true', help='Auto-detect window size from data frequency')
parser.add_argument('--force', action='store_true', help='Clear cache, recompute everything')
parser.add_argument('--testing', action='store_true', help='Enable test mode')
```

**Reads:** `data/vector.parquet`, `data/observations.parquet` (for cohort metadata)
**Writes:** `data/geometry.parquet`

---

### state.py

```bash
# Normal
python -m prism.entry_points.state

# Adaptive
python -m prism.entry_points.state --adaptive

# Force recompute
python -m prism.entry_points.state --force

# Testing
python -m prism.entry_points.state --testing
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--adaptive` | Auto-detect window size from data frequency |
| `--force` | Clear cache, recompute everything |
| `--testing` | Enable test mode |

```python
parser = argparse.ArgumentParser(description='Compute temporal dynamics and state')
parser.add_argument('--adaptive', action='store_true', help='Auto-detect window size from data frequency')
parser.add_argument('--force', action='store_true', help='Clear cache, recompute everything')
parser.add_argument('--testing', action='store_true', help='Enable test mode')
```

**Reads:** `data/geometry.parquet`
**Writes:** `data/state.parquet`

---

## FLAGS TO DELETE

Search and destroy. No deprecation warnings. Just delete.

```bash
# Find all argparse flags in entry_points
grep -r "add_argument" prism/entry_points/ --include="*.py"
```

### Delete These Flags

| Flag | Reason |
|------|--------|
| `--domain` | One domain at a time |
| `--cohort` | Cohorts discovered automatically |
| `--filter-cohort` | Use --signal instead |
| `--dates` | Use --limit instead |
| `--output` | Hardcoded paths |
| `--workers` | Hardcode optimal value |
| `--method` | Hardcode best algorithm |
| `--min-*` | Hardcode sensible defaults |
| `--max-*` | Hardcode sensible defaults |
| `--visualize` | Separate script if needed |
| `--signals` (plural) | Renamed to --signal |
| `--cohorts` (plural) | Deleted |

---

## Config Alignment

### Current Config (Too Complex)

```
config/
├── stride.yaml           # Window/stride settings per domain
├── normalization.yaml    # Normalization per domain  
├── cohorts/              # Predefined cohorts per domain
│   ├── cmapss.yaml
│   ├── femto.yaml
│   └── ...
└── domain_info.json      # Auto-detected domain info
```

### New Config (Simplified)

```
config/
├── engine.yaml           # Engine settings (universal)
├── window.yaml           # Window/stride defaults (overridden by --adaptive)
└── domain.yaml           # Current domain metadata (written by fetch)
```

---

### config/engine.yaml

```yaml
# Engine configuration - universal across all domains
# These are the hardcoded "best" settings

vector_engines:
  hurst:
    min_observations: 100
  entropy:
    min_observations: 50
  garch:
    min_observations: 63
  wavelet:
    min_observations: 64
  spectral:
    min_observations: 64
  lyapunov:
    min_observations: 200
  rqa:
    min_observations: 100
  realized_vol:
    min_observations: 20
  hilbert:
    min_observations: 20

geometry_engines:
  pca:
    n_components: 5
  clustering:
    method: hierarchical  # hardcoded, no CLI option
    linkage: ward
  lof:
    n_neighbors: 20
  mst:
    algorithm: prim

discovery:
  signal_clustering: hierarchical
  cohort_clustering: spectral
  min_cohort_size: 2
```

---

### config/window.yaml

```yaml
# Default window configuration
# Overridden entirely when --adaptive flag is used

default:
  window_size: 252    # observations per window
  stride: 21          # observations between windows
  min_observations: 50

# Adaptive mode calculates these from data frequency
# No per-domain overrides - adaptive figures it out
```

---

### config/domain.yaml

```yaml
# Written by fetch.py, describes current loaded domain
# Read-only for other entry points

name: cmapss
source: NASA C-MAPSS FD001
fetched_at: 2024-01-15T10:30:00Z

# Auto-detected by fetch
observation_count: 20631
signal_count: 2100
date_range:
  start: 1
  end: 362
frequency: cycle  # or 'daily', 'hourly', 'irregular'

# Added by discover
signal_types: 21
cohorts: 4
```

---

## Implementation Checklist

### Phase 1: Delete Old Flags

- [ ] `prism/entry_points/fetch.py` - Remove all flags except --help
- [ ] `prism/entry_points/signal_vector.py` - Keep only: --adaptive, --force, --testing, --limit, --signal
- [ ] `prism/entry_points/geometry.py` - Keep only: --adaptive, --force, --testing
- [ ] `prism/entry_points/state.py` - Keep only: --adaptive, --force, --testing
- [ ] `prism/entry_points/laplace.py` - Keep only: --adaptive, --force, --testing
- [ ] Delete any entry points that are no longer needed

### Phase 2: Update Config

- [ ] Create `config/engine.yaml` with hardcoded engine settings
- [ ] Create `config/window.yaml` with default window settings
- [ ] Create `config/domain.yaml` template (populated by fetch)
- [ ] Delete `config/stride.yaml` (replaced by window.yaml)
- [ ] Delete `config/normalization.yaml` (hardcode in engine)
- [ ] Delete `config/cohorts/` directory (cohorts discovered, not predefined)

### Phase 3: Update Config Loaders

- [ ] `prism/config/loader.py` - Load from new yaml files
- [ ] `prism/config/windows.py` - Read window.yaml, respect --adaptive
- [ ] Delete `prism/config/cascade.py` if no longer needed
- [ ] Delete domain-specific config loading

### Phase 4: Simplify Data Paths

- [ ] `prism/db/parquet_store.py` - Hardcode paths:
  - `data/observations.parquet`
  - `data/vector.parquet`
  - `data/geometry.parquet`
  - `data/state.parquet`
- [ ] Remove all `get_parquet_path(layer, table, domain)` domain parameters
- [ ] Simplify to `get_path(name)` → returns `data/{name}.parquet`

### Phase 5: Create cohort.py

- [ ] Create `prism/entry_points/cohort.py`
- [ ] Implement signal type discovery (Level 1) from raw observations
- [ ] Implement cohort discovery (Level 2) from signal type Laplace fields
- [ ] Implement discovery from vector data (--source vector)
- [ ] Implement --compare mode for accuracy validation
- [ ] Add signal_type and cohort_id columns to observations.parquet

### Phase 6: Verify

```bash
# Each should run with no arguments
python -m prism.entry_points.fetch
python -m prism.entry_points.cohort
python -m prism.entry_points.signal_vector
python -m prism.entry_points.geometry
python -m prism.entry_points.state

# Cohort comparison mode
python -m prism.entry_points.cohort --compare

# Adaptive mode
python -m prism.entry_points.signal_vector --adaptive

# Testing mode
python -m prism.entry_points.signal_vector --testing --limit 100 --signal s1,s2
```

---

## Entry Points to Keep

```
prism/entry_points/
├── __init__.py
├── fetch.py           # Layer 0: Get data
├── cohort.py          # Layer 0.5: Discover signal types + cohorts
├── signal_vector.py   # Layer 1: 51 metrics
├── geometry.py        # Layer 2: Pairwise + structure
├── state.py           # Layer 3: Temporal dynamics
└── testing/           # Test utilities (keep as-is)
```

---

## cohort.py (Renamed from discover.py)

The cohort entry point discovers signal types and cohorts using Laplace field analysis. It can run against **raw observations** OR **vector data** to validate discovery accuracy.

### CLI

```bash
# Discovery on raw observations (default, fast)
python -m prism.entry_points.cohort

# Discovery on vector data (after signal_vector has run)
python -m prism.entry_points.cohort --source vector

# Compare both approaches
python -m prism.entry_points.cohort --compare

# Testing
python -m prism.entry_points.cohort --testing --limit 1000 --signal s1,s2
```

### Flags

| Flag | Description |
|------|-------------|
| `--source {raw,vector}` | Run discovery on raw observations or vector data (default: raw) |
| `--compare` | Run both, output accuracy comparison |
| `--adaptive` | Auto-detect window size from data frequency |
| `--testing` | Enable test mode |
| `--limit N` | [TESTING] Max observations per signal |
| `--signal x,y,z` | [TESTING] Only process these signal_types |

### Implementation

```python
"""
PRISM Cohort Discovery — Find signal types and cohorts via Laplace.

Can run against raw observations OR vector data to validate discovery.

Usage:
    python -m prism.entry_points.cohort              # raw observations
    python -m prism.entry_points.cohort --source vector   # vector data
    python -m prism.entry_points.cohort --compare    # both + accuracy report
"""

import argparse
import polars as pl
from prism.db.parquet_store import get_path
from prism.engines.laplace import compute_laplace_for_series
from prism.engines.geometry.modes import discover_modes


def discover_from_raw(observations: pl.DataFrame) -> pl.DataFrame:
    """
    Level 1: Discover signal types from raw observations.
    Level 2: Discover cohorts from signal type Laplace fields.
    """
    # For each unique signal_id, compute Laplace on raw values
    signal_fields = []
    for signal_id in observations['signal_id'].unique():
        values = observations.filter(pl.col('signal_id') == signal_id)['value'].to_numpy()
        field = compute_laplace_for_series(values)
        signal_fields.append({'signal_id': signal_id, **field})
    
    fields_df = pl.DataFrame(signal_fields)
    
    # Cluster to find signal types (which series are the same sensor?)
    signal_types = cluster_signal_types(fields_df)
    
    # Aggregate by signal type, then cluster to find cohorts
    type_fields = aggregate_by_type(fields_df, signal_types)
    cohorts = cluster_cohorts(type_fields)
    
    return signal_types.join(cohorts, on='signal_type')


def discover_from_vector(vector: pl.DataFrame) -> pl.DataFrame:
    """
    Same discovery but using 51-metric vector data instead of raw.
    """
    # Compute Laplace on vector metrics (already aggregated by signal)
    signal_fields = []
    for signal_id in vector['signal_id'].unique():
        metrics = vector.filter(pl.col('signal_id') == signal_id)
        field = compute_laplace_on_metrics(metrics)
        signal_fields.append({'signal_id': signal_id, **field})
    
    fields_df = pl.DataFrame(signal_fields)
    
    # Same clustering logic
    signal_types = cluster_signal_types(fields_df)
    type_fields = aggregate_by_type(fields_df, signal_types)
    cohorts = cluster_cohorts(type_fields)
    
    return signal_types.join(cohorts, on='signal_type')


def compare_discovery(raw_result: pl.DataFrame, vector_result: pl.DataFrame) -> dict:
    """
    Compare cohort assignments between raw and vector discovery.
    
    Returns accuracy metrics:
    - signal_type_agreement: % of signals assigned to same type
    - cohort_agreement: % of signal types assigned to same cohort
    - adjusted_rand_index: clustering similarity score
    """
    from sklearn.metrics import adjusted_rand_score
    
    # Merge on signal_id
    merged = raw_result.join(vector_result, on='signal_id', suffix='_vector')
    
    # Signal type agreement
    type_match = (merged['signal_type'] == merged['signal_type_vector']).mean()
    
    # Cohort agreement  
    cohort_match = (merged['cohort_id'] == merged['cohort_id_vector']).mean()
    
    # ARI for clustering quality
    ari_types = adjusted_rand_score(
        merged['signal_type'].to_list(),
        merged['signal_type_vector'].to_list()
    )
    ari_cohorts = adjusted_rand_score(
        merged['cohort_id'].to_list(),
        merged['cohort_id_vector'].to_list()
    )
    
    return {
        'signal_type_agreement': type_match,
        'cohort_agreement': cohort_match,
        'ari_signal_types': ari_types,
        'ari_cohorts': ari_cohorts,
    }


def main():
    parser = argparse.ArgumentParser(description='Discover signal types and cohorts via Laplace')
    parser.add_argument('--source', choices=['raw', 'vector'], default='raw',
                        help='Data source: raw observations or vector metrics')
    parser.add_argument('--compare', action='store_true',
                        help='Run both sources and compare accuracy')
    parser.add_argument('--adaptive', action='store_true',
                        help='Auto-detect window size from data frequency')
    parser.add_argument('--testing', action='store_true', help='Enable test mode')
    parser.add_argument('--limit', type=int, default=None,
                        help='[TESTING] Max observations per signal')
    parser.add_argument('--signal', type=str, default=None,
                        help='[TESTING] Comma-separated signal_types')
    args = parser.parse_args()
    
    observations = pl.read_parquet(get_path('observations'))
    
    if args.compare:
        # Run both and compare
        vector = pl.read_parquet(get_path('vector'))
        
        print("Discovering from raw observations...")
        raw_result = discover_from_raw(observations)
        
        print("Discovering from vector data...")
        vector_result = discover_from_vector(vector)
        
        print("\n" + "="*50)
        print("COMPARISON: Raw vs Vector Discovery")
        print("="*50)
        
        metrics = compare_discovery(raw_result, vector_result)
        print(f"Signal Type Agreement: {metrics['signal_type_agreement']:.1%}")
        print(f"Cohort Agreement:      {metrics['cohort_agreement']:.1%}")
        print(f"ARI (Signal Types):    {metrics['ari_signal_types']:.3f}")
        print(f"ARI (Cohorts):         {metrics['ari_cohorts']:.3f}")
        
        # Save both results
        raw_result.write_parquet(get_path('cohorts_raw'))
        vector_result.write_parquet(get_path('cohorts_vector'))
        
        # Use raw as default (update observations)
        result = raw_result
        
    elif args.source == 'vector':
        vector = pl.read_parquet(get_path('vector'))
        result = discover_from_vector(vector)
    else:
        result = discover_from_raw(observations)
    
    # Add signal_type and cohort_id to observations
    observations = observations.join(
        result.select(['signal_id', 'signal_type', 'cohort_id']),
        on='signal_id'
    )
    observations.write_parquet(get_path('observations'))
    
    n_types = result['signal_type'].n_unique()
    n_cohorts = result['cohort_id'].n_unique()
    print(f"\nDiscovered {n_types} signal types, {n_cohorts} cohorts")


if __name__ == "__main__":
    main()
```

### Output Files

**Default (--source raw):**
- Updates `data/observations.parquet` with signal_type, cohort_id columns

**With --compare:**
- `data/cohorts_raw.parquet` - Discovery from raw observations
- `data/cohorts_vector.parquet` - Discovery from vector data
- Updates `data/observations.parquet` with raw discovery results
- Prints comparison metrics

### Validation on C-MAPSS

```bash
# Run full pipeline
python -m prism.entry_points.fetch
python -m prism.entry_points.cohort              # raw discovery
python -m prism.entry_points.signal_vector
python -m prism.entry_points.cohort --compare    # compare raw vs vector

# Expected output:
# Discovered 21 signal types, 4 cohorts
#
# COMPARISON: Raw vs Vector Discovery
# ==================================================
# Signal Type Agreement: 100.0%   (should be perfect - same sensors)
# Cohort Agreement:      85-95%   (interesting if different!)
# ARI (Signal Types):    1.000
# ARI (Cohorts):         0.850
```

**Hypothesis:**
- Signal type discovery should be ~identical (same sensors)
- Cohort discovery might differ - that's interesting data!
  - If raw ≈ vector: discovery can happen early, 51 metrics not needed for structure
  - If raw ≠ vector: 51 metrics reveal structure that raw values don't

### Entry Points to DELETE or MERGE

- [ ] `discover.py` → Renamed to `cohort.py`
- [ ] `laplace.py` → Merge into signal_vector.py (Laplace is part of vector computation)
- [ ] `physics.py` → Move to scripts/ (experimental, not core pipeline)
- [ ] `hybrid.py` → Move to scripts/ (ML experiments, not core pipeline)
- [ ] Any other entry points not in the core pipeline

---

## Hardcoded Values Reference

When Claude Code asks "should this be configurable?", the answer is **NO**.

| Setting | Hardcoded Value | Location |
|---------|-----------------|----------|
| Clustering method | hierarchical (ward) | config/engine.yaml |
| LOF neighbors | 20 | config/engine.yaml |
| PCA components | 5 | config/engine.yaml |
| Min cohort size | 2 | config/engine.yaml |
| Default window | 252 | config/window.yaml |
| Default stride | 21 | config/window.yaml |
| Data directory | data/ | prism/db/parquet_store.py |
| Output filenames | observations, vector, geometry, state, cohorts_raw, cohorts_vector | prism/db/parquet_store.py |

---

## The Final CLI

```bash
# Full pipeline - just run each step
python -m prism.entry_points.fetch
python -m prism.entry_points.cohort
python -m prism.entry_points.signal_vector
python -m prism.entry_points.geometry
python -m prism.entry_points.state

# With adaptive windowing
python -m prism.entry_points.signal_vector --adaptive
python -m prism.entry_points.geometry --adaptive
python -m prism.entry_points.state --adaptive

# Force recompute
python -m prism.entry_points.signal_vector --force

# Testing (quick iteration)
python -m prism.entry_points.signal_vector --testing --limit 100 --signal s1,s2

# Cohort discovery comparison (raw vs vector)
python -m prism.entry_points.cohort --compare
```

**5 entry points. 6 flags max per entry point. No domain flag. No output flag. No method flag.**

---

## Summary

**Before:** 15+ entry points, 20+ flags, config per domain, paths everywhere

**After:** 5 entry points, 6 flags, one config, one data directory

```
fetch → cohort → signal_vector → geometry → state
                   ↓
              (--compare validates raw vs vector discovery)
```

**No backward compatibility. No legacy support. Clean slate.**

*Compute once, query forever. More cowbell.* 🔔
