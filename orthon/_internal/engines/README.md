# PRISM Engines

**33 behavioral measurement engines for signal topology analysis.**

All engines are pure functions or classes that take numerical data and return structured metrics. No side effects, no database access, no interpretation.

**Architecture Note (v2.0):** Engines are pure mathematical functions with no I/O dependencies. They accept NumPy arrays or pandas DataFrames (for scipy/sklearn compatibility) and return Python dicts. The Polars + Parquet storage layer is handled by entry points, not engines.

---

## Quick Start

```python
import numpy as np
from prism.engines import compute_hurst, compute_entropy, PCAEngine, GrangerEngine

# Generate sample data
values = np.random.randn(500)

# Vector engine (single series)
hurst = compute_hurst(values)
print(f"Hurst exponent: {hurst['hurst_exponent']:.3f}")

# Geometry engine (multiple series)
matrix = np.random.randn(100, 5)  # 100 observations, 5 series
pca = PCAEngine()
result = pca.compute(matrix)
print(f"Variance explained: {result['explained_variance_ratio']}")

# State engine (pairwise dynamics)
x, y = np.random.randn(200), np.random.randn(200)
granger = GrangerEngine(max_lag=5)
result = granger.compute(x, y)
print(f"Granger p-value: {result['p_value']:.4f}")
```

---

## Engine Categories

### Vector Engines (9)

Single-signal intrinsic properties. Input: 1D array. Output: dict of metrics.

| Engine | Function | Measures |
|--------|----------|----------|
| **Hurst** | `compute_hurst(values)` | Long-term memory (H > 0.5 = trending) |
| **Entropy** | `compute_entropy(values)` | Complexity via sample/permutation entropy |
| **GARCH** | `compute_garch(values)` | Volatility clustering (alpha, beta) |
| **Wavelet** | `compute_wavelets(values)` | Multi-scale energy distribution |
| **Spectral** | `compute_spectral(values)` | Frequency content via FFT |
| **Lyapunov** | `compute_lyapunov(values)` | Chaos signal (> 0 = chaotic) |
| **RQA** | `compute_rqa(values)` | Recurrence patterns in phase space |
| **Realized Vol** | `compute_realized_vol(values)` | Volatility, drawdown, distribution |
| **Hilbert** | `compute_hilbert(values)` | Amplitude, phase, instantaneous frequency |

```python
from prism.engines import (
    compute_hurst, compute_entropy, compute_garch,
    compute_wavelets, compute_spectral, compute_lyapunov,
    compute_rqa, compute_realized_vol, compute_hilbert
)
```

### Geometry Engines (9)

Multi-signal relational structure. Input: 2D matrix (observations x signals). Output: structural metrics.

| Engine | Class | Measures |
|--------|-------|----------|
| **PCA** | `PCAEngine` | Shared variance, effective dimensionality |
| **Clustering** | `ClusteringEngine` | Natural groupings (K-means, hierarchical) |
| **Distance** | `DistanceEngine` | Pairwise dissimilarity matrix |
| **Mutual Information** | `MutualInformationEngine` | Nonlinear dependence |
| **Copula** | `CopulaEngine` | Tail dependence structure |
| **MST** | `MSTEngine` | Minimum spanning tree topology |
| **LOF** | `LOFEngine` | Local outlier factor scores |
| **Convex Hull** | `ConvexHullEngine` | Geometric extent and centrality |
| **Barycenter** | `BarycenterEngine` | Center of mass in behavioral space |

```python
from prism.engines import (
    PCAEngine, ClusteringEngine, DistanceEngine,
    MutualInformationEngine, CopulaEngine, MSTEngine,
    LOFEngine, ConvexHullEngine, BarycenterEngine
)
```

### State Engines (7)

Multi-signal temporal dynamics. Input: paired signal topology. Output: dynamic relationship metrics.

| Engine | Class | Measures |
|--------|-------|----------|
| **Granger** | `GrangerEngine` | Predictive causality (does X predict Y?) |
| **Cross-Correlation** | `CrossCorrelationEngine` | Lead/lag synchronization |
| **Cointegration** | `CointegrationEngine` | Long-run equilibrium relationship |
| **DTW** | `DTWEngine` | Shape similarity with time warping |
| **DMD** | `DMDEngine` | Dynamic mode decomposition |
| **Transfer Entropy** | `TransferEntropyEngine` | Directed information flow |
| **Coupled Inertia** | `CoupledInertiaEngine` | Coupled momentum dynamics |

```python
from prism.engines import (
    GrangerEngine, CrossCorrelationEngine, CointegrationEngine,
    DTWEngine, DMDEngine, TransferEntropyEngine, CoupledInertiaEngine
)
```

### Temporal Dynamics Engines (5)

Analyze geometry evolution over time. Input: geometry snapshots. Output: dynamics metrics.

| Engine | Class | Measures |
|--------|-------|----------|
| **Energy Dynamics** | `EnergyDynamicsEngine` | Energy flow and conservation |
| **Tension Dynamics** | `TensionDynamicsEngine` | Structural stress evolution |
| **Phase Detector** | `PhaseDetectorEngine` | Phase transitions and regime changes |
| **Cohort Aggregator** | `CohortAggregatorEngine` | Aggregate cohort behavior |
| **Transfer Detector** | `TransferDetectorEngine` | Detect transfer events |

```python
from prism.engines import (
    EnergyDynamicsEngine, TensionDynamicsEngine,
    PhaseDetectorEngine, CohortAggregatorEngine, TransferDetectorEngine
)
```

### Observation Engines (3)

Discontinuity detection at point precision. Run BEFORE windowing.

| Engine | Function | Measures |
|--------|----------|----------|
| **Break Detector** | `get_break_metrics()` | Screen for ALL discontinuities |
| **Heaviside** | `get_heaviside_metrics()` | PERSISTENT level shifts (steps) |
| **Dirac** | `get_dirac_metrics()` | TRANSIENT shocks (impulses) |

```python
from prism.engines import (
    get_break_metrics, get_heaviside_metrics, get_dirac_metrics,
    compute_breaks, identify_steps, identify_impulses
)
```

---

## Registry API

Access engines programmatically via the unified registry.

```python
from prism.engines import (
    # Registries
    ENGINES,              # All 33 engines
    VECTOR_ENGINES,       # 9 vector engines
    GEOMETRY_ENGINES,     # 9 geometry engines
    STATE_ENGINES,        # 7 state engines
    TEMPORAL_DYNAMICS_ENGINES,  # 5 temporal dynamics
    OBSERVATION_ENGINES,  # 3 observation engines

    # Lookup functions
    get_engine,           # Get any engine by name
    get_vector_engine,    # Get vector engine by name
    get_geometry_engine,  # Get geometry engine by name
    get_state_engine,     # Get state engine by name

    # List functions
    list_engines,         # List all engine names
    list_vector_engines,
    list_geometry_engines,
    list_state_engines,
)

# List all engines
print(list_engines())

# Get engine by name
compute_fn = get_engine("hurst")
metrics = compute_fn(values)

# Iterate over all vector engines
for name, fn in VECTOR_ENGINES.items():
    result = fn(values)
    print(f"{name}: {len(result)} metrics")
```

---

## Engine Output Format

All engines return Python dictionaries with typed values.

```python
# Vector engine output
{
    'hurst_exponent': 0.52,       # float
    'std_error': 0.03,            # float
    'r_squared': 0.98,            # float
    'n_points': 500,              # int
}

# Geometry engine output
{
    'explained_variance_ratio': [0.4, 0.25, 0.15, 0.1, 0.1],  # list[float]
    'n_components': 5,            # int
    'total_variance': 1.0,        # float
}

# State engine output
{
    'p_value': 0.03,              # float
    'f_statistic': 4.5,           # float
    'optimal_lag': 2,             # int
    'is_significant': True,       # bool
}
```

---

## Minimum Data Requirements

Each engine has minimum data requirements for reliable results.

| Engine | Minimum Points | Recommended |
|--------|----------------|-------------|
| Hurst | 20 | 100+ |
| Entropy | 20 | 100+ |
| GARCH | 30 | 200+ |
| Wavelet | 32 | 128+ |
| Spectral | 16 | 64+ |
| Lyapunov | 100 | 500+ |
| RQA | 20 | 100+ |

Engines return empty dicts or NaN values when data is insufficient.

---

## Design Principles

1. **Pure Functions**: No side effects, no state, no database access
2. **Consistent Interface**: All vector engines: `f(np.ndarray) -> dict`
3. **Fail Gracefully**: Return empty dict or NaN, never raise on bad data
4. **No Interpretation**: Return raw metrics, don't classify or score
5. **Minimal Dependencies**: Core engines use only NumPy/SciPy

---

## Adding New Engines

1. Create `prism/engines/your_engine.py`
2. Implement compute function or class
3. Add to `__init__.py` registry
4. Add to `__all__` exports

```python
# prism/engines/your_engine.py
import numpy as np

def compute_your_metric(values: np.ndarray) -> dict:
    """Compute your custom metric."""
    if len(values) < 20:
        return {}

    result = your_calculation(values)

    return {
        'your_metric': float(result),
        'n_points': len(values),
    }
```

---

## See Also

- [Main README](../../README.md) - Full project documentation
- [pyproject.toml](../../pyproject.toml) - Package configuration
