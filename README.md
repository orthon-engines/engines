# PRISM: Pure Mathematical Signal Analysis Primitives

**Domain-agnostic signal analysis functions that take numbers in, return numbers out.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Quick Start

```python
import prism
import numpy as np

# Generate test signal
signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000)) + 0.1 * np.random.randn(1000)

# Spectral analysis
dom_freq = prism.dominant_frequency(signal, sample_rate=1000)  # Should be ~10 Hz
spec_flat = prism.spectral_flatness(signal)  # Low for tonal signals

# Complexity measures
perm_ent = prism.permutation_entropy(signal)
samp_ent = prism.sample_entropy(signal)

# Statistics
kurt = prism.kurtosis(signal)
skew = prism.skewness(signal)

# Eigenstructure (for multivariate signals)
multivariate = np.column_stack([signal, np.roll(signal, 10)])
cov = prism.covariance_matrix(multivariate)
eigenvals, eigenvecs = prism.eigendecomposition(cov)
eff_dim = prism.effective_dimension(eigenvals)
```

## What This Is

PRISM provides **pure mathematical functions** for signal analysis. Every function:

- Takes numpy arrays as input
- Returns numbers or arrays as output
- Has zero dependencies on file I/O, configuration files, or data schemas
- Is stateless and deterministic
- Can be used in any signal processing pipeline

## What This Is Not

PRISM is **not** a complete signal processing framework. It doesn't handle:

- Data loading or file I/O
- Pipeline orchestration
- Windowing or segmentation
- Classification or interpretation
- Plotting or visualization

For a complete framework that uses PRISM primitives, see [ORTHON](https://github.com/prism-engines/orthon).

## Architecture

```
PRISM = Pure computation (MIT License)
ORTHON = Pipeline + interpretation (PolyForm Noncommercial)
```

PRISM provides the math. ORTHON provides the workflow.

## Function Categories

### Spectral Analysis
- `power_spectral_density()` - FFT-based power spectrum
- `dominant_frequency()` - Peak frequency detection
- `spectral_flatness()` - Noise vs. tonal content (Wiener entropy)
- `spectral_entropy()` - Shannon entropy of spectrum
- `total_harmonic_distortion()` - THD measurement
- `laplace_transform()` - S-domain analysis

### Time Domain Statistics
- `mean()`, `variance()`, `skewness()`, `kurtosis()` - Moment statistics
- `crest_factor()` - Peak-to-RMS ratio
- `zero_crossings()` - Rate of sign changes
- `turning_points()` - Local extrema count

### Complexity & Entropy
- `permutation_entropy()` - Ordinal pattern complexity
- `sample_entropy()` - Self-similarity measure
- `approximate_entropy()` - Pattern repeatability
- `lempel_ziv_complexity()` - Algorithmic complexity
- `fractal_dimension()` - Box-counting dimension

### Memory & Correlation
- `hurst_exponent()` - Long-range dependence
- `autocorrelation()` - Temporal self-similarity
- `detrended_fluctuation_analysis()` - Scaling behavior

### Eigenstructure & Geometry
- `covariance_matrix()` - Cross-variable relationships
- `eigendecomposition()` - Principal components
- `effective_dimension()` - Participation ratio
- `condition_number()` - Matrix conditioning

### Dynamical Systems
- `lyapunov_exponent()` - Chaos measurement
- `attractor_reconstruction()` - Phase space embedding
- `recurrence_analysis()` - RQA metrics

### Information Theory
- `transfer_entropy()` - Directed information flow
- `granger_causality()` - Predictive causality
- `mutual_information()` - Shared information

### Normalization
- `zscore_normalize()` - Mean/std normalization
- `robust_normalize()` - Median/IQR normalization
- `mad_normalize()` - Median absolute deviation

## Installation

```bash
pip install prism-signal
```

For development:
```bash
git clone https://github.com/prism-engines/prism
cd prism
pip install -e .[dev]
pytest
```

## Citation

If you use PRISM in academic research, please cite:

```bibtex
@software{rudder_prism_2026,
  title = {PRISM: Pure Mathematical Signal Analysis Primitives},
  author = {Rudder, Jason},
  year = {2026},
  license = {MIT},
  url = {https://github.com/prism-engines/prism}
}
```

## License

MIT License. Use freely in academic and commercial projects.

## Relationship to ORTHON

PRISM is the computational engine. [ORTHON](https://github.com/prism-engines/orthon) is the interpretation framework that:

- Loads data and creates manifests
- Orchestrates PRISM functions in pipelines
- Classifies signals and systems
- Generates diagnostic reports
- Provides browser-based exploration

Think of it as: **PRISM = Calculator, ORTHON = Engineer**

## Examples

### Turbofan Engine Analysis
```python
import prism

# Per-sensor analysis
for sensor_data in engine_sensors:
    spectral_entropy = prism.spectral_entropy(sensor_data)
    sample_entropy = prism.sample_entropy(sensor_data)
    lyapunov = prism.lyapunov_exponent(sensor_data)

# Cross-sensor eigenstructure
sensor_matrix = np.column_stack(engine_sensors)
cov = prism.covariance_matrix(sensor_matrix)
eigenvals, _ = prism.eigendecomposition(cov)
eff_dim = prism.effective_dimension(eigenvals)  # Dimensional collapse = failure approaching
```

### Financial Time Series
```python
import prism

# Price analysis
returns = np.diff(np.log(prices))
hurst = prism.hurst_exponent(returns)  # Market efficiency measure
perm_ent = prism.permutation_entropy(returns)  # Market complexity

# Multi-asset analysis
asset_matrix = np.column_stack([asset1_returns, asset2_returns, asset3_returns])
corr = prism.correlation_matrix(asset_matrix)
eigenvals, _ = prism.eigendecomposition(corr)
participation = prism.participation_ratio(eigenvals)  # Diversification measure
```

### Ecological Networks
```python
import prism

# Population time series analysis
for species in species_data:
    coeffs, r2 = prism.trend_fit(species)
    turning_pts = prism.turning_points(species)

# Community-level analysis
community_matrix = np.column_stack(all_species_data)
eigenvals, _ = prism.eigendecomposition(prism.covariance_matrix(community_matrix))
eff_dim = prism.effective_dimension(eigenvals)  # Functional diversity
```

The math is the same. The interpretation changes based on domain.

## Contributing

1. All functions must be pure (no side effects, no file I/O)
2. All functions must have type hints and docstrings
3. All functions must have unit tests with >95% coverage
4. No dependencies beyond numpy and scipy (optional: scikit-learn for complexity measures)

## Credits

- **Avery Rudder** - "Laplace transform IS the state engine" - eigenvalue insight
