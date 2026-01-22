---
title: Vector Layer
description: The first processing layer that extracts 51 behavioral metrics from raw sensor data
---

The Vector Layer transforms raw sensor readings into behavioral metrics that capture signal dynamics.

## What It Does

For each sensor signal, the Vector Layer computes **51 metrics** that describe:

- **Statistical properties** — mean, std, skewness, kurtosis
- **Temporal dynamics** — trends, autocorrelation, stationarity
- **Frequency content** — dominant frequencies, spectral entropy
- **Shape features** — peaks, crossings, complexity measures

With typical industrial datasets having 10-50 sensors, this produces approximately **740,000 features**.

## Why 51 Metrics?

The metric set was designed to be:

1. **Comprehensive** — Capture behavior from multiple perspectives
2. **Computationally efficient** — Run on streaming data
3. **Domain agnostic** — No turbofan-specific or bearing-specific features

## Example Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Central tendency | `mean` | Average value |
| Dispersion | `std` | Standard deviation |
| Shape | `skewness` | Asymmetry of distribution |
| Dynamics | `trend_slope` | Linear trend over window |
| Complexity | `sample_entropy` | Signal regularity measure |

## Usage

```python
from prism.layers import VectorLayer

# Initialize with your data
vector = VectorLayer(df, entity_col='unit_id', time_col='cycle')

# Compute all metrics
features = vector.compute()

# Result: DataFrame with 51 metrics per sensor
```

## Configuration

The Vector Layer respects native sampling frequencies. No interpolation or forward-filling is performed—temporal relationships are preserved exactly as captured.

```yaml
# config.yml
vector_layer:
  window_size: 20        # Rolling window for temporal metrics
  min_periods: 10        # Minimum observations required
  preserve_nulls: true   # Don't interpolate missing data
```

## Output Schema

The output is a Parquet file with columns:

```
entity_id | timestamp | sensor_metric | value
----------|-----------|---------------|------
unit_001  | 50        | temp1_mean    | 489.2
unit_001  | 50        | temp1_std     | 12.4
unit_001  | 50        | temp1_trend   | 0.023
...
```

This feeds directly into the [Geometry Layer](/docs/geometry-layer/) for relationship analysis.
