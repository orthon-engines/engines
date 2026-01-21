"""
Vector Score - Normalized 0-1 activity scoring for vector layer

Computes a composite "activity" score from multiple engine outputs.
Score of 0 = signal behaving calmly/normally
Score of 1 = signal behaving at maximum activity/unusualness

Usage:
    Called by signal_vector.py runner after engine computations.
    Stores results alongside vector metrics in vector/signal parquet.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Union
import polars as pl


@dataclass
class EngineConfig:
    """Configuration for normalizing a single engine's output"""
    baseline: Union[str, float]  # 'median' or literal value
    scale: Union[str, float]     # 'iqr', 'percentile_95', 'abs_max', or literal
    transform: str               # 'abs', 'linear', 'abs_from_half'
    weight: float                # Relative importance


# Default engine configurations
# Map raw engine outputs to 0-1 activity scale
# NOTE: Keys must match actual metric_name from engine output
ENGINE_CONFIGS: Dict[str, EngineConfig] = {
    'hurst_exponent': EngineConfig(
        baseline=0.5,           # Random walk = calm
        scale=0.5,              # Max deviation is 0.5 (to 0 or 1)
        transform='abs',        # Both trending AND mean-reverting = active
        weight=1.0
    ),
    'permutation_entropy': EngineConfig(
        baseline='median',      # Historical median = calm
        scale='iqr',            # IQR for robustness
        transform='linear',     # Higher = more active
        weight=0.8
    ),
    'persistence': EngineConfig(  # from garch engine
        baseline='median',
        scale='percentile_95',  # 95th percentile = max
        transform='linear',
        weight=1.0
    ),
    'lyapunov_exponent': EngineConfig(
        baseline=0,             # Zero = edge of chaos
        scale='abs_max',        # Historical max absolute value
        transform='abs',        # Both positive and negative = active
        weight=0.5
    ),
    'spectral_entropy': EngineConfig(
        baseline='median',
        scale='iqr',
        transform='linear',
        weight=0.6
    ),
    'scale_entropy': EngineConfig(  # from wavelet engine
        baseline='median',
        scale='iqr',
        transform='linear',
        weight=0.6
    ),
    'determinism': EngineConfig(  # from rqa engine
        baseline='median',
        scale='iqr',
        transform='linear',
        weight=0.7
    ),
    'realized_vol': EngineConfig(
        baseline='median',
        scale='percentile_95',
        transform='linear',
        weight=0.9
    ),
}

# Map engine metric names to score output names
ENGINE_TO_SCORE_KEY = {
    'hurst_exponent': 'score_hurst',
    'permutation_entropy': 'score_entropy',
    'persistence': 'score_garch',
    'lyapunov_exponent': 'score_lyapunov',
    'spectral_entropy': 'score_spectral',
    'scale_entropy': 'score_wavelet',
    'determinism': 'score_rqa',
    'realized_vol': 'score_realized_vol',
}


@dataclass
class Baselines:
    """Historical statistics for an engine metric"""
    median: float
    mean: float
    std: float
    iqr: float
    p05: float
    p95: float
    min_val: float
    max_val: float
    abs_max: float


def compute_baselines(values: np.ndarray) -> Optional[Baselines]:
    """Compute historical baseline statistics from array of values"""
    clean = values[~np.isnan(values)]
    if len(clean) < 10:  # Need minimum data
        return None

    return Baselines(
        median=float(np.median(clean)),
        mean=float(np.mean(clean)),
        std=float(np.std(clean)),
        iqr=float(np.percentile(clean, 75) - np.percentile(clean, 25)),
        p05=float(np.percentile(clean, 5)),
        p95=float(np.percentile(clean, 95)),
        min_val=float(np.min(clean)),
        max_val=float(np.max(clean)),
        abs_max=float(np.max(np.abs(clean)))
    )


def normalize_engine_value(
    value: float,
    baselines: Baselines,
    config: EngineConfig
) -> float:
    """
    Normalize a single engine value to 0-1 activity scale.

    Args:
        value: Raw engine output
        baselines: Historical statistics for this engine
        config: Normalization config

    Returns:
        Float in [0, 1] where 0=calm, 1=max activity
    """
    if baselines is None or np.isnan(value):
        return np.nan

    # Determine baseline value
    if config.baseline == 'median':
        baseline = baselines.median
    elif config.baseline == 'mean':
        baseline = baselines.mean
    else:
        baseline = float(config.baseline)

    # Determine scale value
    if config.scale == 'iqr':
        scale = baselines.iqr
    elif config.scale == 'percentile_95':
        scale = baselines.p95 - baselines.median
    elif config.scale == 'abs_max':
        scale = baselines.abs_max
    elif config.scale == 'std':
        scale = baselines.std
    else:
        scale = float(config.scale)

    # Avoid division by zero
    if scale == 0 or np.isnan(scale):
        scale = 1.0

    # Compute deviation based on transform
    if config.transform == 'abs':
        deviation = abs(value - baseline)
    elif config.transform == 'linear':
        deviation = value - baseline
        deviation = max(0, deviation)  # Only positive deviations = activity
    else:
        deviation = value - baseline

    # Normalize to 0-1
    normalized = deviation / scale

    # Clip to [0, 1]
    return float(np.clip(normalized, 0, 1))


def compute_vector_score(
    engine_values: Dict[str, float],
    engine_baselines: Dict[str, Baselines],
    configs: Optional[Dict[str, EngineConfig]] = None
) -> Dict[str, float]:
    """
    Compute composite vector score from multiple engine outputs.

    Args:
        engine_values: {metric_name: raw_value}
        engine_baselines: {metric_name: Baselines}
        configs: Optional override configs

    Returns:
        {
            'vector_score': float,      # Composite 0-1 score
            'score_hurst': float,       # Individual scores
            'score_entropy': float,
            ...
            'n_engines': int,           # How many contributed
            'total_weight': float       # Sum of weights used
        }
    """
    if configs is None:
        configs = ENGINE_CONFIGS

    result = {
        'vector_score': np.nan,
        'score_hurst': np.nan,
        'score_entropy': np.nan,
        'score_garch': np.nan,
        'score_lyapunov': np.nan,
        'score_spectral': np.nan,
        'score_wavelet': np.nan,
        'score_rqa': np.nan,
        'score_realized_vol': np.nan,
        'n_engines': 0,
        'total_weight': 0.0
    }

    weighted_sum = 0.0
    total_weight = 0.0
    n_engines = 0

    for metric_name, value in engine_values.items():
        if np.isnan(value):
            continue

        baselines = engine_baselines.get(metric_name)
        config = configs.get(metric_name)

        if baselines is None or config is None:
            continue

        # Normalize this engine
        score = normalize_engine_value(value, baselines, config)

        if np.isnan(score):
            continue

        # Store individual score
        key = ENGINE_TO_SCORE_KEY.get(metric_name)
        if key:
            result[key] = score

        # Accumulate for weighted average
        weighted_sum += config.weight * score
        total_weight += config.weight
        n_engines += 1

    # Compute composite score
    if total_weight > 0:
        result['vector_score'] = weighted_sum / total_weight

    result['n_engines'] = n_engines
    result['total_weight'] = total_weight

    return result


def compute_baselines_from_history(
    df: pl.DataFrame,
    signal_id: str
) -> Dict[str, Baselines]:
    """
    Compute baselines for all metrics from historical data.

    Args:
        df: Vector metrics DataFrame with signal_id, metric_name, metric_value
        signal_id: Which signal to compute baselines for

    Returns:
        {metric_name: Baselines}
    """
    baselines = {}

    # Filter to this signal
    signal_df = df.filter(pl.col("signal_id") == signal_id)

    if len(signal_df) == 0:
        return baselines

    # Get unique metrics that have configs
    for metric_name in ENGINE_CONFIGS.keys():
        metric_df = signal_df.filter(pl.col("metric_name") == metric_name)

        if len(metric_df) < 10:
            continue

        values = metric_df["metric_value"].to_numpy()
        baseline = compute_baselines(values)

        if baseline is not None:
            baselines[metric_name] = baseline

    return baselines


def compute_score_for_window(
    metrics: Dict[str, float],
    baselines: Dict[str, Baselines]
) -> Dict[str, float]:
    """
    Convenience function: compute vector score for a single window.

    Args:
        metrics: {metric_name: metric_value} from engine outputs
        baselines: Pre-computed baselines for this signal

    Returns:
        Score dict ready to store
    """
    # Filter to metrics we have configs for
    engine_values = {k: v for k, v in metrics.items() if k in ENGINE_CONFIGS}

    return compute_vector_score(engine_values, baselines)
