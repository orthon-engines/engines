"""
Dynamics Engine Runner

Computes dynamical systems metrics for all entities.
Works on ordered index I (time, depth, altitude, etc.)

Outputs: dynamics.parquet
- Lyapunov exponents (stability classification)
- RQA metrics (determinism, recurrence, laminarity)
- Attractor properties (correlation dimension, type)

Architecture: Sequential runner, parallelism handled by orchestrator.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


def process_entity_dynamics(
    unit_id: str,
    entity_obs: pl.DataFrame,
    params: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Process a single entity for dynamics metrics.

    Args:
        unit_id: Entity identifier
        entity_obs: Observations for this entity only
        params: Optional parameters

    Returns:
        List of result dicts (one per signal)
    """
    from prism.engines.signal import lyapunov, attractor
    from prism.primitives.dynamical.rqa import rqa_metrics

    params = params or {}
    min_samples = params.get('min_samples', 100)
    rqa_threshold_pct = params.get('rqa_threshold_percentile', 10.0)
    rqa_min_line = params.get('rqa_min_line', 2)

    results = []
    signals = entity_obs.select('signal_id').unique().to_series().to_list()

    # Check sampling uniformity
    I_values = entity_obs.select('I').unique().sort('I').to_series().to_numpy()
    sampling_cv = 0.0
    if len(I_values) > 1:
        dI = np.diff(I_values)
        if np.mean(dI) > 0:
            sampling_cv = float(np.std(dI) / np.mean(dI))
    sampling_uniform = sampling_cv < 0.1

    for signal_id in signals:
        sig_data = (
            entity_obs
            .filter(pl.col('signal_id') == signal_id)
            .sort('I')
            .select('value')
            .to_series()
            .to_numpy()
        )

        n = len(sig_data)
        if n < min_samples:
            continue

        # Remove NaN
        sig_data = sig_data[~np.isnan(sig_data)]
        if len(sig_data) < min_samples:
            continue

        result_row = {
            'unit_id': unit_id,
            'signal_id': signal_id,
            'n_samples': len(sig_data),
            'sampling_uniform': sampling_uniform,
            'sampling_cv': sampling_cv,
        }

        try:
            # Lyapunov exponent (computed value only, no classification)
            lyap_result = lyapunov.compute(sig_data, min_samples=50)
            result_row['lyapunov'] = lyap_result.get('lyapunov')
            result_row['lyapunov_confidence'] = lyap_result.get('confidence')

            # Get embedding parameters for RQA
            embedding_dim = lyap_result.get('embedding_dim', 3) or 3
            embedding_tau = lyap_result.get('embedding_tau', 1) or 1
            result_row['embedding_dim'] = embedding_dim
            result_row['embedding_tau'] = embedding_tau

        except Exception as e:
            result_row['lyapunov'] = np.nan
            result_row['lyapunov_confidence'] = 0.0
            embedding_dim = 3
            embedding_tau = 1

        try:
            # Attractor properties
            attr_result = attractor.compute(sig_data, embedding_dim=embedding_dim, delay=embedding_tau)
            result_row['correlation_dim'] = attr_result.get('correlation_dim')
            result_row['attractor_type'] = attr_result.get('attractor_type')

        except Exception:
            result_row['correlation_dim'] = np.nan
            result_row['attractor_type'] = 'unknown'

        # RQA metrics (computationally intensive - only if enough data)
        if len(sig_data) >= 200:
            try:
                rqa = rqa_metrics(
                    sig_data,
                    dimension=embedding_dim,
                    delay=embedding_tau,
                    threshold_percentile=rqa_threshold_pct,
                    min_line=rqa_min_line
                )
                result_row['recurrence_rate'] = rqa.get('recurrence_rate')
                result_row['determinism'] = rqa.get('determinism')
                result_row['laminarity'] = rqa.get('laminarity')
                result_row['trapping_time'] = rqa.get('trapping_time')
                result_row['rqa_entropy'] = rqa.get('entropy')
                result_row['max_diagonal'] = rqa.get('max_diagonal')
                result_row['divergence'] = rqa.get('divergence')

            except Exception:
                pass

        results.append(result_row)

    return results


def run_dynamics(
    obs: pl.DataFrame,
    output_dir: Path,
    params: Dict[str, Any] = None
) -> pl.DataFrame:
    """
    Run dynamics engine on observations (sequential).

    For parallel execution, use process_entity_dynamics with joblib from orchestrator.

    Args:
        obs: Observations with unit_id, signal_id, I, value
        output_dir: Where to write dynamics.parquet
        params: Optional parameters (embedding_dim, delay, etc.)

    Returns:
        DataFrame with dynamics metrics per entity/signal
    """
    params = params or {}
    entities = obs.select('unit_id').unique().to_series().to_list()
    all_results = []

    print(f"  Processing {len(entities)} entities...")

    for unit_id in entities:
        entity_obs = obs.filter(pl.col('unit_id') == unit_id)
        entity_results = process_entity_dynamics(unit_id, entity_obs, params)
        all_results.extend(entity_results)

    if not all_results:
        print("  Warning: no dynamics data computed")
        return pl.DataFrame()

    df = pl.DataFrame(all_results)

    # Write output
    output_path = output_dir / 'dynamics.parquet'
    df.write_parquet(output_path)
    print(f"  dynamics.parquet: {len(df):,} rows x {len(df.columns)} cols")

    return df
