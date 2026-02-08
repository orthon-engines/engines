"""
Parallel Dynamics Engine Runner (Digital Ocean)

Computes dynamical systems metrics for all entities using joblib parallelization.
Use on multi-core servers, not on Mac.

Outputs: dynamics.parquet
- Lyapunov exponents (stability classification)
- RQA metrics (determinism, recurrence, laminarity)
- Attractor properties (correlation dimension, type)
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import os


def _process_entity(entity_id: str, obs: pl.DataFrame, params: Dict[str, Any]) -> List[Dict]:
    """Process a single entity - designed for parallel execution."""
    from engines.manifold.signal import lyapunov, attractor
    from engines.primitives.dynamical.rqa import rqa_metrics

    min_samples = params.get('min_samples', 100)
    rqa_threshold_pct = params.get('rqa_threshold_percentile', 10.0)
    rqa_min_line = params.get('rqa_min_line', 2)

    results = []
    entity_obs = obs.filter(pl.col('entity_id') == entity_id)
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
            .select('y')
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
            'entity_id': entity_id,
            'signal_id': signal_id,
            'n_samples': len(sig_data),
            'sampling_uniform': sampling_uniform,
            'sampling_cv': sampling_cv,
        }

        try:
            lyap_result = lyapunov.compute(sig_data, min_samples=50)
            result_row['lyapunov'] = lyap_result.get('lyapunov')
            result_row['stability_class'] = lyap_result.get('stability_class')
            result_row['lyapunov_confidence'] = lyap_result.get('confidence')
            embedding_dim = lyap_result.get('embedding_dim', 3) or 3
            embedding_tau = lyap_result.get('embedding_tau', 1) or 1
            result_row['embedding_dim'] = embedding_dim
            result_row['embedding_tau'] = embedding_tau
        except Exception:
            result_row['lyapunov'] = np.nan
            result_row['stability_class'] = 'unknown'
            embedding_dim = 3
            embedding_tau = 1

        try:
            attr_result = attractor.compute(sig_data, embedding_dim=embedding_dim, delay=embedding_tau)
            result_row['correlation_dim'] = attr_result.get('correlation_dim')
            result_row['attractor_type'] = attr_result.get('attractor_type')
        except Exception:
            result_row['correlation_dim'] = np.nan
            result_row['attractor_type'] = 'unknown'

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


def run_dynamics_parallel(obs: pl.DataFrame, output_dir: Path, params: Dict[str, Any] = None) -> pl.DataFrame:
    """
    Run dynamics engine on observations (parallel across entities).

    Args:
        obs: Observations with entity_id, signal_id, I, y
        output_dir: Where to write dynamics.parquet
        params: Optional parameters (embedding_dim, delay, n_jobs, etc.)

    Returns:
        DataFrame with dynamics metrics per entity/signal
    """
    from joblib import Parallel, delayed

    params = params or {}
    n_jobs = params.get('n_jobs', -1)  # -1 = all cores
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 4

    entities = obs.select('entity_id').unique().to_series().to_list()

    print(f"  [PARALLEL] Processing {len(entities)} entities on {n_jobs} cores...")

    # Parallel processing across entities
    all_results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_process_entity)(entity_id, obs, params)
        for entity_id in entities
    )

    # Flatten results
    results = []
    for entity_results in all_results:
        results.extend(entity_results)

    if not results:
        print("  Warning: no dynamics data computed")
        return pl.DataFrame()

    df = pl.DataFrame(results)

    # Write output
    output_path = output_dir / 'dynamics.parquet'
    df.write_parquet(output_path)
    print(f"  dynamics.parquet: {len(df):,} rows x {len(df.columns)} cols")

    return df
