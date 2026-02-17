"""
Stage 17: Backward FTLE Entry Point
===================================

Backward FTLE reveals attracting structures - where trajectories converge TO.

Same computation as stage_08_ftle with direction='backward':
the time series is reversed before computing FTLE.

Forward FTLE (stage_08) + Backward FTLE (stage_17) together reveal the
full Lagrangian Coherent Structure.

Forward FTLE:  Repelling structures (where trajectories diverge FROM)
Backward FTLE: Attracting structures (where trajectories converge TO)

Inputs:
    - observations.parquet

Output:
    - ftle_backward.parquet

ENGINES computes FTLE values. Prime interprets attractors as regime states.
"""

import numpy as np
import polars as pl
from typing import Optional

from manifold.core.dynamics.ftle import compute as compute_ftle
from manifold.core.dynamics.formal_definitions import classify_stability
from manifold.io.writer import write_output
from manifold.utils import safe_fmt


def run(
    observations_path: str,
    data_path: str = ".",
    min_samples: int = 200,
    method: str = 'rosenstein',
    verbose: bool = True,
    intervention: dict = None,
    direction: str = 'backward',  # Force backward
) -> pl.DataFrame:
    """
    Compute backward FTLE (attracting structures).

    Same computation as forward FTLE but on time-reversed series.

    Args:
        observations_path: Path to observations.parquet
        data_path: Root data directory for output
        min_samples: Minimum samples required per signal
        method: 'rosenstein' or 'kantz'
        verbose: Print progress
        intervention: Optional intervention config dict
        direction: Always 'backward' for this stage

    Returns:
        FTLE DataFrame
    """
    backward = direction == 'backward'

    if verbose:
        print("=" * 70)
        print(f"STAGE 17: FTLE ({'BACKWARD' if backward else 'FORWARD'})")
        if backward:
            print("Backward FTLE - Attracting structures (where trajectories converge TO)")
        else:
            print("Forward FTLE - Repelling structures (where trajectories diverge FROM)")
        print("=" * 70)

    # Check for intervention mode
    intervention_enabled = intervention and intervention.get('enabled', False)
    event_index = intervention.get('event_index', 0) if intervention else 0

    if intervention_enabled and verbose:
        print(f"Intervention mode: event_index={event_index}")

    # Load observations
    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"Loaded observations: {obs.shape}")

    # Get unique signals
    signals = obs['signal_id'].unique().to_list()
    has_cohort = 'cohort' in obs.columns

    if verbose:
        print(f"Processing {len(signals)} signals...")

    results = []

    for signal_id in signals:
        signal_data = obs.filter(pl.col('signal_id') == signal_id).sort('signal_0')

        if has_cohort:
            # Process per cohort
            cohorts = signal_data['cohort'].unique().to_list()
            for cohort in cohorts:
                cohort_data = signal_data.filter(pl.col('cohort') == cohort)
                values = cohort_data['value'].to_numpy()
                s0_values = cohort_data['signal_0'].to_numpy()

                if intervention_enabled:
                    pre_mask = s0_values < event_index
                    post_mask = s0_values >= event_index
                    pre_values = values[pre_mask]
                    post_values = values[post_mask]

                    values_comp = values[::-1].copy() if backward else values
                    pre_values_comp = pre_values[::-1].copy() if backward else pre_values
                    post_values_comp = post_values[::-1].copy() if backward else post_values

                    ftle_full = compute_ftle(values_comp, min_samples=min_samples, method=method)
                    ftle_pre = compute_ftle(pre_values_comp, min_samples=min(min_samples, 50), method=method) if len(pre_values) >= 20 else {'ftle': None, 'ftle_std': None, 'confidence': 0}
                    ftle_post = compute_ftle(post_values_comp, min_samples=min_samples, method=method) if len(post_values) >= min_samples else {'ftle': None, 'ftle_std': None, 'confidence': 0}

                    prefix = 'ftle'
                    ftle_val = ftle_full['ftle']
                    if ftle_val is not None:
                        stability = classify_stability(ftle_val).value
                    elif len(values) < min_samples:
                        stability = 'insufficient_data'
                    else:
                        stability = 'unknown'
                    results.append({
                        'signal_id': signal_id,
                        'cohort': cohort,
                        prefix: ftle_val,
                        f'{prefix}_std': ftle_full['ftle_std'],
                        f'{prefix}_pre': ftle_pre.get('ftle'),
                        f'{prefix}_post': ftle_post.get('ftle'),
                        f'{prefix}_delta': (ftle_post.get('ftle') - ftle_pre.get('ftle')) if ftle_pre.get('ftle') is not None and ftle_post.get('ftle') is not None else None,
                        'embedding_dim': ftle_full['embedding_dim'],
                        'embedding_tau': ftle_full['embedding_tau'],
                        'embedding_dim_method': ftle_full.get('embedding_dim_method'),
                        'tau_method': ftle_full.get('tau_method'),
                        'is_deterministic': ftle_full.get('is_deterministic'),
                        'E1_saturation_dim': ftle_full.get('E1_saturation_dim'),
                        'confidence': ftle_full['confidence'],
                        'n_samples': len(values),
                        'n_pre': len(pre_values),
                        'n_post': len(post_values),
                        'event_index': event_index,
                        'direction': direction,
                        'stability': stability,
                    })
                else:
                    values_comp = values[::-1].copy() if backward else values
                    ftle_result = compute_ftle(
                        values_comp,
                        min_samples=min_samples,
                        method=method,
                    )

                    prefix = 'ftle'
                    ftle_val = ftle_result['ftle']
                    if ftle_val is not None:
                        stability = classify_stability(ftle_val).value
                    elif len(values) < min_samples:
                        stability = 'insufficient_data'
                    else:
                        stability = 'unknown'
                    results.append({
                        'signal_id': signal_id,
                        'cohort': cohort,
                        prefix: ftle_val,
                        f'{prefix}_std': ftle_result['ftle_std'],
                        'embedding_dim': ftle_result['embedding_dim'],
                        'embedding_tau': ftle_result['embedding_tau'],
                        'embedding_dim_method': ftle_result.get('embedding_dim_method'),
                        'tau_method': ftle_result.get('tau_method'),
                        'is_deterministic': ftle_result.get('is_deterministic'),
                        'E1_saturation_dim': ftle_result.get('E1_saturation_dim'),
                        'confidence': ftle_result['confidence'],
                        'n_samples': len(values),
                        'direction': direction,
                        'stability': stability,
                    })
        else:
            values = signal_data['value'].to_numpy()

            values_comp = values[::-1].copy() if backward else values
            ftle_result = compute_ftle(
                values_comp,
                min_samples=min_samples,
                method=method,
            )

            prefix = 'ftle'
            ftle_val = ftle_result['ftle']
            if ftle_val is not None:
                stability = classify_stability(ftle_val).value
            elif len(values) < min_samples:
                stability = 'insufficient_data'
            else:
                stability = 'unknown'
            results.append({
                'signal_id': signal_id,
                prefix: ftle_val,
                f'{prefix}_std': ftle_result['ftle_std'],
                'embedding_dim': ftle_result['embedding_dim'],
                'embedding_tau': ftle_result['embedding_tau'],
                'embedding_dim_method': ftle_result.get('embedding_dim_method'),
                'tau_method': ftle_result.get('tau_method'),
                'is_deterministic': ftle_result.get('is_deterministic'),
                'E1_saturation_dim': ftle_result.get('E1_saturation_dim'),
                'confidence': ftle_result['confidence'],
                'n_samples': len(values),
                'direction': direction,
                'stability': stability,
            })

    # Build DataFrame
    result = pl.from_dicts(results, infer_schema_length=len(results)) if results else pl.DataFrame()

    stage_name = 'ftle_backward' if backward else 'ftle'

    if len(result) > 0:
        write_output(result, data_path, stage_name, verbose=verbose)

    if verbose:
        print(f"Shape: {result.shape}")

        if len(result) > 0 and 'ftle' in result.columns:
            valid = result.filter(pl.col('ftle').is_not_null())
            if len(valid) > 0:
                label = "Backward FTLE" if backward else "Forward FTLE"
                print(f"\n{label} stats (n={len(valid)}):")
                print(f"  Mean: {safe_fmt(valid['ftle'].mean(), '.4f')}")
                print(f"  Std:  {safe_fmt(valid['ftle'].std(), '.4f')}")
                print(f"  Range: [{safe_fmt(valid['ftle'].min(), '.4f')}, {safe_fmt(valid['ftle'].max(), '.4f')}]")

    return result
