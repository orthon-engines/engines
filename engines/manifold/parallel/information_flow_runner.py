"""
Parallel Information Flow Engine Runner (Digital Ocean)

Computes transfer entropy and causal networks between signals using joblib parallelization.
Use on multi-core servers, not on Mac.

Outputs: information_flow.parquet
- Pairwise transfer entropy (directional information flow)
- Granger causality (linear predictive causality)
- Effective TE (bias-corrected)
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from itertools import combinations
import os


def _process_entity(entity_id: str, obs: pl.DataFrame, params: Dict[str, Any]) -> List[Dict]:
    """Process a single entity - designed for parallel execution."""
    from engines.manifold.signal import transfer_entropy, granger

    min_samples = params.get('min_samples', 50)
    te_lag = params.get('te_lag', 1)
    te_bins = params.get('te_bins', 8)
    granger_lag = params.get('granger_lag', 5)
    max_pairs = params.get('max_pairs', 100)

    results = []
    entity_obs = obs.filter(pl.col('entity_id') == entity_id)
    signals = entity_obs.select('signal_id').unique().to_series().to_list()

    if len(signals) < 2:
        return results

    # Build signal data dictionary
    signal_data = {}
    min_len = float('inf')

    for signal_id in signals:
        sig = (
            entity_obs
            .filter(pl.col('signal_id') == signal_id)
            .sort('I')
            .select('y')
            .to_series()
            .to_numpy()
        )
        # Remove NaN
        sig = sig[~np.isnan(sig)]
        if len(sig) >= min_samples:
            signal_data[signal_id] = sig
            min_len = min(min_len, len(sig))

    if len(signal_data) < 2:
        return results

    # Truncate all to same length
    for k in signal_data:
        signal_data[k] = signal_data[k][:int(min_len)]

    # Get all pairs
    signal_ids = list(signal_data.keys())
    all_pairs = list(combinations(signal_ids, 2))

    # Limit pairs if too many
    if len(all_pairs) > max_pairs:
        np.random.shuffle(all_pairs)
        all_pairs = all_pairs[:max_pairs]

    # Compute pairwise transfer entropy
    for sig_a, sig_b in all_pairs:
        y_a = signal_data[sig_a]
        y_b = signal_data[sig_b]

        # A -> B
        try:
            te_ab = transfer_entropy.compute(y_a, y_b, lag=te_lag, n_bins=te_bins)
            te_ab_val = te_ab.get('transfer_entropy')
            ete_ab = te_ab.get('effective_te')
        except Exception:
            te_ab_val = np.nan
            ete_ab = np.nan

        # B -> A
        try:
            te_ba = transfer_entropy.compute(y_b, y_a, lag=te_lag, n_bins=te_bins)
            te_ba_val = te_ba.get('transfer_entropy')
            ete_ba = te_ba.get('effective_te')
        except Exception:
            te_ba_val = np.nan
            ete_ba = np.nan

        # Granger A -> B
        try:
            gr_ab = granger.compute(y_a, y_b, max_lag=granger_lag)
            gr_ab_fstat = gr_ab.get('granger_fstat')
            gr_ab_pval = gr_ab.get('granger_pvalue')
            gr_ab_sig = gr_ab.get('granger_significant', False)
        except Exception:
            gr_ab_fstat = np.nan
            gr_ab_pval = np.nan
            gr_ab_sig = False

        # Granger B -> A
        try:
            gr_ba = granger.compute(y_b, y_a, max_lag=granger_lag)
            gr_ba_fstat = gr_ba.get('granger_fstat')
            gr_ba_pval = gr_ba.get('granger_pvalue')
            gr_ba_sig = gr_ba.get('granger_significant', False)
        except Exception:
            gr_ba_fstat = np.nan
            gr_ba_pval = np.nan
            gr_ba_sig = False

        # Add A -> B row
        results.append({
            'entity_id': entity_id,
            'source': sig_a,
            'target': sig_b,
            'n_samples': len(y_a),
            'transfer_entropy': te_ab_val,
            'effective_te': ete_ab,
            'granger_fstat': gr_ab_fstat,
            'granger_pvalue': gr_ab_pval,
            'granger_significant': gr_ab_sig,
        })

        # Add B -> A row
        results.append({
            'entity_id': entity_id,
            'source': sig_b,
            'target': sig_a,
            'n_samples': len(y_b),
            'transfer_entropy': te_ba_val,
            'effective_te': ete_ba,
            'granger_fstat': gr_ba_fstat,
            'granger_pvalue': gr_ba_pval,
            'granger_significant': gr_ba_sig,
        })

    return results


def run_information_flow_parallel(obs: pl.DataFrame, output_dir: Path, params: Dict[str, Any] = None) -> pl.DataFrame:
    """
    Run information flow engine on observations (parallel across entities).

    Args:
        obs: Observations with entity_id, signal_id, I, y
        output_dir: Where to write information_flow.parquet
        params: Optional parameters (max_lag, te_bins, n_jobs, etc.)

    Returns:
        DataFrame with causal relationships per entity (source->target pairs)
    """
    from joblib import Parallel, delayed

    params = params or {}
    n_jobs = params.get('n_jobs', -1)
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
        print("  Warning: no information flow data computed")
        return pl.DataFrame()

    df = pl.DataFrame(results)

    # Write output
    output_path = output_dir / 'information_flow.parquet'
    df.write_parquet(output_path)
    print(f"  information_flow.parquet: {len(df):,} rows x {len(df.columns)} cols")

    return df
