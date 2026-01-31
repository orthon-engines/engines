"""
Information Flow Engine Runner

Computes transfer entropy and causal networks between signals.

Outputs: information_flow.parquet
- Pairwise transfer entropy (directional information flow)
- Granger causality (linear predictive causality)
- Effective TE (bias-corrected)

Architecture: Sequential runner, parallelism handled by orchestrator.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from itertools import combinations


def process_entity_information_flow(
    unit_id: str,
    entity_obs: pl.DataFrame,
    params: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Process a single entity for information flow metrics.

    Args:
        unit_id: Entity identifier
        entity_obs: Observations for this entity only
        params: Optional parameters

    Returns:
        List of result dicts (one per signal pair direction)
    """
    from prism.engines.signal import transfer_entropy, granger

    params = params or {}
    min_samples = params.get('min_samples', 50)
    te_lag = params.get('te_lag', 1)
    te_bins = params.get('te_bins', 8)
    granger_lag = params.get('granger_lag', 5)
    max_pairs = params.get('max_pairs', 100)  # Limit for large signal counts

    results = []
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
            .select('value')
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

        # A → B
        try:
            te_ab = transfer_entropy.compute(y_a, y_b, lag=te_lag, n_bins=te_bins)
            te_ab_val = te_ab.get('transfer_entropy')
            ete_ab = te_ab.get('effective_te')
        except Exception:
            te_ab_val = np.nan
            ete_ab = np.nan

        # B → A
        try:
            te_ba = transfer_entropy.compute(y_b, y_a, lag=te_lag, n_bins=te_bins)
            te_ba_val = te_ba.get('transfer_entropy')
            ete_ba = te_ba.get('effective_te')
        except Exception:
            te_ba_val = np.nan
            ete_ba = np.nan

        # Granger A → B
        try:
            gr_ab = granger.compute(y_a, y_b, max_lag=granger_lag)
            gr_ab_fstat = gr_ab.get('granger_fstat')
            gr_ab_pval = gr_ab.get('granger_pvalue')
            gr_ab_sig = gr_ab.get('granger_significant', False)
        except Exception:
            gr_ab_fstat = np.nan
            gr_ab_pval = np.nan
            gr_ab_sig = False

        # Granger B → A
        try:
            gr_ba = granger.compute(y_b, y_a, max_lag=granger_lag)
            gr_ba_fstat = gr_ba.get('granger_fstat')
            gr_ba_pval = gr_ba.get('granger_pvalue')
            gr_ba_sig = gr_ba.get('granger_significant', False)
        except Exception:
            gr_ba_fstat = np.nan
            gr_ba_pval = np.nan
            gr_ba_sig = False

        # Add A → B row
        results.append({
            'unit_id': unit_id,
            'source': sig_a,
            'target': sig_b,
            'n_samples': len(y_a),
            'transfer_entropy': te_ab_val,
            'effective_te': ete_ab,
            'granger_fstat': gr_ab_fstat,
            'granger_pvalue': gr_ab_pval,
            'granger_significant': gr_ab_sig,
        })

        # Add B → A row
        results.append({
            'unit_id': unit_id,
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


def run_information_flow(
    obs: pl.DataFrame,
    output_dir: Path,
    params: Dict[str, Any] = None
) -> pl.DataFrame:
    """
    Run information flow engine on observations (sequential).

    For parallel execution, use process_entity_information_flow with joblib from orchestrator.

    Args:
        obs: Observations with unit_id, signal_id, I, value
        output_dir: Where to write information_flow.parquet
        params: Optional parameters (max_lag, te_bins, etc.)

    Returns:
        DataFrame with causal relationships per entity (source→target pairs)
    """
    params = params or {}
    entities = obs.select('unit_id').unique().to_series().to_list()
    all_results = []

    print(f"  Processing {len(entities)} entities...")

    for unit_id in entities:
        entity_obs = obs.filter(pl.col('unit_id') == unit_id)
        entity_results = process_entity_information_flow(unit_id, entity_obs, params)
        all_results.extend(entity_results)

    if not all_results:
        print("  Warning: no information flow data computed")
        return pl.DataFrame()

    df = pl.DataFrame(all_results)

    # Write output
    output_path = output_dir / 'information_flow.parquet'
    df.write_parquet(output_path)
    print(f"  information_flow.parquet: {len(df):,} rows x {len(df.columns)} cols")

    return df
