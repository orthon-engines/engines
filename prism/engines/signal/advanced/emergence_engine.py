"""
Emergence Engine

Computes synergy, redundancy, unique information via Partial Information Decomposition.
Identifies multi-signal interactions that pairwise analysis misses.

Key insight: Emergence = whole > sum of parts (synergy).
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from itertools import combinations

from ..primitives.information import (
    mutual_information, partial_information_decomposition
)


@dataclass
class EmergenceConfig:
    """Configuration for emergence engine."""
    n_bins: int = 10
    min_samples: int = 50


class EmergenceEngine:
    """
    Emergence/Synergy Analysis Engine.

    Computes pairwise mutual information and triplet PID decomposition.

    Outputs:
    - Pairwise: mutual_information between each signal pair
    - Triplets: redundancy, unique_1, unique_2, synergy for each triplet
    - Summary: total_synergy, total_redundancy, emergence_ratio
    """

    ENGINE_TYPE = "advanced"

    def __init__(self, config: Optional[EmergenceConfig] = None):
        self.config = config or EmergenceConfig()

    def compute(
        self,
        signals: np.ndarray,
        signal_names: List[str],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute emergence metrics for multivariate signals.

        Parameters
        ----------
        signals : np.ndarray
            2D array (n_samples, n_signals)
        signal_names : list
            Names for each signal column
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Contains 'pairwise' (list), 'triplets' (list), 'summary' (dict)
        """
        signals = np.asarray(signals)
        if signals.ndim == 1:
            signals = signals.reshape(-1, 1)

        n_samples, n_signals = signals.shape

        if n_samples < self.config.min_samples:
            return self._empty_result(unit_id, signal_names)

        if n_signals < 2:
            return self._empty_result(unit_id, signal_names)

        try:
            pairwise_results = []
            triplet_results = []

            # Pairwise mutual information
            for i in range(n_signals):
                for j in range(i + 1, n_signals):
                    try:
                        mi = mutual_information(
                            signals[:, i], signals[:, j],
                            bins=self.config.n_bins
                        )
                        pairwise_results.append({
                            'signal_a': signal_names[i],
                            'signal_b': signal_names[j],
                            'mutual_information': float(mi),
                        })
                    except Exception:
                        pairwise_results.append({
                            'signal_a': signal_names[i],
                            'signal_b': signal_names[j],
                            'mutual_information': 0.0,
                        })

            # Triplet PID (all combinations of 2 sources -> 1 target)
            total_synergy = 0.0
            total_redundancy = 0.0
            total_unique = 0.0
            n_triplets = 0

            if n_signals >= 3:
                for (i, j) in combinations(range(n_signals), 2):
                    for k in range(n_signals):
                        if k in (i, j):
                            continue
                        try:
                            pid = partial_information_decomposition(
                                signals[:, i],
                                signals[:, j],
                                signals[:, k],
                                n_bins=self.config.n_bins
                            )

                            total = pid.get('total', 0)
                            syn_ratio = pid['synergy'] / total if total > 0 else 0

                            triplet_results.append({
                                'source_1': signal_names[i],
                                'source_2': signal_names[j],
                                'target': signal_names[k],
                                'redundancy': float(pid.get('redundancy', 0)),
                                'unique_1': float(pid.get('unique_1', 0)),
                                'unique_2': float(pid.get('unique_2', 0)),
                                'synergy': float(pid.get('synergy', 0)),
                                'total_info': float(total),
                                'synergy_ratio': float(syn_ratio),
                            })

                            total_synergy += pid.get('synergy', 0)
                            total_redundancy += pid.get('redundancy', 0)
                            total_unique += pid.get('unique_1', 0) + pid.get('unique_2', 0)
                            n_triplets += 1

                        except Exception:
                            continue

            # Summary metrics
            total_info = total_synergy + total_redundancy + total_unique
            emergence_ratio = total_synergy / total_info if total_info > 0 else 0.0
            redundancy_ratio = total_redundancy / total_info if total_info > 0 else 0.0

            summary = {
                'unit_id': unit_id,
                'n_samples': n_samples,
                'n_signals': n_signals,
                'n_pairs': len(pairwise_results),
                'n_triplets': n_triplets,
                'total_synergy': float(total_synergy),
                'total_redundancy': float(total_redundancy),
                'total_unique': float(total_unique),
                'emergence_ratio': float(emergence_ratio),
                'redundancy_ratio': float(redundancy_ratio),
                'mean_pairwise_mi': float(np.mean([p['mutual_information'] for p in pairwise_results])) if pairwise_results else 0.0,
            }

            return {
                'unit_id': unit_id,
                'pairwise': pairwise_results,
                'triplets': triplet_results,
                'summary': summary,
            }

        except Exception as e:
            return self._empty_result(unit_id, signal_names)

    def _empty_result(self, unit_id: str, signal_names: List[str]) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'pairwise': [],
            'triplets': [],
            'summary': {
                'unit_id': unit_id,
                'n_samples': 0,
                'n_signals': len(signal_names),
                'n_pairs': 0,
                'n_triplets': 0,
                'total_synergy': np.nan,
                'total_redundancy': np.nan,
                'total_unique': np.nan,
                'emergence_ratio': np.nan,
                'redundancy_ratio': np.nan,
                'mean_pairwise_mi': np.nan,
            },
        }


def run_emergence_engine(
    observations: pl.DataFrame,
    config: EmergenceConfig,
    signal_columns: List[str],
    output_path: Optional[Path] = None
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Run emergence engine on observations DataFrame.

    Parameters
    ----------
    observations : pl.DataFrame
        Observations with unit_id, signal_id, index, value
    config : EmergenceConfig
        Engine configuration
    signal_columns : list
        Signals to analyze
    output_path : Path, optional
        Base path for output (creates _pairwise, _triplets, _summary parquets)

    Returns
    -------
    tuple of pl.DataFrame
        (pairwise_df, triplets_df, summary_df)
    """
    engine = EmergenceEngine(config)

    entities = observations.select('unit_id').unique().to_series().to_list()
    all_pairwise = []
    all_triplets = []
    all_summary = []

    for unit_id in entities:
        entity_obs = observations.filter(pl.col('unit_id') == unit_id)

        # Pivot to get signals as columns
        signals_data = []
        for sig_col in signal_columns:
            sig_data = (
                entity_obs
                .filter(pl.col('signal_id') == sig_col)
                .sort('index')
                .select('value')
                .to_series()
                .to_numpy()
            )
            signals_data.append(sig_data)

        if len(signals_data) == 0:
            continue

        # Align to minimum length
        min_len = min(len(s) for s in signals_data)
        if min_len < config.min_samples:
            continue

        signals_matrix = np.column_stack([s[:min_len] for s in signals_data])

        result = engine.compute(signals_matrix, signal_columns, unit_id)

        # Add pairwise results
        for pair in result['pairwise']:
            pair['unit_id'] = unit_id
            all_pairwise.append(pair)

        # Add triplet results
        for triplet in result['triplets']:
            triplet['unit_id'] = unit_id
            all_triplets.append(triplet)

        # Add summary
        all_summary.append(result['summary'])

    # Create DataFrames
    df_pairwise = pl.DataFrame(all_pairwise) if all_pairwise else pl.DataFrame({
        'unit_id': [], 'signal_a': [], 'signal_b': [], 'mutual_information': []
    })

    df_triplets = pl.DataFrame(all_triplets) if all_triplets else pl.DataFrame({
        'unit_id': [], 'source_1': [], 'source_2': [], 'target': [],
        'redundancy': [], 'synergy': [], 'synergy_ratio': []
    })

    df_summary = pl.DataFrame(all_summary) if all_summary else pl.DataFrame({
        'unit_id': [], 'emergence_ratio': [], 'redundancy_ratio': []
    })

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_pairwise.write_parquet(output_path.parent / f"{output_path.stem}_pairwise.parquet")
        df_triplets.write_parquet(output_path.parent / f"{output_path.stem}_triplets.parquet")
        df_summary.write_parquet(output_path.parent / f"{output_path.stem}_summary.parquet")

    return df_pairwise, df_triplets, df_summary
