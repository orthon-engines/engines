"""
Topology Engine Runner

Computes persistent homology for phase space embeddings.

Outputs: topology.parquet
- Betti numbers (connected components, loops, voids)
- Persistence entropy (complexity of topological structure)
- Embedding parameters used

Architecture: Sequential runner, parallelism handled by orchestrator.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def process_entity_topology(
    unit_id: str,
    entity_obs: pl.DataFrame,
    params: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Process a single entity for topology metrics.

    Args:
        unit_id: Entity identifier
        entity_obs: Observations for this entity only
        params: Optional parameters

    Returns:
        List of result dicts (one per signal)
    """
    from prism.primitives.embedding import time_delay_embedding, optimal_delay, optimal_dimension
    from prism.primitives.topology import persistence_diagram, betti_numbers, persistence_entropy

    params = params or {}
    min_samples = params.get('min_samples', 100)
    max_dim = params.get('max_dimension', 1)
    subsample_size = params.get('subsample_size', 500)  # Limit for O(n³) computation

    results = []
    signals = entity_obs.select('signal_id').unique().to_series().to_list()

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
        }

        try:
            # Optimal embedding parameters
            tau = optimal_delay(sig_data, max_lag=min(50, len(sig_data) // 10))
            dim = optimal_dimension(sig_data, tau, max_dim=5)

            # Ensure reasonable values
            tau = max(1, min(tau or 1, len(sig_data) // 10))
            dim = max(2, min(dim or 3, 5))

            result_row['embedding_tau'] = tau
            result_row['embedding_dim'] = dim

            # Create embedding
            embedded = time_delay_embedding(sig_data, dimension=dim, delay=tau)

            if len(embedded) < 10:
                continue

            # Subsample if too large (persistence is O(n³))
            if len(embedded) > subsample_size:
                indices = np.random.choice(len(embedded), subsample_size, replace=False)
                indices.sort()
                embedded_sub = embedded[indices]
                result_row['subsampled'] = True
                result_row['subsample_size'] = subsample_size
            else:
                embedded_sub = embedded
                result_row['subsampled'] = False
                result_row['subsample_size'] = len(embedded)

            # Persistence diagram
            dgm = persistence_diagram(embedded_sub, max_dimension=max_dim)

            # Betti numbers
            betti = betti_numbers(dgm)
            for d, b in betti.items():
                result_row[f'betti_{d}'] = b

            # Persistence entropy
            for d in range(max_dim + 1):
                ent = persistence_entropy(dgm, dimension=d)
                result_row[f'persistence_entropy_{d}'] = ent

            # Summary metrics
            # Total persistence (sum of bar lengths)
            for d, bars in dgm.items():
                if len(bars) > 0:
                    persistence = bars[:, 1] - bars[:, 0]
                    finite_pers = persistence[np.isfinite(persistence)]
                    if len(finite_pers) > 0:
                        result_row[f'total_persistence_{d}'] = float(np.sum(finite_pers))
                        result_row[f'max_persistence_{d}'] = float(np.max(finite_pers))
                        result_row[f'mean_persistence_{d}'] = float(np.mean(finite_pers))

        except Exception as e:
            # Topology computation failed - continue with partial results
            result_row['topology_error'] = str(e)[:100]

        results.append(result_row)

    return results


def run_topology(
    obs: pl.DataFrame,
    output_dir: Path,
    params: Dict[str, Any] = None
) -> pl.DataFrame:
    """
    Run topology engine on observations (sequential).

    For parallel execution, use process_entity_topology with joblib from orchestrator.

    Args:
        obs: Observations with unit_id, signal_id, I, value
        output_dir: Where to write topology.parquet
        params: Optional parameters (max_dimension, subsample_size, etc.)

    Returns:
        DataFrame with topology metrics per entity/signal
    """
    params = params or {}
    entities = obs.select('unit_id').unique().to_series().to_list()
    all_results = []

    print(f"  Processing {len(entities)} entities...")

    for unit_id in entities:
        entity_obs = obs.filter(pl.col('unit_id') == unit_id)
        entity_results = process_entity_topology(unit_id, entity_obs, params)
        all_results.extend(entity_results)

    if not all_results:
        print("  Warning: no topology data computed")
        return pl.DataFrame()

    df = pl.DataFrame(all_results)

    # Write output
    output_path = output_dir / 'topology.parquet'
    df.write_parquet(output_path)
    print(f"  topology.parquet: {len(df):,} rows x {len(df.columns)} cols")

    return df
