"""
Parallel Topology Engine Runner (Digital Ocean)

Computes persistent homology for phase space embeddings using joblib parallelization.
Use on multi-core servers, not on Mac.

Outputs: topology.parquet
- Betti numbers (connected components, loops, voids)
- Persistence entropy (complexity of topological structure)
- Embedding parameters used
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import os


def _process_entity(entity_id: str, obs: pl.DataFrame, params: Dict[str, Any]) -> List[Dict]:
    """Process a single entity - designed for parallel execution."""
    from engines.primitives.embedding import time_delay_embedding, optimal_delay, optimal_dimension
    from engines.primitives.topology import persistence_diagram, betti_numbers, persistence_entropy

    min_samples = params.get('min_samples', 100)
    max_dim = params.get('max_dimension', 1)
    subsample_size = params.get('subsample_size', 500)

    results = []
    entity_obs = obs.filter(pl.col('entity_id') == entity_id)
    signals = entity_obs.select('signal_id').unique().to_series().to_list()

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

            # Subsample if too large (persistence is O(n^3))
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
            for d, bars in dgm.items():
                if len(bars) > 0:
                    persistence = bars[:, 1] - bars[:, 0]
                    finite_pers = persistence[np.isfinite(persistence)]
                    if len(finite_pers) > 0:
                        result_row[f'total_persistence_{d}'] = float(np.sum(finite_pers))
                        result_row[f'max_persistence_{d}'] = float(np.max(finite_pers))
                        result_row[f'mean_persistence_{d}'] = float(np.mean(finite_pers))

        except Exception as e:
            result_row['topology_error'] = str(e)[:100]

        results.append(result_row)

    return results


def run_topology_parallel(obs: pl.DataFrame, output_dir: Path, params: Dict[str, Any] = None) -> pl.DataFrame:
    """
    Run topology engine on observations (parallel across entities).

    Args:
        obs: Observations with entity_id, signal_id, I, y
        output_dir: Where to write topology.parquet
        params: Optional parameters (max_dimension, subsample_size, n_jobs, etc.)

    Returns:
        DataFrame with topology metrics per entity/signal
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
        print("  Warning: no topology data computed")
        return pl.DataFrame()

    df = pl.DataFrame(results)

    # Write output
    output_path = output_dir / 'topology.parquet'
    df.write_parquet(output_path)
    print(f"  topology.parquet: {len(df):,} rows x {len(df.columns)} cols")

    return df
