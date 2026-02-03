"""
Signal Vector Runner.

Thin orchestrator that:
1. Reads manifest
2. Loads observations
3. Calls appropriate engines
4. Writes output to parquet

Runner does NOT contain compute logic - only orchestration.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable


# Engine registry - maps names to compute functions
def _load_engine_registry() -> Dict[str, Callable]:
    """Load all signal engines. Each engine has a compute() method."""
    from prism.engines.signal import (
        statistics, memory, complexity, spectral, trend,
        hurst, attractor, lyapunov, garch, dmd,
        envelope, frequency_bands, harmonics,
    )

    return {
        # Core engines (manifest typically specifies these)
        'statistics': statistics.compute,
        'memory': memory.compute,
        'complexity': complexity.compute,
        'spectral': spectral.compute,
        'trend': trend.compute,

        # Individual statistic engines
        'kurtosis': statistics.compute_kurtosis,
        'skewness': statistics.compute_skewness,
        'crest_factor': statistics.compute_crest_factor,

        # Individual memory engines
        'hurst': hurst.compute,
        'dfa': memory.compute_dfa,
        'acf_decay': memory.compute_acf_decay,

        # Individual complexity engines
        'sample_entropy': complexity.compute_sample_entropy,
        'permutation_entropy': complexity.compute_permutation_entropy,
        'approximate_entropy': complexity.compute_approximate_entropy,

        # Individual trend engines
        'mann_kendall': trend.compute_mann_kendall,
        'rate_of_change': trend.compute_rate_of_change,

        # Advanced engines
        'attractor': attractor.compute,
        'lyapunov': lyapunov.compute,
        'garch': garch.compute,
        'dmd': dmd.compute,
        'envelope': envelope.compute,
        'frequency_bands': frequency_bands.compute,
        'harmonics': harmonics.compute,
    }


def run(
    observations_path: str,
    output_path: str,
    manifest: Dict[str, Any],
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal vector computation.

    Args:
        observations_path: Path to observations.parquet
        output_path: Path to write signal_vector.parquet
        manifest: Manifest dict with engine config (REQUIRED)
        verbose: Print progress

    Returns:
        DataFrame with computed features

    Raises:
        ValueError: If manifest is missing required fields
    """
    # Validate manifest - NO DEFAULTS
    _validate_manifest(manifest)

    engines = manifest['engines']
    window_size = manifest['window_size']
    stride = manifest['stride']

    # Load observations
    obs = pl.read_parquet(observations_path)

    if verbose:
        n_signals = obs['signal_id'].n_unique()
        n_obs = len(obs)
        print(f"Loaded {n_obs:,} observations across {n_signals} signals")
    
    # Load engine functions
    engine_registry = _load_engine_registry()
    active_engines = {
        name: engine_registry[name]
        for name in engines
        if name in engine_registry
    }

    if verbose:
        print(f"Running engines: {list(active_engines.keys())}")

    # Process each signal (group by signal_id, NOT unit_id - unit_id is cargo for ORTHON)
    results = []

    for (signal_id,), signal_data in obs.group_by(['signal_id']):
        signal_data = signal_data.sort('I')

        values = signal_data['value'].to_numpy()
        indices = signal_data['I'].to_numpy()

        # Compute features at each window
        signal_results = _compute_signal_features(
            signal_id=signal_id,
            values=values,
            indices=indices,
            engines=active_engines,
            window_size=window_size,
            stride=stride,
        )

        results.extend(signal_results)
    
    # Create DataFrame
    df = pl.DataFrame(results)
    
    # Write output
    df.write_parquet(output_path)
    
    if verbose:
        print(f"Wrote {len(df):,} rows to {output_path}")
    
    return df


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    """Validate manifest has required fields. Reject incomplete manifests."""
    required = ['engines', 'window_size', 'stride']
    missing = [f for f in required if f not in manifest]
    if missing:
        raise ValueError(
            f"Incomplete manifest. Missing required fields: {missing}. "
            f"Manifest must specify: engines, window_size, stride. No defaults."
        )

    if not manifest['engines']:
        raise ValueError("Manifest 'engines' list cannot be empty.")

    if manifest['window_size'] < 1:
        raise ValueError("Manifest 'window_size' must be >= 1.")

    if manifest['stride'] < 1:
        raise ValueError("Manifest 'stride' must be >= 1.")


def _compute_signal_features(
    signal_id: str,
    values: np.ndarray,
    indices: np.ndarray,
    engines: Dict[str, Callable],
    window_size: int,
    stride: int,
) -> List[Dict[str, Any]]:
    """Compute features for one signal."""
    n = len(values)
    results = []

    for i in range(0, n - window_size + 1, stride):
        window = values[i:i + window_size]
        idx = indices[i + window_size - 1]

        row = {
            'signal_id': signal_id,
            'I': int(idx),
        }

        # Run each engine
        for name, engine_fn in engines.items():
            try:
                output = engine_fn(window)
                for key, val in output.items():
                    row[key] = val
            except Exception:
                pass

        results.append(row)

    return results


def run_from_manifest(
    manifest_path: str,
    data_dir: str,
    output_dir: str,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal vector from manifest file.

    Args:
        manifest_path: Path to manifest.yaml
        data_dir: Directory with observations.parquet
        output_dir: Directory for output
        verbose: Print progress

    Returns:
        DataFrame with computed features

    Raises:
        ValueError: If manifest is missing required fields
    """
    import yaml

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    obs_path = Path(data_dir) / 'observations.parquet'
    out_path = Path(output_dir) / 'signal_vector.parquet'

    return run(
        observations_path=str(obs_path),
        output_path=str(out_path),
        manifest=manifest,
        verbose=verbose,
    )
