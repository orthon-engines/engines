"""
Signal Vector Entry Point.

Thin orchestrator that:
1. Reads manifest
2. Loads observations
3. Calls appropriate engines per-signal with per-signal config
4. Writes output to parquet

Entry point does NOT contain compute logic - only orchestration.

================================================================================
WARNING: WINDOW LOGIC BELONGS IN THE MANIFEST - NOT HERE
================================================================================
DO NOT add any window size calculation, adaptive windowing, or default window
values to this file. All windowing parameters MUST come from the manifest.

If the manifest is missing window_size or stride, this entry point MUST fail.
No defaults. No fallbacks. No "smart" calculations.

ORTHON determines window parameters. PRISM executes what the manifest says.
Hardcoding window logic here wrecks everything downstream.
================================================================================
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Callable, Tuple, Set
from collections import defaultdict


def _load_engine_registry() -> Dict[str, Callable]:
    """Load all signal engines. Each engine has a compute() method."""
    from prism.engines.signal import (
        statistics, memory, complexity, spectral, trend,
        hurst, attractor, lyapunov, garch, dmd,
        envelope, frequency_bands, harmonics,
        basin, cycle_counting, lof, pulsation_index, time_constant,
        rate_of_change, variance_growth,
        fundamental_freq, phase_coherence, snr, thd,
    )

    return {
        # Core engines
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
        'perm_entropy': complexity.compute_permutation_entropy,  # alias
        'approximate_entropy': complexity.compute_approximate_entropy,

        # Individual trend engines
        'mann_kendall': trend.compute_mann_kendall,
        'rate_of_change': trend.compute_rate_of_change,

        # Trend aliases (trend.compute returns these)
        'trend_r2': trend.compute,
        'detrend_std': trend.compute,
        'cusum': trend.compute,

        # Spectral aliases (spectral.compute returns these)
        'spectral_entropy': spectral.compute,

        # Frequency band aliases
        'band_power': frequency_bands.compute,
        'frequency_bands': frequency_bands.compute,

        # Advanced engines
        'attractor': attractor.compute,
        'lyapunov': lyapunov.compute,
        'garch': garch.compute,
        'dmd': dmd.compute,
        'envelope': envelope.compute,
        'harmonics': harmonics.compute,

        # Domain-specific engines
        'basin': basin.compute,
        'cycle_counting': cycle_counting.compute,
        'lof': lof.compute,
        'pulsation_index': pulsation_index.compute,
        'time_constant': time_constant.compute,

        # Rate of change (detailed version with mean_rate, max_rate, etc.)
        'rate_of_change_detailed': rate_of_change.compute,

        # Variance growth (non-stationarity detection)
        'variance_growth': variance_growth.compute,

        # Spectral analysis engines
        'fundamental_freq': fundamental_freq.compute,
        'phase_coherence': phase_coherence.compute,
        'snr': snr.compute,
        'thd': thd.compute,
    }


def _validate_engines(
    engine_names: List[str],
    registry: Dict[str, Callable]
) -> Tuple[List[str], List[str]]:
    """Validate engine names against registry. Returns (valid, unknown)."""
    valid = [name for name in engine_names if name in registry]
    unknown = [name for name in engine_names if name not in registry]
    return valid, unknown


def _diagnose_manifest_engines(
    manifest: Dict[str, Any],
    registry: Dict[str, Callable]
) -> Dict[str, Any]:
    """Check all manifest engines against registry."""
    all_engines: Set[str] = set()
    for cohort_signals in manifest.get('cohorts', {}).values():
        for signal_config in cohort_signals.values():
            all_engines.update(signal_config.get('engines', []))

    available = [e for e in all_engines if e in registry]
    missing = [e for e in all_engines if e not in registry]
    coverage = len(available) / len(all_engines) if all_engines else 1.0

    return {
        'available': sorted(available),
        'missing': sorted(missing),
        'coverage': coverage,
        'total_requested': len(all_engines),
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
        manifest: Manifest dict from ORTHON (REQUIRED)
        verbose: Print progress

    Returns:
        DataFrame with computed features

    Raises:
        ValueError: If manifest is missing required fields
    """
    _validate_manifest(manifest)

    # Get defaults from params
    params = manifest.get('params', {})
    default_window = params.get('default_window')
    default_stride = params.get('default_stride')

    # Load engine registry
    engine_registry = _load_engine_registry()

    # Validate engines upfront (PR13)
    diagnosis = _diagnose_manifest_engines(manifest, engine_registry)
    if verbose:
        if diagnosis['missing']:
            print(f"WARNING: Missing engines: {diagnosis['missing']}")
            print(f"  Coverage: {diagnosis['coverage']:.1%} ({len(diagnosis['available'])}/{diagnosis['total_requested']})")
        else:
            print(f"All {diagnosis['total_requested']} engines available")

    # Load observations
    obs = pl.read_parquet(observations_path)

    if verbose:
        n_signals = obs['signal_id'].n_unique()
        n_obs = len(obs)
        print(f"Loaded {n_obs:,} observations across {n_signals} signals")

    # Process each signal according to its manifest config
    results = []
    error_summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for cohort_name, cohort_signals in manifest['cohorts'].items():
        for signal_id, signal_config in cohort_signals.items():
            # Get per-signal config from manifest
            window_size = signal_config.get('window_size') or default_window
            stride = signal_config.get('stride') or default_stride
            engine_names = signal_config.get('engines', [])

            if window_size is None:
                raise ValueError(f"No window_size for signal '{signal_id}' and no default_window in params")
            if stride is None:
                raise ValueError(f"No stride for signal '{signal_id}' and no default_stride in params")
            if not engine_names:
                if verbose:
                    print(f"  Skipping {signal_id}: no engines specified")
                continue

            # Validate engines for this signal (PR13)
            valid_engines, unknown_engines = _validate_engines(engine_names, engine_registry)

            if unknown_engines and verbose:
                print(f"  {signal_id}: skipping unknown engines: {unknown_engines}")

            # Get engine functions
            active_engines = {
                name: engine_registry[name]
                for name in valid_engines
            }

            if verbose:
                print(f"  {signal_id}: window={window_size}, stride={stride}, engines={list(active_engines.keys())}")

            # Get signal data
            signal_data = (
                obs
                .filter(pl.col('signal_id') == signal_id)
                .sort('I')
            )

            if len(signal_data) == 0:
                if verbose:
                    print(f"    Warning: no data for signal '{signal_id}'")
                continue

            values = signal_data['value'].to_numpy()
            indices = signal_data['I'].to_numpy()

            # Compute features at each window
            signal_results, signal_errors = _compute_signal_features(
                signal_id=signal_id,
                values=values,
                indices=indices,
                engines=active_engines,
                window_size=window_size,
                stride=stride,
            )

            # Track errors (PR13)
            for engine_name, count in signal_errors.items():
                error_summary[signal_id][engine_name] += count

            # Convert to DataFrame immediately to preserve column schema
            if signal_results:
                signal_df = pl.DataFrame(signal_results)
                results.append(signal_df)

    # Report error summary (PR13)
    if verbose and error_summary:
        print("\nEngine errors:")
        engine_totals: Dict[str, int] = defaultdict(int)
        for signal_id, engines in error_summary.items():
            for engine, count in engines.items():
                engine_totals[engine] += count
        for engine, count in sorted(engine_totals.items()):
            print(f"  {engine}: {count} failures")

    # Concat all signal DataFrames (handles different column sets properly)
    if not results:
        df = pl.DataFrame()
    elif len(results) == 1:
        df = results[0]
    else:
        df = pl.concat(results, how='diagonal')

    # Write output
    df.write_parquet(output_path)

    if verbose:
        print(f"Wrote {len(df):,} rows to {output_path}")

    return df


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    """Validate manifest has required structure."""
    if 'cohorts' not in manifest:
        raise ValueError("Manifest missing 'cohorts' section.")

    if not manifest['cohorts']:
        raise ValueError("Manifest 'cohorts' is empty.")


def _compute_signal_features(
    signal_id: str,
    values: np.ndarray,
    indices: np.ndarray,
    engines: Dict[str, Callable],
    window_size: int,
    stride: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Compute features for one signal.

    Returns:
        (results_list, error_counts) - PR13: track errors instead of silent pass
    """
    n = len(values)
    results = []
    error_counts: Dict[str, int] = defaultdict(int)

    for i in range(0, n - window_size + 1, stride):
        window = values[i:i + window_size]
        idx = indices[i + window_size - 1]

        row = {
            'signal_id': signal_id,
            'I': int(idx),
        }

        # Run each engine (PR13: track errors)
        for name, engine_fn in engines.items():
            try:
                output = engine_fn(window)
                for key, val in output.items():
                    row[key] = val
            except Exception:
                error_counts[name] += 1

        results.append(row)

    return results, dict(error_counts)


def run_from_manifest(
    manifest_path: str,
    data_dir: str = None,
    output_dir: str = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run signal vector from manifest file.

    Args:
        manifest_path: Path to manifest.yaml
        data_dir: Directory with observations.parquet (optional, derived from manifest)
        output_dir: Directory for output (optional, derived from manifest)
        verbose: Print progress

    Returns:
        DataFrame with computed features

    Raises:
        ValueError: If manifest is missing required fields
    """
    import yaml

    manifest_path = Path(manifest_path).resolve()
    manifest_dir = manifest_path.parent

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    # Derive paths from manifest if not provided
    if data_dir is None:
        # Use manifest's paths.observations or default to manifest directory
        obs_rel = manifest.get('paths', {}).get('observations', 'observations.parquet')
        obs_path = manifest_dir / obs_rel
    else:
        obs_path = Path(data_dir) / 'observations.parquet'

    if output_dir is None:
        # Use manifest directory for output
        out_path = manifest_dir / 'signal_vector.parquet'
    else:
        out_path = Path(output_dir) / 'signal_vector.parquet'

    return run(
        observations_path=str(obs_path),
        output_path=str(out_path),
        manifest=manifest,
        verbose=verbose,
    )


def main():
    """CLI entry point: python -m prism.entry_points.signal_vector <manifest.yaml>"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute signal vectors from manifest',
        usage='python -m prism.entry_points.signal_vector <manifest.yaml>'
    )
    parser.add_argument('manifest', help='Path to manifest.yaml')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run_from_manifest(
        manifest_path=args.manifest,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
