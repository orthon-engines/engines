"""
Signal Vector Entry Point.

Thin orchestrator that:
1. Validates prerequisites (observations, typology, manifest exist)
2. Filters CONSTANT signals (zero variance = zero information)
3. Reads manifest
4. Loads observations
5. Calls appropriate engines per-signal with per-signal config
6. Writes output to parquet

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
import multiprocessing

from joblib import Parallel, delayed

# Hardcoded: always use all available cores
_N_WORKERS = multiprocessing.cpu_count()


# =============================================================================
# ENGINE REQUIREMENTS
# =============================================================================
# Minimum samples required for each engine to produce valid results.
# Engines not listed default to 4 (absolute minimum for any computation).
#
# These define the "ideal" window size for accurate results. Engines receiving
# smaller windows will return NaN for those observations.
# =============================================================================

ENGINE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    # FFT-based engines (need sufficient frequency resolution)
    'spectral': {'min_samples': 64},
    'harmonics': {'min_samples': 64},
    'fundamental_freq': {'min_samples': 64},
    'thd': {'min_samples': 64},
    'frequency_bands': {'min_samples': 64},
    'band_power': {'min_samples': 64},

    # Entropy engines (need sufficient data for pattern detection)
    'sample_entropy': {'min_samples': 64},
    'complexity': {'min_samples': 50},
    'approximate_entropy': {'min_samples': 30},
    'permutation_entropy': {'min_samples': 20},
    'perm_entropy': {'min_samples': 20},

    # Fractal/memory engines (need long series for scaling analysis)
    'hurst': {'min_samples': 128},
    'dfa': {'min_samples': 20},
    'memory': {'min_samples': 20},
    'acf_decay': {'min_samples': 16},

    # Statistical engines (low requirements)
    'statistics': {'min_samples': 4},
    'kurtosis': {'min_samples': 4},
    'skewness': {'min_samples': 4},
    'crest_factor': {'min_samples': 4},

    # Spectral analysis (moderate requirements)
    'snr': {'min_samples': 32},
    'phase_coherence': {'min_samples': 32},

    # Trend engines
    'trend': {'min_samples': 8},
    'mann_kendall': {'min_samples': 8},
    'rate_of_change': {'min_samples': 4},

    # Advanced engines
    'attractor': {'min_samples': 64},
    'lyapunov': {'min_samples': 128},
    'garch': {'min_samples': 64},
    'dmd': {'min_samples': 32},
    'envelope': {'min_samples': 16},
    'variance_growth': {'min_samples': 16},

    # Domain-specific
    'basin': {'min_samples': 32},
    'cycle_counting': {'min_samples': 16},
    'lof': {'min_samples': 20},
    'pulsation_index': {'min_samples': 8},
    'time_constant': {'min_samples': 16},

    # Stationarity engines
    'adf_stat': {'min_samples': 20},
    'variance_ratio': {'min_samples': 20},
}

# Default minimum for unlisted engines
DEFAULT_MIN_SAMPLES = 4


def get_engine_min_samples(engine_name: str) -> int:
    """Get minimum samples required for an engine."""
    return ENGINE_REQUIREMENTS.get(engine_name, {}).get('min_samples', DEFAULT_MIN_SAMPLES)


def validate_engine_can_run(engine_name: str, window_size: int) -> bool:
    """Check if engine can run with given window size."""
    min_required = get_engine_min_samples(engine_name)
    return window_size >= min_required


def group_engines_by_window(
    engines: List[str],
    overrides: Dict[str, int],
    default_window: int,
) -> Dict[int, List[str]]:
    """
    Group engines by their required window size.

    Args:
        engines: List of engine names
        overrides: Dict of {engine_name: window_size} from manifest
        default_window: System default window size

    Returns:
        dict: {window_size: [engine_list]}
    """
    groups: Dict[int, List[str]] = {}

    for engine in engines:
        window = overrides.get(engine, default_window)
        if window not in groups:
            groups[window] = []
        groups[window].append(engine)

    return groups


def _load_engine_registry() -> Dict[str, Callable]:
    """Load all signal engines. Each engine has a compute() method."""
    from prism.engines.signal import (
        statistics, memory, complexity, spectral, trend,
        hurst, attractor, lyapunov, garch, dmd,
        envelope, frequency_bands, harmonics,
        basin, cycle_counting, lof, pulsation_index, time_constant,
        rate_of_change, variance_growth,
        fundamental_freq, phase_coherence, snr, thd,
        adf_stat, variance_ratio,
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

        # Stationarity engines
        'adf_stat': adf_stat.compute,
        'variance_ratio': variance_ratio.compute,
    }


# Global engine registry (loaded once)
_ENGINE_REGISTRY: Dict[str, Callable] = None


def _get_engine_registry() -> Dict[str, Callable]:
    """Get or load the engine registry."""
    global _ENGINE_REGISTRY
    if _ENGINE_REGISTRY is None:
        _ENGINE_REGISTRY = _load_engine_registry()
    return _ENGINE_REGISTRY


def get_signal_data(
    observations: pl.DataFrame,
    cohort_name: str,
    signal_id: str,
) -> np.ndarray:
    """
    Extract signal data from observations.

    Args:
        observations: Observations DataFrame
        cohort_name: Cohort name (unused, signals identified by signal_id)
        signal_id: Signal identifier

    Returns:
        numpy array of signal values sorted by I
    """
    signal_data = (
        observations
        .filter(pl.col('signal_id') == signal_id)
        .sort('I')
    )
    return signal_data['value'].to_numpy()


def run_engine(engine_name: str, window_data: np.ndarray) -> Dict[str, Any]:
    """
    Run a single engine on window data.

    Args:
        engine_name: Name of the engine
        window_data: numpy array of window values

    Returns:
        Dict of {output_key: value}
    """
    registry = _get_engine_registry()
    if engine_name not in registry:
        return {}
    return registry[engine_name](window_data)


def null_output_for_engine(engine_name: str) -> Dict[str, float]:
    """
    Get NaN output for an engine that can't run (insufficient data).

    Args:
        engine_name: Name of the engine

    Returns:
        Dict of {output_key: np.nan}
    """
    registry = _get_engine_registry()
    if engine_name not in registry:
        return {}

    try:
        # Call with minimal data to get output structure
        sample_output = registry[engine_name](np.array([0.0, 0.0, 0.0, 0.0]))
        return {k: np.nan for k in sample_output.keys()}
    except Exception:
        return {}


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


def _compute_single_signal(
    signal_id: str,
    signal_data: np.ndarray,
    signal_config: Dict[str, Any],
    system_window: int,
    system_stride: int,
) -> List[Dict[str, Any]]:
    """
    Compute all windows for one signal.

    This function runs in a worker process.

    Args:
        signal_id: Signal identifier
        signal_data: numpy array of signal values (sorted by I)
        signal_config: Config dict for this signal from manifest
        system_window: System window size
        system_stride: System stride

    Returns:
        List of row dicts for this signal
    """
    engines = signal_config.get('engines', [])
    overrides = signal_config.get('engine_window_overrides', {})

    if not engines or len(signal_data) == 0:
        return []

    # Group engines by window requirement
    engine_groups = group_engines_by_window(engines, overrides, system_window)

    rows = []

    # Compute windows at system stride
    for window_end in range(system_window - 1, len(signal_data), system_stride):
        row = {
            'signal_id': signal_id,
            'I': window_end,
        }

        # Run each engine group with appropriate window
        for window_size, engine_list in engine_groups.items():
            window_start = max(0, window_end - window_size + 1)

            # Skip if not enough data for this window
            if window_end - window_start + 1 < window_size:
                # Fill with NaN for these engines
                for engine in engine_list:
                    row.update(null_output_for_engine(engine))
                continue

            window_data = signal_data[window_start:window_end + 1]

            for engine in engine_list:
                try:
                    engine_output = run_engine(engine, window_data)
                    row.update(engine_output)
                except Exception:
                    row.update(null_output_for_engine(engine))

        rows.append(row)

    return rows


def _prepare_signal_tasks(
    observations: pl.DataFrame,
    manifest: Dict[str, Any],
) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """
    Prepare (signal_id, signal_data, signal_config) tuples for parallel dispatch.

    Returns:
        List of (signal_id, signal_data, signal_config) tuples
    """
    tasks = []

    for cohort_name, cohort_config in manifest['cohorts'].items():
        for signal_id, signal_config in cohort_config.items():
            if not isinstance(signal_config, dict):
                continue

            engines = signal_config.get('engines', [])
            if not engines:
                continue

            signal_data = get_signal_data(observations, cohort_name, signal_id)
            if len(signal_data) == 0:
                continue

            tasks.append((signal_id, signal_data, signal_config))

    return tasks


def compute_signal_vector(
    observations: pl.DataFrame,
    manifest: Dict[str, Any],
    verbose: bool = True,
    progress_interval: int = 100,
    output_path: str = None,
    flush_interval: int = 1000,
) -> pl.DataFrame:
    """
    Compute signal vector with per-engine window support.

    Automatically parallelizes across signals using all available CPU cores.

    Args:
        observations: Observations DataFrame with signal_id, I, value columns
        manifest: Manifest dict with system, cohorts, engine_windows sections
        verbose: Print progress updates
        progress_interval: Print progress every N windows (ignored in parallel mode)
        output_path: Path to write output (enables streaming mode)
        flush_interval: Flush to disk every N windows (ignored in parallel mode)

    Returns:
        DataFrame with computed features per signal per window
    """
    import sys

    system_window = manifest['system']['window']
    system_stride = manifest['system']['stride']

    # Prepare signal tasks
    tasks = _prepare_signal_tasks(observations, manifest)

    if not tasks:
        return pl.DataFrame()

    # Count total windows for progress
    total_windows = 0
    for signal_id, signal_data, signal_config in tasks:
        n_windows = max(0, (len(signal_data) - system_window) // system_stride + 1)
        total_windows += n_windows

    if verbose:
        print(f"Processing {total_windows:,} windows across {len(tasks)} signals using {_N_WORKERS} workers...")
        sys.stdout.flush()

    if len(tasks) == 1:
        # Single signal - no parallelism overhead
        signal_id, signal_data, signal_config = tasks[0]
        all_rows = _compute_single_signal(
            signal_id, signal_data, signal_config, system_window, system_stride
        )
    else:
        # Parallel across signals - always
        results = Parallel(n_jobs=_N_WORKERS, prefer="processes")(
            delayed(_compute_single_signal)(
                signal_id, signal_data, signal_config, system_window, system_stride
            )
            for signal_id, signal_data, signal_config in tasks
        )

        # Flatten results
        all_rows = []
        for signal_rows in results:
            if signal_rows:
                all_rows.extend(signal_rows)

    if verbose:
        print(f"  {len(all_rows):,} rows computed", flush=True)

    if not all_rows:
        return pl.DataFrame()

    # Use infer_schema_length=None to scan ALL rows for schema inference
    # This ensures columns that only appear in some signals are not dropped
    return pl.DataFrame(all_rows, infer_schema_length=None)


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

    # Load engine registry for validation
    engine_registry = _get_engine_registry()

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
        system = manifest.get('system', {})
        print(f"System window={system.get('window')}, stride={system.get('stride')}")

    # Compute signal vector using core function
    # Streaming mode (>1000 windows) writes directly to output_path
    df = compute_signal_vector(obs, manifest, verbose=verbose, output_path=output_path)

    # Check if file was already written by streaming mode
    output_exists = Path(output_path).exists()
    if not output_exists:
        df.write_parquet(output_path)

    if verbose:
        print(f"Wrote {len(df):,} rows to {output_path}")

    return df


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    """Validate manifest has required structure."""
    if 'system' not in manifest:
        raise ValueError("Manifest missing 'system' section.")

    system = manifest['system']
    if 'window' not in system:
        raise ValueError("Manifest 'system' section missing 'window'.")
    if 'stride' not in system:
        raise ValueError("Manifest 'system' section missing 'stride'.")

    if 'cohorts' not in manifest:
        raise ValueError("Manifest missing 'cohorts' section.")

    if not manifest['cohorts']:
        raise ValueError("Manifest 'cohorts' is empty.")


def run_from_manifest(
    manifest_path: str,
    data_dir: str = None,
    output_dir: str = None,
    verbose: bool = True,
    skip_prerequisites: bool = False,
    filter_constants: bool = True,
) -> pl.DataFrame:
    """
    Run signal vector from manifest file.

    Args:
        manifest_path: Path to manifest.yaml
        data_dir: Directory with observations.parquet (optional, derived from manifest)
        output_dir: Directory for output (optional, derived from manifest)
        verbose: Print progress
        skip_prerequisites: If True, skip prerequisite validation (not recommended)
        filter_constants: If True, filter CONSTANT signals from manifest (default)

    Returns:
        DataFrame with computed features

    Raises:
        ValueError: If manifest is missing required fields
        PrerequisiteError: If required files are missing
    """
    import yaml

    manifest_path = Path(manifest_path).resolve()
    manifest_dir = manifest_path.parent

    # Check prerequisites first
    if not skip_prerequisites:
        from prism.validation import check_prerequisites, PrerequisiteError
        if verbose:
            print("Checking prerequisites...")
        check_prerequisites('signal_vector', str(manifest_dir), raise_on_missing=True)
        if verbose:
            print("  Prerequisites satisfied.")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    # Filter CONSTANT signals from manifest
    if filter_constants:
        typology_path = manifest_dir / 'typology.parquet'
        if typology_path.exists():
            from prism.validation import filter_constant_signals
            typology_df = pl.read_parquet(typology_path)
            original_count = _count_manifest_signals(manifest)
            manifest = filter_constant_signals(manifest, typology_df, verbose=False)
            filtered_count = _count_manifest_signals(manifest)
            if verbose and original_count != filtered_count:
                print(f"Filtered {original_count - filtered_count} CONSTANT signal(s)")

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


def _count_manifest_signals(manifest: Dict[str, Any]) -> int:
    """Count total signals in manifest cohorts."""
    count = 0
    for cohort_signals in manifest.get('cohorts', {}).values():
        if isinstance(cohort_signals, dict):
            count += len(cohort_signals)
    return count


def main():
    """CLI entry point: python -m prism.entry_points.signal_vector <manifest.yaml>"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute signal vectors from manifest',
        usage='python -m prism.entry_points.signal_vector <manifest.yaml>'
    )
    parser.add_argument('manifest', help='Path to manifest.yaml')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    parser.add_argument(
        '--skip-prerequisites',
        action='store_true',
        help='Skip prerequisite validation (not recommended)'
    )
    parser.add_argument(
        '--no-filter-constants',
        action='store_true',
        help='Do not filter CONSTANT signals'
    )

    args = parser.parse_args()

    run_from_manifest(
        manifest_path=args.manifest,
        verbose=not args.quiet,
        skip_prerequisites=args.skip_prerequisites,
        filter_constants=not args.no_filter_constants,
    )


if __name__ == '__main__':
    main()
