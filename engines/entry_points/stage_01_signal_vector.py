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
WINDOW SIZING
================================================================================
Window sizes are determined by:
1. Engine config (base_window, min_window, max_window) - from engine's .yaml
2. Signal's window_factor - from typology.parquet
3. Manifest overrides - optional per-signal/per-engine overrides

Effective window = engine.base_window × signal.window_factor

If typology doesn't have window_factor, defaults to 1.0.
If manifest has engine_window_overrides, those take precedence.
================================================================================
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Callable, Tuple, Set, Optional
from collections import defaultdict
import multiprocessing

from joblib import Parallel, delayed

from engines.manifold.registry import get_registry, EngineRegistry

# Hardcoded: always use all available cores
_N_WORKERS = multiprocessing.cpu_count()


# =============================================================================
# ENGINE REQUIREMENTS - NOW LOADED FROM REGISTRY
# =============================================================================
# Engine requirements (min_samples, base_window, etc.) are now defined in
# each engine's config.yaml file and loaded via the EngineRegistry.
#
# Legacy fallback dict for engines not yet migrated to config.yaml format.
# =============================================================================

_LEGACY_ENGINE_REQUIREMENTS: Dict[str, int] = {
    # Engines without config.yaml fall back to these
    'band_power': 64,
    'sample_entropy': 64,
    'approximate_entropy': 30,
    'permutation_entropy': 20,
    'perm_entropy': 20,
    'dfa': 20,
    'acf_decay': 22,
    'kurtosis': 4,
    'skewness': 4,
    'crest_factor': 4,
    'mann_kendall': 8,
    'garch': 64,
    'dmd': 32,
    'envelope': 16,
    'basin': 32,
    'cycle_counting': 16,
    'lof': 20,
    'pulsation_index': 8,
    'time_constant': 16,
    'rqa': 50,
    'peak': 2,
    'rms': 2,
    'correlation_dimension': 64,
    'determinism': 32,
    'embedding_dim': 64,
    'recurrence_rate': 32,
    'hmm': 128,
}

# Default minimum for unlisted engines
DEFAULT_MIN_SAMPLES = 4

# Engines that gate on min_required (config.yaml min_window) instead of
# window_size (alignment requirement). This lets engines compute with
# whatever data is available above their hard math floor.
# PR #12 migrates engines here one at a time for isolated testing.
# Once all engines are migrated, this set and the window_size gate
# can be removed entirely, making min_required the universal gate.
_RELAXED_WINDOW_GATE = {
    'hurst', 'recurrence_rate', 'determinism',
    'correlation_dimension', 'embedding_dim', 'lyapunov',
}


def get_engine_min_samples(engine_name: str) -> int:
    """Get minimum samples required for an engine."""
    registry = get_registry()

    # Try registry first (new config.yaml system)
    if registry.has_engine(engine_name):
        return registry.get_min_samples(engine_name)

    # Fall back to legacy dict
    return _LEGACY_ENGINE_REQUIREMENTS.get(engine_name, DEFAULT_MIN_SAMPLES)


def get_engine_window(engine_name: str, window_factor: float = 1.0) -> int:
    """
    Get effective window size for engine + signal combination.

    Args:
        engine_name: Name of the engine
        window_factor: Signal-specific multiplier from typology (default 1.0)

    Returns:
        Effective window size
    """
    registry = get_registry()

    if registry.has_engine(engine_name):
        return registry.get_window_for_signal(engine_name, window_factor)

    # Fall back to min_samples for legacy engines
    return get_engine_min_samples(engine_name)


def validate_engine_can_run(engine_name: str, window_size: int) -> bool:
    """Check if engine can run with given window size."""
    min_required = get_engine_min_samples(engine_name)
    return window_size >= min_required


def group_engines_by_window(
    engines: List[str],
    overrides: Dict[str, int],
    default_window: int,
    window_factor: float = 1.0,
) -> Dict[int, List[str]]:
    """
    Group engines by their required window size.

    Args:
        engines: List of engine names
        overrides: Dict of {engine_name: window_size} from manifest
        default_window: System default window size
        window_factor: Signal-specific multiplier from typology

    Returns:
        dict: {window_size: [engine_list]}
    """
    groups: Dict[int, List[str]] = {}

    for engine in engines:
        if engine in overrides:
            # Manifest override takes precedence
            window = overrides[engine]
        else:
            # Use registry-based window with factor
            window = get_engine_window(engine, window_factor)
            # Legacy engines not in registry: use their own min_samples, not default_window
            registry = get_registry()
            if not registry.has_engine(engine):
                window = get_engine_min_samples(engine)

        if window not in groups:
            groups[window] = []
        groups[window].append(engine)

    return groups


def _load_legacy_engine_registry() -> Dict[str, Callable]:
    """
    Load legacy engine compute functions.

    This is for engines that don't have config.yaml files yet.
    New engines should use the EngineRegistry instead.
    """
    from engines.manifold.signal import (
        statistics, memory, complexity, spectral, trend,
        hurst, attractor, lyapunov, garch, dmd,
        envelope, frequency_bands, harmonics,
        basin, cycle_counting, lof, pulsation_index, time_constant,
        rate_of_change, variance_growth,
        fundamental_freq, phase_coherence, snr, thd,
        adf_stat, variance_ratio,
        # Discrete/state engines (PR: Missing Discrete Engines)
        dwell_times, level_count, level_histogram, transition_matrix,
        # RQA + amplitude engines
        rqa, peak, rms,
        correlation_dimension, determinism, embedding_dim, recurrence_rate,
        # HMM regime detection
        hmm,
    )
    from engines.manifold.signal import entropy as discrete_entropy

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

        # Discrete/state engines (for DISCRETE, BINARY, STEP, EVENT signals)
        'dwell_times': dwell_times.compute,
        'level_count': level_count.compute,
        'level_histogram': level_histogram.compute,
        'transition_matrix': transition_matrix.compute,
        'entropy': discrete_entropy.compute,

        # RQA engines (full + individual)
        'rqa': rqa.compute,
        'peak': peak.compute,
        'rms': rms.compute,
        'correlation_dimension': correlation_dimension.compute,
        'determinism': determinism.compute,
        'embedding_dim': embedding_dim.compute,
        'recurrence_rate': recurrence_rate.compute,

        # HMM regime detection
        'hmm': hmm.compute,
    }


# Global legacy engine registry (loaded once)
_LEGACY_ENGINE_REGISTRY: Dict[str, Callable] = None


def _get_engine_registry() -> Dict[str, Callable]:
    """
    Get combined engine registry (new + legacy).

    Prefers new registry (config.yaml) over legacy.
    """
    global _LEGACY_ENGINE_REGISTRY
    if _LEGACY_ENGINE_REGISTRY is None:
        _LEGACY_ENGINE_REGISTRY = _load_legacy_engine_registry()
    return _LEGACY_ENGINE_REGISTRY


def get_engine_compute_func(engine_name: str) -> Optional[Callable]:
    """
    Get compute function for an engine.

    Tries new registry first, falls back to legacy.
    """
    registry = get_registry()

    # Try new registry first
    if registry.has_engine(engine_name):
        try:
            return registry.get_compute_func(engine_name)
        except ImportError:
            pass

    # Fall back to legacy
    legacy = _get_engine_registry()
    return legacy.get(engine_name)


def get_signal_data(
    observations: pl.DataFrame,
    cohort_name: str,
    signal_id: str,
) -> np.ndarray:
    """
    Extract signal data from observations.

    Args:
        observations: Observations DataFrame
        cohort_name: Cohort identifier (filters data to this cohort)
        signal_id: Signal identifier

    Returns:
        numpy array of signal values sorted by I
    """
    # Filter by BOTH cohort and signal_id to get per-engine data
    filters = pl.col('signal_id') == signal_id
    if 'cohort' in observations.columns:
        filters = filters & (pl.col('cohort') == cohort_name)

    signal_data = (
        observations
        .filter(filters)
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
    compute_func = get_engine_compute_func(engine_name)
    if compute_func is None:
        return {}
    return compute_func(window_data)


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
    legacy_registry: Dict[str, Callable]
) -> Dict[str, Any]:
    """Check all manifest engines against registries (new + legacy)."""
    all_engines: Set[str] = set()
    for cohort_signals in manifest.get('cohorts', {}).values():
        for signal_config in cohort_signals.values():
            if isinstance(signal_config, dict):
                all_engines.update(signal_config.get('engines', []))

    new_registry = get_registry()
    available = []
    missing = []

    for e in all_engines:
        if new_registry.has_engine(e) or e in legacy_registry:
            available.append(e)
        else:
            missing.append(e)

    coverage = len(available) / len(all_engines) if all_engines else 1.0

    return {
        'available': sorted(available),
        'missing': sorted(missing),
        'coverage': coverage,
        'total_requested': len(all_engines),
    }


def load_window_factors(typology_path: Path) -> Dict[str, float]:
    """
    Load window_factor from typology.parquet.

    Args:
        typology_path: Path to typology.parquet

    Returns:
        Dict mapping signal_id to window_factor (default 1.0 if not present)
    """
    if not typology_path.exists():
        return {}

    typology = pl.read_parquet(typology_path)

    if 'window_factor' not in typology.columns:
        return {}

    return dict(zip(
        typology['signal_id'].to_list(),
        typology['window_factor'].to_list()
    ))


def _compute_single_signal(
    signal_id: str,
    signal_data: np.ndarray,
    signal_config: Dict[str, Any],
    system_window: int,
    system_stride: int,
    window_factor: float = 1.0,
    cohort: str = None,
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
        window_factor: Signal-specific window multiplier from typology
        cohort: Cohort/unit identifier (passed through for grouping)

    Returns:
        List of row dicts for this signal
    """
    engines = signal_config.get('engines', [])
    overrides = signal_config.get('engine_window_overrides', {})

    if not engines or len(signal_data) == 0:
        return []

    # Group engines by window requirement (using window_factor)
    engine_groups = group_engines_by_window(engines, overrides, system_window, window_factor)

    rows = []

    # Compute windows at system stride
    for window_end in range(system_window - 1, len(signal_data), system_stride):
        row = {
            'signal_id': signal_id,
            'I': window_end,
        }
        # Include cohort if provided (for per-unit grouping in downstream stages)
        if cohort:
            row['cohort'] = cohort

        # Run each engine group with appropriate window
        for window_size, engine_list in engine_groups.items():
            window_start = max(0, window_end - window_size + 1)
            window_data = signal_data[window_start:window_end + 1]
            actual_available = len(window_data)

            for engine in engine_list:
                min_required = get_engine_min_samples(engine)
                # Relaxed engines gate on min_required (math floor).
                # Others gate on window_size (alignment requirement).
                gate = min_required if engine in _RELAXED_WINDOW_GATE else window_size
                if actual_available < gate:
                    row.update(null_output_for_engine(engine))
                    continue
                try:
                    engine_output = run_engine(engine, window_data)
                    row.update(engine_output)
                except Exception as e:
                    import sys
                    print(f"WARNING: Engine '{engine}' failed on signal '{signal_id}' I={window_end}: {e}", file=sys.stderr)
                    row.update(null_output_for_engine(engine))

        rows.append(row)

    return rows


def _prepare_signal_tasks(
    observations: pl.DataFrame,
    manifest: Dict[str, Any],
    window_factors: Dict[str, float] = None,
) -> List[Tuple[str, np.ndarray, Dict[str, Any], float, str]]:
    """
    Prepare (signal_id, signal_data, signal_config, window_factor, cohort) tuples for parallel dispatch.

    Args:
        observations: Observations DataFrame
        manifest: Manifest dict
        window_factors: Dict mapping signal_id to window_factor (from typology)

    Returns:
        List of (signal_id, signal_data, signal_config, window_factor, cohort) tuples
    """
    if window_factors is None:
        window_factors = {}

    tasks = []

    # Top-level engine_windows from manifest — these are fixed window sizes
    # (e.g., spectral: 64) that override window_factor scaling for specific engines.
    # Per-signal engine_window_overrides take precedence over these.
    global_engine_windows = {
        k: v for k, v in manifest.get('engine_windows', {}).items()
        if isinstance(v, (int, float))  # Filter out 'note' and other non-numeric keys
    }

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

            # Merge global engine_windows with per-signal overrides.
            # Per-signal overrides take precedence.
            per_signal_overrides = signal_config.get('engine_window_overrides', {})
            merged_overrides = {**global_engine_windows, **per_signal_overrides}
            if merged_overrides != per_signal_overrides:
                # Inject merged overrides into signal_config copy
                signal_config = {**signal_config, 'engine_window_overrides': merged_overrides}

            # Get window factor for this signal (default 1.0)
            factor = window_factors.get(signal_id, 1.0)

            tasks.append((signal_id, signal_data, signal_config, factor, cohort_name))

    return tasks


def compute_signal_vector(
    observations: pl.DataFrame,
    manifest: Dict[str, Any],
    verbose: bool = True,
    progress_interval: int = 100,
    output_path: str = None,
    flush_interval: int = 1000,
    window_factors: Dict[str, float] = None,
) -> pl.DataFrame:
    """
    Compute signal vector with per-engine window support.

    Automatically parallelizes across signals using all available CPU cores.
    Uses window_factor from typology to scale engine windows per-signal.

    Args:
        observations: Observations DataFrame with signal_id, I, value columns
        manifest: Manifest dict with system, cohorts, engine_windows sections
        verbose: Print progress updates
        progress_interval: Print progress every N windows (ignored in parallel mode)
        output_path: Path to write output (enables streaming mode)
        flush_interval: Flush to disk every N windows (ignored in parallel mode)
        window_factors: Dict mapping signal_id to window_factor (from typology)

    Returns:
        DataFrame with computed features per signal per window
    """
    import sys

    system_window = manifest['system']['window']
    system_stride = manifest['system']['stride']

    # Prepare signal tasks (now includes window_factor)
    tasks = _prepare_signal_tasks(observations, manifest, window_factors)

    if not tasks:
        return pl.DataFrame()

    # Count total windows for progress
    total_windows = 0
    for signal_id, signal_data, signal_config, factor, cohort in tasks:
        n_windows = max(0, (len(signal_data) - system_window) // system_stride + 1)
        total_windows += n_windows

    if verbose:
        print(f"Processing {total_windows:,} windows across {len(tasks)} signals using {_N_WORKERS} workers...")
        sys.stdout.flush()

    if len(tasks) == 1:
        # Single signal - no parallelism overhead
        signal_id, signal_data, signal_config, factor, cohort = tasks[0]
        all_rows = _compute_single_signal(
            signal_id, signal_data, signal_config, system_window, system_stride, factor, cohort
        )
    else:
        # Parallel across signals - always
        results = Parallel(n_jobs=_N_WORKERS, prefer="processes")(
            delayed(_compute_single_signal)(
                signal_id, signal_data, signal_config, system_window, system_stride, factor, cohort
            )
            for signal_id, signal_data, signal_config, factor, cohort in tasks
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
    typology_path: str = None,
) -> pl.DataFrame:
    """
    Run signal vector computation.

    Args:
        observations_path: Path to observations.parquet
        output_path: Path to write signal_vector.parquet
        manifest: Manifest dict from ORTHON (REQUIRED)
        verbose: Print progress
        typology_path: Path to typology.parquet (for window_factor)

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

    # Load window factors from typology (if available)
    window_factors = {}
    if typology_path:
        window_factors = load_window_factors(Path(typology_path))
        if verbose and window_factors:
            print(f"Loaded window_factors for {len(window_factors)} signals")

    # Load observations
    obs = pl.read_parquet(observations_path)

    # Validate obs signals vs typology signals
    if typology_path:
        typology_file = Path(typology_path)
        if typology_file.exists():
            typology_df = pl.read_parquet(typology_file)
            obs_signals = set(obs['signal_id'].unique().to_list())
            typo_signals = set(typology_df['signal_id'].unique().to_list())
            missing_from_typology = obs_signals - typo_signals
            if missing_from_typology:
                print(f"\n  WARNING: {len(missing_from_typology)} signals in observations NOT in typology:")
                for sig in sorted(missing_from_typology):
                    n = obs.filter(pl.col('signal_id') == sig).height
                    print(f"    {sig} ({n} observations)")
                print("  Re-run Orthon typology pipeline to include them.\n")

    if verbose:
        n_signals = obs['signal_id'].n_unique()
        n_obs = len(obs)
        print(f"Loaded {n_obs:,} observations across {n_signals} signals")
        system = manifest.get('system', {})
        print(f"System window={system.get('window')}, stride={system.get('stride')}")

    # Compute signal vector using core function
    df = compute_signal_vector(
        obs, manifest, verbose=verbose, output_path=output_path,
        window_factors=window_factors
    )

    # Replace infinities with null in all float columns
    float_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Float32]]
    for col in float_cols:
        df = df.with_columns(
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )

    # Prune dead columns
    meta_cols_keep = {'unit_id', 'I', 'signal_id', 'signal_name', 'n_samples',
                      'window_size', 'cohort'}
    n_rows = len(df)
    if n_rows > 0:
        # Phase 1: Drop columns >90% null globally (universally broken engines)
        dead_cols = [
            c for c in df.columns
            if c not in meta_cols_keep
            and df[c].null_count() / n_rows > 0.90
        ]
        if dead_cols:
            if verbose:
                print(f"  Pruning {len(dead_cols)} dead columns (>90% null)")
            df = df.drop(dead_cols)

        # Phase 2: Drop columns 100% null for any signal (engine not applicable
        # to that signal type — e.g. trend engines on broadband signals).
        # Different signal types have different engine sets in the manifest.
        # The wide format creates null columns for inapplicable engines.
        if 'signal_id' in df.columns:
            feature_cols = [c for c in df.columns if c not in meta_cols_keep]
            signals = df['signal_id'].unique().to_list()
            if len(signals) > 1 and feature_cols:
                inapplicable = set()
                for sig in signals:
                    sig_df = df.filter(pl.col('signal_id') == sig)
                    sig_n = len(sig_df)
                    if sig_n == 0:
                        continue
                    for c in feature_cols:
                        if c not in inapplicable and sig_df[c].null_count() == sig_n:
                            inapplicable.add(c)
                if inapplicable:
                    if verbose:
                        print(f"  Pruning {len(inapplicable)} engine-specific columns (inapplicable to ≥1 signal type)")
                    df = df.drop(list(inapplicable))

    # Always write output (overwrite if exists)
    df.write_parquet(output_path)

    if verbose:
        print(f"  {len(df):,} rows computed")
        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

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
        from engines.validation import check_prerequisites, PrerequisiteError
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
            from engines.validation import filter_constant_signals
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

    # Get typology path for window_factors
    typology_path = manifest_dir / 'typology.parquet'

    return run(
        observations_path=str(obs_path),
        output_path=str(out_path),
        manifest=manifest,
        verbose=verbose,
        typology_path=str(typology_path) if typology_path.exists() else None,
    )


def _count_manifest_signals(manifest: Dict[str, Any]) -> int:
    """Count total signals in manifest cohorts."""
    count = 0
    for cohort_signals in manifest.get('cohorts', {}).values():
        if isinstance(cohort_signals, dict):
            count += len(cohort_signals)
    return count


def main():
    """CLI entry point: python -m engines.entry_points.signal_vector <manifest.yaml>"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute signal vectors from manifest',
        usage='python -m engines.entry_points.signal_vector <manifest.yaml>'
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
