"""
ENGINES Geometry Dynamics Engine

The complete differential geometry framework.
Computes derivatives and curvature for the geometry evolution over time.

"You have position (state_vector).
 You have shape (eigenvalues).
 Now here are the derivatives."

Computes:
- First derivatives (velocity/tangent)
- Second derivatives (acceleration/curvature)
- Third derivatives (jerk/torsion)
- Collapse detection (computed index, not interpretation)
- Phase space analysis

NOTE: ENGINES computes, never classifies. All classification logic
has been removed. ORTHON interprets the computed values.

ARCHITECTURE: This is an ORCHESTRATOR that delegates all compute to ENGINES primitives.
All mathematical operations are performed by engines.* functions.

INPUT:
- state_geometry.parquet (eigenvalues over time)
- signal_geometry.parquet (signal positions over time)

OUTPUT:
- geometry_dynamics.parquet (system-level dynamics)
- signal_dynamics.parquet (per-signal dynamics)

Credit: The emeritus PhD mathematicians who would accept nothing less.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Import ENGINES primitives for all mathematical computation
import engines

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_config


# ============================================================
# CONFIGURATION-DRIVEN DEFAULTS
# ============================================================

def _get_dynamics_config() -> Dict[str, Any]:
    """Get dynamics computation config."""
    config = get_config()
    return {
        'dt': config.get('dynamics.derivatives.dt', 1.0),
        'smooth_window': config.get('dynamics.derivatives.smooth_window', 3),
        'method': config.get('dynamics.derivatives.method', 'central'),
    }


def _get_collapse_config() -> Dict[str, Any]:
    """Get collapse detection config."""
    config = get_config()
    return {
        'threshold_velocity': config.get('dynamics.collapse.threshold_velocity', -0.1),
        'sustained_fraction': config.get('dynamics.collapse.sustained_fraction', 0.3),
        'min_collapse_length': config.get('dynamics.collapse.min_collapse_length', 5),
    }


def _get_phase_space_config() -> Dict[str, Any]:
    """Get phase space config."""
    config = get_config()
    return {
        'embedding_dim': config.get('dynamics.phase_space.embedding_dim', 2),
        'tau': config.get('dynamics.phase_space.tau', 1),
    }


# ============================================================
# DERIVATIVE COMPUTATION
# ============================================================

def compute_derivatives(
    x: np.ndarray,
    dt: Optional[float] = None,
    smooth_window: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compute derivatives up to third order with optional smoothing.

    ARCHITECTURE: Pure orchestration - delegates all math to ENGINES primitives.

    Args:
        x: Time series values
        dt: Time step (from config if not provided)
        smooth_window: Smoothing window for noise reduction (from config if not provided)

    Returns:
        Dict with velocity, acceleration, jerk, curvature, speed
    """
    # Get config values
    dynamics_config = _get_dynamics_config()
    if dt is None:
        dt = dynamics_config['dt']
    if smooth_window is None:
        smooth_window = dynamics_config['smooth_window']

    n = len(x)

    if n < 3:
        return {
            'velocity': np.zeros(n),
            'acceleration': np.zeros(n),
            'jerk': np.zeros(n),
            'curvature': np.zeros(n),
            'speed': np.zeros(n),
        }

    # Optional smoothing (preprocessing, not math)
    if smooth_window > 1 and n > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        x_smooth = np.convolve(x, kernel, mode='same')
    else:
        x_smooth = x

    # ─────────────────────────────────────────────────
    # DERIVATIVES → ENGINES PRIMITIVES
    # ─────────────────────────────────────────────────

    # First derivative (velocity) → ENGINES PRIMITIVE
    dx = engines.first_derivative(x_smooth, dt=dt, method='central')

    # Second derivative (acceleration) → ENGINES PRIMITIVE
    d2x = engines.second_derivative(x_smooth, dt=dt, method='central')

    # Third derivative (jerk) → ENGINES PRIMITIVE
    d3x = engines.jerk(x_smooth, dt=dt)

    # ─────────────────────────────────────────────────
    # 1D CURVATURE: κ = |d²x/dt²| / (1 + (dx/dt)²)^(3/2)
    # Note: engines.curvature is for 2D trajectories (x, y)
    # For 1D time series, we compute curvature directly
    # ─────────────────────────────────────────────────
    denom = (1 + dx**2)**1.5
    curvature = np.where(denom > 1e-10, np.abs(d2x) / denom, 0)

    # Speed (magnitude of velocity)
    speed = np.abs(dx)

    return {
        'velocity': dx,
        'acceleration': d2x,
        'jerk': d3x,
        'curvature': curvature,
        'speed': speed,
    }


def compute_phase_space(
    x: np.ndarray,
    embedding_dim: Optional[int] = None,
    tau: Optional[int] = None
) -> np.ndarray:
    """
    Reconstruct phase space using time-delay embedding.

    ARCHITECTURE: Pure orchestration - delegates to ENGINES primitive.

    Takens' theorem: The attractor can be reconstructed from
    a single time series using delay coordinates.

    Args:
        x: Time series
        embedding_dim: Number of dimensions (from config if not provided)
        tau: Time delay (from config if not provided)

    Returns:
        Phase space coordinates (n_points × embedding_dim)
    """
    # Get config values
    phase_config = _get_phase_space_config()
    if embedding_dim is None:
        embedding_dim = phase_config['embedding_dim']
    if tau is None:
        tau = phase_config['tau']
    # ─────────────────────────────────────────────────
    # ATTRACTOR RECONSTRUCTION → ENGINES PRIMITIVE
    # ─────────────────────────────────────────────────
    return engines.attractor_reconstruction(x, embed_dim=embedding_dim, tau=tau)


# ============================================================
# COLLAPSE DETECTION
# ============================================================

def detect_collapse(
    effective_dim: np.ndarray,
    threshold_velocity: Optional[float] = None,
    sustained_fraction: Optional[float] = None,
    min_collapse_length: Optional[int] = None
) -> Dict[str, Any]:
    """
    Detect dimensional collapse in effective_dim time series.

    Collapse = sustained negative velocity in effective_dim
    indicating the system is losing degrees of freedom.

    Args:
        effective_dim: Effective dimension over time
        threshold_velocity: Velocity below this = collapsing (from config if not provided)
        sustained_fraction: Fraction of points that must be collapsing (from config if not provided)
        min_collapse_length: Minimum consecutive points for collapse (from config if not provided)

    Returns:
        Collapse detection results
    """
    # Get config values
    collapse_config = _get_collapse_config()
    if threshold_velocity is None:
        threshold_velocity = collapse_config['threshold_velocity']
    if sustained_fraction is None:
        sustained_fraction = collapse_config['sustained_fraction']
    if min_collapse_length is None:
        min_collapse_length = collapse_config['min_collapse_length']

    n = len(effective_dim)

    # Return computed values only - no boolean classification
    # collapse_onset_idx = None means no collapse detected
    if n < min_collapse_length:
        return {
            'collapse_onset_idx': None,
            'collapse_onset_fraction': None,
        }

    # Compute derivatives
    deriv = compute_derivatives(effective_dim)
    velocity = deriv['velocity']

    # Identify collapsing regions
    collapsing = velocity < threshold_velocity

    # Find sustained collapse (consecutive points)
    collapse_runs = []
    run_start = None

    for i in range(n):
        if collapsing[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_length = i - run_start
                if run_length >= min_collapse_length:
                    collapse_runs.append((run_start, i, run_length))
                run_start = None

    # Handle run that extends to end
    if run_start is not None:
        run_length = n - run_start
        if run_length >= min_collapse_length:
            collapse_runs.append((run_start, n, run_length))

    if not collapse_runs:
        return {
            'collapse_onset_idx': None,
            'collapse_onset_fraction': None,
        }

    # Take the longest collapse run
    longest_run = max(collapse_runs, key=lambda x: x[2])
    onset_idx, end_idx, duration = longest_run

    # Return computed index/fraction only - no boolean
    return {
        'collapse_onset_idx': int(onset_idx),
        'collapse_onset_fraction': onset_idx / n,
    }


# ============================================================
# GEOMETRY DYNAMICS COMPUTATION
# ============================================================

def compute_geometry_dynamics(
    state_geometry_path: str,
    output_path: str = "geometry_dynamics.parquet",
    dt: Optional[float] = None,
    smooth_window: Optional[int] = None,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute dynamics of geometry evolution.

    For each (unit_id, engine), computes derivatives of:
    - effective_dim
    - eigenvalues
    - total_variance

    Args:
        state_geometry_path: Path to state_geometry.parquet
        output_path: Output path
        dt: Time step (from config if not provided)
        smooth_window: Smoothing window (from config if not provided)
        verbose: Print progress

    Returns:
        Geometry dynamics DataFrame
    """
    # Get config values
    dynamics_config = _get_dynamics_config()
    if dt is None:
        dt = dynamics_config['dt']
    if smooth_window is None:
        smooth_window = dynamics_config['smooth_window']

    if verbose:
        print("=" * 70)
        print("GEOMETRY DYNAMICS ENGINE")
        print("Differential geometry of state evolution")
        print("=" * 70)

    # Load state geometry
    state_geometry = pl.read_parquet(state_geometry_path)

    if verbose:
        print(f"Loaded: {len(state_geometry)} rows")
        print(f"Columns: {state_geometry.columns}")

    # Determine grouping columns - include cohort if present
    has_cohort = 'cohort' in state_geometry.columns
    group_cols = ['cohort', 'engine'] if has_cohort else ['engine']

    # Process each (cohort, engine) or just engine
    results = []
    groups = state_geometry.group_by(group_cols, maintain_order=True)

    for group_key, group in groups:
        if has_cohort:
            if isinstance(group_key, tuple):
                cohort, engine = group_key
            else:
                cohort, engine = None, group_key
        else:
            cohort = None
            engine = group_key[0] if isinstance(group_key, tuple) else group_key
        unit_id = group['unit_id'][0] if 'unit_id' in group.columns else ''
        # Sort by I
        group = group.sort('I')

        I_values = group['I'].to_numpy()
        n = len(I_values)

        if n < 3:
            continue

        # Extract time series
        effective_dim = group['effective_dim'].to_numpy()
        eigenvalue_1 = group['eigenvalue_1'].to_numpy()
        total_variance = group['total_variance'].to_numpy()

        # Compute derivatives
        eff_dim_deriv = compute_derivatives(effective_dim, dt, smooth_window)
        eigen_1_deriv = compute_derivatives(eigenvalue_1, dt, smooth_window)
        variance_deriv = compute_derivatives(total_variance, dt, smooth_window)

        # Detect collapse (computed values only, no classification)
        collapse = detect_collapse(effective_dim)

        # Build result rows - computed values only, NO classification
        for i in range(n):
            row = {
                'I': int(I_values[i]),
                'engine': engine,

                # Effective dimension dynamics
                'effective_dim': effective_dim[i],
                'effective_dim_velocity': eff_dim_deriv['velocity'][i],
                'effective_dim_acceleration': eff_dim_deriv['acceleration'][i],
                'effective_dim_jerk': eff_dim_deriv['jerk'][i],
                'effective_dim_curvature': eff_dim_deriv['curvature'][i],

                # Eigenvalue dynamics
                'eigenvalue_1': eigenvalue_1[i],
                'eigenvalue_1_velocity': eigen_1_deriv['velocity'][i],

                # Variance dynamics
                'total_variance': total_variance[i],
                'variance_velocity': variance_deriv['velocity'][i],

                # Collapse detection (computed index/fraction, not interpretation)
                'collapse_onset_idx': collapse['collapse_onset_idx'],
                'collapse_onset_fraction': collapse['collapse_onset_fraction'],
            }
            # Include cohort if available
            if cohort:
                row['cohort'] = cohort
            if unit_id:
                row['unit_id'] = unit_id
            results.append(row)

    # Build DataFrame
    result = pl.DataFrame(results)
    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        # Summary - computed values only
        if 'collapse_onset_idx' in result.columns:
            n_collapse = result.filter(pl.col('collapse_onset_idx').is_not_null())['engine'].n_unique()
            n_total = result['engine'].n_unique()
            print(f"\nCollapse onset detected: {n_collapse} / {n_total} engines")

    return result


# ============================================================
# SIGNAL DYNAMICS COMPUTATION
# ============================================================

def compute_signal_dynamics(
    signal_geometry_path: str,
    output_path: str = "signal_dynamics.parquet",
    dt: float = 1.0,
    smooth_window: int = 3,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute dynamics of individual signal evolution.

    For each (unit_id, signal_id), computes derivatives of:
    - distance to state
    - coherence to state
    - contribution

    Args:
        signal_geometry_path: Path to signal_geometry.parquet
        output_path: Output path
        dt: Time step
        smooth_window: Smoothing window
        verbose: Print progress

    Returns:
        Signal dynamics DataFrame
    """
    if verbose:
        print("=" * 70)
        print("SIGNAL DYNAMICS ENGINE")
        print("Per-signal trajectory analysis")
        print("=" * 70)

    # Load signal geometry
    signal_geometry = pl.read_parquet(signal_geometry_path)

    # Detect column naming
    distance_cols = [c for c in signal_geometry.columns if c.startswith('distance_')]
    coherence_cols = [c for c in signal_geometry.columns if c.startswith('coherence_')]

    if verbose:
        print(f"Loaded: {len(signal_geometry)} rows")
        print(f"Distance columns: {distance_cols}")
        print(f"Coherence columns: {coherence_cols}")

    # Determine grouping columns - include cohort if present
    has_cohort = 'cohort' in signal_geometry.columns
    signal_col = 'signal_id'
    group_cols = ['cohort', signal_col] if has_cohort else [signal_col]

    # Process each (cohort, signal_id) or just signal_id
    results = []
    groups = signal_geometry.group_by(group_cols, maintain_order=True)
    n_groups = signal_geometry.select(group_cols).unique().height

    if verbose:
        if has_cohort:
            n_cohorts = signal_geometry['cohort'].n_unique()
            print(f"Processing {n_groups} (cohort, signal_id) groups across {n_cohorts} cohorts...")
        else:
            print(f"Processing {n_groups} signal groups...")

    processed = 0
    for group_key, group in groups:
        if has_cohort:
            if isinstance(group_key, tuple):
                cohort, signal_id = group_key
            else:
                cohort, signal_id = None, group_key
        else:
            cohort = None
            signal_id = group_key[0] if isinstance(group_key, tuple) else group_key
        # Skip null signal_id
        if signal_id is None:
            continue
        unit_id = group['unit_id'][0] if 'unit_id' in group.columns else ''

        # Sort by I
        group = group.sort('I')

        I_values = group['I'].to_numpy()
        n = len(I_values)

        if n < 3:
            continue

        # Process each engine's metrics
        for dist_col in distance_cols:
            engine = dist_col.replace('distance_', '')
            coh_col = f'coherence_{engine}'

            if coh_col not in group.columns:
                continue

            distance = group[dist_col].to_numpy()
            coherence = group[coh_col].to_numpy()

            # Skip if all NaN
            if np.all(np.isnan(distance)) or np.all(np.isnan(coherence)):
                continue

            # Compute derivatives
            dist_deriv = compute_derivatives(distance, dt, smooth_window)
            coh_deriv = compute_derivatives(coherence, dt, smooth_window)

            # Build result rows - computed values only, NO classification
            for i in range(n):
                row = {
                    'I': int(I_values[i]),
                    'signal_id': signal_id,
                    'engine': engine,

                    # Distance dynamics
                    'distance': distance[i],
                    'distance_velocity': dist_deriv['velocity'][i],
                    'distance_acceleration': dist_deriv['acceleration'][i],
                    'distance_curvature': dist_deriv['curvature'][i],

                    # Coherence dynamics
                    'coherence': coherence[i],
                    'coherence_velocity': coh_deriv['velocity'][i],
                    'coherence_acceleration': coh_deriv['acceleration'][i],
                }
                # Include cohort if available
                if cohort:
                    row['cohort'] = cohort
                if unit_id:
                    row['unit_id'] = unit_id
                results.append(row)

        processed += 1
        if verbose and processed % 20 == 0:
            print(f"  Processed {processed}/{n_groups} signals...")

    # Build DataFrame
    result = pl.DataFrame(results)
    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        # Summary - computed values only
        if len(result) > 0:
            mean_dist_vel = result['distance_velocity'].mean()
            mean_coh_vel = result['coherence_velocity'].mean()
            print(f"\nMean distance velocity: {mean_dist_vel:.4f}")
            print(f"Mean coherence velocity: {mean_coh_vel:.4f}")

    return result


# ============================================================
# PAIRWISE DYNAMICS
# ============================================================

def compute_pairwise_dynamics(
    signal_pairwise_path: str,
    output_path: str = "pairwise_dynamics.parquet",
    dt: float = 1.0,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Compute dynamics of pairwise relationships.

    How are signal-signal relationships evolving?
    - Coupling strengthening or weakening?
    - Synchronization/desynchronization?
    """
    if verbose:
        print("=" * 70)
        print("PAIRWISE DYNAMICS ENGINE")
        print("Evolution of signal-signal relationships")
        print("=" * 70)

    # Load pairwise
    pairwise = pl.read_parquet(signal_pairwise_path)

    if verbose:
        print(f"Loaded: {len(pairwise)} rows")

    # Determine grouping columns - include cohort if present
    has_cohort = 'cohort' in pairwise.columns
    base_group_cols = ['signal_a', 'signal_b', 'engine']
    group_cols = ['cohort'] + base_group_cols if has_cohort else base_group_cols

    # Process each (cohort, signal_a, signal_b, engine) or just (signal_a, signal_b, engine)
    results = []
    groups = pairwise.group_by(group_cols, maintain_order=True)

    for group_key, group in groups:
        if has_cohort:
            if isinstance(group_key, tuple):
                cohort, sig_a, sig_b, engine = group_key
            else:
                cohort, sig_a, sig_b, engine = None, group_key, None, None
        else:
            cohort = None
            sig_a, sig_b, engine = group_key if isinstance(group_key, tuple) else (group_key, None, None)
        unit_id = group['unit_id'][0] if 'unit_id' in group.columns else ''
        group = group.sort('I')

        I_values = group['I'].to_numpy()
        n = len(I_values)

        if n < 3:
            continue

        correlation = group['correlation'].to_numpy()
        distance = group['distance'].to_numpy()

        # Compute derivatives
        corr_deriv = compute_derivatives(correlation, dt)
        dist_deriv = compute_derivatives(distance, dt)

        # Classification
        # Coupling strengthening if correlation increasing (toward ±1)
        # or distance decreasing

        for i in range(n):
            row = {
                'I': int(I_values[i]),
                'signal_a': sig_a,
                'signal_b': sig_b,
                'engine': engine,

                'correlation': correlation[i],
                'correlation_velocity': corr_deriv['velocity'][i],

                'distance': distance[i],
                'distance_velocity': dist_deriv['velocity'][i],

                # Coupling dynamics - COMPUTED VALUES ONLY (no classification)
                # ORTHON interprets: WHERE coupling_velocity > 0.01 THEN 'STRENGTHENING'
                'coupling_velocity': corr_deriv['velocity'][i] * np.sign(correlation[i]) if np.abs(correlation[i]) > 1e-10 else 0.0,
            }
            # Include cohort if available
            if cohort:
                row['cohort'] = cohort
            if unit_id:
                row['unit_id'] = unit_id
            results.append(row)

    result = pl.DataFrame(results)
    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

    return result


# ============================================================
# FULL DYNAMICS PIPELINE
# ============================================================

def compute_all_dynamics(
    state_geometry_path: str,
    signal_geometry_path: str,
    signal_pairwise_path: str = None,
    output_dir: str = ".",
    dt: float = 1.0,
    verbose: bool = True
) -> Dict[str, pl.DataFrame]:
    """
    Compute all dynamics: geometry, signal, and pairwise.

    The complete differential geometry framework.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Geometry dynamics
    results['geometry'] = compute_geometry_dynamics(
        state_geometry_path,
        str(output_dir / "geometry_dynamics.parquet"),
        dt=dt,
        verbose=verbose
    )

    # Signal dynamics
    results['signal'] = compute_signal_dynamics(
        signal_geometry_path,
        str(output_dir / "signal_dynamics.parquet"),
        dt=dt,
        verbose=verbose
    )

    # Pairwise dynamics (optional)
    if signal_pairwise_path and Path(signal_pairwise_path).exists():
        results['pairwise'] = compute_pairwise_dynamics(
            signal_pairwise_path,
            str(output_dir / "pairwise_dynamics.parquet"),
            dt=dt,
            verbose=verbose
        )

    return results


# ============================================================
# CLI
# ============================================================

def main():
    import sys

    usage = """
Geometry Dynamics Engine - Full differential geometry framework

Usage:
    python geometry_dynamics.py geometry <state_geometry.parquet> [output.parquet]
    python geometry_dynamics.py signal <signal_geometry.parquet> [output.parquet]
    python geometry_dynamics.py pairwise <signal_pairwise.parquet> [output.parquet]
    python geometry_dynamics.py all <state_geometry.parquet> <signal_geometry.parquet> [signal_pairwise.parquet] [output_dir]

Computes (no classification - ENGINES computes, ORTHON interprets):
- Velocity (first derivative)
- Acceleration (second derivative)
- Jerk (third derivative)
- Curvature
- Collapse onset index/fraction
"""

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    mode = sys.argv[1]

    if mode == 'geometry':
        input_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "geometry_dynamics.parquet"
        compute_geometry_dynamics(input_path, output_path)

    elif mode == 'signal':
        input_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "signal_dynamics.parquet"
        compute_signal_dynamics(input_path, output_path)

    elif mode == 'pairwise':
        input_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "pairwise_dynamics.parquet"
        compute_pairwise_dynamics(input_path, output_path)

    elif mode == 'all':
        state_geom = sys.argv[2]
        signal_geom = sys.argv[3]
        pairwise = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].endswith('/') else None
        output_dir = sys.argv[-1] if sys.argv[-1].endswith('/') or len(sys.argv) > 5 else "."
        compute_all_dynamics(state_geom, signal_geom, pairwise, output_dir)

    else:
        print(f"Unknown mode: {mode}")
        print(usage)
        sys.exit(1)


if __name__ == "__main__":
    main()
