"""
PRISM Geometry Dynamics Engine

The complete differential geometry framework.
Computes derivatives, curvature, and trajectory classification
for the geometry evolution over time.

"You have position (state_vector).
 You have shape (eigenvalues).
 Now here are the derivatives."

Computes:
- First derivatives (velocity/tangent)
- Second derivatives (acceleration/curvature)
- Third derivatives (jerk/torsion)
- Trajectory classification
- Collapse detection
- Phase space analysis

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
from enum import Enum


# ============================================================
# TRAJECTORY CLASSIFICATION
# ============================================================

class TrajectoryType(Enum):
    STABLE = "stable"
    CONVERGING = "converging"
    DIVERGING = "diverging"
    OSCILLATING = "oscillating"
    CHAOTIC = "chaotic"
    TRANSIENT = "transient"
    COLLAPSING = "collapsing"
    EXPANDING = "expanding"


class StabilityClass(Enum):
    STABLE = "stable"
    MARGINALLY_STABLE = "marginally_stable"
    UNSTABLE = "unstable"
    CHAOTIC = "chaotic"


# ============================================================
# DERIVATIVE COMPUTATION
# ============================================================

def compute_derivatives(
    x: np.ndarray,
    dt: float = 1.0,
    smooth_window: int = 3
) -> Dict[str, np.ndarray]:
    """
    Compute derivatives up to third order with optional smoothing.

    Args:
        x: Time series values
        dt: Time step
        smooth_window: Smoothing window for noise reduction

    Returns:
        Dict with velocity, acceleration, jerk, curvature, speed
    """
    n = len(x)

    if n < 3:
        return {
            'velocity': np.zeros(n),
            'acceleration': np.zeros(n),
            'jerk': np.zeros(n),
            'curvature': np.zeros(n),
            'speed': np.zeros(n),
        }

    # Optional smoothing
    if smooth_window > 1 and n > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        x_smooth = np.convolve(x, kernel, mode='same')
    else:
        x_smooth = x

    # First derivative (central difference)
    dx = np.zeros(n)
    dx[1:-1] = (x_smooth[2:] - x_smooth[:-2]) / (2 * dt)
    dx[0] = (x_smooth[1] - x_smooth[0]) / dt if n > 1 else 0
    dx[-1] = (x_smooth[-1] - x_smooth[-2]) / dt if n > 1 else 0

    # Second derivative
    d2x = np.zeros(n)
    if n > 2:
        d2x[1:-1] = (x_smooth[2:] - 2*x_smooth[1:-1] + x_smooth[:-2]) / (dt**2)
        d2x[0] = d2x[1] if n > 2 else 0
        d2x[-1] = d2x[-2] if n > 2 else 0

    # Third derivative (jerk)
    d3x = np.zeros(n)
    if n > 3:
        d3x[2:-1] = (d2x[2:] - d2x[:-2])[:-1] / (2 * dt)

    # Curvature: κ = |d²x/dt²| / (1 + (dx/dt)²)^(3/2)
    curvature = np.zeros(n)
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
    embedding_dim: int = 2,
    tau: int = 1
) -> np.ndarray:
    """
    Reconstruct phase space using time-delay embedding.

    Takens' theorem: The attractor can be reconstructed from
    a single time series using delay coordinates.

    Args:
        x: Time series
        embedding_dim: Number of dimensions
        tau: Time delay

    Returns:
        Phase space coordinates (n_points × embedding_dim)
    """
    n = len(x)
    n_vectors = n - (embedding_dim - 1) * tau

    if n_vectors < 1:
        return np.array([])

    phase_space = np.zeros((n_vectors, embedding_dim))

    for i in range(embedding_dim):
        start = i * tau
        end = start + n_vectors
        phase_space[:, i] = x[start:end]

    return phase_space


# ============================================================
# TRAJECTORY CLASSIFICATION
# ============================================================

def classify_trajectory(
    velocity: np.ndarray,
    acceleration: np.ndarray,
    value: np.ndarray = None,
    thresholds: Dict[str, float] = None
) -> TrajectoryType:
    """
    Classify trajectory behavior based on derivatives.

    STABLE: Low velocity and acceleration
    CONVERGING: Moving toward equilibrium, decelerating
    DIVERGING: Moving away from equilibrium, accelerating
    OSCILLATING: Velocity changes sign periodically
    CHAOTIC: High variability in both velocity and acceleration
    COLLAPSING: Sustained negative velocity (for effective_dim)
    EXPANDING: Sustained positive velocity
    """
    thresholds = thresholds or {
        'stable_velocity': 0.01,
        'stable_acceleration': 0.01,
        'oscillation_fraction': 0.3,
        'chaos_cv': 2.0,
        'sustained_fraction': 0.5,
    }

    # Remove NaN
    valid = ~np.isnan(velocity) & ~np.isnan(acceleration)
    if valid.sum() < 3:
        return TrajectoryType.STABLE

    vel = velocity[valid]
    acc = acceleration[valid]

    mean_vel = np.mean(vel)
    std_vel = np.std(vel)
    mean_acc = np.mean(acc)
    std_acc = np.std(acc)

    # Count sign changes in velocity
    sign_changes = np.sum(np.diff(np.sign(vel)) != 0)
    oscillation_ratio = sign_changes / len(vel)

    # Coefficient of variation
    cv_vel = std_vel / (np.abs(mean_vel) + 1e-10)
    cv_acc = std_acc / (np.abs(mean_acc) + 1e-10)

    # Classification logic
    if std_vel < thresholds['stable_velocity'] and np.abs(mean_vel) < thresholds['stable_velocity']:
        return TrajectoryType.STABLE

    if oscillation_ratio > thresholds['oscillation_fraction']:
        return TrajectoryType.OSCILLATING

    if cv_vel > thresholds['chaos_cv'] and cv_acc > thresholds['chaos_cv']:
        return TrajectoryType.CHAOTIC

    # Sustained direction check
    neg_fraction = np.sum(vel < 0) / len(vel)
    pos_fraction = np.sum(vel > 0) / len(vel)

    if neg_fraction > thresholds['sustained_fraction']:
        if mean_acc < 0:  # Accelerating in negative direction
            return TrajectoryType.COLLAPSING
        else:
            return TrajectoryType.CONVERGING

    if pos_fraction > thresholds['sustained_fraction']:
        if mean_acc > 0:  # Accelerating in positive direction
            return TrajectoryType.EXPANDING
        else:
            return TrajectoryType.DIVERGING

    return TrajectoryType.TRANSIENT


def classify_stability(
    lyapunov_exponent: float = None,
    eigenvalue_ratio: float = None,
    velocity_variance: float = None
) -> StabilityClass:
    """
    Classify stability based on available metrics.

    Uses Lyapunov exponent if available, otherwise
    falls back to eigenvalue ratio or velocity variance.
    """
    if lyapunov_exponent is not None:
        if lyapunov_exponent < -0.01:
            return StabilityClass.STABLE
        elif lyapunov_exponent < 0.01:
            return StabilityClass.MARGINALLY_STABLE
        elif lyapunov_exponent < 0.1:
            return StabilityClass.UNSTABLE
        else:
            return StabilityClass.CHAOTIC

    if eigenvalue_ratio is not None:
        if eigenvalue_ratio < 0.3:
            return StabilityClass.STABLE
        elif eigenvalue_ratio < 0.6:
            return StabilityClass.MARGINALLY_STABLE
        else:
            return StabilityClass.UNSTABLE

    if velocity_variance is not None:
        if velocity_variance < 0.01:
            return StabilityClass.STABLE
        elif velocity_variance < 0.1:
            return StabilityClass.MARGINALLY_STABLE
        else:
            return StabilityClass.UNSTABLE

    return StabilityClass.MARGINALLY_STABLE


# ============================================================
# COLLAPSE DETECTION
# ============================================================

def detect_collapse(
    effective_dim: np.ndarray,
    threshold_velocity: float = -0.1,
    sustained_fraction: float = 0.3,
    min_collapse_length: int = 5
) -> Dict[str, Any]:
    """
    Detect dimensional collapse in effective_dim time series.

    Collapse = sustained negative velocity in effective_dim
    indicating the system is losing degrees of freedom.

    Args:
        effective_dim: Effective dimension over time
        threshold_velocity: Velocity below this = collapsing
        sustained_fraction: Fraction of points that must be collapsing
        min_collapse_length: Minimum consecutive points for collapse

    Returns:
        Collapse detection results
    """
    n = len(effective_dim)

    if n < min_collapse_length:
        return {
            'collapse_detected': False,
            'collapse_onset_idx': None,
            'collapse_onset_fraction': None,
            'collapse_velocity': 0.0,
            'collapse_acceleration': 0.0,
            'collapse_duration': 0,
            'total_collapse': 0.0,
        }

    # Compute derivatives
    deriv = compute_derivatives(effective_dim)
    velocity = deriv['velocity']
    acceleration = deriv['acceleration']

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
            'collapse_detected': False,
            'collapse_onset_idx': None,
            'collapse_onset_fraction': None,
            'collapse_velocity': 0.0,
            'collapse_acceleration': 0.0,
            'collapse_duration': 0,
            'total_collapse': 0.0,
        }

    # Take the longest collapse run
    longest_run = max(collapse_runs, key=lambda x: x[2])
    onset_idx, end_idx, duration = longest_run

    # Compute collapse metrics
    collapse_region = slice(onset_idx, end_idx)
    mean_velocity = np.mean(velocity[collapse_region])
    mean_acceleration = np.mean(acceleration[collapse_region])
    total_collapse = effective_dim[onset_idx] - effective_dim[min(end_idx, n-1)]

    return {
        'collapse_detected': True,
        'collapse_onset_idx': int(onset_idx),
        'collapse_onset_fraction': onset_idx / n,
        'collapse_velocity': float(mean_velocity),
        'collapse_acceleration': float(mean_acceleration),
        'collapse_duration': int(duration),
        'total_collapse': float(total_collapse),
        'collapse_runs': collapse_runs,
    }


# ============================================================
# GEOMETRY DYNAMICS COMPUTATION
# ============================================================

def compute_geometry_dynamics(
    state_geometry_path: str,
    output_path: str = "geometry_dynamics.parquet",
    dt: float = 1.0,
    smooth_window: int = 3,
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
        dt: Time step
        smooth_window: Smoothing window
        verbose: Print progress

    Returns:
        Geometry dynamics DataFrame
    """
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

    # Process each engine (unit_id is cargo, not a compute key)
    results = []

    groups = state_geometry.group_by(['engine'], maintain_order=True)

    for group_key, group in groups:
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

        # Classify trajectory
        trajectory = classify_trajectory(
            eff_dim_deriv['velocity'],
            eff_dim_deriv['acceleration'],
            effective_dim
        )

        # Detect collapse
        collapse = detect_collapse(effective_dim)

        # Classify stability
        velocity_var = np.var(eff_dim_deriv['velocity'])
        eigen_ratio = group['ratio_2_1'].mean() if 'ratio_2_1' in group.columns else None
        stability = classify_stability(
            eigenvalue_ratio=eigen_ratio,
            velocity_variance=velocity_var
        )

        # Build result rows
        for i in range(n):
            row = {
                'unit_id': unit_id,
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
                'eigenvalue_1_acceleration': eigen_1_deriv['acceleration'][i],

                # Variance dynamics
                'total_variance': total_variance[i],
                'variance_velocity': variance_deriv['velocity'][i],

                # Speed (magnitude of change)
                'speed': eff_dim_deriv['speed'][i],

                # Classification (constant for unit)
                'trajectory_type': trajectory.value,
                'stability_class': stability.value,

                # Collapse detection (constant for unit)
                'collapse_detected': collapse['collapse_detected'],
                'collapse_onset_idx': collapse['collapse_onset_idx'],
                'collapse_onset_fraction': collapse['collapse_onset_fraction'],
                'collapse_velocity': collapse['collapse_velocity'],
            }
            results.append(row)

    # Build DataFrame
    result = pl.DataFrame(results)
    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        # Summary
        print("\nTrajectory types:")
        for ttype in result['trajectory_type'].unique().to_list():
            count = (result['trajectory_type'] == ttype).sum()
            print(f"  {ttype}: {count}")

        print("\nCollapse detected:")
        n_collapse = result.filter(pl.col('collapse_detected'))['unit_id'].n_unique()
        n_total = result['unit_id'].n_unique()
        print(f"  {n_collapse} / {n_total} units")

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

    # Process each signal_id (unit_id is cargo, not a compute key)
    results = []

    signal_col = 'signal_id' if 'signal_id' in signal_geometry.columns else 'signal_id'
    groups = signal_geometry.group_by([signal_col], maintain_order=True)
    n_groups = signal_geometry.select([signal_col]).unique().height

    if verbose:
        print(f"Processing {n_groups} signal groups...")

    processed = 0
    for group_key, group in groups:
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

            # Classify trajectory
            trajectory = classify_trajectory(
                dist_deriv['velocity'],
                dist_deriv['acceleration'],
                distance
            )

            # Build result rows
            for i in range(n):
                row = {
                    'unit_id': unit_id,
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

                    # Classification
                    'trajectory_type': trajectory.value,

                    # Derived flags
                    'is_converging': dist_deriv['velocity'][i] < 0,
                    'is_aligning': coh_deriv['velocity'][i] > 0,
                }
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

        # Summary
        print("\nSignal trajectory types:")
        for ttype in result['trajectory_type'].unique().to_list():
            count = (result['trajectory_type'] == ttype).sum()
            print(f"  {ttype}: {count}")

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

    # Process each (signal_a, signal_b, engine) - unit_id is cargo
    results = []

    groups = pairwise.group_by(['signal_a', 'signal_b', 'engine'], maintain_order=True)

    for group_key, group in groups:
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
                'unit_id': unit_id,
                'I': int(I_values[i]),
                'signal_a': sig_a,
                'signal_b': sig_b,
                'engine': engine,

                'correlation': correlation[i],
                'correlation_velocity': corr_deriv['velocity'][i],

                'distance': distance[i],
                'distance_velocity': dist_deriv['velocity'][i],

                # Coupling dynamics
                'coupling_strengthening': (
                    (np.abs(correlation[i]) > 0.5 and corr_deriv['velocity'][i] * np.sign(correlation[i]) > 0) or
                    dist_deriv['velocity'][i] < 0
                ),
                'synchronizing': corr_deriv['velocity'][i] > 0.01,
                'desynchronizing': corr_deriv['velocity'][i] < -0.01,
            }
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

Computes:
- Velocity (first derivative)
- Acceleration (second derivative)
- Jerk (third derivative)
- Curvature
- Trajectory classification
- Collapse detection
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
