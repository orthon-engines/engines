"""
Stage 15: FTLE Field Entry Point
================================

Local Finite-Time Lyapunov Exponent fields around centroids and between
adjacent pairs. This is the astrodynamics layer - centroids are planets,
the eigendecomp-reduced manifold is orbital space.

Computes:
    - FTLE ridges between adjacent centroids (Lagrangian Coherent Structures)
    - Basin depth around each centroid (attractor strength)
    - Corridor width between basins (transition ease)
    - Ridge strength (barrier height)

The same math that revealed the Interplanetary Transport Network.
Different planets: your bearings, pumps, turbines.

Dependencies:
    - state_vector.parquet     (centroids = planets)
    - state_geometry.parquet   (eigendecomp = reduced manifold)
    - cohorts.parquet          (basin assignments)
    - ftle.parquet             (single-trajectory FTLE for seeding)

Outputs:
    - ftle_field.parquet       (ridges, basins, corridors)

This stage is OPTIONAL / on-demand. Not part of default pipeline.
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from itertools import combinations

from manifold.io.writer import write_output

try:
    from scipy.spatial import Delaunay
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def find_adjacent_pairs(
    centroid_positions: np.ndarray,
    method: str = 'delaunay',
    k_nearest: int = 3,
) -> List[Tuple[int, int]]:
    """
    Find adjacent centroid pairs for FTLE field computation.
    
    Args:
        centroid_positions: (n_centroids, n_dims) reduced-space positions
        method: 'delaunay' or 'knearest'
        k_nearest: Number of nearest neighbors (if method='knearest')
    
    Returns:
        List of (i, j) index pairs
    """
    n = len(centroid_positions)
    
    if n < 2:
        return []
    
    if n == 2:
        return [(0, 1)]
    
    if method == 'delaunay' and HAS_SCIPY and n >= 3:
        try:
            # Delaunay triangulation finds natural neighbors
            tri = Delaunay(centroid_positions[:, :min(2, centroid_positions.shape[1])])
            pairs = set()
            for simplex in tri.simplices:
                for i, j in combinations(simplex, 2):
                    pairs.add((min(i, j), max(i, j)))
            return sorted(pairs)
        except Exception:
            pass
    
    # Fallback to k-nearest
    pairs = set()
    for i in range(n):
        dists = np.linalg.norm(centroid_positions - centroid_positions[i], axis=1)
        nearest = np.argsort(dists)[1:k_nearest + 1]  # Exclude self
        for j in nearest:
            pairs.add((min(i, j), max(i, j)))
    
    return sorted(pairs)


def compute_local_grid(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    grid_resolution: int = 20,
    neighborhood: float = 2.0,
) -> np.ndarray:
    """
    Create a local grid between two centroids.
    
    Args:
        pos_a: Position of centroid A
        pos_b: Position of centroid B
        grid_resolution: Points per axis
        neighborhood: Grid extent as multiple of inter-centroid distance
    
    Returns:
        Grid points (n_points, n_dims)
    """
    ndim = len(pos_a)
    midpoint = (pos_a + pos_b) / 2
    axis = pos_b - pos_a
    axis_len = np.linalg.norm(axis)
    
    if axis_len < 1e-10:
        return midpoint.reshape(1, -1)
    
    axis_unit = axis / axis_len
    
    # Grid along the connecting axis
    t_range = np.linspace(-0.5 * neighborhood, 0.5 * neighborhood, grid_resolution)
    
    # For 2D+, add perpendicular sampling
    if ndim >= 2:
        # Create perpendicular vector
        perp = np.zeros(ndim)
        if abs(axis_unit[0]) < 0.9:
            perp[0] = 1
        else:
            perp[1] = 1
        perp = perp - np.dot(perp, axis_unit) * axis_unit
        perp = perp / np.linalg.norm(perp)
        
        s_range = np.linspace(-0.3, 0.3, max(3, grid_resolution // 4))
        
        grid_points = []
        for t in t_range:
            for s in s_range:
                point = midpoint + t * axis_len * axis_unit + s * axis_len * perp
                grid_points.append(point)
        
        return np.array(grid_points)
    
    # 1D case
    return (midpoint + t_range.reshape(-1, 1) * axis_len * axis_unit)


def ftle_at_point(
    point: np.ndarray,
    trajectories: np.ndarray,
    trajectory_ids: np.ndarray,
    dt: int = 1,
    k: int = 10,
) -> float:
    """
    Estimate FTLE at a point from nearby trajectory behavior.
    
    Uses Cauchy-Green strain tensor formulation.
    
    Args:
        point: Location in reduced space
        trajectories: All trajectory points (n_points, n_dims + time_idx)
        trajectory_ids: Which trajectory each point belongs to
        dt: Time offset for measuring divergence
        k: Number of nearest neighbors to use
    
    Returns:
        FTLE estimate at this location
    """
    ndim = len(point)
    
    # Find k nearest trajectory points
    dists = np.linalg.norm(trajectories[:, :ndim] - point, axis=1)
    nearest_idx = np.argsort(dists)[:k]
    
    if len(nearest_idx) < 3:
        return 0.0
    
    # For each nearest point, find where it went after dt steps
    initial_cloud = []
    final_cloud = []
    
    for idx in nearest_idx:
        traj_id = trajectory_ids[idx]
        same_traj = trajectory_ids == traj_id
        traj_points = trajectories[same_traj]
        
        # Find current point's position in trajectory
        local_idx = np.where(same_traj)[0] == idx
        if not np.any(local_idx):
            continue
        local_pos = np.where(local_idx)[0][0]
        
        # Check if dt steps later exists
        if local_pos + dt < len(traj_points):
            initial_cloud.append(traj_points[local_pos, :ndim])
            final_cloud.append(traj_points[local_pos + dt, :ndim])
    
    if len(initial_cloud) < 3:
        return 0.0
    
    initial_cloud = np.array(initial_cloud)
    final_cloud = np.array(final_cloud)
    
    # Cauchy-Green strain tensor: C = F^T @ F
    # where F is the deformation gradient
    dx_initial = initial_cloud - initial_cloud.mean(axis=0)
    dx_final = final_cloud - final_cloud.mean(axis=0)
    
    if np.linalg.norm(dx_initial) < 1e-12:
        return 0.0
    
    try:
        # Deformation gradient via least squares
        F, _, _, _ = np.linalg.lstsq(dx_initial, dx_final, rcond=None)
        C = F.T @ F
        
        # FTLE = (1/2T) * ln(max eigenvalue of C)
        eigvals = np.linalg.eigvalsh(C)
        max_eigval = max(eigvals.max(), 1e-12)
        
        return np.log(np.sqrt(max_eigval)) / dt
    except Exception:
        return 0.0


def compute_ridge_metrics(
    ftle_field: np.ndarray,
    grid_points: np.ndarray,
    axis_unit: np.ndarray,
) -> Dict[str, float]:
    """
    Compute ridge metrics from FTLE field.
    
    Args:
        ftle_field: FTLE values at grid points
        grid_points: Grid point locations
        axis_unit: Unit vector from centroid A to B
    
    Returns:
        Ridge metrics
    """
    if len(ftle_field) == 0 or np.all(np.isnan(ftle_field)):
        return {
            'ridge_strength': np.nan,
            'ridge_width': np.nan,
            'corridor_width': np.nan,
        }
    
    # Project grid points onto axis
    projections = np.dot(grid_points, axis_unit)
    
    # Find ridge = maximum FTLE
    valid = ~np.isnan(ftle_field)
    if not np.any(valid):
        return {
            'ridge_strength': np.nan,
            'ridge_width': np.nan,
            'corridor_width': np.nan,
        }
    
    ridge_idx = np.argmax(ftle_field[valid])
    ridge_strength = float(ftle_field[valid][ridge_idx])
    
    # Ridge width: FWHM around peak
    half_max = ridge_strength / 2
    above_half = ftle_field >= half_max
    if np.sum(above_half) > 1:
        proj_above = projections[above_half]
        ridge_width = float(proj_above.max() - proj_above.min())
    else:
        ridge_width = 0.0
    
    # Corridor width: region below threshold where transit is "easy"
    threshold = ridge_strength * 0.3
    below_thresh = ftle_field < threshold
    if np.sum(below_thresh) > 0:
        corridor_width = float(np.sum(below_thresh) / len(ftle_field))
    else:
        corridor_width = 0.0
    
    return {
        'ridge_strength': ridge_strength,
        'ridge_width': ridge_width,
        'corridor_width': corridor_width,
    }


def compute_basin_depth(
    centroid_pos: np.ndarray,
    trajectories: np.ndarray,
    trajectory_ids: np.ndarray,
    radius: float = None,
    n_radial: int = 8,
) -> Dict[str, float]:
    """
    Compute basin depth around a centroid.
    
    Basin depth = mean FTLE in neighborhood (attractor strength).
    
    Args:
        centroid_pos: Centroid position in reduced space
        trajectories: All trajectory points
        trajectory_ids: Trajectory IDs
        radius: Neighborhood radius (auto-compute if None)
        n_radial: Number of radial sample points
    
    Returns:
        Basin metrics
    """
    ndim = len(centroid_pos)
    
    # Auto-compute radius from nearby trajectory spread
    if radius is None:
        dists = np.linalg.norm(trajectories[:, :ndim] - centroid_pos, axis=1)
        radius = np.percentile(dists, 25) if len(dists) > 0 else 1.0
    
    # Sample points around centroid
    ftle_values = []
    
    for angle in np.linspace(0, 2 * np.pi, n_radial, endpoint=False):
        for r in [radius * 0.5, radius, radius * 1.5]:
            if ndim >= 2:
                offset = r * np.array([np.cos(angle), np.sin(angle)] + [0] * (ndim - 2))
            else:
                offset = np.array([r])
            
            point = centroid_pos + offset
            ftle = ftle_at_point(point, trajectories, trajectory_ids)
            ftle_values.append(ftle)
    
    ftle_values = np.array(ftle_values)
    valid = ftle_values != 0
    
    if np.sum(valid) == 0:
        return {
            'basin_depth': np.nan,
            'basin_radius': radius,
        }
    
    return {
        'basin_depth': float(np.mean(ftle_values[valid])),
        'basin_radius': float(radius),
    }


def run(
    state_vector_path: str,
    state_geometry_path: str,
    data_path: str = ".",
    grid_resolution: int = 20,
    neighborhood: float = 2.0,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute local FTLE fields around centroids.

    Args:
        state_vector_path: Path to state_vector.parquet
        state_geometry_path: Path to state_geometry.parquet
        data_path: Root data directory for output
        grid_resolution: Grid points per axis
        neighborhood: Grid extent multiplier
        verbose: Print progress

    Returns:
        FTLE field DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 15: FTLE FIELD")
        print("Local fields around centroids - Lagrangian Coherent Structures")
        print("=" * 70)
    
    # Load inputs
    state_vector = pl.read_parquet(state_vector_path)
    state_geometry = pl.read_parquet(state_geometry_path)
    
    if verbose:
        print(f"State vector: {state_vector.shape}")
        print(f"State geometry: {state_geometry.shape}")
    
    # Get unique cohorts/engines
    engines = state_geometry['engine'].unique().to_list() if 'engine' in state_geometry.columns else ['default']
    
    ridge_results = []
    basin_results = []
    
    for engine in engines:
        if verbose:
            print(f"\nProcessing engine: {engine}")
        
        # Filter to this engine
        if 'engine' in state_geometry.columns:
            geo = state_geometry.filter(pl.col('engine') == engine)
            sv = state_vector.filter(pl.col('engine') == engine) if 'engine' in state_vector.columns else state_vector
        else:
            geo = state_geometry
            sv = state_vector
        
        # Get eigenvalue columns for dimensionality reduction
        # Use first 2-3 principal components
        eigen_cols = [c for c in geo.columns if c.startswith('eigenvalue_')][:3]
        
        if not eigen_cols:
            if verbose:
                print(f"  No eigenvalue columns found, skipping")
            continue
        
        # Get centroid positions in reduced space
        # Group by cohort if available, otherwise use I
        group_cols = ['cohort'] if 'cohort' in sv.columns else []
        
        if len(group_cols) == 0:
            if verbose:
                print(f"  No grouping column, using all points as trajectory")
            continue
        
        # For now, treat each unique cohort as a "centroid"
        cohorts = sv['cohort'].unique().to_list() if 'cohort' in sv.columns else []

        if len(cohorts) < 2:
            if verbose:
                print(f"  Need at least 2 cohorts for ridge detection, got {len(cohorts)}")
            continue

        # Use eigenvalue columns from state_geometry as position coordinates
        # These are the meaningful reduced-space positions, not arbitrary state_vector numerics
        numeric_cols = [c for c in eigen_cols if c in geo.columns]

        if not numeric_cols:
            if verbose:
                print(f"  No eigenvalue columns for centroid positions")
            continue

        # Check sufficient data: need at least 10 cohorts with valid eigenvalue data
        valid_cohort_count = 0
        for cohort in cohorts:
            cohort_geo = geo.filter(pl.col('cohort') == cohort) if 'cohort' in geo.columns else geo
            if len(cohort_geo) > 0:
                vals = cohort_geo.select(numeric_cols).to_numpy()
                if np.isfinite(vals).all():
                    valid_cohort_count += 1

        if valid_cohort_count < 10:
            if verbose:
                print(f"  Skipping engine {engine}: only {valid_cohort_count} cohorts with valid eigenvalue data (need 10)")
            continue
        
        centroid_positions = []
        for cohort in cohorts:
            cohort_geo = geo.filter(pl.col('cohort') == cohort) if 'cohort' in geo.columns else geo
            if len(cohort_geo) == 0:
                centroid_positions.append([np.nan] * len(numeric_cols))
                continue
            pos = [cohort_geo[c].mean() for c in numeric_cols]
            centroid_positions.append(pos)
        
        centroid_positions = np.array(centroid_positions)

        # Filter out cohorts with NaN positions
        valid_mask = np.isfinite(centroid_positions).all(axis=1)
        valid_indices = np.where(valid_mask)[0]
        centroid_positions = centroid_positions[valid_mask]
        cohorts = [cohorts[i] for i in valid_indices]

        if len(cohorts) < 2:
            if verbose:
                print(f"  Need at least 2 valid centroids, got {len(cohorts)}")
            continue

        if verbose:
            print(f"  Found {len(cohorts)} centroids in {len(numeric_cols)}D eigenvalue space")

        # Find adjacent pairs
        pairs = find_adjacent_pairs(centroid_positions)

        if verbose:
            print(f"  Adjacent pairs: {len(pairs)}")

        # Build trajectory array from state_geometry eigenvalue space
        # Each row is a point in the eigenvalue-reduced trajectory
        trajectories = geo.select(numeric_cols).to_numpy()
        trajectory_ids = geo['cohort'].to_numpy() if 'cohort' in geo.columns else np.zeros(len(geo))
        
        # Compute ridge between each pair
        for i, j in pairs:
            pos_a = centroid_positions[i]
            pos_b = centroid_positions[j]
            
            # Create local grid
            grid = compute_local_grid(pos_a, pos_b, grid_resolution, neighborhood)
            
            # Compute FTLE at each grid point
            ftle_field = np.array([
                ftle_at_point(pt, trajectories, trajectory_ids)
                for pt in grid
            ])
            
            # Compute ridge metrics
            axis = pos_b - pos_a
            axis_len = np.linalg.norm(axis)
            axis_unit = axis / axis_len if axis_len > 0 else axis
            
            metrics = compute_ridge_metrics(ftle_field, grid, axis_unit)
            
            # Skip if FTLE field is all zeros (insufficient neighbors)
            valid = ~np.isnan(ftle_field) & (ftle_field != 0)
            if not np.any(valid):
                if verbose:
                    print(f"    Pair ({cohorts[i]}, {cohorts[j]}): all-zero FTLE, skipping")
                continue

            ridge_idx = np.argmax(ftle_field[valid])
            ridge_location = grid[valid][ridge_idx].tolist()

            ridge_results.append({
                'engine': engine,
                'centroid_a': cohorts[i],
                'centroid_b': cohorts[j],
                'ridge_location': str(ridge_location),
                'ridge_strength': metrics['ridge_strength'],
                'ridge_width': metrics['ridge_width'],
                'corridor_width': metrics['corridor_width'],
                'inter_centroid_distance': float(axis_len),
            })
        
        # Compute basin depth for each centroid
        for idx, cohort in enumerate(cohorts):
            basin = compute_basin_depth(
                centroid_positions[idx],
                trajectories,
                trajectory_ids,
            )
            
            # Find escape direction (toward weakest ridge)
            relevant_ridges = [r for r in ridge_results 
                             if r['centroid_a'] == cohort or r['centroid_b'] == cohort]
            if relevant_ridges:
                weakest = min(relevant_ridges, key=lambda x: x['ridge_strength'] or float('inf'))
                escape_dir = weakest['centroid_b'] if weakest['centroid_a'] == cohort else weakest['centroid_a']
            else:
                escape_dir = None
            
            basin_results.append({
                'engine': engine,
                'centroid_id': cohort,
                'basin_depth': basin['basin_depth'],
                'basin_radius': basin['basin_radius'],
                'escape_direction': escape_dir,
                'n_corridors': len(relevant_ridges),
            })
    
    # Combine results
    if ridge_results:
        ridge_df = pl.DataFrame(ridge_results)
    else:
        ridge_df = pl.DataFrame()
    
    if basin_results:
        basin_df = pl.DataFrame(basin_results)
    else:
        basin_df = pl.DataFrame()
    
    # For now, output the ridge data (main output)
    result = ridge_df if len(ridge_df) > 0 else basin_df
    
    if len(result) > 0:
        write_output(result, data_path, 'ftle_field', verbose=verbose)

    if verbose:
        print(f"Ridges: {len(ridge_df)}")
        print(f"Basins: {len(basin_df)}")

        if len(ridge_df) > 0 and 'ridge_strength' in ridge_df.columns:
            valid = ridge_df.filter(pl.col('ridge_strength').is_not_null())
            if len(valid) > 0:
                print(f"\nRidge strength stats:")
                print(f"  Mean: {valid['ridge_strength'].mean():.4f}")
                print(f"  Max:  {valid['ridge_strength'].max():.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 15: FTLE Field",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes local FTLE fields around centroids and between adjacent pairs.

This reveals Lagrangian Coherent Structures (LCS):
  - Ridges = regime boundaries (barriers between operating states)
  - Basin depth = attractor strength (how sticky is this state)
  - Corridor width = transition ease (how easy to move between states)

The same math that revealed the Interplanetary Transport Network
in astrodynamics. Different planets: your bearings, pumps, turbines.

Example:
  python -m engines.entry_points.stage_15_ftle_field \\
      state_vector.parquet state_geometry.parquet -o ftle_field.parquet
"""
    )
    parser.add_argument('state_vector', help='Path to state_vector.parquet')
    parser.add_argument('state_geometry', help='Path to state_geometry.parquet')
    parser.add_argument('-d', '--data-path', default='.',
                        help='Root data directory (default: .)')
    parser.add_argument('--grid-resolution', type=int, default=20,
                        help='Grid points per axis (default: 20)')
    parser.add_argument('--neighborhood', type=float, default=2.0,
                        help='Grid extent multiplier (default: 2.0)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.state_vector,
        args.state_geometry,
        args.data_path,
        grid_resolution=args.grid_resolution,
        neighborhood=args.neighborhood,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
