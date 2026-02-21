"""
Trajectory Feature Extraction for ML
=====================================

Extracts per-engine features from trajectory signatures across
multiple ordering axes (time, htBleed, BPR).

Architecture:
    trajectory_signatures.parquet (per window per engine)
    trajectory_match.parquet (per engine → trajectory assignment)
    → trajectory_features.parquet (one row per engine, all features)

Key insight: same curvature ≠ same state. Two engines at curvature=1.5
are in fundamentally different positions if one is at 20% arc length
and the other at 80%. Features capture both the geometry AND where
on the trajectory the engine sits.

Feature groups:
    Per axis (×3 axes = time, htBleed, BPR):
        summary:     mean, std, min, max, range of each metric
        shape:       early/late values + delta, linear slope
        position:    quartile values (q1/q2/q3 of trajectory)
        weighted:    position-weighted (late life emphasized)
        interaction: curvature × speed, condition spike ratio
        match:       trajectory_id, confidence, distance

    Cross-axis (15 features):
        collapse:    n_axes_collapsing, min/max/mean delta
        agreement:   condition number convergence, trajectory agreement
        divergence:  per-axis spread of collapse magnitude

Output: ~150 features per axis + 15 cross-axis ≈ 480 total.

Notes:
    - trajectory_position is always 1.0 on training data (run-to-failure).
      On test/live data where engines haven't failed yet, this becomes
      the key feature — where on the failure trajectory the engine sits.
    - torsion is currently all NaN (needs 3D embedding). Excluded.
    - 480 features for 100 engines needs feature selection or PCA
      before training. Cross-validated LASSO or tree-based importance
      recommended.
    - FD004 (249 engines, multiple fault modes) will improve ratio.

Usage:
    from trajectory_features import build_feature_matrix

    features = build_feature_matrix({
        'time': {'sigs': 'time/trajectory_signatures.parquet',
                 'match': 'time/trajectory_match.parquet'},
        'htBleed': {'sigs': 'htb/trajectory_signatures.parquet',
                    'match': 'htb/trajectory_match.parquet'},
        'BPR': {'sigs': 'bpr/trajectory_signatures.parquet',
                'match': 'bpr/trajectory_match.parquet'},
    })
    features.write_parquet('trajectory_features.parquet')
"""

import polars as pl
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# Metrics to extract features from (per window in trajectory_signatures)
# ─────────────────────────────────────────────────────────────────────
TRAJECTORY_METRICS = [
    'effective_dim',
    'eigenvalue_1',
    'eigenvalue_2',
    'eigenvalue_3',
    'total_variance',
    'condition_number',
    'speed',
    'curvature',
    'acceleration_magnitude',
    'effective_dim_velocity',
    'effective_dim_acceleration',
    'effective_dim_curvature',
    # torsion excluded: all NaN on 3-eigenvalue signatures
]


def _linear_slope(values: np.ndarray) -> float:
    """Fit linear slope over index. Returns NaN if < 3 valid points."""
    valid = ~np.isnan(values)
    if valid.sum() < 3:
        return np.nan
    x = np.arange(len(values))[valid]
    y = values[valid]
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def _extract_subwindow_features(
    geometry: pl.DataFrame,
    axis_name: str,
) -> pl.DataFrame:
    """
    Extract sub-window dispersion features from cohort_geometry.

    Cohort geometry has 2-3 sub-windows per (cohort, signal_0_end).
    Trajectory signatures pick only the first, masking collapse in
    other sub-windows. This computes per-window min/max/spread across
    sub-windows, then summarizes across the engine's life.

    Returns one row per engine with ~20 features prefixed {axis_name}_sw_.
    """
    p = f"{axis_name}_sw_"
    engines = sorted(geometry['cohort'].unique().to_list())
    rows = []

    for eng in engines:
        eng_geo = geometry.filter(pl.col('cohort') == eng).sort('signal_0_end')
        row = {'cohort': eng}

        if len(eng_geo) == 0:
            rows.append(row)
            continue

        # Per signal_0_end: compute sub-window stats
        windows = sorted(eng_geo['signal_0_end'].unique().to_list())
        n_win = len(windows)

        sw_eff_dim_min = []
        sw_eff_dim_max = []
        sw_eff_dim_spread = []
        sw_cond_max = []
        sw_cond_spread = []
        sw_n_degen = []  # sub-windows with condition > 100

        for w in windows:
            wdata = eng_geo.filter(pl.col('signal_0_end') == w)
            eds = wdata['effective_dim'].to_numpy()
            cns = wdata['condition_number'].to_numpy()

            sw_eff_dim_min.append(np.nanmin(eds))
            sw_eff_dim_max.append(np.nanmax(eds))
            sw_eff_dim_spread.append(np.nanmax(eds) - np.nanmin(eds))
            sw_cond_max.append(np.nanmax(cns))
            sw_cond_spread.append(np.nanmax(cns) - np.nanmin(cns))
            sw_n_degen.append(int(np.nansum(cns > 100)))

        sw_eff_dim_min = np.array(sw_eff_dim_min)
        sw_eff_dim_spread = np.array(sw_eff_dim_spread)
        sw_cond_max = np.array(sw_cond_max)

        # ── Summary: worst-case sub-window track ──
        row[f'{p}eff_dim_min_mean'] = float(np.nanmean(sw_eff_dim_min))
        row[f'{p}eff_dim_min_min'] = float(np.nanmin(sw_eff_dim_min))
        row[f'{p}cond_max_mean'] = float(np.nanmean(sw_cond_max))
        row[f'{p}cond_max_max'] = float(np.nanmax(sw_cond_max))

        # ── Sub-window disagreement ──
        row[f'{p}eff_dim_spread_mean'] = float(np.nanmean(sw_eff_dim_spread))
        row[f'{p}eff_dim_spread_max'] = float(np.nanmax(sw_eff_dim_spread))
        row[f'{p}cond_spread_mean'] = float(np.nanmean(sw_cond_spread))
        row[f'{p}cond_spread_max'] = float(np.nanmax(sw_cond_spread))

        # ── Degenerate sub-windows ──
        total_sub = len(eng_geo)
        n_degen_total = sum(sw_n_degen)
        row[f'{p}n_degenerate'] = n_degen_total
        row[f'{p}frac_degenerate'] = n_degen_total / total_sub if total_sub > 0 else 0.0

        # ── Shape: early vs late worst-case ──
        if n_win >= 4:
            third = n_win // 3
            early_min = sw_eff_dim_min[:third]
            late_min = sw_eff_dim_min[-third:]
            early_cond = sw_cond_max[:third]
            late_cond = sw_cond_max[-third:]

            if len(early_min) > 0 and len(late_min) > 0:
                row[f'{p}eff_dim_min_early'] = float(np.nanmean(early_min))
                row[f'{p}eff_dim_min_late'] = float(np.nanmean(late_min))
                row[f'{p}eff_dim_min_delta'] = float(np.nanmean(late_min) - np.nanmean(early_min))
                row[f'{p}cond_max_early'] = float(np.nanmean(early_cond))
                row[f'{p}cond_max_late'] = float(np.nanmean(late_cond))
                row[f'{p}cond_max_delta'] = float(np.nanmean(late_cond) - np.nanmean(early_cond))
            else:
                for sfx in ['eff_dim_min_early', 'eff_dim_min_late', 'eff_dim_min_delta',
                            'cond_max_early', 'cond_max_late', 'cond_max_delta']:
                    row[f'{p}{sfx}'] = np.nan

            row[f'{p}eff_dim_min_slope'] = _linear_slope(sw_eff_dim_min)
            row[f'{p}eff_dim_spread_slope'] = _linear_slope(sw_eff_dim_spread)
        else:
            for sfx in ['eff_dim_min_early', 'eff_dim_min_late', 'eff_dim_min_delta',
                        'cond_max_early', 'cond_max_late', 'cond_max_delta',
                        'eff_dim_min_slope', 'eff_dim_spread_slope']:
                row[f'{p}{sfx}'] = np.nan

        rows.append(row)

    return pl.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Per-axis feature extraction
# ─────────────────────────────────────────────────────────────────────

def extract_single_axis_features(
    sigs: pl.DataFrame,
    match: pl.DataFrame,
    axis_name: str,
    geometry: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Extract features for one ordering axis.

    Args:
        sigs: trajectory_signatures.parquet (multiple rows per engine)
        match: trajectory_match.parquet (one row per matched engine)
        axis_name: prefix for feature columns ('time', 'htBleed', 'BPR')
        geometry: optional cohort_geometry.parquet for sub-window dispersion
                  features. When provided, adds ~20 {axis}_sw_* features
                  capturing collapse hidden by multi-scale window averaging.

    Returns:
        DataFrame with one row per engine, columns prefixed with axis_name.
    """
    p = f"{axis_name}_"
    engines = sorted(sigs['cohort'].unique().to_list())

    rows = []
    for eng in engines:
        eng_data = sigs.filter(pl.col('cohort') == eng).sort('signal_0_end')
        n = len(eng_data)
        row = {'cohort': eng, f'{p}n_windows': n}

        # ── Arc length: total distance through geometry space ──
        arc = eng_data['arc_length'].to_numpy()
        row[f'{p}arc_length_total'] = arc[-1] if len(arc) > 0 else np.nan

        # Normalized position along trajectory [0, 1]
        norm_pos = arc / arc[-1] if arc[-1] > 0 else np.linspace(0, 1, n)

        # ── Per-metric features ──
        for metric in TRAJECTORY_METRICS:
            if metric not in eng_data.columns:
                continue
            vals = eng_data[metric].to_numpy().astype(float)
            valid = ~np.isnan(vals)

            # Group 1: Summary statistics
            if valid.sum() > 0:
                row[f'{p}{metric}_mean'] = np.nanmean(vals)
                row[f'{p}{metric}_std'] = np.nanstd(vals) if valid.sum() > 1 else 0.0
                row[f'{p}{metric}_min'] = np.nanmin(vals)
                row[f'{p}{metric}_max'] = np.nanmax(vals)
                row[f'{p}{metric}_range'] = np.nanmax(vals) - np.nanmin(vals)
            else:
                for sfx in ['mean', 'std', 'min', 'max', 'range']:
                    row[f'{p}{metric}_{sfx}'] = np.nan

            # Group 2: Trajectory shape — early vs late life
            if n >= 4 and valid.sum() >= 4:
                third = n // 3
                early_valid = vals[:third][~np.isnan(vals[:third])]
                late_valid = vals[-third:][~np.isnan(vals[-third:])]

                if len(early_valid) > 0 and len(late_valid) > 0:
                    row[f'{p}{metric}_early'] = np.mean(early_valid)
                    row[f'{p}{metric}_late'] = np.mean(late_valid)
                    row[f'{p}{metric}_delta'] = np.mean(late_valid) - np.mean(early_valid)
                else:
                    for sfx in ['early', 'late', 'delta']:
                        row[f'{p}{metric}_{sfx}'] = np.nan

                row[f'{p}{metric}_slope'] = _linear_slope(vals)
            else:
                for sfx in ['early', 'late', 'delta', 'slope']:
                    row[f'{p}{metric}_{sfx}'] = np.nan

            # Group 3: Position-aware — values at trajectory quartiles
            if valid.sum() >= 3:
                vp = norm_pos[valid]
                vv = vals[valid]
                for pct, label in [(0.25, 'q1'), (0.50, 'q2'), (0.75, 'q3')]:
                    row[f'{p}{metric}_{label}'] = float(np.interp(pct, vp, vv))
            else:
                for label in ['q1', 'q2', 'q3']:
                    row[f'{p}{metric}_{label}'] = np.nan

        # ── Group 4: Position-weighted features ──
        # Late-life values matter more for failure prediction.
        # Quadratic weighting: weight ∝ (normalized_position)²
        curv = eng_data['curvature'].to_numpy().astype(float)
        spd = eng_data['speed'].to_numpy().astype(float)
        edim = eng_data['effective_dim'].to_numpy().astype(float)
        cond = eng_data['condition_number'].to_numpy().astype(float)

        weights = norm_pos ** 2
        w_sum = np.nansum(weights[~np.isnan(curv)])

        if w_sum > 0:
            row[f'{p}curvature_pos_weighted'] = np.nansum(curv * weights) / w_sum
            row[f'{p}speed_pos_weighted'] = np.nansum(spd * weights) / w_sum
            row[f'{p}eff_dim_pos_weighted'] = np.nansum(edim * weights) / w_sum
            row[f'{p}cond_num_pos_weighted'] = np.nansum(cond * weights) / w_sum
        else:
            for m in ['curvature', 'speed', 'eff_dim', 'cond_num']:
                row[f'{p}{m}_pos_weighted'] = np.nan

        # ── Group 5: Interaction features ──
        # Curvature slope: is trajectory bending more sharply over time?
        if n >= 3 and (~np.isnan(curv)).sum() >= 3:
            row[f'{p}curvature_trend'] = _linear_slope(curv)
            # Curvature × speed product (late): approaching ridge fast?
            cs = curv * spd
            row[f'{p}curvature_speed_product_late'] = np.nanmean(cs[-max(1, n // 3):])
        else:
            row[f'{p}curvature_trend'] = np.nan
            row[f'{p}curvature_speed_product_late'] = np.nan

        # Condition number spike: max / early_mean
        if n >= 4:
            early_cond = cond[:n // 3]
            early_cond_valid = early_cond[~np.isnan(early_cond)]
            if len(early_cond_valid) > 0:
                row[f'{p}cond_spike_ratio'] = np.nanmax(cond) / np.mean(early_cond_valid)
            else:
                row[f'{p}cond_spike_ratio'] = np.nan
        else:
            row[f'{p}cond_spike_ratio'] = np.nan

        rows.append(row)

    features = pl.DataFrame(rows)

    # ── Group 6: Match metadata ──
    match_cols = match.select([
        'cohort',
        pl.col('trajectory_id').alias(f'{p}trajectory_id'),
        pl.col('match_distance').alias(f'{p}match_distance'),
        pl.col('match_confidence').alias(f'{p}match_confidence'),
        pl.col('trajectory_position').alias(f'{p}trajectory_position'),
    ])

    features = features.join(match_cols, on='cohort', how='left')

    # ── Sub-window dispersion features (optional) ──
    if geometry is not None:
        sw_feats = _extract_subwindow_features(geometry, axis_name)
        features = features.join(sw_feats, on='cohort', how='left')

    return features


# ─────────────────────────────────────────────────────────────────────
# Cross-axis feature extraction
# ─────────────────────────────────────────────────────────────────────

def extract_cross_axis_features(
    axis_features: dict,
    collapse_threshold: float = -0.15,
) -> pl.DataFrame:
    """
    Extract features comparing across ordering axes.

    Captures whether an engine collapses on multiple axes (robust signal)
    or only one (axis-specific degradation pathway).

    Args:
        axis_features: dict of axis_name -> single-axis feature DataFrame
        collapse_threshold: effective_dim delta below which counts as collapse
    """
    all_engines = set()
    for df in axis_features.values():
        all_engines |= set(df['cohort'].to_list())

    axis_names = list(axis_features.keys())
    rows = []

    for eng in sorted(all_engines):
        row = {'cohort': eng}

        deltas = {}
        cond_lates = {}
        dim_lates = {}
        traj_ids = {}

        for ax in axis_names:
            df = axis_features[ax]
            eng_row = df.filter(pl.col('cohort') == eng)
            if len(eng_row) == 0:
                continue

            # Collect deltas
            for col_name, target in [
                (f'{ax}_effective_dim_delta', deltas),
                (f'{ax}_condition_number_late', cond_lates),
                (f'{ax}_effective_dim_late', dim_lates),
            ]:
                if col_name in eng_row.columns:
                    val = eng_row[col_name][0]
                    if val is not None and not np.isnan(val):
                        target[ax] = val

            tid_col = f'{ax}_trajectory_id'
            if tid_col in eng_row.columns:
                val = eng_row[tid_col][0]
                if val is not None:
                    traj_ids[ax] = int(val)

        # ── Collapse agreement ──
        n_collapsing = sum(1 for d in deltas.values() if d < collapse_threshold)
        row['cross_n_axes_collapsing'] = n_collapsing
        row['cross_collapse_on_any'] = 1 if n_collapsing > 0 else 0
        row['cross_collapse_on_all'] = 1 if n_collapsing == len(axis_names) else 0

        # ── Delta statistics ──
        if deltas:
            dv = list(deltas.values())
            row['cross_min_dim_delta'] = min(dv)
            row['cross_max_dim_delta'] = max(dv)
            row['cross_mean_dim_delta'] = np.mean(dv)
            row['cross_dim_delta_spread'] = max(dv) - min(dv)
        else:
            for sfx in ['min_dim_delta', 'max_dim_delta', 'mean_dim_delta', 'dim_delta_spread']:
                row[f'cross_{sfx}'] = np.nan

        # ── Condition number agreement ──
        if len(cond_lates) >= 2:
            cv = list(cond_lates.values())
            row['cross_cond_late_mean'] = np.mean(cv)
            row['cross_cond_late_max'] = max(cv)
            row['cross_cond_late_std'] = np.std(cv)
        else:
            for sfx in ['cond_late_mean', 'cond_late_max', 'cond_late_std']:
                row[f'cross_{sfx}'] = np.nan

        # ── Effective dim late-life agreement ──
        if len(dim_lates) >= 2:
            dv = list(dim_lates.values())
            row['cross_dim_late_mean'] = np.mean(dv)
            row['cross_dim_late_min'] = min(dv)
            row['cross_dim_late_std'] = np.std(dv)
        else:
            for sfx in ['dim_late_mean', 'dim_late_min', 'dim_late_std']:
                row[f'cross_{sfx}'] = np.nan

        # ── Trajectory type agreement ──
        if len(traj_ids) >= 2:
            vals = list(traj_ids.values())
            row['cross_trajectory_agreement'] = 1 if len(set(vals)) == 1 else 0
            row['cross_n_distinct_trajectories'] = len(set(vals))
        else:
            row['cross_trajectory_agreement'] = np.nan
            row['cross_n_distinct_trajectories'] = np.nan

        rows.append(row)

    return pl.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Top-level builder
# ─────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    axis_configs: dict,
    collapse_threshold: float = -0.15,
    drop_zero_variance: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Build complete ML feature matrix from multi-axis trajectory data.

    Args:
        axis_configs: dict of axis_name -> {
            'sigs': path to trajectory_signatures.parquet,
            'match': path to trajectory_match.parquet,
            'geometry': (optional) path to cohort_geometry.parquet,
        }
        collapse_threshold: effective_dim delta for collapse detection
        drop_zero_variance: remove features with zero variance
        verbose: print progress

    Returns:
        Feature matrix: one row per engine, all features.
    """
    axis_features = {}

    for axis_name, paths in axis_configs.items():
        if verbose:
            print(f"Extracting {axis_name} features...")
        sigs = pl.read_parquet(paths['sigs'])
        match = pl.read_parquet(paths['match'])
        geometry = None
        if 'geometry' in paths:
            geometry = pl.read_parquet(paths['geometry'])
            if verbose:
                print(f"  + sub-window dispersion from cohort_geometry")
        axis_features[axis_name] = extract_single_axis_features(sigs, match, axis_name, geometry)
        if verbose:
            n_feats = len(axis_features[axis_name].columns) - 1
            print(f"  {n_feats} features for {len(axis_features[axis_name])} engines")

    if verbose:
        print("Extracting cross-axis features...")
    cross = extract_cross_axis_features(axis_features, collapse_threshold)

    # Join on cohort
    result = cross
    for axis_name, feats in axis_features.items():
        result = result.join(feats, on='cohort', how='full', suffix=f'_{axis_name}')

    if drop_zero_variance:
        before = len(result.columns)
        drop_cols = []
        for c in result.columns:
            if c == 'cohort':
                continue
            col = result[c]
            if col.dtype in [pl.Float64, pl.Float32]:
                valid = col.drop_nulls().drop_nans()
                if len(valid) > 0 and valid.std() == 0:
                    drop_cols.append(c)
                elif len(valid) == 0:
                    drop_cols.append(c)
        if drop_cols:
            result = result.drop(drop_cols)
            if verbose:
                print(f"Dropped {len(drop_cols)} zero-variance features: {drop_cols}")

    if verbose:
        n_feats = len(result.columns) - 1
        n_engines = len(result)
        print(f"\nFeature matrix: {n_engines} engines × {n_feats} features")
        print(f"Feature-to-sample ratio: {n_feats / n_engines:.1f}:1")
        if n_feats / n_engines > 5:
            print(f"  ⚠ High ratio — use feature selection (LASSO, tree importance)")

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract ML features from multi-axis trajectory signatures',
    )
    parser.add_argument('--time-sigs', required=True, help='time trajectory_signatures.parquet')
    parser.add_argument('--time-match', required=True, help='time trajectory_match.parquet')
    parser.add_argument('--time-geometry', help='time cohort_geometry.parquet (sub-window features)')
    parser.add_argument('--htbleed-sigs', help='htBleed trajectory_signatures.parquet')
    parser.add_argument('--htbleed-match', help='htBleed trajectory_match.parquet')
    parser.add_argument('--htbleed-geometry', help='htBleed cohort_geometry.parquet')
    parser.add_argument('--bpr-sigs', help='BPR trajectory_signatures.parquet')
    parser.add_argument('--bpr-match', help='BPR trajectory_match.parquet')
    parser.add_argument('--bpr-geometry', help='BPR cohort_geometry.parquet')
    parser.add_argument('-o', '--output', default='trajectory_features.parquet')
    parser.add_argument('--collapse-threshold', type=float, default=-0.15)
    parser.add_argument('-q', '--quiet', action='store_true')

    args = parser.parse_args()

    configs = {'time': {'sigs': args.time_sigs, 'match': args.time_match}}
    if args.time_geometry:
        configs['time']['geometry'] = args.time_geometry
    if args.htbleed_sigs and args.htbleed_match:
        configs['htBleed'] = {'sigs': args.htbleed_sigs, 'match': args.htbleed_match}
        if args.htbleed_geometry:
            configs['htBleed']['geometry'] = args.htbleed_geometry
    if args.bpr_sigs and args.bpr_match:
        configs['BPR'] = {'sigs': args.bpr_sigs, 'match': args.bpr_match}
        if args.bpr_geometry:
            configs['BPR']['geometry'] = args.bpr_geometry

    features = build_feature_matrix(
        configs,
        collapse_threshold=args.collapse_threshold,
        verbose=not args.quiet,
    )
    features.write_parquet(args.output)
    if not args.quiet:
        print(f"Saved: {args.output}")
