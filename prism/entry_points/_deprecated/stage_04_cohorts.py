"""
04: Cohorts Aggregation Entry Point
===================================

Pure orchestration - aggregates window-level metrics into cohort summaries.
No computation engines needed - this is pure aggregation.

Stages: state_vector.parquet + state_geometry.parquet → cohorts.parquet
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional
from scipy import stats


def compute_trend(values: np.ndarray) -> float:
    """
    Compute normalized linear trend slope.

    Returns slope as fraction of mean (change per window).
    """
    if len(values) < 3:
        return 0.0

    try:
        t = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(t, values)

        mean_val = np.mean(values)
        if abs(mean_val) > 1e-10:
            normalized_slope = slope / mean_val
        else:
            normalized_slope = slope

        return float(normalized_slope)
    except Exception:
        return 0.0


def compute_health_score(result: Dict) -> float:
    """
    Compute overall health score (0-1).

    Components:
    - Geometry: eff_dim/n_signals ratio (higher = healthier)
    - Trend: eff_dim_trend (negative = unhealthy)
    - Stability: CSD and collapse events (fewer = healthier)
    """
    scores = []
    weights = []

    # Geometry health (40% weight)
    if 'eff_dim_mean' in result and 'n_signals' in result:
        n_signals = result['n_signals']
        if n_signals > 0:
            eff_dim_ratio = result['eff_dim_mean'] / n_signals
            geometry_score = min(eff_dim_ratio / 0.8, 1.0)
            scores.append(geometry_score)
            weights.append(0.4)

    # Trend health (30% weight)
    if 'eff_dim_trend' in result:
        trend = result['eff_dim_trend']
        trend_score = 1.0 / (1.0 + np.exp(-10 * trend))  # Sigmoid
        scores.append(trend_score)
        weights.append(0.3)

    # Stability health (30% weight)
    n_windows = result.get('n_windows', 1)
    n_events = result.get('n_csd_detected', 0) + result.get('n_collapse_events', 0)
    event_rate = n_events / max(n_windows, 1)
    stability_score = 1.0 - min(event_rate, 1.0)
    scores.append(stability_score)
    weights.append(0.3)

    if scores:
        total_weight = sum(weights)
        health = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return float(np.clip(health, 0, 1))
    else:
        return 0.5


def compute_cohorts(
    state_vector_df: pl.DataFrame,
    state_geometry_df: pl.DataFrame,
    signal_vector_df: Optional[pl.DataFrame] = None,
    cohort_column: str = 'cohort',
    window_column: str = 'window_id',
) -> pl.DataFrame:
    """
    Aggregate window-level metrics into cohort summaries.

    Args:
        state_vector_df: State vector metrics per window
        state_geometry_df: State geometry metrics per window
        signal_vector_df: Optional signal vector metrics
        cohort_column: Column containing cohort identifier
        window_column: Column containing window index

    Returns:
        DataFrame with per-cohort summary metrics
    """
    results = []

    # Use 'I' if window_id not present
    if window_column not in state_vector_df.columns:
        window_column = 'I'

    # Get unique cohorts
    if cohort_column in state_vector_df.columns:
        cohorts = state_vector_df[cohort_column].unique().to_list()
    else:
        cohorts = ['_all_']

    for cohort in cohorts:
        # Filter data for this cohort
        if cohort_column in state_vector_df.columns:
            sv = state_vector_df.filter(pl.col(cohort_column) == cohort)
            sg = state_geometry_df.filter(pl.col(cohort_column) == cohort) if cohort_column in state_geometry_df.columns else state_geometry_df
        else:
            sv = state_vector_df
            sg = state_geometry_df

        n_windows = len(sv)
        if n_windows == 0:
            continue

        result = {
            'cohort': cohort,
            'n_windows': n_windows,
        }

        # Get n_signals from first window
        if 'n_signals' in sv.columns:
            result['n_signals'] = int(sv['n_signals'].head(1).item())

        # === GEOMETRY SUMMARY ===
        if len(sg) > 0 and 'effective_dim' in sg.columns:
            eff_dims = sg['effective_dim'].to_numpy()
            result['eff_dim_mean'] = float(np.mean(eff_dims))
            result['eff_dim_std'] = float(np.std(eff_dims))
            result['eff_dim_min'] = float(np.min(eff_dims))
            result['eff_dim_max'] = float(np.max(eff_dims))
            result['eff_dim_trend'] = compute_trend(eff_dims)

        # === MASS SUMMARY ===
        if 'total_variance' in sv.columns:
            variances = sv['total_variance'].to_numpy()
            result['variance_mean'] = float(np.mean(variances))
            result['variance_std'] = float(np.std(variances))
            result['variance_trend'] = compute_trend(variances)

        # === COUPLING SUMMARY ===
        if 'mean_abs_correlation' in sv.columns:
            correlations = sv['mean_abs_correlation'].to_numpy()
            result['correlation_mean'] = float(np.mean(correlations))
            result['coupling_trend'] = compute_trend(correlations)

        # === STABILITY SUMMARY ===
        if 'mean_autocorr_1' in sv.columns:
            autocorrs = sv['mean_autocorr_1'].drop_nulls().to_numpy()
            if len(autocorrs) > 0:
                n_csd = int(np.sum(autocorrs > 0.9))
                result['n_csd_detected'] = n_csd
                result['max_autocorr_1'] = float(np.max(autocorrs))

        # === HEALTH SCORE ===
        result['health_score'] = compute_health_score(result)

        results.append(result)

    return pl.DataFrame(results) if results else pl.DataFrame()


def interpret_cohort_health(cohort_df: pl.DataFrame) -> List[Dict]:
    """
    Generate interpretations for each cohort.

    Returns:
        List of interpretations with risk_level, description, recommendations
    """
    interpretations = []

    for row in cohort_df.to_dicts():
        cohort = row['cohort']
        health_score = row.get('health_score', 0.5)

        # Risk level
        if health_score >= 0.8:
            risk_level = 'low'
        elif health_score >= 0.6:
            risk_level = 'moderate'
        elif health_score >= 0.4:
            risk_level = 'high'
        else:
            risk_level = 'critical'

        # Build description
        desc_parts = []

        eff_dim_ratio = row.get('eff_dim_mean', 0) / row.get('n_signals', 1) if row.get('n_signals', 0) > 0 else 0
        if eff_dim_ratio > 0.8:
            desc_parts.append("Healthy geometry")
        elif eff_dim_ratio > 0.5:
            desc_parts.append("Moderate coupling")
        else:
            desc_parts.append("HIGH COUPLING")

        eff_dim_trend = row.get('eff_dim_trend', 0)
        if eff_dim_trend < -0.05:
            desc_parts.append("COLLAPSING")
        elif eff_dim_trend > 0.05:
            desc_parts.append("expanding")

        description = "; ".join(desc_parts)

        # Recommendations
        recommendations = []
        if eff_dim_ratio < 0.5:
            recommendations.append("Investigate signal coupling")
        if eff_dim_trend < -0.05:
            recommendations.append("URGENT: Dimensional collapse")
        if not recommendations:
            recommendations.append("Continue monitoring")

        interpretations.append({
            'cohort': cohort,
            'risk_level': risk_level,
            'health_score': health_score,
            'description': description,
            'recommendations': recommendations,
        })

    return interpretations


def main():
    import argparse

    parser = argparse.ArgumentParser(description="04: Cohorts Aggregation")
    parser.add_argument('state_vector', help='State vector parquet')
    parser.add_argument('state_geometry', help='State geometry parquet')
    parser.add_argument('output', help='Output cohorts parquet')

    args = parser.parse_args()

    print("=" * 70)
    print("04: COHORTS AGGREGATION")
    print("=" * 70)

    sv = pl.read_parquet(args.state_vector)
    sg = pl.read_parquet(args.state_geometry)

    print(f"State vector: {len(sv)} windows")
    print(f"State geometry: {len(sg)} windows")

    cohorts = compute_cohorts(sv, sg)
    print(f"Cohorts: {len(cohorts)}")

    cohorts.write_parquet(args.output)
    print(f"Saved: {args.output}")

    # Interpretations
    if len(cohorts) > 0:
        print("\n" + "=" * 70)
        print("COHORT HEALTH SUMMARY")
        print("=" * 70)

        interps = interpret_cohort_health(cohorts)
        for interp in interps:
            print(f"\n{interp['cohort']}:")
            print(f"  Risk: {interp['risk_level'].upper()} (score: {interp['health_score']:.2f})")
            print(f"  {interp['description']}")


if __name__ == "__main__":
    main()
