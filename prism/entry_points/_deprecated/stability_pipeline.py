"""
PRISM Stability Analysis Pipeline

Integrates:
    - signal_geometry.py (cross-signal eigenstructure - B-tipping)
    - lyapunov.py (stability measurement)
    - critical_slowing_down.py (early warning signals)
    - formal_definitions.py (terminology and classification)

This pipeline computes the full dynamical systems assessment for a dataset.

Usage:
    python -m prism.entry_points.stability_pipeline observations.parquet -o stability_report/

Outputs:
    - signal_geometry.parquet (cross-signal eigenstructure)
    - lyapunov_analysis.parquet (per-signal Lyapunov exponents)
    - csd_analysis.parquet (critical slowing down indicators)
    - formal_assessment.json (classification and report)
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

# Import our engines
from prism.engines.dynamics.lyapunov import compute as compute_lyapunov
from prism.engines.dynamics.critical_slowing_down import compute as compute_csd
from prism.engines.state.eigendecomp import compute as compute_eigenvalues
from prism.engines.dynamics.formal_definitions import (
    GeometryMetrics, MassMetrics, EarlyWarningSignals,
    FormalAssessment, AttractorType, StabilityType,
    FailureMode, TippingType, SystemTypology,
    classify_failure_mode, classify_tipping_type, classify_stability,
)


def compute_cross_signal_geometry(
    obs: pl.DataFrame,
    window_size: int = 36,
    step_size: int = 1,
    signal_column: str = 'signal_id',
    date_column: str = 'I',
    value_column: str = 'value',
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute cross-signal geometry (eigenstructure) over rolling windows.

    This is simplified signal geometry for the stability pipeline.
    For full signal geometry, use prism.engines.signal_geometry.

    Args:
        obs: Observations DataFrame
        window_size: Window size for geometry computation
        step_size: Step between windows
        signal_column: Column with signal names
        date_column: Column with time index
        value_column: Column with values

    Returns:
        DataFrame with geometry metrics per window
    """
    # Pivot to wide format
    pivot = obs.pivot(
        on=signal_column,
        index=date_column,
        values=value_column,
    ).sort(date_column)

    signals = [c for c in pivot.columns if c != date_column]
    n_signals = len(signals)
    n_rows = len(pivot)

    if verbose:
        print(f"  Signals: {n_signals}, Time points: {n_rows}")

    results = []

    for i in range(0, n_rows - window_size + 1, step_size):
        window_start = pivot[date_column][i]
        window_end = pivot[date_column][i + window_size - 1]

        # Extract window
        window = pivot[i:i + window_size].select(signals).to_numpy()

        # Remove NaN columns
        valid_cols = ~np.any(np.isnan(window), axis=0)
        if valid_cols.sum() < 2:
            continue

        window = window[:, valid_cols]

        # Use eigendecomp engine (handles centering, SVD, all derived metrics)
        eigen_result = compute_eigenvalues(window, norm_method="none")

        # Skip if insufficient data
        if eigen_result['n_signals'] < 2:
            continue

        eff_dim = eigen_result['effective_dim']
        total_var = eigen_result['total_variance']
        explained_ratio = eigen_result['explained_ratio']
        explained_1 = explained_ratio[0] if len(explained_ratio) > 0 else 0
        condition_number = eigen_result['condition_number']

        # Alignment = dominance of PC1
        alignment = explained_1

        # Correlation statistics (cross-signal specific, not in engine)
        corr_matrix = np.corrcoef(window.T)
        np.fill_diagonal(corr_matrix, 0)
        mean_abs_corr = np.nanmean(np.abs(corr_matrix))
        max_abs_corr = np.nanmax(np.abs(corr_matrix))

        # Coupling fraction (|corr| > 0.5)
        coupling_fraction = np.mean(np.abs(corr_matrix) > 0.5)

        results.append({
            'window_start': window_start,
            'window_end': window_end,
            'eff_dim': float(eff_dim),
            'alignment': float(alignment),
            'total_variance': float(total_var),
            'mean_abs_correlation': float(mean_abs_corr),
            'max_abs_correlation': float(max_abs_corr),
            'coupling_fraction': float(coupling_fraction),
            'condition_number': float(condition_number),
            'n_signals': int(valid_cols.sum()),
        })

    return pl.DataFrame(results)


def run_stability_pipeline(
    observations_path: str,
    output_dir: str = "stability_output",
    window_size: int = 36,
    step_size: int = 1,
    system_type: Optional[str] = None,
    signal_column: str = 'signal_id',
    date_column: str = 'I',
    value_column: str = 'value',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run full stability analysis pipeline.

    Args:
        observations_path: Path to observations.parquet
        output_dir: Directory for outputs
        window_size: Window for geometry computation
        step_size: Step size for rolling windows
        system_type: System typology (DEGRADATION, ACCUMULATION, etc.)
        signal_column, date_column, value_column: Column names
        verbose: Print progress

    Returns:
        Dict with all results and paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if verbose:
        print("=" * 70)
        print("PRISM STABILITY ANALYSIS PIPELINE")
        print("=" * 70)
        print(f"Input: {observations_path}")
        print(f"Output: {output_dir}")
        print()

    # Load observations
    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"Loaded {len(obs):,} observations")
        print(f"Signals: {obs[signal_column].n_unique()}")
        print()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: SIGNAL GEOMETRY (Cross-signal eigenstructure)
    # ─────────────────────────────────────────────────────────────────────
    if verbose:
        print("─" * 70)
        print("STEP 1: Signal Geometry (B-tipping detection)")
        print("─" * 70)

    geometry_path = output_dir / "signal_geometry.parquet"
    geometry_df = compute_cross_signal_geometry(
        obs,
        window_size=window_size,
        step_size=step_size,
        signal_column=signal_column,
        date_column=date_column,
        value_column=value_column,
        verbose=verbose,
    )

    if len(geometry_df) > 0:
        geometry_df.write_parquet(geometry_path)
        if verbose:
            print(f"  Saved: {geometry_path} ({len(geometry_df)} windows)")
    else:
        geometry_path = None
        if verbose:
            print("  Warning: No geometry computed (insufficient data)")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: LYAPUNOV ANALYSIS (Per-signal stability)
    # ─────────────────────────────────────────────────────────────────────
    if verbose:
        print()
        print("─" * 70)
        print("STEP 2: Lyapunov Analysis (stability measurement)")
        print("─" * 70)

    lyapunov_results = []
    signals = obs[signal_column].unique().to_list()

    for signal in signals:
        signal_data = obs.filter(pl.col(signal_column) == signal).sort(date_column)
        values = signal_data[value_column].to_numpy()

        result = compute_lyapunov(values)

        # Add stability classification
        lyap_val = result.get('lyapunov')
        if lyap_val is not None:
            stability = classify_stability(lyap_val)
            result['stability'] = stability.value
            result['lyapunov_exp'] = lyap_val
        else:
            result['stability'] = 'unknown'
            result['lyapunov_exp'] = np.nan

        result['signal_id'] = signal
        lyapunov_results.append(result)

    lyapunov_df = pl.DataFrame(lyapunov_results)
    lyapunov_path = output_dir / "lyapunov_analysis.parquet"
    lyapunov_df.write_parquet(lyapunov_path)

    if verbose:
        print(f"  Computed Lyapunov for {len(signals)} signals")
        stable_count = len([r for r in lyapunov_results
                           if r.get('stability') in ['stable', 'asymptotically_stable']])
        unstable_count = len([r for r in lyapunov_results
                             if r.get('stability') in ['weakly_unstable', 'chaotic', 'unstable']])
        print(f"  Stable: {stable_count}, Unstable: {unstable_count}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: CRITICAL SLOWING DOWN (Early warning)
    # ─────────────────────────────────────────────────────────────────────
    if verbose:
        print()
        print("─" * 70)
        print("STEP 3: Critical Slowing Down (early warning)")
        print("─" * 70)

    csd_results = []

    for signal in signals:
        signal_data = obs.filter(pl.col(signal_column) == signal).sort(date_column)
        values = signal_data[value_column].to_numpy()

        if len(values) >= 100:
            result = compute_csd(values)
            result['signal_id'] = signal
            csd_results.append(result)

    csd_path = None
    csd_df = pl.DataFrame()

    if csd_results:
        # Flatten nested fields
        flat_results = []
        for r in csd_results:
            flat = {k: v for k, v in r.items() if not isinstance(v, list)}
            flat_results.append(flat)

        csd_df = pl.DataFrame(flat_results)
        csd_path = output_dir / "csd_analysis.parquet"
        csd_df.write_parquet(csd_path)

        if verbose:
            detected_count = len([r for r in flat_results if r.get('csd_detected')])
            print(f"  Computed CSD for {len(csd_results)} signals")
            print(f"  CSD detected: {detected_count} signals")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: FORMAL ASSESSMENT
    # ─────────────────────────────────────────────────────────────────────
    if verbose:
        print()
        print("─" * 70)
        print("STEP 4: Formal Assessment")
        print("─" * 70)

    assessment_dict = None
    assessment_path = None

    # Get most recent geometry window
    if geometry_df is not None and len(geometry_df) > 0:
        latest_geom = geometry_df.sort('window_end').tail(1).to_dicts()[0]

        geometry_metrics = GeometryMetrics(
            eff_dim=latest_geom.get('eff_dim', np.nan),
            alignment=latest_geom.get('alignment', np.nan),
            mean_abs_correlation=latest_geom.get('mean_abs_correlation', np.nan),
            max_abs_correlation=latest_geom.get('max_abs_correlation', np.nan),
            coupling_fraction=latest_geom.get('coupling_fraction', np.nan),
            condition_number=latest_geom.get('condition_number', np.nan),
        )

        # Compute geometry trend
        if len(geometry_df) > 10:
            eff_dim_series = geometry_df['eff_dim'].to_numpy()
            t = np.arange(len(eff_dim_series))
            from scipy.stats import linregress
            slope, _, _, _, _ = linregress(t, eff_dim_series)
            geometry_metrics.eff_dim_slope = slope

        mass_metrics = MassMetrics(
            total_variance=latest_geom.get('total_variance', np.nan),
            dominant_signal_mean=0.0,
            drift_rate=0.0,
        )

        # Aggregate CSD across signals
        ews = None
        if len(csd_df) > 0:
            mean_autocorr = csd_df['autocorrelation_lag1'].mean()
            mean_var = csd_df['variance'].mean()
            any_csd = csd_df['csd_detected'].cast(pl.Boolean).any()

            ews = EarlyWarningSignals(
                autocorrelation_lag1=mean_autocorr,
                variance=mean_var,
                skewness=0.0,
                recovery_rate=0.1,
            )
            ews.critical_slowing_detected = any_csd

        # Classify
        sys_type = SystemTypology[system_type.upper()] if system_type else None
        failure_mode = classify_failure_mode(geometry_metrics, mass_metrics, ews, sys_type)
        tipping_type = classify_tipping_type(failure_mode, geometry_metrics, ews)

        # Determine stability from Lyapunov
        valid_lyap = [r['lyapunov_exp'] for r in lyapunov_results
                      if r.get('lyapunov_exp') is not None and np.isfinite(r.get('lyapunov_exp', np.nan))]
        if valid_lyap:
            mean_lyap = np.mean(valid_lyap)
            stability_type = classify_stability(mean_lyap)
        else:
            stability_type = StabilityType.MARGINALLY_STABLE

        # Build assessment
        assessment = FormalAssessment(
            attractor_type=AttractorType.FIXED_POINT,  # Simplification
            stability_type=stability_type,
            failure_mode=failure_mode,
            tipping_type=tipping_type,
            system_typology=sys_type,
            geometry=geometry_metrics,
            mass=mass_metrics,
            ews=ews,
            bifurcation_proximity=1.0 - geometry_metrics.eff_dim / 10 if geometry_metrics.eff_dim < 10 else 0.0,
            transition_probability=0.5,  # Would need model
            expected_severity=0.5,  # Would need model
            lead_time_estimate=None,
        )

        # Generate report
        report = assessment.to_report()

        if verbose:
            print(report)

        # Save assessment
        assessment_dict = {
            'timestamp': datetime.now().isoformat(),
            'failure_mode': failure_mode.value,
            'tipping_type': tipping_type.value,
            'stability': stability_type.value,
            'geometry': {
                'eff_dim': geometry_metrics.eff_dim,
                'alignment': geometry_metrics.alignment,
                'mean_correlation': geometry_metrics.mean_abs_correlation,
                'collapsing': geometry_metrics.is_collapsing(),
            },
            'csd_detected': ews.critical_slowing_detected if ews else False,
            'report': report,
        }

        assessment_path = output_dir / "formal_assessment.json"
        with open(assessment_path, 'w') as f:
            json.dump(assessment_dict, f, indent=2, default=str)

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    if verbose:
        print()
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Outputs:")
        if geometry_path:
            print(f"  - {geometry_path}")
        print(f"  - {lyapunov_path}")
        if csd_path:
            print(f"  - {csd_path}")
        if assessment_path:
            print(f"  - {assessment_path}")

    return {
        'geometry_path': str(geometry_path) if geometry_path else None,
        'lyapunov_path': str(lyapunov_path),
        'csd_path': str(csd_path) if csd_path else None,
        'assessment_path': str(assessment_path) if assessment_path else None,
        'geometry_df': geometry_df,
        'lyapunov_df': lyapunov_df,
        'csd_df': csd_df,
        'assessment': assessment_dict,
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="""
PRISM Stability Analysis Pipeline

Computes full dynamical systems assessment:
1. Signal geometry (cross-signal eigenstructure)
2. Lyapunov exponents (stability)
3. Critical slowing down (early warning)
4. Formal assessment (classification)

Example:
    python -m prism.entry_points.stability_pipeline observations.parquet -o results/
    python -m prism.entry_points.stability_pipeline observations.parquet --system-type ACCUMULATION
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='stability_output',
                        help='Output directory')
    parser.add_argument('-w', '--window-size', type=int, default=36,
                        help='Window size for geometry (default: 36)')
    parser.add_argument('-s', '--step-size', type=int, default=1,
                        help='Step size (default: 1)')
    parser.add_argument('--system-type', choices=['DEGRADATION', 'ACCUMULATION',
                                                   'CONSERVATION', 'OSCILLATORY', 'NETWORK'],
                        help='System typology hint')
    parser.add_argument('--signal-column', default='signal_id',
                        help='Signal column name')
    parser.add_argument('--date-column', default='I',
                        help='Date/time column name')
    parser.add_argument('--value-column', default='value',
                        help='Value column name')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    run_stability_pipeline(
        args.observations,
        args.output,
        window_size=args.window_size,
        step_size=args.step_size,
        system_type=args.system_type,
        signal_column=args.signal_column,
        date_column=args.date_column,
        value_column=args.value_column,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
