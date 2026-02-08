"""
Stage 19: Information Flow Delta Entry Point
============================================

Computes Granger causality independently per segment, then computes deltas.
Shows which causal links appear, disappear, strengthen, or weaken across
a boundary (fault injection, election, regime change).

Manifest config:
    segments:
      - name: pre
        range: [0, 20]
      - name: post
        range: [21, null]

Inputs:
    - observations.parquet

Output:
    - info_flow_delta.parquet

Link status classification:
    emerged:      not significant pre → significant post (new causal path)
    broken:       significant pre → not significant post (feedback disrupted)
    strengthened: significant both, F-stat increased >50%
    weakened:     significant both, F-stat decreased >33%
    stable:       significant both, F-stat similar
    absent:       not significant in either segment
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from engines.manifold.pairwise.causality import compute_granger


def run(
    observations_path: str,
    output_path: str = "info_flow_delta.parquet",
    segments: List[Dict[str, Any]] = None,
    max_lag: int = 5,
    min_samples: int = None,
    p_threshold: float = 0.05,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute per-segment Granger causality and deltas.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for info_flow_delta.parquet
        segments: List of segment definitions
        max_lag: Maximum lag for Granger test
        min_samples: Minimum samples per segment for Granger test
        p_threshold: Significance threshold
        verbose: Print progress

    Returns:
        info_flow_delta DataFrame
    """
    # Adaptive min_samples: Granger at max_lag needs ~3*lag+5 observations minimum
    if min_samples is None:
        min_samples = max(3 * max_lag + 5, 15)

    if verbose:
        print("=" * 70)
        print("STAGE 19: INFORMATION FLOW DELTA")
        print("Per-segment Granger causality with link change classification")
        print("=" * 70)
        print(f"  min_samples={min_samples} (adaptive from max_lag={max_lag})")

    if segments is None or len(segments) < 2:
        if verbose:
            print("Warning: Need at least 2 segments. Using default split at I=20.")
        segments = [
            {'name': 'pre', 'range': [0, 19]},
            {'name': 'post', 'range': [20, None]},
        ]

    if verbose:
        print(f"Segments: {len(segments)}")
        for seg in segments:
            end = seg['range'][1] if seg['range'][1] is not None else 'end'
            print(f"  {seg['name']}: I in [{seg['range'][0]}, {end}]")

    # Load observations
    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"\nLoaded observations: {obs.shape}")

    has_cohort = 'cohort' in obs.columns
    cohorts = obs['cohort'].unique().to_list() if has_cohort else ['all']
    signals = sorted(obs['signal_id'].unique().to_list())

    if verbose:
        print(f"Cohorts: {len(cohorts)}")
        print(f"Signals: {len(signals)}")
        print(f"Signal pairs: {len(signals) * (len(signals) - 1)}")

    results = []

    for cohort_idx, cohort in enumerate(cohorts):
        if verbose and cohort_idx % 10 == 0:
            print(f"  Processing cohort {cohort_idx + 1}/{len(cohorts)}: {cohort}")

        if has_cohort:
            cohort_data = obs.filter(pl.col('cohort') == cohort)
        else:
            cohort_data = obs

        i_max = cohort_data['I'].max()

        # Compute Granger for each segment
        segment_granger = {}

        for seg in segments:
            seg_name = seg['name']
            start_i = seg['range'][0]
            end_i = seg['range'][1] if seg['range'][1] is not None else i_max

            seg_data = cohort_data.filter(
                (pl.col('I') >= start_i) & (pl.col('I') <= end_i)
            )

            # Pivot to wide format (rows = I values, columns = signals)
            try:
                wide = seg_data.pivot(
                    values='value',
                    index='I',
                    on='signal_id',
                ).sort('I')
            except Exception:
                continue

            if wide is None or len(wide) < min_samples:
                if verbose:
                    n_wide = 0 if wide is None else len(wide)
                    print(f"    Skipping segment '{seg_name}': {n_wide} I values < min_samples={min_samples}")
                continue

            # Compute pairwise Granger
            signal_cols = [c for c in wide.columns if c != 'I']
            granger_results = {}

            for source in signal_cols:
                for target in signal_cols:
                    if source == target:
                        continue

                    x = wide[source].to_numpy()
                    y = wide[target].to_numpy()

                    # Remove NaN
                    valid = ~(np.isnan(x) | np.isnan(y))
                    x = x[valid]
                    y = y[valid]

                    if len(x) < min_samples:
                        continue

                    try:
                        result = compute_granger(x, y, max_lag=max_lag)
                        granger_results[(source, target)] = result
                    except Exception:
                        continue

            segment_granger[seg_name] = granger_results

        # Compute deltas between consecutive segments
        seg_names = [s['name'] for s in segments if s['name'] in segment_granger]

        for i in range(len(seg_names) - 1):
            seg_a = seg_names[i]
            seg_b = seg_names[i + 1]

            granger_a = segment_granger.get(seg_a, {})
            granger_b = segment_granger.get(seg_b, {})

            # Get all pairs from both segments
            all_pairs = set(granger_a.keys()) | set(granger_b.keys())

            for source, target in all_pairs:
                result_a = granger_a.get((source, target), {})
                result_b = granger_b.get((source, target), {})

                # Use granger_f and granger_p (engine interface)
                f_a = result_a.get('granger_f')
                f_b = result_b.get('granger_f')
                p_a = result_a.get('granger_p', 1.0)
                p_b = result_b.get('granger_p', 1.0)

                sig_a = p_a < p_threshold if p_a is not None else False
                sig_b = p_b < p_threshold if p_b is not None else False

                # Classify link change
                link_status = classify_link_change(sig_a, sig_b, f_a, f_b)

                results.append({
                    'cohort': cohort,
                    'segment_a': seg_a,
                    'segment_b': seg_b,
                    'source': source,
                    'target': target,
                    'f_stat_a': f_a,
                    'f_stat_b': f_b,
                    'f_stat_delta': (f_b - f_a) if f_a is not None and f_b is not None else None,
                    'p_value_a': p_a,
                    'p_value_b': p_b,
                    'significant_a': sig_a,
                    'significant_b': sig_b,
                    'link_status': link_status,
                })

    # Build output
    if results:
        result = pl.DataFrame(results)
    else:
        result = pl.DataFrame(schema={
            'cohort': pl.Utf8,
            'segment_a': pl.Utf8,
            'segment_b': pl.Utf8,
            'source': pl.Utf8,
            'target': pl.Utf8,
            'f_stat_a': pl.Float64,
            'f_stat_b': pl.Float64,
            'f_stat_delta': pl.Float64,
            'p_value_a': pl.Float64,
            'p_value_b': pl.Float64,
            'significant_a': pl.Boolean,
            'significant_b': pl.Boolean,
            'link_status': pl.Utf8,
        })

    result.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {result.shape}")

        if len(result) > 0:
            print("\nLink status distribution:")
            status_counts = result.group_by('link_status').agg(pl.len().alias('count')).sort('count', descending=True)
            for row in status_counts.iter_rows(named=True):
                print(f"  {row['link_status']}: {row['count']}")

        print()
        print("─" * 50)
        print(f"✓ {Path(output_path).absolute()}")
        print("─" * 50)

    return result


def classify_link_change(sig_a: bool, sig_b: bool, f_a: float, f_b: float) -> str:
    """
    Classify how a causal link changed between segments.

    Args:
        sig_a: Significant in segment A
        sig_b: Significant in segment B
        f_a: F-statistic in segment A
        f_b: F-statistic in segment B

    Returns:
        Link status: emerged, broken, strengthened, weakened, stable, absent
    """
    if not sig_a and sig_b:
        return 'emerged'
    elif sig_a and not sig_b:
        return 'broken'
    elif sig_a and sig_b:
        if f_a is None or f_b is None:
            return 'stable'
        if f_b > f_a * 1.5:
            return 'strengthened'
        elif f_b < f_a * 0.67:
            return 'weakened'
        else:
            return 'stable'
    else:
        return 'absent'


def main():
    parser = argparse.ArgumentParser(
        description="Stage 19: Information Flow Delta",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes Granger causality per segment and classifies link changes.

Link status:
  emerged:      New causal path appeared after boundary
  broken:       Existing feedback loop disrupted
  strengthened: Coupling intensified (F-stat +50%)
  weakened:     Control authority degrading (F-stat -33%)
  stable:       Causal structure preserved
  absent:       No causal relationship in either segment

Example:
  python -m engines.entry_points.stage_19_info_flow_delta \\
      observations.parquet -o info_flow_delta.parquet --split-at 20
"""
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('-o', '--output', default='info_flow_delta.parquet',
                        help='Output path (default: info_flow_delta.parquet)')
    parser.add_argument('--split-at', type=int, default=20,
                        help='I value to split at (default: 20)')
    parser.add_argument('--max-lag', type=int, default=5,
                        help='Maximum lag for Granger test (default: 5)')
    parser.add_argument('--min-samples', type=int, default=None,
                        help='Minimum samples per segment (default: adaptive from max_lag)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    segments = [
        {'name': 'pre', 'range': [0, args.split_at - 1]},
        {'name': 'post', 'range': [args.split_at, None]},
    ]

    run(
        args.observations,
        args.output,
        segments=segments,
        max_lag=args.max_lag,
        min_samples=args.min_samples,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
