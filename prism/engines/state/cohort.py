"""
PRISM State Layer Runner - Cohort State Computation
===================================================

Orchestrates the State Layer to compute system-level dynamic state,
transition detection, and regime tracking.

MEMORY OPTIMIZED: Uses Polars lazy streaming to process large datasets
without loading all data into RAM. Processes 140M+ rows with ~2GB memory.

The State Layer is the final layer of the PRISM Trilogy:
    Layer 1 (Vector):   "How does each signal behave?"
    Layer 2 (Geometry): "How do signals relate?"
    Layer 3 (State):    "What is the system DOING?"

Pipeline:
    signal_field.parquet -> cohort_state.parquet

Output Schema (cohort_state.parquet):
    - domain_id: Domain identifier
    - cohort_id: Cohort identifier
    - window_end: Window date
    - is_transition: Whether this window is a detected transition
    - divergence_zscore: Z-score of total divergence
    - total_divergence: Sum of absolute divergences
    - state_id: Current regime ID
    - state_duration: Windows since last transition
    - state_stability: Stability score (0=unstable, 1=stable)
    - leading_signal: Top responding signal
    - response_order: Top 10 responders (JSON array)
    - n_affected: Count of affected signals

Usage:
    python -m prism.entry_points.cohort_state --domain cheme
    python -m prism.entry_points.cohort_state --domain cheme --zscore-threshold 2.5
    python -m prism.entry_points.cohort_state --domain cheme --classify
"""

import argparse
import json
import os
import polars as pl
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import Counter

from prism.db.parquet_store import (
    get_path,
    get_data_root,
    ensure_directory,
    OBSERVATIONS,
    VECTOR,
    GEOMETRY,
    STATE,
    COHORTS,
)
from prism.db.polars_io import upsert_parquet, write_parquet_atomic
from prism.utils.domain import require_domain


# Key columns for upsert
KEY_COLS = ['domain_id', 'cohort_id', 'window_end']

# Columns needed for state computation (memory optimization)
STATE_COLUMNS = [
    'window_end',
    'signal_id',
    'divergence',
    'gradient_magnitude',
    'is_source',
    'is_sink',
]


def get_cohort_ids(domain: str) -> List[str]:
    """Get all cohort IDs for the domain."""
    members_path = get_path(COHORTS)
    if not members_path.exists():
        return ['default']

    df = pl.read_parquet(members_path)
    cohorts = df['cohort_id'].unique().to_list()
    return cohorts if cohorts else ['default']


def compute_system_divergence_streaming(
    field_path: Path,
    exclude_patterns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Compute system divergence using lazy streaming.

    Memory efficient: processes data in chunks without loading all into RAM.

    Args:
        field_path: Path to signal_field.parquet
        exclude_patterns: Patterns to exclude (e.g., ['FAULT'])

    Returns:
        DataFrame with window_end, total_divergence, avg_gradient_mag, n_signals
    """
    # Build filter expression for exclusions
    filter_expr = pl.lit(True)
    if exclude_patterns:
        for pattern in exclude_patterns:
            filter_expr = filter_expr & ~pl.col('signal_id').str.contains(pattern)

    # Use lazy scan with streaming - never loads all data into RAM
    result = (
        pl.scan_parquet(field_path)
        .select(['window_end', 'signal_id', 'divergence', 'gradient_magnitude'])
        .filter(filter_expr)
        .group_by('window_end')
        .agg([
            pl.col('divergence').abs().sum().alias('total_divergence'),
            pl.col('gradient_magnitude').mean().alias('avg_gradient_mag'),
            pl.col('signal_id').n_unique().alias('n_signals'),
        ])
        .sort('window_end')
        .collect(engine='streaming')
    )

    return result


def detect_transitions_streaming(
    field_path: Path,
    zscore_threshold: float = 3.0,
    exclude_patterns: Optional[List[str]] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Detect transitions using streaming computation.

    Args:
        field_path: Path to signal_field.parquet
        zscore_threshold: Z-score threshold for significance
        exclude_patterns: Patterns to exclude

    Returns:
        Tuple of (system_div DataFrame with z-scores, transitions DataFrame)
    """
    # Step 1: Compute system divergence (streaming)
    system_div = compute_system_divergence_streaming(field_path, exclude_patterns)

    if len(system_div) == 0:
        return system_div, pl.DataFrame()

    # Step 2: Compute z-scores
    median_div = system_div['total_divergence'].median()
    std_div = system_div['total_divergence'].std()

    if std_div is None or std_div == 0:
        system_div = system_div.with_columns([
            pl.lit(0.0).alias('divergence_zscore')
        ])
    else:
        system_div = system_div.with_columns([
            ((pl.col('total_divergence') - median_div) / std_div).alias('divergence_zscore')
        ])

    # Step 3: Filter to transitions
    transitions = system_div.filter(
        pl.col('divergence_zscore').abs() > zscore_threshold
    ).sort('divergence_zscore', descending=True)

    return system_div, transitions


def find_leading_signals_streaming(
    field_path: Path,
    transition_windows: List,
    exclude_patterns: Optional[List[str]] = None,
    top_n: int = 10,
) -> Dict[str, Dict]:
    """
    Find leading signals for each transition window using streaming.

    Args:
        field_path: Path to signal_field.parquet
        transition_windows: List of window dates
        exclude_patterns: Patterns to exclude
        top_n: Number of top responders to return

    Returns:
        Dict mapping window_end (str) to leader info
    """
    if not transition_windows:
        return {}

    # Build filter expression
    filter_expr = pl.col('window_end').is_in(transition_windows)
    if exclude_patterns:
        for pattern in exclude_patterns:
            filter_expr = filter_expr & ~pl.col('signal_id').str.contains(pattern)

    # Stream through data for just transition windows
    window_data = (
        pl.scan_parquet(field_path)
        .select(['window_end', 'signal_id', 'gradient_magnitude', 'divergence'])
        .filter(filter_expr)
        .collect(engine='streaming')
    )

    # Process each window
    results = {}
    for window in transition_windows:
        window_str = str(window)
        wdata = window_data.filter(pl.col('window_end') == window)

        if len(wdata) == 0:
            results[window_str] = {
                'leading_signal': 'unknown',
                'response_order': [],
                'n_affected': 0,
            }
            continue

        # Sort by gradient magnitude
        sorted_data = wdata.sort('gradient_magnitude', descending=True)

        # Get leader and response order
        leading = sorted_data['signal_id'][0]
        response_order = sorted_data['signal_id'].head(top_n).to_list()

        # Count affected (above median)
        median_grad = wdata['gradient_magnitude'].median()
        if median_grad and median_grad > 0:
            n_affected = len(wdata.filter(pl.col('gradient_magnitude') > median_grad))
        else:
            n_affected = len(wdata) // 2

        results[window_str] = {
            'leading_signal': leading,
            'response_order': response_order,
            'n_affected': n_affected,
        }

    return results


def run_cohort_state(
    domain: str,
    zscore_threshold: float = 3.0,
    exclude_patterns: Optional[List[str]] = None,
    classify: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run state computation for all cohorts in a domain.

    MEMORY OPTIMIZED: Uses streaming to process large datasets.

    Args:
        domain: Domain identifier
        zscore_threshold: Z-score threshold for transitions
        exclude_patterns: Signal patterns to exclude
        classify: Whether to run unsupervised classification
        verbose: Print progress

    Returns:
        Summary statistics
    """
    ensure_directory(domain)

    # Default exclude patterns
    if exclude_patterns is None:
        exclude_patterns = ['FAULT']

    # Get field path
    field_path = get_path(VECTOR)
    if not field_path.exists():
        raise FileNotFoundError(f"Field data not found: {field_path}")

    if verbose:
        print("=" * 80)
        print("PRISM STATE LAYER - COHORT STATE COMPUTATION")
        print("=" * 80)
        print(f"Domain: {domain}")
        print(f"Z-score threshold: {zscore_threshold}")
        print(f"Exclude patterns: {exclude_patterns}")
        print(f"Mode: STREAMING (memory optimized)")
        print()

        # Show file size
        size_gb = field_path.stat().st_size / (1024**3)
        print(f"Field data: {size_gb:.2f} GB on disk")

    # Get cohorts
    cohort_ids = get_cohort_ids(domain)
    if verbose:
        print(f"Cohorts: {cohort_ids}")
        print()

    # Step 1: Detect transitions (streaming)
    if verbose:
        print("Step 1: Computing system divergence (streaming)...")

    system_div, transitions_df = detect_transitions_streaming(
        field_path,
        zscore_threshold=zscore_threshold,
        exclude_patterns=exclude_patterns,
    )

    if verbose:
        print(f"  Windows: {len(system_div):,}")
        print(f"  Transitions: {len(transitions_df)}")

    # Step 2: Find leading signals for transitions (streaming)
    transition_windows = transitions_df['window_end'].to_list()

    if verbose and transition_windows:
        print()
        print("Step 2: Finding leading signals (streaming)...")

    leader_info = find_leading_signals_streaming(
        field_path,
        transition_windows,
        exclude_patterns=exclude_patterns,
    )

    # Step 3: Build output records
    if verbose:
        print()
        print("Step 3: Building state records...")

    transition_set = set(str(w) for w in transition_windows)
    computed_at = datetime.now()

    records = []
    state_id = 0
    duration = 0

    # Use first cohort_id (typically one cohort per domain)
    cohort_id = cohort_ids[0] if cohort_ids else 'default'

    for row in system_div.sort('window_end').iter_rows(named=True):
        window_end = row['window_end']
        window_str = str(window_end)

        is_transition = window_str in transition_set

        if is_transition:
            state_id += 1
            duration = 1
            info = leader_info.get(window_str, {})
            leading_signal = info.get('leading_signal')
            response_order = info.get('response_order', [])
            n_affected = info.get('n_affected', 0)
        else:
            duration += 1
            leading_signal = None
            response_order = []
            n_affected = 0

        # Compute stability (reaches 1.0 after 5 windows)
        stability = min(1.0, duration / 5) if duration > 0 else 0.0

        records.append({
            'domain_id': domain,
            'cohort_id': cohort_id,
            'window_end': window_end,
            'is_transition': is_transition,
            'divergence_zscore': row['divergence_zscore'],
            'total_divergence': row['total_divergence'],
            'avg_gradient_mag': row['avg_gradient_mag'],
            'n_signals': row['n_signals'],
            'state_id': state_id,
            'state_duration': duration,
            'state_stability': stability,
            'leading_signal': leading_signal,
            'response_order': json.dumps(response_order) if response_order else None,
            'n_affected': n_affected,
            'computed_at': computed_at,
        })

    if not records:
        print("No results generated")
        return {'total_windows': 0, 'total_transitions': 0}

    result_df = pl.DataFrame(records)

    # Optional classification
    if classify and len(transitions_df) >= 5:
        if verbose:
            print()
            print("Step 4: Running unsupervised classification...")

        try:
            from prism.state import StateClassifier, extract_all_signatures

            # Load minimal data for signature extraction
            sig_data = (
                pl.scan_parquet(field_path)
                .select(['window_end', 'signal_id', 'divergence', 'gradient_magnitude', 'is_source', 'is_sink'])
                .filter(pl.col('window_end').is_in(transition_windows))
                .collect(engine='streaming')
            )

            signatures = []
            for window in transition_windows:
                from prism.state import extract_signature
                try:
                    sig = extract_signature(sig_data, None, window, exclude_patterns)
                    signatures.append(sig)
                except Exception:
                    pass

            if len(signatures) >= 5:
                clf = StateClassifier(mode='unsupervised', n_clusters=min(5, len(signatures) // 2))
                clf.fit(signatures)
                predictions = clf.predict(signatures)

                pred_map = {sig.window_end: int(pred) for sig, pred in zip(signatures, predictions)}

                result_df = result_df.with_columns([
                    pl.col('window_end').cast(pl.Utf8).replace(pred_map, default=-1).alias('cluster_id')
                ])

                if verbose:
                    print(f"  Classified {len(signatures)} transitions into {clf.n_clusters} clusters")
        except Exception as e:
            if verbose:
                print(f"  Classification failed: {e}")

    # Write output
    output_path = get_path(STATE)
    write_parquet_atomic(result_df, output_path)

    total_transitions = len(transitions_df)

    if verbose:
        print()
        print("=" * 80)
        print("STATE COMPUTATION COMPLETE")
        print("=" * 80)
        print(f"Total windows: {len(result_df):,}")
        print(f"Total transitions: {total_transitions}")
        print(f"Total states: {state_id}")
        print(f"Output: {output_path}")

        # Show top transitions
        if total_transitions > 0:
            print()
            print("Top 10 Transitions:")
            print("-" * 60)
            trans_rows = result_df.filter(pl.col('is_transition')).sort(
                'divergence_zscore', descending=True
            )
            for row in trans_rows.head(10).iter_rows(named=True):
                z = row['divergence_zscore']
                leader = row['leading_signal'] or 'N/A'
                print(f"  {row['window_end']}: z={z:+.2f}, leader={leader}")

            # Summarize leading signals
            print()
            print("Leading Signal Summary:")
            print("-" * 60)
            leaders = trans_rows['leading_signal'].drop_nulls().to_list()
            for ind, count in Counter(leaders).most_common(5):
                pct = count / total_transitions * 100
                print(f"  {ind}: {count} ({pct:.1f}%)")

    return {
        'total_windows': len(result_df),
        'total_transitions': total_transitions,
        'total_states': state_id,
        'output_path': str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description='PRISM State Layer - Cohort State Computation (Memory Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m prism.entry_points.cohort_state --domain cheme

  # With custom threshold
  python -m prism.entry_points.cohort_state --domain cheme --zscore-threshold 2.5

  # With unsupervised classification
  python -m prism.entry_points.cohort_state --domain cheme --classify

Memory Usage:
  Uses Polars lazy streaming to process large datasets efficiently.
  140M+ rows can be processed with ~2GB RAM instead of 40GB+.

Output:
  data/{domain}/state/cohort.parquet
"""
    )

    parser.add_argument('--domain', type=str, default=None,
                        help='Domain to process (prompts if not specified)')
    parser.add_argument('--zscore-threshold', type=float, default=3.0,
                        help='Z-score threshold for transition detection (default: 3.0)')
    parser.add_argument('--exclude', type=str, nargs='+', default=['FAULT'],
                        help='Signal patterns to exclude (default: FAULT)')
    parser.add_argument('--classify', action='store_true',
                        help='Run unsupervised classification on transitions')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Domain selection
    domain = require_domain(args.domain, "Select domain for state computation")
    os.environ["PRISM_DOMAIN"] = domain

    run_cohort_state(
        domain=domain,
        zscore_threshold=args.zscore_threshold,
        exclude_patterns=args.exclude,
        classify=args.classify,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
