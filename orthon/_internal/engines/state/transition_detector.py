"""
PRISM State Layer - Transition Detector
=======================================

Detects regime boundaries using Laplace field divergence spikes.
A transition is a statistically significant change in system dynamics.

Key insight: Transitions manifest as divergence singularities in the field.
When a regime change occurs, the total system divergence spikes as signals
collectively respond to the structural shift.

Usage:
    from orthon._internal.state import detect_transitions, find_leading_signals

    system_div, transitions = detect_transitions(field_df, zscore_threshold=3.0)
    leaders = find_leading_signals(field_df, [t.window_end for t in transitions])
"""

import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Transition:
    """
    Detected regime transition.

    Attributes:
        window_start: Start of transition window
        window_end: End of transition window (detection point)
        divergence_zscore: Z-score of total divergence (significance measure)
        total_divergence: Sum of absolute signal divergences
        avg_gradient_magnitude: Mean gradient magnitude across signals
        leading_signal: Signal with highest gradient response
        response_order: Top 10 signals by response magnitude
        n_affected_signals: Count of signals with above-median response
    """
    window_start: str
    window_end: str
    divergence_zscore: float
    total_divergence: float
    avg_gradient_magnitude: float
    leading_signal: str
    response_order: List[str]
    n_affected_signals: int

    def __eq__(self, other):
        if not isinstance(other, Transition):
            return False
        return self.window_end == other.window_end

    def __hash__(self):
        return hash(self.window_end)


def compute_system_divergence(
    field_df: pl.DataFrame,
    exclude_signals: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Compute total system divergence per window.

    System divergence = sum of absolute signal divergences.
    High values indicate regime change (field singularity).

    Args:
        field_df: Signal field data with columns:
            - window_end: Date of window end
            - signal_id: Signal identifier
            - divergence: Laplace field divergence
            - gradient_magnitude: Magnitude of gradient vector
        exclude_signals: List of signal IDs to exclude (e.g., fault labels)

    Returns:
        DataFrame with columns:
            - window_end: Date
            - total_divergence: Sum of |divergence| across signals
            - avg_gradient_mag: Mean gradient magnitude
            - n_signals: Count of signals in window
    """
    if exclude_signals is None:
        exclude_signals = []

    filtered = field_df
    if exclude_signals:
        filtered = field_df.filter(
            ~pl.col('signal_id').is_in(exclude_signals)
        )

    return filtered.group_by('window_end').agg([
        pl.col('divergence').abs().sum().alias('total_divergence'),
        pl.col('gradient_magnitude').mean().alias('avg_gradient_mag'),
        pl.col('signal_id').n_unique().alias('n_signals'),
    ]).sort('window_end')


def detect_transitions(
    field_df: pl.DataFrame,
    zscore_threshold: float = 3.0,
    exclude_signals: Optional[List[str]] = None,
) -> Tuple[pl.DataFrame, List[Transition]]:
    """
    Detect regime transitions using divergence z-scores.

    A transition is flagged when:
    - |divergence_zscore| > threshold (default 3.0)

    The z-score is computed relative to the median and standard deviation
    of total divergence across all windows.

    Args:
        field_df: signal_field.parquet data with columns:
            - window_end, signal_id, divergence, gradient_magnitude
        zscore_threshold: Standard deviations for significance (default 3.0)
        exclude_signals: Signals to exclude (e.g., ['TEP_FAULT'])

    Returns:
        Tuple of:
            - DataFrame of all windows with z-scores (for analysis)
            - List of detected Transition objects (significant transitions)

    Example:
        >>> system_div, transitions = detect_transitions(field_df, zscore_threshold=3.0)
        >>> print(f"Found {len(transitions)} transitions")
        >>> for t in transitions[:5]:
        ...     print(f"  {t.window_end}: z={t.divergence_zscore:.2f}, leader={t.leading_signal}")
    """
    # Compute system divergence
    system_div = compute_system_divergence(field_df, exclude_signals)

    if len(system_div) == 0:
        return system_div, []

    # Compute z-scores using median (robust to outliers)
    median_div = system_div['total_divergence'].median()
    std_div = system_div['total_divergence'].std()

    if std_div is None or std_div == 0:
        system_div = system_div.with_columns([
            pl.lit(0.0).alias('divergence_zscore')
        ])
        return system_div, []

    system_div = system_div.with_columns([
        ((pl.col('total_divergence') - median_div) / std_div).alias('divergence_zscore')
    ])

    # Find transition windows
    transition_windows = system_div.filter(
        pl.col('divergence_zscore').abs() > zscore_threshold
    ).sort('divergence_zscore', descending=True)

    # Build Transition objects with response analysis
    transitions = []

    for row in transition_windows.iter_rows(named=True):
        window_end = row['window_end']

        # Find leading signals for this window
        window_data = field_df.filter(
            pl.col('window_end') == window_end
        )

        # Exclude specified signals from leadership analysis
        if exclude_signals:
            window_data = window_data.filter(
                ~pl.col('signal_id').is_in(exclude_signals)
            )

        window_data = window_data.sort('gradient_magnitude', descending=True)

        if len(window_data) > 0:
            leading = window_data['signal_id'].head(1)[0]
            response_order = window_data['signal_id'].head(10).to_list()

            # Count affected signals (above median gradient)
            median_grad = window_data['gradient_magnitude'].median()
            if median_grad is not None:
                n_affected = len(window_data.filter(
                    pl.col('gradient_magnitude') > median_grad
                ))
            else:
                n_affected = len(window_data) // 2
        else:
            leading = 'unknown'
            response_order = []
            n_affected = 0

        transitions.append(Transition(
            window_start=str(window_end),
            window_end=str(window_end),
            divergence_zscore=row['divergence_zscore'],
            total_divergence=row['total_divergence'],
            avg_gradient_magnitude=row['avg_gradient_mag'],
            leading_signal=leading,
            response_order=response_order,
            n_affected_signals=n_affected,
        ))

    return system_div, transitions


def find_leading_signals(
    field_df: pl.DataFrame,
    transition_windows: List,
    top_n: int = 5,
    exclude_signals: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    For each transition window, identify which signals responded first/strongest.

    Args:
        field_df: Signal field data
        transition_windows: List of window_end dates (str or date objects)
        top_n: Number of top responders per window (default 5)
        exclude_signals: Signals to exclude from analysis

    Returns:
        DataFrame with columns:
            - window_end: Transition window date
            - rank: Response rank (1 = strongest)
            - signal_id: Signal identifier
            - gradient_magnitude: Response magnitude
            - divergence: Signal divergence at transition

    Example:
        >>> leaders = find_leading_signals(field_df, ['2000-07-28', '2000-02-27'])
        >>> print(leaders.filter(pl.col('rank') == 1))  # Show only top responders
    """
    if exclude_signals is None:
        exclude_signals = []

    results = []

    for window in transition_windows:
        # Convert to string if needed for comparison
        window_str = str(window)

        window_data = field_df.filter(
            pl.col('window_end').cast(pl.Utf8) == window_str
        )

        if exclude_signals:
            window_data = window_data.filter(
                ~pl.col('signal_id').is_in(exclude_signals)
            )

        window_data = window_data.sort('gradient_magnitude', descending=True).head(top_n)

        for rank, row in enumerate(window_data.iter_rows(named=True), 1):
            results.append({
                'window_end': window_str,
                'rank': rank,
                'signal_id': row['signal_id'],
                'gradient_magnitude': row['gradient_magnitude'],
                'divergence': row['divergence'],
            })

    if not results:
        return pl.DataFrame({
            'window_end': [],
            'rank': [],
            'signal_id': [],
            'gradient_magnitude': [],
            'divergence': [],
        })

    return pl.DataFrame(results)


def summarize_leading_signals(
    transitions: List[Transition],
) -> pl.DataFrame:
    """
    Summarize which signals lead transitions most frequently.

    Args:
        transitions: List of detected Transition objects

    Returns:
        DataFrame with columns:
            - signal_id: Signal identifier
            - lead_count: Number of times this signal led a transition
            - lead_ratio: Proportion of transitions led
            - avg_zscore: Average z-score when leading

    Example:
        >>> _, transitions = detect_transitions(field_df)
        >>> summary = summarize_leading_signals(transitions)
        >>> print(summary.head(5))  # Top 5 leading signals
    """
    from collections import Counter

    if not transitions:
        return pl.DataFrame({
            'signal_id': [],
            'lead_count': [],
            'lead_ratio': [],
            'avg_zscore': [],
        })

    # Count leadership frequency
    leaders = [t.leading_signal for t in transitions]
    counts = Counter(leaders)

    # Compute average z-score when leading
    leader_zscores = {}
    for t in transitions:
        leader = t.leading_signal
        if leader not in leader_zscores:
            leader_zscores[leader] = []
        leader_zscores[leader].append(abs(t.divergence_zscore))

    records = []
    total = len(transitions)
    for signal, count in counts.most_common():
        records.append({
            'signal_id': signal,
            'lead_count': count,
            'lead_ratio': count / total,
            'avg_zscore': np.mean(leader_zscores[signal]),
        })

    return pl.DataFrame(records)
