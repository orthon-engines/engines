"""
PRISM State Layer - State Signature Extraction
==============================================

Extracts the "fingerprint" of a transition for classification.
Different fault types have different transition signatures based on:
- Divergence characteristics (magnitude, direction)
- Leading signal identity and response strength
- Response cascade (which signals follow, in what order)
- Energy flow topology (sources vs sinks)

Usage:
    from orthon._internal.state import extract_signature, signatures_to_features

    sig = extract_signature(field_df, modes_df, '2000-07-28')
    features = signatures_to_features([sig])
"""

import polars as pl
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class StateSignature:
    """
    The fingerprint of a system state/transition.
    Used to classify what TYPE of transition occurred.

    Attributes:
        window_end: Window date of this signature

        # Divergence characteristics
        divergence_magnitude: Total absolute divergence
        divergence_direction: 'expansion' (+) or 'contraction' (-)

        # Leading signal info
        leading_signal: Signal with strongest response
        leader_gradient_mag: Leader's gradient magnitude
        leader_divergence: Leader's divergence value

        # Response cascade
        response_order: Top 10 responding signals
        response_timing: Gradient magnitudes of top responders
        n_affected: Count of above-median responders
        affected_ratio: Proportion of signals affected

        # Energy flow (from Laplace field topology)
        n_sources: Count of energy-radiating signals
        n_sinks: Count of energy-absorbing signals
        energy_balance: n_sources - n_sinks
    """
    window_end: str

    # Divergence characteristics
    divergence_magnitude: float
    divergence_direction: str  # 'expansion' or 'contraction'

    # Leading signal info
    leading_signal: str
    leader_gradient_mag: float
    leader_divergence: float

    # Response cascade
    response_order: List[str]
    response_timing: List[float]
    n_affected: int
    affected_ratio: float

    # Energy flow
    n_sources: int
    n_sinks: int
    energy_balance: float


def extract_signature(
    field_df: pl.DataFrame,
    modes_df: Optional[pl.DataFrame],
    window_end,
    exclude_signals: Optional[List[str]] = None,
) -> StateSignature:
    """
    Extract the full signature of a transition.

    Args:
        field_df: Signal field data with columns:
            - window_end, signal_id, divergence, gradient_magnitude
            - is_source, is_sink (optional, for energy flow)
        modes_df: Cohort modes data (optional, for cluster context)
        window_end: Date of transition window (str or date)
        exclude_signals: Signals to exclude from analysis

    Returns:
        StateSignature capturing the fingerprint of this transition

    Raises:
        ValueError: If no data exists for the specified window

    Example:
        >>> sig = extract_signature(field_df, None, '2000-07-28')
        >>> print(f"Leader: {sig.leading_signal}, Direction: {sig.divergence_direction}")
    """
    if exclude_signals is None:
        exclude_signals = []

    # Convert window_end to string for comparison
    window_str = str(window_end)

    # Get transition window data
    transition_data = field_df.filter(
        pl.col('window_end').cast(pl.Utf8) == window_str
    )

    if exclude_signals:
        transition_data = transition_data.filter(
            ~pl.col('signal_id').is_in(exclude_signals)
        )

    if len(transition_data) == 0:
        raise ValueError(f"No data for window {window_end}")

    # Divergence characteristics
    total_div = transition_data['divergence'].sum()
    if total_div is None:
        total_div = 0.0
    divergence_direction = 'expansion' if total_div > 0 else 'contraction'

    # Leading signal (highest gradient magnitude)
    sorted_by_grad = transition_data.sort('gradient_magnitude', descending=True)
    leader = sorted_by_grad.head(1)

    leading_signal = leader['signal_id'][0]
    leader_gradient_mag = leader['gradient_magnitude'][0]
    leader_divergence = leader['divergence'][0]

    # Handle None values
    if leader_gradient_mag is None:
        leader_gradient_mag = 0.0
    if leader_divergence is None:
        leader_divergence = 0.0

    # Response cascade
    response_order = sorted_by_grad['signal_id'].head(10).to_list()
    response_timing = sorted_by_grad['gradient_magnitude'].head(10).to_list()
    response_timing = [x if x is not None else 0.0 for x in response_timing]

    # Affected signals (above median gradient)
    median_grad = transition_data['gradient_magnitude'].median()
    if median_grad is None or median_grad == 0:
        median_grad = 0.0
        n_affected = len(transition_data)
    else:
        affected = transition_data.filter(pl.col('gradient_magnitude') > median_grad)
        n_affected = len(affected)

    total_signals = len(transition_data)
    affected_ratio = n_affected / total_signals if total_signals > 0 else 0.0

    # Energy flow (from Laplace field topology)
    if 'is_source' in transition_data.columns:
        n_sources = int(transition_data['is_source'].sum() or 0)
    else:
        n_sources = 0

    if 'is_sink' in transition_data.columns:
        n_sinks = int(transition_data['is_sink'].sum() or 0)
    else:
        n_sinks = 0

    energy_balance = float(n_sources - n_sinks)

    return StateSignature(
        window_end=window_str,
        divergence_magnitude=abs(float(total_div)),
        divergence_direction=divergence_direction,
        leading_signal=leading_signal,
        leader_gradient_mag=float(leader_gradient_mag),
        leader_divergence=float(leader_divergence),
        response_order=response_order,
        response_timing=response_timing,
        n_affected=n_affected,
        affected_ratio=affected_ratio,
        n_sources=n_sources,
        n_sinks=n_sinks,
        energy_balance=energy_balance,
    )


def signatures_to_features(signatures: List[StateSignature]) -> pl.DataFrame:
    """
    Convert signatures to feature matrix for classification.

    Transforms StateSignature objects into a DataFrame suitable for
    machine learning classification of transition types.

    Args:
        signatures: List of StateSignature objects

    Returns:
        DataFrame with numeric features and metadata:
            - window_end: Signature date (metadata)
            - divergence_magnitude: Total divergence magnitude
            - is_expansion: 1 if expansion, 0 if contraction
            - leader_gradient_mag: Leader's gradient magnitude
            - leader_divergence: Leader's divergence value
            - n_affected: Count of affected signals
            - affected_ratio: Proportion affected
            - n_sources: Energy source count
            - n_sinks: Energy sink count
            - energy_balance: n_sources - n_sinks
            - leader_signal: Leading signal ID (metadata)

    Example:
        >>> features = signatures_to_features(signatures)
        >>> X = features.select([c for c in features.columns if c not in ['window_end', 'leader_signal']])
    """
    if not signatures:
        return pl.DataFrame({
            'window_end': [],
            'divergence_magnitude': [],
            'is_expansion': [],
            'leader_gradient_mag': [],
            'leader_divergence': [],
            'n_affected': [],
            'affected_ratio': [],
            'n_sources': [],
            'n_sinks': [],
            'energy_balance': [],
            'leader_signal': [],
        })

    records = []

    for sig in signatures:
        records.append({
            'window_end': sig.window_end,
            'divergence_magnitude': sig.divergence_magnitude,
            'is_expansion': 1 if sig.divergence_direction == 'expansion' else 0,
            'leader_gradient_mag': sig.leader_gradient_mag,
            'leader_divergence': sig.leader_divergence,
            'n_affected': sig.n_affected,
            'affected_ratio': sig.affected_ratio,
            'n_sources': sig.n_sources,
            'n_sinks': sig.n_sinks,
            'energy_balance': sig.energy_balance,
            'leader_signal': sig.leading_signal,
        })

    return pl.DataFrame(records)


def extract_all_signatures(
    field_df: pl.DataFrame,
    window_ends: List,
    modes_df: Optional[pl.DataFrame] = None,
    exclude_signals: Optional[List[str]] = None,
) -> List[StateSignature]:
    """
    Extract signatures for multiple windows.

    Args:
        field_df: Signal field data
        window_ends: List of window dates to extract
        modes_df: Optional modes data
        exclude_signals: Signals to exclude

    Returns:
        List of StateSignature objects (skips windows with no data)

    Example:
        >>> _, transitions = detect_transitions(field_df)
        >>> sigs = extract_all_signatures(field_df, [t.window_end for t in transitions])
    """
    signatures = []

    for window in window_ends:
        try:
            sig = extract_signature(field_df, modes_df, window, exclude_signals)
            signatures.append(sig)
        except ValueError:
            # Skip windows with no data
            continue

    return signatures


def compare_signatures(
    sig_a: StateSignature,
    sig_b: StateSignature,
) -> Dict[str, float]:
    """
    Compare two state signatures and compute similarity metrics.

    Args:
        sig_a: First signature
        sig_b: Second signature

    Returns:
        Dictionary with comparison metrics:
            - divergence_diff: Absolute difference in divergence magnitude
            - direction_match: 1 if same direction, 0 otherwise
            - leader_match: 1 if same leader, 0 otherwise
            - response_overlap: Jaccard similarity of response order
            - energy_diff: Difference in energy balance
    """
    # Divergence magnitude difference
    divergence_diff = abs(sig_a.divergence_magnitude - sig_b.divergence_magnitude)

    # Direction match
    direction_match = 1.0 if sig_a.divergence_direction == sig_b.divergence_direction else 0.0

    # Leader match
    leader_match = 1.0 if sig_a.leading_signal == sig_b.leading_signal else 0.0

    # Response order overlap (Jaccard)
    set_a = set(sig_a.response_order)
    set_b = set(sig_b.response_order)
    if set_a or set_b:
        response_overlap = len(set_a & set_b) / len(set_a | set_b)
    else:
        response_overlap = 0.0

    # Energy balance difference
    energy_diff = abs(sig_a.energy_balance - sig_b.energy_balance)

    return {
        'divergence_diff': divergence_diff,
        'direction_match': direction_match,
        'leader_match': leader_match,
        'response_overlap': response_overlap,
        'energy_diff': energy_diff,
    }
