"""
PRISM State Layer - Regime Tracker
==================================

Tracks system state over time, maintaining history of:
- Current regime (state ID and characteristics)
- Transition history (all detected regime changes)
- State duration (time since last transition)
- Early warning signals (signs of impending transition)

The tracker operates incrementally, updating state as new data arrives.

Usage:
    from orthon._internal.state import RegimeTracker

    tracker = RegimeTracker(zscore_threshold=3.0)
    result = tracker.update(field_df, exclude_signals=['TEP_FAULT'])
    print(f"Current state: {tracker.current_state}")
    print(f"Early warning: {result['early_warning']}")
"""

import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import Counter

from .transition_detector import detect_transitions, Transition
from .state_signature import extract_signature, StateSignature


@dataclass
class RegimeState:
    """
    Current regime state.

    Attributes:
        state_id: Unique identifier for this regime
        start_time: When this regime began
        current_time: Most recent observation time
        duration_windows: Number of windows in this regime
        stability: 0 = unstable (just transitioned), 1 = stable (long duration)
        leading_signals: Top signals from recent transitions
    """
    state_id: int
    start_time: str
    current_time: str
    duration_windows: int
    stability: float  # 0 = unstable, 1 = stable
    leading_signals: List[str]


@dataclass
class RegimeHistory:
    """
    Historical record of regime changes.

    Attributes:
        transitions: All detected Transition objects
        states: All RegimeState objects (regime periods)
        signatures: StateSignature for each transition
    """
    transitions: List[Transition] = field(default_factory=list)
    states: List[RegimeState] = field(default_factory=list)
    signatures: List[StateSignature] = field(default_factory=list)


class RegimeTracker:
    """
    Tracks system regime over time.

    Maintains state across updates, detecting new transitions and
    tracking regime duration/stability.

    Attributes:
        zscore_threshold: Z-score threshold for transition detection
        stability_window: Windows needed for full stability (1.0)
        history: RegimeHistory with all past data
        current_state: Current RegimeState (None if no data)
        state_counter: Running count of regime changes

    Example:
        >>> tracker = RegimeTracker(zscore_threshold=3.0)
        >>> for batch in data_batches:
        ...     result = tracker.update(batch)
        ...     if result['early_warning']:
        ...         print("Warning: Transition may be imminent!")
        >>> summary = tracker.get_summary()
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        stability_window: int = 5,
    ):
        """
        Initialize the tracker.

        Args:
            zscore_threshold: Z-score threshold for transition detection
            stability_window: Number of windows to reach full stability
        """
        self.zscore_threshold = zscore_threshold
        self.stability_window = stability_window

        self.history = RegimeHistory()
        self.current_state: Optional[RegimeState] = None
        self.state_counter = 0

        # For early warning detection
        self._recent_gradients: List[float] = []

    def update(
        self,
        field_df: pl.DataFrame,
        modes_df: Optional[pl.DataFrame] = None,
        exclude_signals: Optional[List[str]] = None,
    ) -> Dict:
        """
        Update tracker with new data.

        Detects new transitions, updates current state, and checks for
        early warning signs of impending transitions.

        Args:
            field_df: Signal field data
            modes_df: Optional cohort modes data
            exclude_signals: Signals to exclude (e.g., fault labels)

        Returns:
            Dictionary with:
                - transitions_detected: List of new Transition objects
                - current_state: Current RegimeState
                - early_warning: Boolean indicating potential imminent transition
                - total_transitions: Total count of transitions in history
        """
        # Detect transitions
        _, transitions = detect_transitions(
            field_df,
            zscore_threshold=self.zscore_threshold,
            exclude_signals=exclude_signals,
        )

        # Find new transitions (not in history)
        new_transitions = []
        existing_windows = {t.window_end for t in self.history.transitions}

        for t in transitions:
            if t.window_end not in existing_windows:
                self.history.transitions.append(t)
                new_transitions.append(t)

                # Extract signature for new transition
                try:
                    sig = extract_signature(
                        field_df, modes_df, t.window_end, exclude_signals
                    )
                    self.history.signatures.append(sig)
                except Exception:
                    pass

        # Update current state
        if new_transitions:
            # New regime started
            self.state_counter += 1
            latest_transition = new_transitions[-1]

            self.current_state = RegimeState(
                state_id=self.state_counter,
                start_time=latest_transition.window_end,
                current_time=latest_transition.window_end,
                duration_windows=1,
                stability=0.0,  # Just transitioned
                leading_signals=[t.leading_signal for t in new_transitions[-3:]],
            )
            self.history.states.append(self.current_state)

        elif self.current_state:
            # Existing regime continues
            self.current_state.duration_windows += 1
            self.current_state.stability = min(
                1.0,
                self.current_state.duration_windows / self.stability_window
            )

            # Update current_time to latest window
            latest_window = field_df.select('window_end').max()[0, 0]
            if latest_window:
                self.current_state.current_time = str(latest_window)

        # Check for early warning
        early_warning = self._check_early_warning(field_df, exclude_signals)

        return {
            'transitions_detected': new_transitions,
            'current_state': self.current_state,
            'early_warning': early_warning,
            'total_transitions': len(self.history.transitions),
        }

    def _check_early_warning(
        self,
        field_df: pl.DataFrame,
        exclude_signals: Optional[List[str]] = None,
    ) -> bool:
        """
        Check for early warning signs of impending transition.

        Looks for elevated gradient magnitude compared to baseline,
        which often precedes a transition.

        Args:
            field_df: Signal field data
            exclude_signals: Signals to exclude

        Returns:
            True if early warning signs detected
        """
        # Filter data
        data = field_df
        if exclude_signals:
            data = data.filter(~pl.col('signal_id').is_in(exclude_signals))

        # Get recent gradient magnitudes
        recent = data.sort('window_end', descending=True)

        # Aggregate by window
        grad_mags = recent.group_by('window_end').agg(
            pl.col('gradient_magnitude').mean().alias('avg_grad')
        ).sort('window_end')

        if len(grad_mags) < 10:
            return False

        grad_values = grad_mags['avg_grad'].to_numpy()

        # Compare recent (last 3) to baseline (rest)
        recent_grad = np.mean(grad_values[-3:])
        baseline_grad = np.mean(grad_values[:-3])

        if baseline_grad == 0 or np.isnan(baseline_grad):
            return False

        # Early warning if recent gradient is 2x baseline
        return recent_grad > 2 * baseline_grad

    def get_summary(self) -> Dict:
        """
        Get tracker summary statistics.

        Returns:
            Dictionary with:
                - total_transitions: Count of all transitions
                - total_states: Count of regime periods
                - current_state_id: ID of current regime
                - current_stability: Stability score of current regime
                - current_duration: Duration of current regime
                - top_leading_signals: Most frequent transition leaders
                - avg_regime_duration: Average duration between transitions
        """
        leaders = [t.leading_signal for t in self.history.transitions]
        top_leaders = [ind for ind, _ in Counter(leaders).most_common(5)]

        # Calculate average regime duration
        if len(self.history.transitions) >= 2:
            durations = []
            sorted_transitions = sorted(
                self.history.transitions,
                key=lambda t: t.window_end
            )
            for i in range(1, len(sorted_transitions)):
                # Simple count between transitions
                durations.append(1)  # Placeholder - would need actual dates
            avg_duration = np.mean(durations) if durations else 0
        else:
            avg_duration = 0

        return {
            'total_transitions': len(self.history.transitions),
            'total_states': self.state_counter,
            'current_state_id': self.current_state.state_id if self.current_state else None,
            'current_stability': self.current_state.stability if self.current_state else None,
            'current_duration': self.current_state.duration_windows if self.current_state else 0,
            'top_leading_signals': top_leaders,
            'avg_regime_duration': avg_duration,
        }

    def get_transition_timeline(self) -> pl.DataFrame:
        """
        Get timeline of all transitions.

        Returns:
            DataFrame with transition history:
                - window_end: Transition date
                - divergence_zscore: Significance score
                - leading_signal: Top responding signal
                - n_affected: Count of affected signals
        """
        if not self.history.transitions:
            return pl.DataFrame({
                'window_end': [],
                'divergence_zscore': [],
                'total_divergence': [],
                'leading_signal': [],
                'n_affected': [],
            })

        records = []
        for t in self.history.transitions:
            records.append({
                'window_end': t.window_end,
                'divergence_zscore': t.divergence_zscore,
                'total_divergence': t.total_divergence,
                'leading_signal': t.leading_signal,
                'n_affected': t.n_affected_signals,
            })

        return pl.DataFrame(records).sort('window_end')

    def reset(self):
        """Reset tracker to initial state."""
        self.history = RegimeHistory()
        self.current_state = None
        self.state_counter = 0
        self._recent_gradients = []


def track_regime_from_file(
    field_path: str,
    zscore_threshold: float = 3.0,
    exclude_signals: Optional[List[str]] = None,
) -> RegimeTracker:
    """
    Convenience function to track regime from a parquet file.

    Args:
        field_path: Path to signal_field.parquet
        zscore_threshold: Z-score threshold for transitions
        exclude_signals: Signals to exclude

    Returns:
        Fitted RegimeTracker with full history
    """
    field_df = pl.read_parquet(field_path)

    tracker = RegimeTracker(zscore_threshold=zscore_threshold)
    tracker.update(field_df, exclude_signals=exclude_signals)

    return tracker
