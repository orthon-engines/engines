"""
PRISM Tension Dynamics Engine
=============================

Computes temporal dynamics of system tension (dispersion) from geometry.structure.

Tension/dispersion represents the disagreement between timescales:
- High dispersion = scouts (63d) and anchors (252d) see different things
- Low dispersion = all timescales agree (stable structure)
- Rising dispersion = growing uncertainty, potential regime change
- Falling dispersion = convergence, stabilization

Input: geometry.structure signal topology (total_dispersion, mean_alignment)
Output: Tension dynamics metrics

Key Metrics:
    - dispersion_velocity: Rate of tension change
    - dispersion_acceleration: Second derivative
    - tension_state: 'building', 'releasing', 'stable'
    - alignment dynamics

Usage:
    from orthon._internal.engines.state.tension_dynamics import TensionDynamicsEngine

    engine = TensionDynamicsEngine()
    result = engine.run(structure_df)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TensionDynamicsResult:
    """Result from tension dynamics computation."""
    dispersion_total: float
    dispersion_velocity: float
    dispersion_acceleration: float
    alignment_mean: float
    alignment_velocity: float
    tension_state: str  # 'building', 'releasing', 'stable'
    coherence: float
    metrics: Dict[str, float]


class TensionDynamicsEngine:
    """
    Compute temporal dynamics of system tension.

    Tension is derived from geometry.structure.total_dispersion which measures
    the average disagreement between timescales across all signals.

    Tension building = system becoming unstable, regime shift possible
    Tension releasing = system converging, stability returning
    """

    def __init__(self, velocity_window: int = 5, accel_window: int = 3):
        """
        Initialize engine.

        Args:
            velocity_window: Window for velocity calculation
            accel_window: Window for acceleration smoothing
        """
        self.velocity_window = velocity_window
        self.accel_window = accel_window

    def run(self, dispersion_series: pd.Series, alignment_series: pd.Series,
            coherence_series: Optional[pd.Series] = None,
            current_idx: int = -1) -> TensionDynamicsResult:
        """
        Compute tension dynamics from dispersion/alignment series.

        Args:
            dispersion_series: Series of total_dispersion values
            alignment_series: Series of mean_alignment values
            coherence_series: Optional series of system_coherence values
            current_idx: Index position to compute dynamics for

        Returns:
            TensionDynamicsResult with all computed metrics
        """
        if len(dispersion_series) < 2:
            return TensionDynamicsResult(
                dispersion_total=dispersion_series.iloc[current_idx] if len(dispersion_series) > 0 else 0,
                dispersion_velocity=0.0,
                dispersion_acceleration=0.0,
                alignment_mean=alignment_series.iloc[current_idx] if len(alignment_series) > 0 else 0,
                alignment_velocity=0.0,
                tension_state='stable',
                coherence=coherence_series.iloc[current_idx] if coherence_series is not None and len(coherence_series) > 0 else 0,
                metrics={}
            )

        # Current values
        current_dispersion = dispersion_series.iloc[current_idx]
        current_alignment = alignment_series.iloc[current_idx]
        current_coherence = coherence_series.iloc[current_idx] if coherence_series is not None else 0.0

        # Dispersion velocity (first derivative)
        disp_diff = dispersion_series.diff()
        disp_velocity = disp_diff.rolling(window=self.velocity_window, min_periods=1).mean()
        dispersion_velocity = disp_velocity.iloc[current_idx] if not pd.isna(disp_velocity.iloc[current_idx]) else 0.0

        # Dispersion acceleration (second derivative)
        disp_accel = disp_diff.diff()
        accel_smooth = disp_accel.rolling(window=self.accel_window, min_periods=1).mean()
        dispersion_acceleration = accel_smooth.iloc[current_idx] if not pd.isna(accel_smooth.iloc[current_idx]) else 0.0

        # Alignment velocity
        align_diff = alignment_series.diff()
        align_velocity = align_diff.rolling(window=self.velocity_window, min_periods=1).mean()
        alignment_velocity = align_velocity.iloc[current_idx] if not pd.isna(align_velocity.iloc[current_idx]) else 0.0

        # Determine tension state
        if dispersion_velocity > 0.001 and current_dispersion > dispersion_series.mean():
            tension_state = 'building'
        elif dispersion_velocity < -0.001 or current_dispersion < dispersion_series.mean() * 0.8:
            tension_state = 'releasing'
        else:
            tension_state = 'stable'

        # Additional metrics
        metrics = {
            'dispersion_ma5': dispersion_series.rolling(5, min_periods=1).mean().iloc[current_idx],
            'dispersion_std': dispersion_series.std() if len(dispersion_series) >= 5 else 0.0,
            'dispersion_percentile': (dispersion_series < current_dispersion).mean() * 100 if len(dispersion_series) > 1 else 50.0,
            'alignment_ma5': alignment_series.rolling(5, min_periods=1).mean().iloc[current_idx],
            'tension_energy_ratio': current_dispersion / (current_alignment + 1e-9),
            'stability_score': current_alignment * (1.0 / (1.0 + abs(dispersion_velocity)))
        }

        return TensionDynamicsResult(
            dispersion_total=float(current_dispersion),
            dispersion_velocity=float(dispersion_velocity),
            dispersion_acceleration=float(dispersion_acceleration),
            alignment_mean=float(current_alignment),
            alignment_velocity=float(alignment_velocity),
            tension_state=tension_state,
            coherence=float(current_coherence),
            metrics=metrics
        )

    def compute_series(self, dispersion_series: pd.Series, alignment_series: pd.Series,
                       coherence_series: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Compute tension dynamics for entire series.

        Returns DataFrame with dynamics metrics for each time point.
        """
        results = []

        for i in range(len(dispersion_series)):
            disp_subset = dispersion_series.iloc[:i+1]
            align_subset = alignment_series.iloc[:i+1]
            coh_subset = coherence_series.iloc[:i+1] if coherence_series is not None else None

            result = self.run(disp_subset, align_subset, coh_subset, current_idx=-1)

            results.append({
                'date': dispersion_series.index[i],
                'dispersion_total': result.dispersion_total,
                'dispersion_velocity': result.dispersion_velocity,
                'dispersion_acceleration': result.dispersion_acceleration,
                'alignment_mean': result.alignment_mean,
                'alignment_velocity': result.alignment_velocity,
                'tension_state': result.tension_state,
                'coherence': result.coherence
            })

        return pd.DataFrame(results)


def compute_tension_dynamics(
    dispersion_values: np.ndarray,
    alignment_values: np.ndarray,
    coherence_values: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Functional interface for tension dynamics computation.

    Args:
        dispersion_values: Array of dispersion values
        alignment_values: Array of alignment values
        coherence_values: Optional array of coherence values

    Returns:
        Dict of tension dynamics metrics
    """
    engine = TensionDynamicsEngine()
    disp_series = pd.Series(dispersion_values)
    align_series = pd.Series(alignment_values)
    coh_series = pd.Series(coherence_values) if coherence_values is not None else None

    result = engine.run(disp_series, align_series, coh_series)

    return {
        'dispersion_total': result.dispersion_total,
        'dispersion_velocity': result.dispersion_velocity,
        'dispersion_acceleration': result.dispersion_acceleration,
        'alignment_mean': result.alignment_mean,
        'alignment_velocity': result.alignment_velocity,
        'tension_state': result.tension_state,
        'coherence': result.coherence,
        **result.metrics
    }
