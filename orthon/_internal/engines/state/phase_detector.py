"""
PRISM Phase Detector Engine
===========================

Detects regime transitions and classifies system phases from geometry dynamics.

A regime shift is confirmed when:
1. anchor_ratio > 1.5 (anchors moved more than scouts)
2. regime_conviction is elevated (high energy into tight structure)
3. Multiple confirmation signals align

Phase Classification:
    - accumulation: Low energy, building tension, low conviction
    - expansion: Rising energy, releasing tension, moderate conviction
    - distribution: High energy, high tension, declining conviction
    - contraction: Falling energy, falling tension, high conviction

Input: Energy and tension dynamics
Output: Phase labels, regime shift detection, confidence scores

Usage:
    from orthon._internal.engines.state.phase_detector import PhaseDetectorEngine

    engine = PhaseDetectorEngine()
    result = engine.run(state_df)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class PhaseDetectionResult:
    """Result from phase detection."""
    phase_label: str
    phase_score: float  # -1 (contraction) to +1 (expansion)
    is_regime_shift: bool
    shift_confidence: float
    regime_conviction: float
    anchor_ratio: float
    signals: Dict[str, Any]


class PhaseDetectorEngine:
    """
    Detect regime transitions and classify system phases.

    Uses multiple signals to identify structural shifts:
    - Energy dynamics (acceleration, z-score)
    - Tension dynamics (velocity, state)
    - Structural metrics (anchor_ratio, regime_conviction)

    Phase classification is based on the combination of:
    - Energy level and trend
    - Tension level and trend
    - Structural conviction
    """

    def __init__(
        self,
        anchor_threshold: float = 1.5,
        conviction_threshold: float = 100,
        energy_zscore_threshold: float = 2.0,
        min_confirmation: int = 2
    ):
        """
        Initialize detector.

        Args:
            anchor_threshold: Minimum anchor_ratio for regime shift
            conviction_threshold: Minimum regime_conviction for shift
            energy_zscore_threshold: Energy z-score for anomaly detection
            min_confirmation: Minimum confirming signals for regime shift
        """
        self.anchor_threshold = anchor_threshold
        self.conviction_threshold = conviction_threshold
        self.energy_zscore_threshold = energy_zscore_threshold
        self.min_confirmation = min_confirmation

    def run(
        self,
        energy_total: float,
        energy_zscore: float,
        energy_trend: str,
        dispersion_total: float,
        tension_state: str,
        alignment: float,
        anchor_ratio: float,
        regime_conviction: float,
        historical_phases: Optional[List[str]] = None
    ) -> PhaseDetectionResult:
        """
        Detect phase and regime shift for current state.

        Args:
            energy_total: Current total energy
            energy_zscore: Energy z-score vs history
            energy_trend: 'rising', 'falling', 'stable'
            dispersion_total: Current system dispersion
            tension_state: 'building', 'releasing', 'stable'
            alignment: Current system alignment
            anchor_ratio: 252d energy / 63d energy
            regime_conviction: energy / dispersion
            historical_phases: Previous phase labels for context

        Returns:
            PhaseDetectionResult
        """
        signals = {}

        # Signal 1: Anchor ratio (structural shift signal)
        anchor_signal = anchor_ratio > self.anchor_threshold
        signals['anchor_signal'] = anchor_signal

        # Signal 2: Conviction level
        conviction_signal = regime_conviction > self.conviction_threshold
        signals['conviction_signal'] = conviction_signal

        # Signal 3: Energy anomaly
        energy_anomaly = abs(energy_zscore) > self.energy_zscore_threshold
        signals['energy_anomaly'] = energy_anomaly

        # Signal 4: Tension building
        tension_building = tension_state == 'building'
        signals['tension_building'] = tension_building

        # Signal 5: Alignment shift (drop in coherence)
        low_alignment = alignment < 0.3
        signals['low_alignment'] = low_alignment

        # Count confirming signals
        regime_shift_signals = [anchor_signal, conviction_signal, energy_anomaly]
        n_confirmations = sum(regime_shift_signals)

        # Regime shift detection
        is_regime_shift = n_confirmations >= self.min_confirmation
        shift_confidence = n_confirmations / len(regime_shift_signals)

        # Phase classification
        phase_label, phase_score = self._classify_phase(
            energy_zscore, energy_trend, tension_state, alignment, regime_conviction
        )

        return PhaseDetectionResult(
            phase_label=phase_label,
            phase_score=phase_score,
            is_regime_shift=is_regime_shift,
            shift_confidence=shift_confidence,
            regime_conviction=regime_conviction,
            anchor_ratio=anchor_ratio,
            signals=signals
        )

    def _classify_phase(
        self,
        energy_zscore: float,
        energy_trend: str,
        tension_state: str,
        alignment: float,
        conviction: float
    ) -> Tuple[str, float]:
        """
        Classify current phase based on state metrics.

        Returns (phase_label, phase_score)
        """
        # Phase score: -1 (contraction) to +1 (expansion)
        # Based on energy trend and tension state

        # Base score from energy
        if energy_trend == 'rising':
            base_score = 0.3
        elif energy_trend == 'falling':
            base_score = -0.3
        else:
            base_score = 0.0

        # Adjust for tension
        if tension_state == 'building':
            base_score += 0.2  # Tension building often precedes expansion
        elif tension_state == 'releasing':
            base_score -= 0.2  # Tension releasing can be post-peak

        # Adjust for energy level
        if energy_zscore > 1.5:
            base_score += 0.3
        elif energy_zscore < -1.5:
            base_score -= 0.3

        # Adjust for alignment
        if alignment < 0.3:
            # Low alignment = uncertainty, often transitional
            base_score *= 0.5

        # Clamp to [-1, 1]
        phase_score = max(-1.0, min(1.0, base_score))

        # Determine phase label
        if phase_score > 0.5:
            phase_label = 'expansion'
        elif phase_score > 0:
            phase_label = 'accumulation'
        elif phase_score > -0.5:
            phase_label = 'distribution'
        else:
            phase_label = 'contraction'

        # Special case: high conviction + high energy + falling alignment
        # This is often a distribution phase before correction
        if conviction > 100 and energy_zscore > 1.0 and alignment < 0.4 and energy_trend != 'rising':
            phase_label = 'distribution'
            phase_score = max(-0.5, min(0, phase_score))

        return phase_label, phase_score

    def detect_transitions(
        self,
        phases_series: pd.Series,
        conviction_series: pd.Series,
        anchor_ratio_series: pd.Series
    ) -> List[Dict[str, Any]]:
        """
        Detect phase transitions in a signal topology.

        Args:
            phases_series: Series of phase labels
            conviction_series: Series of regime_conviction values
            anchor_ratio_series: Series of anchor_ratio values

        Returns:
            List of detected transitions with timing and metrics
        """
        transitions = []

        prev_phase = None
        for i, (idx, phase) in enumerate(phases_series.items()):
            if prev_phase is not None and phase != prev_phase:
                transitions.append({
                    'date': idx,
                    'from_phase': prev_phase,
                    'to_phase': phase,
                    'conviction': conviction_series.iloc[i] if i < len(conviction_series) else 0,
                    'anchor_ratio': anchor_ratio_series.iloc[i] if i < len(anchor_ratio_series) else 0
                })
            prev_phase = phase

        return transitions


def detect_phase(
    energy_total: float,
    energy_zscore: float,
    energy_trend: str,
    dispersion_total: float,
    tension_state: str,
    alignment: float,
    anchor_ratio: float,
    regime_conviction: float
) -> Dict[str, Any]:
    """
    Functional interface for phase detection.

    Returns dict with phase classification and regime shift detection.
    """
    engine = PhaseDetectorEngine()
    result = engine.run(
        energy_total, energy_zscore, energy_trend,
        dispersion_total, tension_state, alignment,
        anchor_ratio, regime_conviction
    )

    return {
        'phase_label': result.phase_label,
        'phase_score': result.phase_score,
        'is_regime_shift': result.is_regime_shift,
        'shift_confidence': result.shift_confidence,
        'regime_conviction': result.regime_conviction,
        'anchor_ratio': result.anchor_ratio,
        **result.signals
    }
