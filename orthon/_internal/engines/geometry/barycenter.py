"""
PRISM Barycenter Engine
=======================

Conviction-weighted center of mass computation across timescales.

"When five year olds run around - normal noise. When adults start running - regime change."

Core Concept:
    - Short windows (63d/21d) are scouts - noisy, early warning
    - Medium windows (126d) are bridges - confirmation
    - Long windows (252d) are anchors - structural truth

Weights are loaded from config/stride.yaml (default: 21d=0.5, 63d=1.0, 126d=2.0, 252d=4.0)

Computes per signal:
    - barycenter: conviction-weighted center of mass across timescales
    - dispersion: tension between shortest and longest window (scout vs anchor)
    - alignment: coherence across all windows (1 = perfect agreement)

Returns cohort-level aggregates for storage in geometry.cohorts.

Phase: Derived (geometry layer)
Normalization: Z-score preferred
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import date

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

from orthon._internal.engines.engine_base import BaseEngine, get_window_dates
from orthon._internal.engines.metadata import EngineMetadata

logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="barycenter",
    engine_type="geometry",
    description="Conviction-weighted center of mass across timescales",
    domains={"structure", "temporal_coherence"},
    requires_window=True,
    deterministic=True,
)


def _load_weights() -> Dict[int, float]:
    """Load barycenter weights from config. Fails if not configured."""
    try:
        from orthon._internal.utils.stride import get_barycenter_weights
        weights = get_barycenter_weights()
        if weights:
            return weights
    except Exception as e:
        raise RuntimeError(f"Failed to load barycenter weights: {e}")

    raise RuntimeError(
        "No barycenter weights configured in config/stride.yaml. "
        "Configure domain-specific window weights before running."
    )


def compute_signal_barycenter(
    vectors: Dict[int, np.ndarray],
    weights: Optional[Dict[int, float]] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    Calculate conviction-weighted barycenter and tension metrics for an signal.

    Args:
        vectors: Dict mapping window_days -> feature vector
        weights: Optional weights per window (loads from config if not provided)

    Returns:
        (barycenter, dispersion, alignment)
        - barycenter: weighted center of mass
        - dispersion: distance between shortest and longest window
        - alignment: coherence across all windows (0-1)
    """
    if weights is None:
        weights = _load_weights()

    # Find available windows that have both vectors and weights
    available_windows = sorted([w for w in vectors.keys() if w in weights])

    if len(available_windows) < 2:
        # Need at least 2 windows for meaningful barycenter
        return None, None, None

    # 1. Weighted Barycenter (Center of Mass)
    total_weight = sum(weights[w] for w in available_windows)
    weighted_sum = np.zeros_like(vectors[available_windows[0]])

    for window_days in available_windows:
        weighted_sum += vectors[window_days] * weights[window_days]

    barycenter = weighted_sum / total_weight

    # 2. Timescale Dispersion (Tension between scouts and anchors)
    # Distance between shortest and longest available windows
    shortest = min(available_windows)
    longest = max(available_windows)
    dispersion = euclidean(vectors[shortest], vectors[longest])

    # 3. Timescale Alignment (How coherent are all windows?)
    # Low variance in distances to barycenter = high alignment
    distances = [euclidean(vectors[w], barycenter) for w in available_windows]
    variance = np.var(distances) if len(distances) > 1 else 0.0
    alignment = 1.0 / (1.0 + variance)

    return barycenter, float(dispersion), float(alignment)


class BarycenterEngine(BaseEngine):
    """
    Barycenter engine for conviction-weighted structural analysis.

    Computes the center of mass across multiple timescales, where longer
    windows carry more weight (conviction). This captures the intuition that
    structural changes in 252-day windows are more significant than 63-day
    window noise.

    Outputs:
        - Cohort-level: mean_dispersion, mean_alignment, n_computed
        - Per-signal: barycenter vector, dispersion, alignment
    """

    name = "barycenter"
    phase = "derived"
    default_normalization = "zscore"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        window_vectors: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        weights: Optional[Dict[int, float]] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Run barycenter analysis.

        Args:
            df: Normalized signal data (rows=dates, cols=signals)
                This is the "current" window's data matrix.
            run_id: Unique run identifier
            window_vectors: Dict mapping signal_id -> {window_days: vector}
                Pre-computed vectors for each signal at each window size.
                If not provided, only returns empty metrics.
            weights: Optional custom weights (loads from config if not provided)

        Returns:
            Dict with cohort-level aggregates:
                - mean_dispersion: Average tension across signals
                - mean_alignment: Average coherence across signals
                - n_computed: Number of signals with valid barycenters
                - per_signal: Dict of per-signal results (for storage)
        """
        if weights is None:
            weights = _load_weights()

        if window_vectors is None:
            logger.warning("No window_vectors provided - barycenter requires multi-window data")
            return {
                'mean_dispersion': None,
                'mean_alignment': None,
                'n_computed': 0,
                'per_signal': {},
            }

        # Compute barycenter for each signal
        dispersions = []
        alignments = []
        per_signal = {}

        for signal_id, vectors in window_vectors.items():
            barycenter, dispersion, alignment = compute_signal_barycenter(
                vectors, weights
            )

            if barycenter is not None:
                dispersions.append(dispersion)
                alignments.append(alignment)
                per_signal[signal_id] = {
                    'barycenter': barycenter,
                    'dispersion': dispersion,
                    'alignment': alignment,
                    'available_windows': sorted(vectors.keys()),
                }

        n_computed = len(per_signal)

        if n_computed == 0:
            logger.warning("No signals had sufficient windows for barycenter computation")
            return {
                'mean_dispersion': None,
                'mean_alignment': None,
                'n_computed': 0,
                'per_signal': {},
            }

        # Compute cohort-level aggregates
        mean_dispersion = float(np.mean(dispersions))
        mean_alignment = float(np.mean(alignments))

        logger.info(
            f"Barycenter complete: {n_computed} signals, "
            f"mean_dispersion={mean_dispersion:.3f}, mean_alignment={mean_alignment:.3f}"
        )

        return {
            'mean_dispersion': mean_dispersion,
            'mean_alignment': mean_alignment,
            'n_computed': n_computed,
            'per_signal': per_signal,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_barycenter(
    window_vectors: Dict[str, Dict[int, np.ndarray]],
    weights: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Compute barycenters for a set of signals.

    Convenience function that wraps BarycenterEngine.

    Args:
        window_vectors: Dict mapping signal_id -> {window_days: vector}
        weights: Optional custom weights

    Returns:
        Dict with mean_dispersion, mean_alignment, n_computed, per_signal
    """
    engine = BarycenterEngine()
    return engine.run(
        df=pd.DataFrame(),  # Not used when window_vectors provided
        run_id="compute_barycenter",
        window_vectors=window_vectors,
        weights=weights,
    )
