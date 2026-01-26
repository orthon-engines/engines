"""
Basin Analysis Engine

Determines basin membership and transition probability.

Approach:
1. The baseline attractor defines "basin 0" (healthy/nominal)
2. Track how close system stays to baseline attractor
3. Detect when system escapes basin 0 (deviation exceeds threshold)
4. Characterize new basin (if any)

Philosophy: Which attractor basin captures us?
"""

import numpy as np
from typing import Dict, Any, List, Optional

try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class BasinAnalyzer:
    """
    Determine basin membership and transition probability.

    Approach:
    1. The baseline attractor defines "basin 0" (healthy/nominal)
    2. Track how close system stays to baseline attractor
    3. Detect when system escapes basin 0 (deviation exceeds threshold)
    4. Characterize new basin (if any)
    """

    def __init__(self, attractor_reconstructor, escape_sigma: float = 3.0):
        """
        Initialize basin analyzer.

        Parameters
        ----------
        attractor_reconstructor : AttractorReconstructor
            Fitted attractor reconstructor
        escape_sigma : float
            Number of standard deviations from baseline to trigger escape
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for basin analysis")

        self.ar = attractor_reconstructor
        self.baseline_embedded = self.ar.baseline_embedded
        self.escape_sigma = escape_sigma

        if len(self.baseline_embedded) == 0:
            self.tree = None
            self.baseline_centroid = np.array([0])
            self.baseline_centroid_dist_mean = 0
            self.baseline_centroid_dist_std = 1
            self.baseline_nn_dist_mean = 1
            self.baseline_nn_dist_std = 1
            self.escape_threshold = np.inf
            return

        # Build KD-tree for baseline attractor
        self.tree = BallTree(self.baseline_embedded)

        # Compute baseline statistics for thresholding
        self._compute_baseline_stats()

    def _compute_baseline_stats(self):
        """
        Compute statistics of distances within baseline attractor.
        """
        # Distances from each baseline point to attractor centroid
        self.baseline_centroid = np.mean(self.baseline_embedded, axis=0)
        dists_to_centroid = np.linalg.norm(
            self.baseline_embedded - self.baseline_centroid, axis=1
        )

        # Distances to nearest neighbor on attractor
        dists, _ = self.tree.query(self.baseline_embedded, k=2)  # k=2 because nearest is self
        nn_dists = dists[:, 1]

        self.baseline_centroid_dist_mean = np.mean(dists_to_centroid)
        self.baseline_centroid_dist_std = np.std(dists_to_centroid)
        if self.baseline_centroid_dist_std < 1e-10:
            self.baseline_centroid_dist_std = 1e-10

        self.baseline_nn_dist_mean = np.mean(nn_dists)
        self.baseline_nn_dist_std = np.std(nn_dists)
        if self.baseline_nn_dist_std < 1e-10:
            self.baseline_nn_dist_std = 1e-10

        # Escape threshold: N sigma from baseline behavior
        self.escape_threshold = (
            self.baseline_centroid_dist_mean +
            self.escape_sigma * self.baseline_centroid_dist_std
        )

    def analyze_window(self, window_embedded: np.ndarray) -> Dict[str, Any]:
        """
        Analyze basin membership for a window.

        Parameters
        ----------
        window_embedded : array
            Embedded state for window

        Returns
        -------
        dict
            Basin metrics including membership, stability, transition probability
        """
        if len(window_embedded) == 0 or self.tree is None:
            return self._empty_result()

        # Ensure 2D
        if window_embedded.ndim == 1:
            window_embedded = window_embedded.reshape(1, -1)

        window_centroid = np.mean(window_embedded, axis=0)

        # Distance from baseline centroid
        dist_from_baseline = np.linalg.norm(window_centroid - self.baseline_centroid)

        # Distance to nearest point on baseline attractor
        dist_to_attractor, nn_idx = self.tree.query([window_centroid], k=1)
        dist_to_attractor = dist_to_attractor[0, 0]

        # Normalized distance (in units of baseline std)
        normalized_dist = (
            (dist_from_baseline - self.baseline_centroid_dist_mean) /
            self.baseline_centroid_dist_std
        )

        # Basin membership
        if dist_from_baseline < self.escape_threshold:
            basin = 0  # Still in baseline basin
        else:
            basin = 1  # Escaped to new basin

        # Basin stability: how deep in basin (negative = deep, positive = near boundary)
        stability = (self.escape_threshold - dist_from_baseline) / self.escape_threshold

        # Transition probability: based on distance to boundary
        boundary_distance = self.escape_threshold - dist_from_baseline
        transition_prob = 1 / (1 + np.exp(boundary_distance / self.baseline_centroid_dist_std))

        # Local density (how many baseline points nearby)
        radius = self.baseline_nn_dist_mean * 3
        n_nearby = self.tree.query_radius([window_centroid], r=radius, count_only=True)[0]
        local_density = n_nearby / len(self.baseline_embedded)

        result = {
            'basin': basin,
            'basin_stability': stability,
            'transition_prob': transition_prob,
            'dist_from_baseline_centroid': dist_from_baseline,
            'dist_to_baseline_attractor': dist_to_attractor,
            'normalized_distance': normalized_dist,
            'local_density': local_density,
            'escape_threshold': self.escape_threshold,
        }

        return result

    def detect_transitions(self, basin_sequence: List[int]) -> List[Dict[str, Any]]:
        """
        Find basin transitions in a sequence.

        Parameters
        ----------
        basin_sequence : list
            List of basin assignments per window

        Returns
        -------
        list of dict
            List of transitions with window, from_basin, to_basin
        """
        transitions = []
        for i in range(1, len(basin_sequence)):
            if basin_sequence[i] != basin_sequence[i-1]:
                transitions.append({
                    'window': i,
                    'from_basin': basin_sequence[i-1],
                    'to_basin': basin_sequence[i],
                })
        return transitions

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result dict."""
        return {
            'basin': -1,
            'basin_stability': np.nan,
            'transition_prob': np.nan,
            'dist_from_baseline_centroid': np.nan,
            'dist_to_baseline_attractor': np.nan,
            'normalized_distance': np.nan,
            'local_density': np.nan,
            'escape_threshold': np.nan,
        }


def compute(
    window_embedded: np.ndarray,
    baseline_embedded: np.ndarray,
    escape_sigma: float = 3.0
) -> Dict[str, Any]:
    """
    Analyze basin membership for a window.

    Parameters
    ----------
    window_embedded : array
        Embedded state for current window
    baseline_embedded : array
        Embedded baseline attractor
    escape_sigma : float
        Number of std devs for escape threshold

    Returns
    -------
    dict
        Basin metrics
    """
    if not HAS_SKLEARN:
        return {
            'basin': -1,
            'basin_stability': np.nan,
            'transition_prob': np.nan,
            'dist_from_baseline_centroid': np.nan,
            'dist_to_baseline_attractor': np.nan,
            'normalized_distance': np.nan,
            'local_density': np.nan,
        }

    # Create minimal attractor reconstructor stub
    class ARStub:
        pass

    ar = ARStub()
    ar.baseline_embedded = baseline_embedded

    analyzer = BasinAnalyzer(ar, escape_sigma=escape_sigma)
    return analyzer.analyze_window(window_embedded)


def detect_basin_transitions(basin_sequence: List[int]) -> List[Dict[str, Any]]:
    """
    Find basin transitions in a sequence.

    Parameters
    ----------
    basin_sequence : list
        List of basin IDs per window

    Returns
    -------
    list of dict
        Transitions with window index, from_basin, to_basin
    """
    transitions = []
    for i in range(1, len(basin_sequence)):
        if basin_sequence[i] != basin_sequence[i-1]:
            transitions.append({
                'window': i,
                'from_basin': basin_sequence[i-1],
                'to_basin': basin_sequence[i],
            })
    return transitions
