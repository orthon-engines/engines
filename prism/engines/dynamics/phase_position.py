"""
Phase Position Engine

Tracks position and velocity on reconstructed attractor.

For each window:
- Project state onto attractor
- Compute phase (if cyclic) or coordinates
- Compute velocity
- Compute deviation from attractor

Philosophy: Where are we on the hidden structure?
"""

import numpy as np
from typing import Dict, Any, Optional

try:
    from sklearn.neighbors import BallTree
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class PhaseTracker:
    """
    Track position and velocity on reconstructed attractor.

    For each window:
    - Project state onto attractor
    - Compute phase (if cyclic) or coordinates
    - Compute velocity
    - Compute deviation from attractor
    """

    def __init__(self, attractor_reconstructor):
        """
        Initialize phase tracker from fitted attractor reconstructor.

        Parameters
        ----------
        attractor_reconstructor : AttractorReconstructor
            Fitted attractor reconstructor with baseline_embedded
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for phase tracking")

        self.ar = attractor_reconstructor
        self.baseline_embedded = self.ar.baseline_embedded

        if len(self.baseline_embedded) == 0:
            self.baseline_centroid = np.array([0])
            self.tree = None
            self.is_cyclic = False
            return

        self.baseline_centroid = np.mean(self.baseline_embedded, axis=0)

        # Build KD-tree for fast nearest neighbor queries
        self.tree = BallTree(self.baseline_embedded)

        # Detect if cyclic (limit cycle)
        self.is_cyclic = self.ar.attractor_type in ['limit_cycle_stable', 'limit_cycle_unstable']

        if self.is_cyclic:
            self._compute_cycle_structure()
        else:
            self.pca = None
            self.cycle_period = np.nan

    def _compute_cycle_structure(self):
        """
        For limit cycles, parameterize by angle from centroid.
        """
        if len(self.baseline_embedded) < 3:
            self.is_cyclic = False
            self.pca = None
            self.cycle_period = np.nan
            return

        # Project to first 2 principal components for phase angle
        n_components = min(2, self.baseline_embedded.shape[1], len(self.baseline_embedded))
        self.pca = PCA(n_components=n_components)
        projected = self.pca.fit_transform(self.baseline_embedded)

        if projected.shape[1] < 2:
            self.is_cyclic = False
            self.cycle_period = np.nan
            return

        # Compute angles
        self.baseline_angles = np.arctan2(projected[:, 1], projected[:, 0])

        # Sort to get canonical cycle order
        self.cycle_order = np.argsort(self.baseline_angles)

        # Compute cycle period (in samples)
        # Count zero-crossings of first PC
        pc1 = projected[:, 0]
        zero_crossings = np.where(np.diff(np.sign(pc1)))[0]
        if len(zero_crossings) >= 2:
            self.cycle_period = np.mean(np.diff(zero_crossings)) * 2
        else:
            self.cycle_period = len(self.baseline_embedded)

    def compute_phase(self, window_embedded: np.ndarray) -> Dict[str, Any]:
        """
        Compute phase position for a window's embedded state.

        Parameters
        ----------
        window_embedded : array
            Embedded trajectory for one window, shape (n, dim)

        Returns
        -------
        dict
            Phase metrics including position, velocity, deviation
        """
        if len(window_embedded) == 0 or self.tree is None:
            return self._empty_result()

        # Ensure 2D
        if window_embedded.ndim == 1:
            window_embedded = window_embedded.reshape(1, -1)

        # Use window centroid for global position
        window_centroid = np.mean(window_embedded, axis=0)

        # Distance to attractor (nearest point on baseline)
        dist_to_attractor, nn_idx = self.tree.query([window_centroid], k=1)
        dist_to_attractor = dist_to_attractor[0, 0]
        nn_idx = nn_idx[0, 0]

        # Distance from baseline centroid
        dist_from_centroid = np.linalg.norm(window_centroid - self.baseline_centroid)

        result = {
            'attractor_deviation': dist_to_attractor,
            'centroid_distance': dist_from_centroid,
            'nearest_baseline_idx': int(nn_idx),
        }

        if self.is_cyclic and self.pca is not None:
            # Compute phase angle
            projected = self.pca.transform([window_centroid])[0]
            phase_angle = np.arctan2(projected[1], projected[0])

            # Convert to 0-1 range
            phase = (phase_angle + np.pi) / (2 * np.pi)

            # Compute phase velocity from window trajectory
            if len(window_embedded) >= 3:
                projected_traj = self.pca.transform(window_embedded)
                angles = np.arctan2(projected_traj[:, 1], projected_traj[:, 0])
                # Unwrap angles to handle wraparound
                angles_unwrapped = np.unwrap(angles)
                phase_velocity = np.mean(np.diff(angles_unwrapped))
            else:
                phase_velocity = np.nan

            result['phase'] = phase
            result['phase_angle'] = phase_angle
            result['phase_velocity'] = phase_velocity
            result['cycle_period'] = self.cycle_period
        else:
            # For non-cyclic, just report coordinates
            result['phase'] = np.nan
            result['phase_angle'] = np.nan
            result['phase_velocity'] = np.nan
            result['cycle_period'] = np.nan

            # Coordinates in embedded space (relative to centroid)
            coords = window_centroid - self.baseline_centroid
            for i, c in enumerate(coords[:5]):  # First 5 dimensions
                result[f'coord_{i}'] = c

        return result

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result dict."""
        return {
            'attractor_deviation': np.nan,
            'centroid_distance': np.nan,
            'nearest_baseline_idx': -1,
            'phase': np.nan,
            'phase_angle': np.nan,
            'phase_velocity': np.nan,
            'cycle_period': np.nan,
        }


def compute(
    window_embedded: np.ndarray,
    baseline_embedded: np.ndarray,
    attractor_type: str = "unknown"
) -> Dict[str, Any]:
    """
    Compute phase position for a window relative to baseline attractor.

    Parameters
    ----------
    window_embedded : array
        Embedded state for current window
    baseline_embedded : array
        Embedded baseline attractor
    attractor_type : str
        Type of attractor (for cycle detection)

    Returns
    -------
    dict
        Phase metrics
    """
    if not HAS_SKLEARN:
        return {
            'attractor_deviation': np.nan,
            'centroid_distance': np.nan,
            'nearest_baseline_idx': -1,
            'phase': np.nan,
            'phase_angle': np.nan,
            'phase_velocity': np.nan,
        }

    # Create minimal attractor reconstructor stub
    class ARStub:
        pass

    ar = ARStub()
    ar.baseline_embedded = baseline_embedded
    ar.attractor_type = attractor_type

    tracker = PhaseTracker(ar)
    return tracker.compute_phase(window_embedded)
