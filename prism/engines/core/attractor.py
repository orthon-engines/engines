"""
Attractor Reconstruction Engine

Discovers hidden dynamical structure using Takens embedding.

Computes:
- Optimal embedding dimension (false nearest neighbors)
- Optimal delay (mutual information)
- Correlation dimension (attractor complexity)
- Largest Lyapunov exponent (stability/chaos)
- Attractor classification

Philosophy: Compute once, melt the mac, query forever.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.spatial.distance import pdist
from scipy.stats import linregress

try:
    from sklearn.neighbors import NearestNeighbors, BallTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class AttractorReconstructor:
    """
    Full Takens embedding and attractor characterization.

    Computes:
    - Optimal embedding dimension (false nearest neighbors)
    - Optimal delay (mutual information)
    - Correlation dimension (attractor complexity)
    - Largest Lyapunov exponent (stability/chaos)
    - Recurrence structure
    """

    def __init__(self, max_dim: int = 10, max_delay: int = 100):
        self.max_dim = max_dim
        self.max_delay = max_delay

        # Results
        self.embedding_dim = None
        self.delay = None
        self.correlation_dim = None
        self.lyapunov_exp = None
        self.attractor_type = None
        self.embedded_trajectory = None
        self.baseline_embedded = None

    def fit(self, signal: np.ndarray, baseline_windows: Optional[int] = None) -> 'AttractorReconstructor':
        """
        Reconstruct attractor from signal.

        Args:
            signal: 1D array of observations
            baseline_windows: If provided, learn attractor from these windows only
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for attractor reconstruction")

        signal = np.asarray(signal).flatten()

        if len(signal) < 50:
            self._set_insufficient_data()
            return self

        if baseline_windows is not None:
            baseline_signal = signal[:baseline_windows]
        else:
            baseline_signal = signal

        if len(baseline_signal) < 50:
            baseline_signal = signal

        # Step 1: Find optimal delay via mutual information
        self.delay = self._optimal_delay(baseline_signal)

        # Step 2: Find optimal embedding dimension via false nearest neighbors
        self.embedding_dim = self._optimal_embedding_dim(baseline_signal, self.delay)

        # Step 3: Create full embedded trajectory
        self.embedded_trajectory = self._embed(signal, self.embedding_dim, self.delay)
        self.baseline_embedded = self._embed(baseline_signal, self.embedding_dim, self.delay)

        if len(self.baseline_embedded) < 20:
            self._set_insufficient_data()
            return self

        # Step 4: Compute correlation dimension
        self.correlation_dim = self._correlation_dimension(self.baseline_embedded)

        # Step 5: Compute largest Lyapunov exponent
        self.lyapunov_exp = self._lyapunov_exponent(self.baseline_embedded)

        # Step 6: Classify attractor type
        self.attractor_type = self._classify_attractor()

        return self

    def _set_insufficient_data(self):
        """Set NaN values when insufficient data."""
        self.embedding_dim = 1
        self.delay = 1
        self.correlation_dim = np.nan
        self.lyapunov_exp = np.nan
        self.attractor_type = "insufficient_data"
        self.embedded_trajectory = np.array([]).reshape(0, 1)
        self.baseline_embedded = np.array([]).reshape(0, 1)

    def _optimal_delay(self, signal: np.ndarray) -> int:
        """
        Find optimal delay using first minimum of mutual information.
        """
        n = len(signal)
        mi_values = []

        for delay in range(1, min(self.max_delay, n // 4)):
            # Compute mutual information via histogram method
            x = signal[:-delay]
            y = signal[delay:]

            # 2D histogram
            bins = max(10, int(np.sqrt(len(x) / 5)))
            hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
            hist_sum = hist_2d.sum()
            if hist_sum == 0:
                continue
            hist_2d = hist_2d / hist_sum

            # Marginals
            px = hist_2d.sum(axis=1)
            py = hist_2d.sum(axis=0)

            # Mutual information
            mi = 0
            for i in range(bins):
                for j in range(bins):
                    if hist_2d[i, j] > 0 and px[i] > 0 and py[j] > 0:
                        mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (px[i] * py[j]))

            mi_values.append(mi)

            # Check for first minimum
            if len(mi_values) >= 3:
                if mi_values[-2] < mi_values[-3] and mi_values[-2] < mi_values[-1]:
                    return delay - 1

        # Fallback: use first local minimum or delay where MI drops to 1/e
        if len(mi_values) == 0:
            return 1

        mi_values = np.array(mi_values)
        threshold = mi_values[0] / np.e
        below_threshold = np.where(mi_values < threshold)[0]
        if len(below_threshold) > 0:
            return below_threshold[0] + 1

        return max(1, len(mi_values) // 4)

    def _optimal_embedding_dim(self, signal: np.ndarray, delay: int) -> int:
        """
        Find optimal embedding dimension using false nearest neighbors.
        """
        n = len(signal)
        threshold = 15.0  # Standard FNN threshold

        for dim in range(1, self.max_dim + 1):
            embedded = self._embed(signal, dim, delay)
            if len(embedded) < 10:
                return dim

            # Find nearest neighbors in dim-dimensional space
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(embedded)
            distances, indices = nbrs.kneighbors(embedded)

            # Check if neighbors are false (would separate in dim+1)
            embedded_plus1 = self._embed(signal, dim + 1, delay)
            if len(embedded_plus1) < len(embedded):
                embedded = embedded[:len(embedded_plus1)]
                distances = distances[:len(embedded_plus1)]
                indices = indices[:len(embedded_plus1)]

            if len(embedded_plus1) == 0:
                return dim

            false_neighbors = 0
            total_checked = 0

            for i in range(len(embedded)):
                nn_idx = indices[i, 1]  # Nearest neighbor (not self)
                if nn_idx >= len(embedded_plus1):
                    continue

                d_current = distances[i, 1]
                if d_current < 1e-10:
                    continue

                # Distance in (dim+1)-dimensional space
                d_next = np.linalg.norm(embedded_plus1[i] - embedded_plus1[nn_idx])

                # Check if false neighbor
                ratio = abs(d_next - d_current) / d_current
                if ratio > threshold:
                    false_neighbors += 1
                total_checked += 1

            if total_checked > 0:
                fnn_ratio = false_neighbors / total_checked
                if fnn_ratio < 0.01:  # Less than 1% false neighbors
                    return dim

        return self.max_dim

    def _embed(self, signal: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """
        Create Takens embedding.

        Returns array of shape (n_points, dim)
        """
        n = len(signal)
        n_points = n - (dim - 1) * delay

        if n_points <= 0:
            return np.array([]).reshape(0, dim)

        embedded = np.zeros((n_points, dim))
        for i in range(dim):
            embedded[:, i] = signal[i * delay : i * delay + n_points]

        return embedded

    def _correlation_dimension(self, embedded: np.ndarray, n_scales: int = 20) -> float:
        """
        Estimate correlation dimension using Grassberger-Procaccia algorithm.
        """
        if len(embedded) < 50:
            return np.nan

        # Compute pairwise distances
        distances = pdist(embedded)
        distances = distances[distances > 0]  # Remove zeros

        if len(distances) < 100:
            return np.nan

        # Range of scales
        r_min = np.percentile(distances, 1)
        r_max = np.percentile(distances, 50)

        if r_min <= 0 or r_max <= r_min:
            return np.nan

        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_scales)

        # Correlation sum C(r) = fraction of pairs within distance r
        C_r = []
        for r in radii:
            count = np.sum(distances < r)
            C_r.append(count / len(distances))

        C_r = np.array(C_r)

        # Linear regression on log-log plot (scaling region)
        valid = (C_r > 0.01) & (C_r < 0.5)  # Scaling region
        if np.sum(valid) < 5:
            return np.nan

        log_r = np.log(radii[valid])
        log_C = np.log(C_r[valid])

        slope, _, r_value, _, _ = linregress(log_r, log_C)

        # Only trust if good linear fit
        if r_value**2 < 0.9:
            return np.nan

        return slope

    def _lyapunov_exponent(self, embedded: np.ndarray, dt: float = 1.0) -> float:
        """
        Estimate largest Lyapunov exponent using Rosenstein's method.
        """
        if len(embedded) < 100:
            return np.nan

        n = len(embedded)

        # Find nearest neighbors (excluding temporally close points)
        min_temporal_separation = self.delay * self.embedding_dim * 2

        nbrs = NearestNeighbors(n_neighbors=min(n // 10, 50), algorithm='ball_tree').fit(embedded)
        distances, indices = nbrs.kneighbors(embedded)

        # For each point, find nearest neighbor with temporal separation
        nn_indices = np.zeros(n, dtype=int)
        nn_distances = np.zeros(n)

        for i in range(n):
            for j in range(1, len(indices[i])):
                nn_idx = indices[i, j]
                if abs(nn_idx - i) > min_temporal_separation:
                    nn_indices[i] = nn_idx
                    nn_distances[i] = distances[i, j]
                    break

        # Track divergence
        max_iter = min(n // 4, 200)
        divergence = np.zeros(max_iter)
        counts = np.zeros(max_iter)

        for i in range(n - max_iter):
            j = nn_indices[i]
            if j == 0 or j + max_iter >= n:
                continue

            d0 = nn_distances[i]
            if d0 < 1e-10:
                continue

            for k in range(max_iter):
                if i + k >= n or j + k >= n:
                    break
                dk = np.linalg.norm(embedded[i + k] - embedded[j + k])
                if dk > 0:
                    divergence[k] += np.log(dk / d0)
                    counts[k] += 1

        # Average divergence
        valid = counts > 10
        if np.sum(valid) < 10:
            return np.nan

        avg_divergence = divergence[valid] / counts[valid]
        time = np.arange(len(avg_divergence)) * dt

        # Linear fit to find Lyapunov exponent
        # Use early portion (before saturation)
        n_fit = min(len(time), 50)
        slope, _, r_value, _, _ = linregress(time[:n_fit], avg_divergence[:n_fit])

        if r_value**2 < 0.8:
            return np.nan

        return slope

    def _classify_attractor(self) -> str:
        """
        Classify attractor type based on computed metrics.
        """
        if self.correlation_dim is None or np.isnan(self.correlation_dim):
            return "unknown"

        if self.lyapunov_exp is None or np.isnan(self.lyapunov_exp):
            lya = 0  # Assume stable if can't compute
        else:
            lya = self.lyapunov_exp

        dim = self.correlation_dim

        # Classification heuristics
        if dim < 0.5:
            return "fixed_point"
        elif dim < 1.5:
            if lya < 0:
                return "limit_cycle_stable"
            else:
                return "limit_cycle_unstable"
        elif dim < 2.5:
            if lya < 0:
                return "torus"
            else:
                return "quasi_periodic"
        else:
            if lya > 0.01:
                return "strange_attractor"
            else:
                return "high_dimensional"

    def get_results(self) -> Dict[str, Any]:
        """Return all computed metrics."""
        return {
            'embedding_dim': self.embedding_dim,
            'delay': self.delay,
            'correlation_dim': self.correlation_dim,
            'lyapunov_exponent': self.lyapunov_exp,
            'attractor_type': self.attractor_type,
            'n_embedded_points': len(self.embedded_trajectory) if self.embedded_trajectory is not None else 0,
        }


def compute(
    signal: np.ndarray,
    baseline_windows: Optional[int] = None,
    max_dim: int = 10,
    max_delay: int = 100
) -> Dict[str, Any]:
    """
    Reconstruct attractor from signal.

    Parameters
    ----------
    signal : array
        1D time series
    baseline_windows : int, optional
        Number of initial windows to use for learning attractor
    max_dim : int
        Maximum embedding dimension to try
    max_delay : int
        Maximum delay to try

    Returns
    -------
    dict
        embedding_dim: optimal embedding dimension
        delay: optimal delay
        correlation_dim: correlation dimension (attractor complexity)
        lyapunov_exponent: largest Lyapunov exponent
        attractor_type: classification string
        n_embedded_points: number of points in embedded trajectory
    """
    ar = AttractorReconstructor(max_dim=max_dim, max_delay=max_delay)
    ar.fit(signal, baseline_windows=baseline_windows)
    return ar.get_results()
