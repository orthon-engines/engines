"""
PRISM DTW (Dynamic Time Warping) Engine

Measures shape similarity between signal topology.

Measures:
- DTW distance matrix
- Optimal warping paths
- Cluster-based similarity groupings

Phase: Unbound
Normalization: Z-score preferred

Performance: Uses numba JIT compilation for 50-200x speedup on distance computation.
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata

# Numba JIT compilation for performance-critical loops
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: identity decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="dtw",
    engine_type="geometry",
    description="Dynamic time warping for shape similarity",
    domains={"similarity", "shape"},
    requires_window=True,
    deterministic=True,
)


@jit(nopython=True, cache=True)
def _dtw_distance_numba(x: np.ndarray, y: np.ndarray, window: int) -> float:
    """
    Numba-JIT compiled DTW distance computation.

    Uses dynamic programming with Sakoe-Chiba band constraint.
    50-200x faster than pure Python for typical signal topology lengths.
    """
    n, m = len(x), len(y)

    # Initialize cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)
        for j in range(j_start, j_end):
            cost = (x[i - 1] - y[j - 1]) ** 2
            # Inline min for numba efficiency
            prev_min = dtw[i - 1, j]
            if dtw[i, j - 1] < prev_min:
                prev_min = dtw[i, j - 1]
            if dtw[i - 1, j - 1] < prev_min:
                prev_min = dtw[i - 1, j - 1]
            dtw[i, j] = cost + prev_min

    return np.sqrt(dtw[n, m])


def _dtw_distance(x: np.ndarray, y: np.ndarray, window: Optional[int] = None) -> float:
    """
    Compute DTW distance between two sequences.

    Uses dynamic programming with optional Sakoe-Chiba band constraint.
    Delegates to numba-compiled version when available.
    """
    n, m = len(x), len(y)

    # Sakoe-Chiba band
    if window is None:
        window = max(n, m)

    return _dtw_distance_numba(x.astype(np.float64), y.astype(np.float64), window)


class DTWEngine(BaseEngine):
    """
    Dynamic Time Warping engine.

    Measures shape similarity between signal topology, allowing for
    temporal warping/alignment.

    Outputs:
        - results.dtw_distances: Pairwise DTW distance matrix
        - results.dtw_clusters: Shape-based cluster assignments
    """

    name = "dtw"
    phase = "derived"
    default_normalization = "zscore"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        window_constraint: Optional[int] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Run DTW analysis.

        Args:
            df: Normalized signal data
            run_id: Unique run identifier
            window_constraint: Sakoe-Chiba band width (None = no constraint)

        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = df_clean.columns.tolist()
        n_signals = len(signals)

        window_start, window_end = get_window_dates(df_clean)

        # Compute pairwise DTW distances
        distances = np.zeros((n_signals, n_signals))

        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                x = df_clean.iloc[:, i].values
                y = df_clean.iloc[:, j].values

                dist = _dtw_distance(x, y, window=window_constraint)
                distances[i, j] = dist
                distances[j, i] = dist

        # Convert to similarity (for comparison with correlation)
        max_dist = distances.max() if distances.max() > 0 else 1.0
        similarity = 1 - (distances / max_dist)

        # Store distance matrix
        records = []
        for i, ind_i in enumerate(signals):
            for j, ind_j in enumerate(signals):
                if i < j:
                    records.append({
                        "signal_1": ind_i,
                        "signal_2": ind_j,
                        "window_start": window_start,
                        "window_end": window_end,
                        "dtw_distance": float(distances[i, j]),
                        "dtw_similarity": float(similarity[i, j]),
                        "run_id": run_id,
                    })

        if records:
            df_results = pd.DataFrame(records)
            ##self.store_results("dtw_distances", df_results, run_id)

        # Summary metrics
        n_pairs = n_signals * (n_signals - 1) // 2
        condensed = squareform(distances)

        metrics = {
            "n_signals": n_signals,
            "n_pairs": n_pairs,
            "n_samples": len(df_clean),
            "avg_dtw_distance": float(np.mean(condensed)) if len(condensed) > 0 else 0.0,
            "max_dtw_distance": float(np.max(condensed)) if len(condensed) > 0 else 0.0,
            "min_dtw_distance": float(np.min(condensed)) if len(condensed) > 0 else 0.0,
            "avg_similarity": float(np.mean(similarity[np.triu_indices(n_signals, k=1)])) if n_pairs > 0 else 0.0,
            "window_constraint": window_constraint,
        }

        logger.info(
            f"DTW complete: {n_signals} signals, "
            f"avg_dist={metrics['avg_dtw_distance']:.4f}"
        )

        return metrics


# =============================================================================
# Standalone function with derivation
# =============================================================================

def compute_dtw_with_derivation(
    x: np.ndarray,
    y: np.ndarray,
    signal_x: str = "X",
    signal_y: str = "Y",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    window_constraint: int = None,
) -> tuple:
    """
    Compute DTW distance with full mathematical derivation.

    Args:
        x: First signal topology
        y: Second signal topology
        signal_x: Name of X signal
        signal_y: Name of Y signal
        window_id: Window identifier
        window_start, window_end: Date range
        window_constraint: Sakoe-Chiba band width (None = no constraint)

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from prism.entry_points.derivations.base import Derivation

    n, m = len(x), len(y)

    deriv = Derivation(
        engine_name="dtw",
        method_name="Dynamic Time Warping",
        signal_id=f"{signal_x}_vs_{signal_y}",
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=n,
        parameters={'window_constraint': window_constraint}
    )

    # Step 1: Input data
    deriv.add_step(
        title="Input Signal Topology",
        equation="x ∈ ℝⁿ, y ∈ ℝᵐ",
        calculation=f"Series {signal_x}:\n"
                    f"  length n = {n}\n"
                    f"  range: [{np.min(x):.4f}, {np.max(x):.4f}]\n\n"
                    f"Series {signal_y}:\n"
                    f"  length m = {m}\n"
                    f"  range: [{np.min(y):.4f}, {np.max(y):.4f}]\n\n"
                    f"DTW allows temporal alignment (warping) to compare shapes",
        result=n,
        result_name="n",
        notes="DTW finds optimal alignment between series of potentially different lengths"
    )

    # Step 2: Cost matrix definition
    deriv.add_step(
        title="Local Cost Matrix",
        equation="c(i,j) = (xᵢ - yⱼ)²",
        calculation=f"Cost matrix C ∈ ℝⁿˣᵐ = {n} × {m} = {n*m} cells\n\n"
                    f"Sample costs:\n"
                    f"  c(1,1) = ({x[0]:.4f} - {y[0]:.4f})² = {(x[0]-y[0])**2:.6f}\n"
                    f"  c(1,m) = ({x[0]:.4f} - {y[-1]:.4f})² = {(x[0]-y[-1])**2:.6f}\n"
                    f"  c(n,1) = ({x[-1]:.4f} - {y[0]:.4f})² = {(x[-1]-y[0])**2:.6f}\n"
                    f"  c(n,m) = ({x[-1]:.4f} - {y[-1]:.4f})² = {(x[-1]-y[-1])**2:.6f}",
        result=(x[0]-y[0])**2,
        result_name="c(1,1)",
        notes="Squared Euclidean distance at each point pair"
    )

    # Step 3: Sakoe-Chiba band constraint
    if window_constraint is None:
        w = max(n, m)
        band_desc = "No constraint (full matrix)"
    else:
        w = window_constraint
        band_desc = f"Window = {w} (constrains warping path)"

    deriv.add_step(
        title="Sakoe-Chiba Band Constraint",
        equation="|i - j| ≤ w (warping path must stay within band)",
        calculation=f"Band width w = {w}\n\n"
                    f"{band_desc}\n\n"
                    f"Without constraint: O(n·m) = O({n*m}) operations\n"
                    f"With constraint w: O(n·w) = O({n*w}) operations\n\n"
                    f"Speedup: {n*m/(n*w):.1f}x",
        result=w,
        result_name="w",
        notes="Band constraint prevents excessive warping and speeds computation"
    )

    # Step 4: Dynamic programming recurrence
    deriv.add_step(
        title="Dynamic Programming Recurrence",
        equation="D(i,j) = c(i,j) + min{D(i-1,j), D(i,j-1), D(i-1,j-1)}",
        calculation=f"Accumulated cost matrix D:\n\n"
                    f"Base cases:\n"
                    f"  D(0,0) = 0\n"
                    f"  D(i,0) = ∞ for i > 0\n"
                    f"  D(0,j) = ∞ for j > 0\n\n"
                    f"Recurrence allows three moves:\n"
                    f"  → (i-1,j): Repeat y value\n"
                    f"  ↑ (i,j-1): Repeat x value\n"
                    f"  ↗ (i-1,j-1): Match points",
        result=0.0,
        result_name="D(0,0)",
        notes="DP finds minimum cost path from (1,1) to (n,m)"
    )

    # Step 5: Compute DTW distance
    dtw_dist = _dtw_distance(x.astype(np.float64), y.astype(np.float64), window=w)

    deriv.add_step(
        title="DTW Distance Computation",
        equation="DTW(x,y) = √D(n,m)",
        calculation=f"Final accumulated cost D(n,m) computed via DP\n\n"
                    f"DTW distance = √D({n},{m})\n"
                    f"DTW distance = {dtw_dist:.6f}",
        result=dtw_dist,
        result_name="DTW",
        notes="Square root converts squared cost to Euclidean-like distance"
    )

    # Step 6: Euclidean comparison
    # Compute standard Euclidean distance for comparison
    if n == m:
        euclidean_dist = np.sqrt(np.sum((x - y) ** 2))
    else:
        # Interpolate shorter to longer for comparison
        from scipy.interpolate import interp1d
        if n > m:
            f = interp1d(np.linspace(0, 1, m), y)
            y_interp = f(np.linspace(0, 1, n))
            euclidean_dist = np.sqrt(np.sum((x - y_interp) ** 2))
        else:
            f = interp1d(np.linspace(0, 1, n), x)
            x_interp = f(np.linspace(0, 1, m))
            euclidean_dist = np.sqrt(np.sum((x_interp - y) ** 2))

    deriv.add_step(
        title="Comparison with Euclidean Distance",
        equation="d_E(x,y) = √Σᵢ(xᵢ - yᵢ)²",
        calculation=f"Euclidean distance (point-to-point): {euclidean_dist:.6f}\n"
                    f"DTW distance (optimal alignment): {dtw_dist:.6f}\n\n"
                    f"DTW / Euclidean ratio: {dtw_dist/euclidean_dist:.4f}\n\n"
                    f"Interpretation:\n"
                    f"  Ratio < 1: DTW finds better alignment than direct mapping\n"
                    f"  Ratio ≈ 1: Series are already well-aligned\n"
                    f"  Current: {'Better alignment via warping' if dtw_dist < euclidean_dist else 'Direct mapping is optimal'}",
        result=euclidean_dist,
        result_name="d_E",
        notes="DTW ≤ Euclidean always (optimal warping can't be worse)"
    )

    # Step 7: Normalized similarity
    # Normalize by path length for comparison
    path_length = n + m  # Approximate path length
    normalized_dist = dtw_dist / np.sqrt(path_length)

    # Convert to similarity (0 = identical, 1 = very different)
    max_possible = np.sqrt(path_length) * max(np.std(x), np.std(y)) * 4  # Rough upper bound
    similarity = 1 - min(dtw_dist / max_possible, 1.0) if max_possible > 0 else 0.5

    deriv.add_step(
        title="Normalized Distance and Similarity",
        equation="DTW_norm = DTW / √(n+m),  Similarity = 1 - DTW/max_DTW",
        calculation=f"Path length ≈ n + m = {path_length}\n"
                    f"Normalized distance = {dtw_dist:.4f} / √{path_length} = {normalized_dist:.6f}\n\n"
                    f"Similarity score: {similarity:.4f}\n\n"
                    f"Interpretation:\n"
                    f"  Similarity > 0.8: Very similar shapes\n"
                    f"  Similarity > 0.5: Moderately similar\n"
                    f"  Similarity < 0.3: Very different shapes",
        result=similarity,
        result_name="sim",
        notes="Similarity allows comparison across different series pairs"
    )

    # Final result
    result = {
        'dtw_distance': float(dtw_dist),
        'euclidean_distance': float(euclidean_dist),
        'normalized_distance': float(normalized_dist),
        'similarity': float(similarity),
        'window_constraint': window_constraint,
        'series_length_x': n,
        'series_length_y': m,
    }

    deriv.final_result = dtw_dist
    deriv.prism_output = dtw_dist

    # Interpretation
    if similarity > 0.8:
        interp = f"**Highly similar** shapes: {signal_x} and {signal_y} (sim={similarity:.3f})."
    elif similarity > 0.5:
        interp = f"**Moderately similar** shapes: {signal_x} and {signal_y} (sim={similarity:.3f})."
    else:
        interp = f"**Different** shapes: {signal_x} and {signal_y} (sim={similarity:.3f})."

    interp += f" DTW distance = {dtw_dist:.4f}."
    if dtw_dist < euclidean_dist * 0.9:
        interp += " Significant temporal warping improves alignment."

    deriv.interpretation = interp

    return result, deriv
