"""
PRISM Distance Engine

Computes geometric distances between signals.

Measures:
- Euclidean distance
- Mahalanobis distance (accounts for covariance)
- Cosine distance (directional similarity)

Phase: Unbound
Normalization: Z-score preferred
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from orthon._internal.engines.engine_base import BaseEngine, get_window_dates
from orthon._internal.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="distance",
    engine_type="geometry",
    description="Geometric distance metrics between series",
    domains={"similarity", "distance"},
    requires_window=True,
    deterministic=True,
)


def _euclidean_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise Euclidean distances.

    Treats each signal as a vector in time-space.
    """
    X = df.values.T  # (n_signals, n_samples)
    n = X.shape[0]

    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(X[i] - X[j])
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def _mahalanobis_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise Mahalanobis distances.

    Accounts for covariance structure of the data.
    Uses time-wise covariance (treating signals as features at each time point).
    """
    X = df.values  # (n_samples, n_signals)
    n_signals = X.shape[1]

    # Compute covariance matrix across signals (correlation structure)
    cov = np.cov(X.T)  # (n_signals, n_signals)

    # Handle 1D case
    if cov.ndim == 0:
        cov = np.array([[cov]])

    # Regularize if singular
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Regularize
        cov_inv = np.linalg.inv(cov + 0.01 * np.eye(n_signals))

    # Compute mean vectors for each signal (its signal topology)
    # Then compute Mahalanobis distance between mean-centered representations
    means = X.mean(axis=0)  # (n_signals,)

    distances = np.zeros((n_signals, n_signals))
    for i in range(n_signals):
        for j in range(i + 1, n_signals):
            # Use the difference in loadings/correlation space
            diff = np.zeros(n_signals)
            diff[i] = 1
            diff[j] = -1
            dist = np.sqrt(np.abs(diff @ cov_inv @ diff))
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def _cosine_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute pairwise cosine distances.

    Measures directional similarity (1 - cosine similarity).
    """
    X = df.values.T  # (n_signals, n_samples)
    n = X.shape[0]

    # Normalize to unit vectors
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    X_norm = X / norms

    # Cosine similarity
    similarity = X_norm @ X_norm.T

    # Convert to distance
    distances = 1 - similarity
    np.fill_diagonal(distances, 0)

    return distances


def _correlation_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute correlation distance.

    Distance = 1 - |correlation|
    """
    corr = df.corr().values
    # Fill NaN with 0 (zero correlation = max distance of 1)
    corr = np.nan_to_num(corr, nan=0.0)
    distances = 1 - np.abs(corr)
    np.fill_diagonal(distances, 0)
    # Ensure symmetry (floating point errors can cause small differences)
    distances = (distances + distances.T) / 2

    return distances


class DistanceEngine(BaseEngine):
    """
    Distance engine.

    Computes multiple geometric distance metrics between signals.

    Outputs:
        - results.distances: Pairwise distance matrices
    """

    name = "distance"
    phase = "derived"
    default_normalization = "zscore"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        methods: list = None,
        **params
    ) -> Dict[str, Any]:
        """
        Run distance analysis.

        Args:
            df: Normalized signal data
            run_id: Unique run identifier
            methods: List of distance methods (default: all)

        Returns:
            Dict with summary metrics
        """
        if methods is None:
            methods = ["euclidean", "mahalanobis", "cosine", "correlation"]

        df_clean = df
        signals = df_clean.columns.tolist()
        n_signals = len(signals)

        window_start, window_end = get_window_dates(df_clean)

        # Compute distance matrices
        distance_matrices = {}

        if "euclidean" in methods:
            distance_matrices["euclidean"] = _euclidean_distance_matrix(df_clean)

        if "mahalanobis" in methods:
            try:
                distance_matrices["mahalanobis"] = _mahalanobis_distance_matrix(df_clean)
            except Exception as e:
                logger.warning(f"Mahalanobis distance failed: {e}")

        if "cosine" in methods:
            distance_matrices["cosine"] = _cosine_distance_matrix(df_clean)

        if "correlation" in methods:
            distance_matrices["correlation"] = _correlation_distance_matrix(df_clean)

        # Store results
        records = []
        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                record = {
                    "signal_1": signals[i],
                    "signal_2": signals[j],
                    "window_start": window_start,
                    "window_end": window_end,
                    "run_id": run_id,
                }

                for method, matrix in distance_matrices.items():
                    record[f"{method}_distance"] = float(matrix[i, j])

                records.append(record)

        if records:
            df_results = pd.DataFrame(records)
            self.store_results("distances", df_results, run_id)

        # Summary metrics
        metrics = {
            "n_signals": n_signals,
            "n_pairs": len(records),
            "n_samples": len(df_clean),
            "methods": methods,
        }

        for method, matrix in distance_matrices.items():
            condensed = squareform(matrix)
            metrics[f"avg_{method}_distance"] = float(np.mean(condensed))
            metrics[f"max_{method}_distance"] = float(np.max(condensed))
            metrics[f"min_{method}_distance"] = float(np.min(condensed))

        logger.info(
            f"Distance complete: {n_signals} signals, "
            f"methods={methods}"
        )

        return metrics


# =============================================================================
# Standalone function with derivation
# =============================================================================

def compute_distance_with_derivation(
    x: np.ndarray,
    y: np.ndarray,
    signal_x: str = "X",
    signal_y: str = "Y",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    method: str = "euclidean",
) -> tuple:
    """
    Compute distance between two series with full mathematical derivation.

    Args:
        x: First signal topology
        y: Second signal topology
        signal_x: Name of X signal
        signal_y: Name of Y signal
        window_id: Window identifier
        window_start, window_end: Date range
        method: 'euclidean', 'cosine', or 'correlation'

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from orthon._internal.entry_points.derivations.base import Derivation

    n = len(x)

    deriv = Derivation(
        engine_name="distance",
        method_name=f"{method.title()} Distance",
        signal_id=f"{signal_x}_vs_{signal_y}",
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=n,
        parameters={'method': method}
    )

    # Step 1: Input data
    deriv.add_step(
        title="Input Vectors",
        equation="x, y ∈ ℝⁿ (signal topology as n-dimensional vectors)",
        calculation=f"Series {signal_x}:\n"
                    f"  n = {n} dimensions (time points)\n"
                    f"  mean = {np.mean(x):.6f}\n"
                    f"  norm = ||x|| = {np.linalg.norm(x):.6f}\n\n"
                    f"Series {signal_y}:\n"
                    f"  n = {n} dimensions\n"
                    f"  mean = {np.mean(y):.6f}\n"
                    f"  norm = ||y|| = {np.linalg.norm(y):.6f}",
        result=n,
        result_name="n",
        notes="Each signal topology is a vector in n-dimensional space"
    )

    if method == "euclidean":
        # Euclidean distance
        diff = x - y
        squared_diff = diff ** 2
        sum_squared = np.sum(squared_diff)
        distance = np.sqrt(sum_squared)

        deriv.add_step(
            title="Point-wise Differences",
            equation="dᵢ = xᵢ - yᵢ",
            calculation=f"Difference vector d = x - y:\n"
                        f"  d[0] = {x[0]:.6f} - {y[0]:.6f} = {diff[0]:.6f}\n"
                        f"  d[1] = {x[1]:.6f} - {y[1]:.6f} = {diff[1]:.6f}\n"
                        f"  ...\n"
                        f"  d[n-1] = {x[-1]:.6f} - {y[-1]:.6f} = {diff[-1]:.6f}\n\n"
                        f"Statistics of d:\n"
                        f"  mean(d) = {np.mean(diff):.6f}\n"
                        f"  std(d) = {np.std(diff):.6f}",
            result=np.mean(diff),
            result_name="μ_d",
            notes="Positive mean: X > Y on average; Negative: Y > X"
        )

        deriv.add_step(
            title="Sum of Squared Differences",
            equation="Σᵢ dᵢ² = Σᵢ (xᵢ - yᵢ)²",
            calculation=f"Squared differences:\n"
                        f"  d[0]² = {diff[0]:.4f}² = {squared_diff[0]:.6f}\n"
                        f"  d[1]² = {diff[1]:.4f}² = {squared_diff[1]:.6f}\n"
                        f"  ...\n\n"
                        f"Sum: Σdᵢ² = {sum_squared:.6f}",
            result=sum_squared,
            result_name="Σd²",
            notes="Sum of squared differences forms the squared distance"
        )

        deriv.add_step(
            title="Euclidean Distance",
            equation="d_E(x,y) = √Σᵢ(xᵢ - yᵢ)²",
            calculation=f"d_E = √{sum_squared:.6f}\n"
                        f"d_E = {distance:.6f}\n\n"
                        f"Normalized by √n: {distance / np.sqrt(n):.6f}",
            result=distance,
            result_name="d_E",
            notes="Euclidean distance in n-dimensional space"
        )

    elif method == "cosine":
        # Cosine distance
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        cosine_sim = dot_product / (norm_x * norm_y) if (norm_x * norm_y) > 0 else 0
        distance = 1 - cosine_sim

        deriv.add_step(
            title="Vector Norms",
            equation="||v|| = √Σᵢvᵢ²",
            calculation=f"Norms:\n"
                        f"  ||{signal_x}|| = √Σxᵢ² = {norm_x:.6f}\n"
                        f"  ||{signal_y}|| = √Σyᵢ² = {norm_y:.6f}",
            result=norm_x,
            result_name="||x||",
            notes="L2 norm measures vector magnitude"
        )

        deriv.add_step(
            title="Dot Product",
            equation="x · y = Σᵢ xᵢyᵢ",
            calculation=f"Dot product:\n"
                        f"  x · y = Σᵢ xᵢyᵢ = {dot_product:.6f}\n\n"
                        f"Sample terms:\n"
                        f"  x[0]·y[0] = {x[0]:.4f}·{y[0]:.4f} = {x[0]*y[0]:.6f}\n"
                        f"  x[1]·y[1] = {x[1]:.4f}·{y[1]:.4f} = {x[1]*y[1]:.6f}",
            result=dot_product,
            result_name="x·y",
            notes="Dot product measures alignment between vectors"
        )

        deriv.add_step(
            title="Cosine Similarity",
            equation="cos(θ) = (x · y) / (||x|| · ||y||)",
            calculation=f"Cosine similarity:\n"
                        f"  cos(θ) = {dot_product:.4f} / ({norm_x:.4f} × {norm_y:.4f})\n"
                        f"  cos(θ) = {cosine_sim:.6f}\n\n"
                        f"Angle: θ = arccos({cosine_sim:.4f}) = {np.degrees(np.arccos(np.clip(cosine_sim, -1, 1))):.2f}°",
            result=cosine_sim,
            result_name="cos(θ)",
            notes="cos(θ) = 1: identical direction; 0: orthogonal; -1: opposite"
        )

        deriv.add_step(
            title="Cosine Distance",
            equation="d_cos(x,y) = 1 - cos(θ)",
            calculation=f"Cosine distance:\n"
                        f"  d_cos = 1 - {cosine_sim:.6f}\n"
                        f"  d_cos = {distance:.6f}\n\n"
                        f"Interpretation:\n"
                        f"  0: Identical direction\n"
                        f"  1: Orthogonal (uncorrelated)\n"
                        f"  2: Opposite direction",
            result=distance,
            result_name="d_cos",
            notes="Cosine distance ignores magnitude, measures only direction"
        )

    else:  # correlation distance
        # Correlation distance
        corr = np.corrcoef(x, y)[0, 1]
        distance = 1 - abs(corr)

        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        deriv.add_step(
            title="Center the Series",
            equation="x̃ = x - μ_x,  ỹ = y - μ_y",
            calculation=f"Centering (remove mean):\n"
                        f"  μ_x = {np.mean(x):.6f}\n"
                        f"  μ_y = {np.mean(y):.6f}\n\n"
                        f"After centering:\n"
                        f"  mean(x̃) = {np.mean(x_centered):.10f} ≈ 0\n"
                        f"  mean(ỹ) = {np.mean(y_centered):.10f} ≈ 0",
            result=np.mean(x),
            result_name="μ_x",
            notes="Centering is required for correlation"
        )

        cov = np.mean(x_centered * y_centered)
        std_x = np.std(x)
        std_y = np.std(y)

        deriv.add_step(
            title="Pearson Correlation",
            equation="r = Cov(x,y) / (σ_x · σ_y)",
            calculation=f"Covariance:\n"
                        f"  Cov(x,y) = E[x̃ · ỹ] = {cov:.6f}\n\n"
                        f"Standard deviations:\n"
                        f"  σ_x = {std_x:.6f}\n"
                        f"  σ_y = {std_y:.6f}\n\n"
                        f"Correlation:\n"
                        f"  r = {cov:.6f} / ({std_x:.6f} × {std_y:.6f})\n"
                        f"  r = {corr:.6f}",
            result=corr,
            result_name="r",
            notes="r ∈ [-1, 1]: strength and direction of linear relationship"
        )

        deriv.add_step(
            title="Correlation Distance",
            equation="d_corr(x,y) = 1 - |r|",
            calculation=f"Correlation distance:\n"
                        f"  d_corr = 1 - |{corr:.6f}|\n"
                        f"  d_corr = 1 - {abs(corr):.6f}\n"
                        f"  d_corr = {distance:.6f}\n\n"
                        f"Interpretation:\n"
                        f"  0: Perfectly (anti-)correlated\n"
                        f"  1: Uncorrelated",
            result=distance,
            result_name="d_corr",
            notes="Uses |r| so negative correlation counts as 'close'"
        )

    # Final result
    result = {
        f'{method}_distance': float(distance),
        'method': method,
        'n_samples': n,
    }

    if method == "euclidean":
        result['normalized_distance'] = float(distance / np.sqrt(n))
    elif method == "cosine":
        result['cosine_similarity'] = float(1 - distance)
    else:
        result['correlation'] = float(1 - distance) if not np.isnan(corr) else 0.0

    deriv.final_result = distance
    deriv.prism_output = distance

    # Interpretation
    if distance < 0.1:
        interp = f"**Very similar**: {signal_x} and {signal_y} ({method} distance = {distance:.4f})."
    elif distance < 0.3:
        interp = f"**Similar**: {signal_x} and {signal_y} ({method} distance = {distance:.4f})."
    elif distance < 0.7:
        interp = f"**Moderately different**: {signal_x} and {signal_y} ({method} distance = {distance:.4f})."
    else:
        interp = f"**Very different**: {signal_x} and {signal_y} ({method} distance = {distance:.4f})."

    deriv.interpretation = interp

    return result, deriv
