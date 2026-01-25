"""
PRISM Lyapunov Exponent Engine

Estimates largest Lyapunov exponent to detect chaos.

Measures:
- Largest Lyapunov exponent (LLE)
- Indicates sensitivity to initial conditions
- Positive LLE → chaotic behavior
- Negative LLE → stable/convergent

Phase: Unbound
Normalization: Z-score preferred

Performance: Uses numba JIT compilation for 10-50x speedup on neighbor search.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import date

import numpy as np
import pandas as pd

from prism.engines.engine_base import BaseEngine
from prism.engines.metadata import EngineMetadata

# Numba JIT compilation for performance-critical loops
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="lyapunov",
    engine_type="vector",
    description="Largest Lyapunov exponent for chaos detection",
    domains={"signal_topology", "chaos"},
    requires_window=True,
    deterministic=True,
)


# =============================================================================
# Vector Engine Contract: Simple function interface
# =============================================================================

def compute_lyapunov_with_derivation(
    values: np.ndarray,
    signal_id: str = "unknown",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    embedding_dim: int = 3,
    time_delay: int = 1,
) -> tuple:
    """
    Compute Lyapunov exponent with full mathematical derivation.

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from prism.entry_points.derivations.base import Derivation

    deriv = Derivation(
        engine_name="lyapunov_exponent",
        method_name="Rosenstein Algorithm (Largest Lyapunov Exponent)",
        signal_id=signal_id,
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=len(values),
        raw_data_sample=values[:10].tolist() if len(values) >= 10 else values.tolist(),
        parameters={
            'embedding_dim': embedding_dim,
            'time_delay': time_delay,
        }
    )

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n < 100:
        deriv.final_result = None
        deriv.interpretation = "Insufficient data (n < 100)"
        return {"lyapunov_exponent": None}, deriv

    # Step 1: Data summary
    deriv.add_step(
        title="Input Data Summary",
        equation="X = {x₁, x₂, ..., xₙ}",
        calculation=f"n = {n}\nRange: [{np.min(values):.4f}, {np.max(values):.4f}]\nMean: {np.mean(values):.4f}\nStd: {np.std(values):.4f}",
        result=n,
        result_name="n"
    )

    # Step 2: Normalize
    mean_val = np.mean(values)
    std_val = np.std(values) + 1e-10
    normalized = (values - mean_val) / std_val

    deriv.add_step(
        title="Normalize Data (Z-score)",
        equation="x̃ᵢ = (xᵢ - μ) / σ",
        calculation=f"μ = {mean_val:.4f}\nσ = {std_val:.4f}\nx̃₀ = ({values[0]:.4f} - {mean_val:.4f}) / {std_val:.4f} = {normalized[0]:.4f}\nx̃₁ = ({values[1]:.4f} - {mean_val:.4f}) / {std_val:.4f} = {normalized[1]:.4f}\n⋮",
        result=normalized[:5].tolist(),
        result_name="x̃",
        notes="Normalization improves numerical stability"
    )

    # Step 3: Time-delay embedding
    dim = embedding_dim
    tau = time_delay
    embedded = _embed_signal_topology(normalized, dim, tau)
    n_vectors = embedded.shape[0]

    deriv.add_step(
        title="Time-Delay Embedding",
        equation="yᵢ = [x̃ᵢ, x̃ᵢ₊τ, x̃ᵢ₊₂τ, ..., x̃ᵢ₊(m-1)τ]",
        calculation=f"Embedding dimension m = {dim}\nTime delay τ = {tau}\nNumber of vectors: {n_vectors}\n\nExample embedded vectors:\ny₀ = [{embedded[0, 0]:.4f}, {embedded[0, 1]:.4f}, {embedded[0, 2]:.4f}]\ny₁ = [{embedded[1, 0]:.4f}, {embedded[1, 1]:.4f}, {embedded[1, 2]:.4f}]\n⋮",
        result=n_vectors,
        result_name="n_vectors",
        notes=f"Created {n_vectors} vectors in {dim}-dimensional phase space"
    )

    # Step 4: Find nearest neighbors
    min_separation = 10
    max_iterations = 50

    # Show example nearest neighbor calculation
    i = 0
    best_j = -1
    best_dist = np.inf
    for j in range(n_vectors):
        if abs(i - j) <= min_separation:
            continue
        dist = np.sqrt(np.sum((embedded[i] - embedded[j])**2))
        if dist < best_dist and dist > 0:
            best_dist = dist
            best_j = j

    deriv.add_step(
        title="Find Nearest Neighbors (Rosenstein Method)",
        equation="d(i,j) = ||yᵢ - yⱼ||₂, with |i-j| > min_separation",
        calculation=f"For point i=0:\n  Search all j where |0-j| > {min_separation}\n  Best neighbor: j = {best_j}\n  Initial distance: d₀ = {best_dist:.6f}\n\nThis is repeated for all i from 0 to {n_vectors - max_iterations - 1}",
        result=best_dist,
        result_name="d₀",
        notes=f"Temporal separation {min_separation} prevents false neighbors"
    )

    # Step 5: Track divergence
    embedded = embedded.astype(np.float64)
    divergence, count = _compute_divergence_numba(embedded, min_separation, max_iterations)

    valid = count > 0
    divergence[valid] /= count[valid]

    deriv.add_step(
        title="Track Divergence Over Time",
        equation="S(k) = (1/N) Σᵢ ln(dᵢ(k) / dᵢ(0))",
        calculation=f"For each pair (i, nearest_neighbor_j), track distance over k steps:\n  S(0) = {divergence[0]:.6f}\n  S(1) = {divergence[1]:.6f}\n  S(2) = {divergence[2]:.6f}\n  ⋮\n  S({max_iterations-1}) = {divergence[-1]:.6f}\n\nNumber of valid pairs at each k: {count[:5].astype(int).tolist()}...",
        result=divergence[:5].tolist(),
        result_name="S(k)",
        notes="Divergence measures how fast nearby trajectories separate"
    )

    # Step 6: Linear regression for LLE
    valid_indices = np.where(valid)[0]
    n_fit = min(20, len(valid_indices))
    t = valid_indices[:n_fit]
    d = divergence[t]

    slope, intercept = np.polyfit(t, d, 1)
    lle = float(slope)

    deriv.add_step(
        title="Estimate Lyapunov Exponent (Linear Fit)",
        equation="S(k) ≈ λ × k + c, where λ is the largest Lyapunov exponent",
        calculation=f"Linear region: k = 0 to {n_fit-1}\nFit: S(k) = {slope:.6f} × k + {intercept:.6f}\n\nData points used:\n  k: {t[:5].tolist()}...\n  S(k): [{d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}, ...]",
        result=lle,
        result_name="λ",
        notes="The slope of log-divergence vs time is the Lyapunov exponent"
    )

    deriv.final_result = lle
    deriv.prism_output = lle

    # Interpretation
    if lle > 0.01:
        interp = f"λ = {lle:.6f} > 0 indicates **chaotic** dynamics. Nearby trajectories diverge exponentially, confirming sensitive dependence on initial conditions."
    elif lle < -0.01:
        interp = f"λ = {lle:.6f} < 0 indicates **stable** dynamics. Nearby trajectories converge, suggesting a stable attractor."
    else:
        interp = f"λ = {lle:.6f} ≈ 0 indicates **marginal** dynamics. The system is at the edge of chaos."

    deriv.interpretation = interp

    result = {
        "lyapunov_exponent": lle,
        "is_chaotic": lle > 0.01,
        "embedding_dim": dim,
    }

    return result, deriv


def compute_lyapunov(values: np.ndarray, embedding_dim: int = 3,
                     time_delay: int = 1) -> dict:
    """
    Compute Lyapunov exponent for a single signal.

    Uses REAL nolds package when available, falls back to simplified method.

    Args:
        values: Array of observed values
        embedding_dim: Dimension for phase space embedding
        time_delay: Time delay for embedding

    Returns:
        Dict with Lyapunov metrics
    """
    # Import the real Lyapunov implementation
    from prism.engines.dynamics.lyapunov import compute as compute_lyapunov_real

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < 100:
        return {
            "lyapunov_exponent": None,
            "is_chaotic": None,
            "embedding_dim": embedding_dim,
            "method": None,
        }

    # Use the real implementation
    result = compute_lyapunov_real(
        values,
        mode='static',
        embedding_dim=embedding_dim,
        delay=time_delay,
    )

    return {
        "lyapunov_exponent": result.get('lyapunov_exponent'),
        "is_chaotic": result.get('is_chaotic'),
        "is_stable": result.get('is_stable'),
        "embedding_dim": result.get('embedding_dimension', embedding_dim),
        "method": result.get('method'),
    }


def _embed_signal_topology(x: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    Time-delay embedding of a signal topology.
    """
    n = len(x)
    n_vectors = n - (dim - 1) * tau

    if n_vectors <= 0:
        return np.array([]).reshape(0, dim)

    embedded = np.zeros((n_vectors, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau:i * tau + n_vectors]

    return embedded


@jit(nopython=True, cache=True)
def _compute_divergence_numba(
    embedded: np.ndarray,
    min_separation: int,
    max_iterations: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-JIT compiled divergence calculation for Rosenstein algorithm.

    Finds nearest neighbors and tracks divergence over time.
    10-50x faster than pure Python for typical series lengths.

    Returns (divergence_sum, count) arrays.
    """
    n = embedded.shape[0]
    dim = embedded.shape[1]

    divergence = np.zeros(max_iterations)
    count = np.zeros(max_iterations)

    for i in range(n - max_iterations):
        # Find nearest neighbor at least min_separation apart
        best_j = -1
        best_dist = np.inf

        for j in range(n):
            # Skip temporal neighbors
            if abs(i - j) <= min_separation:
                continue

            # Compute Euclidean distance
            dist = 0.0
            for d in range(dim):
                diff = embedded[i, d] - embedded[j, d]
                dist += diff * diff
            dist = np.sqrt(dist)

            if dist < best_dist and dist > 0:
                best_dist = dist
                best_j = j

        if best_j < 0 or best_dist == np.inf:
            continue

        initial_dist = best_dist

        # Track divergence over time
        for k in range(max_iterations):
            if i + k >= n or best_j + k >= n:
                break

            # Compute distance at time k
            dist_k = 0.0
            for d in range(dim):
                diff = embedded[i + k, d] - embedded[best_j + k, d]
                dist_k += diff * diff
            dist_k = np.sqrt(dist_k)

            if dist_k > 0:
                divergence[k] += np.log(dist_k / initial_dist)
                count[k] += 1

    return divergence, count


def _rosenstein_lyapunov(
    x: np.ndarray,
    dim: int = 3,
    tau: int = 1,
    min_separation: int = 10,
    max_iterations: int = 50
) -> Tuple[float, np.ndarray]:
    """
    Estimate largest Lyapunov exponent using Rosenstein's algorithm.

    Based on: Rosenstein et al. (1993) "A practical method for
    calculating largest Lyapunov exponents from small data sets"

    Uses numba-JIT compiled inner loop for 10-50x speedup.

    Returns (lyapunov_exponent, divergence_curve)
    """
    # Embed signal topology
    embedded = _embed_signal_topology(x, dim, tau)
    n = embedded.shape[0]

    if n < max_iterations + min_separation:
        return np.nan, np.array([])

    # Ensure float64 for numba
    embedded = embedded.astype(np.float64)

    # Use numba-compiled divergence calculation
    divergence, count = _compute_divergence_numba(embedded, min_separation, max_iterations)

    # Average divergence
    valid = count > 0
    if not np.any(valid):
        return np.nan, np.array([])

    divergence[valid] /= count[valid]

    # Estimate Lyapunov exponent from slope of linear region
    valid_indices = np.where(valid)[0]
    if len(valid_indices) < 5:
        return np.nan, divergence

    # Use first portion of curve (linear region)
    n_fit = min(20, len(valid_indices))
    t = valid_indices[:n_fit]
    d = divergence[t]

    # Linear regression for slope
    slope, _, _, _, _ = np.polyfit(t, d, 1, full=True)
    lyapunov = slope[0] if isinstance(slope, np.ndarray) else slope

    return float(lyapunov), divergence


def _estimate_embedding_params(x: np.ndarray) -> Tuple[int, int]:
    """
    Estimate embedding dimension and time delay.

    Uses autocorrelation for tau and false nearest neighbors for dim.
    Simplified heuristics for robustness.
    """
    n = len(x)

    # Time delay: first zero crossing of autocorrelation
    # or first minimum
    max_lag = min(100, n // 4)
    autocorr = np.correlate(x - x.mean(), x - x.mean(), mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]

    tau = 1
    for i in range(1, max_lag):
        if autocorr[i] < 0.5:  # Threshold approach
            tau = i
            break

    # Embedding dimension: heuristic based on data length
    # Typically 2-10 for signal topology data
    if n < 200:
        dim = 2
    elif n < 500:
        dim = 3
    else:
        dim = 4

    return dim, tau


class LyapunovEngine(BaseEngine):
    """
    Lyapunov exponent engine.

    Estimates the largest Lyapunov exponent to characterize
    system dynamics (chaos vs stability).

    Interpretation:
    - LLE > 0: Chaotic (exponential divergence)
    - LLE ≈ 0: Marginal stability
    - LLE < 0: Stable (convergent)

    Outputs:
        - results.lyapunov: Per-signal Lyapunov estimates
    """

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    name = "lyapunov"
    phase = "derived"
    default_normalization = "zscore"

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        embedding_dim: Optional[int] = None,
        time_delay: Optional[int] = None,
        min_separation: int = 10,
        max_iterations: int = 50,
        **params
    ) -> Dict[str, Any]:
        """
        Run Lyapunov exponent estimation.

        Args:
            df: Normalized signal data
            run_id: Unique run identifier
            embedding_dim: Embedding dimension (auto if None)
            time_delay: Time delay for embedding (auto if None)
            min_separation: Minimum temporal separation for neighbors
            max_iterations: Max iterations for divergence tracking

        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = df_clean.columns.tolist()
        n_signals = len(signals)

        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()

        # Compute Lyapunov exponent for each signal
        records = []
        all_lle = []
        n_chaotic = 0
        n_stable = 0

        for signal in signals:
            x = df_clean[signal].values

            # Estimate embedding parameters if not provided
            if embedding_dim is None or time_delay is None:
                auto_dim, auto_tau = _estimate_embedding_params(x)
                dim = embedding_dim or auto_dim
                tau = time_delay or auto_tau
            else:
                dim, tau = embedding_dim, time_delay

            # Compute Lyapunov exponent
            lle, divergence = _rosenstein_lyapunov(
                x, dim=dim, tau=tau,
                min_separation=min_separation,
                max_iterations=max_iterations
            )

            if not np.isnan(lle):
                all_lle.append(lle)
                if lle > 0.01:
                    n_chaotic += 1
                elif lle < -0.01:
                    n_stable += 1

            records.append({
                "signal_id": signal,
                "window_start": window_start,
                "window_end": window_end,
                "lyapunov_exponent": float(lle) if not np.isnan(lle) else None,
                "embedding_dim": dim,
                "time_delay": tau,
                "is_chaotic": lle > 0.01 if not np.isnan(lle) else None,
                "is_stable": lle < -0.01 if not np.isnan(lle) else None,
                "run_id": run_id,
            })

        if records:
            df_results = pd.DataFrame(records)
            self.store_results("lyapunov", df_results, run_id)

        # Summary metrics
        metrics = {
            "n_signals": n_signals,
            "n_samples": len(df_clean),
            "n_successful": len(all_lle),
            "avg_lyapunov": float(np.mean(all_lle)) if all_lle else None,
            "max_lyapunov": float(np.max(all_lle)) if all_lle else None,
            "min_lyapunov": float(np.min(all_lle)) if all_lle else None,
            "n_chaotic": n_chaotic,
            "n_stable": n_stable,
            "chaotic_fraction": n_chaotic / len(all_lle) if all_lle else 0.0,
        }

        logger.info(
            f"Lyapunov complete: {n_signals} signals, "
            f"avg_LLE={metrics['avg_lyapunov']}, "
            f"chaotic={n_chaotic}, stable={n_stable}"
        )

        return metrics
