"""
PRISM RQA (Recurrence Quantification Analysis) Engine

Analyzes system dynamics through recurrence plots.

Measures:
- Recurrence rate (RR)
- Determinism (DET)
- Laminarity (LAM)
- Entropy of diagonal lines
- Average diagonal/vertical line length

Phase: Unbound
Normalization: Z-score preferred

Performance: Uses numba JIT compilation for 10-50x speedup on line detection.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import date

import numpy as np
import pandas as pd

from orthon._internal.engines.engine_base import BaseEngine
from orthon._internal.engines.metadata import EngineMetadata

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
    name="rqa",
    engine_type="vector",
    description="Recurrence quantification analysis for dynamics",
    domains={"signal_topology", "dynamics"},
    requires_window=True,
    deterministic=True,
)


def _embed_signal_topology(x: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    Time-delay embedding of a signal topology.

    Args:
        x: Input signal topology
        dim: Embedding dimension
        tau: Time delay

    Returns:
        Embedded trajectory matrix (n_vectors x dim)
    """
    n = len(x)
    n_vectors = n - (dim - 1) * tau

    if n_vectors <= 0:
        return np.array([]).reshape(0, dim)

    embedded = np.zeros((n_vectors, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau:i * tau + n_vectors]

    return embedded


def _recurrence_matrix(embedded: np.ndarray, threshold: float) -> np.ndarray:
    """
    Compute recurrence matrix.

    R[i,j] = 1 if ||x_i - x_j|| < threshold
    """
    n = embedded.shape[0]
    if n == 0:
        return np.array([]).reshape(0, 0)

    # Compute pairwise distances
    diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    # Apply threshold
    recurrence = (distances < threshold).astype(int)

    return recurrence


@jit(nopython=True, cache=True)
def _count_diagonal_lines_numba(R: np.ndarray, min_length: int) -> np.ndarray:
    """
    Numba-JIT compiled diagonal line detection.

    Extracts lengths of all diagonal lines >= min_length from recurrence matrix.
    10-50x faster than pure Python for typical matrix sizes.
    """
    n = R.shape[0]
    # Pre-allocate with max possible lines (upper bound)
    max_lines = n * n // 2
    lines = np.zeros(max_lines, dtype=np.int64)
    line_count = 0

    # Check all diagonals (excluding main diagonal)
    for k in range(1, n):
        diag_len = n - k
        current_length = 0

        for i in range(diag_len):
            if R[i, i + k] == 1:
                current_length += 1
            else:
                if current_length >= min_length:
                    lines[line_count] = current_length
                    line_count += 1
                current_length = 0

        if current_length >= min_length:
            lines[line_count] = current_length
            line_count += 1

    return lines[:line_count]


def _diagonal_lines(R: np.ndarray, min_length: int = 2) -> np.ndarray:
    """Extract lengths of diagonal lines from recurrence matrix."""
    if R.shape[0] == 0:
        return np.array([], dtype=np.int64)
    return _count_diagonal_lines_numba(R.astype(np.int64), min_length)


@jit(nopython=True, cache=True)
def _count_vertical_lines_numba(R: np.ndarray, min_length: int) -> np.ndarray:
    """
    Numba-JIT compiled vertical line detection.

    Extracts lengths of all vertical lines >= min_length from recurrence matrix.
    10-50x faster than pure Python for typical matrix sizes.
    """
    n = R.shape[0]
    # Pre-allocate with max possible lines
    max_lines = n * n // 2
    lines = np.zeros(max_lines, dtype=np.int64)
    line_count = 0

    for j in range(n):
        current_length = 0
        for i in range(n):
            if R[i, j] == 1:
                current_length += 1
            else:
                if current_length >= min_length:
                    lines[line_count] = current_length
                    line_count += 1
                current_length = 0

        if current_length >= min_length:
            lines[line_count] = current_length
            line_count += 1

    return lines[:line_count]


def _vertical_lines(R: np.ndarray, min_length: int = 2) -> np.ndarray:
    """Extract lengths of vertical lines from recurrence matrix."""
    if R.shape[0] == 0:
        return np.array([], dtype=np.int64)
    return _count_vertical_lines_numba(R.astype(np.int64), min_length)


# =============================================================================
# Vector Engine Contract: Simple function interface
# =============================================================================

def compute_rqa(values: np.ndarray, embedding_dim: int = 3, time_delay: int = 1,
                threshold_percentile: float = 10.0) -> dict:
    """
    Compute RQA metrics for a single signal.

    Args:
        values: Array of observed values
        embedding_dim: Embedding dimension for phase space reconstruction
        time_delay: Time delay for embedding
        threshold_percentile: Percentile of distances to use as threshold

    Returns:
        Dict of RQA metrics
    """
    # Default return for failures
    null_result = {
        "recurrence_rate": None,
        "determinism": None,
        "laminarity": None,
        "avg_diagonal_length": None,
        "avg_vertical_length": None,
        "entropy": None,
        "max_diagonal_length": None,
    }

    try:
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]

        if len(values) < 20:
            return null_result

        # Check for near-zero variance (problematic for RQA)
        if np.std(values) < 1e-10:
            return null_result

        # Embed signal topology
        embedded = _embed_signal_topology(values, embedding_dim, time_delay)

        if embedded.shape[0] < 10:
            return null_result

        # Compute threshold from distance distribution
        diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        nonzero_distances = distances[distances > 0]

        # Handle constant data (all distances = 0)
        if len(nonzero_distances) == 0:
            n = embedded.shape[0]
            return {
                "recurrence_rate": 1.0,      # Every point equals every other
                "determinism": 1.0,          # Perfectly predictable (trivially)
                "laminarity": 1.0,           # All points in vertical structures
                "avg_diagonal_length": float(n - 1),
                "avg_vertical_length": float(n),
                "entropy": 0.0,              # No complexity
                "max_diagonal_length": float(n - 1),
            }

        threshold = np.percentile(nonzero_distances, threshold_percentile)

        # Compute recurrence matrix
        R = _recurrence_matrix(embedded, threshold)

        return _compute_rqa_metrics(R)

    except Exception:
        # Any failure returns null metrics (don't crash the runner)
        return null_result


def compute_rqa_with_derivation(
    values: np.ndarray,
    signal_id: str = "unknown",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    embedding_dim: int = 3,
    time_delay: int = 1,
    threshold_percentile: float = 10.0,
) -> tuple:
    """
    Compute RQA metrics with full mathematical derivation.

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from orthon._internal.entry_points.derivations.base import Derivation

    deriv = Derivation(
        engine_name="rqa",
        method_name="Recurrence Quantification Analysis",
        signal_id=signal_id,
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=len(values),
        raw_data_sample=values[:10].tolist() if len(values) >= 10 else values.tolist(),
        parameters={
            "embedding_dim": embedding_dim,
            "time_delay": time_delay,
            "threshold_percentile": threshold_percentile,
        }
    )

    n = len(values)
    if n < 20:
        deriv.final_result = None
        deriv.interpretation = "Insufficient data (n < 20)"
        return {}, deriv

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    # Step 1: Input data
    deriv.add_step(
        title="Input Data Summary",
        equation="X = {x₁, x₂, ..., xₙ}",
        calculation=f"n = {n}\nRange: [{np.min(values):.4f}, {np.max(values):.4f}]\nMean: {np.mean(values):.4f}\nStd: {np.std(values):.4f}",
        result=n,
        result_name="n",
        notes="Signal to be analyzed for recurrence patterns"
    )

    # Step 2: Time-delay embedding
    embedded = _embed_signal_topology(values, embedding_dim, time_delay)
    n_vectors = embedded.shape[0]

    deriv.add_step(
        title="Phase Space Reconstruction (Time-Delay Embedding)",
        equation="X⃗ᵢ = [x(i), x(i+τ), x(i+2τ), ..., x(i+(m-1)τ)]",
        calculation=f"Embedding dimension m = {embedding_dim}\nTime delay τ = {time_delay}\nNumber of embedded vectors: {n_vectors}\n\nFirst 3 embedded vectors:\nX⃗₀ = {embedded[0].tolist()}\nX⃗₁ = {embedded[1].tolist()}\nX⃗₂ = {embedded[2].tolist()}",
        result=n_vectors,
        result_name="N_embedded",
        notes="Takens' embedding theorem: reconstruct attractor from scalar signal topology"
    )

    # Step 3: Distance matrix
    diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    nonzero_distances = distances[distances > 0]

    if len(nonzero_distances) == 0:
        deriv.interpretation = "Constant data - all distances zero"
        return {"recurrence_rate": 1.0, "determinism": 1.0}, deriv

    threshold = np.percentile(nonzero_distances, threshold_percentile)

    deriv.add_step(
        title="Compute Distance Matrix",
        equation="D(i,j) = ||X⃗ᵢ - X⃗ⱼ||₂ = √[Σₖ(xᵢₖ - xⱼₖ)²]",
        calculation=f"Distance matrix shape: {distances.shape}\nD(0,1) = ||X⃗₀ - X⃗₁|| = {distances[0,1]:.4f}\nD(0,2) = ||X⃗₀ - X⃗₂|| = {distances[0,2]:.4f}\n\nDistance statistics:\n  Min (non-zero): {np.min(nonzero_distances):.4f}\n  Max: {np.max(distances):.4f}\n  Median: {np.median(nonzero_distances):.4f}",
        result=threshold,
        result_name="ε",
        notes=f"Threshold ε = {threshold_percentile}th percentile of distances"
    )

    # Step 4: Recurrence matrix
    R = _recurrence_matrix(embedded, threshold)
    n_recurrent = R.sum() - n_vectors  # Exclude diagonal

    deriv.add_step(
        title="Construct Recurrence Matrix",
        equation="R(i,j) = Θ(ε - ||X⃗ᵢ - X⃗ⱼ||)",
        calculation=f"R(i,j) = 1 if D(i,j) < ε, else 0\nε = {threshold:.4f}\n\nRecurrence matrix shape: {R.shape}\nTotal recurrent points (excl. diagonal): {n_recurrent}\n\nSample R[0:5, 0:5]:\n{R[:5, :5]}",
        result=n_recurrent,
        result_name="n_recurrent",
        notes="Θ = Heaviside step function"
    )

    # Step 5: RQA metrics
    rr = n_recurrent / (n_vectors * (n_vectors - 1)) if n_vectors > 1 else 0.0

    deriv.add_step(
        title="Recurrence Rate (RR)",
        equation="RR = (1/N²) Σᵢⱼ R(i,j)  [excluding diagonal]",
        calculation=f"RR = {n_recurrent} / ({n_vectors} × {n_vectors - 1})\nRR = {n_recurrent} / {n_vectors * (n_vectors - 1)}",
        result=rr,
        result_name="RR",
        notes="Fraction of phase space that is recurrent"
    )

    # Step 6: Diagonal lines (determinism)
    diag_lines = _diagonal_lines(R, 2)
    if len(diag_lines) > 0 and n_recurrent > 0:
        det = min(diag_lines.sum() / (n_recurrent / 2), 1.0)
        avg_diag = diag_lines.mean()
        max_diag = diag_lines.max()
    else:
        det = 0.0
        avg_diag = 0.0
        max_diag = 0.0

    deriv.add_step(
        title="Determinism (DET) - Diagonal Line Analysis",
        equation="DET = Σₗ≥₂ l·P(l) / Σᵢⱼ R(i,j)",
        calculation=f"Diagonal lines found: {len(diag_lines)}\nLine lengths: {diag_lines[:10].tolist() if len(diag_lines) > 0 else []}\nSum of line lengths: {diag_lines.sum() if len(diag_lines) > 0 else 0}\n\nDET = {diag_lines.sum() if len(diag_lines) > 0 else 0} / {n_recurrent / 2:.0f}",
        result=det,
        result_name="DET",
        notes="High DET = deterministic dynamics; diagonal lines = predictable evolution"
    )

    # Step 7: Vertical lines (laminarity)
    vert_lines = _vertical_lines(R, 2)
    if len(vert_lines) > 0 and n_recurrent > 0:
        lam = min(vert_lines.sum() / (n_recurrent / 2), 1.0)
        avg_vert = vert_lines.mean()
    else:
        lam = 0.0
        avg_vert = 0.0

    deriv.add_step(
        title="Laminarity (LAM) - Vertical Line Analysis",
        equation="LAM = Σᵥ≥₂ v·P(v) / Σᵢⱼ R(i,j)",
        calculation=f"Vertical lines found: {len(vert_lines)}\nLine lengths: {vert_lines[:10].tolist() if len(vert_lines) > 0 else []}\nSum of line lengths: {vert_lines.sum() if len(vert_lines) > 0 else 0}\n\nLAM = {vert_lines.sum() if len(vert_lines) > 0 else 0} / {n_recurrent / 2:.0f}",
        result=lam,
        result_name="LAM",
        notes="High LAM = laminar (trapped) states; system stays in similar states"
    )

    # Step 8: Entropy
    if len(diag_lines) > 0:
        hist, _ = np.histogram(diag_lines, bins=range(2, int(max_diag) + 2))
        hist = hist[hist > 0]
        if len(hist) > 0:
            p = hist / hist.sum()
            entropy = -np.sum(p * np.log(p))
        else:
            entropy = 0.0
    else:
        entropy = 0.0

    deriv.add_step(
        title="Entropy of Diagonal Line Distribution",
        equation="ENTR = -Σₗ p(l) · ln(p(l))",
        calculation=f"Unique line lengths: {len(np.unique(diag_lines)) if len(diag_lines) > 0 else 0}\nLine length distribution entropy",
        result=entropy,
        result_name="ENTR",
        notes="Higher entropy = more complex dynamics"
    )

    # Final result
    result = {
        "recurrence_rate": float(rr),
        "determinism": float(det),
        "laminarity": float(lam),
        "avg_diagonal_length": float(avg_diag),
        "avg_vertical_length": float(avg_vert),
        "entropy": float(entropy),
        "max_diagonal_length": float(max_diag),
    }

    deriv.final_result = det
    deriv.prism_output = det

    # Interpretation
    if det > 0.8:
        interp = f"DET = {det:.3f} indicates **highly deterministic** dynamics. The system evolution is predictable."
    elif det > 0.5:
        interp = f"DET = {det:.3f} indicates **moderately deterministic** dynamics with some stochastic components."
    else:
        interp = f"DET = {det:.3f} indicates **stochastic** dynamics. The system shows weak predictability."

    if lam > 0.8:
        interp += f" LAM = {lam:.3f} indicates the system frequently enters **laminar states** (trapped regions)."

    deriv.interpretation = interp

    return result, deriv


def _compute_rqa_metrics(R: np.ndarray, min_line: int = 2) -> Dict[str, float]:
    """
    Compute RQA metrics from recurrence matrix.
    """
    n = R.shape[0]

    if n == 0:
        return {
            "recurrence_rate": 0.0,
            "determinism": 0.0,
            "laminarity": 0.0,
            "avg_diagonal_length": 0.0,
            "avg_vertical_length": 0.0,
            "entropy": 0.0,
            "max_diagonal_length": 0.0,
        }

    # Recurrence rate: fraction of recurrent points
    n_recurrent = R.sum() - n  # Exclude main diagonal
    rr = n_recurrent / (n * (n - 1)) if n > 1 else 0.0

    # Diagonal lines
    diag_lines = _diagonal_lines(R, min_line)

    # Determinism: fraction of recurrent points in diagonal lines
    if n_recurrent > 0 and len(diag_lines) > 0:
        det = diag_lines.sum() / (n_recurrent / 2)  # Divide by 2 for symmetry
        det = min(det, 1.0)  # Cap at 1
    else:
        det = 0.0

    # Average diagonal line length
    avg_diag = diag_lines.mean() if len(diag_lines) > 0 else 0.0
    max_diag = diag_lines.max() if len(diag_lines) > 0 else 0.0

    # Vertical lines (for laminarity)
    vert_lines = _vertical_lines(R, min_line)

    # Laminarity: fraction of recurrent points in vertical lines
    if n_recurrent > 0 and len(vert_lines) > 0:
        lam = vert_lines.sum() / (n_recurrent / 2)
        lam = min(lam, 1.0)
    else:
        lam = 0.0

    avg_vert = vert_lines.mean() if len(vert_lines) > 0 else 0.0

    # Entropy of diagonal line distribution
    if len(diag_lines) > 0:
        hist, _ = np.histogram(diag_lines, bins=range(min_line, int(max_diag) + 2))
        hist = hist[hist > 0]
        if len(hist) > 0:
            p = hist / hist.sum()
            entropy = -np.sum(p * np.log(p))
        else:
            entropy = 0.0
    else:
        entropy = 0.0

    return {
        "recurrence_rate": float(rr),
        "determinism": float(det),
        "laminarity": float(lam),
        "avg_diagonal_length": float(avg_diag),
        "avg_vertical_length": float(avg_vert),
        "entropy": float(entropy),
        "max_diagonal_length": float(max_diag),
    }


class RQAEngine(BaseEngine):
    """
    Recurrence Quantification Analysis engine.

    Analyzes system dynamics through recurrence plots,
    capturing determinism, laminarity, and complexity.

    Outputs:
        - results.rqa_metrics: Per-signal RQA metrics
    """

    name = "rqa"
    phase = "derived"
    default_normalization = "zscore"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        embedding_dim: int = 3,
        time_delay: int = 1,
        threshold_percentile: float = 10.0,
        min_line_length: int = 2,
        **params
    ) -> Dict[str, Any]:
        """
        Run RQA analysis.

        Args:
            df: Normalized signal data
            run_id: Unique run identifier
            embedding_dim: Embedding dimension (default 3)
            time_delay: Time delay for embedding (default 1)
            threshold_percentile: Percentile for recurrence threshold
            min_line_length: Minimum diagonal/vertical line length

        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = df_clean.columns.tolist()
        n_signals = len(signals)

        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()

        # Compute RQA for each signal
        records = []
        all_rr = []
        all_det = []
        all_lam = []
        all_entropy = []

        for signal in signals:
            x = df_clean[signal].values

            # Embed signal topology
            embedded = _embed_signal_topology(x, embedding_dim, time_delay)

            if embedded.shape[0] < 10:
                logger.warning(f"Insufficient data for RQA on {signal}")
                continue

            # Determine threshold from distance distribution
            diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
            nonzero_distances = distances[distances > 0]

            # Handle constant data (all distances = 0)
            if len(nonzero_distances) == 0:
                n = embedded.shape[0]
                rqa = {
                    "recurrence_rate": 1.0,
                    "determinism": 1.0,
                    "laminarity": 1.0,
                    "avg_diagonal_length": float(n - 1),
                    "avg_vertical_length": float(n),
                    "entropy": 0.0,
                    "max_diagonal_length": float(n - 1),
                }
            else:
                threshold = np.percentile(nonzero_distances, threshold_percentile)

                # Compute recurrence matrix
                R = _recurrence_matrix(embedded, threshold)

                # Compute RQA metrics
                rqa = _compute_rqa_metrics(R, min_line_length)

            all_rr.append(rqa["recurrence_rate"])
            all_det.append(rqa["determinism"])
            all_lam.append(rqa["laminarity"])
            all_entropy.append(rqa["entropy"])

            records.append({
                "signal_id": signal,
                "window_start": window_start,
                "window_end": window_end,
                "recurrence_rate": rqa["recurrence_rate"],
                "determinism": rqa["determinism"],
                "laminarity": rqa["laminarity"],
                "avg_diagonal_length": rqa["avg_diagonal_length"],
                "avg_vertical_length": rqa["avg_vertical_length"],
                "entropy": rqa["entropy"],
                "max_diagonal_length": rqa["max_diagonal_length"],
                "embedding_dim": embedding_dim,
                "time_delay": time_delay,
                "threshold_percentile": threshold_percentile,
                "run_id": run_id,
            })

        if records:
            df_results = pd.DataFrame(records)
            self.store_results("rqa_metrics", df_results, run_id)

        # Summary metrics
        metrics = {
            "n_signals": n_signals,
            "n_samples": len(df_clean),
            "embedding_dim": embedding_dim,
            "time_delay": time_delay,
            "threshold_percentile": threshold_percentile,
            "avg_recurrence_rate": float(np.mean(all_rr)) if all_rr else 0.0,
            "avg_determinism": float(np.mean(all_det)) if all_det else 0.0,
            "avg_laminarity": float(np.mean(all_lam)) if all_lam else 0.0,
            "avg_entropy": float(np.mean(all_entropy)) if all_entropy else 0.0,
        }

        logger.info(
            f"RQA complete: {n_signals} signals, "
            f"avg_det={metrics['avg_determinism']:.4f}, "
            f"avg_lam={metrics['avg_laminarity']:.4f}"
        )

        return metrics
