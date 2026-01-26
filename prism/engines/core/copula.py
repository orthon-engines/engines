"""
PRISM Copula Dependence Engine

Measures tail dependence and non-linear dependence structure.

Measures:
- Tail dependence coefficients (upper/lower)
- Kendall's tau (rank correlation)
- Copula-based dependence metrics

Phase: Unbound
Normalization: Rank transform (to uniform)
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="copula",
    engine_type="geometry",
    description="Tail dependence and non-linear dependence structure",
    domains={"dependence", "tail_risk"},
    requires_window=True,
    deterministic=True,
)


def _empirical_copula(u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    """
    Estimate tail dependence from empirical copula.

    Returns (lower_tail, upper_tail) dependence coefficients.
    """
    n = len(u)

    # Lower tail: lambda_L = lim_{q->0} P(V <= q | U <= q)
    # Upper tail: lambda_U = lim_{q->1} P(V > q | U > q)

    # Use threshold approach
    thresholds = [0.05, 0.10, 0.15]

    lower_deps = []
    upper_deps = []

    for q in thresholds:
        # Lower tail
        mask_lower = u <= q
        if mask_lower.sum() > 0:
            lower_deps.append((v[mask_lower] <= q).mean())

        # Upper tail
        mask_upper = u >= (1 - q)
        if mask_upper.sum() > 0:
            upper_deps.append((v[mask_upper] >= (1 - q)).mean())

    lower_tail = np.mean(lower_deps) if lower_deps else 0.0
    upper_tail = np.mean(upper_deps) if upper_deps else 0.0

    return lower_tail, upper_tail


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Kendall's tau rank correlation."""
    tau, _ = stats.kendalltau(x, y)
    return tau if not np.isnan(tau) else 0.0


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman's rho rank correlation."""
    rho, _ = stats.spearmanr(x, y)
    return rho if not np.isnan(rho) else 0.0


class CopulaEngine(BaseEngine):
    """
    Copula dependence engine.

    Analyzes tail dependence and non-linear dependence structure
    using copula-based methods.

    Outputs:
        - results.copula_dependence: Pairwise dependence metrics
    """

    name = "copula"
    phase = "derived"
    default_normalization = None  # We do rank transform internally

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        **params
    ) -> Dict[str, Any]:
        """
        Run copula dependence analysis.

        Args:
            df: Signal data (will be rank-transformed internally)
            run_id: Unique run identifier

        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = df_clean.columns.tolist()
        n_signals = len(signals)

        window_start, window_end = get_window_dates(df_clean)

        # Transform to uniform marginals (probability integral transform)
        df_uniform = df_clean.rank(pct=True)

        # Compute pairwise dependence metrics
        records = []

        lower_tails = []
        upper_tails = []
        kendalls = []
        spearmans = []

        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                u = df_uniform.iloc[:, i].values
                v = df_uniform.iloc[:, j].values

                # Tail dependence
                lower_tail, upper_tail = _empirical_copula(u, v)

                # Rank correlations
                tau = _kendall_tau(df_clean.iloc[:, i].values, df_clean.iloc[:, j].values)
                rho = _spearman_rho(df_clean.iloc[:, i].values, df_clean.iloc[:, j].values)

                lower_tails.append(lower_tail)
                upper_tails.append(upper_tail)
                kendalls.append(tau)
                spearmans.append(rho)

                records.append({
                    "signal_1": signals[i],
                    "signal_2": signals[j],
                    "window_start": window_start,
                    "window_end": window_end,
                    "lower_tail_dependence": float(lower_tail),
                    "upper_tail_dependence": float(upper_tail),
                    "kendall_tau": float(tau),
                    "spearman_rho": float(rho),
                    "run_id": run_id,
                })

        if records:
            df_results = pd.DataFrame(records)
            self.store_results("copula_dependence", df_results, run_id)

        # Summary metrics
        n_pairs = len(records)

        metrics = {
            "n_signals": n_signals,
            "n_pairs": n_pairs,
            "n_samples": len(df_clean),
            "avg_lower_tail": float(np.mean(lower_tails)) if lower_tails else 0.0,
            "avg_upper_tail": float(np.mean(upper_tails)) if upper_tails else 0.0,
            "max_lower_tail": float(np.max(lower_tails)) if lower_tails else 0.0,
            "max_upper_tail": float(np.max(upper_tails)) if upper_tails else 0.0,
            "avg_kendall_tau": float(np.mean(kendalls)) if kendalls else 0.0,
            "avg_spearman_rho": float(np.mean(spearmans)) if spearmans else 0.0,
            "tail_asymmetry": float(np.mean(upper_tails) - np.mean(lower_tails)) if upper_tails else 0.0,
        }

        logger.info(
            f"Copula complete: {n_signals} signals, "
            f"avg_lower_tail={metrics['avg_lower_tail']:.4f}, "
            f"avg_upper_tail={metrics['avg_upper_tail']:.4f}"
        )

        return metrics
