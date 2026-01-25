"""
PRISM Cointegration Engine

Tests for long-run equilibrium relationships between series.

Measures:
- Engle-Granger test statistics (pairwise)
- Cointegrating vectors (hedge ratios)
- Error correction speeds

Phase: Unbound
Normalization: None (uses levels, not returns)
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
    name="cointegration",
    engine_type="geometry",
    description="Long-run equilibrium relationships between series",
    domains={"dependence", "equilibrium"},
    requires_window=True,
    deterministic=True,
)


def _adf_test(x: np.ndarray, max_lag: int = None) -> Tuple[float, float, int]:
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns (adf_stat, p_value, lags_used)
    """
    n = len(x)

    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** 0.25))

    max_lag = min(max_lag, n // 4)

    # First difference
    dx = np.diff(x)

    # Lagged level
    x_lag = x[:-1]

    # Build regression matrix with lagged differences
    best_aic = np.inf
    best_result = None

    for lag in range(max_lag + 1):
        if lag == 0:
            X = np.column_stack([np.ones(len(dx)), x_lag])
            y = dx
        else:
            # Include lagged differences
            n_obs = len(dx) - lag
            dx_lags = []
            for i in range(lag):
                start = lag - i - 1
                end = start + n_obs
                dx_lags.append(dx[start:end])

            dx_lags = np.column_stack(dx_lags)

            X = np.column_stack([
                np.ones(n_obs),
                x_lag[lag:lag + n_obs],
                dx_lags
            ])
            y = dx[lag:lag + n_obs]

        if len(y) < X.shape[1] + 2:
            continue

        try:
            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta

            # AIC for lag selection
            n_obs = len(y)
            k = X.shape[1]
            sse = np.sum(residuals ** 2)
            aic = n_obs * np.log(sse / n_obs) + 2 * k

            if aic < best_aic:
                best_aic = aic

                # Standard error of gamma coefficient
                mse = sse / (n_obs - k)
                var_beta = mse * np.linalg.inv(X.T @ X)
                se_gamma = np.sqrt(var_beta[1, 1])

                gamma = beta[1]
                adf_stat = gamma / se_gamma
                best_result = (adf_stat, lag)

        except (np.linalg.LinAlgError, ValueError):
            continue

    if best_result is None:
        return None, None, None

    adf_stat, lags = best_result

    # Approximate p-value using MacKinnon critical values
    # For n > 500, critical values: 1%: -3.43, 5%: -2.86, 10%: -2.57
    if adf_stat < -3.43:
        p_value = 0.01
    elif adf_stat < -2.86:
        p_value = 0.05
    elif adf_stat < -2.57:
        p_value = 0.10
    else:
        p_value = 0.5  # Not significant

    return float(adf_stat), p_value, lags


def _engle_granger_test(y: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    """
    Engle-Granger two-step cointegration test.

    Step 1: Regress y on x
    Step 2: Test residuals for stationarity
    """
    n = len(y)

    # Step 1: OLS regression
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    alpha = beta[0]  # Intercept
    hedge_ratio = beta[1]  # Cointegrating coefficient

    # Residuals (spread)
    residuals = y - alpha - hedge_ratio * x

    # Step 2: ADF test on residuals
    adf_stat, p_value, lags = _adf_test(residuals)

    # Half-life of mean reversion (if cointegrated)
    if p_value < 0.10:
        # AR(1) on residuals
        resid_lag = residuals[:-1]
        resid_diff = np.diff(residuals)

        if len(resid_lag) > 2:
            rho = np.corrcoef(resid_lag, resid_diff + resid_lag)[0, 1]
            if rho > 0 and rho < 1:
                half_life = -np.log(2) / np.log(rho)
            else:
                half_life = np.nan
        else:
            half_life = np.nan
    else:
        half_life = np.nan

    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "hedge_ratio": float(hedge_ratio),
        "intercept": float(alpha),
        "half_life": float(half_life) if not np.isnan(half_life) else None,
        "spread_std": float(np.std(residuals)),
        "is_cointegrated": p_value < 0.10,
    }


class CointegrationEngine(BaseEngine):
    """
    Cointegration engine.

    Tests for long-run equilibrium relationships using
    Engle-Granger methodology.

    Outputs:
        - results.cointegration: Pairwise cointegration results
    """

    name = "cointegration"
    phase = "derived"
    default_normalization = None  # Uses levels

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        significance: float = 0.10,
        **params
    ) -> Dict[str, Any]:
        """
        Run cointegration analysis.

        Args:
            df: Signal data (levels, not returns)
            run_id: Unique run identifier
            significance: Significance level for cointegration test

        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = df_clean.columns.tolist()
        n_signals = len(signals)

        window_start, window_end = get_window_dates(df_clean)

        # Test all pairs
        records = []
        n_cointegrated = 0
        all_half_lives = []

        for i in range(n_signals):
            for j in range(i + 1, n_signals):
                y = df_clean.iloc[:, i].values
                x = df_clean.iloc[:, j].values

                try:
                    result = _engle_granger_test(y, x)

                    if result["is_cointegrated"]:
                        n_cointegrated += 1
                        if result["half_life"] is not None:
                            all_half_lives.append(result["half_life"])

                    records.append({
                        "signal_1": signals[i],
                        "signal_2": signals[j],
                        "window_start": window_start,
                        "window_end": window_end,
                        "adf_stat": result["adf_stat"],
                        "p_value": result["p_value"],
                        "hedge_ratio": result["hedge_ratio"],
                        "intercept": result["intercept"],
                        "half_life": result["half_life"],
                        "spread_std": result["spread_std"],
                        "is_cointegrated": result["is_cointegrated"],
                        "run_id": run_id,
                    })

                except Exception as e:
                    logger.warning(f"Cointegration test failed for {signals[i]}-{signals[j]}: {e}")
                    continue

        if records:
            df_results = pd.DataFrame(records)
            ##self.store_results("cointegration", df_results, run_id)

        # Summary metrics
        n_pairs = len(records)

        metrics = {
            "n_signals": n_signals,
            "n_pairs": n_pairs,
            "n_samples": len(df_clean),
            "n_cointegrated": n_cointegrated,
            "cointegration_rate": n_cointegrated / n_pairs if n_pairs > 0 else 0.0,
            "avg_half_life": float(np.mean(all_half_lives)) if all_half_lives else None,
            "significance_level": significance,
        }

        logger.info(
            f"Cointegration complete: {n_signals} signals, "
            f"{n_cointegrated}/{n_pairs} cointegrated pairs"
        )

        return metrics


# =============================================================================
# Standalone function with derivation
# =============================================================================

def compute_cointegration_with_derivation(
    y: np.ndarray,
    x: np.ndarray,
    signal_y: str = "Y",
    signal_x: str = "X",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
) -> tuple:
    """
    Compute Engle-Granger cointegration test with full mathematical derivation.

    Args:
        y: Dependent series (levels, not returns)
        x: Independent series (levels, not returns)
        signal_y: Name of Y signal
        signal_x: Name of X signal
        window_id: Window identifier
        window_start, window_end: Date range

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from prism.entry_points.derivations.base import Derivation

    n = len(y)

    deriv = Derivation(
        engine_name="cointegration",
        method_name="Engle-Granger Two-Step Cointegration Test",
        signal_id=f"{signal_y}_vs_{signal_x}",
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=n,
        parameters={'method': 'engle_granger'}
    )

    # Step 1: Problem statement
    deriv.add_step(
        title="Cointegration Hypothesis",
        equation="H₀: y and x are NOT cointegrated (residuals non-stationary)",
        calculation=f"Testing if {signal_y} and {signal_x} share a long-run equilibrium\n"
                    f"n = {n} observations\n\n"
                    f"Cointegration means: y_t = α + β·x_t + ε_t where ε_t is I(0) stationary",
        result=n,
        result_name="n",
        notes="Two series are cointegrated if a linear combination is stationary"
    )

    # Step 2: OLS regression (Step 1 of Engle-Granger)
    X_design = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    alpha = beta[0]
    hedge_ratio = beta[1]

    deriv.add_step(
        title="Step 1: OLS Cointegrating Regression",
        equation="y_t = α + β·x_t + ε_t (estimate by OLS)",
        calculation=f"Regress {signal_y} on {signal_x}:\n"
                    f"  α (intercept) = {alpha:.6f}\n"
                    f"  β (hedge ratio) = {hedge_ratio:.6f}\n\n"
                    f"Cointegrating relationship:\n"
                    f"  {signal_y} = {alpha:.4f} + {hedge_ratio:.4f}·{signal_x} + ε",
        result=hedge_ratio,
        result_name="β",
        notes="β is the hedge ratio: how much X to hold per unit of Y"
    )

    # Step 3: Compute residuals (spread)
    residuals = y - alpha - hedge_ratio * x
    spread_mean = np.mean(residuals)
    spread_std = np.std(residuals)

    deriv.add_step(
        title="Compute Residuals (Spread)",
        equation="ε̂_t = y_t - α̂ - β̂·x_t",
        calculation=f"Spread statistics:\n"
                    f"  Mean: {spread_mean:.6f} (should be ≈0)\n"
                    f"  Std: {spread_std:.6f}\n"
                    f"  Min: {np.min(residuals):.6f}\n"
                    f"  Max: {np.max(residuals):.6f}\n\n"
                    f"If cointegrated, spread should revert to mean",
        result=spread_std,
        result_name="σ_ε",
        notes="Spread is the 'error correction' term that should be stationary"
    )

    # Step 4: ADF test on residuals (Step 2 of Engle-Granger)
    adf_stat, p_value, lags = _adf_test(residuals)

    deriv.add_step(
        title="Step 2: ADF Test on Residuals",
        equation="Δε̂_t = γ·ε̂_{t-1} + Σδᵢ·Δε̂_{t-i} + u_t",
        calculation=f"Augmented Dickey-Fuller test:\n"
                    f"  H₀: γ = 0 (unit root, non-stationary)\n"
                    f"  H₁: γ < 0 (stationary)\n\n"
                    f"  ADF statistic: {adf_stat:.4f}\n"
                    f"  Lags used: {lags}\n\n"
                    f"Critical values (Engle-Granger):\n"
                    f"  1%: -3.43\n"
                    f"  5%: -2.86\n"
                    f"  10%: -2.57",
        result=adf_stat,
        result_name="ADF",
        notes="More negative ADF stat = stronger evidence of stationarity"
    )

    # Step 5: P-value and decision
    is_cointegrated = p_value < 0.10

    deriv.add_step(
        title="Statistical Decision",
        equation="Reject H₀ if ADF < critical value",
        calculation=f"P-value: {p_value:.4f}\n\n"
                    f"Decision at α = 0.10:\n"
                    f"  ADF = {adf_stat:.4f}\n"
                    f"  Critical (10%) = -2.57\n"
                    f"  {'REJECT H₀' if is_cointegrated else 'FAIL TO REJECT H₀'}\n\n"
                    f"Conclusion: {signal_y} and {signal_x} "
                    f"{'ARE' if is_cointegrated else 'are NOT'} cointegrated",
        result=p_value,
        result_name="p",
        notes="If cointegrated, deviations from equilibrium are temporary"
    )

    # Step 6: Half-life of mean reversion (if cointegrated)
    if is_cointegrated and len(residuals) > 3:
        resid_lag = residuals[:-1]
        resid_next = residuals[1:]
        rho = np.corrcoef(resid_lag, resid_next)[0, 1]
        if 0 < rho < 1:
            half_life = -np.log(2) / np.log(rho)
        else:
            half_life = np.nan
    else:
        half_life = np.nan
        rho = np.nan

    deriv.add_step(
        title="Half-Life of Mean Reversion",
        equation="t_{1/2} = -ln(2) / ln(ρ) where ε_t = ρ·ε_{t-1} + u_t",
        calculation=f"AR(1) on residuals:\n"
                    f"  ρ (autocorrelation) = {rho:.4f if not np.isnan(rho) else 'N/A'}\n\n"
                    f"Half-life calculation:\n"
                    f"  t_{{1/2}} = -ln(2) / ln({rho:.4f if not np.isnan(rho) else 'N/A'})\n"
                    f"  t_{{1/2}} = {half_life:.2f if not np.isnan(half_life) else 'N/A'} periods\n\n"
                    f"Interpretation: Spread reverts halfway to mean in ~{half_life:.1f if not np.isnan(half_life) else 'N/A'} periods",
        result=half_life if not np.isnan(half_life) else None,
        result_name="t_{1/2}",
        notes="Half-life indicates speed of mean reversion"
    )

    # Final result
    result = {
        'adf_stat': float(adf_stat),
        'p_value': float(p_value),
        'hedge_ratio': float(hedge_ratio),
        'intercept': float(alpha),
        'half_life': float(half_life) if not np.isnan(half_life) else None,
        'spread_std': float(spread_std),
        'is_cointegrated': is_cointegrated,
    }

    deriv.final_result = adf_stat
    deriv.prism_output = adf_stat

    # Interpretation
    if is_cointegrated:
        interp = f"**Cointegrated** (p={p_value:.3f}): {signal_y} and {signal_x} share long-run equilibrium."
        interp += f" Hedge ratio β={hedge_ratio:.3f}."
        if not np.isnan(half_life):
            interp += f" Deviations revert with half-life ~{half_life:.1f} periods."
    else:
        interp = f"**Not cointegrated** (p={p_value:.3f}): No long-run equilibrium relationship detected."
        interp += " Spread may drift without bound."

    deriv.interpretation = interp

    return result, deriv
