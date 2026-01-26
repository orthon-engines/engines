"""
PRISM Granger Causality Engine

Tests whether past values of X improve prediction of Y.

Measures:
- F-statistic and p-value per pair
- Directional causality network
- Optimal lag structure

Phase: Unbound
Normalization: None/Z-score (stationarity required)
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import date
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="granger",
    engine_type="geometry",
    description="Granger causality tests for predictive relationships",
    domains={"causality", "dependence"},
    requires_window=True,
    deterministic=True,
)


class GrangerEngine(BaseEngine):
    """
    Granger Causality engine.
    
    Tests pairwise Granger causality between signals.
    
    Outputs:
        - results.granger_causality: Pairwise causality tests
    """
    
    name = "granger"
    phase = "derived"
    default_normalization = None  # Handle stationarity internally

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        max_lag: int = 5,
        significance: float = 0.05,
        ensure_stationary: bool = True,
        **params
    ) -> Dict[str, Any]:
        """
        Run Granger causality analysis.
        
        Args:
            df: Signal data
            run_id: Unique run identifier
            max_lag: Maximum lag to test (default 5)
            significance: Significance level (default 0.05)
            ensure_stationary: Difference if non-stationary (default True)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = list(df_clean.columns)
        n = len(signals)
        
        window_start, window_end = get_window_dates(df_clean)
        
        # Ensure stationarity if requested
        if ensure_stationary:
            df_stationary = self._ensure_stationarity(df_clean)
        else:
            df_stationary = df_clean
        
        # Test all pairs
        results = []
        significant_count = 0
        
        for i, ind1 in enumerate(signals):
            for j, ind2 in enumerate(signals):
                if i == j:
                    continue
                
                x = df_stationary[ind1].values
                y = df_stationary[ind2].values
                
                # Test if ind1 Granger-causes ind2
                try:
                    f_stat, p_value, optimal_lag = self._granger_test(
                        x, y, max_lag
                    )
                    
                    is_significant = p_value < significance
                    if is_significant:
                        significant_count += 1
                    
                    results.append({
                        "signal_from": ind1,
                        "signal_to": ind2,
                        "window_start": window_start,
                        "window_end": window_end,
                        "lag": int(optimal_lag),
                        "f_statistic": float(f_stat),
                        "p_value": float(p_value),
                        "run_id": run_id,
                    })
                    
                except Exception as e:
                    logger.debug(f"Granger test failed for {ind1} -> {ind2}: {e}")
        
        # Note: Detailed pairwise results stored internally
        # Summary metrics returned to geometry.py → results.geometry
        # The store_results call is disabled to avoid schema mismatch
        # with results.granger_causality (signal_from vs signal_1)
        
        # Bonferroni correction info
        n_tests = n * (n - 1)
        bonferroni_threshold = significance / n_tests if n_tests > 0 else significance
        
        metrics = {
            "n_signals": n,
            "n_tests": len(results),
            "significant_pairs": significant_count,
            "significance_level": significance,
            "bonferroni_threshold": float(bonferroni_threshold),
            "max_lag_tested": max_lag,
        }
        
        logger.info(
            f"Granger causality complete: {significant_count}/{len(results)} "
            f"significant pairs at p<{significance}"
        )
        
        return metrics
    
    def _ensure_stationarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Difference non-stationary series.
        Uses simple first difference (could use ADF test for rigor).
        """
        # Simple approach: always difference
        # More rigorous: use ADF test per series
        return df.diff().dropna()
    
    def _granger_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int
    ) -> Tuple[float, float, int]:
        """
        Test if x Granger-causes y.
        
        Returns (F-statistic, p-value, optimal_lag)
        """
        best_f = 0
        best_p = 1.0
        best_lag = 1
        
        for lag in range(1, max_lag + 1):
            try:
                f_stat, p_value = self._granger_test_single_lag(x, y, lag)
                
                if f_stat > best_f:
                    best_f = f_stat
                    best_p = p_value
                    best_lag = lag
                    
            except Exception:
                continue
        
        return best_f, best_p, best_lag
    
    def _granger_test_single_lag(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int
    ) -> Tuple[float, float]:
        """
        Granger causality test at a single lag.
        
        Compares:
        - Restricted model: y_t = a0 + a1*y_{t-1} + ... + a_lag*y_{t-lag}
        - Unrestricted model: y_t = a0 + a1*y_{t-1} + ... + b1*x_{t-1} + ... + b_lag*x_{t-lag}
        
        F-test on the additional explanatory power of x.
        """
        n = len(y)
        
        if n <= 2 * lag + 1:
            raise ValueError("Not enough observations for this lag")
        
        # Build lagged matrices
        y_target = y[lag:]
        n_obs = len(y_target)
        
        # Restricted model regressors (only y lags)
        X_restricted = np.column_stack([
            y[lag-i-1:n-i-1] for i in range(lag)
        ])
        X_restricted = np.column_stack([np.ones(n_obs), X_restricted])
        
        # Unrestricted model regressors (y lags + x lags)
        X_unrestricted = np.column_stack([
            X_restricted,
            *[x[lag-i-1:n-i-1] for i in range(lag)]
        ])
        
        # Fit both models
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Restricted model
            beta_r = np.linalg.lstsq(X_restricted, y_target, rcond=None)[0]
            resid_r = y_target - X_restricted @ beta_r
            rss_r = np.sum(resid_r ** 2)
            
            # Unrestricted model
            beta_u = np.linalg.lstsq(X_unrestricted, y_target, rcond=None)[0]
            resid_u = y_target - X_unrestricted @ beta_u
            rss_u = np.sum(resid_u ** 2)
        
        # F-test
        df_num = lag  # Number of restrictions
        df_den = n_obs - X_unrestricted.shape[1]
        
        if df_den <= 0 or rss_u <= 0:
            raise ValueError("Invalid degrees of freedom")
        
        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
        p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)
        
        return f_stat, p_value


# =============================================================================
# Standalone function with derivation
# =============================================================================

def compute_granger_with_derivation(
    x: np.ndarray,
    y: np.ndarray,
    signal_x: str = "X",
    signal_y: str = "Y",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    max_lag: int = 5,
) -> tuple:
    """
    Compute Granger causality test with full mathematical derivation.

    Tests: Does X Granger-cause Y? (Do past values of X help predict Y?)

    Args:
        x: Signal X (potential cause)
        y: Signal Y (potential effect)
        signal_x, signal_y: Names for the signals
        window_id: Window identifier
        max_lag: Maximum lag to test

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from prism.entry_points.derivations.base import Derivation

    n = len(y)

    deriv = Derivation(
        engine_name="granger_causality",
        method_name="Granger Causality F-Test",
        signal_id=f"{signal_x}_causes_{signal_y}",
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=n,
        raw_data_sample=y[:10].tolist() if len(y) >= 10 else y.tolist(),
        parameters={'max_lag': max_lag}
    )

    if n < 20:
        deriv.final_result = None
        deriv.interpretation = "Insufficient data (n < 20)"
        return {}, deriv

    # Step 1: Input data
    deriv.add_step(
        title="Input Signal Topology",
        equation="X = {x₁, ..., xₙ}, Y = {y₁, ..., yₙ}",
        calculation=f"X ({signal_x}): n={len(x)}, mean={np.mean(x):.4f}, std={np.std(x):.4f}\nY ({signal_y}): n={len(y)}, mean={np.mean(y):.4f}, std={np.std(y):.4f}\n\nQuestion: Does X Granger-cause Y?\n(Do past values of X help predict Y beyond Y's own history?)",
        result=n,
        result_name="n",
        notes="Granger causality ≠ true causality, but tests predictive relationship"
    )

    # Step 2: Stationarity (differencing)
    x_diff = np.diff(x)
    y_diff = np.diff(y)

    deriv.add_step(
        title="Ensure Stationarity (First Difference)",
        equation="Δxₜ = xₜ - xₜ₋₁,  Δyₜ = yₜ - yₜ₋₁",
        calculation=f"Original X: mean={np.mean(x):.4f}\nDifferenced ΔX: mean={np.mean(x_diff):.4f} (should be ≈0)\n\nOriginal Y: mean={np.mean(y):.4f}\nDifferenced ΔY: mean={np.mean(y_diff):.4f}",
        result=len(x_diff),
        result_name="n_diff",
        notes="Differencing removes trends; Granger requires stationary series"
    )

    # Step 3: Choose optimal lag
    best_f = 0
    best_p = 1.0
    best_lag = 1
    lag_results = []

    for lag in range(1, min(max_lag + 1, len(y_diff) // 3)):
        try:
            n_obs = len(y_diff) - lag
            if n_obs <= 2 * lag + 1:
                continue

            y_target = y_diff[lag:]

            # Restricted model: Y ~ Y_lags only
            X_r = np.column_stack([np.ones(n_obs)] + [y_diff[lag-i-1:len(y_diff)-i-1] for i in range(lag)])
            beta_r = np.linalg.lstsq(X_r, y_target, rcond=None)[0]
            rss_r = np.sum((y_target - X_r @ beta_r) ** 2)

            # Unrestricted model: Y ~ Y_lags + X_lags
            X_u = np.column_stack([X_r] + [x_diff[lag-i-1:len(x_diff)-i-1] for i in range(lag)])
            beta_u = np.linalg.lstsq(X_u, y_target, rcond=None)[0]
            rss_u = np.sum((y_target - X_u @ beta_u) ** 2)

            # F-test
            df_num = lag
            df_den = n_obs - X_u.shape[1]
            if df_den > 0 and rss_u > 0:
                f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
                p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)
                lag_results.append((lag, f_stat, p_value, rss_r, rss_u))

                if f_stat > best_f:
                    best_f = f_stat
                    best_p = p_value
                    best_lag = lag
        except:
            continue

    if not lag_results:
        deriv.interpretation = "Could not compute Granger test - insufficient data"
        return {}, deriv

    # Step 4: Show the models
    deriv.add_step(
        title="Define Restricted and Unrestricted Models",
        equation="Restricted:   Yₜ = α₀ + Σᵢ αᵢYₜ₋ᵢ + εₜ\nUnrestricted: Yₜ = α₀ + Σᵢ αᵢYₜ₋ᵢ + Σⱼ βⱼXₜ₋ⱼ + εₜ",
        calculation=f"At lag = {best_lag}:\n\nRestricted model (Y only):\n  Yₜ = α₀ + α₁Yₜ₋₁ + ... + α_{best_lag}Yₜ₋{best_lag}\n\nUnrestricted model (Y + X):\n  Yₜ = α₀ + α₁Yₜ₋₁ + ... + β₁Xₜ₋₁ + ... + β_{best_lag}Xₜ₋{best_lag}",
        result=best_lag,
        result_name="lag",
        notes="Null hypothesis H₀: β₁ = β₂ = ... = β_lag = 0 (X adds no predictive power)"
    )

    # Step 5: Show RSS comparison
    best_result = [r for r in lag_results if r[0] == best_lag][0]
    _, f_stat, p_val, rss_r, rss_u = best_result

    deriv.add_step(
        title="Compare Residual Sum of Squares",
        equation="RSS_r = Σ(Yₜ - Ŷₜ|restricted)²\nRSS_u = Σ(Yₜ - Ŷₜ|unrestricted)²",
        calculation=f"RSS_restricted (Y only): {rss_r:.4f}\nRSS_unrestricted (Y + X): {rss_u:.4f}\n\nReduction: {rss_r - rss_u:.4f} ({(rss_r - rss_u)/rss_r*100:.1f}%)\n\nIf X helps predict Y, RSS_u < RSS_r",
        result=rss_u,
        result_name="RSS_u",
        notes="Lower RSS = better fit; question is whether improvement is significant"
    )

    # Step 6: F-test
    n_obs = len(y_diff) - best_lag
    df_num = best_lag
    df_den = n_obs - (1 + 2*best_lag)

    deriv.add_step(
        title="F-Test for Significance",
        equation="F = [(RSS_r - RSS_u)/q] / [RSS_u/(n-k)]",
        calculation=f"q = {df_num} (number of restrictions = lag)\nn-k = {df_den} (degrees of freedom)\n\nF = [({rss_r:.4f} - {rss_u:.4f})/{df_num}] / [{rss_u:.4f}/{df_den}]\nF = {(rss_r - rss_u)/df_num:.4f} / {rss_u/df_den:.4f}\nF = {f_stat:.4f}",
        result=f_stat,
        result_name="F",
        notes=f"Compare to F({df_num}, {df_den}) distribution"
    )

    # Step 7: P-value
    deriv.add_step(
        title="Compute P-Value",
        equation="p = P(F > F_observed | H₀)",
        calculation=f"F-statistic: {f_stat:.4f}\nDegrees of freedom: ({df_num}, {df_den})\n\np-value = 1 - F_cdf({f_stat:.4f})\np-value = {p_val:.6f}",
        result=p_val,
        result_name="p",
        notes="If p < 0.05, reject H₀: X does Granger-cause Y"
    )

    # Step 8: Results across all lags
    lag_table = "\n".join([f"  Lag {l}: F={f:.3f}, p={p:.4f}" for l, f, p, _, _ in lag_results[:5]])

    deriv.add_step(
        title="Results Across Lags",
        equation="Select lag with highest F-statistic",
        calculation=f"Lag scan results:\n{lag_table}\n\nOptimal lag: {best_lag}\nBest F: {best_f:.4f}\nBest p: {best_p:.6f}",
        result=best_lag,
        result_name="optimal_lag",
        notes="Higher F = stronger evidence of Granger causality"
    )

    # Final result
    result = {
        'f_statistic': float(best_f),
        'p_value': float(best_p),
        'optimal_lag': int(best_lag),
        'significant_005': best_p < 0.05,
        'significant_001': best_p < 0.01,
    }

    deriv.final_result = best_f
    deriv.prism_output = best_f

    # Interpretation
    if best_p < 0.01:
        interp = f"**Strong evidence** that {signal_x} Granger-causes {signal_y} (p={best_p:.4f} < 0.01)."
        interp += f" Past values of {signal_x} significantly improve predictions of {signal_y}."
    elif best_p < 0.05:
        interp = f"**Moderate evidence** that {signal_x} Granger-causes {signal_y} (p={best_p:.4f} < 0.05)."
    elif best_p < 0.10:
        interp = f"**Weak evidence** of Granger causality (p={best_p:.4f}). Marginally significant at 10% level."
    else:
        interp = f"**No significant evidence** that {signal_x} Granger-causes {signal_y} (p={best_p:.4f})."
        interp += f" Past values of {signal_x} do not improve Y predictions beyond Y's own history."

    interp += f" Optimal lag = {best_lag} periods."

    deriv.interpretation = interp

    return result, deriv
