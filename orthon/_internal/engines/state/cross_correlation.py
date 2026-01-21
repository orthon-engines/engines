"""
PRISM Cross-Correlation Engine

Measures lead/lag relationships between signals.

Measures:
- Pairwise correlation at various lags
- Optimal lag per pair
- Lead/lag network structure

Phase: Unbound
Normalization: Z-score preferred
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import date

import numpy as np
import pandas as pd
from scipy import signal

from orthon._internal.engines.engine_base import BaseEngine, get_window_dates
from orthon._internal.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="cross_correlation",
    engine_type="geometry",
    description="Pairwise correlation and lead/lag relationships",
    domains={"correlation", "dependence"},
    requires_window=True,
    deterministic=True,
)


class CrossCorrelationEngine(BaseEngine):
    """
    Cross-correlation engine for lead/lag analysis.
    
    Outputs:
        - results.correlation_matrix: Pairwise correlations at lag 0
        - results.cross_correlation: Full lag structure per pair
    """
    
    name = "cross_correlation"
    phase = "derived"
    default_normalization = "zscore"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        max_lag: int = 20,
        store_full_lags: bool = False,
        **params
    ) -> Dict[str, Any]:
        """
        Run cross-correlation analysis.
        
        Args:
            df: Normalized signal data
            run_id: Unique run identifier
            max_lag: Maximum lag to compute (default 20)
            store_full_lags: Store full lag structure (default False, just optimal)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = list(df_clean.columns)
        n = len(signals)
        
        window_start, window_end = get_window_dates(df_clean)
        
        # Compute lag-0 correlation matrix
        corr_matrix = df_clean.corr()
        self._store_correlation_matrix(corr_matrix, window_start, window_end, run_id)
        
        # Compute cross-correlations with lags
        lag_results = []
        
        for i, ind1 in enumerate(signals):
            for j, ind2 in enumerate(signals):
                if i >= j:  # Skip self and duplicates
                    continue
                
                x = df_clean[ind1].values
                y = df_clean[ind2].values
                
                # Compute cross-correlation
                xcorr = self._cross_correlate(x, y, max_lag)
                
                # Find optimal lag
                optimal_lag = xcorr["lag"].iloc[xcorr["correlation"].abs().argmax()]
                optimal_corr = xcorr.loc[xcorr["lag"] == optimal_lag, "correlation"].values[0]
                
                lag_results.append({
                    "signal_id_1": ind1,
                    "signal_id_2": ind2,
                    "window_start": window_start,
                    "window_end": window_end,
                    "optimal_lag": int(optimal_lag),
                    "optimal_correlation": float(optimal_corr),
                    "lag_0_correlation": float(corr_matrix.loc[ind1, ind2]),
                    "run_id": run_id,
                })
        
        # Store results
        lag_df = pd.DataFrame(lag_results)
        self._store_lag_results(lag_df, run_id)
        
        # Summary metrics
        if len(lag_results) > 0:
            avg_abs_corr = corr_matrix.abs().values[np.triu_indices(n, k=1)].mean()
            leading_pairs = sum(1 for r in lag_results if r["optimal_lag"] < 0)
            lagging_pairs = sum(1 for r in lag_results if r["optimal_lag"] > 0)
        else:
            avg_abs_corr = 0
            leading_pairs = 0
            lagging_pairs = 0
        
        metrics = {
            "n_signals": n,
            "n_pairs": len(lag_results),
            "avg_abs_correlation": float(avg_abs_corr),
            "max_correlation": float(corr_matrix.abs().values[np.triu_indices(n, k=1)].max()) if n > 1 else 0,
            "leading_pairs": leading_pairs,
            "lagging_pairs": lagging_pairs,
            "synchronous_pairs": len(lag_results) - leading_pairs - lagging_pairs,
        }
        
        logger.info(
            f"Cross-correlation complete: {metrics['n_pairs']} pairs, "
            f"avg |r|={metrics['avg_abs_correlation']:.3f}"
        )
        
        return metrics
    
    def _cross_correlate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int
    ) -> pd.DataFrame:
        """
        Compute normalized cross-correlation at multiple lags.
        
        Positive lag means x leads y.
        """
        n = len(x)
        lags = range(-max_lag, max_lag + 1)
        correlations = []
        
        for lag in lags:
            if lag < 0:
                # x lags y (y leads)
                x_slice = x[-lag:]
                y_slice = y[:lag]
            elif lag > 0:
                # x leads y
                x_slice = x[:-lag]
                y_slice = y[lag:]
            else:
                x_slice = x
                y_slice = y
            
            if len(x_slice) > 0:
                corr = np.corrcoef(x_slice, y_slice)[0, 1]
            else:
                corr = np.nan
            
            correlations.append({"lag": lag, "correlation": corr})
        
        return pd.DataFrame(correlations)
    
    def _store_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store correlation matrix to results.correlation_matrix."""
        records = []
        signals = list(corr_matrix.columns)
        
        for i, ind1 in enumerate(signals):
            for j, ind2 in enumerate(signals):
                if i > j:  # Store lower triangle only
                    continue
                records.append({
                    "signal_id_1": ind1,
                    "signal_id_2": ind2,
                    "window_start": window_start,
                    "window_end": window_end,
                    "correlation": float(corr_matrix.loc[ind1, ind2]),
                    "run_id": run_id,
                })
        
        df = pd.DataFrame(records)
        ##self.store_results("correlation_matrix", df, run_id)
    
    def _store_lag_results(self, df: pd.DataFrame, run_id: str):
        """Store lag analysis results."""
        # Store to a dedicated cross-correlation table
        # For now, we'll skip since schema doesn't have this table
        # Could add: results.cross_correlation_lags
        pass


# =============================================================================
# Standalone function with derivation
# =============================================================================

def compute_cross_correlation_with_derivation(
    x: np.ndarray,
    y: np.ndarray,
    signal_x: str = "X",
    signal_y: str = "Y",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    max_lag: int = 20,
) -> tuple:
    """
    Compute cross-correlation with full mathematical derivation.

    Args:
        x: First signal topology
        y: Second signal topology
        signal_x: Name of X signal
        signal_y: Name of Y signal
        window_id: Window identifier
        window_start, window_end: Date range
        max_lag: Maximum lag to compute

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from orthon._internal.entry_points.derivations.base import Derivation

    n = len(x)

    deriv = Derivation(
        engine_name="cross_correlation",
        method_name="Cross-Correlation Analysis",
        signal_id=f"{signal_x}_vs_{signal_y}",
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=n,
        parameters={'max_lag': max_lag}
    )

    # Step 1: Input data
    deriv.add_step(
        title="Input Signal Topology",
        equation="x = {x₁, x₂, ..., xₙ}, y = {y₁, y₂, ..., yₙ}",
        calculation=f"Series {signal_x}:\n"
                    f"  n = {n} observations\n"
                    f"  mean = {np.mean(x):.6f}\n"
                    f"  std = {np.std(x):.6f}\n\n"
                    f"Series {signal_y}:\n"
                    f"  n = {n} observations\n"
                    f"  mean = {np.mean(y):.6f}\n"
                    f"  std = {np.std(y):.6f}",
        result=n,
        result_name="n",
        notes="Cross-correlation measures linear relationship at various time shifts"
    )

    # Step 2: Standardize
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)
    x_norm = (x - x_mean) / x_std if x_std > 0 else x - x_mean
    y_norm = (y - y_mean) / y_std if y_std > 0 else y - y_mean

    deriv.add_step(
        title="Standardize Series (Z-score)",
        equation="z_x = (x - μ_x) / σ_x,  z_y = (y - μ_y) / σ_y",
        calculation=f"Standardization:\n"
                    f"  z_{signal_x} = ({signal_x} - {x_mean:.4f}) / {x_std:.4f}\n"
                    f"  z_{signal_y} = ({signal_y} - {y_mean:.4f}) / {y_std:.4f}\n\n"
                    f"After standardization:\n"
                    f"  mean(z_x) = {np.mean(x_norm):.6f} ≈ 0\n"
                    f"  std(z_x) = {np.std(x_norm):.6f} ≈ 1",
        result=0.0,
        result_name="μ_z",
        notes="Standardization ensures correlation is in [-1, 1]"
    )

    # Step 3: Lag-0 correlation
    corr_0 = np.corrcoef(x, y)[0, 1]

    deriv.add_step(
        title="Lag-0 (Synchronous) Correlation",
        equation="r(0) = (1/n) Σᵢ z_x[i] · z_y[i]",
        calculation=f"Correlation at lag 0:\n"
                    f"  r(0) = Σ(z_x · z_y) / n\n"
                    f"  r(0) = {corr_0:.6f}\n\n"
                    f"Interpretation:\n"
                    f"  |r| > 0.7: Strong relationship\n"
                    f"  |r| > 0.3: Moderate relationship\n"
                    f"  |r| < 0.3: Weak relationship\n\n"
                    f"  r(0) = {corr_0:.4f} → {'Strong' if abs(corr_0) > 0.7 else 'Moderate' if abs(corr_0) > 0.3 else 'Weak'}",
        result=corr_0,
        result_name="r(0)",
        notes="Lag-0 measures contemporaneous relationship"
    )

    # Step 4: Compute cross-correlation at all lags
    lags = list(range(-max_lag, max_lag + 1))
    correlations = []

    for lag in lags:
        if lag < 0:
            # x lags y (y leads)
            x_slice = x[-lag:]
            y_slice = y[:lag]
        elif lag > 0:
            # x leads y
            x_slice = x[:-lag]
            y_slice = y[lag:]
        else:
            x_slice = x
            y_slice = y

        if len(x_slice) > 1:
            corr = np.corrcoef(x_slice, y_slice)[0, 1]
        else:
            corr = np.nan
        correlations.append(corr)

    correlations = np.array(correlations)

    # Find optimal lag
    valid_mask = ~np.isnan(correlations)
    if valid_mask.any():
        abs_corrs = np.abs(correlations)
        abs_corrs[~valid_mask] = -np.inf
        optimal_idx = np.argmax(abs_corrs)
        optimal_lag = lags[optimal_idx]
        optimal_corr = correlations[optimal_idx]
    else:
        optimal_lag = 0
        optimal_corr = corr_0

    deriv.add_step(
        title="Cross-Correlation Function",
        equation="r(k) = (1/(n-|k|)) Σᵢ z_x[i] · z_y[i+k]",
        calculation=f"Computed r(k) for k ∈ [{-max_lag}, {max_lag}]\n\n"
                    f"Sample correlations:\n"
                    f"  r({-max_lag}) = {correlations[0]:.4f}\n"
                    f"  r(-5) = {correlations[max_lag-5]:.4f}\n"
                    f"  r(0) = {correlations[max_lag]:.4f}\n"
                    f"  r(5) = {correlations[max_lag+5]:.4f}\n"
                    f"  r({max_lag}) = {correlations[-1]:.4f}",
        result=correlations.tolist(),
        result_name="r(k)",
        notes="Positive lag means X leads Y"
    )

    # Step 5: Optimal lag analysis
    deriv.add_step(
        title="Optimal Lag Detection",
        equation="k* = argmax_k |r(k)|",
        calculation=f"Maximum absolute correlation:\n"
                    f"  k* = {optimal_lag}\n"
                    f"  r(k*) = {optimal_corr:.6f}\n\n"
                    f"Lead/Lag interpretation:\n"
                    f"  k* < 0: {signal_y} leads {signal_x} by {abs(optimal_lag)} periods\n"
                    f"  k* = 0: Synchronous relationship\n"
                    f"  k* > 0: {signal_x} leads {signal_y} by {optimal_lag} periods\n\n"
                    f"Result: {'Synchronous' if optimal_lag == 0 else f'{signal_x} leads by {optimal_lag}' if optimal_lag > 0 else f'{signal_y} leads by {abs(optimal_lag)}'}",
        result=optimal_lag,
        result_name="k*",
        notes="Optimal lag reveals lead/lag structure"
    )

    # Step 6: Significance test
    # Under null hypothesis, r ~ N(0, 1/sqrt(n))
    se = 1 / np.sqrt(n)
    z_stat = optimal_corr / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    is_significant = p_value < 0.05

    deriv.add_step(
        title="Significance Test",
        equation="z = r(k*) / (1/√n),  H₀: ρ(k*) = 0",
        calculation=f"Test statistic:\n"
                    f"  SE = 1/√{n} = {se:.6f}\n"
                    f"  z = {optimal_corr:.4f} / {se:.4f} = {z_stat:.4f}\n"
                    f"  p-value = {p_value:.6f}\n\n"
                    f"Decision (α = 0.05):\n"
                    f"  {'SIGNIFICANT' if is_significant else 'NOT SIGNIFICANT'}\n"
                    f"  The relationship {'is' if is_significant else 'is NOT'} statistically significant",
        result=p_value,
        result_name="p",
        notes="Tests whether optimal correlation differs from zero"
    )

    # Final result
    result = {
        'lag_0_correlation': float(corr_0),
        'optimal_lag': int(optimal_lag),
        'optimal_correlation': float(optimal_corr),
        'p_value': float(p_value),
        'is_significant': is_significant,
        'max_lag': max_lag,
    }

    deriv.final_result = optimal_corr
    deriv.prism_output = optimal_corr

    # Interpretation
    if optimal_lag == 0:
        interp = f"**Synchronous**: {signal_x} and {signal_y} move together (r={corr_0:.3f})."
    elif optimal_lag > 0:
        interp = f"**{signal_x} leads** {signal_y} by {optimal_lag} periods (r={optimal_corr:.3f})."
    else:
        interp = f"**{signal_y} leads** {signal_x} by {abs(optimal_lag)} periods (r={optimal_corr:.3f})."

    if is_significant:
        interp += " Relationship is statistically significant."
    else:
        interp += " Relationship is NOT statistically significant."

    deriv.interpretation = interp

    return result, deriv
