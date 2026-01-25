"""
PRISM GARCH Engine

Conditional volatility modeling.

Measures:
- GARCH parameters (persistence)
- Conditional variance series
- Volatility clustering

Phase: Unbound
Normalization: Returns
"""

import logging
from typing import Dict, Any, Optional
from datetime import date
import warnings

import numpy as np
import pandas as pd

from prism.engines.engine_base import BaseEngine
from prism.engines.metadata import EngineMetadata
from prism.entry_points.derivations.base import Derivation


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="garch",
    engine_type="vector",
    description="GARCH volatility modeling and persistence",
    domains={"signal_topology", "volatility"},
    requires_window=True,
    deterministic=False,  # Optimization-based
)


# Try to import arch library
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    logger.warning("arch library not installed. GARCH will use simplified estimation.")


# =============================================================================
# Vector Engine Contract: Simple function interface
# =============================================================================

def compute_garch(values: np.ndarray) -> dict:
    """
    Compute GARCH volatility metrics for a single signal.

    Uses REAL arch package MLE when available, falls back to moment matching.

    Args:
        values: Array of observed values (levels, not returns)

    Returns:
        Dict of GARCH metrics
    """
    # Import the real GARCH implementation
    from prism.engines.volatility.garch import compute as compute_garch_real

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < 30:
        return {
            "omega": None,
            "alpha": None,
            "beta": None,
            "persistence": None,
            "unconditional_vol": None,
            "method": None,
        }

    # Use the real implementation
    result = compute_garch_real(values, mode='static')

    return {
        "omega": result.get('omega'),
        "alpha": result.get('alpha'),
        "beta": result.get('beta'),
        "persistence": result.get('persistence'),
        "unconditional_vol": result.get('unconditional_volatility'),
        "is_stationary": result.get('is_stationary'),
        "method": result.get('method'),
    }


def compute_garch_with_derivation(
    values: np.ndarray,
    signal_id: str = "unknown",
    window_id: str = "unknown",
    window_start: str = "",
    window_end: str = "",
) -> tuple:
    """
    Compute GARCH(1,1) parameters with full mathematical derivation.

    Uses moment matching method for educational clarity.

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from datetime import datetime

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    # Initialize derivation
    derivation = Derivation(
        engine_name="garch",
        method_name="GARCH(1,1) via Moment Matching",
        signal_id=signal_id,
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=n,
        generated_at=datetime.now(),
    )

    derivation.purpose = (
        "GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models "
        "time-varying volatility. It captures volatility clustering: large price changes "
        "tend to be followed by large changes, and small by small."
    )

    derivation.definition = """
The GARCH(1,1) model for conditional variance:

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

Where:
- σ²_t = conditional variance at time t
- ω = baseline variance (intercept, ω > 0)
- α = ARCH coefficient (shock impact, α ≥ 0)
- β = GARCH coefficient (persistence, β ≥ 0)
- ε_t = return innovation (r_t - μ)

### Key Properties

| Parameter | Interpretation |
|-----------|----------------|
| α + β | Persistence (should be < 1 for stationarity) |
| β/(α+β) | Proportion of persistence from past variance |
| α/(α+β) | Proportion of persistence from past shocks |
| ω/(1-α-β) | Unconditional (long-run) variance |
"""

    # Store raw data sample
    derivation.raw_data_sample = values[:10].tolist()

    # Step 1: Input data summary
    derivation.add_step(
        title="Input Price/Level Data",
        equation="P = {P₁, P₂, ..., Pₙ}",
        calculation=f"n = {n}\nRange: [{values.min():.4f}, {values.max():.4f}]\nMean: {values.mean():.4f}",
        result=n,
        result_name="n",
        notes="Input signal topology (prices or levels)"
    )

    # Step 2: Convert to log returns
    log_values = np.log(np.maximum(values, 1e-10))
    returns = np.diff(log_values)
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    n_returns = len(returns)

    derivation.add_step(
        title="Compute Log Returns",
        equation="r_t = log(P_t) - log(P_{t-1}) = log(P_t/P_{t-1})",
        calculation=f"r₁ = log({values[1]:.4f}) - log({values[0]:.4f}) = {returns[0]:.6f}\nr₂ = log({values[2]:.4f}) - log({values[1]:.4f}) = {returns[1]:.6f}\n...\nn_returns = {n_returns}\nMean return: {returns.mean():.6f}\nReturn std: {returns.std():.6f}",
        result=n_returns,
        result_name="n_returns",
        notes="Log returns are approximately normally distributed"
    )

    # Step 3: Compute sample variance
    sample_var = np.var(returns)
    sample_std = np.sqrt(sample_var)

    derivation.add_step(
        title="Compute Sample Variance (Unconditional)",
        equation="σ² = (1/n) Σ(r_t - r̄)²",
        calculation=f"r̄ = {returns.mean():.6f}\nσ² = (1/{n_returns}) Σ(r_t - {returns.mean():.6f})²\n   = {sample_var:.8f}\nσ  = {sample_std:.6f}",
        result=sample_var,
        result_name="σ²",
        notes="This estimates the unconditional (long-run) variance"
    )

    # Step 4: Compute squared returns
    r2 = returns ** 2

    derivation.add_step(
        title="Compute Squared Returns (Volatility Proxy)",
        equation="ε²_t = r²_t",
        calculation=f"ε²₁ = {returns[0]:.6f}² = {r2[0]:.8f}\nε²₂ = {returns[1]:.6f}² = {r2[1]:.8f}\n...\nMean(ε²) = {r2.mean():.8f}\nMax(ε²) = {r2.max():.8f}",
        result=r2.mean(),
        result_name="mean_ε²",
        notes="Squared returns proxy for instantaneous variance"
    )

    # Step 5: Compute autocorrelation of squared returns
    if len(r2) > 1:
        acf1 = np.corrcoef(r2[:-1], r2[1:])[0, 1]
        if np.isnan(acf1):
            acf1 = 0.0
    else:
        acf1 = 0.0

    derivation.add_step(
        title="Compute Autocorrelation of Squared Returns",
        equation="ρ₁ = Corr(ε²_t, ε²_{t-1})",
        calculation=f"ρ₁ = Cov(ε²_t, ε²_{{t-1}}) / [Std(ε²_t) · Std(ε²_{{t-1}})]\n   = {acf1:.6f}",
        result=acf1,
        result_name="ρ₁",
        notes="Positive autocorrelation indicates volatility clustering"
    )

    # Step 6: Estimate GARCH parameters via moment matching
    # Standard assumption: persistence = 0.9 for typical signal topology
    persistence = 0.9
    alpha = min(max(0.05, acf1 * 0.5), 0.3)
    beta = persistence - alpha
    omega = sample_var * (1 - persistence)
    unconditional_vol = np.sqrt(sample_var)

    derivation.add_step(
        title="Estimate GARCH(1,1) Parameters (Moment Matching)",
        equation="Assume persistence α + β = 0.9 (typical)\nα ≈ ρ₁ · 0.5 (bounded to [0.05, 0.3])\nβ = (α + β) - α\nω = σ² · (1 - α - β)",
        calculation=f"Persistence target: 0.9\nα = min(max(0.05, {acf1:.4f} × 0.5), 0.3) = {alpha:.4f}\nβ = 0.9 - {alpha:.4f} = {beta:.4f}\nω = {sample_var:.8f} × (1 - 0.9) = {omega:.8f}\n\nVerification:\n  α + β = {alpha + beta:.4f} (persistence)\n  ω/(1-α-β) = {omega/(1-persistence):.8f} ≈ {sample_var:.8f} (unconditional var)",
        result=persistence,
        result_name="α+β",
        notes="Moment matching provides closed-form estimates"
    )

    # Step 7: Interpret results
    derivation.add_step(
        title="Interpret GARCH Parameters",
        equation="Half-life of volatility shock: t_{1/2} = log(0.5)/log(α+β)",
        calculation=f"Persistence: α + β = {persistence:.4f}\n  → Volatility shocks decay slowly\n\nHalf-life: t_{{1/2}} = log(0.5)/log({persistence:.4f})\n         = {np.log(0.5)/np.log(persistence):.1f} periods\n\nShock impact: α = {alpha:.4f}\n  → {alpha*100:.1f}% of new shock feeds into next variance\n\nMemory: β = {beta:.4f}\n  → {beta*100:.1f}% of past variance carries forward\n\nUnconditional volatility: √(ω/(1-α-β)) = {unconditional_vol:.6f}",
        result=persistence,
        result_name="persistence",
        notes="High persistence (>0.9) indicates slowly decaying volatility shocks"
    )

    # Set final result
    derivation.final_result = persistence
    derivation.prism_output = persistence

    result = {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(persistence),
        "unconditional_vol": float(unconditional_vol),
    }

    return result, derivation


class GARCHEngine(BaseEngine):
    """
    GARCH engine for volatility analysis.

    Models time-varying volatility and captures volatility clustering.

    Outputs:
        - results.geometry_fingerprints: GARCH parameters
    """

    name = "garch"
    phase = "derived"
    default_normalization = "returns"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        p: int = 1,
        q: int = 1,
        **params
    ) -> Dict[str, Any]:
        """
        Run GARCH analysis.
        
        Args:
            df: Returns data
            run_id: Unique run identifier
            p: GARCH lag order (default 1)
            q: ARCH lag order (default 1)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = list(df_clean.columns)
        
        window_start = df_clean.index.min().date()
        window_end = df_clean.index.max().date()
        
        results = []
        
        for signal in signals:
            returns = df_clean[signal].values * 100  # Scale for numerical stability
            
            try:
                if HAS_ARCH:
                    garch_result = self._fit_arch_garch(returns, p, q)
                else:
                    garch_result = self._fit_simple_garch(returns, p, q)
                
                results.append({
                    "signal_id": signal,
                    **garch_result,
                })
                
            except Exception as e:
                logger.warning(f"GARCH failed for {signal}: {e}")
                results.append({
                    "signal_id": signal,
                    "omega": np.nan,
                    "alpha": np.nan,
                    "beta": np.nan,
                    "persistence": np.nan,
                    "unconditional_vol": np.nan,
                })
        
        # Store results
        self._store_garch(results, window_start, window_end, run_id)
        
        # Summary metrics
        df_results = pd.DataFrame(results)
        valid = df_results.dropna(subset=["persistence"])
        
        metrics = {
            "n_signals": len(signals),
            "n_successful": len(valid),
            "p": p,
            "q": q,
            "avg_persistence": float(valid["persistence"].mean()) if len(valid) > 0 else None,
            "high_persistence_count": int((valid["persistence"] > 0.9).sum()) if len(valid) > 0 else 0,
            "has_arch_library": HAS_ARCH,
        }
        
        logger.info(
            f"GARCH complete: {metrics['n_successful']}/{len(signals)} successful, "
            f"avg persistence={metrics['avg_persistence']:.3f}" if metrics['avg_persistence'] else ""
        )
        
        return metrics
    
    def _fit_arch_garch(
        self,
        returns: np.ndarray,
        p: int,
        q: int
    ) -> Dict[str, float]:
        """Fit GARCH using arch library."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            model = arch_model(returns, vol='Garch', p=p, q=q, rescale=False)
            result = model.fit(disp='off')
            
            params = result.params
            omega = params.get('omega', 0)
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            
            persistence = alpha + beta
            unconditional_vol = np.sqrt(omega / (1 - persistence)) if persistence < 1 else np.nan
            
            return {
                "omega": float(omega),
                "alpha": float(alpha),
                "beta": float(beta),
                "persistence": float(persistence),
                "unconditional_vol": float(unconditional_vol) if not np.isnan(unconditional_vol) else None,
                "log_likelihood": float(result.loglikelihood),
            }
    
    def _fit_simple_garch(
        self,
        returns: np.ndarray,
        p: int,
        q: int
    ) -> Dict[str, float]:
        """
        Simple GARCH(1,1) estimation via variance targeting.
        Fallback when arch library not available.
        """
        n = len(returns)
        
        # Sample variance
        sample_var = np.var(returns)
        
        # Squared returns
        r2 = returns ** 2
        
        # Simple moment estimation for GARCH(1,1)
        # Assumes alpha + beta = 0.9 (typical)
        persistence = 0.9
        
        # Autocorrelation of squared returns
        acf1 = np.corrcoef(r2[:-1], r2[1:])[0, 1]
        
        # alpha ≈ acf1 * (1 - beta)
        # With constraint alpha + beta = persistence
        alpha = min(max(0.05, acf1 * 0.5), 0.3)
        beta = persistence - alpha
        
        # omega from unconditional variance
        omega = sample_var * (1 - persistence)
        
        unconditional_vol = np.sqrt(sample_var)
        
        return {
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
            "persistence": float(persistence),
            "unconditional_vol": float(unconditional_vol),
            "log_likelihood": None,  # Not computed in simple version
        }
    
    def _store_garch(
        self,
        results: list,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store GARCH parameters as geometry fingerprints."""
        records = []
        
        for r in results:
            for dim in ["alpha", "beta", "persistence", "unconditional_vol"]:
                value = r.get(dim)
                if value is not None and not np.isnan(value):
                    records.append({
                        "signal_id": r["signal_id"],
                        "window_start": window_start,
                        "window_end": window_end,
                        "dimension": f"garch_{dim}",
                        "value": float(value),
                        "run_id": run_id,
                    })
        
        if records:
            df = pd.DataFrame(records)
            self.store_results("geometry_fingerprints", df, run_id)
