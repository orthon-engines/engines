"""
PRISM Transfer Entropy Engine

Measures directional information flow between signals.

Measures:
- Transfer entropy (bits) from X to Y
- Effective transfer entropy (bias-corrected)
- Asymmetry (net information flow direction)

Phase: Unbound
Normalization: Discretization required
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple
from datetime import date

import numpy as np
import pandas as pd
from collections import Counter

VERBOSE = os.getenv('PRISM_VERBOSE', '0') == '1'

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="transfer_entropy",
    engine_type="geometry",
    description="Directional information flow between series",
    domains={"causality", "information"},
    requires_window=True,
    deterministic=True,
)


class TransferEntropyEngine(BaseEngine):
    """
    Transfer Entropy engine for information flow analysis.
    
    Measures how much knowing the past of X reduces uncertainty about Y,
    beyond what knowing the past of Y provides.
    
    Outputs:
        - results.transfer_entropy: Pairwise TE values
    """
    
    name = "transfer_entropy"
    phase = "derived"
    default_normalization = None  # Discretize internally

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        n_bins: int = 8,
        lag: int = 1,
        n_shuffles: int = 100,
        **params
    ) -> Dict[str, Any]:
        """
        Run transfer entropy analysis.
        
        Args:
            df: Signal data
            run_id: Unique run identifier
            n_bins: Number of bins for discretization
            lag: Lag for transfer entropy (default 1)
            n_shuffles: Number of shuffles for significance testing
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = list(df_clean.columns)
        n = len(signals)
        
        if len(df_clean) < 500:
            logger.warning(
                f"Transfer entropy requires substantial data. "
                f"Got {len(df_clean)} samples, recommend 500+"
            )
        
        window_start, window_end = get_window_dates(df_clean)
        
        # Discretize
        df_discrete = self._discretize(df_clean, n_bins)
        
        # Compute pairwise TE
        results = []
        
        for i, ind1 in enumerate(signals):
            for j, ind2 in enumerate(signals):
                if i == j:
                    continue
                
                x = df_discrete[ind1].values
                y = df_discrete[ind2].values
                
                # TE from ind1 to ind2
                te, te_eff, p_value = self._transfer_entropy(
                    x, y, lag, n_shuffles
                )
                
                results.append({
                    "signal_from": ind1,
                    "signal_to": ind2,
                    "window_start": window_start,
                    "window_end": window_end,
                    "transfer_entropy": float(te),
                    "effective_te": float(te_eff),
                    "p_value": float(p_value),
                    "lag": lag,
                    "run_id": run_id,
                })
        
        # Store results
        if results:
            self._store_te_results(pd.DataFrame(results), run_id)
        
        # Find strongest information flows
        df_results = pd.DataFrame(results)
        significant = df_results[df_results["p_value"] < 0.05]
        
        # Net flow analysis
        net_flows = {}
        for ind in signals:
            outflow = df_results[df_results["signal_from"] == ind]["effective_te"].sum()
            inflow = df_results[df_results["signal_to"] == ind]["effective_te"].sum()
            net_flows[ind] = outflow - inflow
        
        # Most influential (net positive flow)
        top_influencer = max(net_flows, key=net_flows.get) if net_flows else None
        
        metrics = {
            "n_signals": n,
            "n_pairs": len(results),
            "avg_te": float(df_results["transfer_entropy"].mean()),
            "max_te": float(df_results["transfer_entropy"].max()),
            "significant_pairs": len(significant),
            "lag": lag,
            "n_bins": n_bins,
            "top_influencer": top_influencer,
        }
        
        logger.info(
            f"Transfer entropy complete: {metrics['significant_pairs']} "
            f"significant flows out of {metrics['n_pairs']}"
        )
        
        return metrics
    
    def _discretize(self, df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
        """Discretize using quantile bins."""
        result = pd.DataFrame(index=df.index)
        for col in df.columns:
            result[col] = pd.qcut(
                df[col], q=n_bins, labels=False, duplicates="drop"
            )
        return result
    
    def _transfer_entropy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int,
        n_shuffles: int,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """
        Compute transfer entropy from x to y.

        TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})

        Returns (TE, effective_TE, p_value)
        """
        # Build state sequences
        y_past = y[:-lag]
        x_past = x[:-lag]
        y_future = y[lag:]

        # Joint and marginal probabilities
        te = self._compute_te(x_past, y_past, y_future)

        # Shuffle test for significance (seeded for determinism)
        rng = np.random.default_rng(seed)
        shuffled_tes = []
        for _ in range(n_shuffles):
            x_shuffled = rng.permutation(x_past)
            te_shuffled = self._compute_te(x_shuffled, y_past, y_future)
            shuffled_tes.append(te_shuffled)

        # Effective TE (bias-corrected)
        bias = np.mean(shuffled_tes)
        te_effective = te - bias

        # P-value
        p_value = np.mean([1 if t >= te else 0 for t in shuffled_tes])

        return te, max(0, te_effective), p_value
    
    def _compute_te(
        self,
        x_past: np.ndarray,
        y_past: np.ndarray,
        y_future: np.ndarray
    ) -> float:
        """Compute transfer entropy using empirical probabilities."""
        n = len(y_future)
        
        # Count joint occurrences
        # P(y_t, y_{t-1}, x_{t-1})
        joint_xyz = Counter(zip(y_future, y_past, x_past))
        # P(y_{t-1}, x_{t-1})
        joint_xz = Counter(zip(y_past, x_past))
        # P(y_t, y_{t-1})
        joint_yz = Counter(zip(y_future, y_past))
        # P(y_{t-1})
        marginal_z = Counter(y_past)
        
        te = 0.0
        for (yt, yt_1, xt_1), count in joint_xyz.items():
            p_xyz = count / n
            p_xz = joint_xz[(yt_1, xt_1)] / n
            p_yz = joint_yz[(yt, yt_1)] / n
            p_z = marginal_z[yt_1] / n
            
            if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
                te += p_xyz * np.log2((p_xyz * p_z) / (p_xz * p_yz))
        
        return max(0, te)
    
    def _store_te_results(self, df: pd.DataFrame, run_id: str):
        """Store TE results."""
        # Would need table: results.transfer_entropy
        pass


# =============================================================================
# Standalone function with derivation
# =============================================================================

def compute_transfer_entropy_with_derivation(
    x: np.ndarray,
    y: np.ndarray,
    signal_x: str = "X",
    signal_y: str = "Y",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    n_bins: int = 8,
    lag: int = 1,
    n_shuffles: int = 100,
) -> tuple:
    """
    Compute transfer entropy with full mathematical derivation.

    Args:
        x: Source series (information flows FROM x)
        y: Target series (information flows TO y)
        signal_x: Name of X signal
        signal_y: Name of Y signal
        window_id: Window identifier
        window_start, window_end: Date range
        n_bins: Number of bins for discretization
        lag: Time lag
        n_shuffles: Number of shuffles for significance

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from prism.entry_points.derivations.base import Derivation

    n = len(x)

    deriv = Derivation(
        engine_name="transfer_entropy",
        method_name="Transfer Entropy (Information Flow)",
        signal_id=f"{signal_x}_to_{signal_y}",
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=n,
        parameters={'n_bins': n_bins, 'lag': lag, 'n_shuffles': n_shuffles}
    )

    # Step 1: Problem statement
    deriv.add_step(
        title="Transfer Entropy Definition",
        equation="TE(X→Y) = H(Yₜ | Yₜ₋₁) - H(Yₜ | Yₜ₋₁, Xₜ₋₁)",
        calculation=f"Measures information flow from {signal_x} to {signal_y}\n"
                    f"n = {n} observations\n"
                    f"lag = {lag}\n\n"
                    f"TE quantifies: How much does knowing X's past reduce\n"
                    f"uncertainty about Y's future, beyond Y's own past?",
        result=n,
        result_name="n",
        notes="TE is asymmetric: TE(X→Y) ≠ TE(Y→X)"
    )

    # Step 2: Discretization
    def discretize(arr, n_bins):
        from scipy.stats import rankdata
        ranks = rankdata(arr, method='ordinal')
        bins = np.floor((ranks - 1) / len(arr) * n_bins).astype(int)
        return np.clip(bins, 0, n_bins - 1)

    x_discrete = discretize(x, n_bins)
    y_discrete = discretize(y, n_bins)

    deriv.add_step(
        title="Discretization (Quantile Binning)",
        equation="xᵈ = floor(rank(x) / n × B) where B = number of bins",
        calculation=f"B = {n_bins} bins (equiprobable)\n\n"
                    f"Original {signal_x}: range [{np.min(x):.4f}, {np.max(x):.4f}]\n"
                    f"Discretized: {n_bins} states {{0, 1, ..., {n_bins-1}}}\n\n"
                    f"Original {signal_y}: range [{np.min(y):.4f}, {np.max(y):.4f}]\n"
                    f"Discretized: {n_bins} states {{0, 1, ..., {n_bins-1}}}",
        result=n_bins,
        result_name="B",
        notes="Discretization enables entropy computation via counting"
    )

    # Step 3: Build state sequences
    y_past = y_discrete[:-lag]
    x_past = x_discrete[:-lag]
    y_future = y_discrete[lag:]

    deriv.add_step(
        title="Construct State Sequences",
        equation="Yₜ₋₁ (past Y), Xₜ₋₁ (past X), Yₜ (future Y)",
        calculation=f"With lag = {lag}:\n"
                    f"  Yₜ₋₁ : Y[0:{n-lag}] → {n-lag} values\n"
                    f"  Xₜ₋₁ : X[0:{n-lag}] → {n-lag} values\n"
                    f"  Yₜ   : Y[{lag}:{n}] → {n-lag} values\n\n"
                    f"Joint state space: {n_bins}³ = {n_bins**3} possible triplets",
        result=n-lag,
        result_name="n_eff",
        notes="Each time t maps to triplet (Yₜ, Yₜ₋₁, Xₜ₋₁)"
    )

    # Step 4: Compute joint probabilities
    joint_xyz = Counter(zip(y_future, y_past, x_past))
    joint_xz = Counter(zip(y_past, x_past))
    joint_yz = Counter(zip(y_future, y_past))
    marginal_z = Counter(y_past)

    n_eff = len(y_future)

    deriv.add_step(
        title="Empirical Probability Distributions",
        equation="P̂(a,b,c) = count(a,b,c) / n",
        calculation=f"Joint distributions:\n"
                    f"  P(Yₜ, Yₜ₋₁, Xₜ₋₁): {len(joint_xyz)} unique states\n"
                    f"  P(Yₜ₋₁, Xₜ₋₁): {len(joint_xz)} unique states\n"
                    f"  P(Yₜ, Yₜ₋₁): {len(joint_yz)} unique states\n"
                    f"  P(Yₜ₋₁): {len(marginal_z)} unique states\n\n"
                    f"Maximum possible states:\n"
                    f"  P(Yₜ, Yₜ₋₁, Xₜ₋₁): {n_bins**3}\n"
                    f"  P(Yₜ₋₁, Xₜ₋₁): {n_bins**2}",
        result=len(joint_xyz),
        result_name="states",
        notes="Sparse representation: only observed states counted"
    )

    # Step 5: Transfer entropy computation
    te = 0.0
    for (yt, yt_1, xt_1), count in joint_xyz.items():
        p_xyz = count / n_eff
        p_xz = joint_xz[(yt_1, xt_1)] / n_eff
        p_yz = joint_yz[(yt, yt_1)] / n_eff
        p_z = marginal_z[yt_1] / n_eff

        if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
            te += p_xyz * np.log2((p_xyz * p_z) / (p_xz * p_yz))

    te = max(0, te)

    deriv.add_step(
        title="Transfer Entropy Calculation",
        equation="TE = Σ P(Yₜ,Yₜ₋₁,Xₜ₋₁) · log₂[P(Yₜ,Yₜ₋₁,Xₜ₋₁)·P(Yₜ₋₁) / P(Yₜ₋₁,Xₜ₋₁)·P(Yₜ,Yₜ₋₁)]",
        calculation=f"Summing over all observed triplets:\n\n"
                    f"  TE({signal_x} → {signal_y}) = {te:.6f} bits\n\n"
                    f"Interpretation:\n"
                    f"  TE = 0: No information flow\n"
                    f"  TE > 0: {signal_x}'s past provides information about {signal_y}'s future\n"
                    f"  TE ≈ log₂(B) = {np.log2(n_bins):.2f} bits: Maximum possible",
        result=te,
        result_name="TE",
        notes="TE in bits: information gain from knowing X's past"
    )

    # Step 6: Shuffle test for significance
    rng = np.random.default_rng(42)
    shuffled_tes = []

    for _ in range(n_shuffles):
        x_shuffled = rng.permutation(x_past)
        te_s = 0.0
        joint_xyz_s = Counter(zip(y_future, y_past, x_shuffled))
        joint_xz_s = Counter(zip(y_past, x_shuffled))

        for (yt, yt_1, xt_1), count in joint_xyz_s.items():
            p_xyz = count / n_eff
            p_xz = joint_xz_s[(yt_1, xt_1)] / n_eff
            p_yz = joint_yz[(yt, yt_1)] / n_eff
            p_z = marginal_z[yt_1] / n_eff

            if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
                te_s += p_xyz * np.log2((p_xyz * p_z) / (p_xz * p_yz))

        shuffled_tes.append(max(0, te_s))

    bias = np.mean(shuffled_tes)
    te_effective = max(0, te - bias)
    p_value = np.mean([1 if t >= te else 0 for t in shuffled_tes])

    deriv.add_step(
        title="Significance Test (Shuffle Test)",
        equation="H₀: TE = 0 (no information flow)",
        calculation=f"Shuffle test ({n_shuffles} permutations):\n"
                    f"  Observed TE: {te:.6f} bits\n"
                    f"  Null mean (bias): {bias:.6f} bits\n"
                    f"  Null std: {np.std(shuffled_tes):.6f} bits\n\n"
                    f"Effective TE (bias-corrected):\n"
                    f"  TE_eff = {te:.4f} - {bias:.4f} = {te_effective:.6f} bits\n\n"
                    f"P-value: {p_value:.4f}\n"
                    f"{'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at α = 0.05",
        result=p_value,
        result_name="p",
        notes="Shuffling destroys temporal structure, creating null distribution"
    )

    # Step 7: Interpretation
    is_significant = p_value < 0.05

    deriv.add_step(
        title="Information Flow Direction",
        equation="Net flow = TE(X→Y) - TE(Y→X)",
        calculation=f"TE({signal_x} → {signal_y}) = {te:.4f} bits\n"
                    f"Effective TE = {te_effective:.4f} bits\n"
                    f"Significant: {'Yes' if is_significant else 'No'}\n\n"
                    f"To determine dominant flow direction,\n"
                    f"compare with TE({signal_y} → {signal_x})",
        result=te_effective,
        result_name="TE_eff",
        notes="Positive effective TE indicates genuine information flow"
    )

    # Final result
    result = {
        'transfer_entropy': float(te),
        'effective_te': float(te_effective),
        'bias': float(bias),
        'p_value': float(p_value),
        'is_significant': is_significant,
        'n_bins': n_bins,
        'lag': lag,
        # Confidence metrics
        'te_null_std': float(np.std(shuffled_tes)),
        'te_z_score': float((te - bias) / (np.std(shuffled_tes) + 1e-10)),
    }

    if VERBOSE:
        result['null_distribution'] = [float(t) for t in shuffled_tes]

    deriv.final_result = te_effective
    deriv.prism_output = te_effective

    # Interpretation
    if is_significant and te_effective > 0.1:
        interp = f"**Strong information flow** from {signal_x} to {signal_y} (TE_eff={te_effective:.3f} bits)."
    elif is_significant:
        interp = f"**Weak but significant** information flow from {signal_x} to {signal_y} (TE_eff={te_effective:.3f} bits)."
    else:
        interp = f"**No significant** information flow from {signal_x} to {signal_y} (p={p_value:.3f})."

    deriv.interpretation = interp

    return result, deriv
