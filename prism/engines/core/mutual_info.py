"""
PRISM Mutual Information Engine

Measures non-linear dependence between signals.

Measures:
- Mutual information (bits)
- Normalized mutual information
- Non-linear dependency strength

Phase: Unbound
Normalization: Discretization required
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_regression

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="mutual_information",
    engine_type="geometry",
    description="Non-linear dependence via mutual information",
    domains={"dependence", "information"},
    requires_window=True,
    deterministic=True,
)


class MutualInformationEngine(BaseEngine):
    """
    Mutual Information engine for non-linear dependence.
    
    Captures dependencies that correlation misses.
    
    Outputs:
        - results.mutual_information: Pairwise MI values
    """
    
    name = "mutual_information"
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
        method: str = "binned",
        **params
    ) -> Dict[str, Any]:
        """
        Run mutual information analysis.
        
        Args:
            df: Signal data
            run_id: Unique run identifier
            n_bins: Number of bins for discretization
            method: 'binned' or 'knn' (k-nearest neighbors)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        signals = list(df_clean.columns)
        n = len(signals)
        
        window_start, window_end = get_window_dates(df_clean)
        
        # Discretize for binned MI
        if method == "binned":
            df_discrete = self._discretize(df_clean, n_bins)
        
        # Compute pairwise MI
        results = []
        mi_matrix = np.zeros((n, n))
        
        for i, ind1 in enumerate(signals):
            for j, ind2 in enumerate(signals):
                if i > j:
                    continue
                
                if method == "binned":
                    x = df_discrete[ind1].values
                    y = df_discrete[ind2].values
                    mi = mutual_info_score(x, y)
                    nmi = normalized_mutual_info_score(x, y)
                else:  # knn
                    x = df_clean[ind1].values.reshape(-1, 1)
                    y = df_clean[ind2].values
                    mi = mutual_info_regression(x, y, random_state=42)[0]
                    nmi = mi  # No normalized version for continuous
                
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
                
                if i != j:
                    results.append({
                        "signal_id_1": ind1,
                        "signal_id_2": ind2,
                        "window_start": window_start,
                        "window_end": window_end,
                        "mutual_information": float(mi),
                        "normalized_mi": float(nmi),
                        "run_id": run_id,
                    })
        
        # Store results
        if results:
            self._store_mi_results(pd.DataFrame(results), run_id)
        
        # Compare with correlation
        corr_matrix = df_clean.corr().values
        upper_idx = np.triu_indices(n, k=1)
        
        mi_values = mi_matrix[upper_idx]
        corr_values = np.abs(corr_matrix[upper_idx])
        
        # Non-linear excess: cases where MI is high but correlation is low
        if len(mi_values) > 0:
            nonlinear_excess = np.mean(mi_values[corr_values < 0.3])
        else:
            nonlinear_excess = 0
        
        metrics = {
            "n_signals": n,
            "n_pairs": len(results),
            "avg_mi": float(np.mean(mi_values)) if len(mi_values) > 0 else 0,
            "max_mi": float(np.max(mi_values)) if len(mi_values) > 0 else 0,
            "method": method,
            "n_bins": n_bins if method == "binned" else None,
            "nonlinear_excess": float(nonlinear_excess) if not np.isnan(nonlinear_excess) else 0,
        }
        
        logger.info(
            f"Mutual information complete: {metrics['n_pairs']} pairs, "
            f"avg MI={metrics['avg_mi']:.4f}"
        )
        
        return metrics
    
    def _discretize(self, df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
        """Discretize continuous data using quantile bins."""
        result = pd.DataFrame(index=df.index)
        for col in df.columns:
            # Skip constant columns (can't discretize)
            if df[col].std() == 0 or df[col].nunique() < 2:
                result[col] = 0  # Assign all to same bin
            else:
                try:
                    result[col] = pd.qcut(
                        df[col], q=n_bins, labels=False, duplicates="drop"
                    )
                except ValueError:
                    # Fall back to cut if qcut fails
                    result[col] = pd.cut(
                        df[col], bins=n_bins, labels=False
                    )
        # Fill any remaining NaN with 0
        result = result.fillna(0).astype(int)
        return result
    
    def _store_mi_results(self, df: pd.DataFrame, run_id: str):
        """Store MI results."""
        # Would need table: results.mutual_information
        # For now, skip or store as generic
        pass


# =============================================================================
# Standalone function with derivation
# =============================================================================

def compute_mutual_information_with_derivation(
    x: np.ndarray,
    y: np.ndarray,
    signal_x: str = "X",
    signal_y: str = "Y",
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    n_bins: int = 8,
) -> tuple:
    """
    Compute mutual information with full mathematical derivation.

    Args:
        x: First signal topology
        y: Second signal topology
        signal_x: Name of X signal
        signal_y: Name of Y signal
        window_id: Window identifier
        window_start, window_end: Date range
        n_bins: Number of bins for discretization

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from prism.entry_points.derivations.base import Derivation

    n = len(x)

    deriv = Derivation(
        engine_name="mutual_information",
        method_name="Mutual Information (Non-linear Dependence)",
        signal_id=f"{signal_x}_vs_{signal_y}",
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=n,
        parameters={'n_bins': n_bins}
    )

    # Step 1: Problem statement
    deriv.add_step(
        title="Mutual Information Definition",
        equation="I(X;Y) = H(X) + H(Y) - H(X,Y) = H(X) - H(X|Y)",
        calculation=f"Measures shared information between {signal_x} and {signal_y}\n"
                    f"n = {n} observations\n\n"
                    f"MI captures ALL dependencies (linear AND non-linear)\n"
                    f"Unlike correlation which only captures linear relationships",
        result=n,
        result_name="n",
        notes="MI ≥ 0; MI = 0 iff X and Y are independent"
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
        equation="xᵈ = floor(rank(x) / n × B)",
        calculation=f"B = {n_bins} bins (equiprobable quantiles)\n\n"
                    f"{signal_x}:\n"
                    f"  Original range: [{np.min(x):.4f}, {np.max(x):.4f}]\n"
                    f"  Discrete states: {{0, 1, ..., {n_bins-1}}}\n\n"
                    f"{signal_y}:\n"
                    f"  Original range: [{np.min(y):.4f}, {np.max(y):.4f}]\n"
                    f"  Discrete states: {{0, 1, ..., {n_bins-1}}}",
        result=n_bins,
        result_name="B",
        notes="Equiprobable binning maximizes entropy estimation accuracy"
    )

    # Step 3: Compute marginal entropies
    from collections import Counter

    px = Counter(x_discrete)
    py = Counter(y_discrete)

    def entropy(counts, n):
        h = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                h -= p * np.log2(p)
        return h

    h_x = entropy(px, n)
    h_y = entropy(py, n)

    deriv.add_step(
        title="Marginal Entropies",
        equation="H(X) = -Σᵢ P(xᵢ) log₂ P(xᵢ)",
        calculation=f"Entropy of {signal_x}:\n"
                    f"  H({signal_x}) = {h_x:.6f} bits\n"
                    f"  Max possible: log₂({n_bins}) = {np.log2(n_bins):.4f} bits\n"
                    f"  Efficiency: {h_x / np.log2(n_bins) * 100:.1f}%\n\n"
                    f"Entropy of {signal_y}:\n"
                    f"  H({signal_y}) = {h_y:.6f} bits\n"
                    f"  Max possible: log₂({n_bins}) = {np.log2(n_bins):.4f} bits\n"
                    f"  Efficiency: {h_y / np.log2(n_bins) * 100:.1f}%",
        result=h_x,
        result_name="H(X)",
        notes="Entropy measures uncertainty/information content"
    )

    # Step 4: Compute joint entropy
    pxy = Counter(zip(x_discrete, y_discrete))
    h_xy = entropy(pxy, n)

    deriv.add_step(
        title="Joint Entropy",
        equation="H(X,Y) = -Σᵢⱼ P(xᵢ,yⱼ) log₂ P(xᵢ,yⱼ)",
        calculation=f"Joint distribution:\n"
                    f"  {len(pxy)} unique (x,y) pairs observed\n"
                    f"  Max possible: {n_bins}² = {n_bins**2}\n\n"
                    f"Joint entropy:\n"
                    f"  H({signal_x},{signal_y}) = {h_xy:.6f} bits\n"
                    f"  Max possible: 2·log₂({n_bins}) = {2*np.log2(n_bins):.4f} bits\n\n"
                    f"If independent: H(X,Y) = H(X) + H(Y) = {h_x + h_y:.4f} bits",
        result=h_xy,
        result_name="H(X,Y)",
        notes="H(X,Y) ≤ H(X) + H(Y) with equality iff independent"
    )

    # Step 5: Mutual Information
    mi = h_x + h_y - h_xy

    deriv.add_step(
        title="Mutual Information Calculation",
        equation="I(X;Y) = H(X) + H(Y) - H(X,Y)",
        calculation=f"Mutual Information:\n"
                    f"  I({signal_x};{signal_y}) = {h_x:.4f} + {h_y:.4f} - {h_xy:.4f}\n"
                    f"  I({signal_x};{signal_y}) = {mi:.6f} bits\n\n"
                    f"Information shared between {signal_x} and {signal_y}:\n"
                    f"  {mi:.4f} bits out of min({h_x:.4f}, {h_y:.4f}) = {min(h_x, h_y):.4f} possible",
        result=mi,
        result_name="I",
        notes="MI quantifies how much knowing X reduces uncertainty about Y"
    )

    # Step 6: Normalized Mutual Information
    nmi = mi / max(np.sqrt(h_x * h_y), 1e-10)  # Geometric mean normalization
    nmi_max = 2 * mi / (h_x + h_y) if (h_x + h_y) > 0 else 0  # Arithmetic mean normalization

    deriv.add_step(
        title="Normalized Mutual Information",
        equation="NMI = I(X;Y) / √(H(X)·H(Y))",
        calculation=f"Normalization (geometric mean):\n"
                    f"  NMI = {mi:.4f} / √({h_x:.4f}×{h_y:.4f})\n"
                    f"  NMI = {mi:.4f} / {np.sqrt(h_x * h_y):.4f}\n"
                    f"  NMI = {nmi:.6f}\n\n"
                    f"Alternative (arithmetic mean):\n"
                    f"  NMI_arith = 2×{mi:.4f} / ({h_x:.4f}+{h_y:.4f})\n"
                    f"  NMI_arith = {nmi_max:.6f}",
        result=nmi,
        result_name="NMI",
        notes="NMI ∈ [0, 1]: 0 = independent, 1 = perfect dependence"
    )

    # Step 7: Compare with correlation
    corr = np.corrcoef(x, y)[0, 1]

    # Estimate MI from correlation (for Gaussian)
    mi_gaussian = -0.5 * np.log2(1 - corr**2) if abs(corr) < 1 else np.inf

    deriv.add_step(
        title="Comparison with Correlation",
        equation="For Gaussian: I_G = -½ log₂(1 - r²)",
        calculation=f"Correlation:\n"
                    f"  r = {corr:.6f}\n"
                    f"  r² = {corr**2:.6f}\n\n"
                    f"MI if Gaussian:\n"
                    f"  I_G = -½ log₂(1 - {corr**2:.4f})\n"
                    f"  I_G = {mi_gaussian:.6f} bits\n\n"
                    f"Actual MI: {mi:.6f} bits\n"
                    f"Non-linearity index: {mi - mi_gaussian:.6f} bits\n"
                    f"  {'Positive' if mi > mi_gaussian else 'Negative'}: MI {'exceeds' if mi > mi_gaussian else 'below'} Gaussian prediction",
        result=corr,
        result_name="r",
        notes="MI > I_G suggests non-linear dependencies"
    )

    # Final result
    result = {
        'mutual_information': float(mi),
        'normalized_mi': float(nmi),
        'entropy_x': float(h_x),
        'entropy_y': float(h_y),
        'joint_entropy': float(h_xy),
        'correlation': float(corr),
        'mi_gaussian': float(mi_gaussian) if not np.isinf(mi_gaussian) else None,
        'nonlinear_excess': float(mi - mi_gaussian) if not np.isinf(mi_gaussian) else None,
        'n_bins': n_bins,
    }

    deriv.final_result = mi
    deriv.prism_output = mi

    # Interpretation
    if nmi > 0.7:
        interp = f"**Strong dependence** between {signal_x} and {signal_y} (NMI={nmi:.3f})."
    elif nmi > 0.3:
        interp = f"**Moderate dependence** between {signal_x} and {signal_y} (NMI={nmi:.3f})."
    else:
        interp = f"**Weak dependence** between {signal_x} and {signal_y} (NMI={nmi:.3f})."

    if mi > mi_gaussian * 1.2 and not np.isinf(mi_gaussian):
        interp += f" **Non-linear** component detected (MI exceeds Gaussian by {(mi/mi_gaussian - 1)*100:.0f}%)."

    deriv.interpretation = interp

    return result, deriv
