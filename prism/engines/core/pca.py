"""
PRISM PCA Engine

Principal Component Analysis for structure analysis.

Measures:
- Variance explained by each component
- Loading matrix (signal weights)
- Effective dimensionality

Phase: Unbound
Normalization: Z-score required
"""

import logging
from typing import Dict, Any, Optional
from datetime import date

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="pca",
    engine_type="geometry",
    description="Principal component analysis for dimensionality",
    domains={"structure", "dimensionality"},
    requires_window=True,
    deterministic=True,
)


class PCAEngine(BaseEngine):
    """
    Principal Component Analysis engine.

    Outputs:
        - results.pca_loadings: Signal loadings per component
        - results.pca_variance: Variance explained per component
    """

    name = "pca"
    phase = "derived"
    default_normalization = "zscore"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        n_components: Optional[int] = None,
        **params
    ) -> Dict[str, Any]:
        """
        Run PCA analysis.
        
        Args:
            df: Normalized signal data (rows=dates, cols=signals)
            run_id: Unique run identifier
            n_components: Number of components (default: all)
        
        Returns:
            Dict with summary metrics
        """
        # Clean layer guarantees no NaN
        df_clean = df
        
        if len(df_clean) < 3 * len(df_clean.columns):
            logger.warning(
                f"Limited samples ({len(df_clean)}) for {len(df_clean.columns)} signals. "
                f"Recommended: {3 * len(df_clean.columns)}+"
            )
        
        # Fit PCA
        n_comp = n_components or min(len(df_clean), len(df_clean.columns))
        pca = PCA(n_components=n_comp)
        pca.fit(df_clean.values)
        
        # Extract results
        loadings = pd.DataFrame(
            pca.components_.T,
            index=df_clean.columns,
            columns=[f"PC{i+1}" for i in range(n_comp)]
        )
        
        variance = pd.DataFrame({
            "component": range(1, n_comp + 1),
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        })
        
        # Store results
        window_start, window_end = get_window_dates(df_clean)
        
        self._store_loadings(loadings, window_start, window_end, run_id)
        self._store_variance(variance, window_start, window_end, run_id)
        
        # Compute summary metrics
        n_components_90 = np.searchsorted(
            np.cumsum(pca.explained_variance_ratio_), 0.9
        ) + 1
        
        metrics = {
            "n_signals": len(df_clean.columns),
            "n_samples": len(df_clean),
            "n_components": n_comp,
            "variance_pc1": float(pca.explained_variance_ratio_[0]),
            "variance_pc2": float(pca.explained_variance_ratio_[1]) if n_comp > 1 else 0,
            "components_for_90pct": int(n_components_90),
            "effective_dimensionality": float(
                1 / np.sum(pca.explained_variance_ratio_ ** 2)
            ),
        }
        
        logger.info(
            f"PCA complete: PC1 explains {metrics['variance_pc1']:.1%}, "
            f"{metrics['components_for_90pct']} components for 90%"
        )
        
        return metrics
    
    def _store_loadings(
        self,
        loadings: pd.DataFrame,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store loadings to results.pca_loadings."""
        # Reshape: one row per (signal, component)
        records = []
        for signal_id in loadings.index:
            for i, col in enumerate(loadings.columns):
                records.append({
                    "signal_id": signal_id,
                    "window_start": window_start,
                    "window_end": window_end,
                    "component": i + 1,
                    "loading": float(loadings.loc[signal_id, col]),
                    "run_id": run_id,
                })
        
        df = pd.DataFrame(records)
        self.store_results("pca_loadings", df, run_id)
    
    def _store_variance(
        self,
        variance: pd.DataFrame,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store variance to results.pca_variance."""
        df = variance.copy()
        df["window_start"] = window_start
        df["window_end"] = window_end
        df["run_id"] = run_id
        
        # Reorder columns to match schema
        df = df[[
            "window_start", "window_end", "component",
            "variance_explained", "cumulative_variance", "run_id"
        ]]
        
        self.store_results("pca_variance", df, run_id)


# =============================================================================
# Standalone function with derivation
# =============================================================================

def compute_pca_with_derivation(
    data: np.ndarray,
    signal_ids: list = None,
    window_id: str = "0",
    window_start: str = None,
    window_end: str = None,
    n_components: int = None,
) -> tuple:
    """
    Compute PCA with full mathematical derivation.

    Args:
        data: Matrix of shape (n_samples, n_features) - rows=time, cols=signals
        signal_ids: List of signal names
        window_id: Window identifier
        window_start, window_end: Date range
        n_components: Number of components (default: all)

    Returns:
        tuple: (result_dict, Derivation object)
    """
    from prism.entry_points.derivations.base import Derivation

    n_samples, n_features = data.shape
    if signal_ids is None:
        signal_ids = [f"X{i}" for i in range(n_features)]

    deriv = Derivation(
        engine_name="pca",
        method_name="Principal Component Analysis",
        signal_id=f"cohort_{n_features}_signals",
        window_id=window_id,
        window_start=window_start,
        window_end=window_end,
        sample_size=n_samples,
        parameters={'n_features': n_features, 'n_components': n_components}
    )

    # Step 1: Input data summary
    deriv.add_step(
        title="Input Data Matrix",
        equation="X ∈ ℝⁿˣᵖ where n=samples, p=features",
        calculation=f"X shape: {data.shape}\nn = {n_samples} time points\np = {n_features} signals\n\nSignals: {signal_ids[:5]}{'...' if len(signal_ids) > 5 else ''}",
        result=n_features,
        result_name="p",
        notes="Each column is an signal, each row is a time point"
    )

    # Step 2: Center the data
    means = np.mean(data, axis=0)
    X_centered = data - means

    deriv.add_step(
        title="Center the Data (Remove Mean)",
        equation="X̃ = X - μ  where μⱼ = (1/n) Σᵢ Xᵢⱼ",
        calculation=f"Column means:\n" + "\n".join([
            f"  μ({signal_ids[i]}) = {means[i]:.4f}" for i in range(min(5, n_features))
        ]) + ("\n  ..." if n_features > 5 else ""),
        result=means[:5].tolist(),
        result_name="μ",
        notes="Centering ensures PC1 passes through the data centroid"
    )

    # Step 3: Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)

    deriv.add_step(
        title="Compute Covariance Matrix",
        equation="Σ = (1/(n-1)) X̃ᵀX̃",
        calculation=f"Covariance matrix Σ ∈ ℝᵖˣᵖ = {cov_matrix.shape}\n\nSample entries:\n  Σ[0,0] = Var({signal_ids[0]}) = {cov_matrix[0,0]:.4f}\n  Σ[0,1] = Cov({signal_ids[0]},{signal_ids[1] if n_features > 1 else 'X1'}) = {cov_matrix[0,1] if n_features > 1 else 0:.4f}",
        result=cov_matrix[0, 0],
        result_name="Σ",
        notes="Covariance matrix captures linear relationships between all signal pairs"
    )

    # Step 4: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort by decreasing eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    deriv.add_step(
        title="Eigendecomposition of Covariance Matrix",
        equation="Σv = λv  (solve for eigenvalues λ and eigenvectors v)",
        calculation=f"Eigenvalues (sorted):\n  λ₁ = {eigenvalues[0]:.4f}\n  λ₂ = {eigenvalues[1] if len(eigenvalues) > 1 else 0:.4f}\n  λ₃ = {eigenvalues[2] if len(eigenvalues) > 2 else 0:.4f}\n  ...\n\nTotal variance: Σλ = {np.sum(eigenvalues):.4f}",
        result=eigenvalues[:3].tolist(),
        result_name="λ",
        notes="Eigenvalues = variance explained by each principal component"
    )

    # Step 5: Compute variance explained
    total_var = np.sum(eigenvalues)
    var_explained = eigenvalues / total_var if total_var > 0 else eigenvalues
    cumulative_var = np.cumsum(var_explained)

    deriv.add_step(
        title="Variance Explained Ratio",
        equation="VEᵢ = λᵢ / Σⱼλⱼ",
        calculation=f"Variance explained:\n  PC1: {var_explained[0]:.4f} ({var_explained[0]*100:.1f}%)\n  PC2: {var_explained[1] if len(var_explained) > 1 else 0:.4f} ({var_explained[1]*100 if len(var_explained) > 1 else 0:.1f}%)\n  PC3: {var_explained[2] if len(var_explained) > 2 else 0:.4f} ({var_explained[2]*100 if len(var_explained) > 2 else 0:.1f}%)\n\nCumulative:\n  PC1: {cumulative_var[0]*100:.1f}%\n  PC1-2: {cumulative_var[1]*100 if len(cumulative_var) > 1 else 0:.1f}%\n  PC1-3: {cumulative_var[2]*100 if len(cumulative_var) > 2 else 0:.1f}%",
        result=var_explained[0],
        result_name="VE₁",
        notes="PC1 captures the direction of maximum variance"
    )

    # Step 6: Loadings (first PC)
    pc1_loadings = eigenvectors[:, 0]

    deriv.add_step(
        title="PC1 Loadings (Signal Weights)",
        equation="PC1 = Σⱼ wⱼXⱼ  where wⱼ = eigenvector₁[j]",
        calculation=f"PC1 loadings:\n" + "\n".join([
            f"  w({signal_ids[i]}) = {pc1_loadings[i]:.4f}" for i in range(min(5, n_features))
        ]) + ("\n  ..." if n_features > 5 else ""),
        result=pc1_loadings[:5].tolist(),
        result_name="w₁",
        notes="Loadings show each signal's contribution to PC1"
    )

    # Step 7: Effective dimensionality
    eff_dim = 1 / np.sum(var_explained ** 2) if np.sum(var_explained ** 2) > 0 else 1

    deriv.add_step(
        title="Effective Dimensionality (Participation Ratio)",
        equation="d_eff = 1 / Σᵢ(VEᵢ)²",
        calculation=f"d_eff = 1 / ({var_explained[0]:.4f}² + {var_explained[1] if len(var_explained) > 1 else 0:.4f}² + ...)\nd_eff = 1 / {np.sum(var_explained ** 2):.4f}",
        result=eff_dim,
        result_name="d_eff",
        notes="Number of 'effective' independent dimensions; low = concentrated variance"
    )

    # Step 8: Components for 90% variance
    n_comp_90 = int(np.searchsorted(cumulative_var, 0.9) + 1)

    deriv.add_step(
        title="Components for 90% Variance",
        equation="k₉₀ = min{k : Σᵢ₌₁ᵏ VEᵢ ≥ 0.90}",
        calculation=f"Cumulative variance:\n" + "\n".join([
            f"  PC1-{i+1}: {cumulative_var[i]*100:.1f}%" for i in range(min(n_comp_90 + 1, len(cumulative_var)))
        ]) + f"\n\nk₉₀ = {n_comp_90}",
        result=n_comp_90,
        result_name="k₉₀",
        notes=f"{n_comp_90} of {n_features} components needed for 90% variance"
    )

    # Final result
    result = {
        'variance_pc1': float(var_explained[0]),
        'variance_pc2': float(var_explained[1]) if len(var_explained) > 1 else 0,
        'variance_pc3': float(var_explained[2]) if len(var_explained) > 2 else 0,
        'cumulative_variance_3': float(cumulative_var[2]) if len(cumulative_var) > 2 else float(cumulative_var[-1]),
        'effective_dimensionality': float(eff_dim),
        'components_for_90pct': n_comp_90,
        'n_features': n_features,
    }

    deriv.final_result = var_explained[0]
    deriv.prism_output = var_explained[0]

    # Interpretation
    if var_explained[0] > 0.6:
        interp = f"PC1 explains {var_explained[0]*100:.1f}% of variance - **strong dominant factor** drives the cohort."
    elif var_explained[0] > 0.3:
        interp = f"PC1 explains {var_explained[0]*100:.1f}% - **moderate structure** with multiple important factors."
    else:
        interp = f"PC1 explains only {var_explained[0]*100:.1f}% - **diverse behavior**, no single dominant pattern."

    interp += f" Effective dimensionality d_eff = {eff_dim:.1f} (of {n_features} possible)."

    if n_comp_90 <= 3:
        interp += f" Only {n_comp_90} components needed for 90% variance - **low-dimensional structure**."

    deriv.interpretation = interp

    return result, deriv
