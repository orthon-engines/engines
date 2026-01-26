"""
PRISM DMD/Koopman Engine

Dynamic Mode Decomposition for dynamic mode results.

Measures:
- Eigenvalues (growth/decay rates, frequencies)
- Mode amplitudes/energies
- Reconstruction error (structural change signal)
- Dominant oscillatory modes

Phase: Structure
Normalization: Z-score preferred
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import date

import numpy as np
import pandas as pd
from scipy import linalg

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="dmd",
    engine_type="geometry",
    description="Dynamic mode decomposition for oscillatory patterns",
    domains={"structure", "dynamics"},
    requires_window=True,
    deterministic=True,
)


class DMDEngine(BaseEngine):
    """
    Dynamic Mode Decomposition engine.
    
    Analyzes system dynamics via modes and eigenvalues.
    Captures oscillatory behavior, stability, and dynamical results.
    
    Aligns with PRISM's "waves/vibrations" framing.
    
    Outputs:
        - results.dmd_modes: Eigenvalue summaries
        - results.dmd_reconstruction: Reconstruction error
    """
    
    name = "dmd"
    phase = "structure"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA
    default_normalization = "zscore"
    
    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        rank: Optional[int] = None,
        delay_embedding: int = 1,
        **params
    ) -> Dict[str, Any]:
        """
        Run DMD analysis.
        
        Args:
            df: Normalized signal data
            run_id: Unique run identifier
            rank: Truncation rank (None = auto via SVD threshold)
            delay_embedding: Time-delay embedding (1 = no embedding)
        
        Returns:
            Dict with summary metrics
        """
        df_clean = df
        
        window_start, window_end = get_window_dates(df_clean)
        
        # Prepare data matrix (features x time)
        X = df_clean.values.T  # Shape: (n_features, n_samples)
        
        # Apply time-delay embedding if requested
        if delay_embedding > 1:
            X = self._delay_embed(X, delay_embedding)
        
        # Run DMD
        eigenvalues, modes, amplitudes, reconstruction_error = self._compute_dmd(
            X, rank
        )
        
        # Extract mode characteristics
        mode_info = self._analyze_modes(eigenvalues, amplitudes)
        
        # Store results
        self._store_modes(
            mode_info, window_start, window_end, run_id
        )
        
        # Summary metrics
        n_modes = len(eigenvalues)
        stable_modes = sum(1 for ev in eigenvalues if np.abs(ev) <= 1.0)
        
        # Dominant frequency
        frequencies = np.abs(np.angle(eigenvalues)) / (2 * np.pi)
        dominant_freq = frequencies[np.argmax(amplitudes)] if len(amplitudes) > 0 else 0
        
        metrics = {
            "n_modes": n_modes,
            "rank_used": rank or n_modes,
            "reconstruction_error": float(reconstruction_error),
            "stable_modes": stable_modes,
            "unstable_modes": n_modes - stable_modes,
            "dominant_frequency": float(dominant_freq),
            "total_energy": float(np.sum(amplitudes ** 2)),
            "delay_embedding": delay_embedding,
        }
        
        logger.info(
            f"DMD complete: {n_modes} modes, "
            f"recon error={reconstruction_error:.4f}, "
            f"stable={stable_modes}/{n_modes}"
        )
        
        return metrics
    
    def _delay_embed(self, X: np.ndarray, delays: int) -> np.ndarray:
        """
        Time-delay embedding to augment state space.
        
        Converts X(t) to [X(t), X(t-1), ..., X(t-delays+1)]
        """
        n_features, n_samples = X.shape
        n_embedded = n_samples - delays + 1
        
        embedded = np.zeros((n_features * delays, n_embedded))
        
        for d in range(delays):
            start = delays - 1 - d
            end = start + n_embedded
            embedded[d * n_features:(d + 1) * n_features, :] = X[:, start:end]
        
        return embedded
    
    def _compute_dmd(
        self,
        X: np.ndarray,
        rank: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Exact DMD algorithm.
        
        Given snapshots X = [x_1, ..., x_m], find A such that x_{k+1} ≈ A x_k
        
        Returns (eigenvalues, modes, amplitudes, reconstruction_error)
        """
        # Split into X (past) and Y (future)
        X_past = X[:, :-1]
        X_future = X[:, 1:]
        
        # SVD of X_past
        U, s, Vh = linalg.svd(X_past, full_matrices=False)
        
        # Determine rank
        if rank is None:
            # Auto-select based on energy capture (99%)
            cumulative_energy = np.cumsum(s ** 2) / np.sum(s ** 2)
            rank = np.searchsorted(cumulative_energy, 0.99) + 1
            rank = max(1, min(rank, len(s)))
        
        rank = min(rank, len(s))
        
        # Truncate
        U_r = U[:, :rank]
        s_r = s[:rank]
        Vh_r = Vh[:rank, :]
        
        # Build reduced A matrix: A_tilde = U_r^T @ Y @ V_r @ S_r^{-1}
        A_tilde = U_r.T @ X_future @ Vh_r.T @ np.diag(1 / s_r)
        
        # Eigendecomposition
        eigenvalues, W = linalg.eig(A_tilde)
        
        # DMD modes (in original space)
        modes = X_future @ Vh_r.T @ np.diag(1 / s_r) @ W
        
        # Mode amplitudes (initial condition projection)
        x0 = X[:, 0]
        amplitudes = np.abs(np.linalg.lstsq(modes, x0, rcond=None)[0])
        
        # Reconstruction error
        X_reconstructed = self._reconstruct(modes, eigenvalues, amplitudes, X.shape[1])
        error = np.linalg.norm(X - X_reconstructed, 'fro') / np.linalg.norm(X, 'fro')
        
        return eigenvalues, modes, amplitudes, error
    
    def _reconstruct(
        self,
        modes: np.ndarray,
        eigenvalues: np.ndarray,
        amplitudes: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """Reconstruct signal topology from DMD modes."""
        n_modes = len(eigenvalues)
        n_features = modes.shape[0]
        
        X_recon = np.zeros((n_features, n_steps), dtype=complex)
        
        for k in range(n_steps):
            for j in range(n_modes):
                X_recon[:, k] += amplitudes[j] * (eigenvalues[j] ** k) * modes[:, j]
        
        return np.real(X_recon)
    
    def _analyze_modes(
        self,
        eigenvalues: np.ndarray,
        amplitudes: np.ndarray
    ) -> list:
        """Extract characteristics of each mode."""
        modes = []
        
        for i, (ev, amp) in enumerate(zip(eigenvalues, amplitudes)):
            # Growth rate (log of magnitude)
            growth_rate = np.log(np.abs(ev))
            
            # Frequency (angle)
            frequency = np.abs(np.angle(ev)) / (2 * np.pi)
            
            # Period (if oscillatory)
            period = 1 / frequency if frequency > 0.01 else np.inf
            
            # Stability
            is_stable = np.abs(ev) <= 1.0
            
            modes.append({
                "mode_idx": i,
                "eigenvalue_real": float(np.real(ev)),
                "eigenvalue_imag": float(np.imag(ev)),
                "magnitude": float(np.abs(ev)),
                "growth_rate": float(growth_rate),
                "frequency": float(frequency),
                "period": float(period) if not np.isinf(period) else None,
                "amplitude": float(amp),
                "energy": float(amp ** 2),
                "is_stable": is_stable,
            })
        
        return modes
    
    def _store_modes(
        self,
        mode_info: list,
        window_start: date,
        window_end: date,
        run_id: str,
    ):
        """Store DMD mode information."""
        # Store top modes as geometry fingerprints
        # Sort by energy
        sorted_modes = sorted(mode_info, key=lambda x: x["energy"], reverse=True)
        
        records = []
        for i, mode in enumerate(sorted_modes[:10]):  # Top 10 modes
            records.append({
                "signal_id": f"dmd_mode_{i}",
                "window_start": window_start,
                "window_end": window_end,
                "dimension": "growth_rate",
                "value": mode["growth_rate"],
                "run_id": run_id,
            })
            records.append({
                "signal_id": f"dmd_mode_{i}",
                "window_start": window_start,
                "window_end": window_end,
                "dimension": "frequency",
                "value": mode["frequency"],
                "run_id": run_id,
            })
            records.append({
                "signal_id": f"dmd_mode_{i}",
                "window_start": window_start,
                "window_end": window_end,
                "dimension": "energy",
                "value": mode["energy"],
                "run_id": run_id,
            })
        
        if records:
            df = pd.DataFrame(records)
            ##self.store_results("geometry_fingerprints", df, run_id)
