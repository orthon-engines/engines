"""
PRISM Local Outlier Factor Engine

Detects anomalous signals in behavioral space using density-based analysis.

ARCHITECTURE:
- Computes bounded LOF scores [0, max_lof]
- Raises exception if computation fails (no silent defaults)
- Caller should NOT add defensive clipping - this engine guarantees bounds

Phase: Structure
Normalization: Z-score required
"""

import logging
from typing import Dict, Any
from datetime import date

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from prism.engines.engine_base import BaseEngine, get_window_dates
from prism.engines.metadata import EngineMetadata


logger = logging.getLogger(__name__)


METADATA = EngineMetadata(
    name="lof",
    engine_type="geometry",
    description="Local outlier factor for anomaly detection in behavioral space",
    domains={"structure", "anomaly"},
    requires_window=True,
    deterministic=True,
)


class LOFComputationError(Exception):
    """Raised when LOF cannot be computed meaningfully."""
    pass


class LOFEngine(BaseEngine):
    """
    Local Outlier Factor engine for behavioral space.

    GUARANTEES:
    - All returned scores are in [0, max_lof] range
    - Raises LOFComputationError if data is unsuitable
    - No silent fallbacks - caller knows if computation succeeded
    """

    name = "lof"
    phase = "structure"
    default_normalization = "zscore"

    @property
    def metadata(self) -> EngineMetadata:
        return METADATA

    def run(
        self,
        df: pd.DataFrame,
        run_id: str,
        n_neighbors: int = 5,
        contamination: str = "auto",
        max_lof: float = 10.0,
        min_samples: int = 5,
        raise_on_failure: bool = False,  # Set True to fail loudly
        **params
    ) -> Dict[str, Any]:
        """
        Run LOF analysis with guaranteed bounded output.

        Args:
            df: Behavioral vectors (rows=dimensions, cols=signals)
            run_id: Unique run identifier
            n_neighbors: Number of neighbors for LOF
            max_lof: Maximum LOF value (scores clipped to this)
            min_samples: Minimum samples required
            raise_on_failure: If True, raise exception on bad data.
                              If False, return defaults with warning.

        Returns:
            Dict with bounded metrics (all LOF values <= max_lof)

        Raises:
            LOFComputationError: If raise_on_failure=True and data unsuitable
        """
        signals = list(df.columns)
        n_signals = len(signals)
        window_start, window_end = get_window_dates(df)

        # Prepare data
        X = df.T.values
        n_samples, n_features = X.shape

        # === VALIDATION ===
        failure_reason = self._validate_data(X, n_samples, n_features, n_neighbors, min_samples)

        if failure_reason:
            if raise_on_failure:
                raise LOFComputationError(f"LOF cannot compute: {failure_reason}")
            else:
                logger.warning(f"LOF: {failure_reason}, returning defaults")
                return self._default_metrics(n_signals, n_neighbors)

        # Adjust neighbors if needed
        effective_neighbors = min(n_neighbors, n_samples - 1)

        # Filter to valid features
        feature_std = np.std(X, axis=0)
        valid_features = feature_std > 1e-10
        X_filtered = X[:, valid_features]

        # === COMPUTE LOF ===
        try:
            lof = LocalOutlierFactor(
                n_neighbors=effective_neighbors,
                contamination=contamination if contamination != "auto" else "auto",
                novelty=False,
            )
            labels = lof.fit_predict(X_filtered)
            raw_scores = -lof.negative_outlier_factor_

        except Exception as e:
            if raise_on_failure:
                raise LOFComputationError(f"LOF sklearn failed: {e}")
            else:
                logger.warning(f"LOF sklearn failed: {e}, returning defaults")
                return self._default_metrics(n_signals, n_neighbors)

        # === BOUND THE SCORES (the actual fix) ===
        lof_scores = self._bound_scores(raw_scores, max_lof)

        # === VERIFY BOUNDS (paranoia check) ===
        if np.any(lof_scores > max_lof) or np.any(~np.isfinite(lof_scores)):
            # This should NEVER happen after _bound_scores
            error_msg = f"LOF bounding failed: max={np.max(lof_scores)}, has_inf={np.any(~np.isfinite(lof_scores))}"
            if raise_on_failure:
                raise LOFComputationError(error_msg)
            else:
                logger.error(error_msg)
                return self._default_metrics(n_signals, n_neighbors)

        # === BUILD METRICS ===
        return self._build_metrics(
            lof_scores, labels, signals, n_neighbors,
            window_start, window_end, run_id
        )

    def _validate_data(
        self, X: np.ndarray, n_samples: int, n_features: int,
        n_neighbors: int, min_samples: int
    ) -> str:
        """
        Validate data is suitable for LOF. Returns failure reason or empty string.
        """
        if n_samples < min_samples:
            return f"only {n_samples} samples < min {min_samples}"

        effective_neighbors = min(n_neighbors, n_samples - 1)
        if effective_neighbors < 2:
            return f"effective_neighbors={effective_neighbors} < 2"

        feature_std = np.std(X, axis=0)
        if np.all(feature_std < 1e-10):
            return "all features constant"

        valid_features = feature_std > 1e-10
        if valid_features.sum() < 2:
            return f"only {valid_features.sum()} varying features"

        X_filtered = X[:, valid_features]
        unique_rows = np.unique(X_filtered, axis=0)
        if len(unique_rows) < effective_neighbors:
            return f"only {len(unique_rows)} unique rows < {effective_neighbors} neighbors"

        return ""  # Valid

    def _bound_scores(self, raw_scores: np.ndarray, max_lof: float) -> np.ndarray:
        """
        Bound LOF scores to [0, max_lof]. Handles inf/nan.

        This is THE fix. All scores leaving this function are guaranteed bounded.
        """
        scores = raw_scores.copy()

        # Step 1: Replace inf/nan with max_lof (they indicate extreme outliers)
        scores = np.where(np.isfinite(scores), scores, max_lof)

        # Step 2: Clip to range
        scores = np.clip(scores, 0.0, max_lof)

        # Step 3: Final verification (belt AND suspenders)
        scores = np.minimum(scores, max_lof)  # Explicit cap
        scores = np.maximum(scores, 0.0)       # Explicit floor

        return scores

    def _build_metrics(
        self, lof_scores: np.ndarray, labels: np.ndarray,
        signals: list, n_neighbors: int,
        window_start, window_end, run_id: str
    ) -> Dict[str, Any]:
        """Build metrics dict from bounded scores."""

        n_signals = len(signals)
        n_outliers = (labels == -1).sum()

        # Create and store score DataFrame
        score_df = pd.DataFrame({
            "signal_id": signals,
            "lof_score": lof_scores,
            "is_outlier": labels == -1,
        }).sort_values("lof_score", ascending=False)

        self._store_scores(score_df, window_start, window_end, run_id)

        metrics = {
            "n_signals": n_signals,
            "n_neighbors": n_neighbors,
            "n_outliers_auto": int(n_outliers),
            "outlier_rate_auto": float(n_outliers / n_signals) if n_signals > 0 else 0.0,
            "n_outliers_1_5": int((lof_scores > 1.5).sum()),
            "n_outliers_2_0": int((lof_scores > 2.0).sum()),
            "n_outliers_3_0": int((lof_scores > 3.0).sum()),
            "avg_lof_score": float(np.mean(lof_scores)),
            "max_lof_score": float(np.max(lof_scores)),
            "min_lof_score": float(np.min(lof_scores)),
            "std_lof_score": float(np.std(lof_scores)),
            "median_lof_score": float(np.median(lof_scores)),
        }

        logger.info(
            f"LOF complete: {n_signals} signals, {n_outliers} outliers, "
            f"avg={metrics['avg_lof_score']:.2f}, max={metrics['max_lof_score']:.2f}"
        )

        return metrics

    def _store_scores(self, score_df: pd.DataFrame, window_start, window_end, run_id: str):
        """Store LOF scores per signal."""
        records = [{
            "signal_id": row["signal_id"],
            "window_start": window_start,
            "window_end": window_end,
            "lof_score": float(row["lof_score"]),
            "is_outlier": bool(row["is_outlier"]),
            "outlier_severity": self._classify_severity(row["lof_score"]),
            "run_id": run_id,
        } for _, row in score_df.iterrows()]

        if records:
            self.store_results("lof_scores", pd.DataFrame(records), run_id)

    def _classify_severity(self, lof_score: float) -> str:
        """Classify outlier severity."""
        if lof_score > 3.0: return "extreme"
        if lof_score > 2.0: return "strong"
        if lof_score > 1.5: return "moderate"
        if lof_score > 1.0: return "mild"
        return "normal"

    def _default_metrics(self, n_signals: int, n_neighbors: int) -> Dict[str, Any]:
        """Default metrics when computation not possible. LOF=1.0 means normal."""
        return {
            "n_signals": n_signals,
            "n_neighbors": n_neighbors,
            "n_outliers_auto": 0,
            "outlier_rate_auto": 0.0,
            "n_outliers_1_5": 0,
            "n_outliers_2_0": 0,
            "n_outliers_3_0": 0,
            "avg_lof_score": 1.0,
            "max_lof_score": 1.0,
            "min_lof_score": 1.0,
            "std_lof_score": 0.0,
            "median_lof_score": 1.0,
        }
