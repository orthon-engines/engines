"""
Canonical engine base for PRISM.

Defines the contract shared by vector and geometry engines.
This file contains no domain-specific logic.
"""

import logging
from abc import ABC, abstractmethod, abstractproperty

from prism.engines.metadata import EngineMetadata
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import uuid

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def get_window_dates(df: pd.DataFrame) -> Tuple[date, date]:
    """
    Safely extract window start/end dates from a DataFrame.

    For signal topology data (DatetimeIndex): returns min/max dates
    For behavioral vectors (string index): returns today's date for both

    This allows geometry engines to work with both:
    - Traditional signal topology (rows=dates, cols=signals)
    - Behavioral vectors (rows=signals, cols=dimensions)
    """
    today = date.today()

    if df.empty:
        return today, today

    try:
        # Try to get dates from DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.min().date(), df.index.max().date()

        # Try to convert index to datetime (suppress warnings for non-date strings)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idx = pd.to_datetime(df.index, errors='coerce')
            if not idx.isna().all():
                valid_dates = idx.dropna()
                if len(valid_dates) > 0:
                    return valid_dates.min().date(), valid_dates.max().date()
    except Exception:
        pass

    # Default: use today's date (behavioral vectors have no inherent date range)
    return today, today


@dataclass
class EngineResult:
    """Result of an engine run."""
    engine_name: str
    run_id: str
    success: bool
    started_at: datetime
    completed_at: Optional[datetime] = None
    window_start: Optional[date] = None
    window_end: Optional[date] = None
    normalization: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def runtime_seconds(self) -> Optional[float]:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def summary(self) -> str:
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Engine: {self.engine_name}",
            f"Run ID: {self.run_id}",
            f"Status: {status}",
            f"Runtime: {self.runtime_seconds:.2f}s",
        ]
        if self.window_start and self.window_end:
            lines.append(f"Window: {self.window_start} to {self.window_end}")
        if self.normalization:
            lines.append(f"Normalization: {self.normalization}")
        if self.metrics:
            lines.append(f"Metrics: {self.metrics}")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


class BaseEngine(ABC):
    """
    Abstract base class for all PRISM analysis engines.

    Subclasses must implement:
        - name: Engine identifier
        - phase: Which phase this engine belongs to
        - run(): Execute the analysis

    Usage:
        class PCAEngine(BaseEngine):
            name = "pca"
            phase = "derived"

            def run(self, signals, window_start, window_end, **params):
                # ... implementation
                return results_df
    """

    # Subclasses must define these
    name: str = "base"
    phase: str = "derived"  # 'derived', 'structure', 'binding'

    # Default normalization (override in subclass if needed)
    default_normalization: Optional[str] = None  # 'zscore', 'minmax', 'returns', etc.

    @property
    def metadata(self) -> EngineMetadata:
        """Return engine metadata. Subclasses must implement."""
        raise NotImplementedError("Subclasses must define metadata property")

    def __repr__(self) -> str:
        """Engine self-identification."""
        try:
            m = self.metadata
            return f"<Engine {m.name} ({m.engine_type})>"
        except NotImplementedError:
            return f"<Engine {self.name}>"

    def __init__(self):
        """Initialize engine."""
        pass

    # -------------------------------------------------------------------------
    # Normalization (engine calls what it needs)
    # -------------------------------------------------------------------------

    def normalize_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalization (mean=0, std=1) per column."""
        return (df - df.mean()) / df.std()

    def normalize_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalization (0-1) per column."""
        return (df - df.min()) / (df.max() - df.min())

    def normalize_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to percentage returns."""
        return df.pct_change().dropna()

    def normalize_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to ranks (uniform distribution)."""
        return df.rank(pct=True)

    def normalize_diff(self, df: pd.DataFrame) -> pd.DataFrame:
        """First difference (for stationarity)."""
        return df.diff().dropna()

    def discretize(
        self,
        df: pd.DataFrame,
        n_bins: int = 8,
        method: str = "quantile"
    ) -> pd.DataFrame:
        """
        Discretize continuous data into bins.

        Args:
            df: Input DataFrame
            n_bins: Number of bins
            method: 'quantile' or 'uniform'
        """
        result = df.copy()
        for col in result.columns:
            if method == "quantile":
                result[col] = pd.qcut(
                    result[col], q=n_bins, labels=False, duplicates="drop"
                )
            else:
                result[col] = pd.cut(
                    result[col], bins=n_bins, labels=False
                )
        return result

    # -------------------------------------------------------------------------
    # Run Orchestration
    # -------------------------------------------------------------------------

    def execute(
        self,
        names: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        normalization: Optional[str] = None,
        **params
    ) -> EngineResult:
        """
        Execute the engine with logging and error handling.

        Engines always run when invoked. No gating.
        Engines read from signals.all and see only (date, value, name).

        Args:
            names: List of signal names (None = all signals)
            start_date: Window start
            end_date: Window end
            normalization: Override default normalization
            **params: Engine-specific parameters

        Returns:
            EngineResult with status and metrics
        """
        run_id = self._generate_run_id()
        started_at = datetime.now()

        result = EngineResult(
            engine_name=self.name,
            run_id=run_id,
            success=False,
            started_at=started_at,
            window_start=start_date,
            window_end=end_date,
            normalization=normalization or self.default_normalization,
            parameters=params,
        )

        try:
            # Log run start
            self._record_run_start(result)

            # Load data from signals.all
            if names:
                df = self.load_signals(names, start_date, end_date)
            else:
                df = self.load_all_signals(start_date, end_date)

            if df.empty:
                raise ValueError("No data available for specified signals/window")

            # Update window from actual data
            result.window_start = df.index.min().date()
            result.window_end = df.index.max().date()

            # Apply normalization if specified
            norm = normalization or self.default_normalization
            if norm:
                df = self._apply_normalization(df, norm)
                result.normalization = norm

            # Run the actual analysis
            logger.info(f"Running {self.name} on {len(df.columns)} signals")
            metrics = self.run(df, run_id=run_id, **params)

            result.success = True
            result.metrics = metrics or {}

        except Exception as e:
            logger.exception(f"Engine {self.name} failed: {e}")
            result.error = str(e)

        finally:
            result.completed_at = datetime.now()
            self._record_run_complete(result)

        return result

    def _apply_normalization(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply specified normalization method."""
        methods = {
            "zscore": self.normalize_zscore,
            "minmax": self.normalize_minmax,
            "returns": self.normalize_returns,
            "rank": self.normalize_rank,
            "diff": self.normalize_diff,
        }

        if method not in methods:
            raise ValueError(f"Unknown normalization: {method}. Options: {list(methods.keys())}")

        return methods[method](df)

    @abstractmethod
    def run(self, df: pd.DataFrame, run_id: str, **params) -> Dict[str, Any]:
        """
        Execute the analysis. Subclasses must implement.

        Args:
            df: Prepared DataFrame (normalized if applicable)
            run_id: Unique run identifier
            **params: Engine-specific parameters

        Returns:
            Dict of metrics/summary statistics
        """
        pass

    # -------------------------------------------------------------------------
    # Result Storage
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # Meta Logging
    # -------------------------------------------------------------------------

    def _record_run_start(self, result: EngineResult):
        """Record engine run start. No-op - metadata tracked via EngineResult."""
        pass

    def _record_run_complete(self, result: EngineResult):
        """Update engine run record on completion. No-op - metadata tracked via EngineResult."""
        pass

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"{self.name}_{timestamp}_{short_uuid}"
