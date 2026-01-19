"""
PRISM Energy Dynamics Engine
============================

Computes temporal dynamics of system energy from geometry.displacement.

Energy represents the kinetic motion of the behavioral manifold - how fast
and how much the system is changing. This engine transforms point-in-time
energy readings into temporal features.

Input: geometry.displacement signal topology
Output: Energy dynamics metrics (MA, acceleration, z-scores)

Key Metrics:
    - energy_ma5/ma20: Smoothed energy trends
    - energy_acceleration: Rate of change in energy
    - energy_zscore: Current energy vs historical distribution

Usage:
    from prism.engines.state.energy_dynamics import EnergyDynamicsEngine

    engine = EnergyDynamicsEngine()
    result = engine.run(displacement_df)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EnergyDynamicsResult:
    """Result from energy dynamics computation."""
    energy_total: float
    energy_ma5: Optional[float]
    energy_ma20: Optional[float]
    energy_acceleration: Optional[float]
    energy_zscore: Optional[float]
    energy_trend: str  # 'rising', 'falling', 'stable'
    metrics: Dict[str, float]


class EnergyDynamicsEngine:
    """
    Compute temporal dynamics of system energy.

    Energy is derived from geometry.displacement.energy_total which measures
    the weighted movement of signals between snapshots.

    High energy = system is changing rapidly
    Low energy = system is stable
    Rising energy = acceleration in change
    Falling energy = deceleration, stabilization
    """

    def __init__(self, ma_short: int = None, ma_long: int = None, zscore_window: int = None):
        """
        Initialize engine.

        Args:
            ma_short: Short moving average window. If None, loads from config.
            ma_long: Long moving average window. If None, loads from config.
            zscore_window: Window for z-score calculation. If None, loads from config.
        """
        # Load window parameters from config if not provided
        if ma_short is None or ma_long is None or zscore_window is None:
            config = self._load_window_config()
            ma_short = ma_short or config.get('ma_short', config.get('min_samples', 5))
            ma_long = ma_long or config.get('ma_long', config.get('window_samples', 20))
            zscore_window = zscore_window or config.get('zscore_window', config.get('window_samples', 20))

        self.ma_short = ma_short
        self.ma_long = ma_long
        self.zscore_window = zscore_window

    def _load_window_config(self) -> dict:
        """Load window parameters from domain_info or domain.yaml."""
        import os
        import json
        from pathlib import Path

        # Try domain_info.json first
        try:
            from prism.db.parquet_store import get_data_root
            domain_info_path = get_data_root() / "domain_info.json"
            if domain_info_path.exists():
                with open(domain_info_path) as f:
                    return json.load(f)
        except Exception:
            pass

        # Try domain.yaml
        try:
            from prism.config.loader import load_clock_config
            return load_clock_config()
        except Exception:
            pass

        raise RuntimeError(
            "No window configuration found. "
            "Run signal_vector with --adaptive flag first, or configure config/domain.yaml"
        )

    def run(self, energy_series: pd.Series, current_idx: int = -1) -> EnergyDynamicsResult:
        """
        Compute energy dynamics from a signal topology of energy values.

        Args:
            energy_series: Series of energy_total values indexed by date
            current_idx: Index position to compute dynamics for (default: last)

        Returns:
            EnergyDynamicsResult with all computed metrics
        """
        if len(energy_series) < 2:
            return EnergyDynamicsResult(
                energy_total=energy_series.iloc[current_idx] if len(energy_series) > 0 else 0,
                energy_ma5=None,
                energy_ma20=None,
                energy_acceleration=None,
                energy_zscore=None,
                energy_trend='stable',
                metrics={}
            )

        # Current energy
        current_energy = energy_series.iloc[current_idx]

        # Moving averages
        ma5 = energy_series.rolling(window=self.ma_short, min_periods=1).mean()
        ma20 = energy_series.rolling(window=self.ma_long, min_periods=1).mean()

        energy_ma5 = ma5.iloc[current_idx] if len(ma5) > abs(current_idx) else None
        energy_ma20 = ma20.iloc[current_idx] if len(ma20) > abs(current_idx) else None

        # Acceleration (rate of change in energy)
        if len(energy_series) >= 3:
            delta1 = energy_series.diff()
            delta2 = delta1.diff()  # Second derivative
            energy_acceleration = delta2.iloc[current_idx] if not pd.isna(delta2.iloc[current_idx]) else 0.0
        else:
            energy_acceleration = 0.0

        # Z-score (current vs historical distribution)
        if len(energy_series) >= self.zscore_window:
            window_data = energy_series.iloc[max(0, current_idx - self.zscore_window + 1):current_idx + 1]
            mean_e = window_data.mean()
            std_e = window_data.std()
            if std_e > 0:
                energy_zscore = (current_energy - mean_e) / std_e
            else:
                energy_zscore = 0.0
        elif len(energy_series) >= 5:
            mean_e = energy_series.mean()
            std_e = energy_series.std()
            if std_e > 0:
                energy_zscore = (current_energy - mean_e) / std_e
            else:
                energy_zscore = 0.0
        else:
            energy_zscore = 0.0

        # Determine trend
        if energy_ma5 and energy_ma20:
            if energy_ma5 > energy_ma20 * 1.1:
                energy_trend = 'rising'
            elif energy_ma5 < energy_ma20 * 0.9:
                energy_trend = 'falling'
            else:
                energy_trend = 'stable'
        else:
            energy_trend = 'stable'

        # Additional metrics
        metrics = {
            'energy_volatility': energy_series.std() if len(energy_series) >= 5 else 0.0,
            'energy_min': energy_series.min(),
            'energy_max': energy_series.max(),
            'energy_range': energy_series.max() - energy_series.min(),
            'energy_percentile': (energy_series < current_energy).mean() * 100 if len(energy_series) > 1 else 50.0
        }

        return EnergyDynamicsResult(
            energy_total=float(current_energy),
            energy_ma5=float(energy_ma5) if energy_ma5 is not None else None,
            energy_ma20=float(energy_ma20) if energy_ma20 is not None else None,
            energy_acceleration=float(energy_acceleration),
            energy_zscore=float(energy_zscore),
            energy_trend=energy_trend,
            metrics=metrics
        )

    def compute_series(self, energy_series: pd.Series) -> pd.DataFrame:
        """
        Compute energy dynamics for entire series.

        Returns DataFrame with dynamics metrics for each time point.
        """
        results = []

        for i in range(len(energy_series)):
            subset = energy_series.iloc[:i+1]
            result = self.run(subset, current_idx=-1)

            results.append({
                'date': energy_series.index[i],
                'energy_total': result.energy_total,
                'energy_ma5': result.energy_ma5,
                'energy_ma20': result.energy_ma20,
                'energy_acceleration': result.energy_acceleration,
                'energy_zscore': result.energy_zscore,
                'energy_trend': result.energy_trend
            })

        return pd.DataFrame(results)


def compute_energy_dynamics(
    energy_values: np.ndarray,
    ma_short: int = None,
    ma_long: int = None,
    zscore_window: int = None
) -> Dict[str, Any]:
    """
    Functional interface for energy dynamics computation.

    Args:
        energy_values: Array of energy values
        ma_short: Short MA window. If None, loads from config.
        ma_long: Long MA window. If None, loads from config.
        zscore_window: Z-score window. If None, loads from config.

    Returns:
        Dict of energy dynamics metrics
    """
    engine = EnergyDynamicsEngine(ma_short, ma_long, zscore_window)
    series = pd.Series(energy_values)
    result = engine.run(series)

    return {
        'energy_total': result.energy_total,
        'energy_ma5': result.energy_ma5,
        'energy_ma20': result.energy_ma20,
        'energy_acceleration': result.energy_acceleration,
        'energy_zscore': result.energy_zscore,
        'energy_trend': result.energy_trend,
        **result.metrics
    }
