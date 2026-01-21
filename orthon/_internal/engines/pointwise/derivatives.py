"""
Derivatives Engine
==================

Point-wise engine for computing temporal derivatives.

Computes:
- Velocity (first derivative)
- Acceleration (second derivative)
- Jerk (third derivative)

These are fundamental for state trajectory computation -
position, velocity, acceleration are the core state components.
"""

import numpy as np
from typing import Tuple, Optional, Literal

from orthon._internal.core.signals.types import DenseSignal


# =============================================================================
# DERIVATIVES ENGINE
# =============================================================================

class DerivativesEngine:
    """
    Point-wise derivatives engine.

    Computes velocity, acceleration, and jerk from raw observations.

    Usage:
        engine = DerivativesEngine()
        velocity, acceleration, jerk = engine.compute(timestamps, values)
    """

    def __init__(
        self,
        method: Literal['gradient', 'finite_diff', 'savgol'] = 'gradient',
        savgol_window: int = 5,
        savgol_order: int = 2,
    ):
        """
        Initialize derivatives engine.

        Parameters:
            method: Differentiation method
                - 'gradient': NumPy gradient (central differences)
                - 'finite_diff': Simple finite differences
                - 'savgol': Savitzky-Golay filter (smoothed derivatives)
            savgol_window: Window size for Savitzky-Golay
            savgol_order: Polynomial order for Savitzky-Golay
        """
        self.method = method
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order

    def _compute_derivative(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        order: int = 1,
    ) -> np.ndarray:
        """Compute derivative using configured method."""
        # Get time differences
        if len(timestamps) > 1:
            dt = np.diff(timestamps.astype(np.float64))
            dt = np.concatenate([[dt[0]], dt])  # Pad first element
        else:
            dt = np.ones_like(values)

        current = values.copy()

        for _ in range(order):
            if self.method == 'gradient':
                current = np.gradient(current) / (dt + 1e-10)
            elif self.method == 'finite_diff':
                diff = np.diff(current, prepend=current[0])
                current = diff / (dt + 1e-10)
            elif self.method == 'savgol':
                from scipy.signal import savgol_filter
                window = min(self.savgol_window, len(current))
                if window % 2 == 0:
                    window -= 1
                if window < 3:
                    current = np.gradient(current) / (dt + 1e-10)
                else:
                    current = savgol_filter(
                        current,
                        window,
                        min(self.savgol_order, window - 1),
                        deriv=1,
                    ) / (dt + 1e-10)

        return current

    def compute(
        self,
        signal_id: str,
        timestamps: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[DenseSignal, DenseSignal, DenseSignal]:
        """
        Compute velocity, acceleration, and jerk.

        Returns:
            (velocity, acceleration, jerk) as DenseSignal objects
        """
        velocity_values = self._compute_derivative(values, timestamps, order=1)
        acceleration_values = self._compute_derivative(values, timestamps, order=2)
        jerk_values = self._compute_derivative(values, timestamps, order=3)

        velocity = DenseSignal(
            signal_id=f"{signal_id}_velocity",
            timestamps=timestamps,
            values=velocity_values,
            source_signal=signal_id,
            engine='derivatives',
            parameters={'metric': 'velocity', 'method': self.method},
        )

        acceleration = DenseSignal(
            signal_id=f"{signal_id}_acceleration",
            timestamps=timestamps,
            values=acceleration_values,
            source_signal=signal_id,
            engine='derivatives',
            parameters={'metric': 'acceleration', 'method': self.method},
        )

        jerk = DenseSignal(
            signal_id=f"{signal_id}_jerk",
            timestamps=timestamps,
            values=jerk_values,
            source_signal=signal_id,
            engine='derivatives',
            parameters={'metric': 'jerk', 'method': self.method},
        )

        return velocity, acceleration, jerk


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_velocity(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    method: str = 'gradient',
) -> DenseSignal:
    """Compute first derivative (velocity)."""
    if len(timestamps) > 1:
        dt = np.diff(timestamps.astype(np.float64))
        dt = np.concatenate([[dt[0]], dt])
    else:
        dt = np.ones_like(values)

    velocity_values = np.gradient(values) / (dt + 1e-10)

    return DenseSignal(
        signal_id=f"{signal_id}_velocity",
        timestamps=timestamps,
        values=velocity_values,
        source_signal=signal_id,
        engine='derivatives',
        parameters={'metric': 'velocity', 'method': method},
    )


def compute_acceleration(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    method: str = 'gradient',
) -> DenseSignal:
    """Compute second derivative (acceleration)."""
    if len(timestamps) > 1:
        dt = np.diff(timestamps.astype(np.float64))
        dt = np.concatenate([[dt[0]], dt])
    else:
        dt = np.ones_like(values)

    velocity = np.gradient(values) / (dt + 1e-10)
    acceleration_values = np.gradient(velocity) / (dt + 1e-10)

    return DenseSignal(
        signal_id=f"{signal_id}_acceleration",
        timestamps=timestamps,
        values=acceleration_values,
        source_signal=signal_id,
        engine='derivatives',
        parameters={'metric': 'acceleration', 'method': method},
    )


def compute_jerk(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    method: str = 'gradient',
) -> DenseSignal:
    """Compute third derivative (jerk)."""
    if len(timestamps) > 1:
        dt = np.diff(timestamps.astype(np.float64))
        dt = np.concatenate([[dt[0]], dt])
    else:
        dt = np.ones_like(values)

    velocity = np.gradient(values) / (dt + 1e-10)
    acceleration = np.gradient(velocity) / (dt + 1e-10)
    jerk_values = np.gradient(acceleration) / (dt + 1e-10)

    return DenseSignal(
        signal_id=f"{signal_id}_jerk",
        timestamps=timestamps,
        values=jerk_values,
        source_signal=signal_id,
        engine='derivatives',
        parameters={'metric': 'jerk', 'method': method},
    )


# =============================================================================
# HIGHER-ORDER DERIVATIVES
# =============================================================================

def compute_nth_derivative(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    n: int = 1,
    method: str = 'gradient',
) -> DenseSignal:
    """
    Compute nth derivative.

    Parameters:
        n: Order of derivative (1=velocity, 2=acceleration, etc.)
    """
    if len(timestamps) > 1:
        dt = np.diff(timestamps.astype(np.float64))
        dt = np.concatenate([[dt[0]], dt])
    else:
        dt = np.ones_like(values)

    result = values.copy()
    for _ in range(n):
        result = np.gradient(result) / (dt + 1e-10)

    derivative_names = {
        1: 'velocity',
        2: 'acceleration',
        3: 'jerk',
        4: 'snap',
        5: 'crackle',
        6: 'pop',
    }
    metric_name = derivative_names.get(n, f'derivative_{n}')

    return DenseSignal(
        signal_id=f"{signal_id}_{metric_name}",
        timestamps=timestamps,
        values=result,
        source_signal=signal_id,
        engine='derivatives',
        parameters={'metric': metric_name, 'order': n, 'method': method},
    )


# =============================================================================
# CURVATURE
# =============================================================================

def compute_curvature(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
) -> DenseSignal:
    """
    Compute path curvature.

    Curvature κ = |y''| / (1 + y'²)^(3/2)

    High curvature indicates sharp changes in the signal.
    """
    if len(timestamps) > 1:
        dt = np.diff(timestamps.astype(np.float64))
        dt = np.concatenate([[dt[0]], dt])
    else:
        dt = np.ones_like(values)

    y_prime = np.gradient(values) / (dt + 1e-10)
    y_double_prime = np.gradient(y_prime) / (dt + 1e-10)

    # Curvature formula
    curvature = np.abs(y_double_prime) / (1 + y_prime ** 2) ** 1.5

    return DenseSignal(
        signal_id=f"{signal_id}_curvature",
        timestamps=timestamps,
        values=curvature,
        source_signal=signal_id,
        engine='derivatives',
        parameters={'metric': 'curvature'},
    )
