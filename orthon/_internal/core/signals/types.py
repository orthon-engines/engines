"""
PRISM Signal Types
==================

Core signal types for the PRISM architecture:
- DenseSignal: Values at every timestamp (native resolution)
- SparseSignal: Values at subset of timestamps (windowed engines)
- LaplaceField: 2D structure (time × frequency) from running Laplace

These types form the foundation for the point-wise vs windowed engine split.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
import numpy as np
from datetime import datetime


# =============================================================================
# DENSE SIGNAL - Native Resolution
# =============================================================================

@dataclass
class DenseSignal:
    """
    Signal with value at every timestamp.

    Produced by point-wise engines (Hilbert, derivatives).
    Full temporal resolution preserved.
    """
    signal_id: str
    timestamps: np.ndarray          # Native timestamps
    values: np.ndarray              # Value at each timestamp
    source_signal: str              # Parent signal ID
    engine: str                     # Which engine produced this
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.timestamps) != len(self.values):
            raise ValueError(
                f"Timestamp length ({len(self.timestamps)}) must match "
                f"values length ({len(self.values)})"
            )

    def __len__(self) -> int:
        return len(self.values)

    @property
    def resolution(self) -> float:
        """Median time between samples."""
        if len(self.timestamps) < 2:
            return np.inf
        return float(np.median(np.diff(self.timestamps.astype(float))))

    @property
    def is_valid(self) -> bool:
        """Check if signal has valid data."""
        return len(self.values) > 0 and not np.all(np.isnan(self.values))

    def at(self, t: float) -> float:
        """Value at time t (exact match or interpolate)."""
        idx = np.searchsorted(self.timestamps.astype(float), t)
        if idx == 0:
            return float(self.values[0])
        if idx >= len(self.values):
            return float(self.values[-1])
        # Linear interpolation (only for query, not stored)
        t0, t1 = float(self.timestamps[idx-1]), float(self.timestamps[idx])
        v0, v1 = float(self.values[idx-1]), float(self.values[idx])
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0 + 1e-10)

    def up_to(self, t: float) -> tuple:
        """All data up to time t (for running Laplace)."""
        mask = self.timestamps.astype(float) <= t
        return self.timestamps[mask], self.values[mask]

    def slice(self, start: float, end: float) -> 'DenseSignal':
        """Get signal slice between timestamps."""
        ts_float = self.timestamps.astype(float)
        mask = (ts_float >= start) & (ts_float <= end)
        return DenseSignal(
            signal_id=self.signal_id,
            timestamps=self.timestamps[mask],
            values=self.values[mask],
            source_signal=self.source_signal,
            engine=self.engine,
            parameters=self.parameters,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signal_id': self.signal_id,
            'timestamps': self.timestamps.tolist(),
            'values': self.values.tolist(),
            'source_signal': self.source_signal,
            'engine': self.engine,
            'parameters': self.parameters,
        }


# =============================================================================
# SPARSE SIGNAL - Windowed Resolution
# =============================================================================

@dataclass
class SparseSignal:
    """
    Signal with value at window midpoints only.

    Produced by windowed engines (entropy, hurst).
    Fundamentally cannot be faster than window size.
    """
    signal_id: str
    timestamps: np.ndarray          # Window midpoint times
    values: np.ndarray              # Metric value per window
    source_signal: str              # Parent signal ID
    engine: str                     # Which engine produced this
    window_size: int                # Samples per window
    window_duration: float          # Time span per window
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.timestamps) != len(self.values):
            raise ValueError(
                f"Timestamp length ({len(self.timestamps)}) must match "
                f"values length ({len(self.values)})"
            )

    def __len__(self) -> int:
        return len(self.values)

    @property
    def resolution(self) -> float:
        """This signal's native resolution (window duration)."""
        return self.window_duration

    @property
    def is_valid(self) -> bool:
        """Check if signal has valid data."""
        return len(self.values) > 0 and not np.all(np.isnan(self.values))

    def at(self, t: float) -> Optional[float]:
        """
        Most recent value at or before time t.

        NO INTERPOLATION — that would fabricate data.
        Returns None if t is before first window.
        """
        ts_float = self.timestamps.astype(float)
        mask = ts_float <= t
        if not mask.any():
            return None
        return float(self.values[mask][-1])

    def up_to(self, t: float) -> tuple:
        """All completed windows up to time t."""
        ts_float = self.timestamps.astype(float)
        mask = ts_float <= t
        return self.timestamps[mask], self.values[mask]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signal_id': self.signal_id,
            'timestamps': self.timestamps.tolist(),
            'values': self.values.tolist(),
            'window_size': self.window_size,
            'window_duration': self.window_duration,
            'source_signal': self.source_signal,
            'engine': self.engine,
            'parameters': self.parameters,
        }


# =============================================================================
# LAPLACE FIELD - 2D Time × Frequency Structure
# =============================================================================

@dataclass
class LaplaceField:
    """
    Running Laplace transform F(s, t) for a signal.

    2D structure: time × frequency
    Updated incrementally as new observations arrive.
    """
    signal_id: str
    timestamps: np.ndarray          # Times where F(s) was computed
    s_values: np.ndarray            # Laplace s-values (frequency axis)
    field: np.ndarray               # Shape: [len(timestamps), len(s_values)]
    source_type: Literal['dense', 'sparse'] = 'dense'

    def __post_init__(self):
        expected_shape = (len(self.timestamps), len(self.s_values))
        if self.field.shape != expected_shape:
            raise ValueError(
                f"Field shape {self.field.shape} doesn't match expected "
                f"{expected_shape} from timestamps × s_values"
            )

    def at(self, t: float) -> np.ndarray:
        """F(s) vector at time t."""
        ts_float = self.timestamps.astype(float)
        idx = np.searchsorted(ts_float, t)
        idx = min(idx, len(self.timestamps) - 1)
        return self.field[idx]

    def power_spectrum(self, t: float) -> np.ndarray:
        """Power at each s-value at time t."""
        F_s = self.at(t)
        return np.abs(F_s) ** 2

    @property
    def frequency_range(self) -> tuple:
        """Range of frequencies this field covers meaningfully."""
        # For sparse signals, high-freq content is minimal
        return (float(self.s_values[0]), float(self.s_values[-1]))

    @property
    def magnitude(self) -> np.ndarray:
        """Magnitude |F(s,t)| at each (t, s) point."""
        return np.abs(self.field)

    @property
    def phase(self) -> np.ndarray:
        """Phase arg(F(s,t)) at each (t, s) point."""
        return np.angle(self.field)

    @property
    def total_energy(self) -> np.ndarray:
        """Total energy across all frequencies at each timestamp."""
        return np.sum(self.magnitude ** 2, axis=1)

    def gradient_t(self) -> np.ndarray:
        """
        Temporal gradient ∂F/∂t.

        Represents velocity in Laplace space - how fast the
        frequency content is changing.
        """
        return np.gradient(self.field, axis=0)

    def gradient_s(self) -> np.ndarray:
        """
        Frequency gradient ∂F/∂s.

        Represents how energy is distributed across frequencies.
        """
        return np.gradient(self.field, axis=1)

    def divergence_at_t(self) -> np.ndarray:
        """
        Divergence in s-space at each timestamp: Σ ∂²F/∂s².

        Positive = energy dispersing (healthy)
        Negative = energy concentrating (degradation)
        """
        return np.sum(np.gradient(np.gradient(self.field, axis=1), axis=1), axis=1)

    def dominant_frequency(self) -> DenseSignal:
        """
        Extract dominant frequency at each timestamp.

        Returns DenseSignal of the s-value with maximum magnitude.
        """
        dominant_idx = np.argmax(self.magnitude, axis=1)
        dominant_s = self.s_values[dominant_idx]

        return DenseSignal(
            signal_id=f"{self.signal_id}_dominant_freq",
            timestamps=self.timestamps,
            values=np.real(dominant_s),  # Real part of dominant frequency
            source_signal=self.signal_id,
            engine='laplace',
            parameters={'derived': 'dominant_frequency'},
        )

    def spectral_centroid(self) -> DenseSignal:
        """
        Spectral centroid at each timestamp.

        Weighted average of frequencies by their magnitudes.
        """
        weights = self.magnitude
        total_weight = np.sum(weights, axis=1, keepdims=True) + 1e-10
        centroid = np.sum(
            np.real(self.s_values[np.newaxis, :]) * weights, axis=1
        ) / total_weight.squeeze()

        return DenseSignal(
            signal_id=f"{self.signal_id}_spectral_centroid",
            timestamps=self.timestamps,
            values=centroid,
            source_signal=self.signal_id,
            engine='laplace',
            parameters={'derived': 'spectral_centroid'},
        )


# =============================================================================
# GEOMETRY SNAPSHOT - System State at Time t
# =============================================================================

@dataclass
class GeometrySnapshot:
    """
    System geometry at a single timestamp.
    """
    timestamp: float
    coupling_matrix: np.ndarray     # [n_signals × n_signals]
    divergence: float               # Scalar: source (+) or sink (-)
    mode_labels: np.ndarray         # Cluster label per signal
    mode_coherence: np.ndarray      # Coherence per mode
    signal_ids: List[str]           # Signal ordering

    @property
    def n_signals(self) -> int:
        return len(self.signal_ids)

    @property
    def n_modes(self) -> int:
        """Number of distinct modes."""
        return len(np.unique(self.mode_labels))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'n_signals': self.n_signals,
            'divergence': self.divergence,
            'n_modes': self.n_modes,
            'signal_ids': self.signal_ids,
        }


# =============================================================================
# STATE TRAJECTORY - Position, Velocity, Acceleration Over Time
# =============================================================================

@dataclass
class StateTrajectory:
    """
    System state over time.

    Position, velocity, acceleration computed as direct derivatives.
    """
    timestamps: np.ndarray
    position: np.ndarray            # G(t) - geometry vector at each t
    velocity: np.ndarray            # dG/dt
    acceleration: np.ndarray        # d²G/dt²

    def __post_init__(self):
        n_t = len(self.timestamps)
        if self.position.shape[0] != n_t:
            raise ValueError(f"Position rows ({self.position.shape[0]}) must match timestamps ({n_t})")
        if self.velocity.shape[0] != n_t:
            raise ValueError(f"Velocity rows ({self.velocity.shape[0]}) must match timestamps ({n_t})")
        if self.acceleration.shape[0] != n_t:
            raise ValueError(f"Acceleration rows ({self.acceleration.shape[0]}) must match timestamps ({n_t})")

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def n_dimensions(self) -> int:
        return self.position.shape[1] if len(self.position.shape) > 1 else 1

    @property
    def speed(self) -> np.ndarray:
        """Scalar speed (magnitude of velocity) at each timestamp."""
        if len(self.velocity.shape) == 1:
            return np.abs(self.velocity)
        return np.linalg.norm(self.velocity, axis=1)

    @property
    def acceleration_magnitude(self) -> np.ndarray:
        """Scalar acceleration magnitude at each timestamp."""
        if len(self.acceleration.shape) == 1:
            return np.abs(self.acceleration)
        return np.linalg.norm(self.acceleration, axis=1)

    def failure_risk(self, t: float) -> Dict[str, Any]:
        """Failure indicators at time t."""
        ts_float = self.timestamps.astype(float)
        idx = np.searchsorted(ts_float, t)
        idx = min(idx, len(self.timestamps) - 1)

        v = self.velocity[idx]
        a = self.acceleration[idx]

        v_mag = float(np.linalg.norm(v)) if len(v.shape) > 0 else abs(float(v))
        a_mag = float(np.linalg.norm(a)) if len(a.shape) > 0 else abs(float(a))

        # Acceleration direction: positive dot product means speeding up
        if v_mag > 0 and a_mag > 0:
            if len(v.shape) > 0:
                a_direction = float(np.dot(v, a) / (v_mag * a_mag + 1e-10))
            else:
                a_direction = 1.0 if (v > 0) == (a > 0) else -1.0
        else:
            a_direction = 0.0

        return {
            'velocity_magnitude': v_mag,
            'acceleration_magnitude': a_mag,
            'is_accelerating': a_mag > 0,
            'acceleration_direction': a_direction,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'n_timestamps': len(self.timestamps),
            'n_dimensions': self.n_dimensions,
            'mean_speed': float(np.mean(self.speed)),
            'max_speed': float(np.max(self.speed)),
            'mean_acceleration': float(np.mean(self.acceleration_magnitude)),
            'max_acceleration': float(np.max(self.acceleration_magnitude)),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_dense_signal(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    source_signal: str,
    engine: str,
    **parameters
) -> DenseSignal:
    """Factory function to create DenseSignal with validation."""
    return DenseSignal(
        signal_id=signal_id,
        timestamps=np.asarray(timestamps),
        values=np.asarray(values),
        source_signal=source_signal,
        engine=engine,
        parameters=parameters,
    )


def create_sparse_signal(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    window_size: int,
    window_duration: float,
    source_signal: str,
    engine: str,
    **parameters
) -> SparseSignal:
    """Factory function to create SparseSignal with validation."""
    return SparseSignal(
        signal_id=signal_id,
        timestamps=np.asarray(timestamps),
        values=np.asarray(values),
        source_signal=source_signal,
        engine=engine,
        window_size=window_size,
        window_duration=window_duration,
        parameters=parameters,
    )
