"""
PRISM Laplace Module
====================

Unified module for Laplace operations:

1. DISCRETE OPERATORS (gradient, laplacian)
   - compute_gradient: ∇f - first derivative
   - compute_laplacian: ∇²f - second derivative
   - compute_laplace_for_series: Apply to metric series
   - compute_divergence_for_signal: Sum laplacians across metrics

2. RUNNING LAPLACE TRANSFORM
   - RunningLaplace: O(1) incremental transform F(s,t) = ∫f(τ)e^(-sτ)dτ
   - compute_laplace_field: Batch computation
   - Derived quantities: gradient, divergence, energy

Mathematical Foundation:
------------------------
Discrete Operators:
  GRADIENT: ∇V(t) = (V(t+1) - V(t-1)) / 2
  LAPLACIAN: ∇²V(t) = V(t+1) - 2V(t) + V(t-1)
  DIVERGENCE: Σ ∇²V across all metrics (SOURCE > 0, SINK < 0)

Laplace Transform:
  F(s, t) = ∫₀ᵗ f(τ) e^(-sτ) dτ
  O(1) update: F(s, t+Δt) = F(s, t) + f(t+Δt) × e^(-s×(t+Δt)) × Δt
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from prism.core.signals.types import LaplaceField, DenseSignal


# =============================================================================
# DISCRETE OPERATORS (gradient, laplacian, divergence)
# =============================================================================

def compute_gradient(values: np.ndarray) -> np.ndarray:
    """
    Compute first derivative (gradient) with consistent accuracy at boundaries.

    Interior: Central difference O(h²)
        ∇f(t) = (f(t+1) - f(t-1)) / 2

    Boundaries: One-sided second-order O(h²)
        ∇f(0) = (-3f(0) + 4f(1) - f(2)) / 2
        ∇f(n) = (3f(n) - 4f(n-1) + f(n-2)) / 2

    Args:
        values: 1D array of values

    Returns:
        Array of gradients (same length as input)
    """
    n = len(values)
    gradient = np.full(n, np.nan)

    if n < 2:
        return gradient

    # Interior: central difference O(h²)
    if n >= 3:
        gradient[1:-1] = (values[2:] - values[:-2]) / 2.0

    # Boundaries: one-sided second-order O(h²)
    if n >= 3:
        gradient[0] = (-3*values[0] + 4*values[1] - values[2]) / 2.0
        gradient[-1] = (3*values[-1] - 4*values[-2] + values[-3]) / 2.0
    elif n == 2:
        gradient[0] = (values[1] - values[0])
        gradient[-1] = (values[-1] - values[-2])

    return gradient


def compute_laplacian(values: np.ndarray) -> np.ndarray:
    """
    Compute second derivative (Laplacian) with consistent accuracy at boundaries.

    Interior: Central difference O(h²)
        ∇²f(t) = f(t+1) - 2f(t) + f(t-1)

    Boundaries: One-sided second-order O(h²) (requires n >= 4)
        ∇²f(0) = 2f(0) - 5f(1) + 4f(2) - f(3)
        ∇²f(n) = 2f(n) - 5f(n-1) + 4f(n-2) - f(n-3)

    Args:
        values: 1D array of values

    Returns:
        Array of laplacians (same length as input)
    """
    n = len(values)
    laplacian = np.full(n, np.nan)

    if n < 3:
        return laplacian

    # Interior: central difference O(h²)
    laplacian[1:-1] = values[2:] - 2 * values[1:-1] + values[:-2]

    # Boundaries: one-sided second-order O(h²) (requires 4+ points)
    if n >= 4:
        laplacian[0] = 2*values[0] - 5*values[1] + 4*values[2] - values[3]
        laplacian[-1] = 2*values[-1] - 5*values[-2] + 4*values[-3] - values[-4]

    return laplacian


def compute_laplace_for_series(
    signal_id: str,
    dates: List[datetime],
    values: np.ndarray,
    engine: str,
    metric_name: str,
) -> List[Dict[str, Any]]:
    """
    Compute Laplace field quantities for a single metric series.

    Args:
        signal_id: The signal identifier
        dates: List of window_end dates (sorted ascending)
        values: Array of metric values corresponding to dates
        engine: Engine name (e.g., 'hurst')
        metric_name: Metric name (e.g., 'hurst_exponent')

    Returns:
        List of dicts with field quantities per date
    """
    n = len(values)
    if n < 3:
        return []

    gradient = compute_gradient(values)
    laplacian = compute_laplacian(values)

    results = []
    for i in range(n):
        if np.isnan(gradient[i]) and np.isnan(laplacian[i]):
            continue

        row = {
            'signal_id': signal_id,
            'window_end': dates[i],
            'engine': engine,
            'metric_name': metric_name,
            'metric_value': float(values[i]) if not np.isnan(values[i]) else None,
            'gradient': float(gradient[i]) if not np.isnan(gradient[i]) else None,
            'laplacian': float(laplacian[i]) if not np.isnan(laplacian[i]) else None,
            'gradient_magnitude': abs(float(gradient[i])) if not np.isnan(gradient[i]) else None,
        }
        results.append(row)

    return results


def compute_divergence_for_signal(
    field_rows: List[Dict[str, Any]],
) -> Dict[datetime, Dict[str, float]]:
    """
    Compute divergence (sum of laplacians) per window for a signal.

    Divergence = Σ ∇²V across all metrics at time t
    - Positive = SOURCE (energy injection)
    - Negative = SINK (energy absorption)
    """
    by_window = defaultdict(list)
    for row in field_rows:
        by_window[row['window_end']].append(row)

    results = {}
    for window_end, rows in by_window.items():
        laplacians = [r['laplacian'] for r in rows if r['laplacian'] is not None]
        grad_mags = [r['gradient_magnitude'] for r in rows if r['gradient_magnitude'] is not None]

        results[window_end] = {
            'divergence': sum(laplacians) if laplacians else 0.0,
            'total_gradient_mag': sum(grad_mags) if grad_mags else 0.0,
            'mean_gradient_mag': np.mean(grad_mags) if grad_mags else 0.0,
            'n_metrics': len(rows),
        }

    return results


def add_divergence_to_field_rows(
    field_rows: List[Dict[str, Any]],
    divergence_by_window: Dict[datetime, Dict[str, float]],
    source_threshold: float = 0.1,
    sink_threshold: float = -0.1,
) -> List[Dict[str, Any]]:
    """
    Add divergence and source/sink flags to field rows.
    """
    for row in field_rows:
        window_end = row['window_end']
        div_info = divergence_by_window.get(window_end, {})

        row['divergence'] = div_info.get('divergence')
        row['total_gradient_mag'] = div_info.get('total_gradient_mag')
        row['mean_gradient_mag'] = div_info.get('mean_gradient_mag')
        row['n_metrics'] = div_info.get('n_metrics')
        row['is_source'] = row['divergence'] > source_threshold
        row['is_sink'] = row['divergence'] < sink_threshold

    return field_rows


# =============================================================================
# RUNNING LAPLACE TRANSFORM
# =============================================================================

@dataclass
class RunningLaplace:
    """
    Incremental Laplace transform with O(1) update per observation.

    The Laplace transform captures frequency content while preserving
    causality - only past observations contribute to F(s, t).

    Usage:
        laplace = RunningLaplace(s_values=[0.01, 0.1, 1.0, 10.0])
        for t, value in observations:
            laplace.update(t, value)
        field = laplace.get_field()

    Parameters:
        s_values: Complex frequency values to compute
                  Real part = decay rate, Imaginary part = oscillation
        normalize: Whether to normalize by time span (default True)
    """
    s_values: np.ndarray = field(default_factory=lambda: np.array([
        0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0
    ]))
    normalize: bool = True

    # Internal state (initialized on first update)
    _signal_id: str = ""
    _timestamps: List[datetime] = field(default_factory=list)
    _values: List[float] = field(default_factory=list)
    _field_history: List[np.ndarray] = field(default_factory=list)
    _current_field: Optional[np.ndarray] = None
    _t0: Optional[float] = None
    _initialized: bool = False

    def __post_init__(self):
        self.s_values = np.asarray(self.s_values, dtype=np.complex128)
        if len(self.s_values) == 0:
            raise ValueError("s_values must not be empty")

    def reset(self, signal_id: str = "") -> None:
        """Reset state for new signal."""
        self._signal_id = signal_id
        self._timestamps = []
        self._values = []
        self._field_history = []
        self._current_field = np.zeros(len(self.s_values), dtype=np.complex128)
        self._t0 = None
        self._initialized = False

    def _to_numeric_time(self, t: Union[datetime, float, int]) -> float:
        """Convert timestamp to numeric value."""
        if isinstance(t, datetime):
            return t.timestamp()
        return float(t)

    def update(self, t: Union[datetime, float, int], value: float) -> np.ndarray:
        """
        Update Laplace transform with new observation.

        F(s, t+Δt) = F(s, t) + f(t+Δt) × e^(-s×(t+Δt)) × Δt

        Returns the current field values (one per s value).
        """
        t_numeric = self._to_numeric_time(t)

        if not self._initialized:
            self._current_field = np.zeros(len(self.s_values), dtype=np.complex128)
            self._t0 = t_numeric
            self._initialized = True

        # Compute relative time from start
        tau = t_numeric - self._t0

        # Compute Δt
        if len(self._timestamps) > 0:
            t_prev = self._to_numeric_time(self._timestamps[-1])
            dt = t_numeric - t_prev
        else:
            dt = 1.0  # First observation, assume unit time step

        # Update: F(s, t) += value × e^(-s×tau) × dt
        exponential = np.exp(-self.s_values * tau)
        self._current_field += value * exponential * dt

        # Store history
        self._timestamps.append(t)
        self._values.append(value)
        self._field_history.append(self._current_field.copy())

        return self._current_field.copy()

    def get_current(self) -> np.ndarray:
        """Get current Laplace field values."""
        if self._current_field is None:
            return np.zeros(len(self.s_values), dtype=np.complex128)
        return self._current_field.copy()

    def get_field(self) -> LaplaceField:
        """
        Get complete LaplaceField structure.

        Returns 2D field [n_timestamps × n_s] with all history.
        """
        if len(self._timestamps) == 0:
            return LaplaceField(
                signal_id=self._signal_id,
                timestamps=np.array([]),
                s_values=self.s_values,
                field=np.zeros((0, len(self.s_values)), dtype=np.complex128),
            )

        # Stack history into 2D array [n_t × n_s]
        field_array = np.row_stack(self._field_history)

        # Normalize if requested
        if self.normalize and len(self._timestamps) > 1:
            t_span = self._to_numeric_time(self._timestamps[-1]) - self._to_numeric_time(self._timestamps[0])
            if t_span > 0:
                field_array = field_array / t_span

        return LaplaceField(
            signal_id=self._signal_id,
            timestamps=np.array(self._timestamps),
            s_values=self.s_values,
            field=field_array,
        )

    def get_magnitude_at(self, t: Union[datetime, float, int]) -> np.ndarray:
        """Get magnitude |F(s,t)| at specific timestamp."""
        idx = None
        for i, ts in enumerate(self._timestamps):
            if self._to_numeric_time(ts) == self._to_numeric_time(t):
                idx = i
                break

        if idx is None:
            return np.full(len(self.s_values), np.nan)

        return np.abs(self._field_history[idx])

    def get_dominant_frequency_at(self, t: Union[datetime, float, int]) -> float:
        """Get dominant frequency (s with max magnitude) at timestamp."""
        magnitudes = self.get_magnitude_at(t)
        if np.all(np.isnan(magnitudes)):
            return np.nan
        idx = np.argmax(magnitudes)
        return float(np.real(self.s_values[idx]))


# =============================================================================
# BATCH COMPUTATION
# =============================================================================

def compute_laplace_field(
    signal_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    s_values: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> LaplaceField:
    """
    Compute Laplace field for a complete signal.

    This is the batch version - use RunningLaplace for streaming.

    Parameters:
        signal_id: Signal identifier
        timestamps: Array of timestamps
        values: Array of signal values
        s_values: Complex frequency values (default: logarithmic range)
        normalize: Whether to normalize by time span

    Returns:
        LaplaceField with complete 2D structure
    """
    if s_values is None:
        # Default logarithmic range covering multiple scales
        s_values = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])

    laplace = RunningLaplace(s_values=s_values, normalize=normalize)
    laplace.reset(signal_id)

    for t, v in zip(timestamps, values):
        laplace.update(t, v)

    return laplace.get_field()


# =============================================================================
# DERIVED QUANTITIES
# =============================================================================

def laplace_gradient(field: LaplaceField) -> DenseSignal:
    """
    Compute temporal gradient of Laplace field.

    ∂F/∂t represents the "velocity" in Laplace space -
    how fast the frequency content is changing.

    Returns the norm of gradient across all frequencies.
    """
    grad = field.gradient_t()  # [n_t × n_s]
    grad_norm = np.linalg.norm(grad, axis=1)  # Norm across s values

    return DenseSignal(
        signal_id=f"{field.signal_id}_laplace_grad",
        timestamps=field.timestamps,
        values=grad_norm.real,  # Real part of gradient norm
        source_signal=field.signal_id,
        engine='laplace',
        parameters={'derived': 'gradient_norm'},
    )


def laplace_divergence(field: LaplaceField) -> DenseSignal:
    """
    Compute divergence in frequency space.

    Positive divergence = energy dispersing (healthy)
    Negative divergence = energy concentrating (degradation)
    """
    div = field.divergence_at_t()  # Returns array of divergence per timestamp

    return DenseSignal(
        signal_id=f"{field.signal_id}_laplace_div",
        timestamps=field.timestamps,
        values=div.real,  # Real part of divergence
        source_signal=field.signal_id,
        engine='laplace',
        parameters={'derived': 'divergence'},
    )


def laplace_energy(field: LaplaceField) -> DenseSignal:
    """
    Total energy in Laplace field at each timestamp.

    E(t) = Σ_s |F(s,t)|²

    Increasing energy = more pronounced frequency components
    Decreasing energy = signal becoming more uniform
    """
    energy = field.total_energy

    return DenseSignal(
        signal_id=f"{field.signal_id}_laplace_energy",
        timestamps=field.timestamps,
        values=energy,
        source_signal=field.signal_id,
        engine='laplace',
        parameters={'derived': 'total_energy'},
    )


# =============================================================================
# SCALE-SPECIFIC ANALYSIS
# =============================================================================

def decompose_by_scale(
    field: LaplaceField,
    scale_boundaries: Optional[List[float]] = None,
) -> List[DenseSignal]:
    """
    Decompose Laplace field by frequency scales.

    Parameters:
        field: LaplaceField to decompose
        scale_boundaries: Boundaries between scales (default: [0.1, 1.0, 10.0])

    Returns:
        List of DenseSignals, one per scale band
    """
    if scale_boundaries is None:
        scale_boundaries = [0.1, 1.0, 10.0]

    s_real = np.real(field.s_values)
    signals = []

    # Add lower bound
    boundaries = [0.0] + list(scale_boundaries) + [np.inf]

    for i in range(len(boundaries) - 1):
        low, high = boundaries[i], boundaries[i + 1]
        mask = (s_real >= low) & (s_real < high)

        if not np.any(mask):
            continue

        # Extract magnitude for this scale band
        # field.magnitude has shape [n_t × n_s], so sum over masked s values
        band_magnitude = np.sum(field.magnitude[:, mask], axis=1)

        scale_name = f"scale_{low:.2f}_{high:.2f}"
        signals.append(DenseSignal(
            signal_id=f"{field.signal_id}_{scale_name}",
            timestamps=field.timestamps,
            values=band_magnitude,
            source_signal=field.signal_id,
            engine='laplace',
            parameters={'scale_low': low, 'scale_high': high},
        ))

    return signals


def frequency_shift(field: LaplaceField) -> DenseSignal:
    """
    Track shift in dominant frequency over time.

    Frequency shift is an early indicator of regime change -
    the system starts resonating at different frequencies before
    behavior visibly changes.
    """
    dominant = field.dominant_frequency()

    # Compute shift as first difference
    shifts = np.diff(dominant.values, prepend=dominant.values[0])

    return DenseSignal(
        signal_id=f"{field.signal_id}_freq_shift",
        timestamps=field.timestamps,
        values=shifts,
        source_signal=field.signal_id,
        engine='laplace',
        parameters={'derived': 'frequency_shift'},
    )
