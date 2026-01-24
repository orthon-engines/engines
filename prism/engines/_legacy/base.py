"""
Engine Base Classes
===================

Standard interface for all PRISM engines.

Three computation modes:
    - static: Entire signal → single result
    - windowed: Rolling windows → time series result
    - point: At time t → single result

Usage:
    class MyEngine(WindowedEngine):
        def _compute_static(self, signal, **params):
            # Return dict with metrics
            return {'my_metric': value}

    # Then call:
    result = MyEngine().compute(signal, mode='windowed', window_size=200, step_size=20)
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List


@dataclass
class EngineConfig:
    """Configuration for windowed computation."""
    window_size: int = 200
    step_size: int = 20
    min_samples: int = 30


class BaseEngine(ABC):
    """
    Abstract base class for all engines.

    Subclasses must implement _compute_static().
    """

    # Engine metadata
    name: str = "base_engine"
    description: str = "Base engine class"

    # Whether this engine supports windowing
    supports_windowing: bool = True

    # Minimum samples required
    min_samples: int = 30

    def compute(
        self,
        signal: np.ndarray,
        mode: str = 'static',
        t: Optional[int] = None,
        window_size: int = 200,
        step_size: int = 20,
        **params
    ) -> Dict[str, Any]:
        """
        Compute engine metric(s).

        Args:
            signal: 1D numpy array
            mode:
                'static' - entire signal, returns single values
                'windowed' - rolling windows, returns time series
                'point' - at time t, returns single values
            t: Time index for point mode
            window_size: Window size for windowed/point modes
            step_size: Step between windows for windowed mode
            **params: Engine-specific parameters

        Returns:
            Dict with metric names as keys.
            For mode='windowed', values are arrays.
            For mode='static' or 'point', values are scalars.
        """
        signal = np.asarray(signal).flatten()

        if mode == 'static':
            return self._compute_static(signal, **params)
        elif mode == 'windowed':
            return self._compute_windowed(signal, window_size, step_size, **params)
        elif mode == 'point':
            return self._compute_point(signal, t, window_size, **params)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'static', 'windowed', or 'point'.")

    @abstractmethod
    def _compute_static(self, signal: np.ndarray, **params) -> Dict[str, Any]:
        """Compute on entire signal. Must be implemented by subclass."""
        pass

    def _compute_windowed(
        self,
        signal: np.ndarray,
        window_size: int,
        step_size: int,
        **params
    ) -> Dict[str, Any]:
        """
        Compute over rolling windows.

        Default implementation calls _compute_static on each window.
        Override for more efficient implementations.
        """
        if not self.supports_windowing:
            raise ValueError(f"{self.name} does not support windowed mode")

        n = len(signal)
        if n < window_size:
            raise ValueError(f"Signal length {n} < window_size {window_size}")

        # Collect results from each window
        results_list: List[Dict[str, Any]] = []
        t_values: List[int] = []

        for start in range(0, n - window_size + 1, step_size):
            end = start + window_size
            window = signal[start:end]

            try:
                result = self._compute_static(window, **params)
                results_list.append(result)
                t_values.append(start + window_size // 2)  # Center of window
            except Exception:
                # Skip failed windows, append NaN
                if results_list:
                    nan_result = {k: np.nan for k in results_list[0].keys()}
                    results_list.append(nan_result)
                    t_values.append(start + window_size // 2)

        if not results_list:
            return {'t': np.array([]), 'window_size': window_size, 'step_size': step_size}

        # Combine into arrays
        output = {
            't': np.array(t_values),
            'window_size': window_size,
            'step_size': step_size,
        }

        for key in results_list[0].keys():
            values = [r.get(key, np.nan) for r in results_list]
            output[key] = np.array(values)

        return output

    def _compute_point(
        self,
        signal: np.ndarray,
        t: int,
        window_size: int,
        **params
    ) -> Dict[str, Any]:
        """
        Compute at specific time t.

        Default implementation extracts window around t and calls _compute_static.
        """
        if t is None:
            raise ValueError("t is required for point mode")

        n = len(signal)

        # Center window on t
        half_window = window_size // 2
        start = max(0, t - half_window)
        end = min(n, start + window_size)

        # Adjust start if end hit the boundary
        if end - start < window_size:
            start = max(0, end - window_size)

        window = signal[start:end]

        if len(window) < self.min_samples:
            return self._nan_result(t=t, window_start=start, window_end=end)

        result = self._compute_static(window, **params)
        result['t'] = t
        result['window_start'] = start
        result['window_end'] = end

        return result

    def _nan_result(self, **extras) -> Dict[str, Any]:
        """Return result with NaN values. Override in subclass."""
        result = {'error': 'insufficient_samples'}
        result.update(extras)
        return result


class StaticOnlyEngine(BaseEngine):
    """
    Engine that only supports static mode.

    Use for engines where windowing doesn't make sense
    (e.g., FFT dominant frequency, embedding dimension).
    """
    supports_windowing = False

    def _compute_windowed(self, *args, **kwargs):
        raise ValueError(f"{self.name} only supports static mode (global signal property)")

    def _compute_point(self, *args, **kwargs):
        raise ValueError(f"{self.name} only supports static mode (global signal property)")


# =============================================================================
# Functional Interface
# =============================================================================

def compute_windowed(
    compute_func,
    signal: np.ndarray,
    window_size: int = 200,
    step_size: int = 20,
    min_samples: int = 30,
    **params
) -> Dict[str, np.ndarray]:
    """
    Generic windowed computation wrapper.

    Takes a static compute function and applies it over rolling windows.

    Args:
        compute_func: Function that takes signal and returns dict
        signal: 1D array
        window_size: Size of each window
        step_size: Step between windows
        min_samples: Minimum samples required
        **params: Passed to compute_func

    Returns:
        Dict with arrays (one value per window)
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    if n < window_size:
        raise ValueError(f"Signal length {n} < window_size {window_size}")

    results_list = []
    t_values = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = signal[start:end]

        if len(window) < min_samples:
            continue

        try:
            result = compute_func(window, **params)
            results_list.append(result)
            t_values.append(start + window_size // 2)
        except Exception:
            if results_list:
                nan_result = {k: np.nan for k in results_list[0].keys()}
                results_list.append(nan_result)
                t_values.append(start + window_size // 2)

    if not results_list:
        return {'t': np.array([]), 'window_size': window_size, 'step_size': step_size}

    output = {
        't': np.array(t_values),
        'window_size': window_size,
        'step_size': step_size,
    }

    for key in results_list[0].keys():
        values = [r.get(key, np.nan) for r in results_list]
        output[key] = np.array(values)

    return output


def compute_point(
    compute_func,
    signal: np.ndarray,
    t: int,
    window_size: int = 200,
    min_samples: int = 30,
    **params
) -> Dict[str, Any]:
    """
    Point-in-time computation wrapper.

    Args:
        compute_func: Function that takes signal and returns dict
        signal: 1D array
        t: Time index
        window_size: Window size around t
        min_samples: Minimum samples required
        **params: Passed to compute_func

    Returns:
        Dict with scalar values at time t
    """
    signal = np.asarray(signal).flatten()
    n = len(signal)

    half_window = window_size // 2
    start = max(0, t - half_window)
    end = min(n, start + window_size)

    if end - start < window_size:
        start = max(0, end - window_size)

    window = signal[start:end]

    if len(window) < min_samples:
        return {'error': 'insufficient_samples', 't': t, 'window_start': start, 'window_end': end}

    result = compute_func(window, **params)
    result['t'] = t
    result['window_start'] = start
    result['window_end'] = end

    return result
