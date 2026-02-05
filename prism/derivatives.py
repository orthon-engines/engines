"""
PRISM Numerical Derivatives Primitives

Pure mathematical functions for computing numerical derivatives
and differential operations on signals.
"""

import numpy as np
from typing import Optional


def first_derivative(
    values: np.ndarray,
    dt: float = 1.0,
    method: str = 'central'
) -> np.ndarray:
    """
    Compute first derivative (rate of change).

    Args:
        values: Input time series
        dt: Time step
        method: 'forward', 'backward', or 'central' difference

    Returns:
        First derivative array
    """
    values = np.asarray(values, dtype=np.float64)

    if method == 'forward':
        # Forward difference: (f[i+1] - f[i]) / dt
        deriv = np.diff(values) / dt
        # Pad to same length
        deriv = np.append(deriv, deriv[-1])

    elif method == 'backward':
        # Backward difference: (f[i] - f[i-1]) / dt
        deriv = np.diff(values) / dt
        # Pad to same length
        deriv = np.insert(deriv, 0, deriv[0])

    elif method == 'central':
        # Central difference: (f[i+1] - f[i-1]) / (2*dt)
        deriv = np.gradient(values, dt)

    else:
        raise ValueError(f"Unknown method: {method}")

    return deriv


def second_derivative(
    values: np.ndarray,
    dt: float = 1.0,
    method: str = 'central'
) -> np.ndarray:
    """
    Compute second derivative (acceleration/curvature).

    Args:
        values: Input time series
        dt: Time step
        method: 'central' or 'finite_difference'

    Returns:
        Second derivative array
    """
    values = np.asarray(values, dtype=np.float64)

    if method == 'central':
        # Second derivative via gradient of gradient
        deriv = np.gradient(np.gradient(values, dt), dt)

    elif method == 'finite_difference':
        # (f[i+1] - 2*f[i] + f[i-1]) / dt^2
        n = len(values)
        deriv = np.zeros(n)
        for i in range(1, n - 1):
            deriv[i] = (values[i + 1] - 2 * values[i] + values[i - 1]) / (dt ** 2)
        deriv[0] = deriv[1]
        deriv[-1] = deriv[-2]

    else:
        raise ValueError(f"Unknown method: {method}")

    return deriv


def gradient(
    values: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute gradient (equivalent to first derivative for 1D signals).

    Uses numpy's gradient function which handles edges appropriately.

    Args:
        values: Input array (can be multi-dimensional)
        dt: Spacing

    Returns:
        Gradient array
    """
    values = np.asarray(values, dtype=np.float64)
    return np.gradient(values, dt)


def laplacian(
    values: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute Laplacian (second spatial derivative).

    For 1D, this is equivalent to second derivative.
    For 2D, it's d²f/dx² + d²f/dy².

    Args:
        values: Input array
        dt: Spacing

    Returns:
        Laplacian array
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim == 1:
        return second_derivative(values, dt)

    elif values.ndim == 2:
        # 2D Laplacian
        lap_x = np.zeros_like(values)
        lap_y = np.zeros_like(values)

        # d²f/dx²
        lap_x[:, 1:-1] = (values[:, 2:] - 2 * values[:, 1:-1] + values[:, :-2]) / (dt ** 2)
        lap_x[:, 0] = lap_x[:, 1]
        lap_x[:, -1] = lap_x[:, -2]

        # d²f/dy²
        lap_y[1:-1, :] = (values[2:, :] - 2 * values[1:-1, :] + values[:-2, :]) / (dt ** 2)
        lap_y[0, :] = lap_y[1, :]
        lap_y[-1, :] = lap_y[-2, :]

        return lap_x + lap_y

    else:
        raise ValueError(f"Laplacian not implemented for {values.ndim}D arrays")


def finite_difference(
    values: np.ndarray,
    order: int = 1,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute finite difference of specified order.

    Args:
        values: Input array
        order: Derivative order (1, 2, 3, ...)
        dt: Time step

    Returns:
        Finite difference array
    """
    values = np.asarray(values, dtype=np.float64)
    result = values.copy()

    for _ in range(order):
        result = np.diff(result) / dt
        # Pad to maintain length
        result = np.append(result, result[-1])

    return result


def velocity(
    values: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute velocity (first time derivative).

    Alias for first_derivative with more intuitive naming.

    Args:
        values: Position time series
        dt: Time step

    Returns:
        Velocity array
    """
    return first_derivative(values, dt, method='central')


def acceleration(
    values: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute acceleration (second time derivative).

    Alias for second_derivative with more intuitive naming.

    Args:
        values: Position time series
        dt: Time step

    Returns:
        Acceleration array
    """
    return second_derivative(values, dt, method='central')


def jerk(
    values: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute jerk (third time derivative).

    Jerk is the rate of change of acceleration.

    Args:
        values: Position time series
        dt: Time step

    Returns:
        Jerk array
    """
    values = np.asarray(values, dtype=np.float64)
    return np.gradient(np.gradient(np.gradient(values, dt), dt), dt)


def curvature(
    x: np.ndarray,
    y: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Compute curvature of a 2D trajectory.

    κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)

    Args:
        x: X coordinates
        y: Y coordinates
        dt: Time step

    Returns:
        Curvature array
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # First derivatives
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)

    # Second derivatives
    d2x = np.gradient(dx, dt)
    d2y = np.gradient(dy, dt)

    # Curvature formula
    numerator = np.abs(dx * d2y - dy * d2x)
    denominator = (dx ** 2 + dy ** 2) ** 1.5

    # Avoid division by zero
    curvature = np.where(denominator > 1e-12, numerator / denominator, 0)

    return curvature


def smoothed_derivative(
    values: np.ndarray,
    dt: float = 1.0,
    window: int = 5,
    order: int = 1
) -> np.ndarray:
    """
    Compute smoothed derivative using Savitzky-Golay filter.

    Args:
        values: Input time series
        dt: Time step
        window: Window length for smoothing (must be odd)
        order: Derivative order

    Returns:
        Smoothed derivative
    """
    from scipy.signal import savgol_filter

    values = np.asarray(values, dtype=np.float64)

    if window % 2 == 0:
        window += 1  # Must be odd

    # Polynomial order for Savitzky-Golay (must be less than window)
    polyorder = min(3, window - 1)

    deriv = savgol_filter(values, window, polyorder, deriv=order, delta=dt)

    return deriv


def integral(
    values: np.ndarray,
    dt: float = 1.0,
    method: str = 'trapezoid',
    initial: float = 0.0
) -> np.ndarray:
    """
    Compute cumulative integral (antiderivative).

    Args:
        values: Input time series (derivative values)
        dt: Time step
        method: 'trapezoid' or 'simpson'
        initial: Initial value

    Returns:
        Cumulative integral array
    """
    from scipy import integrate

    values = np.asarray(values, dtype=np.float64)

    if method == 'trapezoid':
        result = integrate.cumulative_trapezoid(values, dx=dt, initial=0)
    elif method == 'simpson':
        # Simpson's rule requires cumulative computation
        result = np.zeros(len(values))
        for i in range(1, len(values)):
            result[i] = integrate.simpson(values[:i + 1], dx=dt)
    else:
        raise ValueError(f"Unknown method: {method}")

    return result + initial
