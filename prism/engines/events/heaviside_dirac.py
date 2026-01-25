"""
Heaviside and Dirac Event Functions

HYBRID APPROACH:
    1. Detect events (steps, spikes) in time series
    2. Compute REAL H(t - t0) and delta(t - t0) centered at each event
    3. Store for downstream use (convolution, system ID, decomposition)

The functions are computed blind. They become relevant when they do.

Mathematical definitions:

    Heaviside step function:
        H(t - t0) = 0  for t < t0
        H(t - t0) = 1  for t >= t0

    Dirac delta (distribution):
        delta(t - t0) = 0     for t != t0
        integral(delta(t) dt) = 1

    Relationship:
        dH/dt = delta(t)   (derivative of step is impulse)
        integral(delta) = H (integral of impulse is step)

Usage:
    from prism.engines.events import compute, heaviside, dirac_delta_discrete

    # Full decomposition
    result = compute(t, x)

    # Just the mathematical function
    H = heaviside(t, t0=50)  # Step at t=50
    delta = dirac_delta_discrete(t, t0=50, dt=0.01)  # Impulse at t=50

References:
    Lighthill (1958) "Introduction to Fourier Analysis and Generalised Functions"
    Schwartz (1950) "Theorie des distributions"
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import signal as sig
from scipy.ndimage import gaussian_filter1d


@dataclass
class StepEvent:
    """Detected step change."""
    t_index: int          # Index where step occurs
    t_value: float        # Time value (if available)
    magnitude: float      # Step size (after - before)
    direction: str        # 'up' or 'down'


@dataclass
class SpikeEvent:
    """Detected impulse/spike."""
    t_index: int          # Index of spike
    t_value: float        # Time value
    magnitude: float      # Spike height
    width: int            # Approximate width in samples


# =============================================================================
# EVENT DETECTION
# =============================================================================

def detect_steps(
    x: np.ndarray,
    threshold: float = None,
    min_separation: int = 10,
) -> List[StepEvent]:
    """
    Detect step changes (level shifts) in signal.

    Uses derivative + threshold to find abrupt level changes.

    Args:
        x: Signal array
        threshold: Jump threshold (default: 3 * std(diff))
        min_separation: Minimum samples between steps

    Returns:
        List of StepEvent objects
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) < 3:
        return []

    if threshold is None:
        diff_std = np.std(np.diff(x))
        if diff_std < 1e-10:
            return []
        threshold = 3 * diff_std

    # First difference
    dx = np.diff(x)

    # Find large jumps
    step_indices = np.where(np.abs(dx) > threshold)[0]

    if len(step_indices) == 0:
        return []

    # Merge nearby detections
    events = []
    last_idx = -min_separation - 1

    for idx in step_indices:
        if idx - last_idx >= min_separation:
            magnitude = dx[idx]
            events.append(StepEvent(
                t_index=idx,
                t_value=float(idx),
                magnitude=float(magnitude),
                direction='up' if magnitude > 0 else 'down',
            ))
            last_idx = idx

    return events


def detect_spikes(
    x: np.ndarray,
    threshold: float = None,
    prominence: float = None,
) -> List[SpikeEvent]:
    """
    Detect impulse/spike events in signal.

    Uses scipy.signal.find_peaks with prominence filtering.

    Args:
        x: Signal array
        threshold: Height threshold (default: mean + 3*std)
        prominence: Peak prominence (default: 2*std)

    Returns:
        List of SpikeEvent objects
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) < 3:
        return []

    x_std = np.std(x)
    x_mean = np.mean(x)

    if x_std < 1e-10:
        return []

    if threshold is None:
        threshold = x_mean + 3 * x_std

    if prominence is None:
        prominence = 2 * x_std

    events = []

    # Find positive peaks
    try:
        peaks, properties = sig.find_peaks(
            x,
            height=threshold,
            prominence=prominence,
        )

        for idx in peaks:
            width_result = sig.peak_widths(x, [idx])
            width = int(max(1, width_result[0][0])) if len(width_result[0]) > 0 else 1

            events.append(SpikeEvent(
                t_index=int(idx),
                t_value=float(idx),
                magnitude=float(x[idx] - np.median(x)),
                width=width,
            ))
    except Exception:
        pass

    # Find negative spikes (troughs)
    try:
        x_neg = -x
        neg_threshold = -threshold + 2 * x_mean

        troughs, _ = sig.find_peaks(
            x_neg,
            height=-neg_threshold,
            prominence=prominence,
        )

        for idx in troughs:
            width_result = sig.peak_widths(x_neg, [idx])
            width = int(max(1, width_result[0][0])) if len(width_result[0]) > 0 else 1

            events.append(SpikeEvent(
                t_index=int(idx),
                t_value=float(idx),
                magnitude=float(x[idx] - np.median(x)),  # Negative
                width=width,
            ))
    except Exception:
        pass

    # Sort by time
    events.sort(key=lambda e: e.t_index)

    return events


# =============================================================================
# HEAVISIDE FUNCTION - THE REAL THING
# =============================================================================

def heaviside(t: np.ndarray, t0: float = 0) -> np.ndarray:
    """
    Heaviside step function: H(t - t0)

        H(t - t0) = 0  for t < t0
        H(t - t0) = 1  for t >= t0

    This is the REAL Heaviside function.

    Args:
        t: Time array
        t0: Step location

    Returns:
        H(t - t0) array

    Example:
        >>> t = np.linspace(0, 10, 100)
        >>> H = heaviside(t, t0=5)  # Step at t=5
        >>> H[:50].sum() == 0  # All zeros before t=5
        True
    """
    return np.heaviside(t - t0, 1.0)  # 1.0 at discontinuity


def heaviside_scaled(
    t: np.ndarray,
    t0: float,
    magnitude: float,
    baseline: float = 0,
) -> np.ndarray:
    """
    Scaled Heaviside: baseline + magnitude * H(t - t0)

    Represents a step from `baseline` to `baseline + magnitude` at t0.

    Args:
        t: Time array
        t0: Step location
        magnitude: Step size
        baseline: Level before step

    Returns:
        Scaled step function
    """
    return baseline + magnitude * heaviside(t, t0)


def heaviside_ramp(
    t: np.ndarray,
    t0: float,
    t1: float,
) -> np.ndarray:
    """
    Ramp function: smooth transition from 0 to 1.

    R(t) = 0                    for t < t0
    R(t) = (t - t0) / (t1 - t0) for t0 <= t <= t1
    R(t) = 1                    for t > t1

    This is a smoothed version of Heaviside.

    Args:
        t: Time array
        t0: Ramp start
        t1: Ramp end

    Returns:
        Ramp function (0 to 1)
    """
    ramp = np.zeros_like(t, dtype=float)
    mask_rising = (t >= t0) & (t <= t1)
    mask_high = t > t1

    if t1 > t0:
        ramp[mask_rising] = (t[mask_rising] - t0) / (t1 - t0)
    ramp[mask_high] = 1.0

    return ramp


# =============================================================================
# DIRAC DELTA - THE REAL THING (discrete approximations)
# =============================================================================

def dirac_delta_discrete(
    t: np.ndarray,
    t0: float,
    dt: float = None,
) -> np.ndarray:
    """
    Discrete Dirac delta: delta(t - t0)

    In discrete time, delta[n] = 1/dt at n=n0, 0 elsewhere.
    Scaled so that sum(delta) * dt = 1 (integral property).

    This is the REAL discrete Dirac delta (Kronecker delta normalized).

    Args:
        t: Time array
        t0: Impulse location
        dt: Sample spacing (auto-detected if None)

    Returns:
        delta(t - t0) array

    Example:
        >>> t = np.linspace(0, 10, 1000)
        >>> delta = dirac_delta_discrete(t, t0=5, dt=0.01)
        >>> np.sum(delta) * 0.01  # Should be approximately 1
        1.0
    """
    if dt is None:
        dt = t[1] - t[0] if len(t) > 1 else 1.0

    delta = np.zeros_like(t, dtype=float)

    # Find closest index to t0
    idx = np.argmin(np.abs(t - t0))

    # Unit impulse scaled by 1/dt so integral = 1
    delta[idx] = 1.0 / dt

    return delta


def dirac_delta_gaussian(
    t: np.ndarray,
    t0: float,
    sigma: float = None,
) -> np.ndarray:
    """
    Gaussian approximation to Dirac delta.

        delta(t) = lim_{sigma->0} (1 / (sigma * sqrt(2*pi))) * exp(-t^2 / (2*sigma^2))

    For numerical work, small but finite sigma.

    Args:
        t: Time array
        t0: Impulse location
        sigma: Width (default: 2 samples)

    Returns:
        Gaussian approximation to delta(t - t0)
    """
    if sigma is None:
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        sigma = 2 * dt

    # Gaussian normalized to integrate to 1
    delta = np.exp(-0.5 * ((t - t0) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    return delta


def dirac_delta_sinc(
    t: np.ndarray,
    t0: float,
    bandwidth: float = None,
) -> np.ndarray:
    """
    Sinc function approximation to Dirac delta (band-limited).

        delta(t) = lim_{B->inf} B * sinc(B * t)

    Useful for band-limited signals.

    Args:
        t: Time array
        t0: Impulse location
        bandwidth: Bandwidth parameter

    Returns:
        Band-limited approximation to delta(t - t0)
    """
    if bandwidth is None:
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        bandwidth = 1.0 / (2 * dt)  # Nyquist

    tau = t - t0
    # sinc(x) = sin(pi*x) / (pi*x)
    delta = bandwidth * np.sinc(bandwidth * tau)

    return delta


def dirac_delta_scaled(
    t: np.ndarray,
    t0: float,
    magnitude: float,
    dt: float = None,
) -> np.ndarray:
    """
    Scaled discrete Dirac delta.

    Represents an impulse of strength `magnitude` at t0.

    Args:
        t: Time array
        t0: Impulse location
        magnitude: Impulse strength
        dt: Sample spacing

    Returns:
        Scaled impulse
    """
    if dt is None:
        dt = t[1] - t[0] if len(t) > 1 else 1.0

    return magnitude * dirac_delta_discrete(t, t0, dt) * dt


# =============================================================================
# CONSTRUCT BASIS FUNCTIONS FROM DETECTED EVENTS
# =============================================================================

def construct_heaviside_basis(
    t: np.ndarray,
    step_events: List[StepEvent],
) -> Dict[str, Any]:
    """
    Construct Heaviside functions for each detected step event.

    For each event at t0 with magnitude a:
        - H(t - t0): Unit step function
        - a * H(t - t0): Scaled step

    Returns:
        Dict with:
            - Individual H(t - t0) for each event
            - Combined step function (sum of scaled Heavisides)
            - Event metadata
    """
    n = len(t)

    individual = []
    combined = np.zeros(n)

    for event in step_events:
        t0 = t[event.t_index] if event.t_index < len(t) else t[-1]

        # The real Heaviside
        H = heaviside(t, t0)

        # Scaled version
        H_scaled = event.magnitude * H

        individual.append({
            't0': float(t0),
            't_index': event.t_index,
            'magnitude': event.magnitude,
            'direction': event.direction,
            'H': H,                    # Unit step
            'H_scaled': H_scaled,      # Scaled by event magnitude
        })

        combined += H_scaled

    return {
        'n_events': len(step_events),
        'events': individual,
        'combined': combined,
        'step_function': combined,
        'heaviside_decomposition': individual,
    }


def construct_dirac_basis(
    t: np.ndarray,
    spike_events: List[SpikeEvent],
    method: str = 'discrete',  # 'discrete', 'gaussian', or 'sinc'
) -> Dict[str, Any]:
    """
    Construct Dirac delta functions for each detected spike event.

    For each event at t0 with magnitude b:
        - delta(t - t0): Unit impulse
        - b * delta(t - t0): Scaled impulse

    Returns:
        Dict with:
            - Individual delta(t - t0) for each event
            - Combined impulse train
            - Event metadata
    """
    n = len(t)
    dt = t[1] - t[0] if len(t) > 1 else 1.0

    individual = []
    combined = np.zeros(n)

    for event in spike_events:
        t0 = t[event.t_index] if event.t_index < len(t) else t[-1]

        # The real Dirac delta
        if method == 'discrete':
            delta = dirac_delta_discrete(t, t0, dt)
        elif method == 'gaussian':
            sigma = max(dt, event.width * dt / 2)
            delta = dirac_delta_gaussian(t, t0, sigma)
        elif method == 'sinc':
            delta = dirac_delta_sinc(t, t0)
        else:
            delta = dirac_delta_discrete(t, t0, dt)

        # Scaled version
        delta_scaled = event.magnitude * delta * dt

        individual.append({
            't0': float(t0),
            't_index': event.t_index,
            'magnitude': event.magnitude,
            'width': event.width,
            'delta': delta,               # Unit impulse
            'delta_scaled': delta_scaled, # Scaled by event magnitude
        })

        combined += delta_scaled

    return {
        'n_events': len(spike_events),
        'events': individual,
        'combined': combined,
        'impulse_train': combined,
        'dirac_decomposition': individual,
    }


# =============================================================================
# SIGNAL DECOMPOSITION
# =============================================================================

def decompose_signal(
    t: np.ndarray,
    x: np.ndarray,
    step_threshold: float = None,
    spike_threshold: float = None,
) -> Dict[str, Any]:
    """
    Decompose signal into step + impulse + residual components.

        x(t) = sum_i(a_i * H(t - t_i)) + sum_j(b_j * delta(t - t_j)) + r(t)

    Where:
        - a_i * H(t - t_i) = step components (level shifts)
        - b_j * delta(t - t_j) = impulse components (spikes)
        - r(t) = residual (smooth variation)

    This decomposition is useful for:
        - System identification (steps/impulses are test inputs)
        - Anomaly analysis (impulses may be faults)
        - Baseline extraction

    Args:
        t: Time array
        x: Signal array
        step_threshold: Threshold for step detection
        spike_threshold: Threshold for spike detection

    Returns:
        Dict with components and metadata
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    # Detect events
    steps = detect_steps(x, threshold=step_threshold)
    spikes = detect_spikes(x, threshold=spike_threshold)

    # Construct basis functions
    H_basis = construct_heaviside_basis(t, steps)
    delta_basis = construct_dirac_basis(t, spikes)

    # Step component
    step_component = H_basis['combined']

    # Impulse component (smoothed slightly for subtraction)
    impulse_component = delta_basis['combined']
    impulse_smooth = gaussian_filter1d(impulse_component, sigma=2) if len(impulse_component) > 0 else impulse_component

    # Baseline (DC level before first step)
    if len(steps) > 0 and steps[0].t_index > 0:
        baseline = np.mean(x[:steps[0].t_index])
    else:
        baseline = np.mean(x[:min(10, len(x))])

    # Residual
    residual = x - baseline - step_component - impulse_smooth

    # Reconstruction quality
    reconstruction = baseline + step_component + impulse_smooth
    mse = np.mean((x - reconstruction)**2)
    r2 = 1 - mse / np.var(x) if np.var(x) > 0 else 0

    return {
        # Detected events
        'n_steps': len(steps),
        'n_spikes': len(spikes),
        'step_events': steps,
        'spike_events': spikes,

        # Components
        'baseline': float(baseline),
        'step_component': step_component,
        'impulse_component': impulse_component,
        'residual': residual,

        # Basis functions (for downstream use)
        'heaviside_basis': H_basis,
        'dirac_basis': delta_basis,

        # Quality metrics
        'reconstruction': reconstruction,
        'mse': float(mse),
        'r2': float(r2),

        # For system identification
        'input_signals': {
            'steps': H_basis['combined'],
            'impulses': delta_basis['combined'],
        },
    }


# =============================================================================
# MAIN COMPUTE FUNCTION
# =============================================================================

def compute(
    x: np.ndarray,
    t: np.ndarray = None,
    step_threshold: float = None,
    spike_threshold: float = None,
    include_basis: bool = True,
) -> Dict[str, Any]:
    """
    Compute Heaviside/Dirac event analysis.

    Detects step and spike events, computes real H(t-t0) and delta(t-t0)
    centered at each event.

    Args:
        x: Signal array
        t: Time array (default: integer indices)
        step_threshold: Threshold for step detection
        spike_threshold: Threshold for spike detection
        include_basis: Whether to include full basis functions

    Returns:
        Dict with:
            - Event counts and locations
            - Decomposition components
            - Real mathematical functions
            - Quality metrics
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if t is None:
        t = np.arange(n, dtype=float)
    else:
        t = np.asarray(t, dtype=float)

    # Full decomposition
    decomp = decompose_signal(t, x, step_threshold, spike_threshold)

    # Summary statistics
    result = {
        # Counts
        'n_steps': decomp['n_steps'],
        'n_spikes': decomp['n_spikes'],
        'n_events_total': decomp['n_steps'] + decomp['n_spikes'],

        # Event locations (indices)
        'step_locations': [e.t_index for e in decomp['step_events']],
        'spike_locations': [e.t_index for e in decomp['spike_events']],

        # Event magnitudes
        'step_magnitudes': [e.magnitude for e in decomp['step_events']],
        'spike_magnitudes': [e.magnitude for e in decomp['spike_events']],

        # Summary stats
        'max_step_magnitude': max([abs(e.magnitude) for e in decomp['step_events']], default=None),
        'max_spike_magnitude': max([abs(e.magnitude) for e in decomp['spike_events']], default=None),
        'mean_step_magnitude': np.mean([abs(e.magnitude) for e in decomp['step_events']]) if decomp['step_events'] else None,
        'mean_spike_magnitude': np.mean([abs(e.magnitude) for e in decomp['spike_events']]) if decomp['spike_events'] else None,

        # Quality
        'decomposition_r2': decomp['r2'],
        'decomposition_mse': decomp['mse'],
        'baseline': decomp['baseline'],
    }

    if include_basis:
        # Include full decomposition for downstream use
        result['decomposition'] = decomp
        result['step_component'] = decomp['step_component']
        result['impulse_component'] = decomp['impulse_component']
        result['residual'] = decomp['residual']

    return result
