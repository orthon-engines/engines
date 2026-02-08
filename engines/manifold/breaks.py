"""
ENGINES Break Detection Engine

Detects discontinuities in signals:
- Steps (Heaviside): sustained level changes
- Impulses (Dirac): transient spikes that return to baseline
- Gradual shifts: slow transitions between regimes

Uses three complementary detectors:
1. CUSUM (cumulative sum) — sensitive to mean shifts
2. Derivative magnitude — catches sharp transitions
3. Local outlier detection — finds isolated spikes

ENGINES computes break locations and metrics.
ORTHON classifies break types.
"""

import numpy as np
from typing import Dict, List, Any, Optional


def compute(
    y: np.ndarray,
    signal_id: str,
    sensitivity: float = 1.0,
    min_spacing: int = 10,
    context_window: int = 50,
) -> List[Dict[str, Any]]:
    """
    Detect breaks in a single signal.

    Args:
        y: Signal values (1D array, ordered by I)
        signal_id: Signal identifier (passed through to output)
        sensitivity: Detection sensitivity multiplier (lower = more breaks)
                     Default 1.0 uses adaptive thresholds.
        min_spacing: Minimum samples between breaks (prevents clustering)
        context_window: Samples before/after break for pre/post level

    Returns:
        List of break dicts matching breaks.parquet schema.
        Empty list if no breaks detected.
    """
    y = np.asarray(y, dtype=float).flatten()
    n = len(y)

    if n < 2 * context_window + min_spacing:
        return []

    # Clean NaN (interpolate for detection, but record original positions)
    nan_mask = np.isnan(y)
    if nan_mask.all():
        return []
    if nan_mask.any():
        y = _interpolate_nan(y)

    # Compute local noise floor (MAD-based, robust)
    noise_floor = _local_noise_floor(y, window=context_window)

    # --- Detector 1: CUSUM for mean shifts ---
    cusum_breaks = _detect_cusum(y, noise_floor, sensitivity)

    # --- Detector 2: Derivative magnitude for sharp transitions ---
    deriv_breaks = _detect_derivative(y, noise_floor, sensitivity)

    # --- Detector 3: Local outlier for isolated spikes ---
    spike_breaks = _detect_spikes(y, noise_floor, sensitivity)

    # Merge all candidates
    all_candidates = cusum_breaks + deriv_breaks + spike_breaks

    # Deduplicate: if multiple detectors find the same break, keep highest SNR
    merged = _merge_breaks(all_candidates, min_spacing)

    # Enrich with context
    breaks = []
    for b in merged:
        idx = b['I']
        enriched = _enrich_break(y, idx, context_window, noise_floor)
        enriched['signal_id'] = signal_id
        enriched['I'] = idx
        breaks.append(enriched)

    return breaks


# ============================================================
# DETECTOR 1: CUSUM (Cumulative Sum)
# ============================================================
# Detects sustained shifts in mean. Classic Heaviside detector.
# Page's CUSUM test: accumulates deviations from expected mean.

def _detect_cusum(
    y: np.ndarray,
    noise_floor: np.ndarray,
    sensitivity: float,
) -> List[Dict[str, Any]]:
    """
    CUSUM detector for mean shifts.

    Accumulates positive and negative deviations.
    When cumulative sum exceeds threshold → break detected.
    Threshold adapts to local noise floor.
    """
    n = len(y)
    candidates = []

    # Adaptive threshold: k * local_noise
    k = 4.0 / sensitivity  # Higher sensitivity → lower threshold

    # Running mean (exponential moving average for efficiency)
    alpha = 0.05  # Slow adaptation
    running_mean = y[0]

    cusum_pos = 0.0
    cusum_neg = 0.0
    last_reset = 0

    for i in range(1, n):
        deviation = y[i] - running_mean
        local_noise = noise_floor[i] if noise_floor[i] > 1e-10 else 1.0

        cusum_pos = max(0, cusum_pos + deviation / local_noise - 0.5)
        cusum_neg = max(0, cusum_neg - deviation / local_noise - 0.5)

        threshold = k

        if cusum_pos > threshold or cusum_neg > threshold:
            candidates.append({
                'I': i,
                'detector': 'cusum',
                'cusum_value': max(cusum_pos, cusum_neg),
            })
            cusum_pos = 0.0
            cusum_neg = 0.0
            running_mean = y[i]
            last_reset = i
        else:
            running_mean = alpha * y[i] + (1 - alpha) * running_mean

    return candidates


# ============================================================
# DETECTOR 2: Derivative Magnitude
# ============================================================
# Catches sharp transitions via first difference.
# Complements CUSUM by detecting fast changes CUSUM may lag on.

def _detect_derivative(
    y: np.ndarray,
    noise_floor: np.ndarray,
    sensitivity: float,
) -> List[Dict[str, Any]]:
    """
    Derivative-based detector for sharp transitions.

    Flags points where |dy/dt| exceeds threshold relative
    to local noise floor.
    """
    diff = np.diff(y)
    candidates = []

    # Threshold: magnitude must exceed k * local_noise
    k = 5.0 / sensitivity

    for i in range(len(diff)):
        local_noise = noise_floor[i + 1] if noise_floor[i + 1] > 1e-10 else 1.0
        score = abs(diff[i]) / local_noise

        if score > k:
            candidates.append({
                'I': i + 1,
                'detector': 'derivative',
                'deriv_score': score,
            })

    return candidates


# ============================================================
# DETECTOR 3: Local Outlier (Spike Detection)
# ============================================================
# Finds isolated spikes using local MAD.
# These are Dirac-like: large deviation that returns to baseline.

def _detect_spikes(
    y: np.ndarray,
    noise_floor: np.ndarray,
    sensitivity: float,
) -> List[Dict[str, Any]]:
    """
    Local outlier detector for isolated spikes (Dirac impulses).

    Uses sliding window MAD to find points that are extreme
    relative to their local neighborhood.
    """
    n = len(y)
    candidates = []
    half_window = 25

    k = 5.0 / sensitivity  # MAD threshold for spike

    for i in range(half_window, n - half_window):
        # Local window excluding the point itself
        window = np.concatenate([
            y[i - half_window:i],
            y[i + 1:i + half_window + 1],
        ])

        local_median = np.median(window)
        local_mad = np.median(np.abs(window - local_median))
        scaled_mad = 1.4826 * local_mad

        if scaled_mad < 1e-10:
            continue

        score = abs(y[i] - local_median) / scaled_mad

        if score > k:
            candidates.append({
                'I': i,
                'detector': 'spike',
                'spike_score': score,
            })

    return candidates


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _local_noise_floor(y: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Compute local noise floor using rolling MAD.

    Returns array same length as y with local noise estimate.
    Uses 1.4826 * MAD for Gaussian consistency.
    """
    n = len(y)
    noise = np.zeros(n)
    half = window // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half)
        segment = y[start:end]
        median = np.median(segment)
        mad = np.median(np.abs(segment - median))
        noise[i] = 1.4826 * mad

    return noise


def _merge_breaks(
    candidates: List[Dict],
    min_spacing: int,
) -> List[Dict]:
    """
    Merge candidates from multiple detectors.

    If multiple detectors find breaks within min_spacing of each other,
    keep the one with the highest detection score and merge detector info.
    """
    if not candidates:
        return []

    def get_score(c: Dict) -> float:
        """Get detection score from any detector type."""
        return max(
            c.get('cusum_value', 0),
            c.get('deriv_score', 0),
            c.get('spike_score', 0),
        )

    # Sort by I
    candidates.sort(key=lambda x: x['I'])

    merged = [candidates[0]]
    for c in candidates[1:]:
        if c['I'] - merged[-1]['I'] < min_spacing:
            # Keep higher scoring one
            if get_score(c) > get_score(merged[-1]):
                c['confirmed_by'] = merged[-1].get('confirmed_by', 1) + 1
                merged[-1] = c
            else:
                merged[-1]['confirmed_by'] = merged[-1].get('confirmed_by', 1) + 1
        else:
            merged.append(c)

    return merged


def _enrich_break(
    y: np.ndarray,
    idx: int,
    context_window: int,
    noise_floor: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute full break metrics at a detected location.

    Measures magnitude, direction, sharpness, duration, pre/post levels, SNR.
    All values suitable for breaks.parquet schema.

    For step changes: magnitude = post_level - pre_level
    For impulses: magnitude = peak deviation from local median
    """
    n = len(y)

    # Pre and post windows
    pre_start = max(0, idx - context_window)
    post_end = min(n, idx + context_window)

    pre_values = y[pre_start:idx]
    post_values = y[idx + 1:post_end] if idx + 1 < n else np.array([y[idx]])

    pre_level = float(np.median(pre_values)) if len(pre_values) > 0 else float(y[idx])
    post_level = float(np.median(post_values)) if len(post_values) > 0 else float(y[idx])

    # Local baseline (median excluding the break point)
    local_baseline = (pre_level + post_level) / 2

    # Raw level change (step magnitude)
    raw_level_change = post_level - pre_level

    # Peak deviation from baseline (impulse magnitude)
    peak_deviation = y[idx] - local_baseline

    # Use whichever is larger: level change or peak deviation
    # This captures both Heaviside steps and Dirac impulses
    if abs(peak_deviation) > abs(raw_level_change):
        raw_magnitude = peak_deviation
        is_impulse = True
    else:
        raw_magnitude = raw_level_change
        is_impulse = False

    # MAD-normalize magnitude for cross-signal comparability
    local_noise = noise_floor[idx] if noise_floor[idx] > 1e-10 else 1.0
    magnitude = raw_magnitude / local_noise

    # Direction
    if abs(raw_magnitude) < 1e-10:
        direction = 0
    else:
        direction = 1 if raw_magnitude > 0 else -1

    # Duration: how many consecutive samples are "in transition"
    # For impulses, duration is typically 1-2
    if is_impulse:
        duration = 1
    else:
        duration = _measure_transition_duration(y, idx, local_noise)

    # Sharpness: magnitude per sample of transition
    sharpness = abs(magnitude) / max(duration, 1)

    # SNR: break magnitude vs local noise
    snr = abs(raw_magnitude) / local_noise

    return {
        'magnitude': float(magnitude),
        'direction': int(direction),
        'sharpness': float(sharpness),
        'duration': int(duration),
        'pre_level': float(pre_level),
        'post_level': float(post_level),
        'snr': float(snr),
    }


def _measure_transition_duration(
    y: np.ndarray,
    idx: int,
    noise: float,
    max_duration: int = 20,
) -> int:
    """
    Measure how many samples the transition spans.

    Walks forward from break point until signal stabilizes
    (consecutive values within noise of post-level).
    Returns 1 for instantaneous (Dirac-like), higher for gradual.
    """
    n = len(y)
    if idx + 2 >= n:
        return 1

    # Post-break stable level (median of next 20-50 points)
    post_end = min(n, idx + 50)
    post_level = np.median(y[idx + 5:post_end]) if idx + 5 < post_end else y[idx + 1]

    for d in range(1, min(max_duration, n - idx)):
        if abs(y[idx + d] - post_level) < 2 * noise:
            return d

    return max_duration


def _interpolate_nan(y: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values for detection purposes."""
    y = y.copy()
    nans = np.isnan(y)
    if nans.all():
        return y

    x = np.arange(len(y))
    y[nans] = np.interp(x[nans], x[~nans], y[~nans])
    return y


def summarize_breaks(breaks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics for typology enrichment.

    Returns scalar metrics suitable for adding to typology row.

    Args:
        breaks: List of break dicts from compute()

    Returns:
        Dict with n_breaks, mean_break_spacing, mean_magnitude,
        max_magnitude, mean_sharpness, max_sharpness
    """
    if not breaks:
        return {
            'n_breaks': 0,
            'mean_break_spacing': 0.0,
            'mean_magnitude': 0.0,
            'max_magnitude': 0.0,
            'mean_sharpness': 0.0,
            'max_sharpness': 0.0,
        }

    n = len(breaks)
    magnitudes = [abs(b['magnitude']) for b in breaks]
    sharpnesses = [b['sharpness'] for b in breaks]

    # Break spacing
    locations = sorted(b['I'] for b in breaks)
    if len(locations) > 1:
        spacings = [locations[i+1] - locations[i] for i in range(len(locations)-1)]
        mean_spacing = float(np.mean(spacings))
    else:
        mean_spacing = 0.0

    return {
        'n_breaks': n,
        'mean_break_spacing': mean_spacing,
        'mean_magnitude': float(np.mean(magnitudes)),
        'max_magnitude': float(np.max(magnitudes)),
        'mean_sharpness': float(np.mean(sharpnesses)),
        'max_sharpness': float(np.max(sharpnesses)),
    }
