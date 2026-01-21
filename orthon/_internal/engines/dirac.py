"""
PRISM Dirac Engine
==================

Measures IMPULSE/SHOCK characteristics in signal topology.
Deployed when break_detector identifies transient spikes that revert.

WHAT IT MEASURES:
    Dirac impulses represent sudden shocks that REVERT toward baseline.
    Examples: Flash crashes, news shocks, supply disruptions that resolve,
    equipment alarms that clear, temporary faults.

PHYSICS INTERPRETATION:
    A Dirac impulse is energy that enters the system and DISSIPATES.
    The system absorbs the shock and returns toward equilibrium.
    
                    │
                    │  ← impulse_magnitude
    Before: ────────┼────────  After (decays back)
                    
    Unlike Heaviside (step), Dirac reverts to prior level.

OUTPUTS:
    Per-impulse metrics:
        - impulse_time: When did the impulse occur?
        - impulse_magnitude: How big was the spike?
        - impulse_direction: Up (+1) or down (-1)?
        - decay_rate: How fast did it revert? (half-life)
        - energy: Integrated deviation from baseline
        - reversion_ratio: How much did it revert? (1.0 = full)
    
    Aggregate metrics (for behavioral vector):
        - dirac_n_impulses: Total impulse count
        - dirac_mean_magnitude: Average absolute spike size
        - dirac_total_energy: Total energy injected
        - dirac_mean_decay_rate: Average decay rate
        - dirac_up_ratio: Fraction of impulses that go up
        - dirac_mean_reversion: Average reversion completeness

RELATIONSHIP TO OTHER ENGINES:
    break_detector → finds ALL discontinuities
    heaviside → analyzes breaks that are PERSISTENT (steps)
    dirac → analyzes breaks that REVERT (impulses)

Usage:
    from orthon._internal.engines.dirac import compute_dirac, get_dirac_metrics
    
    # With break detection results
    metrics = compute_dirac(values, break_indices)
    
    # Standalone (will detect breaks internally)
    metrics = get_dirac_metrics(values)

Author: PRISM Team
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'pre_window': 20,          # Observations before impulse for baseline
    'decay_window': 30,        # Window to measure decay
    'reversion_threshold': 0.5,  # Must revert at least this fraction
    'min_impulse_magnitude': 1.0,  # Minimum z-score to count
    'energy_integration_window': 20,  # Window for energy calculation
}

MIN_OBSERVATIONS = 50


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ImpulseEvent:
    """Single Dirac impulse event."""
    index: int
    time: Any  # Could be datetime, int cycle, etc.
    magnitude: float  # Signed: positive = up, negative = down
    direction: int  # +1 or -1
    baseline: float  # Pre-impulse level
    peak_value: float  # Value at impulse
    decay_rate: float  # Exponential decay rate (higher = faster decay)
    half_life: float  # Time to decay to half magnitude
    energy: float  # Integrated deviation from baseline
    reversion_ratio: float  # How much it reverted (1.0 = full)


# =============================================================================
# IMPULSE DETECTION AND MEASUREMENT
# =============================================================================

def identify_impulses(
    values: np.ndarray,
    break_indices: np.ndarray,
    config: Dict = None,
) -> List[ImpulseEvent]:
    """
    Analyze break points to identify which are Dirac impulses.
    
    A break is classified as an IMPULSE if:
    1. It reverts toward baseline after the spike
    2. The decay is measurable
    
    Args:
        values: Signal values
        break_indices: Indices where breaks were detected
        config: Configuration parameters
    
    Returns:
        List of ImpulseEvent objects for confirmed impulses
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    n = len(values)
    impulses = []
    
    pre_win = config['pre_window']
    decay_win = config['decay_window']
    reversion_thresh = config['reversion_threshold']
    
    # Compute global scale for normalization
    global_std = np.std(values)
    if global_std < 1e-10:
        global_std = 1.0
    
    for idx in break_indices:
        # Need enough data before and after
        if idx < pre_win or idx >= n - decay_win:
            continue
        
        # Measure baseline (pre-impulse level)
        pre_values = values[idx - pre_win : idx]
        baseline = np.mean(pre_values)
        baseline_std = np.std(pre_values)
        
        # Peak value at impulse
        peak_value = values[idx]
        
        # Impulse magnitude (signed)
        magnitude = peak_value - baseline
        
        # Skip tiny impulses
        if abs(magnitude) / global_std < config['min_impulse_magnitude']:
            continue
        
        # Measure decay: how does the series evolve after impulse?
        post_values = values[idx + 1 : idx + 1 + decay_win]
        
        if len(post_values) < 5:
            continue
        
        # Measure reversion: how much did it return toward baseline?
        final_value = np.mean(post_values[-5:])
        
        if abs(magnitude) > 1e-10:
            reversion_ratio = 1.0 - abs(final_value - baseline) / abs(magnitude)
            reversion_ratio = max(0.0, min(1.0, reversion_ratio))
        else:
            reversion_ratio = 0.0
        
        # Only count as impulse if it reverts sufficiently
        if reversion_ratio < reversion_thresh:
            continue  # This is more like a Heaviside step
        
        # Measure decay rate (fit exponential decay)
        decay_rate, half_life = _fit_decay(post_values, baseline, magnitude)
        
        # Compute energy (integrated deviation from baseline)
        energy_win = min(config['energy_integration_window'], len(post_values))
        energy_values = values[idx : idx + energy_win]
        energy = np.sum(np.abs(energy_values - baseline))
        
        # Create impulse event
        impulse = ImpulseEvent(
            index=idx,
            time=idx,  # Could be replaced with actual timestamp
            magnitude=magnitude,
            direction=1 if magnitude > 0 else -1,
            baseline=baseline,
            peak_value=peak_value,
            decay_rate=decay_rate,
            half_life=half_life,
            energy=energy,
            reversion_ratio=reversion_ratio,
        )
        
        impulses.append(impulse)
    
    return impulses


def _fit_decay(
    post_values: np.ndarray,
    baseline: float,
    magnitude: float,
) -> Tuple[float, float]:
    """
    Fit exponential decay to post-impulse values.
    
    Model: deviation(t) = magnitude * exp(-decay_rate * t)
    
    Returns:
        decay_rate: Exponential decay constant
        half_life: Time to decay to half magnitude
    """
    n = len(post_values)
    
    if n < 3 or abs(magnitude) < 1e-10:
        return 0.0, float('inf')
    
    # Compute deviation from baseline
    deviations = post_values - baseline
    
    # Normalize by initial magnitude
    normalized = np.abs(deviations) / abs(magnitude)
    
    # Fit log-linear regression: log(deviation) = -decay_rate * t
    # Only use points where deviation is still positive
    valid = normalized > 0.01
    
    if np.sum(valid) < 3:
        # Not enough valid points for fit
        # Estimate from simple ratio
        if len(normalized) >= 2 and normalized[0] > 0.01:
            decay_rate = -np.log(max(0.01, normalized[-1] / normalized[0])) / n
        else:
            decay_rate = 0.1  # Default moderate decay
    else:
        t = np.arange(n)[valid]
        y = np.log(normalized[valid])
        
        # Linear regression
        if len(t) >= 2:
            slope, _ = np.polyfit(t, y, 1)
            decay_rate = -slope
        else:
            decay_rate = 0.1
    
    # Ensure positive decay rate
    decay_rate = max(0.001, decay_rate)
    
    # Half-life: time for exp(-decay_rate * t) = 0.5
    half_life = np.log(2) / decay_rate
    
    return float(decay_rate), float(half_life)


def compute_dirac(
    values: np.ndarray,
    break_indices: np.ndarray = None,
    config: Dict = None,
) -> Dict[str, Any]:
    """
    Full Dirac impulse analysis.
    
    Args:
        values: Signal values
        break_indices: Pre-detected break indices (optional)
        config: Configuration parameters
    
    Returns:
        Dict with impulse analysis results
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    n = len(values)
    
    # If no break indices provided, detect them
    if break_indices is None or len(break_indices) == 0:
        from orthon._internal.engines.break_detector import compute_breaks
        break_result = compute_breaks(values)
        break_indices = break_result['break_indices']
    
    # No breaks = no impulses
    if len(break_indices) == 0:
        return {
            'impulses': [],
            'n_impulses': 0,
            'impulse_indices': [],
        }
    
    # Identify which breaks are Dirac impulses
    impulses = identify_impulses(values, break_indices, config)
    
    # Extract arrays for analysis
    impulse_indices = [imp.index for imp in impulses]
    magnitudes = [imp.magnitude for imp in impulses]
    directions = [imp.direction for imp in impulses]
    decay_rates = [imp.decay_rate for imp in impulses]
    half_lives = [imp.half_life for imp in impulses]
    energies = [imp.energy for imp in impulses]
    reversions = [imp.reversion_ratio for imp in impulses]
    
    # Compute intervals between impulses
    if len(impulse_indices) >= 2:
        intervals = np.diff(impulse_indices)
        mean_interval = float(np.mean(intervals))
    else:
        intervals = []
        mean_interval = n if len(impulse_indices) <= 1 else 0
    
    return {
        'impulses': impulses,
        'n_impulses': len(impulses),
        'impulse_indices': impulse_indices,
        'magnitudes': magnitudes,
        'directions': directions,
        'decay_rates': decay_rates,
        'half_lives': half_lives,
        'energies': energies,
        'reversions': reversions,
        'intervals': intervals,
        'mean_interval': mean_interval,
    }


# =============================================================================
# METRICS FOR BEHAVIORAL VECTOR
# =============================================================================

def get_dirac_metrics(
    values: np.ndarray,
    break_indices: np.ndarray = None,
    config: Dict = None,
) -> Dict[str, float]:
    """
    Get Dirac metrics suitable for behavioral vector.
    
    Returns metrics prefixed with 'dirac_' for inclusion
    alongside hurst, entropy, etc.
    """
    if len(values) < MIN_OBSERVATIONS:
        return {
            'dirac_n_impulses': 0.0,
            'dirac_mean_magnitude': 0.0,
            'dirac_max_magnitude': 0.0,
            'dirac_total_energy': 0.0,
            'dirac_mean_decay_rate': 0.0,
            'dirac_mean_half_life': -1.0,
            'dirac_up_ratio': 0.5,
            'dirac_mean_reversion': 0.0,
            'dirac_mean_interval': -1.0,
        }
    
    result = compute_dirac(values, break_indices, config)
    
    n_impulses = result['n_impulses']
    
    if n_impulses == 0:
        return {
            'dirac_n_impulses': 0.0,
            'dirac_mean_magnitude': 0.0,
            'dirac_max_magnitude': 0.0,
            'dirac_total_energy': 0.0,
            'dirac_mean_decay_rate': 0.0,
            'dirac_mean_half_life': -1.0,
            'dirac_up_ratio': 0.5,
            'dirac_mean_reversion': 0.0,
            'dirac_mean_interval': -1.0,
        }
    
    magnitudes = np.array(result['magnitudes'])
    directions = np.array(result['directions'])
    decay_rates = np.array(result['decay_rates'])
    half_lives = np.array(result['half_lives'])
    energies = np.array(result['energies'])
    reversions = np.array(result['reversions'])
    
    # Filter infinite half-lives for mean calculation
    finite_half_lives = half_lives[np.isfinite(half_lives)]
    mean_half_life = float(np.mean(finite_half_lives)) if len(finite_half_lives) > 0 else -1.0
    
    return {
        'dirac_n_impulses': float(n_impulses),
        'dirac_mean_magnitude': float(np.mean(np.abs(magnitudes))),
        'dirac_max_magnitude': float(np.max(np.abs(magnitudes))),
        'dirac_total_energy': float(np.sum(energies)),
        'dirac_mean_decay_rate': float(np.mean(decay_rates)),
        'dirac_mean_half_life': mean_half_life,
        'dirac_up_ratio': float(np.mean(directions > 0)),
        'dirac_mean_reversion': float(np.mean(reversions)),
        'dirac_mean_interval': result['mean_interval'],
    }


# =============================================================================
# IMPULSE RESPONSE RECONSTRUCTION
# =============================================================================

def reconstruct_impulse_response(
    values: np.ndarray,
    impulses: List[ImpulseEvent],
    response_window: int = 30,
) -> np.ndarray:
    """
    Reconstruct the impulse response component of the signal.
    
    Useful for:
    - Visualizing the impulse structure
    - Separating impulse component from baseline
    - Measuring goodness of fit
    
    Returns:
        Array of same length as values with impulse response approximation
    """
    n = len(values)
    impulse_component = np.zeros(n)
    
    for imp in impulses:
        idx = imp.index
        mag = imp.magnitude
        decay = imp.decay_rate
        
        # Generate decaying response
        for t in range(min(response_window, n - idx)):
            response = mag * np.exp(-decay * t)
            impulse_component[idx + t] += response
    
    return impulse_component


def compute_impulse_residual(
    values: np.ndarray,
    impulses: List[ImpulseEvent],
) -> Dict[str, float]:
    """
    Compute residual after removing impulse response.
    
    Returns statistics on how well impulses explain the signal variance.
    """
    if not impulses:
        return {
            'impulse_explained_variance': 0.0,
            'residual_std': float(np.std(values)),
            'residual_mean': float(np.mean(values)),
        }
    
    # Compute baseline (mean of non-impulse regions)
    impulse_regions = set()
    for imp in impulses:
        for t in range(imp.index, min(imp.index + 30, len(values))):
            impulse_regions.add(t)
    
    non_impulse_mask = [i not in impulse_regions for i in range(len(values))]
    
    if any(non_impulse_mask):
        baseline = np.mean(values[np.array(non_impulse_mask)])
    else:
        baseline = np.mean(values)
    
    impulse_component = reconstruct_impulse_response(values, impulses)
    reconstructed = baseline + impulse_component
    residual = values - reconstructed
    
    total_var = np.var(values)
    residual_var = np.var(residual)
    
    if total_var > 1e-10:
        explained_ratio = 1.0 - (residual_var / total_var)
    else:
        explained_ratio = 0.0
    
    return {
        'impulse_explained_variance': max(0.0, explained_ratio),
        'residual_std': float(np.std(residual)),
        'residual_mean': float(np.mean(residual)),
    }


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PRISM Dirac Engine')
    parser.add_argument('--test', action='store_true', help='Run tests')
    
    args = parser.parse_args()
    
    if args.test:
        print("=" * 70)
        print("PRISM Dirac Engine - Test")
        print("=" * 70)
        
        np.random.seed(42)
        
        # Test 1: Clear impulse (spike and revert)
        print("\n1. CLEAR IMPULSE (flash crash style)")
        print("-" * 50)
        n = 300
        values = 100 + np.random.randn(n) * 0.5
        # Add impulse at t=100: spike down, then recover
        values[100] = 85  # Sharp drop
        for t in range(1, 30):
            values[100 + t] = 85 + 15 * (1 - np.exp(-0.2 * t)) + np.random.randn() * 0.3
        
        metrics = get_dirac_metrics(values)
        print(f"   Impulses detected: {metrics['dirac_n_impulses']}")
        print(f"   Mean magnitude: {metrics['dirac_mean_magnitude']:.3f}")
        print(f"   Mean decay rate: {metrics['dirac_mean_decay_rate']:.3f}")
        print(f"   Mean half-life: {metrics['dirac_mean_half_life']:.1f}")
        print(f"   Mean reversion: {metrics['dirac_mean_reversion']:.3f}")
        
        # Test 2: Multiple impulses
        print("\n2. MULTIPLE IMPULSES (news shocks)")
        print("-" * 50)
        values = 50 + np.random.randn(n) * 0.3
        # Add several impulses
        for t_imp in [50, 120, 200]:
            direction = np.random.choice([-1, 1])
            values[t_imp] = 50 + direction * 8
            for t in range(1, 25):
                recovery = direction * 8 * np.exp(-0.15 * t)
                values[t_imp + t] = 50 + recovery + np.random.randn() * 0.3
        
        metrics = get_dirac_metrics(values)
        print(f"   Impulses detected: {metrics['dirac_n_impulses']}")
        print(f"   Total energy: {metrics['dirac_total_energy']:.1f}")
        print(f"   Up ratio: {metrics['dirac_up_ratio']:.2f}")
        
        # Test 3: No impulses (smooth)
        print("\n3. NO IMPULSES (smooth random walk)")
        print("-" * 50)
        values = np.cumsum(np.random.randn(n) * 0.1) + 100
        
        metrics = get_dirac_metrics(values)
        print(f"   Impulses detected: {metrics['dirac_n_impulses']}")
        
        # Test 4: Step function (should NOT be detected as impulse)
        print("\n4. STEP FUNCTION (should NOT count as impulse)")
        print("-" * 50)
        values = np.zeros(n)
        values[:150] = 20 + np.random.randn(150) * 0.3
        values[150:] = 30 + np.random.randn(150) * 0.3  # Step, doesn't revert
        
        metrics = get_dirac_metrics(values)
        print(f"   Impulses detected: {metrics['dirac_n_impulses']} (should be 0)")
        print(f"   Mean reversion: {metrics['dirac_mean_reversion']:.3f}")
        
        # Test 5: Mixed (impulses + noise)
        print("\n5. REALISTIC: Impulses in noisy baseline")
        print("-" * 50)
        values = 100 + np.random.randn(500) * 1.0
        # Add several realistic impulses
        impulse_times = [80, 200, 350]
        for t_imp in impulse_times:
            mag = np.random.choice([-1, 1]) * (5 + np.random.rand() * 5)
            values[t_imp] = 100 + mag
            decay = 0.1 + np.random.rand() * 0.2
            for t in range(1, 40):
                values[t_imp + t] = 100 + mag * np.exp(-decay * t) + np.random.randn() * 1.0
        
        metrics = get_dirac_metrics(values)
        print(f"   Impulses detected: {metrics['dirac_n_impulses']} (expected ~3)")
        print(f"   Mean decay rate: {metrics['dirac_mean_decay_rate']:.3f}")
        print(f"   Mean half-life: {metrics['dirac_mean_half_life']:.1f}")
        
        print("\n" + "=" * 70)
        print("Dirac Engine Test Complete")
        print("=" * 70)
