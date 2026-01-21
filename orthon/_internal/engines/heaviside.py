"""
PRISM Heaviside Engine
======================

Measures STEP FUNCTION characteristics in signal topology.
Deployed when break_detector identifies persistent level shifts.

WHAT IT MEASURES:
    Heaviside functions represent sudden, PERSISTENT changes in level.
    Examples: Fed rate decisions, policy changes, regime shifts, 
    equipment failures that don't recover.

PHYSICS INTERPRETATION:
    A Heaviside step is energy that enters the system and STAYS.
    The system reaches a new equilibrium at a different level.
    
    Before: ────────────┐
                        │  ← step_magnitude
    After:              └──────────────
                        
    Unlike Dirac (impulse), Heaviside doesn't revert.

OUTPUTS:
    Per-step metrics:
        - step_time: When did the step occur?
        - step_magnitude: How big was the jump?
        - step_direction: Up (+1) or down (-1)?
        - pre_level: Mean level before step
        - post_level: Mean level after step
        - step_sharpness: How sudden? (1 = instantaneous)
    
    Aggregate metrics (for behavioral vector):
        - heaviside_n_steps: Total step count
        - heaviside_mean_magnitude: Average absolute step size
        - heaviside_net_displacement: Sum of all steps (signed)
        - heaviside_up_ratio: Fraction of steps that go up
        - heaviside_mean_duration: Average time between steps
        - heaviside_largest_step: Maximum step magnitude

RELATIONSHIP TO OTHER ENGINES:
    break_detector → finds ALL discontinuities
    heaviside → analyzes breaks that are PERSISTENT (steps)
    dirac → analyzes breaks that REVERT (impulses)

Usage:
    from orthon._internal.engines.heaviside import compute_heaviside, get_heaviside_metrics
    
    # With break detection results
    metrics = compute_heaviside(values, break_indices)
    
    # Standalone (will detect breaks internally)
    metrics = get_heaviside_metrics(values)

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
    'pre_window': 20,         # Observations before step to measure pre-level
    'post_window': 20,        # Observations after step to measure post-level
    'persistence_threshold': 0.7,  # Post level must stay within this fraction of step
    'min_step_magnitude': 0.5,     # Minimum z-score to count as step
    'sharpness_window': 5,    # Window to measure transition sharpness
}

MIN_OBSERVATIONS = 50


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class StepEvent:
    """Single Heaviside step event."""
    index: int
    time: Any  # Could be datetime, int cycle, etc.
    magnitude: float  # Signed: positive = up, negative = down
    direction: int  # +1 or -1
    pre_level: float
    post_level: float
    sharpness: float  # 0-1, higher = more instantaneous
    persistence: float  # 0-1, higher = more persistent


# =============================================================================
# STEP DETECTION AND MEASUREMENT
# =============================================================================

def identify_steps(
    values: np.ndarray,
    break_indices: np.ndarray,
    config: Dict = None,
) -> List[StepEvent]:
    """
    Analyze break points to identify which are Heaviside steps.
    
    A break is classified as a STEP if:
    1. The post-break level persists (doesn't revert)
    2. The transition is relatively sharp
    
    Args:
        values: Signal values
        break_indices: Indices where breaks were detected
        config: Configuration parameters
    
    Returns:
        List of StepEvent objects for confirmed steps
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    n = len(values)
    steps = []
    
    pre_win = config['pre_window']
    post_win = config['post_window']
    persist_thresh = config['persistence_threshold']
    
    # Compute global scale for normalization
    global_std = np.std(values)
    if global_std < 1e-10:
        global_std = 1.0
    
    for idx in break_indices:
        # Need enough data before and after
        if idx < pre_win or idx >= n - post_win:
            continue
        
        # Measure pre and post levels
        pre_values = values[idx - pre_win : idx]
        post_values = values[idx + 1 : idx + 1 + post_win]
        
        pre_level = np.mean(pre_values)
        post_level = np.mean(post_values)
        
        # Step magnitude (signed)
        magnitude = post_level - pre_level
        
        # Skip tiny steps
        if abs(magnitude) / global_std < config['min_step_magnitude']:
            continue
        
        # Measure persistence: does post-level stay stable?
        # Compare end of post-window to beginning of post-window
        if len(post_values) >= 10:
            early_post = np.mean(post_values[:5])
            late_post = np.mean(post_values[-5:])
            
            # If late_post is closer to pre_level than early_post, it's reverting (Dirac, not Heaviside)
            reversion = abs(late_post - pre_level) / (abs(early_post - pre_level) + 1e-10)
            persistence = 1.0 - min(1.0, max(0.0, reversion - persist_thresh))
        else:
            persistence = 0.5  # Unknown
        
        # Only count as step if persistent
        if persistence < 0.5:
            continue  # This is more like a Dirac impulse
        
        # Measure sharpness: how instantaneous is the transition?
        sharp_win = config['sharpness_window']
        if idx >= sharp_win and idx < n - sharp_win:
            # Compare actual transition to ideal step
            transition = values[idx - sharp_win : idx + sharp_win + 1]
            ideal_step = np.concatenate([
                np.full(sharp_win, pre_level),
                [values[idx]],
                np.full(sharp_win, post_level)
            ])
            
            # Sharpness = 1 - normalized deviation from ideal
            deviation = np.mean(np.abs(transition - ideal_step)) / (abs(magnitude) + 1e-10)
            sharpness = max(0.0, 1.0 - deviation)
        else:
            sharpness = 0.5  # Unknown
        
        # Create step event
        step = StepEvent(
            index=idx,
            time=idx,  # Could be replaced with actual timestamp
            magnitude=magnitude,
            direction=1 if magnitude > 0 else -1,
            pre_level=pre_level,
            post_level=post_level,
            sharpness=sharpness,
            persistence=persistence,
        )
        
        steps.append(step)
    
    return steps


def compute_heaviside(
    values: np.ndarray,
    break_indices: np.ndarray = None,
    config: Dict = None,
) -> Dict[str, Any]:
    """
    Full Heaviside step analysis.
    
    Args:
        values: Signal values
        break_indices: Pre-detected break indices (optional)
        config: Configuration parameters
    
    Returns:
        Dict with step analysis results
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    n = len(values)
    
    # If no break indices provided, detect them
    if break_indices is None or len(break_indices) == 0:
        from orthon._internal.engines.break_detector import compute_breaks
        break_result = compute_breaks(values)
        break_indices = break_result['break_indices']
    
    # No breaks = no steps
    if len(break_indices) == 0:
        return {
            'steps': [],
            'n_steps': 0,
            'step_indices': [],
        }
    
    # Identify which breaks are Heaviside steps
    steps = identify_steps(values, break_indices, config)
    
    # Extract arrays for analysis
    step_indices = [s.index for s in steps]
    magnitudes = [s.magnitude for s in steps]
    directions = [s.direction for s in steps]
    sharpnesses = [s.sharpness for s in steps]
    persistences = [s.persistence for s in steps]
    
    # Compute intervals between steps
    if len(step_indices) >= 2:
        intervals = np.diff(step_indices)
        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))
    else:
        intervals = []
        mean_interval = n if len(step_indices) <= 1 else 0
        std_interval = 0
    
    return {
        'steps': steps,
        'n_steps': len(steps),
        'step_indices': step_indices,
        'magnitudes': magnitudes,
        'directions': directions,
        'sharpnesses': sharpnesses,
        'persistences': persistences,
        'intervals': intervals,
        'mean_interval': mean_interval,
        'std_interval': std_interval,
    }


# =============================================================================
# METRICS FOR BEHAVIORAL VECTOR
# =============================================================================

def get_heaviside_metrics(
    values: np.ndarray,
    break_indices: np.ndarray = None,
    config: Dict = None,
) -> Dict[str, float]:
    """
    Get Heaviside metrics suitable for behavioral vector.
    
    Returns metrics prefixed with 'heaviside_' for inclusion
    alongside hurst, entropy, etc.
    """
    if len(values) < MIN_OBSERVATIONS:
        return {
            'heaviside_n_steps': 0.0,
            'heaviside_mean_magnitude': 0.0,
            'heaviside_max_magnitude': 0.0,
            'heaviside_net_displacement': 0.0,
            'heaviside_up_ratio': 0.5,
            'heaviside_mean_interval': -1.0,
            'heaviside_mean_sharpness': 0.0,
            'heaviside_mean_persistence': 0.0,
        }
    
    result = compute_heaviside(values, break_indices, config)
    
    n_steps = result['n_steps']
    
    if n_steps == 0:
        return {
            'heaviside_n_steps': 0.0,
            'heaviside_mean_magnitude': 0.0,
            'heaviside_max_magnitude': 0.0,
            'heaviside_net_displacement': 0.0,
            'heaviside_up_ratio': 0.5,
            'heaviside_mean_interval': -1.0,
            'heaviside_mean_sharpness': 0.0,
            'heaviside_mean_persistence': 0.0,
        }
    
    magnitudes = np.array(result['magnitudes'])
    directions = np.array(result['directions'])
    sharpnesses = np.array(result['sharpnesses'])
    persistences = np.array(result['persistences'])
    
    return {
        'heaviside_n_steps': float(n_steps),
        'heaviside_mean_magnitude': float(np.mean(np.abs(magnitudes))),
        'heaviside_max_magnitude': float(np.max(np.abs(magnitudes))),
        'heaviside_net_displacement': float(np.sum(magnitudes)),
        'heaviside_up_ratio': float(np.mean(directions > 0)),
        'heaviside_mean_interval': result['mean_interval'],
        'heaviside_mean_sharpness': float(np.mean(sharpnesses)),
        'heaviside_mean_persistence': float(np.mean(persistences)),
    }


# =============================================================================
# STEP FUNCTION RECONSTRUCTION
# =============================================================================

def reconstruct_step_function(
    values: np.ndarray,
    steps: List[StepEvent],
) -> np.ndarray:
    """
    Reconstruct the underlying step function from detected steps.
    
    Useful for:
    - Visualizing the step structure
    - Separating step component from noise
    - Measuring goodness of fit
    
    Returns:
        Array of same length as values with step function approximation
    """
    n = len(values)
    step_func = np.zeros(n)
    
    if not steps:
        step_func[:] = np.mean(values)
        return step_func
    
    # Sort steps by index
    sorted_steps = sorted(steps, key=lambda s: s.index)
    
    # Build step function
    current_level = sorted_steps[0].pre_level
    step_func[:sorted_steps[0].index] = current_level
    
    for i, step in enumerate(sorted_steps):
        current_level = step.post_level
        
        if i + 1 < len(sorted_steps):
            end_idx = sorted_steps[i + 1].index
        else:
            end_idx = n
        
        step_func[step.index:end_idx] = current_level
    
    return step_func


def compute_step_residual(
    values: np.ndarray,
    steps: List[StepEvent],
) -> Dict[str, float]:
    """
    Compute residual after removing step function.
    
    Returns statistics on how well steps explain the signal.
    """
    step_func = reconstruct_step_function(values, steps)
    residual = values - step_func
    
    total_var = np.var(values)
    residual_var = np.var(residual)
    
    if total_var > 1e-10:
        explained_ratio = 1.0 - (residual_var / total_var)
    else:
        explained_ratio = 0.0
    
    return {
        'step_explained_variance': explained_ratio,
        'residual_std': float(np.std(residual)),
        'residual_mean': float(np.mean(residual)),
    }


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PRISM Heaviside Engine')
    parser.add_argument('--test', action='store_true', help='Run tests')
    
    args = parser.parse_args()
    
    if args.test:
        print("=" * 70)
        print("PRISM Heaviside Engine - Test")
        print("=" * 70)
        
        np.random.seed(42)
        
        # Test 1: Clear step function
        print("\n1. CLEAR STEP FUNCTION (Fed rate style)")
        print("-" * 50)
        n = 300
        values = np.zeros(n)
        values[:100] = 2.0 + np.random.randn(100) * 0.1
        values[100:200] = 2.5 + np.random.randn(100) * 0.1  # Step up
        values[200:] = 2.25 + np.random.randn(100) * 0.1    # Step down
        
        metrics = get_heaviside_metrics(values)
        print(f"   Steps detected: {metrics['heaviside_n_steps']}")
        print(f"   Mean magnitude: {metrics['heaviside_mean_magnitude']:.3f}")
        print(f"   Net displacement: {metrics['heaviside_net_displacement']:.3f}")
        print(f"   Up ratio: {metrics['heaviside_up_ratio']:.2f}")
        
        # Test 2: Multiple steps (staircase)
        print("\n2. STAIRCASE (multiple steps up)")
        print("-" * 50)
        values = np.zeros(n)
        for i in range(6):
            start = i * 50
            end = (i + 1) * 50
            values[start:end] = i * 1.0 + np.random.randn(50) * 0.1
        
        metrics = get_heaviside_metrics(values)
        print(f"   Steps detected: {metrics['heaviside_n_steps']}")
        print(f"   Net displacement: {metrics['heaviside_net_displacement']:.3f}")
        print(f"   Mean interval: {metrics['heaviside_mean_interval']:.1f}")
        
        # Test 3: No steps (smooth)
        print("\n3. NO STEPS (random walk)")
        print("-" * 50)
        values = np.cumsum(np.random.randn(n) * 0.1)
        
        metrics = get_heaviside_metrics(values)
        print(f"   Steps detected: {metrics['heaviside_n_steps']}")
        
        # Test 4: Mixed with Dirac (should filter out impulses)
        print("\n4. MIXED: STEPS + IMPULSES (should only count steps)")
        print("-" * 50)
        values = np.zeros(n)
        values[:100] = 5.0 + np.random.randn(100) * 0.1
        values[100:] = 8.0 + np.random.randn(200) * 0.1  # Step at 100
        # Add impulse at 200 (should NOT be counted as step)
        values[200] = 15.0  # Spike
        values[201:210] = 8.0 + np.linspace(3, 0, 9)  # Decay back
        
        metrics = get_heaviside_metrics(values)
        print(f"   Steps detected: {metrics['heaviside_n_steps']} (should be ~1, not 2)")
        print(f"   Mean persistence: {metrics['heaviside_mean_persistence']:.3f}")
        
        print("\n" + "=" * 70)
        print("Heaviside Engine Test Complete")
        print("=" * 70)
