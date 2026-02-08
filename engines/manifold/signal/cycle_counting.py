"""
Cycle Counting Engine.

Rainflow cycle counting for fatigue analysis.
Counts stress/strain cycles and their ranges.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute(y: np.ndarray) -> Dict[str, float]:
    """
    Perform rainflow cycle counting.

    Args:
        y: Signal values (stress, strain, load)

    Returns:
        dict with n_cycles, max_range, mean_range, damage_index,
              n_full_cycles, n_half_cycles
    """
    result = {
        'n_cycles': 0,
        'n_full_cycles': 0,
        'n_half_cycles': 0,
        'max_range': np.nan,
        'mean_range': np.nan,
        'damage_index': np.nan
    }

    # Handle NaN values
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 4:
        return result

    try:
        # Try to use rainflow library if available
        try:
            import rainflow
            cycles = list(rainflow.extract_cycles(y))

            if len(cycles) > 0:
                ranges = [c[0] for c in cycles]
                counts = [c[2] for c in cycles]

                n_full = sum(1 for c in counts if c == 1.0)
                n_half = sum(1 for c in counts if c == 0.5)

                # Damage index using Palmgren-Miner (S-N slope of 3)
                max_range = max(ranges) if ranges else 1.0
                damage = sum((r / max_range) ** 3 * c for r, c in zip(ranges, counts))

                result = {
                    'n_cycles': int(sum(counts)),
                    'n_full_cycles': n_full,
                    'n_half_cycles': n_half,
                    'max_range': float(max(ranges)),
                    'mean_range': float(np.mean(ranges)),
                    'damage_index': float(damage)
                }
                return result

        except ImportError:
            pass  # Fall through to manual implementation

        # Manual implementation: Simple peak-valley counting
        # Step 1: Find turning points (peaks and valleys)
        turning_points = _extract_turning_points(y)

        if len(turning_points) < 2:
            return result

        # Step 2: Count cycles from turning points
        # Each pair of consecutive extrema forms a half-cycle
        ranges = []
        for i in range(len(turning_points) - 1):
            r = abs(turning_points[i+1] - turning_points[i])
            ranges.append(r)

        if not ranges:
            return result

        ranges = np.array(ranges)

        # Convert to full cycles (pair half-cycles by range)
        # Sort ranges and pair similar ones
        sorted_ranges = np.sort(ranges)[::-1]
        n_half = len(sorted_ranges)
        n_full = n_half // 2
        n_remaining = n_half % 2

        max_range = float(np.max(ranges))
        mean_range = float(np.mean(ranges))

        # Damage calculation (each half-cycle contributes 0.5)
        damage = float(np.sum((ranges / max_range) ** 3) * 0.5)

        result = {
            'n_cycles': n_full + n_remaining,
            'n_full_cycles': n_full,
            'n_half_cycles': n_remaining,
            'max_range': max_range,
            'mean_range': mean_range,
            'damage_index': damage
        }

    except Exception:
        pass

    return result


def _extract_turning_points(y: np.ndarray) -> List[float]:
    """Extract peaks and valleys from signal."""
    if len(y) < 3:
        return list(y)

    turning = [y[0]]  # Include first point

    for i in range(1, len(y) - 1):
        # Check if it's a local extremum (peak or valley)
        is_peak = y[i] > y[i-1] and y[i] > y[i+1]
        is_valley = y[i] < y[i-1] and y[i] < y[i+1]

        if is_peak or is_valley:
            turning.append(y[i])

    turning.append(y[-1])  # Include last point

    return turning
