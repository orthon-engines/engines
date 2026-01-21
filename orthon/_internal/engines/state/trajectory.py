"""
State Trajectory
================

Compute state as direct derivatives of geometry.
No windowing — pure calculus.
"""

import numpy as np
from typing import List, Dict

from orthon._internal.core.signals.types import GeometrySnapshot, StateTrajectory
from orthon._internal.engines.geometry.snapshot import snapshot_to_vector


def compute_state_trajectory(
    snapshots: List[GeometrySnapshot],
) -> StateTrajectory:
    """
    Compute position, velocity, acceleration from geometry snapshots.

    Args:
        snapshots: List of GeometrySnapshot (one per timestamp)

    Returns:
        StateTrajectory with direct derivatives
    """
    if len(snapshots) == 0:
        return StateTrajectory(
            timestamps=np.array([]),
            position=np.array([[]]),
            velocity=np.array([[]]),
            acceleration=np.array([[]]),
        )

    timestamps = np.array([s.timestamp for s in snapshots])

    # Convert each geometry snapshot to a vector
    positions = []
    for snap in snapshots:
        pos = snapshot_to_vector(snap)
        positions.append(pos)

    positions = np.array(positions)  # [n_timestamps, n_features]

    # Velocity: dG/dt (direct derivative)
    velocity = np.gradient(positions, timestamps, axis=0)

    # Acceleration: d²G/dt² (derivative of velocity)
    acceleration = np.gradient(velocity, timestamps, axis=0)

    return StateTrajectory(
        timestamps=timestamps,
        position=positions,
        velocity=velocity,
        acceleration=acceleration,
    )


def detect_failure_acceleration(
    state: StateTrajectory,
    velocity_threshold: float = 0.1,
    acceleration_threshold: float = 0.05,
) -> np.ndarray:
    """
    Detect timestamps where system shows failure signature.

    Failure signature:
    - Velocity magnitude above threshold (system is moving)
    - Acceleration positive and increasing (system is speeding up)

    Args:
        state: StateTrajectory to analyze
        velocity_threshold: Minimum velocity to consider "moving"
        acceleration_threshold: Minimum acceleration to consider "accelerating"

    Returns:
        Boolean array: True at timestamps showing failure signature
    """
    v_mag = state.speed
    a_mag = state.acceleration_magnitude

    # Acceleration direction: is it in same direction as velocity?
    # Positive dot product = speeding up
    if len(state.velocity.shape) > 1 and state.velocity.shape[1] > 1:
        a_direction = np.einsum('ij,ij->i', state.velocity, state.acceleration)
    else:
        # 1D case
        a_direction = state.velocity.flatten() * state.acceleration.flatten()

    # Failure signature
    is_moving = v_mag > velocity_threshold
    is_accelerating = a_mag > acceleration_threshold
    speeding_up = a_direction > 0

    return is_moving & is_accelerating & speeding_up


def compute_state_metrics(state: StateTrajectory) -> Dict[str, float]:
    """
    Compute summary metrics from state trajectory.

    Returns:
        Dict with velocity and acceleration statistics
    """
    v_mag = state.speed
    a_mag = state.acceleration_magnitude

    return {
        'mean_velocity': float(np.mean(v_mag)),
        'max_velocity': float(np.max(v_mag)),
        'std_velocity': float(np.std(v_mag)),
        'mean_acceleration': float(np.mean(a_mag)),
        'max_acceleration': float(np.max(a_mag)),
        'std_acceleration': float(np.std(a_mag)),
        'n_timestamps': len(state.timestamps),
    }


def find_acceleration_events(
    state: StateTrajectory,
    threshold: float = 2.0,
) -> List[Dict]:
    """
    Find significant acceleration events.

    Args:
        state: StateTrajectory to analyze
        threshold: Number of std devs above mean for "significant"

    Returns:
        List of dicts with event details
    """
    a_mag = state.acceleration_magnitude
    mean_a = np.mean(a_mag)
    std_a = np.std(a_mag) + 1e-10

    # Find peaks above threshold
    significant = (a_mag - mean_a) / std_a > threshold

    events = []
    in_event = False
    event_start = 0

    for i, is_sig in enumerate(significant):
        if is_sig and not in_event:
            in_event = True
            event_start = i
        elif not is_sig and in_event:
            in_event = False
            # Record event
            peak_idx = event_start + np.argmax(a_mag[event_start:i])
            events.append({
                'start_idx': event_start,
                'end_idx': i,
                'peak_idx': peak_idx,
                'timestamp': float(state.timestamps[peak_idx]),
                'peak_acceleration': float(a_mag[peak_idx]),
                'duration': i - event_start,
            })

    return events


def compute_trajectory_curvature(state: StateTrajectory) -> np.ndarray:
    """
    Compute path curvature at each timestamp.

    κ = |a_perp| / |v|^2

    High curvature = sharp turns in behavioral space
    Low curvature = smooth trajectory
    """
    v = state.velocity
    a = state.acceleration

    if len(v.shape) == 1 or v.shape[1] == 1:
        # 1D case: curvature is 0
        return np.zeros(len(state.timestamps))

    v_norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-10

    # a_perp = a - (a·v̂)v̂
    v_unit = v / v_norm
    a_parallel = np.sum(a * v_unit, axis=1, keepdims=True) * v_unit
    a_perp = a - a_parallel

    curvature = np.linalg.norm(a_perp, axis=1) / (v_norm.squeeze() ** 2 + 1e-10)
    return curvature
