"""
Ørthon Signal Typology: Main Orchestrator
==========================================

Entry point for Signal Typology analysis.

Usage:
    from signal_typology import analyze_signal, analyze_windowed
    
    # Single window analysis
    typology = analyze_signal(series, entity_id="FD002_U001", signal_id="temp_hpc_outlet")
    
    # Windowed analysis with transition detection
    typologies = analyze_windowed(series, window_size=50, step_size=10, ...)

Output:
    SignalTypology object containing:
    - Six orthogonal axis measurements
    - Structural discontinuity detection
    - Archetype classification
    - Regime transition status
    - Human-readable summary
"""

import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple
import warnings

from .models import (
    SignalTypology, TransitionType,
    MemoryAxis, InformationAxis, RecurrenceAxis,
    VolatilityAxis, FrequencyAxis, DynamicsAxis,
    StructuralDiscontinuity
)

from .engines import (
    measure_memory_axis,
    measure_information_axis,
    measure_recurrence_axis,
    measure_volatility_axis,
    measure_frequency_axis,
    measure_dynamics_axis,
    measure_discontinuity
)

from .archetypes import (
    compute_fingerprint,
    match_archetype,
    compute_boundary_proximity,
    diagnose_differential,
    generate_summary,
    compute_confidence,
    ARCHETYPES
)


# =============================================================================
# SINGLE SIGNAL ANALYSIS
# =============================================================================

def analyze_signal(
    series: np.ndarray,
    entity_id: str = "unknown",
    signal_id: str = "unknown",
    window_start: datetime = None,
    window_end: datetime = None,
    previous_typology: SignalTypology = None,
    hurst_method: str = 'dfa'
) -> SignalTypology:
    """
    Analyze a single time series and return complete Signal Typology.
    
    Args:
        series: 1D numpy array of observations
        entity_id: Identifier for the entity (e.g., "FD002_U001")
        signal_id: Identifier for the signal (e.g., "temp_hpc_outlet")
        window_start: Start datetime of the window
        window_end: End datetime of the window
        previous_typology: Previous window's typology (for transition detection)
        hurst_method: Method for Hurst estimation ('dfa' or 'rs')
    
    Returns:
        SignalTypology with complete analysis
    """
    
    # Validate input
    series = np.asarray(series).flatten()
    n = len(series)
    
    if n < 20:
        warnings.warn(f"Series too short ({n} points). Minimum 20 recommended.")
    
    if window_start is None:
        window_start = datetime.now()
    if window_end is None:
        window_end = datetime.now()
    
    # Initialize typology
    typology = SignalTypology(
        entity_id=entity_id,
        signal_id=signal_id,
        window_start=window_start,
        window_end=window_end,
        n_observations=n
    )
    
    # Handle degenerate cases
    if n < 10:
        typology.summary = "⚠️ Insufficient data for analysis"
        typology.confidence = 0.0
        return typology
    
    # Check for constant series
    if np.std(series) < 1e-10:
        typology.summary = "⚠️ Constant series — no variation to analyze"
        typology.confidence = 0.0
        return typology
    
    # ==========================================================================
    # MEASURE ALL SIX ORTHOGONAL AXES
    # ==========================================================================
    
    # Axis 1: Memory
    prev_entropy = None
    if previous_typology is not None:
        prev_entropy = previous_typology.information.entropy_permutation
    
    typology.memory = measure_memory_axis(series, method=hurst_method)
    
    # Axis 2: Information
    typology.information = measure_information_axis(series, previous_entropy=prev_entropy)
    
    # Axis 3: Recurrence
    typology.recurrence = measure_recurrence_axis(series)
    
    # Axis 4: Volatility
    typology.volatility = measure_volatility_axis(series)
    
    # Axis 5: Frequency
    typology.frequency = measure_frequency_axis(series)
    
    # Axis 6: Dynamics
    typology.dynamics = measure_dynamics_axis(series)
    
    # ==========================================================================
    # STRUCTURAL DISCONTINUITY DETECTION
    # ==========================================================================
    
    typology.discontinuity = measure_discontinuity(series)
    
    # ==========================================================================
    # ARCHETYPE MATCHING
    # ==========================================================================
    
    # Compute fingerprint
    fingerprint = compute_fingerprint(typology)
    typology.fingerprint = fingerprint
    
    # Match to archetypes
    primary, primary_dist, secondary, secondary_dist = match_archetype(
        fingerprint,
        discontinuity_dirac=typology.discontinuity.dirac.detected,
        discontinuity_heaviside=typology.discontinuity.heaviside.detected
    )
    
    typology.archetype = primary.name
    typology.archetype_distance = primary_dist
    typology.secondary_archetype = secondary.name
    typology.secondary_distance = secondary_dist
    
    # Boundary proximity
    typology.boundary_proximity = compute_boundary_proximity(
        fingerprint, primary, secondary, primary_dist, secondary_dist
    )
    
    # ==========================================================================
    # TRANSITION DETECTION
    # ==========================================================================
    
    if previous_typology is not None:
        axes_moving, axes_stable, diagnosis = diagnose_differential(
            typology, previous_typology
        )
        typology.axes_moving = axes_moving
        typology.axes_stable = axes_stable
        
        # Determine transition type
        if len(axes_moving) == 0:
            typology.regime_transition = TransitionType.NONE
        elif len(axes_moving) >= 4 or typology.discontinuity.heaviside.detected:
            typology.regime_transition = TransitionType.IN_PROGRESS
        elif typology.boundary_proximity < 0.3:
            typology.regime_transition = TransitionType.APPROACHING
        elif typology.archetype != previous_typology.archetype:
            typology.regime_transition = TransitionType.COMPLETED
        else:
            typology.regime_transition = TransitionType.NONE
    else:
        typology.axes_moving = []
        typology.axes_stable = ["memory", "information", "recurrence", 
                               "volatility", "frequency", "dynamics"]
        typology.regime_transition = TransitionType.NONE
    
    # ==========================================================================
    # SUMMARY GENERATION
    # ==========================================================================
    
    generate_summary(typology, previous_typology)
    typology.confidence = compute_confidence(typology)
    
    return typology


# =============================================================================
# WINDOWED ANALYSIS
# =============================================================================

def analyze_windowed(
    series: np.ndarray,
    window_size: int = 50,
    step_size: int = 10,
    entity_id: str = "unknown",
    signal_id: str = "unknown",
    timestamps: List[datetime] = None,
    hurst_method: str = 'dfa'
) -> List[SignalTypology]:
    """
    Analyze time series using rolling windows.
    
    Enables:
    - Tracking typology evolution over time
    - Detecting regime transitions
    - Differential diagnosis (what changed between windows)
    
    Args:
        series: 1D numpy array of observations
        window_size: Number of observations per window
        step_size: Step between windows
        entity_id: Entity identifier
        signal_id: Signal identifier
        timestamps: Optional list of timestamps (one per observation)
        hurst_method: Method for Hurst estimation
    
    Returns:
        List of SignalTypology objects, one per window
    """
    
    series = np.asarray(series).flatten()
    n = len(series)
    
    if n < window_size:
        warnings.warn(f"Series ({n}) shorter than window ({window_size})")
        return [analyze_signal(series, entity_id, signal_id)]
    
    # Generate window start positions
    starts = list(range(0, n - window_size + 1, step_size))
    
    # Generate timestamps if not provided
    if timestamps is None:
        timestamps = [datetime.now() for _ in range(n)]
    
    typologies = []
    previous = None
    
    for start in starts:
        end = start + window_size
        window = series[start:end]
        
        window_start = timestamps[start]
        window_end = timestamps[end - 1]
        
        typology = analyze_signal(
            series=window,
            entity_id=entity_id,
            signal_id=signal_id,
            window_start=window_start,
            window_end=window_end,
            previous_typology=previous,
            hurst_method=hurst_method
        )
        
        typologies.append(typology)
        previous = typology
    
    return typologies


# =============================================================================
# BATCH ANALYSIS
# =============================================================================

def analyze_batch(
    signals: dict,
    entity_id: str = "unknown",
    window_size: int = 50,
    step_size: int = 10,
    timestamps: List[datetime] = None
) -> dict:
    """
    Analyze multiple signals from the same entity.
    
    Args:
        signals: Dict mapping signal_id -> numpy array
        entity_id: Entity identifier
        window_size: Window size for each signal
        step_size: Step size for each signal
        timestamps: Shared timestamps (assumed same for all signals)
    
    Returns:
        Dict mapping signal_id -> List[SignalTypology]
    """
    
    results = {}
    
    for signal_id, series in signals.items():
        results[signal_id] = analyze_windowed(
            series=series,
            window_size=window_size,
            step_size=step_size,
            entity_id=entity_id,
            signal_id=signal_id,
            timestamps=timestamps
        )
    
    return results


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_typologies(
    a: SignalTypology, 
    b: SignalTypology
) -> dict:
    """
    Compare two typologies and return differences.
    
    Useful for comparing:
    - Same signal at different times
    - Different signals at same time
    - Healthy vs degraded states
    """
    
    return {
        'fingerprint_distance': float(np.linalg.norm(a.fingerprint - b.fingerprint)),
        'archetype_match': a.archetype == b.archetype,
        
        'delta_hurst': b.memory.hurst_exponent - a.memory.hurst_exponent,
        'delta_entropy': b.information.entropy_permutation - a.information.entropy_permutation,
        'delta_determinism': b.recurrence.determinism - a.recurrence.determinism,
        'delta_volatility': b.volatility.garch_persistence - a.volatility.garch_persistence,
        'delta_bandwidth': b.frequency.spectral_bandwidth - a.frequency.spectral_bandwidth,
        'delta_lyapunov': b.dynamics.lyapunov_exponent - a.dynamics.lyapunov_exponent,
        
        'a_archetype': a.archetype,
        'b_archetype': b.archetype,
        
        'a_summary': a.summary,
        'b_summary': b.summary
    }


def find_regime_changes(typologies: List[SignalTypology]) -> List[int]:
    """
    Find indices where regime changes occurred.
    
    Returns:
        List of window indices where archetype changed
    """
    changes = []
    
    for i in range(1, len(typologies)):
        if typologies[i].archetype != typologies[i-1].archetype:
            changes.append(i)
        elif typologies[i].regime_transition == TransitionType.IN_PROGRESS:
            changes.append(i)
    
    return changes


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def typologies_to_dataframe(typologies: List[SignalTypology]):
    """
    Convert list of typologies to pandas DataFrame.
    
    Requires pandas.
    """
    import pandas as pd
    
    records = [t.to_dict() for t in typologies]
    return pd.DataFrame(records)


def typologies_to_parquet(typologies: List[SignalTypology], filepath: str):
    """
    Export typologies to Parquet file.
    
    Requires pandas and pyarrow/fastparquet.
    """
    df = typologies_to_dataframe(typologies)
    df.to_parquet(filepath, index=False)


# =============================================================================
# QUICK ANALYSIS
# =============================================================================

def quick_typology(series: np.ndarray) -> str:
    """
    Get quick one-line typology summary.
    
    Useful for rapid exploration.
    """
    typology = analyze_signal(series)
    return f"{typology.archetype} | H={typology.memory.hurst_exponent:.2f} " \
           f"E={typology.information.entropy_permutation:.2f} " \
           f"D={typology.recurrence.determinism:.2f} | " \
           f"conf={typology.confidence:.0%}"


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Generate example signals
    np.random.seed(42)
    
    # Trending signal (Hurst > 0.5)
    trending = np.cumsum(np.random.randn(200) + 0.1)
    
    # Mean-reverting signal (Hurst < 0.5)
    mean_reverting = np.zeros(200)
    mean_reverting[0] = 0
    for i in range(1, 200):
        mean_reverting[i] = 0.3 * mean_reverting[i-1] + np.random.randn()
    
    # Random walk (Hurst ≈ 0.5)
    random_walk = np.cumsum(np.random.randn(200))
    
    # Chaotic (Lorenz-like)
    chaotic = np.sin(np.linspace(0, 50, 200)) + 0.5 * np.sin(np.linspace(0, 127, 200))
    chaotic += np.random.randn(200) * 0.3
    
    print("=" * 70)
    print("ØRTHON SIGNAL TYPOLOGY - DEMONSTRATION")
    print("=" * 70)
    
    for name, series in [("Trending", trending), 
                          ("Mean-Reverting", mean_reverting),
                          ("Random Walk", random_walk),
                          ("Quasi-Periodic", chaotic)]:
        print(f"\n{name}:")
        print("-" * 40)
        
        typology = analyze_signal(series, signal_id=name.lower().replace(" ", "_"))
        
        print(f"  Archetype: {typology.archetype}")
        print(f"  Secondary: {typology.secondary_archetype} (dist={typology.secondary_distance:.3f})")
        print(f"  Boundary proximity: {typology.boundary_proximity:.0%}")
        print(f"  Confidence: {typology.confidence:.0%}")
        print()
        print(f"  6D Fingerprint:")
        print(f"    Memory (H):     {typology.memory.hurst_exponent:.3f} [{typology.memory.memory_class.value}]")
        print(f"    Information:    {typology.information.entropy_permutation:.3f} [{typology.information.information_class.value}]")
        print(f"    Recurrence:     {typology.recurrence.determinism:.3f} [{typology.recurrence.recurrence_class.value}]")
        print(f"    Volatility:     {typology.volatility.garch_persistence:.3f} [{typology.volatility.volatility_class.value}]")
        print(f"    Frequency BW:   {typology.frequency.spectral_bandwidth:.3f} [{typology.frequency.frequency_class.value}]")
        print(f"    Dynamics (λ):   {typology.dynamics.lyapunov_exponent:.3f} [{typology.dynamics.dynamics_class.value}]")
        print()
        print(f"  Discontinuities:")
        print(f"    Dirac:     {'✓' if typology.discontinuity.dirac.detected else '✗'} (count={typology.discontinuity.dirac.count})")
        print(f"    Heaviside: {'✓' if typology.discontinuity.heaviside.detected else '✗'} (count={typology.discontinuity.heaviside.count})")
        print()
        print(f"  Summary: {typology.summary.split(chr(10))[0]}")
    
    print("\n" + "=" * 70)
    print("WINDOWED ANALYSIS (detecting transitions)")
    print("=" * 70)
    
    # Create a signal with regime change
    regime_change = np.concatenate([
        np.cumsum(np.random.randn(100) + 0.1),  # Trending
        np.random.randn(100) * 2                 # Random/volatile
    ])
    
    typologies = analyze_windowed(
        regime_change, 
        window_size=40, 
        step_size=20,
        signal_id="regime_change_demo"
    )
    
    print(f"\nAnalyzed {len(typologies)} windows:")
    for i, t in enumerate(typologies):
        transition = ""
        if t.regime_transition != TransitionType.NONE:
            transition = f" ⚠️ {t.regime_transition.value}"
        print(f"  Window {i}: {t.archetype:20s} | conf={t.confidence:.0%}{transition}")
    
    changes = find_regime_changes(typologies)
    if changes:
        print(f"\nRegime changes detected at windows: {changes}")
