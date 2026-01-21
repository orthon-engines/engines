"""
PRISM Windowed Engines
======================

Engines that require a distribution of values (window-based).

These engines compute metrics over windows of observations
and produce SparseSignal outputs with fewer timestamps than input.

Engines:
    - entropy: Sample, permutation, spectral entropy
    - hurst: Hurst exponent (R/S analysis)
    - lyapunov: Lyapunov exponents (chaos)
    - rqa: Recurrence quantification analysis
    - garch: GARCH volatility modeling

These engines are wrappers around the existing engine implementations
in prism/engines/, adapted to produce SparseSignal outputs.
"""

from typing import List

# List of windowed engines (require distribution, produce sparse output)
WINDOWED_ENGINES: List[str] = [
    'entropy',      # Sample entropy, permutation entropy, spectral entropy
    'hurst',        # Hurst exponent via R/S rescaled range
    'lyapunov',     # Lyapunov exponents
    'rqa',          # Recurrence quantification analysis
    'garch',        # GARCH volatility model parameters
    'spectral',     # Spectral density analysis
    'wavelet',      # Wavelet decomposition
]

# Minimum observations required per engine
MIN_OBSERVATIONS = {
    'entropy': 30,
    'hurst': 50,
    'lyapunov': 50,
    'rqa': 50,
    'garch': 100,
    'spectral': 64,
    'wavelet': 32,
}


def get_min_observations(engine: str) -> int:
    """Get minimum observations required for an engine."""
    return MIN_OBSERVATIONS.get(engine, 30)


def is_windowed_engine(engine: str) -> bool:
    """Check if an engine is windowed (requires distribution)."""
    return engine.lower() in WINDOWED_ENGINES


# Note: The actual engine implementations remain in prism/engines/
# This module provides classification and metadata for the new architecture.
# Future work will add SparseSignal-producing wrappers here.
