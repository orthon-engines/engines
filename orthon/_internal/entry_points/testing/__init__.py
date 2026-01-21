"""
PRISM Testing - Synthetic data generation and validation utilities.

This module provides tools for generating test data and validating
the PRISM analysis pipeline.

Available generators:
- dynamical: Generate datasets from chaotic systems (Lorenz, Rossler)
- pendulum: Pendulum regime transitions
- dynamic_vector: Dynamic vector computation utilities
- dynamic_state: Dynamic state computation utilities
"""

from orthon._internal.entry_points.testing.dynamical import generate_datasets
from orthon._internal.entry_points.testing.pendulum import generate_pendulum_regime

__all__ = [
    'generate_datasets',
    'generate_pendulum_regime',
]
