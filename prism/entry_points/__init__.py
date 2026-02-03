"""
PRISM Entry Points.

Thin orchestrators that:
1. Read manifest
2. Load data
3. Call engines
4. Write output

Entry points do NOT contain compute logic - only orchestration.
"""

from .signal_vector import run, run_from_manifest

__all__ = ['run', 'run_from_manifest']
