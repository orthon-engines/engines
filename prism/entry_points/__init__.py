"""
PRISM Entry Points.

Thin orchestrators that:
1. Read manifest
2. Load data
3. Call engines
4. Write output

Entry points do NOT contain compute logic - only orchestration.
"""

__all__ = ['run', 'run_from_manifest']


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning when running as __main__."""
    if name in ('run', 'run_from_manifest'):
        from .signal_vector import run, run_from_manifest
        return run if name == 'run' else run_from_manifest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
