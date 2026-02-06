"""
01: Signal Vector Entry Point
=============================

Pure orchestration - calls signal engines from engines/signal/.
No computation logic lives here.

Stages: observations.parquet → signal_vector.parquet

Credit: Entry point architecture ensuring separation of concerns.
"""

# Import from the existing clean implementation
from .signal_vector import (
    compute_signal_vector,
    run,
    run_from_manifest,
    get_engine_compute_func,
    get_engine_min_samples,
    get_engine_window,
    validate_engine_can_run,
    group_engines_by_window,
    load_window_factors,
    main,
)

__all__ = [
    'compute_signal_vector',
    'run',
    'run_from_manifest',
    'get_engine_compute_func',
    'get_engine_min_samples',
    'get_engine_window',
    'validate_engine_can_run',
    'group_engines_by_window',
    'load_window_factors',
    'main',
]

if __name__ == '__main__':
    main()
