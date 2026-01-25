#!/usr/bin/env python3
"""
PRISM Systems Entry Point
=========================

Orchestrates fleet/cross-entity calculations.

REQUIRES: All prior stages (vector, geometry, dynamics, state, physics)

Answers: What is the FLEET doing?

Unlike other stages, Systems is indexed by WINDOW only (not entity).
It aggregates across all entities to provide fleet-level metrics.

Engines (6 - all future, user to implement):
    fleet_health, entity_clustering, correlated_failures,
    leading_indicators, cohort_tracking, fleet_trajectory

Usage:
    python -m prism.entry_points.systems
    python -m prism.entry_points.systems --force

Output:
    data/systems.parquet - window indexed (NOT entity indexed)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, VECTOR, GEOMETRY, DYNAMICS, PHYSICS
from prism.db.polars_io import read_parquet, write_parquet_atomic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE REGISTRY
# =============================================================================

# Systems engines (fleet-level analysis)
ENGINES = {
    'fleet_status': 'prism.engines.systems.fleet_status:compute',
    'entity_ranking': 'prism.engines.systems.entity_ranking:compute',
    'leading_indicator': 'prism.engines.systems.leading_indicator:compute',
    'correlated_trajectories': 'prism.engines.systems.correlated_trajectories:compute',
}


def load_engine(spec: str):
    """Load engine function from module:function spec."""
    module_path, func_name = spec.rsplit(':', 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


# =============================================================================
# CONFIG
# =============================================================================

def load_config(data_path: Path) -> Dict[str, Any]:
    """Load config.json or config.yaml from data directory."""
    import json

    config_path = data_path / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    yaml_path = data_path / 'config.yaml'
    if yaml_path.exists():
        import yaml
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    return {}


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def run_systems_engines(
    vector_df: pl.DataFrame,
    geometry_df: pl.DataFrame,
    dynamics_df: pl.DataFrame,
    state_df: pl.DataFrame,
    physics_df: pl.DataFrame,
    config: Dict[str, Any],
    engines: List[str] = None,
) -> pl.DataFrame:
    """
    Orchestrate systems engine execution.

    Pure orchestration - no computation here, just routing.

    NOTE: Systems engines operate across ALL entities for each window.
    Output is indexed by window only, not by entity.
    """
    if engines is None:
        engines = list(ENGINES.keys())

    if not engines:
        logger.warning("No systems engines implemented yet")
        return pl.DataFrame({'window': [], 'status': []})

    # Get unique windows
    window_col = 'window' if 'window' in vector_df.columns else None
    if window_col is None:
        logger.warning("No window column found - systems requires windowed data")
        return pl.DataFrame({'window': [], 'status': []})

    windows = vector_df[window_col].unique().sort().to_list()

    results = []

    for window in windows:
        # Filter all dataframes to this window
        v_df = vector_df.filter(pl.col('window') == window)
        g_df = geometry_df.filter(pl.col('window') == window) if 'window' in geometry_df.columns else geometry_df
        d_df = dynamics_df.filter(pl.col('window') == window) if 'window' in dynamics_df.columns else dynamics_df
        s_df = state_df.filter(pl.col('window') == window) if 'window' in state_df.columns else state_df
        p_df = physics_df.filter(pl.col('window') == window) if 'window' in physics_df.columns else physics_df

        row = {'window': window}

        for engine_name in engines:
            if engine_name not in ENGINES:
                continue

            try:
                engine = load_engine(ENGINES[engine_name])
                result = engine(
                    vector_df=v_df,
                    geometry_df=g_df,
                    dynamics_df=d_df,
                    state_df=s_df,
                    physics_df=p_df,
                    config=config,
                )

                # Flatten engine output into row
                if isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, (int, float, bool)) or v is None:
                            row[f"{engine_name}_{k}"] = v

            except Exception as e:
                logger.debug(f"{engine_name} skipped: {e}")

        results.append(row)

    return pl.DataFrame(results) if results else pl.DataFrame({'window': []})


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM Systems (Fleet)")
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--engines', '-e', nargs='+', choices=list(ENGINES.keys()) or None)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Systems (Fleet-Level)")
    logger.info("=" * 60)

    if not ENGINES:
        logger.warning("No systems engines implemented yet")
        logger.warning("Systems engines aggregate across entities - user must define algorithms")
        return 0

    ensure_directory()
    data_path = Path(get_path(VECTOR)).parent

    # Check dependencies
    deps = {
        'vector': get_path(VECTOR),
        'geometry': get_path(GEOMETRY),
        'dynamics': get_path(DYNAMICS),
    }

    # State and physics are optional for systems
    state_path = data_path / 'state.parquet'
    physics_path = data_path / 'physics.parquet'

    missing = [name for name, path in deps.items() if not path.exists()]
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        return 1

    output_path = data_path / 'systems.parquet'
    if output_path.exists() and not args.force:
        logger.info("systems.parquet exists, use --force to recompute")
        return 0

    # Load dependencies
    vector_df = read_parquet(deps['vector'])
    geometry_df = read_parquet(deps['geometry'])
    dynamics_df = read_parquet(deps['dynamics'])
    state_df = read_parquet(state_path) if state_path.exists() else pl.DataFrame()
    physics_df = read_parquet(physics_path) if physics_path.exists() else pl.DataFrame()
    config = load_config(data_path)

    logger.info(f"Entities: {vector_df['entity_id'].n_unique() if 'entity_id' in vector_df.columns else 1}")
    logger.info(f"Engines: {args.engines or list(ENGINES.keys())}")

    # Run
    start = time.time()
    systems_df = run_systems_engines(
        vector_df, geometry_df, dynamics_df, state_df, physics_df,
        config, args.engines
    )

    logger.info(f"Complete: {time.time() - start:.1f}s")
    logger.info(f"Output: {len(systems_df)} rows, {len(systems_df.columns)} columns")

    # Save
    write_parquet_atomic(systems_df, output_path)
    logger.info(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
