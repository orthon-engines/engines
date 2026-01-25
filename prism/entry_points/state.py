#!/usr/bin/env python3
"""
PRISM State Entry Point
=======================

Orchestrates state space position calculations.

REQUIRES: vector.parquet, geometry.parquet, dynamics.parquet

Answers: WHERE are we in state space?

Engines (6 existing + 3 future):
    Existing: transfer_entropy, granger, coupled_inertia, tension_dynamics,
              energy_dynamics, cohort
    Future: phase_position, attractor, basin

Usage:
    python -m prism.entry_points.state
    python -m prism.entry_points.state --force

Output:
    data/state.parquet - entity + window indexed
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, VECTOR, GEOMETRY, DYNAMICS
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

ENGINES = {
    'transfer_entropy': 'prism.engines.state.transfer_entropy:compute',
    'granger': 'prism.engines.state.granger:compute',
    'coupled_inertia': 'prism.engines.state.coupled_inertia:compute',
    'tension_dynamics': 'prism.engines.state.tension_dynamics:compute',
    'energy_dynamics': 'prism.engines.state.energy_dynamics:compute',
    'cohort': 'prism.engines.state.cohort:compute',
    # Future engines (user to implement)
    # 'phase_position': 'prism.engines.state.phase_position:compute',
    # 'attractor': 'prism.engines.state.attractor:compute',
    # 'basin': 'prism.engines.state.basin:compute',
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

def run_state_engines(
    vector_df: pl.DataFrame,
    geometry_df: pl.DataFrame,
    dynamics_df: pl.DataFrame,
    config: Dict[str, Any],
    engines: List[str] = None,
) -> pl.DataFrame:
    """
    Orchestrate state engine execution.

    Pure orchestration - no computation here, just routing.
    """
    if engines is None:
        engines = list(ENGINES.keys())

    results = []
    entities = vector_df['entity_id'].unique().to_list() if 'entity_id' in vector_df.columns else ['default']

    for entity_id in entities:
        # Filter data for this entity
        v_df = vector_df.filter(pl.col('entity_id') == entity_id) if 'entity_id' in vector_df.columns else vector_df
        g_df = geometry_df.filter(pl.col('entity_id') == entity_id) if 'entity_id' in geometry_df.columns else geometry_df
        d_df = dynamics_df.filter(pl.col('entity_id') == entity_id) if 'entity_id' in dynamics_df.columns else dynamics_df

        row = {'entity_id': entity_id}

        for engine_name in engines:
            if engine_name not in ENGINES:
                continue

            try:
                engine = load_engine(ENGINES[engine_name])
                result = engine(
                    vector_df=v_df,
                    geometry_df=g_df,
                    dynamics_df=d_df,
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

    return pl.DataFrame(results) if results else pl.DataFrame({'entity_id': []})


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PRISM State")
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--engines', '-e', nargs='+', choices=list(ENGINES.keys()))
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM State")
    logger.info("=" * 60)

    ensure_directory()
    data_path = Path(get_path(VECTOR)).parent

    # Check dependencies
    vector_path = get_path(VECTOR)
    geometry_path = get_path(GEOMETRY)
    dynamics_path = get_path(DYNAMICS)

    missing = []
    if not vector_path.exists():
        missing.append('vector')
    if not geometry_path.exists():
        missing.append('geometry')
    if not dynamics_path.exists():
        missing.append('dynamics')

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Run the pipeline in order: vector → geometry → dynamics → state")
        return 1

    output_path = data_path / 'state.parquet'
    if output_path.exists() and not args.force:
        logger.info("state.parquet exists, use --force to recompute")
        return 0

    # Load dependencies
    vector_df = read_parquet(vector_path)
    geometry_df = read_parquet(geometry_path)
    dynamics_df = read_parquet(dynamics_path)
    config = load_config(data_path)

    logger.info(f"Vector: {len(vector_df)} rows")
    logger.info(f"Geometry: {len(geometry_df)} rows")
    logger.info(f"Dynamics: {len(dynamics_df)} rows")
    logger.info(f"Engines: {args.engines or list(ENGINES.keys())}")

    # Run
    start = time.time()
    state_df = run_state_engines(vector_df, geometry_df, dynamics_df, config, args.engines)

    logger.info(f"Complete: {time.time() - start:.1f}s")
    logger.info(f"Output: {len(state_df)} rows, {len(state_df.columns)} columns")

    # Save
    write_parquet_atomic(state_df, output_path)
    logger.info(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
