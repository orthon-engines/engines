#!/usr/bin/env python3
"""
PRISM Architecture Validator
============================

Validates the PRISM pipeline outputs for correct architecture:

1. Vector: n_entities × n_signals rows, includes Laplace metrics
2. Geometry: n_entities rows, includes precision matrix
3. Dynamics: n_entities rows (NOT per signal!), uses Mahalanobis
4. Physics: n_entities rows

Usage:
    python -m prism.entry_points.validate
"""

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from prism.db.parquet_store import get_path, OBSERVATIONS, VECTOR, GEOMETRY, DYNAMICS, PHYSICS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def validate_architecture(data_path: Path) -> bool:
    """
    Verify the PRISM architecture is correct.

    Returns True if valid, False if errors found.
    """
    errors = []
    warnings = []

    # Check files exist
    obs_path = data_path / 'observations.parquet'
    vec_path = data_path / 'vector.parquet'
    geo_path = data_path / 'geometry.parquet'
    dyn_path = data_path / 'dynamics.parquet'
    phy_path = data_path / 'physics.parquet'

    if not obs_path.exists():
        errors.append("observations.parquet not found")
        return False

    obs = pl.read_parquet(obs_path)
    n_entities = obs['entity_id'].n_unique()
    n_signals = obs['signal_id'].n_unique()

    logger.info(f"Observations: {n_entities} entities, {n_signals} signals")

    # ==========================================================================
    # VECTOR: n_entities × n_signals rows, includes Laplace
    # ==========================================================================
    if vec_path.exists():
        vec = pl.read_parquet(vec_path)
        expected_rows = n_entities * n_signals

        if len(vec) != expected_rows:
            warnings.append(f"vector.parquet: expected {expected_rows} rows, got {len(vec)}")

        # Check Laplace metrics
        laplace_cols = [c for c in vec.columns if 'gradient' in c.lower() or 'laplacian' in c.lower() or 'divergence' in c.lower()]
        if len(laplace_cols) < 9:
            errors.append(f"vector.parquet: missing Laplace metrics, found only {len(laplace_cols)}: {laplace_cols}")
        else:
            logger.info(f"  ✓ Vector: {len(vec)} rows, {len(laplace_cols)} Laplace metrics (local geometry)")
    else:
        errors.append("vector.parquet not found")

    # ==========================================================================
    # GEOMETRY: n_entities rows, has precision matrix
    # ==========================================================================
    if geo_path.exists():
        geo = pl.read_parquet(geo_path)

        if len(geo) != n_entities:
            errors.append(f"geometry.parquet: expected {n_entities} rows (one per entity), got {len(geo)}")

        # Check precision matrix (binary blob or legacy JSON)
        has_precision = (
            'precision_matrix_blob' in geo.columns or
            'covariance_inverse_json' in geo.columns or
            'precision_matrix' in geo.columns
        )
        if not has_precision:
            errors.append("geometry.parquet: missing precision matrix (needed for Mahalanobis)")
        else:
            fmt = "binary" if 'precision_matrix_blob' in geo.columns else "JSON"
            logger.info(f"  ✓ Geometry: {len(geo)} rows (one per entity), has precision matrix ({fmt})")
    else:
        errors.append("geometry.parquet not found")

    # ==========================================================================
    # DYNAMICS: n_entities rows (NOT per signal!), uses Mahalanobis
    # ==========================================================================
    if dyn_path.exists():
        dyn = pl.read_parquet(dyn_path)

        if len(dyn) != n_entities:
            if len(dyn) == n_entities * n_signals:
                errors.append(
                    f"dynamics.parquet: got {len(dyn)} rows (n_entities × n_signals)\n"
                    "  → WRONG! Should be one row per ENTITY, not per signal.\n"
                    "  → hd_slope is computed across ALL signals together."
                )
            else:
                warnings.append(f"dynamics.parquet: expected {n_entities} rows, got {len(dyn)}")

        # Check hd_slope
        if 'hd_slope' not in dyn.columns:
            errors.append("dynamics.parquet: missing hd_slope column")

        # Check distance metric
        if 'distance_metric' in dyn.columns:
            metrics = dyn['distance_metric'].unique().to_list()
            if 'mahalanobis' not in metrics:
                warnings.append(f"dynamics.parquet: distance_metric is {metrics}, should include 'mahalanobis'")
            else:
                mahal_count = dyn.filter(pl.col('distance_metric') == 'mahalanobis').height
                logger.info(f"  ✓ Dynamics: {len(dyn)} rows (one per entity), Mahalanobis: {mahal_count}/{len(dyn)}")
        else:
            logger.info(f"  ✓ Dynamics: {len(dyn)} rows (one per entity)")
    else:
        errors.append("dynamics.parquet not found")

    # ==========================================================================
    # PHYSICS: n_entities rows
    # ==========================================================================
    if phy_path.exists():
        phy = pl.read_parquet(phy_path)

        if len(phy) != n_entities:
            warnings.append(f"physics.parquet: expected {n_entities} rows, got {len(phy)}")
        else:
            logger.info(f"  ✓ Physics: {len(phy)} rows (one per entity)")

        # Check Hamiltonian
        if 'hamiltonian_H' not in phy.columns:
            warnings.append("physics.parquet: missing hamiltonian_H column")
    else:
        warnings.append("physics.parquet not found (optional)")

    # ==========================================================================
    # Report
    # ==========================================================================
    logger.info("")

    if errors:
        logger.error("=" * 60)
        logger.error("ARCHITECTURE VALIDATION FAILED")
        logger.error("=" * 60)
        for e in errors:
            logger.error(f"  ✗ {e}")
        return False

    if warnings:
        logger.warning("Warnings:")
        for w in warnings:
            logger.warning(f"  ⚠ {w}")

    logger.info("=" * 60)
    logger.info("ARCHITECTURE VALIDATED")
    logger.info("=" * 60)
    logger.info("")
    logger.info("The Two Geometries:")
    logger.info("  - Laplace (local):     Vector layer - gradient, laplacian, divergence per signal")
    logger.info("  - Mahalanobis (global): Dynamics layer - distance on manifold per entity")
    logger.info("")
    logger.info("Pipeline:")
    logger.info("  Vector (WHAT) → Geometry (WHERE) → Dynamics (HOW) → Physics (WHY)")
    logger.info("")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="PRISM Architecture Validator"
    )
    parser.add_argument('--data-path', '-d', type=str, default=None,
                        help='Path to data directory (default: uses PRISM_DATA_PATH)')

    args = parser.parse_args()

    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = get_path(OBSERVATIONS).parent

    logger.info("=" * 60)
    logger.info("PRISM Architecture Validator")
    logger.info("=" * 60)
    logger.info(f"Data path: {data_path}")
    logger.info("")

    success = validate_architecture(data_path)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
