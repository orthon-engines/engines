#!/usr/bin/env python3
"""
PRISM Fields Entry Point
========================

Real Navier-Stokes field analysis. Not inspired by. The real equations.

    dv/dt + (v . nabla)v = -nabla(p)/rho + nu * nabla^2(v) + f

REQUIRES: 3D velocity field data (u, v, w components)

This layer operates on FIELD DATA, not time series.
- Time series: shape (nt,) - one value per time step
- Field data: shape (nx, ny, nz) or (nx, ny, nz, nt) - spatial grid

Output:
    data/fields.parquet - Flow analysis metrics:
        - reynolds_number: Re = UL/nu
        - taylor_reynolds_number: Re_lambda
        - flow_regime: laminar/transitional/turbulent
        - mean_tke: Turbulent kinetic energy [m^2/s^2]
        - mean_dissipation: Energy dissipation rate [m^2/s^3]
        - mean_enstrophy: Vorticity intensity [1/s^2]
        - mean_helicity: Helical motion measure [m/s^2]
        - kolmogorov_length: Smallest turbulent scale [m]
        - taylor_microscale: Intermediate scale [m]
        - integral_length_scale: Largest eddy scale [m]
        - energy_spectrum_slope: Should be ~ -5/3 for turbulence
        - is_kolmogorov_turbulence: True if slope ~ -5/3

Usage:
    # With velocity field data
    python -m prism.entry_points.fields --data /path/to/velocity_data/

    # With synthetic test data
    python -m prism.entry_points.fields --synthetic --nx 64

    # Required config in data/config.yaml:
    fields:
        dx: 0.001  # Grid spacing x [m]
        dy: 0.001  # Grid spacing y [m]
        dz: 0.001  # Grid spacing z [m]
        nu: 1.0e-6 # Kinematic viscosity [m^2/s]

References:
    Pope (2000) "Turbulent Flows"
    Kolmogorov (1941) "Local structure of turbulence"
"""

import argparse
import logging
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import polars as pl

from prism.db.parquet_store import get_path, ensure_directory, FIELDS
from prism.db.polars_io import write_parquet_atomic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

from prism.config.validator import ConfigurationError


def load_config(data_path: Path) -> Dict[str, Any]:
    """
    Load fields config from data directory.

    REQUIRES explicit grid spacing and viscosity. NO DEFAULTS.

    Raises:
        ConfigurationError: If config not found or incomplete
    """
    config_path = data_path / 'config.yaml'

    if not config_path.exists():
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: config.yaml not found\n"
            f"{'='*60}\n"
            f"Location: {config_path}\n\n"
            f"Fields analysis requires explicit configuration:\n\n"
            f"  fields:\n"
            f"    dx: 0.001  # Grid spacing [m]\n"
            f"    dy: 0.001\n"
            f"    dz: 0.001\n"
            f"    nu: 1.0e-6  # Kinematic viscosity [m^2/s]\n"
            f"\n"
            f"NO DEFAULTS. NO FALLBACKS. Know your physics.\n"
            f"{'='*60}"
        )

    with open(config_path) as f:
        user_config = yaml.safe_load(f) or {}

    fields_config = user_config.get('fields', {})

    required = ['dx', 'dy', 'dz', 'nu']
    missing = [k for k in required if k not in fields_config]

    if missing:
        raise ConfigurationError(
            f"\n{'='*60}\n"
            f"CONFIGURATION ERROR: Missing fields config\n"
            f"{'='*60}\n"
            f"File: {config_path}\n"
            f"Missing: {missing}\n\n"
            f"Add to config.yaml:\n\n"
            f"  fields:\n"
            f"    dx: 0.001  # Grid spacing x [m]\n"
            f"    dy: 0.001  # Grid spacing y [m]\n"
            f"    dz: 0.001  # Grid spacing z [m]\n"
            f"    nu: 1.0e-6  # Kinematic viscosity [m^2/s]\n"
            f"\n"
            f"Common values:\n"
            f"  Water at 20C: nu = 1.0e-6 m^2/s\n"
            f"  Air at 20C:   nu = 1.5e-5 m^2/s\n"
            f"  JHTDB:        nu = 0.000185 m^2/s\n"
            f"\n"
            f"NO DEFAULTS. NO FALLBACKS. Know your physics.\n"
            f"{'='*60}"
        )

    return fields_config


# =============================================================================
# DATA LOADING
# =============================================================================

def load_velocity_data(data_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load velocity field data from directory.

    Expected files:
        u.npy or u.npz - x-component of velocity
        v.npy or v.npz - y-component of velocity
        w.npy or w.npz - z-component of velocity

    Returns:
        Dict with 'u', 'v', 'w' arrays
    """
    velocity = {}

    for component in ['u', 'v', 'w']:
        npy_path = data_dir / f'{component}.npy'
        npz_path = data_dir / f'{component}.npz'

        if npy_path.exists():
            velocity[component] = np.load(npy_path)
            logger.info(f"Loaded {npy_path.name}: shape {velocity[component].shape}")
        elif npz_path.exists():
            with np.load(npz_path) as npz:
                # Assume first array in npz
                key = list(npz.keys())[0]
                velocity[component] = npz[key]
            logger.info(f"Loaded {npz_path.name}: shape {velocity[component].shape}")
        else:
            raise FileNotFoundError(
                f"Velocity component '{component}' not found.\n"
                f"Expected: {npy_path} or {npz_path}"
            )

    return velocity


def create_synthetic_data(nx: int, ny: int, nz: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Create synthetic turbulent velocity field for testing.

    Uses random Fourier modes with k^(-5/3) energy spectrum.
    This is for testing only - not real turbulence data.
    """
    from prism.utils.fields_orchestrator import create_synthetic_turbulence

    logger.info(f"Creating synthetic turbulence: {nx}x{ny}x{nz}")
    return create_synthetic_turbulence(nx, ny, nz, Re_target=1000.0, seed=seed)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRISM Fields - Real Navier-Stokes Analysis"
    )
    parser.add_argument('--data', '-d', type=Path,
                        help='Directory containing velocity field data (u.npy, v.npy, w.npy)')
    parser.add_argument('--synthetic', '-s', action='store_true',
                        help='Use synthetic turbulence data for testing')
    parser.add_argument('--nx', type=int, default=64,
                        help='Grid size for synthetic data (default: 64)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Recompute even if output exists')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Fields Engine")
    logger.info("Real Navier-Stokes. Not inspired by. The real equations.")
    logger.info("=" * 60)

    ensure_directory()
    data_path = get_path(FIELDS).parent

    output_path = get_path(FIELDS)

    if output_path.exists() and not args.force:
        logger.info("fields.parquet exists, use --force to recompute")
        return 0

    # Load or create velocity data
    if args.synthetic:
        velocity_data = create_synthetic_data(args.nx, args.nx, args.nx)
        # Use default config for synthetic
        config = {
            'dx': 2 * np.pi / args.nx,
            'dy': 2 * np.pi / args.nx,
            'dz': 2 * np.pi / args.nx,
            'nu': 1e-4,  # Moderate viscosity for synthetic
        }
        entity_id = f"synthetic_{args.nx}"
    elif args.data:
        if not args.data.exists():
            logger.error(f"Data directory not found: {args.data}")
            return 1
        velocity_data = load_velocity_data(args.data)
        config = load_config(data_path)
        entity_id = args.data.name
    else:
        logger.error("Must specify --data or --synthetic")
        parser.print_help()
        return 1

    # Run analysis
    from prism.utils.fields_orchestrator import FieldsOrchestrator

    logger.info(f"Grid config: dx={config['dx']}, dy={config['dy']}, dz={config['dz']}")
    logger.info(f"Viscosity: nu={config['nu']} m^2/s")

    orchestrator = FieldsOrchestrator(config)

    start = time.time()
    df = orchestrator.run(velocity_data, entity_id=entity_id)
    elapsed = time.time() - start

    # Log key results
    if len(df) > 0:
        row = df.row(0, named=True)
        logger.info("")
        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info(f"Flow regime:           {row['flow_regime']}")
        logger.info(f"Reynolds number:       {row['reynolds_number']:.2f}")
        logger.info(f"Taylor Re:             {row['taylor_reynolds_number']:.2f}")
        logger.info(f"Turbulence intensity:  {row['turbulence_intensity']:.4f}")
        logger.info("")
        logger.info(f"Mean TKE:              {row['mean_tke']:.6f} m^2/s^2")
        logger.info(f"Mean dissipation:      {row['mean_dissipation']:.6e} m^2/s^3")
        logger.info(f"Mean enstrophy:        {row['mean_enstrophy']:.6f} 1/s^2")
        logger.info(f"Mean vorticity:        {row['mean_vorticity']:.6f} 1/s")
        logger.info("")
        logger.info(f"Kolmogorov length:     {row['kolmogorov_length']:.6e} m")
        logger.info(f"Taylor microscale:     {row['taylor_microscale']:.6e} m")
        logger.info(f"Integral scale:        {row['integral_length_scale']:.6e} m")
        logger.info("")
        if row['energy_spectrum_slope'] is not None:
            logger.info(f"Energy spectrum slope: {row['energy_spectrum_slope']:.3f}")
            logger.info(f"Expected (Kolmogorov): -1.667")
            logger.info(f"Is Kolmogorov turb:    {row['is_kolmogorov_turbulence']}")
        logger.info("=" * 60)

        write_parquet_atomic(df, output_path)
        logger.info(f"Wrote {output_path}")
        logger.info(f"  {len(df)} rows, {len(df.columns)} columns in {elapsed:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
