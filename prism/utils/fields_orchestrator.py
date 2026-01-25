"""
Fields Orchestrator

Runs Navier-Stokes analysis on velocity field data.

Input: 3D velocity field data (u, v, w components)
Output: fields.parquet with real fluid dynamics metrics

No approximations. No "inspired by." The real Navier-Stokes.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from prism.engines.fields.navier_stokes import (
    VelocityField,
    analyze_velocity_field,
)


class FieldsOrchestrator:
    """
    Orchestrates Navier-Stokes field analysis.

    Input: Velocity field data (3D or 4D arrays)
    Output: fields.parquet with flow analysis results

    Required config (NO DEFAULTS - must be explicit):
        dx, dy, dz: Grid spacing [m]
        nu: Kinematic viscosity [m^2/s]
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fields orchestrator.

        Args:
            config: Must contain:
                - dx: Grid spacing in x [m]
                - dy: Grid spacing in y [m]
                - dz: Grid spacing in z [m]
                - nu: Kinematic viscosity [m^2/s]
                - dt: Time step [s] (optional, for time-varying fields)
                - rho: Density [kg/m^3] (optional, default 1.0)
        """
        self.config = config

        # Required config - NO DEFAULTS
        self.dx = config.get('dx')
        self.dy = config.get('dy')
        self.dz = config.get('dz')
        self.nu = config.get('nu')

        if None in [self.dx, self.dy, self.dz, self.nu]:
            raise ValueError(
                "Fields analysis requires explicit grid and viscosity config:\n"
                "  dx, dy, dz: Grid spacing [m]\n"
                "  nu: Kinematic viscosity [m^2/s]\n"
                "\n"
                "Example for water at 20C:\n"
                "  dx: 0.001  # 1mm grid\n"
                "  dy: 0.001\n"
                "  dz: 0.001\n"
                "  nu: 1.0e-6  # water kinematic viscosity\n"
                "\n"
                "Example for JHTDB isotropic turbulence:\n"
                "  dx: 0.00614  # 2*pi/1024\n"
                "  dy: 0.00614\n"
                "  dz: 0.00614\n"
                "  nu: 0.000185"
            )

        # Optional config
        self.dt = config.get('dt', 1.0)
        self.rho = config.get('rho', 1.0)

    def run(self, velocity_data: Dict[str, np.ndarray], entity_id: str = None) -> pl.DataFrame:
        """
        Run analysis on velocity field data.

        Args:
            velocity_data: Dict with keys 'u', 'v', 'w' containing velocity arrays
                Each array should be shape (nx, ny, nz) or (nx, ny, nz, nt)
            entity_id: Optional identifier for this field snapshot

        Returns:
            DataFrame with analysis results
        """
        # Validate input
        required_keys = ['u', 'v', 'w']
        for key in required_keys:
            if key not in velocity_data:
                raise ValueError(f"velocity_data missing required key: {key}")

        field = VelocityField(
            u=velocity_data['u'],
            v=velocity_data['v'],
            w=velocity_data['w'],
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            dt=self.dt,
            nu=self.nu,
            rho=self.rho,
        )

        results = analyze_velocity_field(field)

        # Remove field data (starts with _)
        scalar_results = {k: v for k, v in results.items() if not k.startswith('_')}

        # Add entity_id if provided
        if entity_id is not None:
            scalar_results['entity_id'] = entity_id

        return pl.DataFrame([scalar_results])

    def run_batch(
        self,
        velocity_snapshots: List[Dict[str, np.ndarray]],
        entity_ids: List[str] = None
    ) -> pl.DataFrame:
        """
        Run analysis on multiple velocity field snapshots.

        Args:
            velocity_snapshots: List of velocity data dicts
            entity_ids: Optional list of identifiers (one per snapshot)

        Returns:
            DataFrame with all results concatenated
        """
        if entity_ids is None:
            entity_ids = [f"snapshot_{i}" for i in range(len(velocity_snapshots))]

        if len(entity_ids) != len(velocity_snapshots):
            raise ValueError("entity_ids length must match velocity_snapshots length")

        results = []
        for vel_data, eid in zip(velocity_snapshots, entity_ids):
            df = self.run(vel_data, entity_id=eid)
            results.append(df)

        return pl.concat(results)


def load_velocity_field_from_npy(
    u_path: Path,
    v_path: Path,
    w_path: Path
) -> Dict[str, np.ndarray]:
    """
    Load velocity field from numpy files.

    Args:
        u_path: Path to u-component .npy file
        v_path: Path to v-component .npy file
        w_path: Path to w-component .npy file

    Returns:
        Dictionary with 'u', 'v', 'w' arrays
    """
    return {
        'u': np.load(u_path),
        'v': np.load(v_path),
        'w': np.load(w_path),
    }


def create_synthetic_turbulence(
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    Re_target: float = 1000.0,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Create synthetic turbulent velocity field for testing.

    Uses random Fourier modes with k^(-5/3) energy spectrum.
    This is for testing only - not real turbulence data.

    Args:
        nx, ny, nz: Grid dimensions
        Re_target: Target Reynolds number (approximate)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'u', 'v', 'w' arrays
    """
    np.random.seed(seed)

    # Wavenumber grid
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.fftfreq(nz) * nz

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[0, 0, 0] = 1.0  # Avoid division by zero

    # Energy spectrum: E(k) ~ k^(-5/3) in inertial range
    # Scale so that integral gives desired energy
    E_k = K**(-5/3)
    E_k[0, 0, 0] = 0.0  # No energy at k=0

    # Random phases
    phase_u = np.random.uniform(0, 2*np.pi, (nx, ny, nz))
    phase_v = np.random.uniform(0, 2*np.pi, (nx, ny, nz))
    phase_w = np.random.uniform(0, 2*np.pi, (nx, ny, nz))

    # Velocity in Fourier space
    amplitude = np.sqrt(E_k / 3)  # Split energy equally among components
    u_hat = amplitude * np.exp(1j * phase_u)
    v_hat = amplitude * np.exp(1j * phase_v)
    w_hat = amplitude * np.exp(1j * phase_w)

    # Enforce divergence-free condition: k . u_hat = 0
    # Project out the compressible part
    k_dot_u = KX * u_hat + KY * v_hat + KZ * w_hat
    k_sq = K**2
    k_sq[0, 0, 0] = 1.0

    u_hat -= KX * k_dot_u / k_sq
    v_hat -= KY * k_dot_u / k_sq
    w_hat -= KZ * k_dot_u / k_sq

    # Transform to physical space
    u = np.real(np.fft.ifftn(u_hat))
    v = np.real(np.fft.ifftn(v_hat))
    w = np.real(np.fft.ifftn(w_hat))

    # Scale to target RMS velocity
    u_rms_target = Re_target * 1e-6 / (2 * np.pi)  # Rough scaling
    u_rms_current = np.sqrt(np.mean(u**2 + v**2 + w**2) / 3)

    if u_rms_current > 0:
        scale = u_rms_target / u_rms_current
        u *= scale
        v *= scale
        w *= scale

    return {'u': u, 'v': v, 'w': w}
