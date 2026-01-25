"""
Reynolds Number Engine

Computes Reynolds number - the dimensionless ratio of inertial to viscous forces.

THE EQUATION:
    Re = ρvL/μ = vL/ν

Where:
    ρ = density [kg/m³]
    v = velocity [m/s]
    L = characteristic length [m]
    μ = dynamic viscosity [Pa·s = kg/(m·s)]
    ν = kinematic viscosity [m²/s] (ν = μ/ρ)

Outputs:
    - reynolds: Reynolds number (dimensionless)
    - reynolds_mean: Mean Re over time series
    - reynolds_max: Maximum Re
    - flow_regime: 'laminar' (<2300), 'transitional' (2300-4000), 'turbulent' (>4000)
                   Note: thresholds are for pipe flow. Other geometries differ.

REQUIRES from config:
    One of:
        - density + dynamic_viscosity
        - kinematic_viscosity
    AND:
        - characteristic_length (pipe diameter, chord length, etc.)

    Velocity from signals or config.

References:
    - Osborne Reynolds (1883)
    - Standard fluid mechanics (White, Munson, etc.)
"""

import numpy as np
from typing import Dict, Any, Optional, Union
import polars as pl


def compute(
    velocity: Union[np.ndarray, float] = None,
    df: pl.DataFrame = None,
    constants: Dict[str, Any] = None,
    config: Dict[str, Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute Reynolds number.

    Args:
        velocity: Velocity array [m/s] or scalar
        df: DataFrame with velocity signal
        constants: Physical constants dict containing:
            - density [kg/m³] AND dynamic_viscosity [Pa·s]
            - OR kinematic_viscosity [m²/s]
            - AND characteristic_length [m]
        config: Alternative source for constants (checks global_constants)

    Returns:
        Dict with Reynolds number metrics
    """
    # Get constants from config if not provided directly
    if constants is None:
        constants = {}
    if config is not None:
        constants = {**config.get('global_constants', {}), **constants}

    # Extract required constants
    L = constants.get('characteristic_length') or constants.get('diameter') or constants.get('pipe_diameter')

    # Get kinematic viscosity (directly or computed from density/dynamic_viscosity)
    nu = constants.get('kinematic_viscosity')

    if nu is None:
        rho = constants.get('density') or constants.get('fluid_density')
        mu = constants.get('dynamic_viscosity') or constants.get('viscosity')

        if rho is not None and mu is not None:
            nu = mu / rho

    # Validate we have what we need
    if L is None:
        return _null_result("Missing characteristic_length (or diameter) in constants")

    if nu is None:
        return _null_result("Missing kinematic_viscosity OR (density AND dynamic_viscosity) in constants")

    # Convert units if needed (handle common cases)
    L = _convert_length(L, constants)
    nu = _convert_viscosity(nu, constants)

    # Get velocity
    if velocity is not None:
        v = np.asarray(velocity, dtype=np.float64)
    elif df is not None:
        v = _extract_velocity(df)
        if v is None:
            return _null_result("Could not extract velocity from DataFrame")
    else:
        # Check for velocity in constants
        v_const = constants.get('velocity') or constants.get('flow_velocity')
        if v_const is not None:
            v = np.array([float(v_const)])
        else:
            return _null_result("No velocity provided")

    # Handle NaN
    v_clean = v[~np.isnan(v)] if isinstance(v, np.ndarray) else np.array([v])

    if len(v_clean) == 0:
        return _null_result("All velocity values are NaN")

    # THE CALCULATION: Re = vL/ν
    Re = v_clean * L / nu

    # Flow regime classification (pipe flow thresholds)
    Re_mean = float(np.mean(Re))

    if Re_mean < 2300:
        regime = 'laminar'
    elif Re_mean < 4000:
        regime = 'transitional'
    else:
        regime = 'turbulent'

    return {
        'reynolds': Re.tolist() if len(Re) > 1 else float(Re[0]),
        'reynolds_mean': Re_mean,
        'reynolds_min': float(np.min(Re)),
        'reynolds_max': float(np.max(Re)),
        'reynolds_std': float(np.std(Re)) if len(Re) > 1 else 0.0,
        'flow_regime': regime,
        'characteristic_length': float(L),
        'kinematic_viscosity': float(nu),
        'is_laminar': Re_mean < 2300,
        'is_turbulent': Re_mean > 4000,
    }


def _extract_velocity(df: pl.DataFrame) -> Optional[np.ndarray]:
    """Extract velocity from DataFrame."""
    # Look for velocity column
    velocity_patterns = ['velocity', 'speed', 'flow_velocity', 'v_']

    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in velocity_patterns):
            return df[col].to_numpy()

    # Look for 'value' column with signal_id containing velocity
    if 'signal_id' in df.columns and 'value' in df.columns:
        for pattern in velocity_patterns:
            filtered = df.filter(pl.col('signal_id').str.contains(pattern))
            if len(filtered) > 0:
                return filtered['value'].to_numpy()

    return None


def _convert_length(L: float, constants: Dict[str, Any]) -> float:
    """Convert length to meters if unit hints present."""
    unit = constants.get('length_unit') or constants.get('diameter_unit', '')

    if 'in' in str(unit).lower() or 'inch' in str(unit).lower():
        return L * 0.0254  # inches to meters
    elif 'ft' in str(unit).lower() or 'feet' in str(unit).lower():
        return L * 0.3048  # feet to meters
    elif 'cm' in str(unit).lower():
        return L * 0.01  # cm to meters
    elif 'mm' in str(unit).lower():
        return L * 0.001  # mm to meters

    return float(L)  # Assume SI (meters)


def _convert_viscosity(nu: float, constants: Dict[str, Any]) -> float:
    """Convert kinematic viscosity to m²/s if unit hints present."""
    unit = constants.get('viscosity_unit', '')

    if 'cst' in str(unit).lower() or 'centistokes' in str(unit).lower():
        return nu * 1e-6  # centistokes to m²/s
    elif 'st' in str(unit).lower() or 'stokes' in str(unit).lower():
        return nu * 1e-4  # stokes to m²/s

    return float(nu)  # Assume SI (m²/s)


def _null_result(reason: str) -> Dict[str, Any]:
    """Return null result with reason."""
    return {
        'reynolds': None,
        'reynolds_mean': None,
        'reynolds_min': None,
        'reynolds_max': None,
        'reynolds_std': None,
        'flow_regime': None,
        'characteristic_length': None,
        'kinematic_viscosity': None,
        'is_laminar': None,
        'is_turbulent': None,
        'reynolds_error': reason,
    }
