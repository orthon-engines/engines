"""
Pressure Drop Engine (Darcy-Weisbach)

Computes pressure drop in pipe flow using the Darcy-Weisbach equation.

THE EQUATION:
    ΔP = f × (L/D) × (ρv²/2)

Where:
    ΔP = pressure drop [Pa]
    f = Darcy friction factor (dimensionless)
    L = pipe length [m]
    D = pipe diameter [m]
    ρ = fluid density [kg/m³]
    v = flow velocity [m/s]

FRICTION FACTOR CALCULATION:
    Laminar flow (Re < 2300):
        f = 64/Re  (Hagen-Poiseuille)

    Turbulent flow (Re > 4000):
        Colebrook-White equation (implicit):
        1/√f = -2.0 log₁₀(ε/(3.7D) + 2.51/(Re√f))

        Solved iteratively or using Swamee-Jain explicit approximation:
        f = 0.25 / [log₁₀(ε/(3.7D) + 5.74/Re^0.9)]²

    Where ε = pipe roughness [m]

Outputs:
    - pressure_drop: ΔP [Pa]
    - friction_factor: f (dimensionless)
    - head_loss: h_L = ΔP/(ρg) [m]
    - power_loss: P = ΔP × Q [W]

REQUIRES from config:
    - pipe_length [m]
    - pipe_diameter [m]
    - density [kg/m³]
    - pipe_roughness [m] (optional, default 0.000045 for commercial steel)

References:
    - Darcy (1857), Weisbach (1845)
    - Colebrook (1939)
    - Swamee & Jain (1976)
"""

import numpy as np
from typing import Dict, Any, Optional, Union
import polars as pl


# Default pipe roughness values [m]
ROUGHNESS_VALUES = {
    'smooth': 0.0,
    'glass': 0.0,
    'plastic': 0.0000015,
    'pvc': 0.0000015,
    'copper': 0.0000015,
    'steel': 0.000045,
    'commercial_steel': 0.000045,
    'galvanized': 0.00015,
    'cast_iron': 0.00026,
    'concrete': 0.0003,
    'riveted_steel': 0.003,
}


def compute(
    velocity: Union[np.ndarray, float] = None,
    df: pl.DataFrame = None,
    constants: Dict[str, Any] = None,
    config: Dict[str, Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute pressure drop using Darcy-Weisbach equation.

    Args:
        velocity: Flow velocity [m/s]
        df: DataFrame with velocity signal
        constants: Physical constants dict containing:
            - pipe_length [m]
            - pipe_diameter [m]
            - density [kg/m³]
            - kinematic_viscosity [m²/s] OR (dynamic_viscosity [Pa·s])
            - pipe_roughness [m] (optional)
        config: Alternative source for constants

    Returns:
        Dict with pressure drop metrics
    """
    # Get constants
    if constants is None:
        constants = {}
    if config is not None:
        constants = {**config.get('global_constants', {}), **constants}

    # Extract required constants
    L = constants.get('pipe_length') or constants.get('length')
    D = constants.get('pipe_diameter') or constants.get('diameter') or constants.get('characteristic_length')
    rho = constants.get('density') or constants.get('fluid_density')

    # Kinematic viscosity
    nu = constants.get('kinematic_viscosity')
    if nu is None:
        mu = constants.get('dynamic_viscosity') or constants.get('viscosity')
        if mu is not None and rho is not None:
            nu = mu / rho

    # Pipe roughness (optional, with default)
    epsilon = constants.get('pipe_roughness') or constants.get('roughness')
    if epsilon is None:
        material = constants.get('pipe_material', 'steel').lower()
        epsilon = ROUGHNESS_VALUES.get(material, 0.000045)

    # Validate required constants
    missing = []
    if L is None:
        missing.append('pipe_length')
    if D is None:
        missing.append('pipe_diameter')
    if rho is None:
        missing.append('density')
    if nu is None:
        missing.append('kinematic_viscosity (or dynamic_viscosity)')

    if missing:
        return _null_result(f"Missing constants: {', '.join(missing)}")

    # Convert units if needed
    D = _convert_length(D, constants, 'diameter')
    L = _convert_length(L, constants, 'length')

    # Get velocity
    if velocity is not None:
        v = np.asarray(velocity, dtype=np.float64)
    elif df is not None:
        v = _extract_velocity(df)
        if v is None:
            return _null_result("Could not extract velocity from DataFrame")
    else:
        v_const = constants.get('velocity') or constants.get('flow_velocity')
        if v_const is not None:
            v = np.array([float(v_const)])
        else:
            return _null_result("No velocity provided")

    # Clean NaN
    v_clean = v[~np.isnan(v)] if isinstance(v, np.ndarray) else np.array([v])
    if len(v_clean) == 0:
        return _null_result("All velocity values are NaN")

    # Calculate Reynolds number
    Re = v_clean * D / nu

    # Calculate friction factor
    f = np.array([_friction_factor(re, D, epsilon) for re in Re])

    # DARCY-WEISBACH: ΔP = f × (L/D) × (ρv²/2)
    delta_P = f * (L / D) * (rho * v_clean**2 / 2)

    # Head loss: h_L = ΔP / (ρg)
    g = 9.81  # m/s²
    head_loss = delta_P / (rho * g)

    # Power loss (if flow rate known)
    Q = constants.get('flow_rate')  # m³/s
    if Q is None:
        # Calculate from velocity and pipe area
        A = np.pi * (D/2)**2
        Q = v_clean * A
    power_loss = delta_P * Q

    return {
        'pressure_drop': float(np.mean(delta_P)),
        'pressure_drop_min': float(np.min(delta_P)),
        'pressure_drop_max': float(np.max(delta_P)),
        'friction_factor': float(np.mean(f)),
        'head_loss': float(np.mean(head_loss)),
        'power_loss': float(np.mean(power_loss)),
        'reynolds_mean': float(np.mean(Re)),
        'pipe_length': float(L),
        'pipe_diameter': float(D),
        'pipe_roughness': float(epsilon),
        'relative_roughness': float(epsilon / D),
        'units': {
            'pressure_drop': 'Pa',
            'head_loss': 'm',
            'power_loss': 'W',
        },
    }


def _friction_factor(Re: float, D: float, epsilon: float) -> float:
    """
    Calculate Darcy friction factor.

    Uses:
        - Hagen-Poiseuille for laminar (Re < 2300)
        - Swamee-Jain approximation for turbulent (Re > 4000)
        - Linear interpolation for transitional
    """
    if Re <= 0:
        return 0.0

    if Re < 2300:
        # Laminar: f = 64/Re
        return 64.0 / Re

    elif Re > 4000:
        # Turbulent: Swamee-Jain explicit approximation
        # f = 0.25 / [log₁₀(ε/(3.7D) + 5.74/Re^0.9)]²
        relative_roughness = epsilon / D
        term = relative_roughness / 3.7 + 5.74 / (Re ** 0.9)

        if term <= 0:
            term = 1e-10

        f = 0.25 / (np.log10(term) ** 2)
        return f

    else:
        # Transitional: interpolate between laminar and turbulent
        f_lam = 64.0 / 2300
        Re_turb = 4000
        relative_roughness = epsilon / D
        term = relative_roughness / 3.7 + 5.74 / (Re_turb ** 0.9)
        f_turb = 0.25 / (np.log10(term) ** 2)

        # Linear interpolation
        t = (Re - 2300) / (4000 - 2300)
        return f_lam + t * (f_turb - f_lam)


def _extract_velocity(df: pl.DataFrame) -> Optional[np.ndarray]:
    """Extract velocity from DataFrame."""
    velocity_patterns = ['velocity', 'speed', 'flow_velocity', 'v_']

    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in velocity_patterns):
            return df[col].to_numpy()

    if 'signal_id' in df.columns and 'value' in df.columns:
        for pattern in velocity_patterns:
            filtered = df.filter(pl.col('signal_id').str.contains(pattern))
            if len(filtered) > 0:
                return filtered['value'].to_numpy()

    return None


def _convert_length(value: float, constants: Dict[str, Any], param: str) -> float:
    """Convert length to meters."""
    unit_key = f'{param}_unit'
    unit = constants.get(unit_key, '')

    if 'in' in str(unit).lower():
        return value * 0.0254
    elif 'ft' in str(unit).lower():
        return value * 0.3048
    elif 'cm' in str(unit).lower():
        return value * 0.01
    elif 'mm' in str(unit).lower():
        return value * 0.001

    return float(value)


def _null_result(reason: str) -> Dict[str, Any]:
    """Return null result with reason."""
    return {
        'pressure_drop': None,
        'pressure_drop_min': None,
        'pressure_drop_max': None,
        'friction_factor': None,
        'head_loss': None,
        'power_loss': None,
        'reynolds_mean': None,
        'pipe_length': None,
        'pipe_diameter': None,
        'pipe_roughness': None,
        'relative_roughness': None,
        'units': None,
        'pressure_drop_error': reason,
    }
