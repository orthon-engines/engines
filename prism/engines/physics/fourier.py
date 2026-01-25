"""
Fourier's Law of Heat Conduction

The fundamental law governing heat transfer by conduction:

    q = -k * (dT/dx)

Where:
    q = heat flux [W/m²]
    k = thermal conductivity [W/(m·K)]
    dT/dx = temperature gradient [K/m]

For steady-state 1D conduction through a slab:
    Q = k * A * ΔT / L

Where:
    Q = heat transfer rate [W]
    A = cross-sectional area [m²]
    ΔT = temperature difference [K]
    L = thickness [m]

Thermal resistance (conduction):
    R_cond = L / (k * A) [K/W]

For cylindrical geometry:
    Q = 2π * k * L * ΔT / ln(r_outer/r_inner)
    R_cond = ln(r_outer/r_inner) / (2π * k * L)

For spherical geometry:
    Q = 4π * k * r_inner * r_outer * ΔT / (r_outer - r_inner)
"""

import numpy as np
from typing import Dict, Any, Optional


def compute_heat_flux(
    thermal_conductivity: float,
    temperature_gradient: float
) -> Dict[str, float]:
    """
    Compute heat flux from Fourier's law.

    Parameters
    ----------
    thermal_conductivity : float
        k [W/(m·K)]
    temperature_gradient : float
        dT/dx [K/m] (positive = temperature increasing in x direction)

    Returns
    -------
    dict
        heat_flux: q [W/m²] (negative = heat flows opposite to gradient)
    """
    q = -thermal_conductivity * temperature_gradient

    return {
        'heat_flux': q,
        'thermal_conductivity': thermal_conductivity,
        'temperature_gradient': temperature_gradient
    }


def compute_conduction_slab(
    thermal_conductivity: float,
    area: float,
    temperature_difference: float,
    thickness: float
) -> Dict[str, float]:
    """
    Steady-state heat conduction through a plane slab.

    Parameters
    ----------
    thermal_conductivity : float
        k [W/(m·K)]
    area : float
        A [m²]
    temperature_difference : float
        ΔT = T_hot - T_cold [K]
    thickness : float
        L [m]

    Returns
    -------
    dict
        heat_rate: Q [W]
        thermal_resistance: R [K/W]
        heat_flux: q [W/m²]
    """
    if thickness <= 0:
        return {'heat_rate': np.nan, 'thermal_resistance': np.nan, 'heat_flux': np.nan}

    Q = thermal_conductivity * area * temperature_difference / thickness
    R = thickness / (thermal_conductivity * area)
    q = Q / area

    return {
        'heat_rate': Q,
        'thermal_resistance': R,
        'heat_flux': q,
        'thermal_conductivity': thermal_conductivity,
        'area': area,
        'temperature_difference': temperature_difference,
        'thickness': thickness
    }


def compute_conduction_cylinder(
    thermal_conductivity: float,
    length: float,
    temperature_difference: float,
    r_inner: float,
    r_outer: float
) -> Dict[str, float]:
    """
    Steady-state radial heat conduction through a cylindrical shell.

    Parameters
    ----------
    thermal_conductivity : float
        k [W/(m·K)]
    length : float
        L [m] (axial length)
    temperature_difference : float
        ΔT = T_inner - T_outer [K]
    r_inner : float
        Inner radius [m]
    r_outer : float
        Outer radius [m]

    Returns
    -------
    dict
        heat_rate: Q [W]
        thermal_resistance: R [K/W]
    """
    if r_inner <= 0 or r_outer <= r_inner:
        return {'heat_rate': np.nan, 'thermal_resistance': np.nan}

    ln_ratio = np.log(r_outer / r_inner)
    Q = 2 * np.pi * thermal_conductivity * length * temperature_difference / ln_ratio
    R = ln_ratio / (2 * np.pi * thermal_conductivity * length)

    return {
        'heat_rate': Q,
        'thermal_resistance': R,
        'r_inner': r_inner,
        'r_outer': r_outer,
        'length': length
    }


def compute_conduction_sphere(
    thermal_conductivity: float,
    temperature_difference: float,
    r_inner: float,
    r_outer: float
) -> Dict[str, float]:
    """
    Steady-state radial heat conduction through a spherical shell.

    Parameters
    ----------
    thermal_conductivity : float
        k [W/(m·K)]
    temperature_difference : float
        ΔT = T_inner - T_outer [K]
    r_inner : float
        Inner radius [m]
    r_outer : float
        Outer radius [m]

    Returns
    -------
    dict
        heat_rate: Q [W]
        thermal_resistance: R [K/W]
    """
    if r_inner <= 0 or r_outer <= r_inner:
        return {'heat_rate': np.nan, 'thermal_resistance': np.nan}

    Q = 4 * np.pi * thermal_conductivity * r_inner * r_outer * temperature_difference / (r_outer - r_inner)
    R = (r_outer - r_inner) / (4 * np.pi * thermal_conductivity * r_inner * r_outer)

    return {
        'heat_rate': Q,
        'thermal_resistance': R,
        'r_inner': r_inner,
        'r_outer': r_outer
    }


def compute_composite_wall(
    layers: list,
    area: float,
    T_hot: float,
    T_cold: float,
    h_hot: Optional[float] = None,
    h_cold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Heat transfer through composite wall with multiple layers.

    Parameters
    ----------
    layers : list of dict
        Each dict: {'k': thermal_conductivity [W/(m·K)], 'L': thickness [m]}
    area : float
        A [m²]
    T_hot : float
        Hot side temperature [K]
    T_cold : float
        Cold side temperature [K]
    h_hot : float, optional
        Hot side convection coefficient [W/(m²·K)]
    h_cold : float, optional
        Cold side convection coefficient [W/(m²·K)]

    Returns
    -------
    dict
        heat_rate: Q [W]
        total_resistance: R_total [K/W]
        layer_resistances: list of R [K/W]
        interface_temperatures: list of T [K]
    """
    resistances = []

    # Convection resistance on hot side
    if h_hot is not None and h_hot > 0:
        R_conv_hot = 1 / (h_hot * area)
        resistances.append(('convection_hot', R_conv_hot))

    # Conduction resistances
    for i, layer in enumerate(layers):
        k = layer['k']
        L = layer['L']
        R = L / (k * area)
        resistances.append((f'layer_{i}', R))

    # Convection resistance on cold side
    if h_cold is not None and h_cold > 0:
        R_conv_cold = 1 / (h_cold * area)
        resistances.append(('convection_cold', R_conv_cold))

    R_total = sum(r for _, r in resistances)
    Q = (T_hot - T_cold) / R_total

    # Compute interface temperatures
    temperatures = [T_hot]
    T_current = T_hot
    for name, R in resistances:
        T_current = T_current - Q * R
        temperatures.append(T_current)

    return {
        'heat_rate': Q,
        'total_resistance': R_total,
        'layer_resistances': dict(resistances),
        'interface_temperatures': temperatures,
        'overall_U': 1 / (R_total * area) if R_total > 0 else np.nan  # Overall heat transfer coefficient
    }
