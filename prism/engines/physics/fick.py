"""
Fick's Laws of Diffusion

Fick's First Law (steady-state diffusion):
    J = -D * (dC/dx)

Where:
    J = molar flux [mol/(m²·s)]
    D = diffusion coefficient [m²/s]
    dC/dx = concentration gradient [mol/m⁴]

Fick's Second Law (transient diffusion):
    ∂C/∂t = D * ∂²C/∂x²

Analogous to Fourier's law for heat transfer:
    Heat: q = -k * (dT/dx)
    Mass: J = -D * (dC/dx)

Key diffusivity correlations:
    - Wilke-Chang (liquids)
    - Chapman-Enskog (gases)
    - Stokes-Einstein (dilute solutions)
"""

import numpy as np
from typing import Dict, Optional


def compute_molar_flux(
    diffusivity: float,
    concentration_gradient: float
) -> Dict[str, float]:
    """
    Fick's first law: molar flux from concentration gradient.

    J = -D * (dC/dx)

    Parameters
    ----------
    diffusivity : float
        D [m²/s]
    concentration_gradient : float
        dC/dx [mol/m⁴]

    Returns
    -------
    dict
        molar_flux: J [mol/(m²·s)]
    """
    J = -diffusivity * concentration_gradient

    return {
        'molar_flux': J,
        'diffusivity': diffusivity,
        'concentration_gradient': concentration_gradient
    }


def compute_mass_transfer_slab(
    diffusivity: float,
    area: float,
    concentration_difference: float,
    thickness: float
) -> Dict[str, float]:
    """
    Steady-state diffusion through a stagnant film/slab.

    Analogous to heat conduction through slab.

    N = D * A * ΔC / L

    Parameters
    ----------
    diffusivity : float
        D [m²/s]
    area : float
        A [m²]
    concentration_difference : float
        ΔC = C_high - C_low [mol/m³]
    thickness : float
        L [m] (film thickness)

    Returns
    -------
    dict
        molar_rate: N [mol/s]
        mass_transfer_resistance: R [s/m³]
    """
    if thickness <= 0:
        return {'molar_rate': np.nan, 'mass_transfer_resistance': np.nan}

    N = diffusivity * area * concentration_difference / thickness
    R = thickness / (diffusivity * area)  # Mass transfer resistance

    return {
        'molar_rate': N,
        'mass_transfer_resistance': R,
        'molar_flux': N / area,
        'diffusivity': diffusivity,
        'thickness': thickness
    }


def equimolar_counterdiffusion(
    diffusivity: float,
    total_pressure: float,
    temperature: float,
    partial_pressure_1: float,
    partial_pressure_2: float,
    thickness: float
) -> Dict[str, float]:
    """
    Equimolar counterdiffusion of ideal gases.

    For gases A and B diffusing in opposite directions:
    N_A = -N_B = (D * P_total) / (R * T * L) * (p_A1 - p_A2)

    Parameters
    ----------
    diffusivity : float
        D_AB [m²/s]
    total_pressure : float
        P_total [Pa]
    temperature : float
        T [K]
    partial_pressure_1 : float
        p_A at position 1 [Pa]
    partial_pressure_2 : float
        p_A at position 2 [Pa]
    thickness : float
        L [m]

    Returns
    -------
    dict
        molar_flux: N_A [mol/(m²·s)]
    """
    R = 8.314  # J/(mol·K)

    N_A = (diffusivity * total_pressure / (R * temperature * thickness)) * (partial_pressure_1 - partial_pressure_2)

    return {
        'molar_flux': N_A,
        'diffusivity': diffusivity,
        'total_pressure': total_pressure,
        'temperature': temperature
    }


def diffusion_through_stagnant_film(
    diffusivity: float,
    total_pressure: float,
    temperature: float,
    partial_pressure_A1: float,
    partial_pressure_A2: float,
    thickness: float
) -> Dict[str, float]:
    """
    Diffusion of A through stagnant B (Stefan diffusion).

    Used for evaporation into still air, etc.

    N_A = (D * P) / (R * T * L) * ln[(P - p_A2)/(P - p_A1)]

    Or using log-mean:
    N_A = (D * P) / (R * T * L * p_B_lm) * (p_A1 - p_A2)

    Parameters
    ----------
    diffusivity : float
        D_AB [m²/s]
    total_pressure : float
        P [Pa]
    temperature : float
        T [K]
    partial_pressure_A1 : float
        p_A at interface [Pa]
    partial_pressure_A2 : float
        p_A in bulk [Pa]
    thickness : float
        Film thickness L [m]

    Returns
    -------
    dict
        molar_flux: N_A [mol/(m²·s)]
        enhancement_factor: ratio vs equimolar
    """
    R = 8.314  # J/(mol·K)

    p_B1 = total_pressure - partial_pressure_A1
    p_B2 = total_pressure - partial_pressure_A2

    if p_B1 <= 0 or p_B2 <= 0:
        return {'molar_flux': np.nan, 'warning': 'Invalid partial pressures'}

    # Log-mean partial pressure of B
    p_B_lm = (p_B2 - p_B1) / np.log(p_B2 / p_B1) if abs(p_B2 - p_B1) > 1e-10 else p_B1

    N_A = (diffusivity * total_pressure / (R * temperature * thickness * p_B_lm)) * (partial_pressure_A1 - partial_pressure_A2)

    # Enhancement compared to equimolar counterdiffusion
    N_equimolar = (diffusivity * total_pressure / (R * temperature * thickness)) * (partial_pressure_A1 - partial_pressure_A2)
    enhancement = N_A / N_equimolar if N_equimolar != 0 else 1.0

    return {
        'molar_flux': N_A,
        'p_B_log_mean': p_B_lm,
        'enhancement_factor': enhancement
    }


def wilke_chang(
    temperature: float,
    solvent_viscosity: float,
    solute_molar_volume: float,
    solvent_molecular_weight: float,
    association_factor: float = 1.0
) -> Dict[str, float]:
    """
    Wilke-Chang correlation for liquid diffusivity.

    D_AB = 7.4×10⁻⁸ * (φ*M_B)^0.5 * T / (μ_B * V_A^0.6)

    Parameters
    ----------
    temperature : float
        T [K]
    solvent_viscosity : float
        μ_B [cP]
    solute_molar_volume : float
        V_A at normal boiling point [cm³/mol]
    solvent_molecular_weight : float
        M_B [g/mol]
    association_factor : float
        φ: 2.6 for water, 1.9 for methanol, 1.5 for ethanol, 1.0 for non-associated

    Returns
    -------
    dict
        diffusivity: D_AB [cm²/s]
        diffusivity_SI: D_AB [m²/s]
    """
    D_AB = 7.4e-8 * ((association_factor * solvent_molecular_weight) ** 0.5) * temperature / (
        solvent_viscosity * (solute_molar_volume ** 0.6)
    )

    return {
        'diffusivity_cgs': D_AB,
        'diffusivity': D_AB * 1e-4,  # Convert cm²/s to m²/s
        'correlation': 'wilke_chang',
        'association_factor': association_factor
    }


def chapman_enskog(
    temperature: float,
    pressure: float,
    molecular_weight_A: float,
    molecular_weight_B: float,
    sigma_AB: float,
    omega_D: float
) -> Dict[str, float]:
    """
    Chapman-Enskog equation for gas diffusivity.

    D_AB = 1.858×10⁻³ * T^1.5 / (P * σ_AB² * Ω_D) * [(1/M_A + 1/M_B)]^0.5

    Parameters
    ----------
    temperature : float
        T [K]
    pressure : float
        P [atm]
    molecular_weight_A : float
        M_A [g/mol]
    molecular_weight_B : float
        M_B [g/mol]
    sigma_AB : float
        Collision diameter [Å]
    omega_D : float
        Collision integral (dimensionless, typically 0.5-2.5)

    Returns
    -------
    dict
        diffusivity: D_AB [cm²/s]
        diffusivity_SI: D_AB [m²/s]
    """
    M_term = np.sqrt(1/molecular_weight_A + 1/molecular_weight_B)
    D_AB = 1.858e-3 * (temperature ** 1.5) * M_term / (pressure * (sigma_AB ** 2) * omega_D)

    return {
        'diffusivity_cgs': D_AB,
        'diffusivity': D_AB * 1e-4,  # Convert to m²/s
        'correlation': 'chapman_enskog'
    }


def stokes_einstein(
    temperature: float,
    viscosity: float,
    particle_radius: float
) -> Dict[str, float]:
    """
    Stokes-Einstein equation for diffusivity of large molecules/particles.

    D = k_B * T / (6π * μ * r)

    Parameters
    ----------
    temperature : float
        T [K]
    viscosity : float
        μ [Pa·s]
    particle_radius : float
        r [m]

    Returns
    -------
    dict
        diffusivity: D [m²/s]
    """
    k_B = 1.380649e-23  # Boltzmann constant [J/K]

    D = k_B * temperature / (6 * np.pi * viscosity * particle_radius)

    return {
        'diffusivity': D,
        'correlation': 'stokes_einstein',
        'temperature': temperature,
        'particle_radius': particle_radius
    }


def penetration_depth(
    diffusivity: float,
    time: float
) -> Dict[str, float]:
    """
    Characteristic penetration depth for transient diffusion.

    δ = √(D*t)  or  δ = √(4*D*t) for 99% penetration

    From solution to Fick's second law with semi-infinite medium.

    Parameters
    ----------
    diffusivity : float
        D [m²/s]
    time : float
        t [s]

    Returns
    -------
    dict
        penetration_depth: δ [m]
        fourier_number: Fo = D*t/δ² (should be ~1)
    """
    delta = np.sqrt(diffusivity * time)
    delta_99 = np.sqrt(4 * diffusivity * time)  # 99% of equilibrium

    return {
        'penetration_depth': delta,
        'penetration_depth_99pct': delta_99,
        'diffusivity': diffusivity,
        'time': time
    }
