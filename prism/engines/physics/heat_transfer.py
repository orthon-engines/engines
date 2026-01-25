"""
Convective Heat Transfer Correlations

Heat transfer by convection:
    q = h * (T_surface - T_fluid)
    Q = h * A * ΔT

Where:
    h = convective heat transfer coefficient [W/(m²·K)]

The Nusselt number relates h to fluid properties:
    Nu = h * L / k

Where L is characteristic length and k is fluid thermal conductivity.

Key correlations implemented:
    - Dittus-Boelter (turbulent pipe flow)
    - Sieder-Tate (viscosity variation)
    - Gnielinski (transitional/turbulent)
    - Churchill-Chu (natural convection, vertical plate)
    - Flat plate (laminar/turbulent)
    - Sphere in cross-flow
"""

import numpy as np
from typing import Dict, Optional


def dittus_boelter(
    Re: float,
    Pr: float,
    heating: bool = True
) -> Dict[str, float]:
    """
    Dittus-Boelter correlation for turbulent flow in smooth tubes.

    Valid for:
        - Re > 10,000
        - 0.6 < Pr < 160
        - L/D > 10

    Nu = 0.023 * Re^0.8 * Pr^n

    Where n = 0.4 (heating) or 0.3 (cooling)

    Parameters
    ----------
    Re : float
        Reynolds number
    Pr : float
        Prandtl number
    heating : bool
        True if fluid is being heated, False if cooled

    Returns
    -------
    dict
        nusselt: Nu
        correlation: name
    """
    if Re < 10000:
        return {'nusselt': np.nan, 'warning': 'Re < 10000, use transitional correlation'}

    n = 0.4 if heating else 0.3
    Nu = 0.023 * (Re ** 0.8) * (Pr ** n)

    return {
        'nusselt': Nu,
        'correlation': 'dittus_boelter',
        'reynolds': Re,
        'prandtl': Pr,
        'exponent_n': n
    }


def sieder_tate(
    Re: float,
    Pr: float,
    mu_bulk: float,
    mu_wall: float
) -> Dict[str, float]:
    """
    Sieder-Tate correlation accounting for viscosity variation.

    Valid for:
        - Re > 10,000
        - 0.7 < Pr < 16,700
        - L/D > 10

    Nu = 0.027 * Re^0.8 * Pr^(1/3) * (μ_bulk/μ_wall)^0.14

    Parameters
    ----------
    Re : float
        Reynolds number (bulk)
    Pr : float
        Prandtl number (bulk)
    mu_bulk : float
        Dynamic viscosity at bulk temperature
    mu_wall : float
        Dynamic viscosity at wall temperature

    Returns
    -------
    dict
        nusselt: Nu
    """
    if Re < 10000:
        return {'nusselt': np.nan, 'warning': 'Re < 10000'}

    viscosity_ratio = (mu_bulk / mu_wall) ** 0.14
    Nu = 0.027 * (Re ** 0.8) * (Pr ** (1/3)) * viscosity_ratio

    return {
        'nusselt': Nu,
        'correlation': 'sieder_tate',
        'viscosity_correction': viscosity_ratio
    }


def gnielinski(
    Re: float,
    Pr: float,
    f: Optional[float] = None
) -> Dict[str, float]:
    """
    Gnielinski correlation for transitional and turbulent pipe flow.

    Valid for:
        - 3000 < Re < 5×10^6
        - 0.5 < Pr < 2000

    Nu = (f/8)(Re - 1000)Pr / [1 + 12.7(f/8)^0.5(Pr^(2/3) - 1)]

    Uses Petukhov friction factor if not provided:
        f = (0.790 ln(Re) - 1.64)^(-2)

    Parameters
    ----------
    Re : float
        Reynolds number
    Pr : float
        Prandtl number
    f : float, optional
        Darcy friction factor (computed if not provided)

    Returns
    -------
    dict
        nusselt: Nu
        friction_factor: f
    """
    if Re < 3000:
        return {'nusselt': np.nan, 'warning': 'Re < 3000, flow is laminar'}

    # Petukhov friction factor
    if f is None:
        f = (0.790 * np.log(Re) - 1.64) ** (-2)

    numerator = (f / 8) * (Re - 1000) * Pr
    denominator = 1 + 12.7 * np.sqrt(f / 8) * (Pr ** (2/3) - 1)
    Nu = numerator / denominator

    return {
        'nusselt': Nu,
        'correlation': 'gnielinski',
        'friction_factor': f,
        'reynolds': Re,
        'prandtl': Pr
    }


def laminar_pipe(
    constant_wall_temp: bool = True
) -> Dict[str, float]:
    """
    Fully developed laminar flow in a circular pipe.

    Nu = 3.66 (constant wall temperature)
    Nu = 4.36 (constant heat flux)

    Valid for Re < 2300, fully developed flow.

    Parameters
    ----------
    constant_wall_temp : bool
        True for constant T_wall, False for constant q"

    Returns
    -------
    dict
        nusselt: Nu
    """
    Nu = 3.66 if constant_wall_temp else 4.36

    return {
        'nusselt': Nu,
        'correlation': 'laminar_pipe',
        'boundary_condition': 'constant_T' if constant_wall_temp else 'constant_q'
    }


def flat_plate_laminar(
    Re_x: float,
    Pr: float
) -> Dict[str, float]:
    """
    Laminar boundary layer on a flat plate.

    Local Nusselt number:
        Nu_x = 0.332 * Re_x^0.5 * Pr^(1/3)

    Average Nusselt number (from leading edge):
        Nu_L = 0.664 * Re_L^0.5 * Pr^(1/3)

    Valid for:
        - Re_x < 5×10^5
        - Pr ≥ 0.6

    Parameters
    ----------
    Re_x : float
        Reynolds number at position x (or L for average)
    Pr : float
        Prandtl number

    Returns
    -------
    dict
        nusselt_local: Nu_x
        nusselt_average: Nu_avg
    """
    Nu_local = 0.332 * np.sqrt(Re_x) * (Pr ** (1/3))
    Nu_avg = 0.664 * np.sqrt(Re_x) * (Pr ** (1/3))  # = 2 * Nu_local at same x

    return {
        'nusselt_local': Nu_local,
        'nusselt_average': Nu_avg,
        'correlation': 'flat_plate_laminar',
        'reynolds': Re_x,
        'prandtl': Pr
    }


def flat_plate_turbulent(
    Re_x: float,
    Pr: float
) -> Dict[str, float]:
    """
    Turbulent boundary layer on a flat plate.

    Local Nusselt number:
        Nu_x = 0.0296 * Re_x^0.8 * Pr^(1/3)

    Average (assuming turbulent from leading edge):
        Nu_L = 0.037 * Re_L^0.8 * Pr^(1/3)

    Valid for:
        - 5×10^5 < Re_x < 10^7
        - 0.6 < Pr < 60

    Parameters
    ----------
    Re_x : float
        Reynolds number
    Pr : float
        Prandtl number

    Returns
    -------
    dict
        nusselt_local: Nu_x
        nusselt_average: Nu_avg
    """
    Nu_local = 0.0296 * (Re_x ** 0.8) * (Pr ** (1/3))
    Nu_avg = 0.037 * (Re_x ** 0.8) * (Pr ** (1/3))

    return {
        'nusselt_local': Nu_local,
        'nusselt_average': Nu_avg,
        'correlation': 'flat_plate_turbulent',
        'reynolds': Re_x,
        'prandtl': Pr
    }


def churchill_chu_vertical_plate(
    Ra: float,
    Pr: float
) -> Dict[str, float]:
    """
    Churchill-Chu correlation for natural convection on vertical plate.

    Valid for all Ra:

    Nu = {0.825 + 0.387 * Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27)}^2

    Parameters
    ----------
    Ra : float
        Rayleigh number = Gr × Pr
    Pr : float
        Prandtl number

    Returns
    -------
    dict
        nusselt: Nu
    """
    if Ra < 0:
        return {'nusselt': np.nan, 'warning': 'Ra must be positive'}

    bracket = (1 + (0.492 / Pr) ** (9/16)) ** (8/27)
    Nu = (0.825 + 0.387 * (Ra ** (1/6)) / bracket) ** 2

    return {
        'nusselt': Nu,
        'correlation': 'churchill_chu_vertical',
        'rayleigh': Ra,
        'prandtl': Pr
    }


def sphere_crossflow(
    Re: float,
    Pr: float,
    mu_inf: float,
    mu_s: float
) -> Dict[str, float]:
    """
    Whitaker correlation for sphere in cross-flow.

    Nu = 2 + (0.4*Re^0.5 + 0.06*Re^(2/3)) * Pr^0.4 * (μ_∞/μ_s)^0.25

    Valid for:
        - 3.5 < Re < 7.6×10^4
        - 0.71 < Pr < 380
        - 1.0 < μ_∞/μ_s < 3.2

    Parameters
    ----------
    Re : float
        Reynolds number (based on diameter)
    Pr : float
        Prandtl number
    mu_inf : float
        Dynamic viscosity at free-stream temperature
    mu_s : float
        Dynamic viscosity at surface temperature

    Returns
    -------
    dict
        nusselt: Nu
    """
    viscosity_ratio = (mu_inf / mu_s) ** 0.25
    Nu = 2 + (0.4 * np.sqrt(Re) + 0.06 * (Re ** (2/3))) * (Pr ** 0.4) * viscosity_ratio

    return {
        'nusselt': Nu,
        'correlation': 'whitaker_sphere',
        'reynolds': Re,
        'prandtl': Pr,
        'viscosity_ratio': viscosity_ratio
    }


def compute_h_from_nusselt(
    Nu: float,
    k: float,
    L: float
) -> Dict[str, float]:
    """
    Convert Nusselt number to heat transfer coefficient.

    h = Nu * k / L

    Parameters
    ----------
    Nu : float
        Nusselt number
    k : float
        Thermal conductivity of fluid [W/(m·K)]
    L : float
        Characteristic length [m]

    Returns
    -------
    dict
        h: heat transfer coefficient [W/(m²·K)]
    """
    h = Nu * k / L

    return {
        'h': h,
        'nusselt': Nu,
        'thermal_conductivity': k,
        'characteristic_length': L
    }


def overall_heat_transfer_coefficient(
    h_hot: float,
    h_cold: float,
    wall_thickness: float,
    k_wall: float,
    fouling_hot: float = 0.0,
    fouling_cold: float = 0.0
) -> Dict[str, float]:
    """
    Overall heat transfer coefficient for heat exchanger.

    For flat plate (per unit area):
    1/U = 1/h_hot + R_f_hot + L/k + R_f_cold + 1/h_cold

    Parameters
    ----------
    h_hot : float
        Hot side convection coefficient [W/(m²·K)]
    h_cold : float
        Cold side convection coefficient [W/(m²·K)]
    wall_thickness : float
        Wall thickness [m]
    k_wall : float
        Wall thermal conductivity [W/(m·K)]
    fouling_hot : float
        Hot side fouling resistance [m²·K/W]
    fouling_cold : float
        Cold side fouling resistance [m²·K/W]

    Returns
    -------
    dict
        U: overall coefficient [W/(m²·K)]
        resistances: breakdown of thermal resistances
    """
    R_conv_hot = 1 / h_hot
    R_conv_cold = 1 / h_cold
    R_wall = wall_thickness / k_wall

    R_total = R_conv_hot + fouling_hot + R_wall + fouling_cold + R_conv_cold
    U = 1 / R_total

    return {
        'U': U,
        'R_total': R_total,
        'resistances': {
            'convection_hot': R_conv_hot,
            'fouling_hot': fouling_hot,
            'wall': R_wall,
            'fouling_cold': fouling_cold,
            'convection_cold': R_conv_cold
        },
        'controlling_resistance': max(
            ('convection_hot', R_conv_hot),
            ('convection_cold', R_conv_cold),
            ('wall', R_wall),
            key=lambda x: x[1]
        )[0]
    }
