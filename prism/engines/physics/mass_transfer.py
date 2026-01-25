"""
Convective Mass Transfer Correlations

Mass transfer by convection:
    N_A = k_c * (C_As - C_A∞)

Where:
    k_c = mass transfer coefficient [m/s]

The Sherwood number relates k_c to molecular diffusion:
    Sh = k_c * L / D_AB

Analogous to heat transfer:
    Heat: Nu = h*L/k
    Mass: Sh = k_c*L/D

Key correlations implemented:
    - Dittus-Boelter analog (turbulent pipe)
    - Flat plate (laminar/turbulent)
    - Sphere in cross-flow (Froessling/Ranz-Marshall)
    - Packed bed
    - Falling film
"""

import numpy as np
from typing import Dict, Optional


def chilton_colburn_analogy(
    Nu: Optional[float] = None,
    Sh: Optional[float] = None,
    Re: float = None,
    Pr: float = None,
    Sc: float = None,
    f: float = None
) -> Dict[str, float]:
    """
    Chilton-Colburn analogy relating heat, mass, and momentum transfer.

    j_H = j_D = f/2

    Where:
        j_H = St_H * Pr^(2/3) = (Nu/Re*Pr) * Pr^(2/3)  [heat transfer]
        j_D = St_D * Sc^(2/3) = (Sh/Re*Sc) * Sc^(2/3)  [mass transfer]
        f = friction factor

    This allows conversion between heat and mass transfer correlations.

    Parameters
    ----------
    Nu : float, optional
        Nusselt number (if computing Sh)
    Sh : float, optional
        Sherwood number (if computing Nu)
    Re : float
        Reynolds number
    Pr : float
        Prandtl number
    Sc : float
        Schmidt number
    f : float, optional
        Friction factor

    Returns
    -------
    dict
        Converted quantities
    """
    results = {}

    if Nu is not None and Pr is not None and Sc is not None:
        # Convert Nu to Sh
        # Nu/Pr^(1/3) = Sh/Sc^(1/3)  (for same geometry)
        Sh_calc = Nu * (Sc / Pr) ** (1/3)
        results['sherwood_from_nusselt'] = Sh_calc
        results['j_H'] = (Nu / (Re * Pr)) * (Pr ** (2/3)) if Re else np.nan

    if Sh is not None and Pr is not None and Sc is not None:
        # Convert Sh to Nu
        Nu_calc = Sh * (Pr / Sc) ** (1/3)
        results['nusselt_from_sherwood'] = Nu_calc
        results['j_D'] = (Sh / (Re * Sc)) * (Sc ** (2/3)) if Re else np.nan

    if f is not None:
        results['j_factor'] = f / 2
        if Re and Pr:
            results['Nu_from_f'] = (f / 2) * Re * Pr / (Pr ** (2/3))
        if Re and Sc:
            results['Sh_from_f'] = (f / 2) * Re * Sc / (Sc ** (2/3))

    return results


def pipe_turbulent(
    Re: float,
    Sc: float
) -> Dict[str, float]:
    """
    Sherwood number for turbulent flow in pipes (Dittus-Boelter analog).

    Sh = 0.023 * Re^0.8 * Sc^(1/3)

    Valid for:
        - Re > 10,000
        - 0.6 < Sc < 3000
        - L/D > 10

    Parameters
    ----------
    Re : float
        Reynolds number
    Sc : float
        Schmidt number

    Returns
    -------
    dict
        sherwood: Sh
    """
    if Re < 10000:
        return {'sherwood': np.nan, 'warning': 'Re < 10000, not turbulent'}

    Sh = 0.023 * (Re ** 0.8) * (Sc ** (1/3))

    return {
        'sherwood': Sh,
        'correlation': 'pipe_turbulent',
        'reynolds': Re,
        'schmidt': Sc
    }


def gilliland_sherwood(
    Re: float,
    Sc: float
) -> Dict[str, float]:
    """
    Gilliland-Sherwood correlation for mass transfer in pipes.

    Sh = 0.023 * Re^0.83 * Sc^0.44

    Valid for:
        - 2000 < Re < 35000
        - 0.6 < Sc < 2.5

    Parameters
    ----------
    Re : float
        Reynolds number
    Sc : float
        Schmidt number

    Returns
    -------
    dict
        sherwood: Sh
    """
    Sh = 0.023 * (Re ** 0.83) * (Sc ** 0.44)

    return {
        'sherwood': Sh,
        'correlation': 'gilliland_sherwood',
        'reynolds': Re,
        'schmidt': Sc
    }


def flat_plate_laminar(
    Re_x: float,
    Sc: float
) -> Dict[str, float]:
    """
    Laminar boundary layer mass transfer on flat plate.

    Local: Sh_x = 0.332 * Re_x^0.5 * Sc^(1/3)
    Average: Sh_L = 0.664 * Re_L^0.5 * Sc^(1/3)

    Analogous to heat transfer correlation with Sc replacing Pr.

    Parameters
    ----------
    Re_x : float
        Reynolds number at position x
    Sc : float
        Schmidt number

    Returns
    -------
    dict
        sherwood_local: Sh_x
        sherwood_average: Sh_avg
    """
    Sh_local = 0.332 * np.sqrt(Re_x) * (Sc ** (1/3))
    Sh_avg = 0.664 * np.sqrt(Re_x) * (Sc ** (1/3))

    return {
        'sherwood_local': Sh_local,
        'sherwood_average': Sh_avg,
        'correlation': 'flat_plate_laminar',
        'reynolds': Re_x,
        'schmidt': Sc
    }


def flat_plate_turbulent(
    Re_x: float,
    Sc: float
) -> Dict[str, float]:
    """
    Turbulent boundary layer mass transfer on flat plate.

    Local: Sh_x = 0.0296 * Re_x^0.8 * Sc^(1/3)
    Average: Sh_L = 0.037 * Re_L^0.8 * Sc^(1/3)

    Parameters
    ----------
    Re_x : float
        Reynolds number
    Sc : float
        Schmidt number

    Returns
    -------
    dict
        sherwood_local: Sh_x
        sherwood_average: Sh_avg
    """
    Sh_local = 0.0296 * (Re_x ** 0.8) * (Sc ** (1/3))
    Sh_avg = 0.037 * (Re_x ** 0.8) * (Sc ** (1/3))

    return {
        'sherwood_local': Sh_local,
        'sherwood_average': Sh_avg,
        'correlation': 'flat_plate_turbulent',
        'reynolds': Re_x,
        'schmidt': Sc
    }


def froessling(
    Re: float,
    Sc: float
) -> Dict[str, float]:
    """
    Froessling equation for mass transfer from sphere in stagnant fluid.

    Sh = 2 + 0.552 * Re^0.5 * Sc^(1/3)

    The "2" represents pure diffusion limit (Sh = 2 for Re → 0).

    Parameters
    ----------
    Re : float
        Reynolds number (based on diameter)
    Sc : float
        Schmidt number

    Returns
    -------
    dict
        sherwood: Sh
    """
    Sh = 2 + 0.552 * np.sqrt(Re) * (Sc ** (1/3))

    return {
        'sherwood': Sh,
        'correlation': 'froessling',
        'reynolds': Re,
        'schmidt': Sc,
        'diffusion_limit': 2.0
    }


def ranz_marshall(
    Re: float,
    Sc: float
) -> Dict[str, float]:
    """
    Ranz-Marshall correlation for sphere in flowing fluid.

    Sh = 2 + 0.6 * Re^0.5 * Sc^(1/3)

    Widely used for droplets, particles, bubbles.

    Valid for:
        - 2 < Re < 200
        - 0.6 < Sc < 2.5 (gases)
        - Can extend to liquids

    Parameters
    ----------
    Re : float
        Reynolds number
    Sc : float
        Schmidt number

    Returns
    -------
    dict
        sherwood: Sh
    """
    Sh = 2 + 0.6 * np.sqrt(Re) * (Sc ** (1/3))

    return {
        'sherwood': Sh,
        'correlation': 'ranz_marshall',
        'reynolds': Re,
        'schmidt': Sc
    }


def packed_bed(
    Re: float,
    Sc: float,
    void_fraction: float = 0.4
) -> Dict[str, float]:
    """
    Mass transfer in packed bed (Wakao-Funazkri correlation).

    Sh = 2 + 1.1 * Re^0.6 * Sc^(1/3)

    Where Re is based on superficial velocity and particle diameter.

    Parameters
    ----------
    Re : float
        Reynolds number (superficial velocity, particle diameter)
    Sc : float
        Schmidt number
    void_fraction : float
        Bed void fraction ε

    Returns
    -------
    dict
        sherwood: Sh
        volumetric_coefficient: k_c * a_v (if void fraction given)
    """
    Sh = 2 + 1.1 * (Re ** 0.6) * (Sc ** (1/3))

    # Specific surface area for spheres: a_v = 6(1-ε)/d_p
    # Volumetric coefficient: k_c * a_v

    return {
        'sherwood': Sh,
        'correlation': 'wakao_funazkri',
        'reynolds': Re,
        'schmidt': Sc,
        'void_fraction': void_fraction
    }


def falling_film(
    Re_film: float,
    Sc: float
) -> Dict[str, float]:
    """
    Mass transfer for laminar falling film.

    Sh = 0.69 * (Re_film * Sc * δ/L)^(1/3)

    For fully developed:
    Sh = 3.41 (constant wall concentration)

    Parameters
    ----------
    Re_film : float
        Film Reynolds number = 4Γ/μ (Γ = mass flow rate per unit width)
    Sc : float
        Schmidt number

    Returns
    -------
    dict
        sherwood: Sh
    """
    # Fully developed laminar falling film
    if Re_film < 20:  # Laminar regime
        Sh = 3.41
        regime = 'laminar_developed'
    else:
        # Turbulent falling film (Henstock-Hanratty)
        Sh = 0.0113 * (Re_film ** 0.835) * (Sc ** 0.5)
        regime = 'turbulent'

    return {
        'sherwood': Sh,
        'correlation': 'falling_film',
        'regime': regime,
        'reynolds_film': Re_film,
        'schmidt': Sc
    }


def compute_kc_from_sherwood(
    Sh: float,
    D_AB: float,
    L: float
) -> Dict[str, float]:
    """
    Convert Sherwood number to mass transfer coefficient.

    k_c = Sh * D_AB / L

    Parameters
    ----------
    Sh : float
        Sherwood number
    D_AB : float
        Diffusivity [m²/s]
    L : float
        Characteristic length [m]

    Returns
    -------
    dict
        k_c: mass transfer coefficient [m/s]
    """
    k_c = Sh * D_AB / L

    return {
        'k_c': k_c,
        'sherwood': Sh,
        'diffusivity': D_AB,
        'characteristic_length': L
    }


def overall_mass_transfer_coefficient(
    k_c_gas: float,
    k_c_liquid: float,
    H: float,
    gas_side_controlling: Optional[bool] = None
) -> Dict[str, float]:
    """
    Overall mass transfer coefficient for gas-liquid systems.

    For transfer from gas to liquid:
    1/K_L = 1/k_L + 1/(H*k_G)
    1/K_G = H/k_L + 1/k_G

    Where H = Henry's law constant (C_L = H * p_A)

    Parameters
    ----------
    k_c_gas : float
        Gas-side coefficient k_G [mol/(m²·s·Pa)]
    k_c_liquid : float
        Liquid-side coefficient k_L [m/s]
    H : float
        Henry's constant [mol/(m³·Pa)]
    gas_side_controlling : bool, optional
        Hint about controlling resistance

    Returns
    -------
    dict
        K_G: Overall gas-side coefficient
        K_L: Overall liquid-side coefficient
        controlling_resistance: which side limits transfer
    """
    # Resistance analysis
    R_gas = 1 / k_c_gas if k_c_gas > 0 else np.inf
    R_liquid = 1 / (H * k_c_liquid) if (H > 0 and k_c_liquid > 0) else np.inf

    R_total = R_gas + R_liquid
    K_G = 1 / R_total if R_total > 0 else np.nan

    # Liquid-side overall coefficient
    R_liquid_direct = 1 / k_c_liquid if k_c_liquid > 0 else np.inf
    R_gas_liquid = H / k_c_gas if k_c_gas > 0 else np.inf
    K_L = 1 / (R_liquid_direct + R_gas_liquid) if (R_liquid_direct + R_gas_liquid) > 0 else np.nan

    # Determine controlling resistance
    if R_gas > 2 * R_liquid:
        controlling = 'gas_side'
    elif R_liquid > 2 * R_gas:
        controlling = 'liquid_side'
    else:
        controlling = 'both_significant'

    return {
        'K_G': K_G,
        'K_L': K_L,
        'R_gas': R_gas,
        'R_liquid': R_liquid,
        'controlling_resistance': controlling,
        'gas_resistance_fraction': R_gas / R_total if R_total > 0 else np.nan
    }
