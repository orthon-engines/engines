"""
Chemical Reaction Kinetics

Core reaction engineering calculations:

Rate Laws:
    - Arrhenius equation: k = A * exp(-Ea/RT)
    - Power law: r = k * C_A^n * C_B^m
    - Michaelis-Menten: r = V_max * C / (K_m + C)

Reactor Design:
    - Batch: t = N_A0 * ∫(dX/(-r_A*V))
    - CSTR: V/F_A0 = X/(-r_A)
    - PFR: V/F_A0 = ∫(dX/(-r_A))

Performance Metrics:
    - Conversion: X = (N_A0 - N_A) / N_A0
    - Yield: Y = moles desired / moles limiting reacted
    - Selectivity: S = rate desired / rate undesired
"""

import numpy as np
from typing import Dict, Optional, List


# Universal gas constant
R = 8.314  # J/(mol·K)


def arrhenius(
    A: float,
    Ea: float,
    T: float
) -> Dict[str, float]:
    """
    Arrhenius equation for rate constant.

    k = A * exp(-Ea/RT)

    Parameters
    ----------
    A : float
        Pre-exponential factor [same units as k]
    Ea : float
        Activation energy [J/mol]
    T : float
        Temperature [K]

    Returns
    -------
    dict
        k: rate constant
        exponential_factor: exp(-Ea/RT)
    """
    exp_factor = np.exp(-Ea / (R * T))
    k = A * exp_factor

    return {
        'k': k,
        'pre_exponential': A,
        'activation_energy': Ea,
        'temperature': T,
        'exponential_factor': exp_factor,
        'Ea_RT': Ea / (R * T)
    }


def arrhenius_two_temperatures(
    k1: float,
    T1: float,
    k2: Optional[float] = None,
    T2: Optional[float] = None,
    Ea: Optional[float] = None
) -> Dict[str, float]:
    """
    Arrhenius equation with two temperature points.

    ln(k2/k1) = -Ea/R * (1/T2 - 1/T1)

    Can solve for Ea given k1, k2, T1, T2
    Or solve for k2 given k1, T1, T2, Ea

    Parameters
    ----------
    k1 : float
        Rate constant at T1
    T1 : float
        Temperature 1 [K]
    k2 : float, optional
        Rate constant at T2
    T2 : float, optional
        Temperature 2 [K]
    Ea : float, optional
        Activation energy [J/mol]

    Returns
    -------
    dict
        Ea or k2 (whichever was missing)
    """
    if k2 is not None and T2 is not None and Ea is None:
        # Solve for Ea
        ln_ratio = np.log(k2 / k1)
        inv_T_diff = 1/T2 - 1/T1
        Ea = -R * ln_ratio / inv_T_diff
        A = k1 / np.exp(-Ea / (R * T1))
        return {
            'activation_energy': Ea,
            'pre_exponential': A,
            'Ea_kJ_mol': Ea / 1000
        }
    elif T2 is not None and Ea is not None and k2 is None:
        # Solve for k2
        ln_ratio = -Ea / R * (1/T2 - 1/T1)
        k2 = k1 * np.exp(ln_ratio)
        return {
            'k2': k2,
            'T2': T2,
            'ratio_k2_k1': k2 / k1
        }
    else:
        return {
            'activation_energy': float('nan'),
            'pre_exponential': float('nan'),
            'k2': float('nan'),
            'error': 'Provide (k1, T1, k2, T2) or (k1, T1, T2, Ea)'
        }


def power_law_rate(
    k: float,
    concentrations: Dict[str, float],
    orders: Dict[str, float]
) -> Dict[str, float]:
    """
    Power law rate expression.

    r = k * ∏(C_i^n_i)

    Parameters
    ----------
    k : float
        Rate constant
    concentrations : dict
        {species: concentration} [mol/m³]
    orders : dict
        {species: reaction order}

    Returns
    -------
    dict
        rate: r [mol/(m³·s)]
        overall_order: sum of orders
    """
    r = k
    for species, conc in concentrations.items():
        n = orders.get(species, 0)
        r *= conc ** n

    overall_order = sum(orders.values())

    return {
        'rate': r,
        'rate_constant': k,
        'overall_order': overall_order,
        'concentrations': concentrations,
        'orders': orders
    }


def michaelis_menten(
    V_max: float,
    K_m: float,
    S: float
) -> Dict[str, float]:
    """
    Michaelis-Menten kinetics for enzyme reactions.

    r = V_max * S / (K_m + S)

    Where:
        V_max = maximum reaction rate
        K_m = Michaelis constant (substrate conc at half V_max)
        S = substrate concentration

    Parameters
    ----------
    V_max : float
        Maximum rate [mol/(m³·s)]
    K_m : float
        Michaelis constant [mol/m³]
    S : float
        Substrate concentration [mol/m³]

    Returns
    -------
    dict
        rate: r
        saturation_fraction: S/(K_m + S)
    """
    r = V_max * S / (K_m + S)
    saturation = S / (K_m + S)

    return {
        'rate': r,
        'V_max': V_max,
        'K_m': K_m,
        'substrate': S,
        'saturation_fraction': saturation,
        'rate_fraction_of_Vmax': r / V_max
    }


def langmuir_hinshelwood(
    k: float,
    K_A: float,
    K_B: float,
    P_A: float,
    P_B: float
) -> Dict[str, float]:
    """
    Langmuir-Hinshelwood kinetics for surface reactions.

    r = k * K_A*P_A * K_B*P_B / (1 + K_A*P_A + K_B*P_B)²

    For A + B → products on a surface.

    Parameters
    ----------
    k : float
        Surface rate constant
    K_A : float
        Adsorption equilibrium constant for A [1/Pa]
    K_B : float
        Adsorption equilibrium constant for B [1/Pa]
    P_A : float
        Partial pressure of A [Pa]
    P_B : float
        Partial pressure of B [Pa]

    Returns
    -------
    dict
        rate: r
        coverage_A: θ_A
        coverage_B: θ_B
    """
    denominator = 1 + K_A * P_A + K_B * P_B
    theta_A = K_A * P_A / denominator
    theta_B = K_B * P_B / denominator

    r = k * K_A * P_A * K_B * P_B / (denominator ** 2)

    return {
        'rate': r,
        'coverage_A': theta_A,
        'coverage_B': theta_B,
        'total_coverage': theta_A + theta_B,
        'vacant_fraction': 1 / denominator
    }


def conversion(
    N_A0: float,
    N_A: float
) -> Dict[str, float]:
    """
    Conversion of limiting reactant.

    X = (N_A0 - N_A) / N_A0

    Parameters
    ----------
    N_A0 : float
        Initial moles of A
    N_A : float
        Final moles of A

    Returns
    -------
    dict
        conversion: X (0 to 1)
    """
    X = (N_A0 - N_A) / N_A0

    return {
        'conversion': X,
        'moles_reacted': N_A0 - N_A,
        'moles_remaining': N_A,
        'percent_conversion': X * 100
    }


def yield_and_selectivity(
    N_desired: float,
    N_A0: float,
    N_A: float,
    N_undesired: float,
    stoich_desired: float = 1.0,
    stoich_undesired: float = 1.0
) -> Dict[str, float]:
    """
    Yield and selectivity for multiple reactions.

    Yield: Y = N_desired / (N_A0 - N_A) * stoich ratio
    Selectivity: S = N_desired / (N_desired + N_undesired)

    Parameters
    ----------
    N_desired : float
        Moles of desired product
    N_A0 : float
        Initial moles of limiting reactant
    N_A : float
        Final moles of limiting reactant
    N_undesired : float
        Moles of undesired product
    stoich_desired : float
        Stoichiometric coefficient (moles A per mole desired)
    stoich_undesired : float
        Stoichiometric coefficient (moles A per mole undesired)

    Returns
    -------
    dict
        yield: Y
        selectivity: S
        overall_yield: conversion × selectivity
    """
    moles_reacted = N_A0 - N_A
    X = moles_reacted / N_A0

    # Yield based on moles reacted
    Y = N_desired * stoich_desired / moles_reacted if moles_reacted > 0 else 0

    # Selectivity
    total_products = N_desired + N_undesired
    S = N_desired / total_products if total_products > 0 else 0

    return {
        'yield': Y,
        'selectivity': S,
        'conversion': X,
        'overall_yield': X * S,
        'N_desired': N_desired,
        'N_undesired': N_undesired
    }


def batch_reactor_time(
    N_A0: float,
    V: float,
    k: float,
    X_final: float,
    order: int = 1
) -> Dict[str, float]:
    """
    Batch reactor design equation for nth order reaction.

    For first order: t = (1/k) * ln(1/(1-X))
    For second order: t = (1/k*C_A0) * X/(1-X)
    For zero order: t = C_A0*X/k

    Parameters
    ----------
    N_A0 : float
        Initial moles [mol]
    V : float
        Volume [m³]
    k : float
        Rate constant [appropriate units]
    X_final : float
        Final conversion
    order : int
        Reaction order (0, 1, or 2)

    Returns
    -------
    dict
        time: t [s]
    """
    C_A0 = N_A0 / V

    if order == 0:
        t = C_A0 * X_final / k
    elif order == 1:
        t = (1 / k) * np.log(1 / (1 - X_final))
    elif order == 2:
        t = (1 / (k * C_A0)) * X_final / (1 - X_final)
    else:
        return {
            'time': float('nan'),
            'C_A0': float('nan'),
            'conversion': float('nan'),
            'error': 'Only orders 0, 1, 2 implemented'
        }

    return {
        'time': t,
        'C_A0': C_A0,
        'conversion': X_final,
        'order': order
    }


def cstr_volume(
    F_A0: float,
    X: float,
    k: float,
    C_A0: float,
    order: int = 1
) -> Dict[str, float]:
    """
    CSTR design equation.

    V/F_A0 = X/(-r_A)

    For first order: V = F_A0*X / (k*C_A0*(1-X))

    Parameters
    ----------
    F_A0 : float
        Inlet molar flow rate [mol/s]
    X : float
        Conversion
    k : float
        Rate constant
    C_A0 : float
        Inlet concentration [mol/m³]
    order : int
        Reaction order

    Returns
    -------
    dict
        volume: V [m³]
        space_time: τ [s]
    """
    C_A = C_A0 * (1 - X)

    if order == 1:
        r_A = k * C_A
    elif order == 2:
        r_A = k * C_A ** 2
    elif order == 0:
        r_A = k
    else:
        return {
            'volume': float('nan'),
            'space_time': float('nan'),
            'exit_concentration': float('nan'),
            'rate': float('nan'),
            'conversion': float('nan'),
            'error': 'Only orders 0, 1, 2 implemented'
        }

    V = F_A0 * X / r_A if r_A > 0 else np.inf
    tau = V * C_A0 / F_A0  # Space time

    return {
        'volume': V,
        'space_time': tau,
        'exit_concentration': C_A,
        'rate': r_A,
        'conversion': X
    }


def pfr_volume(
    F_A0: float,
    X_final: float,
    k: float,
    C_A0: float,
    order: int = 1
) -> Dict[str, float]:
    """
    PFR design equation (analytical solutions).

    V/F_A0 = ∫[0 to X] dX/(-r_A)

    For first order: V/F_A0 = (1/k*C_A0) * ln(1/(1-X))
    For second order: V/F_A0 = (1/k*C_A0²) * X/(1-X)

    Parameters
    ----------
    F_A0 : float
        Inlet molar flow rate [mol/s]
    X_final : float
        Exit conversion
    k : float
        Rate constant
    C_A0 : float
        Inlet concentration [mol/m³]
    order : int
        Reaction order

    Returns
    -------
    dict
        volume: V [m³]
        space_time: τ [s]
    """
    v_0 = F_A0 / C_A0  # Volumetric flow rate

    if order == 1:
        integral = (1 / (k * C_A0)) * np.log(1 / (1 - X_final))
    elif order == 2:
        integral = (1 / (k * C_A0 ** 2)) * X_final / (1 - X_final)
    elif order == 0:
        integral = C_A0 * X_final / k
    else:
        return {
            'volume': float('nan'),
            'space_time': float('nan'),
            'volumetric_flow': float('nan'),
            'conversion': float('nan'),
            'error': 'Only orders 0, 1, 2 implemented'
        }

    V = F_A0 * integral
    tau = V / v_0

    return {
        'volume': V,
        'space_time': tau,
        'volumetric_flow': v_0,
        'conversion': X_final
    }


def residence_time_distribution(
    t: np.ndarray,
    E_t: np.ndarray
) -> Dict[str, float]:
    """
    Analyze residence time distribution.

    E(t) = exit age distribution (normalized so ∫E dt = 1)

    Mean residence time: τ_m = ∫ t*E(t) dt
    Variance: σ² = ∫ (t - τ_m)²*E(t) dt

    Parameters
    ----------
    t : array
        Time values [s]
    E_t : array
        E(t) values [1/s]

    Returns
    -------
    dict
        mean_residence_time: τ_m
        variance: σ²
        dispersion_number: σ²/τ_m² (measures deviation from plug flow)
    """
    # Normalize E(t)
    dt = np.diff(t, prepend=t[0])
    integral = np.sum(E_t * dt)
    E_normalized = E_t / integral if integral > 0 else E_t

    # Mean residence time
    tau_m = np.sum(t * E_normalized * dt)

    # Variance
    variance = np.sum((t - tau_m) ** 2 * E_normalized * dt)

    # Dispersion number
    dispersion = variance / (tau_m ** 2) if tau_m > 0 else np.nan

    return {
        'mean_residence_time': tau_m,
        'variance': variance,
        'standard_deviation': np.sqrt(variance),
        'dispersion_number': dispersion,
        'reactor_type': (
            'plug_flow' if dispersion < 0.01 else
            'mixed_flow' if dispersion > 1 else
            'intermediate'
        )
    }


def equilibrium_constant(
    delta_G_rxn: float,
    T: float
) -> Dict[str, float]:
    """
    Equilibrium constant from Gibbs free energy.

    K = exp(-ΔG°_rxn / RT)

    Parameters
    ----------
    delta_G_rxn : float
        Standard Gibbs free energy of reaction [J/mol]
    T : float
        Temperature [K]

    Returns
    -------
    dict
        K: equilibrium constant
        ln_K: natural log of K
    """
    ln_K = -delta_G_rxn / (R * T)
    K = np.exp(ln_K)

    return {
        'K': K,
        'ln_K': ln_K,
        'delta_G': delta_G_rxn,
        'temperature': T,
        'spontaneous_forward': K > 1
    }
