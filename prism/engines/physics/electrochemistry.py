"""
Electrode Kinetics Engines

Butler-Volmer equation, Tafel kinetics, mass transfer limitations.
"""

import numpy as np
from typing import Dict, Any, Optional


# Physical constants
F = 96485.33212  # Faraday constant [C/mol]
R = 8.314462618  # Gas constant [J/(mol·K)]


def butler_volmer(eta: float, i0: float, alpha: float = 0.5,
                  n: int = 1, T: float = 298.15) -> Dict[str, Any]:
    """
    Butler-Volmer equation for electrode kinetics.

    i = i₀ [exp(αnFη/RT) - exp(-(1-α)nFη/RT)]

    Parameters
    ----------
    eta : float
        Overpotential [V] (E - E_eq)
    i0 : float
        Exchange current density [A/m²]
    alpha : float
        Transfer coefficient (0 < α < 1, typically 0.5)
    n : int
        Number of electrons transferred
    T : float
        Temperature [K]

    Returns
    -------
    dict
        i: Current density [A/m²]
        i_anodic: Anodic component
        i_cathodic: Cathodic component
    """
    # Dimensionless overpotential
    f = F / (R * T)

    # Anodic and cathodic components
    i_anodic = i0 * np.exp(alpha * n * f * eta)
    i_cathodic = i0 * np.exp(-(1 - alpha) * n * f * eta)

    i = i_anodic - i_cathodic

    return {
        'i': float(i),
        'i_anodic': float(i_anodic),
        'i_cathodic': float(i_cathodic),
        'eta': eta,
        'i0': i0,
        'alpha': alpha,
        'n': n,
        'temperature': T,
        'equation': 'i = i₀[exp(αnFη/RT) - exp(-(1-α)nFη/RT)]',
    }


def tafel(eta: float, i0: float, b: float = None, alpha: float = 0.5,
          n: int = 1, T: float = 298.15) -> Dict[str, Any]:
    """
    Tafel equation for high overpotential kinetics.

    Anodic:  η = a + b·log(i)  where b = 2.303RT/(αnF)
    Cathodic: η = a - b·log(i)  where b = 2.303RT/((1-α)nF)

    Parameters
    ----------
    eta : float
        Overpotential [V]
    i0 : float
        Exchange current density [A/m²]
    b : float, optional
        Tafel slope [V/decade]. If None, calculated from α
    alpha : float
        Transfer coefficient (default 0.5)
    n : int
        Number of electrons
    T : float
        Temperature [K]

    Returns
    -------
    dict
        i: Current density [A/m²]
        b_anodic: Anodic Tafel slope
        b_cathodic: Cathodic Tafel slope
    """
    # Tafel slopes
    b_anodic = 2.303 * R * T / (alpha * n * F)
    b_cathodic = 2.303 * R * T / ((1 - alpha) * n * F)

    if b is None:
        b = b_anodic if eta > 0 else b_cathodic

    # Tafel equation: log(i/i0) = η/b (anodic) or -η/b (cathodic)
    if eta > 0:
        # Anodic
        log_i = np.log10(i0) + eta / b_anodic
    else:
        # Cathodic
        log_i = np.log10(i0) - eta / b_cathodic

    i = 10 ** log_i

    return {
        'i': float(i),
        'b_anodic': float(b_anodic),
        'b_cathodic': float(b_cathodic),
        'b_used': float(b),
        'eta': eta,
        'i0': i0,
        'region': 'anodic' if eta > 0 else 'cathodic',
        'equation': 'η = a ± b·log(i)',
    }


def limiting_current(i_L: float, i: float, eta: float = None,
                     i0: float = None, alpha: float = 0.5,
                     n: int = 1, T: float = 298.15) -> Dict[str, Any]:
    """
    Current with mass transfer limitation.

    i = i_L / [1 + (i_L/i_kinetic)]

    At limiting current, surface concentration → 0.

    Parameters
    ----------
    i_L : float
        Limiting current density [A/m²]
    i : float
        Actual current density [A/m²]
    eta : float, optional
        Overpotential [V] for kinetic calculation
    i0 : float, optional
        Exchange current density [A/m²]
    alpha : float
        Transfer coefficient
    n : int
        Number of electrons
    T : float
        Temperature [K]

    Returns
    -------
    dict
        i_total: Total current density
        limitation: 'kinetic', 'mixed', or 'mass transfer'
        efficiency: i/i_L ratio
    """
    ratio = i / i_L

    if ratio < 0.1:
        limitation = 'kinetic'
    elif ratio > 0.9:
        limitation = 'mass transfer'
    else:
        limitation = 'mixed'

    # Concentration overpotential
    if ratio < 1:
        eta_conc = (R * T / (n * F)) * np.log(1 - ratio)
    else:
        eta_conc = float('-inf')

    return {
        'i': float(i),
        'i_L': i_L,
        'i_over_i_L': float(ratio),
        'limitation': limitation,
        'eta_concentration': float(eta_conc) if eta_conc > float('-inf') else None,
        'surface_concentration_ratio': float(1 - ratio) if ratio < 1 else 0,
        'equation': 'C_s/C_b = 1 - i/i_L',
    }


def mixed_potential(E_a: float, E_c: float, i0_a: float, i0_c: float,
                    b_a: float, b_c: float) -> Dict[str, Any]:
    """
    Mixed potential theory for corrosion.

    At mixed potential: i_anodic = |i_cathodic|

    Parameters
    ----------
    E_a : float
        Equilibrium potential of anodic reaction [V]
    E_c : float
        Equilibrium potential of cathodic reaction [V]
    i0_a : float
        Exchange current density for anodic reaction [A/m²]
    i0_c : float
        Exchange current density for cathodic reaction [A/m²]
    b_a : float
        Anodic Tafel slope [V/decade]
    b_c : float
        Cathodic Tafel slope [V/decade]

    Returns
    -------
    dict
        E_corr: Corrosion potential [V]
        i_corr: Corrosion current density [A/m²]
    """
    # At E_corr: i_a = i_c
    # i0_a * 10^((E_corr - E_a)/b_a) = i0_c * 10^((E_c - E_corr)/b_c)

    # Solving analytically:
    # (E_corr - E_a)/b_a + log(i0_a) = (E_c - E_corr)/b_c + log(i0_c)

    numerator = (E_c / b_c + E_a / b_a + np.log10(i0_c) - np.log10(i0_a))
    denominator = (1 / b_a + 1 / b_c)

    E_corr = numerator / denominator

    # Corrosion current
    i_corr = i0_a * 10 ** ((E_corr - E_a) / b_a)

    return {
        'E_corr': float(E_corr),
        'i_corr': float(i_corr),
        'E_a': E_a,
        'E_c': E_c,
        'polarization_anodic': float(E_corr - E_a),
        'polarization_cathodic': float(E_c - E_corr),
        'equation': 'At E_corr: i_anodic = i_cathodic',
    }


def polarization_curve(eta_range: np.ndarray, i0: float, alpha: float = 0.5,
                       n: int = 1, T: float = 298.15,
                       i_L: float = None) -> Dict[str, Any]:
    """
    Generate polarization curve data.

    Parameters
    ----------
    eta_range : array
        Overpotential range [V]
    i0 : float
        Exchange current density [A/m²]
    alpha : float
        Transfer coefficient
    n : int
        Number of electrons
    T : float
        Temperature [K]
    i_L : float, optional
        Limiting current density [A/m²]

    Returns
    -------
    dict
        eta: Overpotentials
        i: Current densities
        log_abs_i: Log of absolute current density
    """
    eta_range = np.asarray(eta_range)
    f = F / (R * T)

    # Butler-Volmer
    i = i0 * (np.exp(alpha * n * f * eta_range) -
              np.exp(-(1 - alpha) * n * f * eta_range))

    # Apply mass transfer limitation if specified
    if i_L is not None:
        # Cathodic limiting: for negative i
        i = np.where(i < 0, np.maximum(i, -i_L), i)

    log_abs_i = np.log10(np.abs(i) + 1e-20)  # Avoid log(0)

    return {
        'eta': eta_range.tolist(),
        'i': i.tolist(),
        'log_abs_i': log_abs_i.tolist(),
        'i0': i0,
        'alpha': alpha,
        'n': n,
        'i_L': i_L,
    }


def exchange_current_density(i0_ref: float, T_ref: float, T: float,
                             E_a: float = 50000) -> Dict[str, Any]:
    """
    Temperature dependence of exchange current density.

    i₀(T) = i₀(T_ref) exp[(-E_a/R)(1/T - 1/T_ref)]

    Parameters
    ----------
    i0_ref : float
        Exchange current density at reference temperature [A/m²]
    T_ref : float
        Reference temperature [K]
    T : float
        Temperature of interest [K]
    E_a : float
        Activation energy [J/mol] (default 50 kJ/mol)

    Returns
    -------
    dict
        i0: Exchange current density at T [A/m²]
        ratio: i0(T)/i0(T_ref)
    """
    exponent = (-E_a / R) * (1 / T - 1 / T_ref)
    ratio = np.exp(exponent)
    i0 = i0_ref * ratio

    return {
        'i0': float(i0),
        'i0_ref': i0_ref,
        'ratio': float(ratio),
        'T': T,
        'T_ref': T_ref,
        'E_a': E_a,
        'equation': 'i₀(T) = i₀(T_ref)·exp[(-E_a/R)(1/T - 1/T_ref)]',
    }


def _nan_result(reason: str, keys: list) -> Dict[str, Any]:
    """Return NaN result with error reason."""
    result = {k: float('nan') for k in keys}
    result['error'] = reason
    return result


def compute(signal: np.ndarray = None, eta: float = None, i0: float = None,
            alpha: float = 0.5, n: int = 1, T: float = 298.15,
            **kwargs) -> Dict[str, Any]:
    """
    Main entry point for electrode kinetics calculations.
    """
    if eta is not None and i0 is not None:
        return butler_volmer(eta, i0, alpha, n, T)

    if 'E_a' in kwargs and 'E_c' in kwargs:
        return mixed_potential(kwargs['E_a'], kwargs['E_c'],
                              kwargs.get('i0_a', 1e-6), kwargs.get('i0_c', 1e-9),
                              kwargs.get('b_a', 0.06), kwargs.get('b_c', 0.12))

    return _nan_result(
        'Insufficient parameters for kinetics calculation',
        ['i', 'i_anodic', 'i_cathodic', 'E_corr', 'i_corr']
    )
