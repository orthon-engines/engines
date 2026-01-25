"""
Thermodynamics Calculations

Core thermodynamic relationships for chemical engineering:

First Law: ΔU = Q - W
Second Law: ΔS_universe ≥ 0

Key state functions:
    - Enthalpy: H = U + PV
    - Entropy: dS = δQ_rev/T
    - Gibbs free energy: G = H - TS
    - Helmholtz free energy: A = U - TS

Equations of State:
    - Ideal gas: PV = nRT
    - Van der Waals: (P + a/V²)(V - b) = RT
    - Peng-Robinson
    - Redlich-Kwong

Phase Equilibrium:
    - Fugacity and fugacity coefficient
    - Activity and activity coefficient
    - Raoult's law, Henry's law
"""

import numpy as np
from typing import Dict, Optional, Tuple


# Universal gas constant
R = 8.314  # J/(mol·K)


def ideal_gas(
    P: Optional[float] = None,
    V: Optional[float] = None,
    n: Optional[float] = None,
    T: Optional[float] = None
) -> Dict[str, float]:
    """
    Ideal gas equation of state: PV = nRT

    Provide any 3 of 4 variables, solves for the 4th.

    Parameters
    ----------
    P : float, optional
        Pressure [Pa]
    V : float, optional
        Volume [m³]
    n : float, optional
        Moles [mol]
    T : float, optional
        Temperature [K]

    Returns
    -------
    dict
        All four state variables
    """
    params = {'P': P, 'V': V, 'n': n, 'T': T}
    none_count = sum(1 for v in params.values() if v is None)

    if none_count != 1:
        return {'error': 'Provide exactly 3 of 4 parameters'}

    if P is None:
        P = n * R * T / V
    elif V is None:
        V = n * R * T / P
    elif n is None:
        n = P * V / (R * T)
    elif T is None:
        T = P * V / (n * R)

    return {
        'pressure': P,
        'volume': V,
        'moles': n,
        'temperature': T,
        'molar_volume': V / n if n > 0 else np.nan,
        'compressibility_factor': 1.0  # Z = 1 for ideal gas
    }


def van_der_waals(
    T: float,
    V_m: float,
    a: float,
    b: float
) -> Dict[str, float]:
    """
    Van der Waals equation of state.

    (P + a/V_m²)(V_m - b) = RT

    Accounts for:
        - Molecular attraction (a term)
        - Molecular volume (b term)

    Parameters
    ----------
    T : float
        Temperature [K]
    V_m : float
        Molar volume [m³/mol]
    a : float
        Attraction parameter [Pa·m⁶/mol²]
    b : float
        Covolume [m³/mol]

    Returns
    -------
    dict
        pressure: P [Pa]
        compressibility_factor: Z = PV/(nRT)
    """
    if V_m <= b:
        return {'pressure': np.nan, 'error': 'V_m must be > b'}

    P = R * T / (V_m - b) - a / (V_m ** 2)
    Z = P * V_m / (R * T)

    return {
        'pressure': P,
        'compressibility_factor': Z,
        'equation_of_state': 'van_der_waals',
        'departure_from_ideal': Z - 1
    }


def peng_robinson(
    T: float,
    V_m: float,
    T_c: float,
    P_c: float,
    omega: float
) -> Dict[str, float]:
    """
    Peng-Robinson equation of state.

    P = RT/(V_m - b) - a*α/(V_m² + 2bV_m - b²)

    Where:
        a = 0.45724 * R²*T_c² / P_c
        b = 0.07780 * R*T_c / P_c
        α = [1 + κ(1 - √(T/T_c))]²
        κ = 0.37464 + 1.54226ω - 0.26992ω²

    Parameters
    ----------
    T : float
        Temperature [K]
    V_m : float
        Molar volume [m³/mol]
    T_c : float
        Critical temperature [K]
    P_c : float
        Critical pressure [Pa]
    omega : float
        Acentric factor (dimensionless)

    Returns
    -------
    dict
        pressure: P [Pa]
        compressibility_factor: Z
        fugacity_coefficient: φ
    """
    # Parameters
    a = 0.45724 * (R ** 2) * (T_c ** 2) / P_c
    b = 0.07780 * R * T_c / P_c
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * (omega ** 2)
    alpha = (1 + kappa * (1 - np.sqrt(T / T_c))) ** 2

    # Pressure
    denom = V_m ** 2 + 2 * b * V_m - b ** 2
    if denom == 0 or V_m <= b:
        return {'pressure': np.nan, 'error': 'Invalid molar volume'}

    P = R * T / (V_m - b) - a * alpha / denom
    Z = P * V_m / (R * T)

    # Fugacity coefficient (simplified)
    A = a * alpha * P / ((R * T) ** 2)
    B = b * P / (R * T)
    sqrt2 = np.sqrt(2)

    # ln(φ) = Z - 1 - ln(Z - B) - A/(2√2*B) * ln[(Z + (1+√2)B)/(Z + (1-√2)B)]
    if Z > B and Z + (1 + sqrt2) * B > 0 and Z + (1 - sqrt2) * B > 0:
        ln_phi = Z - 1 - np.log(Z - B) - A / (2 * sqrt2 * B) * np.log(
            (Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B)
        )
        phi = np.exp(ln_phi)
    else:
        phi = np.nan

    return {
        'pressure': P,
        'compressibility_factor': Z,
        'fugacity_coefficient': phi,
        'a_parameter': a * alpha,
        'b_parameter': b,
        'equation_of_state': 'peng_robinson'
    }


def enthalpy_ideal_gas(
    Cp_coeffs: Tuple[float, ...],
    T: float,
    T_ref: float = 298.15
) -> Dict[str, float]:
    """
    Ideal gas enthalpy from heat capacity polynomial.

    Cp = A + B*T + C*T² + D*T³

    ΔH = ∫[T_ref to T] Cp dT

    Parameters
    ----------
    Cp_coeffs : tuple
        (A, B, C, D) coefficients for Cp [J/(mol·K)]
    T : float
        Temperature [K]
    T_ref : float
        Reference temperature [K]

    Returns
    -------
    dict
        delta_H: enthalpy change [J/mol]
    """
    A, B, C, D = Cp_coeffs if len(Cp_coeffs) == 4 else (*Cp_coeffs, 0, 0, 0)[:4]

    def H_integral(T):
        return A * T + B * T**2 / 2 + C * T**3 / 3 + D * T**4 / 4

    delta_H = H_integral(T) - H_integral(T_ref)

    return {
        'delta_H': delta_H,
        'H_at_T': H_integral(T),
        'T': T,
        'T_ref': T_ref,
        'Cp_at_T': A + B * T + C * T**2 + D * T**3
    }


def entropy_ideal_gas(
    Cp_coeffs: Tuple[float, ...],
    T: float,
    P: float,
    T_ref: float = 298.15,
    P_ref: float = 101325.0
) -> Dict[str, float]:
    """
    Ideal gas entropy from heat capacity polynomial.

    ΔS = ∫[T_ref to T] (Cp/T) dT - R*ln(P/P_ref)

    Parameters
    ----------
    Cp_coeffs : tuple
        (A, B, C, D) coefficients for Cp
    T : float
        Temperature [K]
    P : float
        Pressure [Pa]
    T_ref : float
        Reference temperature [K]
    P_ref : float
        Reference pressure [Pa]

    Returns
    -------
    dict
        delta_S: entropy change [J/(mol·K)]
    """
    A, B, C, D = Cp_coeffs if len(Cp_coeffs) == 4 else (*Cp_coeffs, 0, 0, 0)[:4]

    def S_integral(T):
        return A * np.log(T) + B * T + C * T**2 / 2 + D * T**3 / 3

    delta_S_T = S_integral(T) - S_integral(T_ref)
    delta_S_P = -R * np.log(P / P_ref)
    delta_S = delta_S_T + delta_S_P

    return {
        'delta_S': delta_S,
        'delta_S_temperature': delta_S_T,
        'delta_S_pressure': delta_S_P,
        'T': T,
        'P': P
    }


def gibbs_free_energy(
    H: float,
    S: float,
    T: float
) -> Dict[str, float]:
    """
    Gibbs free energy: G = H - TS

    For reaction spontaneity:
        ΔG < 0: spontaneous
        ΔG = 0: equilibrium
        ΔG > 0: non-spontaneous

    Parameters
    ----------
    H : float
        Enthalpy [J/mol]
    S : float
        Entropy [J/(mol·K)]
    T : float
        Temperature [K]

    Returns
    -------
    dict
        G: Gibbs free energy [J/mol]
    """
    G = H - T * S

    return {
        'G': G,
        'H': H,
        'TS': T * S,
        'T': T,
        'spontaneous': G < 0
    }


def clausius_clapeyron(
    T1: float,
    P1: float,
    T2: Optional[float] = None,
    P2: Optional[float] = None,
    delta_H_vap: float = None
) -> Dict[str, float]:
    """
    Clausius-Clapeyron equation for vapor pressure.

    ln(P2/P1) = -ΔH_vap/R * (1/T2 - 1/T1)

    Provide T1, P1, delta_H_vap, and either T2 or P2.

    Parameters
    ----------
    T1 : float
        Temperature 1 [K]
    P1 : float
        Pressure 1 [Pa]
    T2 : float, optional
        Temperature 2 [K]
    P2 : float, optional
        Pressure 2 [Pa]
    delta_H_vap : float
        Enthalpy of vaporization [J/mol]

    Returns
    -------
    dict
        Missing P2 or T2, and the relationship
    """
    if T2 is not None:
        # Solve for P2
        ln_ratio = -delta_H_vap / R * (1/T2 - 1/T1)
        P2 = P1 * np.exp(ln_ratio)
        return {
            'P2': P2,
            'T2': T2,
            'ln_P2_P1': ln_ratio
        }
    elif P2 is not None:
        # Solve for T2
        ln_ratio = np.log(P2 / P1)
        inv_T2 = 1/T1 - ln_ratio * R / delta_H_vap
        T2 = 1 / inv_T2
        return {
            'T2': T2,
            'P2': P2,
            'ln_P2_P1': ln_ratio
        }
    else:
        return {'error': 'Provide either T2 or P2'}


def antoine_equation(
    A: float,
    B: float,
    C: float,
    T: Optional[float] = None,
    P: Optional[float] = None
) -> Dict[str, float]:
    """
    Antoine equation for vapor pressure.

    log₁₀(P) = A - B/(T + C)

    Where P is in mmHg and T is in °C (common convention).

    Parameters
    ----------
    A, B, C : float
        Antoine coefficients
    T : float, optional
        Temperature [°C]
    P : float, optional
        Vapor pressure [mmHg]

    Returns
    -------
    dict
        P or T (whichever was missing)
    """
    if T is not None:
        log_P = A - B / (T + C)
        P = 10 ** log_P
        return {
            'vapor_pressure_mmHg': P,
            'vapor_pressure_Pa': P * 133.322,
            'temperature_C': T
        }
    elif P is not None:
        log_P = np.log10(P)
        T = B / (A - log_P) - C
        return {
            'temperature_C': T,
            'temperature_K': T + 273.15,
            'vapor_pressure_mmHg': P
        }
    else:
        return {'error': 'Provide either T or P'}


def raoults_law(
    x_i: float,
    P_sat_i: float
) -> Dict[str, float]:
    """
    Raoult's law for ideal solution vapor-liquid equilibrium.

    p_i = x_i * P_sat_i

    Where:
        p_i = partial pressure of component i in vapor
        x_i = mole fraction in liquid
        P_sat_i = saturation pressure of pure i

    Parameters
    ----------
    x_i : float
        Liquid mole fraction
    P_sat_i : float
        Pure component saturation pressure [Pa]

    Returns
    -------
    dict
        partial_pressure: p_i [Pa]
    """
    p_i = x_i * P_sat_i

    return {
        'partial_pressure': p_i,
        'liquid_mole_fraction': x_i,
        'saturation_pressure': P_sat_i,
        'activity_coefficient': 1.0  # Ideal solution
    }


def henrys_law(
    x_i: float,
    H_i: float
) -> Dict[str, float]:
    """
    Henry's law for dilute solutions.

    p_i = x_i * H_i

    Where H_i is Henry's constant.

    Parameters
    ----------
    x_i : float
        Mole fraction in liquid (dilute)
    H_i : float
        Henry's constant [Pa]

    Returns
    -------
    dict
        partial_pressure: p_i [Pa]
    """
    p_i = x_i * H_i

    return {
        'partial_pressure': p_i,
        'liquid_mole_fraction': x_i,
        'henrys_constant': H_i
    }


def fugacity(
    P: float,
    phi: float
) -> Dict[str, float]:
    """
    Fugacity from fugacity coefficient.

    f = φ * P

    For ideal gas: φ = 1, f = P
    For real gas: φ ≠ 1

    Parameters
    ----------
    P : float
        Pressure [Pa]
    phi : float
        Fugacity coefficient (dimensionless)

    Returns
    -------
    dict
        fugacity: f [Pa]
    """
    f = phi * P

    return {
        'fugacity': f,
        'fugacity_coefficient': phi,
        'pressure': P,
        'departure_from_ideal': phi - 1
    }


def activity(
    x_i: float,
    gamma_i: float
) -> Dict[str, float]:
    """
    Activity from activity coefficient.

    a_i = γ_i * x_i

    For ideal solution: γ = 1, a = x
    For non-ideal: γ ≠ 1

    Parameters
    ----------
    x_i : float
        Mole fraction
    gamma_i : float
        Activity coefficient

    Returns
    -------
    dict
        activity: a_i (dimensionless)
    """
    a_i = gamma_i * x_i

    return {
        'activity': a_i,
        'activity_coefficient': gamma_i,
        'mole_fraction': x_i,
        'ideal_behavior': abs(gamma_i - 1) < 0.01
    }
