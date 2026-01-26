"""
Activity Coefficient Models

Margules, Van Laar, Wilson, NRTL, UNIQUAC, UNIFAC.
Predict deviations from ideal solution behavior.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple


# Gas constant
R = 8.314  # J/(mol·K)


def ideal_solution(x: np.ndarray) -> Dict[str, Any]:
    """
    Ideal solution model: gamma = 1 for all components.

    Parameters
    ----------
    x : array
        Mole fractions

    Returns
    -------
    dict
        gamma: Activity coefficients (all 1.0)
        G_E: Excess Gibbs energy (0)
    """
    x = np.asarray(x)
    n = len(x)

    return {
        'gamma': [1.0] * n,
        'G_E': 0.0,
        'model': 'ideal_solution',
        'equation': 'γ_i = 1',
    }


def margules_two_suffix(x1: float, A12: float, A21: float) -> Dict[str, Any]:
    """
    Two-suffix Margules equation for binary mixtures.

    ln(γ₁) = x₂² [A₁₂ + 2(A₂₁ - A₁₂)x₁]
    ln(γ₂) = x₁² [A₂₁ + 2(A₁₂ - A₂₁)x₂]

    Parameters
    ----------
    x1 : float
        Mole fraction of component 1
    A12 : float
        Margules parameter A₁₂
    A21 : float
        Margules parameter A₂₁

    Returns
    -------
    dict
        gamma1, gamma2: Activity coefficients
        ln_gamma1, ln_gamma2: Natural log of activity coefficients
        G_E_RT: Excess Gibbs energy / RT
    """
    x2 = 1 - x1

    ln_gamma1 = x2**2 * (A12 + 2 * (A21 - A12) * x1)
    ln_gamma2 = x1**2 * (A21 + 2 * (A12 - A21) * x2)

    gamma1 = np.exp(ln_gamma1)
    gamma2 = np.exp(ln_gamma2)

    # Excess Gibbs energy
    G_E_RT = x1 * x2 * (A12 * x2 + A21 * x1)

    return {
        'gamma': [float(gamma1), float(gamma2)],
        'gamma1': float(gamma1),
        'gamma2': float(gamma2),
        'ln_gamma1': float(ln_gamma1),
        'ln_gamma2': float(ln_gamma2),
        'G_E_RT': float(G_E_RT),
        'model': 'margules_two_suffix',
        'equation': 'ln(γ₁) = x₂²[A₁₂ + 2(A₂₁-A₁₂)x₁]',
    }


def margules(x1: float, A12: float, A21: float = None) -> Dict[str, Any]:
    """
    Margules equation (one or two parameter).

    One-parameter (symmetric): A12 = A21 = A
    Two-parameter (asymmetric): A12 ≠ A21

    Parameters
    ----------
    x1 : float
        Mole fraction of component 1
    A12 : float
        First Margules parameter
    A21 : float, optional
        Second Margules parameter (if None, symmetric)

    Returns
    -------
    dict
        gamma: Activity coefficients
    """
    if A21 is None:
        A21 = A12  # Symmetric

    return margules_two_suffix(x1, A12, A21)


def van_laar(x1: float, A: float, B: float) -> Dict[str, Any]:
    """
    Van Laar equation for binary mixtures.

    ln(γ₁) = A / [1 + (Ax₁)/(Bx₂)]²
    ln(γ₂) = B / [1 + (Bx₂)/(Ax₁)]²

    Parameters
    ----------
    x1 : float
        Mole fraction of component 1
    A : float
        Van Laar parameter A
    B : float
        Van Laar parameter B

    Returns
    -------
    dict
        gamma1, gamma2: Activity coefficients
    """
    x2 = 1 - x1

    # Avoid division by zero
    if x1 < 1e-10:
        ln_gamma1 = A
        ln_gamma2 = 0.0
    elif x2 < 1e-10:
        ln_gamma1 = 0.0
        ln_gamma2 = B
    else:
        ln_gamma1 = A / (1 + (A * x1) / (B * x2))**2
        ln_gamma2 = B / (1 + (B * x2) / (A * x1))**2

    gamma1 = np.exp(ln_gamma1)
    gamma2 = np.exp(ln_gamma2)

    # Excess Gibbs energy
    if x1 > 1e-10 and x2 > 1e-10:
        G_E_RT = A * B * x1 * x2 / (A * x1 + B * x2)
    else:
        G_E_RT = 0.0

    return {
        'gamma': [float(gamma1), float(gamma2)],
        'gamma1': float(gamma1),
        'gamma2': float(gamma2),
        'ln_gamma1': float(ln_gamma1),
        'ln_gamma2': float(ln_gamma2),
        'G_E_RT': float(G_E_RT),
        'model': 'van_laar',
        'equation': 'ln(γ₁) = A/[1 + Ax₁/(Bx₂)]²',
    }


def wilson(x: np.ndarray, Lambda: np.ndarray) -> Dict[str, Any]:
    """
    Wilson equation for multicomponent mixtures.

    ln(γ_i) = 1 - ln(Σⱼ x_j Λ_ij) - Σₖ (x_k Λ_ki / Σⱼ x_j Λ_kj)

    Parameters
    ----------
    x : array
        Mole fractions
    Lambda : 2D array
        Wilson parameters Λ_ij (Λ_ii = 1)

    Returns
    -------
    dict
        gamma: Activity coefficients
        G_E_RT: Excess Gibbs energy / RT
    """
    x = np.asarray(x)
    Lambda = np.asarray(Lambda)
    n = len(x)

    # Ensure diagonal is 1
    Lambda = Lambda.copy()
    np.fill_diagonal(Lambda, 1.0)

    # Calculate ln(gamma)
    ln_gamma = np.zeros(n)

    for i in range(n):
        # First sum: ln(Σⱼ x_j Λ_ij)
        sum1 = np.sum(x * Lambda[i, :])

        # Second sum: Σₖ (x_k Λ_ki / Σⱼ x_j Λ_kj)
        sum2 = 0.0
        for k in range(n):
            denom = np.sum(x * Lambda[k, :])
            if denom > 1e-10:
                sum2 += x[k] * Lambda[k, i] / denom

        ln_gamma[i] = 1 - np.log(sum1) - sum2

    gamma = np.exp(ln_gamma)

    # Excess Gibbs energy
    G_E_RT = -np.sum(x * np.log(np.dot(Lambda, x)))

    return {
        'gamma': gamma.tolist(),
        'ln_gamma': ln_gamma.tolist(),
        'G_E_RT': float(G_E_RT),
        'model': 'wilson',
        'equation': 'ln(γᵢ) = 1 - ln(Σⱼxⱼλᵢⱼ) - Σₖ(xₖΛₖᵢ/Σⱼxⱼλₖⱼ)',
    }


def wilson_binary(x1: float, Lambda12: float, Lambda21: float) -> Dict[str, Any]:
    """
    Wilson equation for binary systems.

    ln(γ₁) = -ln(x₁ + Λ₁₂x₂) + x₂(Λ₁₂/(x₁+Λ₁₂x₂) - Λ₂₁/(Λ₂₁x₁+x₂))
    ln(γ₂) = -ln(Λ₂₁x₁ + x₂) - x₁(Λ₁₂/(x₁+Λ₁₂x₂) - Λ₂₁/(Λ₂₁x₁+x₂))

    Parameters
    ----------
    x1 : float
        Mole fraction of component 1
    Lambda12 : float
        Wilson parameter Λ₁₂
    Lambda21 : float
        Wilson parameter Λ₂₁

    Returns
    -------
    dict
        gamma: Activity coefficients
    """
    x2 = 1 - x1

    # Avoid numerical issues
    eps = 1e-10

    term1 = x1 + Lambda12 * x2
    term2 = Lambda21 * x1 + x2

    if term1 < eps:
        term1 = eps
    if term2 < eps:
        term2 = eps

    common = Lambda12 / term1 - Lambda21 / term2

    ln_gamma1 = -np.log(term1) + x2 * common
    ln_gamma2 = -np.log(term2) - x1 * common

    gamma1 = np.exp(ln_gamma1)
    gamma2 = np.exp(ln_gamma2)

    # Excess Gibbs energy
    G_E_RT = -x1 * np.log(term1) - x2 * np.log(term2)

    return {
        'gamma': [float(gamma1), float(gamma2)],
        'gamma1': float(gamma1),
        'gamma2': float(gamma2),
        'ln_gamma1': float(ln_gamma1),
        'ln_gamma2': float(ln_gamma2),
        'G_E_RT': float(G_E_RT),
        'model': 'wilson',
    }


def nrtl(x: np.ndarray, tau: np.ndarray, alpha: np.ndarray) -> Dict[str, Any]:
    """
    NRTL (Non-Random Two-Liquid) model for multicomponent mixtures.

    G_ij = exp(-α_ij τ_ij)

    ln(γ_i) = (Σⱼ τ_ji G_ji x_j)/(Σₖ G_ki x_k) +
              Σⱼ [(x_j G_ij)/(Σₖ G_kj x_k)] [τ_ij - (Σₘ τ_mj G_mj x_m)/(Σₖ G_kj x_k)]

    Parameters
    ----------
    x : array
        Mole fractions
    tau : 2D array
        NRTL parameters τ_ij (τ_ii = 0)
    alpha : 2D array
        Non-randomness parameters α_ij (α_ij = α_ji)

    Returns
    -------
    dict
        gamma: Activity coefficients
        G_E_RT: Excess Gibbs energy / RT
    """
    x = np.asarray(x)
    tau = np.asarray(tau)
    alpha = np.asarray(alpha)
    n = len(x)

    # Ensure diagonal tau is 0
    tau = tau.copy()
    np.fill_diagonal(tau, 0.0)

    # Calculate G_ij
    G = np.exp(-alpha * tau)
    np.fill_diagonal(G, 1.0)

    # Calculate activity coefficients
    ln_gamma = np.zeros(n)

    for i in range(n):
        # Denominator sum: Σₖ G_ki x_k
        sum_Gki_xk = np.sum(G[:, i] * x)

        # First term: (Σⱼ τ_ji G_ji x_j) / sum_Gki_xk
        term1 = np.sum(tau[:, i] * G[:, i] * x) / sum_Gki_xk

        # Second term
        term2 = 0.0
        for j in range(n):
            sum_Gkj_xk = np.sum(G[:, j] * x)
            sum_tau_G_x = np.sum(tau[:, j] * G[:, j] * x)

            term2 += (x[j] * G[i, j] / sum_Gkj_xk) * (tau[i, j] - sum_tau_G_x / sum_Gkj_xk)

        ln_gamma[i] = term1 + term2

    gamma = np.exp(ln_gamma)

    # Excess Gibbs energy
    G_E_RT = 0.0
    for i in range(n):
        sum_tau_G_x = np.sum(tau[:, i] * G[:, i] * x)
        sum_G_x = np.sum(G[:, i] * x)
        G_E_RT += x[i] * sum_tau_G_x / sum_G_x

    return {
        'gamma': gamma.tolist(),
        'ln_gamma': ln_gamma.tolist(),
        'G_E_RT': float(G_E_RT),
        'model': 'nrtl',
        'equation': 'G_ij = exp(-α_ij·τ_ij)',
    }


def nrtl_binary(x1: float, tau12: float, tau21: float, alpha: float = 0.3) -> Dict[str, Any]:
    """
    NRTL for binary systems.

    Parameters
    ----------
    x1 : float
        Mole fraction of component 1
    tau12 : float
        NRTL parameter τ₁₂
    tau21 : float
        NRTL parameter τ₂₁
    alpha : float
        Non-randomness parameter (default 0.3)

    Returns
    -------
    dict
        gamma: Activity coefficients
    """
    x2 = 1 - x1

    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)

    # Denominators
    x1_plus_x2G21 = x1 + x2 * G21
    x2_plus_x1G12 = x2 + x1 * G12

    ln_gamma1 = x2**2 * (tau21 * (G21 / x1_plus_x2G21)**2 +
                        tau12 * G12 / x2_plus_x1G12**2)

    ln_gamma2 = x1**2 * (tau12 * (G12 / x2_plus_x1G12)**2 +
                        tau21 * G21 / x1_plus_x2G21**2)

    gamma1 = np.exp(ln_gamma1)
    gamma2 = np.exp(ln_gamma2)

    # Excess Gibbs energy
    G_E_RT = x1 * x2 * (tau21 * G21 / x1_plus_x2G21 +
                        tau12 * G12 / x2_plus_x1G12)

    return {
        'gamma': [float(gamma1), float(gamma2)],
        'gamma1': float(gamma1),
        'gamma2': float(gamma2),
        'ln_gamma1': float(ln_gamma1),
        'ln_gamma2': float(ln_gamma2),
        'G_E_RT': float(G_E_RT),
        'alpha': alpha,
        'model': 'nrtl',
    }


def uniquac(x: np.ndarray, r: np.ndarray, q: np.ndarray,
            tau: np.ndarray, z: int = 10) -> Dict[str, Any]:
    """
    UNIQUAC (Universal Quasi-Chemical) model.

    Parameters
    ----------
    x : array
        Mole fractions
    r : array
        Volume parameters r_i
    q : array
        Surface parameters q_i
    tau : 2D array
        UNIQUAC binary parameters τ_ij
    z : int
        Coordination number (default 10)

    Returns
    -------
    dict
        gamma: Activity coefficients
        gamma_C: Combinatorial contribution
        gamma_R: Residual contribution
    """
    x = np.asarray(x)
    r = np.asarray(r)
    q = np.asarray(q)
    tau = np.asarray(tau)
    n = len(x)

    # Volume fractions
    phi = x * r / np.sum(x * r)

    # Surface fractions
    theta = x * q / np.sum(x * q)

    # Segment fractions
    l = (z / 2) * (r - q) - (r - 1)

    # Combinatorial contribution
    ln_gamma_C = np.zeros(n)
    for i in range(n):
        ln_gamma_C[i] = (np.log(phi[i] / x[i]) + (z / 2) * q[i] * np.log(theta[i] / phi[i]) +
                        l[i] - (phi[i] / x[i]) * np.sum(x * l))

    # Residual contribution
    ln_gamma_R = np.zeros(n)
    for i in range(n):
        sum1 = np.sum(theta * tau[:, i])
        sum2 = 0.0
        for j in range(n):
            sum2 += theta[j] * tau[i, j] / np.sum(theta * tau[:, j])

        ln_gamma_R[i] = q[i] * (1 - np.log(sum1) - sum2)

    ln_gamma = ln_gamma_C + ln_gamma_R
    gamma = np.exp(ln_gamma)
    gamma_C = np.exp(ln_gamma_C)
    gamma_R = np.exp(ln_gamma_R)

    return {
        'gamma': gamma.tolist(),
        'gamma_C': gamma_C.tolist(),
        'gamma_R': gamma_R.tolist(),
        'ln_gamma': ln_gamma.tolist(),
        'phi': phi.tolist(),
        'theta': theta.tolist(),
        'model': 'uniquac',
    }


def unifac(x: np.ndarray, groups: List[Dict], T: float) -> Dict[str, Any]:
    """
    UNIFAC (UNIQUAC Functional-group Activity Coefficients) model.

    Simplified implementation - full UNIFAC requires extensive group tables.

    Parameters
    ----------
    x : array
        Mole fractions
    groups : list of dicts
        Group composition for each molecule
        [{'CH3': 2, 'CH2': 1}, {'CH3': 1, 'OH': 1}]
    T : float
        Temperature [K]

    Returns
    -------
    dict
        gamma: Activity coefficients (placeholder for full implementation)
        note: Implementation notes
    """
    # UNIFAC requires extensive group interaction parameter tables
    # This is a placeholder showing the structure

    return {
        'gamma': [1.0] * len(x),
        'note': 'Full UNIFAC requires group interaction parameter tables (Gmehling et al.)',
        'model': 'unifac',
        'temperature': T,
        'groups': groups,
        'reference': 'Fredenslund et al., Vapor-Liquid Equilibria using UNIFAC',
    }


def compute(x: np.ndarray = None, model: str = 'ideal', **kwargs) -> Dict[str, Any]:
    """
    Main entry point for activity coefficient calculations.

    Parameters
    ----------
    x : array
        Mole fractions
    model : str
        'ideal', 'margules', 'van_laar', 'wilson', 'nrtl', 'uniquac', 'unifac'
    **kwargs
        Model-specific parameters

    Returns
    -------
    dict
        gamma: Activity coefficients
        model: Model used
    """
    if x is None:
        return {'gamma': float('nan'), 'error': 'Mole fractions required'}

    x = np.asarray(x)

    if model == 'ideal':
        return ideal_solution(x)

    elif model == 'margules':
        if len(x) == 2:
            return margules(x[0], kwargs.get('A12', 0), kwargs.get('A21'))
        return {'gamma': float('nan'), 'error': 'Margules requires binary mixture'}

    elif model == 'van_laar':
        if len(x) == 2:
            return van_laar(x[0], kwargs.get('A', 0), kwargs.get('B', 0))
        return {'gamma': float('nan'), 'error': 'Van Laar requires binary mixture'}

    elif model == 'wilson':
        if 'Lambda' in kwargs:
            return wilson(x, kwargs['Lambda'])
        elif len(x) == 2 and 'Lambda12' in kwargs:
            return wilson_binary(x[0], kwargs['Lambda12'], kwargs.get('Lambda21', 1))
        return {'gamma': float('nan'), 'error': 'Wilson requires Lambda matrix or Lambda12/Lambda21'}

    elif model == 'nrtl':
        if 'tau' in kwargs and 'alpha' in kwargs:
            return nrtl(x, kwargs['tau'], kwargs['alpha'])
        elif len(x) == 2:
            return nrtl_binary(x[0], kwargs.get('tau12', 0), kwargs.get('tau21', 0),
                              kwargs.get('alpha', 0.3))
        return {'gamma': float('nan'), 'error': 'NRTL requires tau and alpha matrices'}

    elif model == 'uniquac':
        return uniquac(x, kwargs['r'], kwargs['q'], kwargs['tau'])

    elif model == 'unifac':
        return unifac(x, kwargs.get('groups', []), kwargs.get('T', 298.15))

    return {'gamma': float('nan'), 'error': f'Unknown model: {model}'}
