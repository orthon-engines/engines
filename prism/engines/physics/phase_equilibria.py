"""
Phase Equilibria
================

Thermodynamic phase equilibrium calculations:
- VLE (Vapor-Liquid Equilibrium)
- LLE (Liquid-Liquid Equilibrium)
- Flash calculations

Stream mode: These are typically single-point calculations,
invoked per thermodynamic state.
"""

import numpy as np
from typing import Dict, Any, List, Optional


def compute_vle(
    T: float,
    P: float,
    z: List[float],
    K_values: Optional[List[float]] = None,
    antoine_params: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Vapor-Liquid Equilibrium calculation.
    
    Args:
        T: Temperature [K]
        P: Pressure [Pa]
        z: Feed composition (mole fractions)
        K_values: Equilibrium ratios (if known)
        antoine_params: Antoine equation parameters for each component
    
    Returns:
        V: Vapor fraction
        x: Liquid composition
        y: Vapor composition
    """
    z = np.array(z)
    n = len(z)
    
    # Calculate K values if not provided
    if K_values is None:
        if antoine_params is None:
            # Assume ideal Raoult's law with rough estimates
            K_values = np.ones(n)
        else:
            # Antoine equation: log10(Psat) = A - B/(C+T)
            K_values = []
            for params in antoine_params:
                A, B, C = params['A'], params['B'], params['C']
                Psat = 10 ** (A - B / (C + T))  # Pa
                K_values.append(Psat / P)
            K_values = np.array(K_values)
    else:
        K_values = np.array(K_values)
    
    # Rachford-Rice equation solver
    def rachford_rice(V):
        return np.sum(z * (K_values - 1) / (1 + V * (K_values - 1)))
    
    # Check if single phase
    f0 = rachford_rice(0)
    f1 = rachford_rice(1)
    
    if f0 < 0:
        # All liquid
        return {'V': 0.0, 'x': z.tolist(), 'y': z.tolist(), 'phase': 'liquid'}
    if f1 > 0:
        # All vapor
        return {'V': 1.0, 'x': z.tolist(), 'y': z.tolist(), 'phase': 'vapor'}
    
    # Two-phase: solve by bisection
    V_low, V_high = 0.0, 1.0
    for _ in range(50):
        V_mid = (V_low + V_high) / 2
        f_mid = rachford_rice(V_mid)
        if abs(f_mid) < 1e-10:
            break
        if f_mid > 0:
            V_low = V_mid
        else:
            V_high = V_mid
    
    V = V_mid
    x = z / (1 + V * (K_values - 1))
    y = K_values * x
    
    return {
        'V': float(V),
        'x': x.tolist(),
        'y': y.tolist(),
        'K_values': K_values.tolist(),
        'phase': 'two_phase'
    }


def compute_flash(
    z: List[float],
    K_values: List[float],
    tol: float = 1e-8,
    max_iter: int = 50
) -> Dict[str, Any]:
    """
    Isothermal flash calculation.
    
    Solves Rachford-Rice equation to find vapor fraction
    and phase compositions.
    """
    z = np.array(z)
    K = np.array(K_values)
    
    # Rachford-Rice
    def RR(V):
        return np.sum(z * (K - 1) / (1 + V * (K - 1)))
    
    # Initial bounds check
    f0, f1 = RR(0), RR(1)
    
    if f0 <= 0:
        x = z.copy()
        y = K * x / np.sum(K * x)
        return {'V': 0.0, 'x': x.tolist(), 'y': y.tolist()}
    
    if f1 >= 0:
        y = z.copy()
        x = y / K
        x = x / np.sum(x)
        return {'V': 1.0, 'x': x.tolist(), 'y': y.tolist()}
    
    # Bisection
    V_lo, V_hi = 0.0, 1.0
    for _ in range(max_iter):
        V = (V_lo + V_hi) / 2
        f = RR(V)
        if abs(f) < tol:
            break
        if f > 0:
            V_lo = V
        else:
            V_hi = V
    
    x = z / (1 + V * (K - 1))
    y = K * x
    
    return {
        'V': float(V),
        'x': x.tolist(),
        'y': y.tolist()
    }


def compute(calculation: str, **kwargs) -> Dict[str, Any]:
    """
    Run phase equilibria calculation.
    
    Args:
        calculation: 'vle' or 'flash'
    """
    if calculation == 'vle':
        return compute_vle(**kwargs)
    elif calculation == 'flash':
        return compute_flash(**kwargs)
    else:
        raise ValueError(f"Unknown calculation: {calculation}")
