"""
Separation Processes
====================

Unit operation calculations for:
- Distillation (stage-by-stage)
- Absorption (mass transfer)
- Extraction (liquid-liquid)
- Membrane separation

These require iterative solvers - not expressible in SQL.
"""

import numpy as np
from typing import Dict, Any, List, Optional


def compute_distillation_stages(
    x_feed: float,
    x_distillate: float,
    x_bottoms: float,
    q: float,
    alpha: float,
    R: float
) -> Dict[str, Any]:
    """
    McCabe-Thiele distillation analysis.
    
    Args:
        x_feed: Feed composition (mole fraction light)
        x_distillate: Distillate composition target
        x_bottoms: Bottoms composition target
        q: Feed thermal condition (1=sat liquid, 0=sat vapor)
        alpha: Relative volatility
        R: Reflux ratio (L/D)
    
    Returns:
        n_stages: Number of theoretical stages
        feed_stage: Optimal feed stage location
    """
    # Operating lines
    def rectifying(x):
        return (R / (R + 1)) * x + x_distillate / (R + 1)
    
    def stripping(x):
        # Material balance determines slope
        V_bar = 1 - q  # Vapor below feed
        L_bar = R + q  # Liquid below feed
        slope = L_bar / V_bar if V_bar > 0 else float('inf')
        return slope * (x - x_bottoms) + x_bottoms
    
    def equilibrium(x):
        # y = alpha*x / (1 + (alpha-1)*x)
        return alpha * x / (1 + (alpha - 1) * x)
    
    # Step off stages from top
    stages = []
    x = x_distillate
    y = x_distillate
    
    feed_stage = None
    
    for i in range(100):  # Max iterations
        # Step to equilibrium
        # Solve: y = alpha*x_eq / (1 + (alpha-1)*x_eq)
        x_eq = y / (alpha - (alpha - 1) * y)
        
        stages.append({'stage': i + 1, 'x': x_eq, 'y': y})
        
        if x_eq <= x_bottoms:
            break
        
        # Step to operating line
        if x_eq > x_feed and feed_stage is None:
            y_next = rectifying(x_eq)
        else:
            if feed_stage is None:
                feed_stage = i + 1
            y_next = stripping(x_eq)
        
        x = x_eq
        y = y_next
    
    return {
        'n_stages': len(stages),
        'feed_stage': feed_stage or len(stages),
        'stages': stages
    }


def compute_absorption(
    y_in: float,
    y_out: float,
    x_in: float = 0.0,
    L_G: float = 1.5,
    H: float = 1.0
) -> Dict[str, Any]:
    """
    Gas absorption column design.
    
    Args:
        y_in: Inlet gas composition
        y_out: Target outlet gas composition
        x_in: Inlet liquid composition
        L_G: Liquid to gas ratio
        H: Henry's law constant (y = H*x)
    
    Returns:
        NTU: Number of transfer units
        HTU: Height of transfer unit (assumed = 1)
        height: Total column height
    """
    # Absorption factor
    A = L_G / H
    
    # Number of transfer units
    if abs(A - 1) < 0.01:
        NTU = (y_in - y_out) / (y_out - H * x_in)
    else:
        y_star_in = H * x_in
        NTU = np.log((y_in - y_star_in) / (y_out - y_star_in) * (1 - 1/A) + 1/A) / (1 - 1/A)
    
    HTU = 1.0  # Would come from correlations
    height = NTU * HTU
    
    return {
        'NTU': float(NTU),
        'HTU': float(HTU),
        'height': float(height),
        'absorption_factor': float(A)
    }


def compute(operation: str, **kwargs) -> Dict[str, Any]:
    """
    Run separation calculation.
    
    Args:
        operation: 'distillation' or 'absorption'
    """
    if operation == 'distillation':
        return compute_distillation_stages(**kwargs)
    elif operation == 'absorption':
        return compute_absorption(**kwargs)
    else:
        raise ValueError(f"Unknown operation: {operation}")
