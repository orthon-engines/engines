"""
Fluid Mechanics Fundamentals

Core fluid mechanics for chemical engineering:

Conservation Laws:
    - Continuity: ∂ρ/∂t + ∇·(ρv) = 0
    - Momentum (Navier-Stokes): ρ(Dv/Dt) = -∇P + μ∇²v + ρg
    - Energy: ρCp(DT/Dt) = k∇²T + Φ

Key Equations:
    - Bernoulli: P/ρ + v²/2 + gz = constant
    - Hagen-Poiseuille (laminar pipe flow)
    - Darcy-Weisbach (friction losses)
    - Mechanical energy balance

Flow Regimes:
    - Laminar: Re < 2100
    - Transitional: 2100 < Re < 4000
    - Turbulent: Re > 4000
"""

import numpy as np
from typing import Dict, Optional


# Gravitational acceleration
g = 9.81  # m/s²


def continuity_equation(
    rho_1: float,
    v_1: float,
    A_1: float,
    rho_2: Optional[float] = None,
    v_2: Optional[float] = None,
    A_2: Optional[float] = None
) -> Dict[str, float]:
    """
    Continuity equation for steady flow.

    ρ₁v₁A₁ = ρ₂v₂A₂

    For incompressible: v₁A₁ = v₂A₂

    Parameters
    ----------
    rho_1 : float
        Density at point 1 [kg/m³]
    v_1 : float
        Velocity at point 1 [m/s]
    A_1 : float
        Cross-sectional area at point 1 [m²]
    rho_2, v_2, A_2 : float, optional
        Properties at point 2 (provide any 2 of 3)

    Returns
    -------
    dict
        mass_flow_rate: ṁ [kg/s]
        volumetric_flow_rate: Q [m³/s]
        missing variable at point 2
    """
    m_dot = rho_1 * v_1 * A_1  # Mass flow rate
    Q = v_1 * A_1  # Volumetric flow rate (if incompressible)

    results = {
        'mass_flow_rate': m_dot,
        'volumetric_flow_rate': Q
    }

    # Solve for missing variable at point 2
    if rho_2 is not None and v_2 is not None:
        A_2 = m_dot / (rho_2 * v_2)
        results['A_2'] = A_2
    elif rho_2 is not None and A_2 is not None:
        v_2 = m_dot / (rho_2 * A_2)
        results['v_2'] = v_2
    elif v_2 is not None and A_2 is not None:
        rho_2 = m_dot / (v_2 * A_2)
        results['rho_2'] = rho_2
    elif A_2 is not None:
        # Assume incompressible (rho_2 = rho_1)
        v_2 = Q / A_2
        results['v_2'] = v_2
        results['rho_2'] = rho_1

    return results


def bernoulli_equation(
    P_1: float,
    v_1: float,
    z_1: float,
    rho: float,
    P_2: Optional[float] = None,
    v_2: Optional[float] = None,
    z_2: Optional[float] = None,
    h_loss: float = 0.0
) -> Dict[str, float]:
    """
    Bernoulli equation (with optional head loss).

    P₁/ρg + v₁²/2g + z₁ = P₂/ρg + v₂²/2g + z₂ + h_loss

    Expressed as head [m]:
    Total head = pressure head + velocity head + elevation head

    Parameters
    ----------
    P_1 : float
        Pressure at point 1 [Pa]
    v_1 : float
        Velocity at point 1 [m/s]
    z_1 : float
        Elevation at point 1 [m]
    rho : float
        Fluid density [kg/m³]
    P_2, v_2, z_2 : float, optional
        Properties at point 2 (provide any 2 of 3)
    h_loss : float
        Head loss between points [m]

    Returns
    -------
    dict
        Missing variable and heads
    """
    # Convert to heads
    h_pressure_1 = P_1 / (rho * g)
    h_velocity_1 = v_1**2 / (2 * g)
    h_total_1 = h_pressure_1 + h_velocity_1 + z_1

    results = {
        'pressure_head_1': h_pressure_1,
        'velocity_head_1': h_velocity_1,
        'elevation_head_1': z_1,
        'total_head_1': h_total_1,
        'head_loss': h_loss
    }

    h_total_2 = h_total_1 - h_loss

    # Solve for missing variable
    if P_2 is not None and v_2 is not None:
        h_pressure_2 = P_2 / (rho * g)
        h_velocity_2 = v_2**2 / (2 * g)
        z_2 = h_total_2 - h_pressure_2 - h_velocity_2
        results['z_2'] = z_2
    elif P_2 is not None and z_2 is not None:
        h_pressure_2 = P_2 / (rho * g)
        h_velocity_2 = h_total_2 - h_pressure_2 - z_2
        v_2 = np.sqrt(2 * g * h_velocity_2) if h_velocity_2 >= 0 else np.nan
        results['v_2'] = v_2
    elif v_2 is not None and z_2 is not None:
        h_velocity_2 = v_2**2 / (2 * g)
        h_pressure_2 = h_total_2 - h_velocity_2 - z_2
        P_2 = h_pressure_2 * rho * g
        results['P_2'] = P_2
    else:
        results['total_head_2'] = h_total_2

    return results


def hagen_poiseuille(
    mu: float,
    L: float,
    D: float,
    delta_P: Optional[float] = None,
    Q: Optional[float] = None
) -> Dict[str, float]:
    """
    Hagen-Poiseuille equation for laminar pipe flow.

    Q = πD⁴ΔP / (128μL)

    Valid only for:
        - Re < 2100 (laminar flow)
        - Fully developed flow
        - Newtonian fluid

    Parameters
    ----------
    mu : float
        Dynamic viscosity [Pa·s]
    L : float
        Pipe length [m]
    D : float
        Pipe diameter [m]
    delta_P : float, optional
        Pressure drop [Pa]
    Q : float, optional
        Volumetric flow rate [m³/s]

    Returns
    -------
    dict
        Q or delta_P (whichever missing)
        average_velocity: v_avg
        wall_shear_stress: τ_w
    """
    if delta_P is not None:
        Q = np.pi * (D**4) * delta_P / (128 * mu * L)
    elif Q is not None:
        delta_P = 128 * mu * L * Q / (np.pi * D**4)
    else:
        return {'error': 'Provide either delta_P or Q'}

    A = np.pi * (D/2)**2
    v_avg = Q / A
    tau_w = D * delta_P / (4 * L)  # Wall shear stress

    return {
        'volumetric_flow_rate': Q,
        'pressure_drop': delta_P,
        'average_velocity': v_avg,
        'wall_shear_stress': tau_w,
        'max_velocity': 2 * v_avg,  # v_max = 2*v_avg for laminar
        'valid_regime': 'laminar (verify Re < 2100)'
    }


def friction_factor(
    Re: float,
    epsilon_D: float = 0.0
) -> Dict[str, float]:
    """
    Darcy friction factor for pipe flow.

    Laminar (Re < 2100): f = 64/Re

    Turbulent smooth (Blasius): f = 0.316/Re^0.25

    Turbulent rough (Swamee-Jain):
    f = 0.25 / [log₁₀(ε/3.7D + 5.74/Re^0.9)]²

    Parameters
    ----------
    Re : float
        Reynolds number
    epsilon_D : float
        Relative roughness ε/D (0 for smooth)

    Returns
    -------
    dict
        f: Darcy friction factor
        regime: flow regime
    """
    if Re < 2100:
        f = 64 / Re
        regime = 'laminar'
    elif Re < 4000:
        # Transitional - interpolate or use turbulent
        f = 0.316 / (Re ** 0.25)  # Blasius as approximation
        regime = 'transitional'
    else:
        if epsilon_D < 1e-6:
            # Smooth pipe (Blasius)
            f = 0.316 / (Re ** 0.25)
        else:
            # Swamee-Jain (explicit approximation to Colebrook)
            f = 0.25 / (np.log10(epsilon_D / 3.7 + 5.74 / (Re ** 0.9))) ** 2
        regime = 'turbulent'

    return {
        'friction_factor': f,
        'reynolds': Re,
        'relative_roughness': epsilon_D,
        'regime': regime
    }


def darcy_weisbach(
    f: float,
    L: float,
    D: float,
    v: float,
    rho: float
) -> Dict[str, float]:
    """
    Darcy-Weisbach equation for head/pressure loss.

    h_f = f * (L/D) * (v²/2g)
    ΔP = f * (L/D) * (ρv²/2)

    Parameters
    ----------
    f : float
        Darcy friction factor
    L : float
        Pipe length [m]
    D : float
        Pipe diameter [m]
    v : float
        Average velocity [m/s]
    rho : float
        Fluid density [kg/m³]

    Returns
    -------
    dict
        head_loss: h_f [m]
        pressure_drop: ΔP [Pa]
    """
    h_f = f * (L / D) * (v**2 / (2 * g))
    delta_P = f * (L / D) * (rho * v**2 / 2)

    return {
        'head_loss': h_f,
        'pressure_drop': delta_P,
        'friction_factor': f,
        'L_over_D': L / D
    }


def minor_losses(
    v: float,
    rho: float,
    K_values: Dict[str, float]
) -> Dict[str, float]:
    """
    Minor losses from fittings, valves, etc.

    h_m = K * v²/2g
    ΔP_m = K * ρv²/2

    Parameters
    ----------
    v : float
        Velocity [m/s]
    rho : float
        Density [kg/m³]
    K_values : dict
        {fitting_name: K_value}

    Returns
    -------
    dict
        Total head loss and pressure drop
        Individual contributions
    """
    velocity_head = v**2 / (2 * g)
    dynamic_pressure = rho * v**2 / 2

    K_total = sum(K_values.values())
    h_total = K_total * velocity_head
    delta_P_total = K_total * dynamic_pressure

    individual_losses = {name: K * velocity_head for name, K in K_values.items()}

    return {
        'total_head_loss': h_total,
        'total_pressure_drop': delta_P_total,
        'K_total': K_total,
        'individual_head_losses': individual_losses
    }


def pump_power(
    Q: float,
    H: float,
    rho: float,
    eta: float = 1.0
) -> Dict[str, float]:
    """
    Pump power calculation.

    Hydraulic power: P_h = ρgQH
    Shaft power: P_s = P_h / η

    Parameters
    ----------
    Q : float
        Volumetric flow rate [m³/s]
    H : float
        Total head [m]
    rho : float
        Fluid density [kg/m³]
    eta : float
        Pump efficiency (0 to 1)

    Returns
    -------
    dict
        hydraulic_power: P_h [W]
        shaft_power: P_s [W]
    """
    P_h = rho * g * Q * H
    P_s = P_h / eta if eta > 0 else np.inf

    return {
        'hydraulic_power_W': P_h,
        'hydraulic_power_kW': P_h / 1000,
        'shaft_power_W': P_s,
        'shaft_power_kW': P_s / 1000,
        'efficiency': eta
    }


def navier_stokes_1d_steady(
    mu: float,
    dp_dx: float,
    y: np.ndarray,
    H: float
) -> Dict[str, np.ndarray]:
    """
    1D steady Navier-Stokes solution (Couette/Poiseuille flow).

    For flow between parallel plates:
    μ(d²u/dy²) = dP/dx

    Solution: u(y) = (1/2μ)(dP/dx)(y² - Hy)

    Parameters
    ----------
    mu : float
        Dynamic viscosity [Pa·s]
    dp_dx : float
        Pressure gradient [Pa/m] (negative for flow in +x)
    y : array
        Position across channel [m]
    H : float
        Channel height [m]

    Returns
    -------
    dict
        velocity profile u(y)
        max velocity
        average velocity
    """
    u = (1 / (2 * mu)) * dp_dx * (y**2 - H * y)

    u_max = -(H**2 * dp_dx) / (8 * mu)
    u_avg = -(H**2 * dp_dx) / (12 * mu)  # u_avg = (2/3) * u_max

    return {
        'y': y,
        'velocity': u,
        'max_velocity': u_max,
        'average_velocity': u_avg,
        'flow_type': 'Poiseuille (pressure-driven)'
    }


def orifice_meter(
    D_pipe: float,
    D_orifice: float,
    delta_P: float,
    rho: float,
    C_d: float = 0.61
) -> Dict[str, float]:
    """
    Orifice meter flow measurement.

    Q = C_d * A_o * √(2ΔP / (ρ(1 - β⁴)))

    Where β = D_orifice / D_pipe

    Parameters
    ----------
    D_pipe : float
        Pipe diameter [m]
    D_orifice : float
        Orifice diameter [m]
    delta_P : float
        Pressure drop across orifice [Pa]
    rho : float
        Fluid density [kg/m³]
    C_d : float
        Discharge coefficient (typically 0.6-0.65)

    Returns
    -------
    dict
        Q: volumetric flow rate [m³/s]
        velocity: through orifice [m/s]
    """
    beta = D_orifice / D_pipe
    A_o = np.pi * (D_orifice / 2)**2

    Q = C_d * A_o * np.sqrt(2 * delta_P / (rho * (1 - beta**4)))
    v_o = Q / A_o

    return {
        'volumetric_flow_rate': Q,
        'orifice_velocity': v_o,
        'beta_ratio': beta,
        'discharge_coefficient': C_d
    }


def venturi_meter(
    D_pipe: float,
    D_throat: float,
    delta_P: float,
    rho: float,
    C_d: float = 0.98
) -> Dict[str, float]:
    """
    Venturi meter flow measurement.

    Same equation as orifice but higher C_d due to gradual contraction.

    Parameters
    ----------
    D_pipe : float
        Pipe diameter [m]
    D_throat : float
        Throat diameter [m]
    delta_P : float
        Pressure difference [Pa]
    rho : float
        Fluid density [kg/m³]
    C_d : float
        Discharge coefficient (typically 0.95-0.99)

    Returns
    -------
    dict
        Q: volumetric flow rate [m³/s]
    """
    beta = D_throat / D_pipe
    A_t = np.pi * (D_throat / 2)**2

    Q = C_d * A_t * np.sqrt(2 * delta_P / (rho * (1 - beta**4)))

    return {
        'volumetric_flow_rate': Q,
        'throat_velocity': Q / A_t,
        'beta_ratio': beta,
        'discharge_coefficient': C_d,
        'permanent_pressure_loss': 0.1 * delta_P  # ~10% for venturi
    }
