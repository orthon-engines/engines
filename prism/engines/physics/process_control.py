"""
Process Control Fundamentals

Core process control concepts for ChemE:

Transfer Functions:
    - First order: G(s) = K / (τs + 1)
    - Second order: G(s) = K / (τ²s² + 2ζτs + 1)
    - Time delay: e^(-θs)

PID Control:
    - P: Gc(s) = Kc
    - PI: Gc(s) = Kc(1 + 1/(τI*s))
    - PID: Gc(s) = Kc(1 + 1/(τI*s) + τD*s)

Stability Analysis:
    - Poles and zeros
    - Routh-Hurwitz criterion
    - Bode plots (gain margin, phase margin)

Tuning Methods:
    - Ziegler-Nichols
    - Cohen-Coon
    - IMC (Internal Model Control)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def first_order_response(
    K: float,
    tau: float,
    t: np.ndarray,
    step_size: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    First-order system step response.

    G(s) = K / (τs + 1)

    y(t) = K * step_size * (1 - exp(-t/τ))

    Parameters
    ----------
    K : float
        Steady-state gain
    tau : float
        Time constant [s]
    t : array
        Time values [s]
    step_size : float
        Magnitude of step input

    Returns
    -------
    dict
        y: response values
        time_to_63pct: τ (time to reach 63.2% of final)
        time_to_95pct: 3τ (time to reach 95%)
        time_to_99pct: 5τ (time to reach 99%)
    """
    y = K * step_size * (1 - np.exp(-t / tau))

    return {
        'time': t,
        'response': y,
        'final_value': K * step_size,
        'time_constant': tau,
        'time_to_63pct': tau,
        'time_to_95pct': 3 * tau,
        'time_to_99pct': 5 * tau
    }


def second_order_response(
    K: float,
    tau: float,
    zeta: float,
    t: np.ndarray,
    step_size: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Second-order system step response.

    G(s) = K / (τ²s² + 2ζτs + 1)

    Parameters
    ----------
    K : float
        Steady-state gain
    tau : float
        Time constant [s]
    zeta : float
        Damping ratio (ζ)
        - ζ < 1: underdamped (oscillatory)
        - ζ = 1: critically damped
        - ζ > 1: overdamped
    t : array
        Time values [s]
    step_size : float
        Step input magnitude

    Returns
    -------
    dict
        y: response values
        overshoot: percent overshoot (if underdamped)
        settling_time: time to stay within 2% of final
    """
    y_ss = K * step_size

    if zeta < 1:
        # Underdamped
        omega_d = np.sqrt(1 - zeta**2) / tau
        y = y_ss * (1 - np.exp(-zeta * t / tau) * (
            np.cos(omega_d * t) + zeta / np.sqrt(1 - zeta**2) * np.sin(omega_d * t)
        ))

        # Overshoot
        overshoot = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
        peak_time = np.pi * tau / np.sqrt(1 - zeta**2)
        settling_time = 4 * tau / zeta  # 2% criterion

    elif zeta == 1:
        # Critically damped
        y = y_ss * (1 - (1 + t/tau) * np.exp(-t/tau))
        overshoot = 0.0
        peak_time = np.nan
        settling_time = 4.75 * tau

    else:
        # Overdamped
        r1 = (-zeta + np.sqrt(zeta**2 - 1)) / tau
        r2 = (-zeta - np.sqrt(zeta**2 - 1)) / tau
        tau1 = -1 / r1
        tau2 = -1 / r2
        y = y_ss * (1 - tau1 / (tau1 - tau2) * np.exp(-t/tau1) + tau2 / (tau1 - tau2) * np.exp(-t/tau2))
        overshoot = 0.0
        peak_time = np.nan
        settling_time = 4 * max(tau1, tau2)

    return {
        'time': t,
        'response': y,
        'final_value': y_ss,
        'damping_ratio': zeta,
        'damping_type': 'underdamped' if zeta < 1 else ('critically_damped' if zeta == 1 else 'overdamped'),
        'percent_overshoot': overshoot * 100 if zeta < 1 else 0,
        'peak_time': peak_time,
        'settling_time_2pct': settling_time,
        'natural_frequency': 1 / tau
    }


def time_delay_pade(
    theta: float,
    order: int = 1
) -> Dict[str, Tuple]:
    """
    Padé approximation for time delay.

    e^(-θs) ≈ N(s) / D(s)

    First-order: (1 - θs/2) / (1 + θs/2)
    Second-order: (1 - θs/2 + θ²s²/12) / (1 + θs/2 + θ²s²/12)

    Parameters
    ----------
    theta : float
        Time delay [s]
    order : int
        Order of approximation (1 or 2)

    Returns
    -------
    dict
        numerator: coefficients [a_n, ..., a_1, a_0]
        denominator: coefficients [b_n, ..., b_1, b_0]
    """
    if order == 1:
        num = (1, -theta/2)  # 1 - θs/2
        den = (1, theta/2)   # 1 + θs/2
    elif order == 2:
        num = (theta**2/12, -theta/2, 1)  # θ²s²/12 - θs/2 + 1
        den = (theta**2/12, theta/2, 1)   # θ²s²/12 + θs/2 + 1
    else:
        return {'error': 'Only orders 1 and 2 implemented'}

    return {
        'numerator': num,
        'denominator': den,
        'time_delay': theta,
        'approximation_order': order
    }


def pid_controller(
    Kc: float,
    tau_I: Optional[float] = None,
    tau_D: Optional[float] = None
) -> Dict[str, str]:
    """
    PID controller transfer function.

    P:   Gc(s) = Kc
    PI:  Gc(s) = Kc(1 + 1/(τI*s)) = Kc(τI*s + 1)/(τI*s)
    PID: Gc(s) = Kc(1 + 1/(τI*s) + τD*s)

    Parameters
    ----------
    Kc : float
        Controller gain
    tau_I : float, optional
        Integral time [s]
    tau_D : float, optional
        Derivative time [s]

    Returns
    -------
    dict
        controller_type: P, PI, PD, or PID
        transfer_function: symbolic representation
    """
    if tau_I is None and tau_D is None:
        controller_type = 'P'
        tf = f'{Kc}'
    elif tau_D is None:
        controller_type = 'PI'
        tf = f'{Kc}(1 + 1/({tau_I}s))'
    elif tau_I is None:
        controller_type = 'PD'
        tf = f'{Kc}(1 + {tau_D}s)'
    else:
        controller_type = 'PID'
        tf = f'{Kc}(1 + 1/({tau_I}s) + {tau_D}s)'

    return {
        'controller_type': controller_type,
        'Kc': Kc,
        'tau_I': tau_I,
        'tau_D': tau_D,
        'transfer_function': tf
    }


def ziegler_nichols_closed_loop(
    Kcu: float,
    Pu: float,
    controller_type: str = 'PID'
) -> Dict[str, float]:
    """
    Ziegler-Nichols closed-loop (ultimate gain) tuning.

    From sustained oscillations at Kc = Kcu with period Pu:

    P:   Kc = 0.5*Kcu
    PI:  Kc = 0.45*Kcu, τI = Pu/1.2
    PID: Kc = 0.6*Kcu, τI = Pu/2, τD = Pu/8

    Parameters
    ----------
    Kcu : float
        Ultimate gain (gain at marginal stability)
    Pu : float
        Ultimate period [s]
    controller_type : str
        'P', 'PI', or 'PID'

    Returns
    -------
    dict
        Kc, tau_I, tau_D: tuning parameters
    """
    if controller_type == 'P':
        Kc = 0.5 * Kcu
        tau_I = None
        tau_D = None
    elif controller_type == 'PI':
        Kc = 0.45 * Kcu
        tau_I = Pu / 1.2
        tau_D = None
    elif controller_type == 'PID':
        Kc = 0.6 * Kcu
        tau_I = Pu / 2
        tau_D = Pu / 8
    else:
        return {'error': 'Unknown controller type'}

    return {
        'Kc': Kc,
        'tau_I': tau_I,
        'tau_D': tau_D,
        'controller_type': controller_type,
        'Kcu': Kcu,
        'Pu': Pu,
        'method': 'Ziegler-Nichols (closed-loop)'
    }


def ziegler_nichols_open_loop(
    K: float,
    tau: float,
    theta: float,
    controller_type: str = 'PID'
) -> Dict[str, float]:
    """
    Ziegler-Nichols open-loop (process reaction curve) tuning.

    From step response: gain K, time constant τ, delay θ

    P:   Kc = τ/(K*θ)
    PI:  Kc = 0.9*τ/(K*θ), τI = 3.33*θ
    PID: Kc = 1.2*τ/(K*θ), τI = 2*θ, τD = 0.5*θ

    Parameters
    ----------
    K : float
        Process gain
    tau : float
        Time constant [s]
    theta : float
        Time delay [s]

    Returns
    -------
    dict
        Kc, tau_I, tau_D: tuning parameters
    """
    if controller_type == 'P':
        Kc = tau / (K * theta)
        tau_I = None
        tau_D = None
    elif controller_type == 'PI':
        Kc = 0.9 * tau / (K * theta)
        tau_I = 3.33 * theta
        tau_D = None
    elif controller_type == 'PID':
        Kc = 1.2 * tau / (K * theta)
        tau_I = 2 * theta
        tau_D = 0.5 * theta
    else:
        return {'error': 'Unknown controller type'}

    return {
        'Kc': Kc,
        'tau_I': tau_I,
        'tau_D': tau_D,
        'controller_type': controller_type,
        'K': K,
        'tau': tau,
        'theta': theta,
        'method': 'Ziegler-Nichols (open-loop)'
    }


def imc_tuning(
    K: float,
    tau: float,
    theta: float,
    tau_c: Optional[float] = None,
    controller_type: str = 'PI'
) -> Dict[str, float]:
    """
    Internal Model Control (IMC) tuning.

    For FOPDT process: G(s) = K*exp(-θs) / (τs + 1)

    Closed-loop time constant τc determines aggressiveness.
    Rule of thumb: τc ≥ max(0.1τ, 0.8θ)

    PI (θ ≈ 0):  Kc = τ/(K*τc), τI = τ
    PID:         Kc = (2τ + θ)/(2K*(τc + θ)), τI = τ + θ/2, τD = τθ/(2τ + θ)

    Parameters
    ----------
    K : float
        Process gain
    tau : float
        Process time constant [s]
    theta : float
        Process time delay [s]
    tau_c : float, optional
        Closed-loop time constant [s]
    controller_type : str
        'PI' or 'PID'

    Returns
    -------
    dict
        Kc, tau_I, tau_D: tuning parameters
    """
    if tau_c is None:
        tau_c = max(0.1 * tau, 0.8 * theta, 1.0)

    if controller_type == 'PI':
        Kc = tau / (K * (tau_c + theta))
        tau_I = tau
        tau_D = None
    elif controller_type == 'PID':
        Kc = (2 * tau + theta) / (2 * K * (tau_c + theta))
        tau_I = tau + theta / 2
        tau_D = tau * theta / (2 * tau + theta)
    else:
        return {'error': 'Unknown controller type'}

    return {
        'Kc': Kc,
        'tau_I': tau_I,
        'tau_D': tau_D,
        'tau_c': tau_c,
        'controller_type': controller_type,
        'method': 'IMC'
    }


def stability_margins(
    gain_crossover_freq: float,
    phase_crossover_freq: float,
    gain_at_phase_crossover: float,
    phase_at_gain_crossover: float
) -> Dict[str, float]:
    """
    Calculate gain margin and phase margin.

    Gain Margin (GM): Additional gain before instability
        GM = 1 / |G(jω_pc)| where phase = -180°

    Phase Margin (PM): Additional phase lag before instability
        PM = 180° + phase(G(jω_gc)) where |G| = 1

    Parameters
    ----------
    gain_crossover_freq : float
        Frequency where |G(jω)| = 1 [rad/s]
    phase_crossover_freq : float
        Frequency where phase = -180° [rad/s]
    gain_at_phase_crossover : float
        |G(jω)| at phase crossover
    phase_at_gain_crossover : float
        Phase [degrees] at gain crossover

    Returns
    -------
    dict
        gain_margin: GM
        phase_margin: PM [degrees]
        stable: whether system has positive margins
    """
    GM = 1 / gain_at_phase_crossover if gain_at_phase_crossover > 0 else np.inf
    GM_dB = 20 * np.log10(GM) if GM > 0 and GM != np.inf else np.nan
    PM = 180 + phase_at_gain_crossover

    return {
        'gain_margin': GM,
        'gain_margin_dB': GM_dB,
        'phase_margin_deg': PM,
        'gain_crossover_freq': gain_crossover_freq,
        'phase_crossover_freq': phase_crossover_freq,
        'stable': GM > 1 and PM > 0
    }


def poles_and_zeros(
    numerator: List[float],
    denominator: List[float]
) -> Dict[str, np.ndarray]:
    """
    Find poles and zeros of transfer function.

    G(s) = (b_m*s^m + ... + b_0) / (a_n*s^n + ... + a_0)

    Zeros: roots of numerator
    Poles: roots of denominator

    Stability: all poles must have negative real parts.

    Parameters
    ----------
    numerator : list
        Coefficients [b_m, b_{m-1}, ..., b_0]
    denominator : list
        Coefficients [a_n, a_{n-1}, ..., a_0]

    Returns
    -------
    dict
        zeros: array of zeros
        poles: array of poles
        stable: True if all poles have Re < 0
    """
    zeros = np.roots(numerator) if len(numerator) > 1 else np.array([])
    poles = np.roots(denominator)

    stable = all(np.real(poles) < 0)
    dominant_pole = poles[np.argmax(np.real(poles))]  # Closest to imaginary axis

    return {
        'zeros': zeros,
        'poles': poles,
        'stable': stable,
        'dominant_pole': dominant_pole,
        'dominant_time_constant': -1 / np.real(dominant_pole) if np.real(dominant_pole) < 0 else np.inf
    }


def closed_loop_transfer_function(
    Gp_num: List[float],
    Gp_den: List[float],
    Gc_num: List[float],
    Gc_den: List[float]
) -> Dict[str, List[float]]:
    """
    Closed-loop transfer function with unity feedback.

    G_cl(s) = Gc*Gp / (1 + Gc*Gp)

    Parameters
    ----------
    Gp_num : list
        Process numerator coefficients
    Gp_den : list
        Process denominator coefficients
    Gc_num : list
        Controller numerator coefficients
    Gc_den : list
        Controller denominator coefficients

    Returns
    -------
    dict
        numerator: closed-loop numerator
        denominator: closed-loop denominator
    """
    # Multiply numerators and denominators
    open_loop_num = np.convolve(Gc_num, Gp_num)
    open_loop_den = np.convolve(Gc_den, Gp_den)

    # Closed-loop: num / (den + num)
    cl_num = open_loop_num
    cl_den = np.polyadd(open_loop_den, open_loop_num)

    return {
        'numerator': cl_num.tolist(),
        'denominator': cl_den.tolist(),
        'open_loop_numerator': open_loop_num.tolist(),
        'open_loop_denominator': open_loop_den.tolist()
    }
