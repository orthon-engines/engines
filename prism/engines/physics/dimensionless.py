"""
Dimensionless Numbers - The ChemE Alphabet

Every ChemE student memorizes these ratios. Now they compute them.

Numbers included:
    Re - Reynolds (inertia/viscous) - in reynolds.py
    Pr - Prandtl (momentum/thermal diffusivity)
    Sc - Schmidt (momentum/mass diffusivity)
    Nu - Nusselt (convective/conductive heat)
    Sh - Sherwood (convective/diffusive mass)
    Pe - Peclet (advection/diffusion)
    Da - Damköhler (reaction/transport)
    We - Weber (inertia/surface tension)
    Fr - Froude (inertia/gravity)
    Gr - Grashof (buoyancy/viscous)
    Ra - Rayleigh (Gr × Pr, natural convection)
    Bi - Biot (surface/internal thermal resistance)
    St - Stanton (heat transfer/thermal capacity)
    Le - Lewis (thermal/mass diffusivity)

All equations are THE equations. No approximations.
"""

import numpy as np
from typing import Dict, Any, Optional, Union


def compute_prandtl(
    kinematic_viscosity: float = None,
    thermal_diffusivity: float = None,
    dynamic_viscosity: float = None,
    heat_capacity: float = None,
    thermal_conductivity: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Prandtl Number: Pr = ν/α = Cp·μ/k

    Ratio of momentum diffusivity to thermal diffusivity.

    Args:
        kinematic_viscosity: ν [m²/s]
        thermal_diffusivity: α [m²/s]
        OR
        dynamic_viscosity: μ [Pa·s]
        heat_capacity: Cp [J/(kg·K)]
        thermal_conductivity: k [W/(m·K)]

    Typical values:
        Liquid metals: 0.01-0.03
        Gases: 0.7-1.0
        Water: ~7
        Oils: 100-10000
    """
    if kinematic_viscosity is not None and thermal_diffusivity is not None:
        Pr = kinematic_viscosity / thermal_diffusivity
    elif dynamic_viscosity is not None and heat_capacity is not None and thermal_conductivity is not None:
        Pr = (heat_capacity * dynamic_viscosity) / thermal_conductivity
    else:
        return {'prandtl': None, 'error': 'Need (ν, α) or (μ, Cp, k)'}

    return {
        'prandtl': float(Pr),
        'momentum_dominated': Pr > 1,
        'thermal_dominated': Pr < 1,
    }


def compute_schmidt(
    kinematic_viscosity: float = None,
    mass_diffusivity: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Schmidt Number: Sc = ν/D

    Ratio of momentum diffusivity to mass diffusivity.
    Mass transfer analog of Prandtl.

    Args:
        kinematic_viscosity: ν [m²/s]
        mass_diffusivity: D [m²/s]

    Typical values:
        Gases: ~1
        Liquids: 100-10000
    """
    if kinematic_viscosity is None or mass_diffusivity is None:
        return {'schmidt': None, 'error': 'Need ν and D'}

    Sc = kinematic_viscosity / mass_diffusivity

    return {
        'schmidt': float(Sc),
        'momentum_dominated': Sc > 1,
        'mass_dominated': Sc < 1,
    }


def compute_nusselt(
    heat_transfer_coeff: float = None,
    characteristic_length: float = None,
    thermal_conductivity: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Nusselt Number: Nu = hL/k

    Ratio of convective to conductive heat transfer.

    Args:
        heat_transfer_coeff: h [W/(m²·K)]
        characteristic_length: L [m]
        thermal_conductivity: k [W/(m·K)]

    Nu = 1 means pure conduction.
    Nu > 1 means convection enhances heat transfer.
    """
    if None in [heat_transfer_coeff, characteristic_length, thermal_conductivity]:
        return {'nusselt': None, 'error': 'Need h, L, and k'}

    Nu = (heat_transfer_coeff * characteristic_length) / thermal_conductivity

    return {
        'nusselt': float(Nu),
        'convection_dominated': Nu > 1,
    }


def compute_sherwood(
    mass_transfer_coeff: float = None,
    characteristic_length: float = None,
    mass_diffusivity: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Sherwood Number: Sh = kL/D

    Ratio of convective to diffusive mass transfer.
    Mass transfer analog of Nusselt.

    Args:
        mass_transfer_coeff: k [m/s]
        characteristic_length: L [m]
        mass_diffusivity: D [m²/s]
    """
    if None in [mass_transfer_coeff, characteristic_length, mass_diffusivity]:
        return {'sherwood': None, 'error': 'Need k, L, and D'}

    Sh = (mass_transfer_coeff * characteristic_length) / mass_diffusivity

    return {
        'sherwood': float(Sh),
        'convection_dominated': Sh > 1,
    }


def compute_peclet(
    velocity: float = None,
    characteristic_length: float = None,
    diffusivity: float = None,
    peclet_type: str = 'thermal',
    **kwargs,
) -> Dict[str, Any]:
    """
    Peclet Number: Pe = vL/α (thermal) or Pe = vL/D (mass)

    Ratio of advective to diffusive transport.
    Pe = Re × Pr (thermal) or Pe = Re × Sc (mass)

    Args:
        velocity: v [m/s]
        characteristic_length: L [m]
        diffusivity: α or D [m²/s]
        peclet_type: 'thermal' or 'mass'
    """
    if None in [velocity, characteristic_length, diffusivity]:
        return {'peclet': None, 'error': 'Need v, L, and diffusivity'}

    Pe = (velocity * characteristic_length) / diffusivity

    return {
        'peclet': float(Pe),
        'peclet_type': peclet_type,
        'advection_dominated': Pe > 1,
        'diffusion_dominated': Pe < 1,
    }


def compute_damkohler(
    reaction_rate: float = None,
    transport_rate: float = None,
    residence_time: float = None,
    rate_constant: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Damköhler Number: Da = reaction rate / transport rate

    For first-order: Da = k·τ (rate constant × residence time)

    Args:
        reaction_rate: [1/s] or [mol/(m³·s)]
        transport_rate: [1/s]
        OR
        residence_time: τ [s]
        rate_constant: k [1/s]

    Da >> 1: Reaction fast, transport limited
    Da << 1: Reaction slow, kinetically limited
    """
    if reaction_rate is not None and transport_rate is not None:
        Da = reaction_rate / transport_rate
    elif residence_time is not None and rate_constant is not None:
        Da = rate_constant * residence_time
    else:
        return {'damkohler': None, 'error': 'Need (reaction_rate, transport_rate) or (k, τ)'}

    return {
        'damkohler': float(Da),
        'transport_limited': Da > 1,
        'kinetically_limited': Da < 1,
    }


def compute_weber(
    density: float = None,
    velocity: float = None,
    characteristic_length: float = None,
    surface_tension: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Weber Number: We = ρv²L/σ

    Ratio of inertial to surface tension forces.
    Important for droplets, bubbles, sprays.

    Args:
        density: ρ [kg/m³]
        velocity: v [m/s]
        characteristic_length: L [m] (droplet diameter)
        surface_tension: σ [N/m]

    We >> 1: Inertia dominates, droplet breakup
    We << 1: Surface tension dominates, stable droplets
    """
    if None in [density, velocity, characteristic_length, surface_tension]:
        return {'weber': None, 'error': 'Need ρ, v, L, and σ'}

    We = (density * velocity**2 * characteristic_length) / surface_tension

    return {
        'weber': float(We),
        'inertia_dominated': We > 1,
        'surface_tension_dominated': We < 1,
        'droplet_breakup_likely': We > 12,  # Critical Weber for breakup
    }


def compute_froude(
    velocity: float = None,
    characteristic_length: float = None,
    gravity: float = 9.81,
    **kwargs,
) -> Dict[str, Any]:
    """
    Froude Number: Fr = v/√(gL)

    Ratio of inertial to gravitational forces.
    Important for open channel flow, ship hydrodynamics.

    Args:
        velocity: v [m/s]
        characteristic_length: L [m]
        gravity: g [m/s²] (default 9.81)

    Fr < 1: Subcritical (tranquil) flow
    Fr = 1: Critical flow
    Fr > 1: Supercritical (rapid) flow
    """
    if None in [velocity, characteristic_length]:
        return {'froude': None, 'error': 'Need v and L'}

    Fr = velocity / np.sqrt(gravity * characteristic_length)

    return {
        'froude': float(Fr),
        'subcritical': Fr < 1,
        'critical': abs(Fr - 1) < 0.01,
        'supercritical': Fr > 1,
    }


def compute_grashof(
    gravity: float = 9.81,
    beta: float = None,
    delta_T: float = None,
    characteristic_length: float = None,
    kinematic_viscosity: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Grashof Number: Gr = gβΔTL³/ν²

    Ratio of buoyancy to viscous forces.
    Drives natural convection.

    Args:
        gravity: g [m/s²]
        beta: β [1/K] thermal expansion coefficient
        delta_T: ΔT [K] temperature difference
        characteristic_length: L [m]
        kinematic_viscosity: ν [m²/s]

    For ideal gas: β = 1/T (absolute temperature)
    """
    if None in [beta, delta_T, characteristic_length, kinematic_viscosity]:
        return {'grashof': None, 'error': 'Need β, ΔT, L, and ν'}

    Gr = (gravity * beta * delta_T * characteristic_length**3) / kinematic_viscosity**2

    return {
        'grashof': float(Gr),
        'buoyancy_dominated': Gr > 1e4,
    }


def compute_rayleigh(
    grashof: float = None,
    prandtl: float = None,
    gravity: float = 9.81,
    beta: float = None,
    delta_T: float = None,
    characteristic_length: float = None,
    kinematic_viscosity: float = None,
    thermal_diffusivity: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Rayleigh Number: Ra = Gr × Pr = gβΔTL³/(να)

    Combined buoyancy and diffusion parameter.
    Critical for natural convection onset.

    Args:
        grashof: Gr (if already computed)
        prandtl: Pr (if already computed)
        OR compute from primitives

    Ra < 10³: Conduction dominates
    Ra > 10⁶: Turbulent natural convection
    """
    if grashof is not None and prandtl is not None:
        Ra = grashof * prandtl
    elif all(v is not None for v in [beta, delta_T, characteristic_length, kinematic_viscosity, thermal_diffusivity]):
        Ra = (gravity * beta * delta_T * characteristic_length**3) / (kinematic_viscosity * thermal_diffusivity)
    else:
        return {'rayleigh': None, 'error': 'Need (Gr, Pr) or (β, ΔT, L, ν, α)'}

    return {
        'rayleigh': float(Ra),
        'conduction_regime': Ra < 1e3,
        'laminar_convection': 1e3 < Ra < 1e6,
        'turbulent_convection': Ra > 1e6,
    }


def compute_biot(
    heat_transfer_coeff: float = None,
    characteristic_length: float = None,
    thermal_conductivity_solid: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Biot Number: Bi = hL/k_solid

    Ratio of surface to internal thermal resistance.
    Determines if lumped capacitance is valid.

    Args:
        heat_transfer_coeff: h [W/(m²·K)]
        characteristic_length: L [m] (V/A for lumped)
        thermal_conductivity_solid: k [W/(m·K)]

    Bi < 0.1: Lumped capacitance valid (uniform T)
    Bi > 0.1: Internal temperature gradients matter
    """
    if None in [heat_transfer_coeff, characteristic_length, thermal_conductivity_solid]:
        return {'biot': None, 'error': 'Need h, L, and k_solid'}

    Bi = (heat_transfer_coeff * characteristic_length) / thermal_conductivity_solid

    return {
        'biot': float(Bi),
        'lumped_valid': Bi < 0.1,
        'internal_gradients': Bi > 0.1,
    }


def compute_lewis(
    thermal_diffusivity: float = None,
    mass_diffusivity: float = None,
    schmidt: float = None,
    prandtl: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Lewis Number: Le = α/D = Sc/Pr

    Ratio of thermal to mass diffusivity.
    Important for simultaneous heat and mass transfer.

    Args:
        thermal_diffusivity: α [m²/s]
        mass_diffusivity: D [m²/s]
        OR
        schmidt: Sc
        prandtl: Pr

    Le = 1: Heat and mass transfer at same rate
    Le > 1: Heat diffuses faster than mass
    Le < 1: Mass diffuses faster than heat
    """
    if thermal_diffusivity is not None and mass_diffusivity is not None:
        Le = thermal_diffusivity / mass_diffusivity
    elif schmidt is not None and prandtl is not None:
        Le = schmidt / prandtl
    else:
        return {'lewis': None, 'error': 'Need (α, D) or (Sc, Pr)'}

    return {
        'lewis': float(Le),
        'heat_diffuses_faster': Le > 1,
        'mass_diffuses_faster': Le < 1,
    }


def compute_stanton(
    nusselt: float = None,
    reynolds: float = None,
    prandtl: float = None,
    heat_transfer_coeff: float = None,
    density: float = None,
    velocity: float = None,
    heat_capacity: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Stanton Number: St = Nu/(Re·Pr) = h/(ρvCp)

    Ratio of heat transfer to thermal capacity of flow.
    Modified Nusselt for flow systems.

    Args:
        nusselt, reynolds, prandtl: Dimensionless numbers
        OR
        heat_transfer_coeff: h [W/(m²·K)]
        density: ρ [kg/m³]
        velocity: v [m/s]
        heat_capacity: Cp [J/(kg·K)]
    """
    if nusselt is not None and reynolds is not None and prandtl is not None:
        if reynolds * prandtl == 0:
            return {'stanton': None, 'error': 'Re·Pr cannot be zero'}
        St = nusselt / (reynolds * prandtl)
    elif all(v is not None for v in [heat_transfer_coeff, density, velocity, heat_capacity]):
        St = heat_transfer_coeff / (density * velocity * heat_capacity)
    else:
        return {'stanton': None, 'error': 'Need (Nu, Re, Pr) or (h, ρ, v, Cp)'}

    return {
        'stanton': float(St),
    }


# Convenience function to compute all applicable numbers
def compute_all(constants: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all dimensionless numbers possible from given constants.
    """
    results = {}

    # Try each number
    pr = compute_prandtl(**constants)
    if pr.get('prandtl') is not None:
        results['prandtl'] = pr['prandtl']

    sc = compute_schmidt(**constants)
    if sc.get('schmidt') is not None:
        results['schmidt'] = sc['schmidt']

    nu = compute_nusselt(**constants)
    if nu.get('nusselt') is not None:
        results['nusselt'] = nu['nusselt']

    sh = compute_sherwood(**constants)
    if sh.get('sherwood') is not None:
        results['sherwood'] = sh['sherwood']

    pe = compute_peclet(**constants)
    if pe.get('peclet') is not None:
        results['peclet'] = pe['peclet']

    da = compute_damkohler(**constants)
    if da.get('damkohler') is not None:
        results['damkohler'] = da['damkohler']

    we = compute_weber(**constants)
    if we.get('weber') is not None:
        results['weber'] = we['weber']

    fr = compute_froude(**constants)
    if fr.get('froude') is not None:
        results['froude'] = fr['froude']

    gr = compute_grashof(**constants)
    if gr.get('grashof') is not None:
        results['grashof'] = gr['grashof']

    ra = compute_rayleigh(**constants, grashof=results.get('grashof'), prandtl=results.get('prandtl'))
    if ra.get('rayleigh') is not None:
        results['rayleigh'] = ra['rayleigh']

    bi = compute_biot(**constants)
    if bi.get('biot') is not None:
        results['biot'] = bi['biot']

    le = compute_lewis(**constants, schmidt=results.get('schmidt'), prandtl=results.get('prandtl'))
    if le.get('lewis') is not None:
        results['lewis'] = le['lewis']

    return results
