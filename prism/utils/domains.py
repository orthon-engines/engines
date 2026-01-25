"""
PRISM Domain Definitions

Metadata for ORTHON UI to build domain-specific wizards.

Usage:
    from prism.domains import DOMAINS, get_required_inputs

    # Get all domains
    for domain in DOMAINS.values():
        print(domain['name'], domain['equations'])

    # Get required inputs for selected equations
    inputs = get_required_inputs(['PIPE_REYNOLDS', 'PRESSURE_DROP'])
    # Returns: {'density': {...}, 'kinematic_viscosity': {...}, 'pipe_network': {...}}
"""

from typing import Dict, List, Set, Any
from prism.utils.capability import Capability, CAPABILITY_REQUIREMENTS


# =============================================================================
# DOMAIN DEFINITIONS
# =============================================================================

DOMAINS: Dict[str, Dict[str, Any]] = {
    'chemical': {
        'name': 'Chemical Engineering',
        'icon': '🧪',
        'equations': [
            'PIPE_REYNOLDS',
            'FLOW_REGIME',
            'FRICTION_FACTOR',
            'PRESSURE_DROP',
            'HEAD_LOSS',
            'PIPE_POWER_LOSS',
            'GIBBS_FREE_ENERGY',
            'ENTHALPY',
            'ENTROPY_THERMO',
            'CHEMICAL_POTENTIAL',
        ],
        'signal_hints': ['temperature', 'pressure', 'flow', 'velocity', 'concentration', 'pH'],
    },

    'electrical': {
        'name': 'Electrical Engineering',
        'icon': '⚡',
        'equations': [
            'TRANSFER_FUNCTION',
            'FREQUENCY_RESPONSE',
            'POLES_ZEROS',
            'IMPULSE_RESPONSE',
            'GRANGER_CAUSALITY',
        ],
        'signal_hints': ['voltage', 'current', 'power', 'frequency', 'impedance'],
    },

    'mechanical': {
        'name': 'Mechanical Engineering',
        'icon': '⚙️',
        'equations': [
            'KINETIC_ENERGY',
            'POTENTIAL_ENERGY',
            'HAMILTONIAN',
            'LAGRANGIAN',
            'MOMENTUM',
            'ANGULAR_MOMENTUM',
            'WORK',
            'POWER',
            'ROTATIONAL_KE',
            'PHASE_SPACE',
            'ENERGY_CONSERVATION',
        ],
        'signal_hints': ['position', 'velocity', 'acceleration', 'force', 'torque', 'angle'],
    },

    'fluids': {
        'name': 'Fluid Dynamics',
        'icon': '🌊',
        'equations': [
            'VORTICITY',
            'STRAIN_TENSOR',
            'Q_CRITERION',
            'TURBULENT_KE',
            'DISSIPATION',
            'ENERGY_SPECTRUM',
            'REYNOLDS_NUMBER',
            'KOLMOGOROV_SCALES',
        ],
        'signal_hints': ['velocity_x', 'velocity_y', 'velocity_z', 'pressure', 'density'],
    },

    'thermal': {
        'name': 'Thermal/Heat Transfer',
        'icon': '🔥',
        'equations': [
            'HEAT_FLUX',
            'LAPLACIAN_T',
            'GIBBS_FREE_ENERGY',
            'ENTHALPY',
        ],
        'signal_hints': ['temperature', 'heat_flux', 'thermal_conductivity'],
    },

    'signals': {
        'name': 'Signal Processing',
        'icon': '📊',
        'equations': [
            'STATISTICS',
            'ENTROPY',
            'MEMORY',
            'SPECTRAL',
            'STATIONARITY',
            'VOLATILITY',
            'EVENT_DETECTION',
            'HEAVISIDE_DIRAC',
        ],
        'signal_hints': ['sensor', 'measurement', 'signal', 'data'],
    },

    'dynamical': {
        'name': 'Dynamical Systems',
        'icon': '🔄',
        'equations': [
            'CHAOS',
            'RECURRENCE',
            'PHASE_SPACE',
            'DYNAMICS',
            'GEOMETRY',
        ],
        'signal_hints': ['state', 'trajectory', 'attractor'],
    },
}


# =============================================================================
# INPUT DEFINITIONS
# =============================================================================

INPUT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # Constants
    'mass': {
        'label': 'Mass',
        'units': 'kg',
        'type': 'float',
        'min': 0,
        'placeholder': '2.0',
    },
    'spring_constant': {
        'label': 'Spring Constant',
        'units': 'N/m',
        'type': 'float',
        'min': 0,
        'placeholder': '50.0',
    },
    'damping_coefficient': {
        'label': 'Damping Coefficient',
        'units': 'N·s/m',
        'type': 'float',
        'min': 0,
        'placeholder': '1.0',
    },
    'moment_of_inertia': {
        'label': 'Moment of Inertia',
        'units': 'kg·m²',
        'type': 'float',
        'min': 0,
        'placeholder': '0.5',
    },
    'density': {
        'label': 'Density',
        'units': 'kg/m³',
        'type': 'float',
        'min': 0,
        'placeholder': '1000.0',
    },
    'kinematic_viscosity': {
        'label': 'Kinematic Viscosity',
        'units': 'm²/s',
        'type': 'float',
        'min': 0,
        'placeholder': '1e-6',
        'scientific': True,
    },
    'Cp': {
        'label': 'Heat Capacity (Cp)',
        'units': 'J/(mol·K)',
        'type': 'float',
        'min': 0,
        'placeholder': '29.1',
    },
    'Cv': {
        'label': 'Heat Capacity (Cv)',
        'units': 'J/(mol·K)',
        'type': 'float',
        'min': 0,
        'placeholder': '20.8',
    },
    'n_moles': {
        'label': 'Amount of Substance',
        'units': 'mol',
        'type': 'float',
        'min': 0,
        'placeholder': '1.0',
    },

    # Pipe network (special type)
    'pipe_network': {
        'label': 'Pipe Geometry',
        'type': 'pipe_table',
        'columns': ['signal', 'diameter', 'length', 'roughness'],
        'required_columns': ['signal', 'diameter'],
    },

    # Spatial grid
    'dx': {'label': 'Grid Spacing (x)', 'units': 'm', 'type': 'float', 'placeholder': '0.01'},
    'dy': {'label': 'Grid Spacing (y)', 'units': 'm', 'type': 'float', 'placeholder': '0.01'},
    'dz': {'label': 'Grid Spacing (z)', 'units': 'm', 'type': 'float', 'placeholder': '0.01'},
    'dt': {'label': 'Time Step', 'units': 's', 'type': 'float', 'placeholder': '0.001'},

    # Relationships
    'io_pair': {
        'label': 'Input/Output Signal Pair',
        'type': 'signal_pair',
        'fields': ['input_signal', 'output_signal'],
    },
    'velocity_field': {
        'label': 'Velocity Field Components',
        'type': 'signal_triplet',
        'fields': ['u (x-velocity)', 'v (y-velocity)', 'w (z-velocity)'],
    },
}


# =============================================================================
# EQUATION METADATA
# =============================================================================

EQUATION_INFO: Dict[str, Dict[str, str]] = {
    # Level 0
    'STATISTICS': {'formula': 'μ, σ, skew, kurt', 'description': 'Basic statistics'},
    'ENTROPY': {'formula': 'H = -Σ p log p', 'description': 'Information entropy'},
    'MEMORY': {'formula': 'H (Hurst)', 'description': 'Long-range dependence'},
    'SPECTRAL': {'formula': 'F(ω)', 'description': 'Frequency analysis'},
    'STATIONARITY': {'formula': 'ADF, KPSS', 'description': 'Stationarity tests'},
    'VOLATILITY': {'formula': 'σ(t)', 'description': 'Time-varying variance'},
    'EVENT_DETECTION': {'formula': 'steps, spikes', 'description': 'Discontinuity detection'},
    'HEAVISIDE_DIRAC': {'formula': 'H(t-t₀), δ(t-t₀)', 'description': 'Step/impulse at events'},
    'RECURRENCE': {'formula': 'RQA', 'description': 'Recurrence quantification'},
    'CHAOS': {'formula': 'λ (Lyapunov)', 'description': 'Chaos detection'},
    'GEOMETRY': {'formula': 'PCA, manifold', 'description': 'Geometric structure'},
    'DYNAMICS': {'formula': 'trajectory', 'description': 'Dynamical behavior'},

    # Level 2 - Mechanical
    'KINETIC_ENERGY': {'formula': 'T = ½mv²', 'description': 'Energy of motion'},
    'POTENTIAL_ENERGY': {'formula': 'V = ½kx²', 'description': 'Stored energy'},
    'HAMILTONIAN': {'formula': 'H = T + V', 'description': 'Total energy'},
    'LAGRANGIAN': {'formula': 'L = T - V', 'description': 'Action principle'},
    'MOMENTUM': {'formula': 'p = mv', 'description': 'Linear momentum'},
    'ANGULAR_MOMENTUM': {'formula': 'L = r × p', 'description': 'Rotational momentum'},
    'ROTATIONAL_KE': {'formula': 'T = ½Iω²', 'description': 'Rotational kinetic energy'},
    'WORK': {'formula': 'W = ∫F·dx', 'description': 'Work done'},
    'POWER': {'formula': 'P = F·v', 'description': 'Rate of work'},
    'PHASE_SPACE': {'formula': '(q, p)', 'description': 'Phase space trajectory'},
    'ENERGY_CONSERVATION': {'formula': 'dH/dt ≈ 0', 'description': 'Energy conservation check'},

    # Level 3 - Thermodynamic
    'GIBBS_FREE_ENERGY': {'formula': 'G = H - TS', 'description': 'Available work'},
    'ENTHALPY': {'formula': 'H = U + PV', 'description': 'Heat content'},
    'ENTROPY_THERMO': {'formula': 'S = ∫dQ/T', 'description': 'Thermodynamic entropy'},
    'CHEMICAL_POTENTIAL': {'formula': 'μ = ∂G/∂n', 'description': 'Chemical potential'},

    # Level 3 - Control
    'TRANSFER_FUNCTION': {'formula': 'G(s) = Y(s)/U(s)', 'description': 'System transfer function'},
    'FREQUENCY_RESPONSE': {'formula': 'G(jω)', 'description': 'Bode plot'},
    'POLES_ZEROS': {'formula': 'poles, zeros', 'description': 'Stability analysis'},
    'IMPULSE_RESPONSE': {'formula': 'h(t)', 'description': 'Impulse response'},
    'GRANGER_CAUSALITY': {'formula': 'X → Y', 'description': 'Causal relationships'},
    'TRANSFER_ENTROPY': {'formula': 'T_X→Y', 'description': 'Information transfer'},

    # Level 4 - Spatial
    'VORTICITY': {'formula': 'ω = ∇ × v', 'description': 'Fluid rotation'},
    'STRAIN_TENSOR': {'formula': 'S_ij', 'description': 'Deformation rate'},
    'Q_CRITERION': {'formula': 'Q > 0', 'description': 'Vortex identification'},
    'TURBULENT_KE': {'formula': 'k = ½⟨u\'²⟩', 'description': 'Turbulent kinetic energy'},
    'DISSIPATION': {'formula': 'ε = 2ν⟨S_ij S_ij⟩', 'description': 'Energy dissipation'},
    'ENERGY_SPECTRUM': {'formula': 'E(k) ∝ k⁻⁵/³', 'description': 'Energy spectrum'},
    'REYNOLDS_NUMBER': {'formula': 'Re = UL/ν', 'description': 'Flow regime'},
    'KOLMOGOROV_SCALES': {'formula': 'η, τ_η, v_η', 'description': 'Smallest turbulent scales'},
    'HEAT_FLUX': {'formula': 'q = -k∇T', 'description': 'Heat conduction'},
    'LAPLACIAN_T': {'formula': '∇²T', 'description': 'Temperature diffusion'},

    # Pipe flow
    'PIPE_REYNOLDS': {'formula': 'Re = vD/ν', 'description': 'Pipe flow Reynolds number'},
    'FLOW_REGIME': {'formula': 'Laminar/Turb', 'description': 'Flow classification'},
    'FRICTION_FACTOR': {'formula': 'f (Darcy)', 'description': 'Friction factor'},
    'PRESSURE_DROP': {'formula': 'ΔP = f(L/D)(ρv²/2)', 'description': 'Pressure loss'},
    'HEAD_LOSS': {'formula': 'h_f = ΔP/(ρg)', 'description': 'Head loss'},
    'PIPE_POWER_LOSS': {'formula': 'P = ΔP·Q', 'description': 'Pumping power'},
}


# =============================================================================
# API FUNCTIONS (for ORTHON to call)
# =============================================================================

def get_required_inputs(equations: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get all required inputs for a set of equations.

    Args:
        equations: List of equation names (e.g., ['PIPE_REYNOLDS', 'PRESSURE_DROP'])

    Returns:
        Dict of input_name -> input_definition
    """
    required = set()

    for eq_name in equations:
        try:
            cap = Capability[eq_name]
            reqs = CAPABILITY_REQUIREMENTS.get(cap, {})

            # Collect constants
            for const in reqs.get('constants', []):
                required.add(const)

            # Check for pipe network
            if reqs.get('pipe_network'):
                required.add('pipe_network')

            # Check for I/O pair
            if reqs.get('io_pair'):
                required.add('io_pair')

            # Check for spatial
            spatial = reqs.get('spatial')
            if spatial == 'velocity_field':
                required.add('velocity_field')
                required.add('dx')
                required.add('dy')
                required.add('dz')

        except KeyError:
            pass  # Unknown equation

    return {name: INPUT_DEFINITIONS.get(name, {'label': name}) for name in required}


def get_equations_for_domain(domain: str) -> List[Dict[str, Any]]:
    """
    Get equation info for a domain.

    Returns list of {name, formula, description, requires}
    """
    if domain not in DOMAINS:
        return []

    result = []
    for eq_name in DOMAINS[domain]['equations']:
        info = EQUATION_INFO.get(eq_name, {})
        try:
            cap = Capability[eq_name]
            reqs = CAPABILITY_REQUIREMENTS.get(cap, {})
        except KeyError:
            reqs = {}

        result.append({
            'name': eq_name,
            'formula': info.get('formula', ''),
            'description': info.get('description', ''),
            'requires': reqs,
        })

    return result


def validate_inputs(equations: List[str], inputs: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate that all required inputs are provided.

    Returns dict of missing input_name -> error message
    """
    required = get_required_inputs(equations)
    errors = {}

    for name, defn in required.items():
        if name not in inputs or inputs[name] is None:
            errors[name] = f"Missing required input: {defn.get('label', name)}"
        elif defn.get('type') == 'float':
            try:
                val = float(inputs[name])
                if defn.get('min') is not None and val < defn['min']:
                    errors[name] = f"{defn['label']} must be >= {defn['min']}"
            except (TypeError, ValueError):
                errors[name] = f"{defn['label']} must be a number"

    return errors


def generate_config(
    domain: str,
    equations: List[str],
    signals: List[str],
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a PRISM config dict from wizard inputs.

    This is what ORTHON calls after the user completes the wizard.
    """
    config = {
        'window_size': inputs.get('window_size', 100),
        'window_stride': inputs.get('window_stride', 50),
        'signals': [{'name': s} for s in signals],
        'constants': {},
    }

    # Add constants
    for const in ['mass', 'spring_constant', 'damping_coefficient', 'moment_of_inertia',
                  'density', 'kinematic_viscosity', 'Cp', 'Cv', 'n_moles', 'dx', 'dy', 'dz', 'dt']:
        if const in inputs and inputs[const] is not None:
            config['constants'][const] = float(inputs[const])

    # Add pipe network
    if 'pipe_network' in inputs and inputs['pipe_network']:
        config['pipe_network'] = {'pipes': inputs['pipe_network']}

    # Add I/O pair
    if 'io_pair' in inputs and inputs['io_pair']:
        config['relationships'] = {
            'control': {
                'input': inputs['io_pair'].get('input'),
                'output': inputs['io_pair'].get('output'),
            }
        }

    # Add velocity field
    if 'velocity_field' in inputs and inputs['velocity_field']:
        config['spatial'] = {
            'type': 'velocity_field',
            'u': inputs['velocity_field'].get('u'),
            'v': inputs['velocity_field'].get('v'),
            'w': inputs['velocity_field'].get('w'),
        }

    return config


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print("PRISM Domains:")
    for key, domain in DOMAINS.items():
        print(f"  {domain['icon']} {domain['name']}: {len(domain['equations'])} equations")

    print("\nExample - Chemical Engineering inputs for PIPE_REYNOLDS + PRESSURE_DROP:")
    inputs = get_required_inputs(['PIPE_REYNOLDS', 'PRESSURE_DROP'])
    for name, defn in inputs.items():
        print(f"  {name}: {defn.get('label', name)} [{defn.get('units', '')}]")
