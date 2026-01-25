"""
PRISM Discipline Registry

Defines available disciplines with their requirements and engines.

Each discipline specifies:
- Required/optional signals
- Required/optional constants (with units)
- Available engines
"""

DISCIPLINES = {
    "thermodynamics": {
        "name": "Thermodynamics",
        "description": "Energy, entropy, equations of state",
        "icon": "fire",
        "signals": {
            "required_any": ["temperature", "pressure"],  # Need at least one
            "optional": ["volume", "internal_energy", "enthalpy"],
        },
        "constants": {
            "optional": {
                "gas_constant": {"unit": "J/(mol·K)", "default": 8.314},
                "molar_mass": {"unit": "kg/mol"},
                "heat_capacity_cp": {"unit": "J/(kg·K)"},
                "heat_capacity_cv": {"unit": "J/(kg·K)"},
                "critical_temperature": {"unit": "K"},
                "critical_pressure": {"unit": "Pa"},
                "acentric_factor": {"unit": "dimensionless"},
            }
        },
        "engines": [
            "gibbs_free_energy",
            "enthalpy_ideal_gas",
            "entropy_ideal_gas",
            "ideal_gas",
            "van_der_waals",
            "peng_robinson",
            "clausius_clapeyron",
            "antoine_equation",
            "raoults_law",
            "henrys_law",
            "fugacity",
            "activity",
        ],
    },

    "transport": {
        "name": "Transport Phenomena",
        "description": "Heat, mass, and momentum transfer",
        "icon": "waves",
        "subdisciplines": {
            "momentum": {
                "name": "Momentum Transfer",
                "signals": {
                    "required_any": ["velocity", "flow_rate"],
                },
                "constants": {
                    "required": {
                        "density": {"unit": "kg/m³"},
                        "viscosity": {"unit": "Pa·s"},
                    },
                    "optional": {
                        "diameter": {"unit": "m"},
                        "length": {"unit": "m"},
                        "roughness": {"unit": "m"},
                    }
                },
                "engines": [
                    "reynolds",
                    "pressure_drop",
                    "friction_factor",
                    "bernoulli_equation",
                    "hagen_poiseuille",
                    "pump_power",
                ],
            },
            "heat": {
                "name": "Heat Transfer",
                "signals": {
                    "required": ["temperature"],
                },
                "constants": {
                    "required": {
                        "thermal_conductivity": {"unit": "W/(m·K)"},
                    },
                    "optional": {
                        "heat_transfer_coeff": {"unit": "W/(m²·K)"},
                        "thermal_diffusivity": {"unit": "m²/s"},
                    }
                },
                "engines": [
                    "compute_heat_flux",
                    "compute_conduction_slab",
                    "dittus_boelter",
                    "gnielinski",
                    "compute_prandtl",
                    "compute_nusselt",
                    "overall_heat_transfer_coefficient",
                ],
            },
            "mass": {
                "name": "Mass Transfer",
                "signals": {
                    "required": ["concentration"],
                },
                "constants": {
                    "required": {
                        "diffusivity": {"unit": "m²/s"},
                    },
                    "optional": {
                        "mass_transfer_coeff": {"unit": "m/s"},
                    }
                },
                "engines": [
                    "compute_molar_flux",
                    "compute_mass_transfer_slab",
                    "compute_schmidt",
                    "compute_sherwood",
                    "ranz_marshall",
                    "wilke_chang",
                ],
            },
        },
        "engines": [
            # Dimensionless numbers (need various constants)
            "compute_peclet",
            "compute_weber",
            "compute_froude",
            "compute_grashof",
            "compute_rayleigh",
            "compute_biot",
            "compute_lewis",
            "compute_stanton",
            "chilton_colburn_analogy",
        ],
    },

    "reaction": {
        "name": "Reaction Engineering",
        "description": "Kinetics, reactor design, yields",
        "icon": "flask",
        "signals": {
            "required_any": ["concentration", "conversion", "temperature"],
        },
        "constants": {
            "optional": {
                "activation_energy": {"unit": "J/mol"},
                "pre_exponential": {"unit": "1/s"},
                "reaction_order": {"unit": "dimensionless"},
                "reactor_volume": {"unit": "m³"},
                "flow_rate": {"unit": "m³/s"},
                "K_m": {"unit": "mol/m³", "description": "Michaelis constant"},
                "V_max": {"unit": "mol/(m³·s)", "description": "Maximum rate"},
            }
        },
        "engines": [
            "arrhenius",
            "arrhenius_two_temperatures",
            "power_law_rate",
            "michaelis_menten",
            "langmuir_hinshelwood",
            "conversion",
            "yield_and_selectivity",
            "batch_reactor_time",
            "cstr_volume",
            "pfr_volume",
            "residence_time_distribution",
            "equilibrium_constant",
            "compute_damkohler",
        ],
    },

    "controls": {
        "name": "Process Control",
        "description": "Dynamics, stability, feedback",
        "icon": "sliders",
        "signals": {
            "required": ["process_variable"],  # The thing being controlled
            "optional": ["setpoint", "manipulated_variable", "controller_output"],
        },
        "constants": {
            "optional": {
                "time_constant": {"unit": "s"},
                "dead_time": {"unit": "s"},
                "gain": {"unit": "dimensionless"},
                "kp": {"unit": "dimensionless", "description": "Proportional gain"},
                "ki": {"unit": "1/s", "description": "Integral gain"},
                "kd": {"unit": "s", "description": "Derivative gain"},
            }
        },
        "engines": [
            "first_order_response",
            "second_order_response",
            "time_delay_pade",
            "pid_controller",
            "ziegler_nichols_closed_loop",
            "ziegler_nichols_open_loop",
            "imc_tuning",
            "stability_margins",
            "poles_and_zeros",
            "closed_loop_transfer_function",
        ],
    },

    "mechanics": {
        "name": "Mechanical Systems",
        "description": "Vibration, fatigue, stress, energy",
        "icon": "gear",
        "signals": {
            "required_any": ["vibration", "acceleration", "displacement", "strain", "stress", "velocity"],
        },
        "constants": {
            "optional": {
                "mass": {"unit": "kg"},
                "stiffness": {"unit": "N/m"},
                "damping": {"unit": "N·s/m"},
                "youngs_modulus": {"unit": "Pa"},
                "poissons_ratio": {"unit": "dimensionless"},
                "moment_of_inertia": {"unit": "kg·m²"},
            }
        },
        "engines": [
            "compute_kinetic_energy",
            "compute_potential_energy_harmonic",
            "compute_potential_energy_gravitational",
            "compute_hamiltonian",
            "compute_lagrangian",
            "compute_linear_momentum",
            "compute_angular_momentum",
            "compute_work",
            "compute_power",
        ],
    },

    "electrical": {
        "name": "Electrical Systems",
        "description": "Impedance, power, batteries",
        "icon": "zap",
        "signals": {
            "required_any": ["voltage", "current", "power", "impedance"],
        },
        "constants": {
            "optional": {
                "resistance": {"unit": "Ω"},
                "capacitance": {"unit": "F"},
                "inductance": {"unit": "H"},
                "nominal_capacity": {"unit": "Ah"},
                "nominal_voltage": {"unit": "V"},
            }
        },
        "engines": [
            # Electrical-specific engines (to be implemented)
            "impedance_spectrum",
            "power_factor",
            "soh",  # State of Health (batteries)
            "capacity_fade",
            "internal_resistance",
        ],
    },

    "fluid_dynamics": {
        "name": "Fluid Dynamics (CFD)",
        "description": "Velocity fields, vorticity, turbulence",
        "icon": "wind",
        "signals": {
            "required": ["u", "v"],  # Velocity components
            "optional": ["w", "pressure"],
        },
        "constants": {
            "required": {
                "density": {"unit": "kg/m³"},
                "viscosity": {"unit": "Pa·s"},
            },
            "optional": {
                "reference_velocity": {"unit": "m/s"},
                "reference_length": {"unit": "m"},
            }
        },
        "spatial": True,  # Indicates Level 4 / Fields analysis
        "engines": [
            "vorticity",
            "divergence",
            "q_criterion",
            "strain_rate",
            "tke",
            "navier_stokes_residual",
        ],
    },

    "dimensionless": {
        "name": "Dimensionless Analysis",
        "description": "All dimensionless numbers",
        "icon": "hash",
        "signals": {
            "optional": ["velocity", "temperature", "concentration"],
        },
        "constants": {
            "optional": {
                "density": {"unit": "kg/m³"},
                "viscosity": {"unit": "Pa·s"},
                "thermal_conductivity": {"unit": "W/(m·K)"},
                "heat_capacity": {"unit": "J/(kg·K)"},
                "diffusivity": {"unit": "m²/s"},
                "characteristic_length": {"unit": "m"},
                "characteristic_velocity": {"unit": "m/s"},
            }
        },
        "engines": [
            "compute_prandtl",
            "compute_schmidt",
            "compute_nusselt",
            "compute_sherwood",
            "compute_peclet",
            "compute_damkohler",
            "compute_weber",
            "compute_froude",
            "compute_grashof",
            "compute_rayleigh",
            "compute_biot",
            "compute_lewis",
            "compute_stanton",
            "compute_all_dimensionless",
        ],
    },
}


def get_discipline(name: str) -> dict:
    """Get discipline by name."""
    return DISCIPLINES.get(name)


def list_disciplines() -> list:
    """List all available discipline names."""
    return list(DISCIPLINES.keys())


def get_all_engines() -> list:
    """Get list of all engines across all disciplines."""
    engines = set()
    for disc in DISCIPLINES.values():
        engines.update(disc.get('engines', []))
        for sub in disc.get('subdisciplines', {}).values():
            engines.update(sub.get('engines', []))
    return sorted(engines)
