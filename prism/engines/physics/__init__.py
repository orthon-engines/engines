"""
PRISM Physics & Chemical Engineering Engines

All engines compute REAL physics equations with proper units.

Philosophy:
- When physical constants are known -> compute absolute values in Joules, etc.
- When physical constants are unknown -> compute specific (per-unit) values
- Always honest about what was computed (is_specific flag, units, equation)

=== CLASSICAL MECHANICS ===
1. kinetic_energy: T = ½mv²  [J or J/kg]
2. potential_energy: V = ½kx², V = mgh  [J or specific]
3. hamiltonian: H = T + V  [J]
4. lagrangian: L = T - V  [J]
5. momentum: p = mv, L = r × p  [kg·m/s, kg·m²/s]
6. work_energy: W = ∫F·dx, P = F·v  [J, W]

=== FLUID MECHANICS ===
7. reynolds: Re = ρvL/μ
8. pressure_drop: Darcy-Weisbach equation
9. fluid_mechanics: Bernoulli, Hagen-Poiseuille, continuity

=== HEAT TRANSFER ===
10. fourier: q = -k(dT/dx), conduction through walls
11. heat_transfer: Nusselt correlations (Dittus-Boelter, Churchill-Chu, etc.)

=== MASS TRANSFER ===
12. fick: J = -D(dC/dx), diffusion
13. mass_transfer: Sherwood correlations (Ranz-Marshall, etc.)

=== THERMODYNAMICS ===
14. thermodynamics: Equations of state (ideal, van der Waals, Peng-Robinson)
15. gibbs_free_energy: G = H - TS  [J or J/mol]

=== DIMENSIONLESS NUMBERS ===
16. dimensionless: Re, Pr, Sc, Nu, Sh, Pe, Da, We, Fr, Gr, Ra, Bi, Le, St

=== REACTION ENGINEERING ===
17. reaction_kinetics: Arrhenius, Michaelis-Menten, reactor design

=== PROCESS CONTROL ===
18. process_control: Transfer functions, PID tuning, stability analysis

No fake physics. No assumed constants. No cover bands.
"""

# === Classical Mechanics ===
from prism.engines.physics.kinetic_energy import (
    compute_kinetic_energy,
    compute_kinetic_energy_rotational,
    compute as compute_kinetic,
)

from prism.engines.physics.potential_energy import (
    compute_potential_energy_harmonic,
    compute_potential_energy_gravitational,
    estimate_spring_constant,
    compute as compute_potential,
)

from prism.engines.physics.hamiltonian import (
    compute_hamiltonian,
    compute_hamiltons_equations,
    compute as compute_hamilton,
)

from prism.engines.physics.lagrangian import (
    compute_lagrangian,
    compute_action,
    check_euler_lagrange,
    compute as compute_lagrange,
)

from prism.engines.physics.momentum import (
    compute_linear_momentum,
    compute_angular_momentum,
    check_momentum_conservation,
    compute_impulse,
    compute as compute_momentum,
)

from prism.engines.physics.work_energy import (
    compute_work,
    compute_power,
    verify_work_energy_theorem,
    compute_conservative_force_test,
    compute_mechanical_energy,
    compute as compute_work_energy,
)

# === Fluid Mechanics ===
from prism.engines.physics.reynolds import (
    compute as compute_reynolds,
)

from prism.engines.physics.pressure_drop import (
    compute as compute_pressure_drop,
)

from prism.engines.physics.fluid_mechanics import (
    continuity_equation,
    bernoulli_equation,
    hagen_poiseuille,
    friction_factor,
    darcy_weisbach,
    minor_losses,
    pump_power,
    orifice_meter,
    venturi_meter,
)

# === Heat Transfer ===
from prism.engines.physics.fourier import (
    compute_heat_flux,
    compute_conduction_slab,
    compute_conduction_cylinder,
    compute_conduction_sphere,
    compute_composite_wall,
)

from prism.engines.physics.heat_transfer import (
    dittus_boelter,
    sieder_tate,
    gnielinski,
    laminar_pipe,
    flat_plate_laminar as heat_flat_plate_laminar,
    flat_plate_turbulent as heat_flat_plate_turbulent,
    churchill_chu_vertical_plate,
    sphere_crossflow,
    compute_h_from_nusselt,
    overall_heat_transfer_coefficient,
)

# === Mass Transfer ===
from prism.engines.physics.fick import (
    compute_molar_flux,
    compute_mass_transfer_slab,
    equimolar_counterdiffusion,
    diffusion_through_stagnant_film,
    wilke_chang,
    chapman_enskog,
    stokes_einstein,
    penetration_depth,
)

from prism.engines.physics.mass_transfer import (
    chilton_colburn_analogy,
    pipe_turbulent as mass_pipe_turbulent,
    gilliland_sherwood,
    flat_plate_laminar as mass_flat_plate_laminar,
    flat_plate_turbulent as mass_flat_plate_turbulent,
    froessling,
    ranz_marshall,
    packed_bed,
    falling_film,
    compute_kc_from_sherwood,
    overall_mass_transfer_coefficient,
)

# === Thermodynamics ===
from prism.engines.physics.thermodynamics import (
    ideal_gas,
    van_der_waals,
    peng_robinson,
    enthalpy_ideal_gas,
    entropy_ideal_gas,
    gibbs_free_energy,
    clausius_clapeyron,
    antoine_equation,
    raoults_law,
    henrys_law,
    fugacity,
    activity,
)

from prism.engines.physics.gibbs_free_energy import (
    compute_gibbs_free_energy,
    compute_gibbs_ideal_gas,
    compute_gibbs_change,
    compute_chemical_potential,
    compute as compute_gibbs,
)

# === Dimensionless Numbers ===
from prism.engines.physics.dimensionless import (
    compute_prandtl,
    compute_schmidt,
    compute_nusselt,
    compute_sherwood,
    compute_peclet,
    compute_damkohler,
    compute_weber,
    compute_froude,
    compute_grashof,
    compute_rayleigh,
    compute_biot,
    compute_lewis,
    compute_stanton,
    compute_all as compute_all_dimensionless,
)

# === Reaction Kinetics ===
from prism.engines.physics.reaction_kinetics import (
    arrhenius,
    arrhenius_two_temperatures,
    power_law_rate,
    michaelis_menten,
    langmuir_hinshelwood,
    conversion,
    yield_and_selectivity,
    batch_reactor_time,
    cstr_volume,
    pfr_volume,
    residence_time_distribution,
    equilibrium_constant,
)

# === Process Control ===
from prism.engines.physics.process_control import (
    first_order_response,
    second_order_response,
    time_delay_pade,
    pid_controller,
    ziegler_nichols_closed_loop,
    ziegler_nichols_open_loop,
    imc_tuning,
    stability_margins,
    poles_and_zeros,
    closed_loop_transfer_function,
)


__all__ = [
    # === Classical Mechanics ===
    # Kinetic Energy
    'compute_kinetic_energy',
    'compute_kinetic_energy_rotational',
    'compute_kinetic',
    # Potential Energy
    'compute_potential_energy_harmonic',
    'compute_potential_energy_gravitational',
    'estimate_spring_constant',
    'compute_potential',
    # Hamiltonian
    'compute_hamiltonian',
    'compute_hamiltons_equations',
    'compute_hamilton',
    # Lagrangian
    'compute_lagrangian',
    'compute_action',
    'check_euler_lagrange',
    'compute_lagrange',
    # Momentum
    'compute_linear_momentum',
    'compute_angular_momentum',
    'check_momentum_conservation',
    'compute_impulse',
    'compute_momentum',
    # Work-Energy
    'compute_work',
    'compute_power',
    'verify_work_energy_theorem',
    'compute_conservative_force_test',
    'compute_mechanical_energy',
    'compute_work_energy',

    # === Fluid Mechanics ===
    'compute_reynolds',
    'compute_pressure_drop',
    'continuity_equation',
    'bernoulli_equation',
    'hagen_poiseuille',
    'friction_factor',
    'darcy_weisbach',
    'minor_losses',
    'pump_power',
    'orifice_meter',
    'venturi_meter',

    # === Heat Transfer ===
    'compute_heat_flux',
    'compute_conduction_slab',
    'compute_conduction_cylinder',
    'compute_conduction_sphere',
    'compute_composite_wall',
    'dittus_boelter',
    'sieder_tate',
    'gnielinski',
    'laminar_pipe',
    'heat_flat_plate_laminar',
    'heat_flat_plate_turbulent',
    'churchill_chu_vertical_plate',
    'sphere_crossflow',
    'compute_h_from_nusselt',
    'overall_heat_transfer_coefficient',

    # === Mass Transfer ===
    'compute_molar_flux',
    'compute_mass_transfer_slab',
    'equimolar_counterdiffusion',
    'diffusion_through_stagnant_film',
    'wilke_chang',
    'chapman_enskog',
    'stokes_einstein',
    'penetration_depth',
    'chilton_colburn_analogy',
    'mass_pipe_turbulent',
    'gilliland_sherwood',
    'mass_flat_plate_laminar',
    'mass_flat_plate_turbulent',
    'froessling',
    'ranz_marshall',
    'packed_bed',
    'falling_film',
    'compute_kc_from_sherwood',
    'overall_mass_transfer_coefficient',

    # === Thermodynamics ===
    'ideal_gas',
    'van_der_waals',
    'peng_robinson',
    'enthalpy_ideal_gas',
    'entropy_ideal_gas',
    'gibbs_free_energy',
    'clausius_clapeyron',
    'antoine_equation',
    'raoults_law',
    'henrys_law',
    'fugacity',
    'activity',
    'compute_gibbs_free_energy',
    'compute_gibbs_ideal_gas',
    'compute_gibbs_change',
    'compute_chemical_potential',
    'compute_gibbs',

    # === Dimensionless Numbers ===
    'compute_prandtl',
    'compute_schmidt',
    'compute_nusselt',
    'compute_sherwood',
    'compute_peclet',
    'compute_damkohler',
    'compute_weber',
    'compute_froude',
    'compute_grashof',
    'compute_rayleigh',
    'compute_biot',
    'compute_lewis',
    'compute_stanton',
    'compute_all_dimensionless',

    # === Reaction Kinetics ===
    'arrhenius',
    'arrhenius_two_temperatures',
    'power_law_rate',
    'michaelis_menten',
    'langmuir_hinshelwood',
    'conversion',
    'yield_and_selectivity',
    'batch_reactor_time',
    'cstr_volume',
    'pfr_volume',
    'residence_time_distribution',
    'equilibrium_constant',

    # === Process Control ===
    'first_order_response',
    'second_order_response',
    'time_delay_pade',
    'pid_controller',
    'ziegler_nichols_closed_loop',
    'ziegler_nichols_open_loop',
    'imc_tuning',
    'stability_margins',
    'poles_and_zeros',
    'closed_loop_transfer_function',
]
