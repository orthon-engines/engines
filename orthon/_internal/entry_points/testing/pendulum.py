"""
Double Pendulum Phase Shift Generator
=====================================

Creates a double pendulum dataset with a KNOWN regime change:
- Phase 1 (t=0-25): Stable, low-energy oscillation (small angles)
- Phase 2 (t=25-50): Chaotic, high-energy dynamics (kicked to π)

This is the ideal PRISM validation case because:
1. Ground truth regime boundary is known (t=25, row 2500)
2. Stable phase should show periodic behavior
3. Chaotic phase should show deterministic chaos
4. PRISM should detect the transition

Usage:
    python -m prism.entry_points.generate_pendulum_regime

Then run:
    python -m prism.entry_points.signal_vector --signal --domain pendulum_regime --force --report signal
"""

import numpy as np
import polars as pl
from datetime import date, timedelta
from scipy.integrate import solve_ivp

from orthon._internal.db.parquet_store import get_data_root
from orthon._internal.db.polars_io import write_parquet_atomic


def double_pendulum_dynamics(t, state, m1=1, m2=1, L1=1, L2=1, g=9.81):
    """
    Proper Lagrangian formulation of double pendulum.
    Uses mass matrix inversion for numerical stability.
    """
    th1, w1, th2, w2 = state

    # Mass matrix
    M = np.array([
        [(m1 + m2) * L1, m2 * L2 * np.cos(th1 - th2)],
        [m2 * L1 * np.cos(th1 - th2), m2 * L2]
    ])

    # Force vector
    f = np.array([
        -m2 * L2 * w2**2 * np.sin(th1 - th2) - (m1 + m2) * g * np.sin(th1),
        m2 * L1 * w1**2 * np.sin(th1 - th2) - m2 * g * np.sin(th2)
    ])

    # Solve for accelerations
    accel = np.linalg.solve(M, f)
    return [w1, accel[0], w2, accel[1]]


def compute_energy(state, m1=1, m2=1, L1=1, L2=1, g=9.81):
    """Compute total mechanical energy (should be conserved)."""
    th1, w1, th2, w2 = state

    # Kinetic energy
    T = 0.5 * (m1 + m2) * L1**2 * w1**2 + \
        0.5 * m2 * L2**2 * w2**2 + \
        m2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2)

    # Potential energy (relative to hanging position)
    V = -(m1 + m2) * g * L1 * np.cos(th1) - m2 * g * L2 * np.cos(th2)

    return T + V


def generate_pendulum_regime():
    """Generate double pendulum with phase shift."""

    print("=" * 70)
    print("DOUBLE PENDULUM PHASE SHIFT GENERATOR")
    print("=" * 70)

    # Phase 1: Stable (low energy, small angles)
    print("\n[Phase 1] STABLE OSCILLATION (t=0-25)")
    print("    Initial: θ1=θ2=0.1 rad, ω1=ω2=0")

    t_stable = np.linspace(0, 25, 2500)
    stable_init = [0.1, 0, 0.1, 0]  # Small angles = periodic
    sol_stable = solve_ivp(
        double_pendulum_dynamics,
        [0, 25],
        stable_init,
        t_eval=t_stable,
        method='RK45'
    )

    E_stable = compute_energy(sol_stable.y[:, 0])
    print(f"    Points: {len(t_stable)}")
    print(f"    Energy: {E_stable:.4f} J")

    # Phase 2: Chaotic (high energy kick)
    print("\n[Phase 2] CHAOTIC DYNAMICS (t=25-50)")
    print("    Kick: θ1=π, θ2=π/2, ω1=2, ω2=-1")

    t_chaotic = np.linspace(25, 50, 2500)
    kick_state = [np.pi, 2.0, np.pi/2, -1.0]  # High energy state
    sol_chaotic = solve_ivp(
        double_pendulum_dynamics,
        [25, 50],
        kick_state,
        t_eval=t_chaotic,
        method='RK45'
    )

    E_chaotic = compute_energy(kick_state)
    print(f"    Points: {len(t_chaotic)}")
    print(f"    Energy: {E_chaotic:.4f} J")
    print(f"    Energy ratio (chaotic/stable): {E_chaotic/E_stable:.1f}x")

    # Concatenate
    print("\n[Concatenating phases]")
    t_full = np.concatenate([sol_stable.t, sol_chaotic.t])
    theta1 = np.concatenate([sol_stable.y[0], sol_chaotic.y[0]])
    omega1 = np.concatenate([sol_stable.y[1], sol_chaotic.y[1]])
    theta2 = np.concatenate([sol_stable.y[2], sol_chaotic.y[2]])
    omega2 = np.concatenate([sol_stable.y[3], sol_chaotic.y[3]])

    trajectory_df = pl.DataFrame({
        't': t_full,
        'theta1': theta1,
        'omega1': omega1,
        'theta2': theta2,
        'omega2': omega2
    })

    # Create observations
    base_date = date(2020, 1, 1)
    rows = []
    for i in range(len(t_full)):
        obs_date = base_date + timedelta(days=i)
        rows.append({'signal_id': 'pend_theta1', 'obs_date': obs_date, 'value': float(theta1[i])})
        rows.append({'signal_id': 'pend_omega1', 'obs_date': obs_date, 'value': float(omega1[i])})
        rows.append({'signal_id': 'pend_theta2', 'obs_date': obs_date, 'value': float(theta2[i])})
        rows.append({'signal_id': 'pend_omega2', 'obs_date': obs_date, 'value': float(omega2[i])})

    obs_df = pl.DataFrame(rows)

    # Save to data directory
    data_root = get_data_root()
    data_root.mkdir(parents=True, exist_ok=True)

    traj_path = data_root / 'pendulum_trajectory.parquet'
    obs_path = data_root / 'observations.parquet'

    write_parquet_atomic(trajectory_df, traj_path)
    write_parquet_atomic(obs_df, obs_path)

    print(f"\n    Total points: {len(t_full)}")
    print(f"    Observations: {len(obs_df)}")
    print(f"    Saved: {obs_path}")

    print("\n" + "=" * 70)
    print("GROUND TRUTH")
    print("=" * 70)
    print(f"  Regime transition: t=25.0 (row 2500)")
    print(f"  Transition date: {base_date + timedelta(days=2500)}")
    print(f"  Phase 1 (stable): rows 0-2499")
    print(f"  Phase 2 (chaotic): rows 2500-4999")

    print("\n" + "=" * 70)
    print("EXPECTED PRISM BEHAVIOR")
    print("=" * 70)
    print("  Phase 1 (stable):")
    print("    - High RQA determinism (~0.99)")
    print("    - Low sample entropy (<0.1)")
    print("    - Periodic spectral signature")
    print("  Phase 2 (chaotic):")
    print("    - Still high RQA det (deterministic chaos)")
    print("    - Higher sample entropy")
    print("    - Broadband spectral signature")
    print("  Transition:")
    print("    - Break detector should flag t=25")
    print("    - Laplace divergence spike")
    print("    - Discontinuity engines should activate")

    print("\n" + "=" * 70)
    print("RUN PRISM")
    print("=" * 70)
    print("  python -m prism.entry_points.signal_vector --signal --domain pendulum_regime --force --report signal")

    return {
        'domain': domain,
        'total_points': len(t_full),
        'transition_row': 2500,
        'transition_time': 25.0,
        'energy_stable': E_stable,
        'energy_chaotic': E_chaotic
    }


if __name__ == '__main__':
    generate_pendulum_regime()
