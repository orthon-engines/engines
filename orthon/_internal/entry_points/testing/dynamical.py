"""
Dynamical Systems Generator
===========================

Generates 4 deterministic dynamical systems for PRISM validation.

Usage:
    python -m prism.entry_points.generate_dynamical

Output:
    data/<domain>/raw/observations.parquet for each system
"""

import numpy as np
import polars as pl
from datetime import date, timedelta
from scipy.integrate import solve_ivp

from orthon._internal.db.parquet_store import get_data_root
from orthon._internal.db.polars_io import write_parquet_atomic


def create_observations(t: np.ndarray, data: dict, domain: str) -> pl.DataFrame:
    """Convert trajectory arrays to PRISM observations format."""
    base_date = date(2020, 1, 1)
    n_points = len(t)

    rows = []
    for i in range(n_points):
        obs_date = base_date + timedelta(days=i)
        for name, values in data.items():
            rows.append({
                'signal_id': f'{domain}_{name}',
                'obs_date': obs_date,
                'value': float(values[i])
            })

    return pl.DataFrame(rows)


def save_domain(domain: str, trajectory_df: pl.DataFrame, obs_df: pl.DataFrame):
    """Save to data directory."""
    data_root = get_data_root()
    data_root.mkdir(parents=True, exist_ok=True)

    write_parquet_atomic(trajectory_df, data_root / f'{domain}_trajectory.parquet')
    write_parquet_atomic(obs_df, data_root / 'observations.parquet')

    return data_root / 'observations.parquet'


def generate_datasets():
    """Generate all 4 dynamical systems."""

    # 1. Lorenz System
    print("\n[1/4] LORENZ SYSTEM")
    def lorenz(t, state, sigma=10, rho=28, beta=8/3):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    sol_lorenz = solve_ivp(lorenz, [0, 50], [1.0, 1.0, 1.0],
                           t_eval=np.linspace(0, 50, 5000))
    trajectory_df = pl.DataFrame({
        't': sol_lorenz.t, 'x': sol_lorenz.y[0],
        'y': sol_lorenz.y[1], 'z': sol_lorenz.y[2]
    })
    obs_df = create_observations(sol_lorenz.t, {'x': sol_lorenz.y[0], 'y': sol_lorenz.y[1], 'z': sol_lorenz.y[2]}, 'lorenz')
    path = save_domain('lorenz', trajectory_df, obs_df)
    print(f"    Points: {len(sol_lorenz.t):,}")
    print(f"    Saved: {path}")

    # 2. Rössler Attractor
    print("\n[2/4] RÖSSLER ATTRACTOR")
    def rossler(t, state, a=0.2, b=0.2, c=5.7):
        x, y, z = state
        return [-y - z, x + a * y, b + z * (x - c)]

    sol_rossler = solve_ivp(rossler, [0, 100], [0.1, 0.0, 0.1],
                            t_eval=np.linspace(0, 100, 10000))
    trajectory_df = pl.DataFrame({
        't': sol_rossler.t, 'x': sol_rossler.y[0],
        'y': sol_rossler.y[1], 'z': sol_rossler.y[2]
    })
    obs_df = create_observations(sol_rossler.t, {'x': sol_rossler.y[0], 'y': sol_rossler.y[1], 'z': sol_rossler.y[2]}, 'rossler')
    path = save_domain('rossler', trajectory_df, obs_df)
    print(f"    Points: {len(sol_rossler.t):,}")
    print(f"    Saved: {path}")

    # 3. Double Pendulum
    print("\n[3/4] DOUBLE PENDULUM")
    def double_pendulum(t, state, m1=1, m2=1, L1=1, L2=1, g=9.81):
        th1, w1, th2, w2 = state
        delta = th2 - th1

        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        d_w1 = (m2 * L1 * w1**2 * np.sin(delta) * np.cos(delta) +
                m2 * g * np.sin(th2) * np.cos(delta) +
                m2 * L2 * w2**2 * np.sin(delta) -
                (m1 + m2) * g * np.sin(th1)) / den1

        den2 = (L2 / L1) * den1
        d_w2 = (-m2 * L2 * w2**2 * np.sin(delta) * np.cos(delta) +
                (m1 + m2) * g * np.sin(th1) * np.cos(delta) -
                (m1 + m2) * L1 * w1**2 * np.sin(delta) -
                (m1 + m2) * g * np.sin(th2)) / den2

        return [w1, d_w1, w2, d_w2]

    sol_pendulum = solve_ivp(double_pendulum, [0, 20], [np.pi/2, 0, np.pi/2, 0],
                             t_eval=np.linspace(0, 20, 2000))
    trajectory_df = pl.DataFrame({
        't': sol_pendulum.t,
        'theta1': sol_pendulum.y[0], 'omega1': sol_pendulum.y[1],
        'theta2': sol_pendulum.y[2], 'omega2': sol_pendulum.y[3]
    })
    obs_df = create_observations(sol_pendulum.t, {
        'theta1': sol_pendulum.y[0], 'omega1': sol_pendulum.y[1],
        'theta2': sol_pendulum.y[2], 'omega2': sol_pendulum.y[3]
    }, 'pendulum')
    path = save_domain('pendulum', trajectory_df, obs_df)
    print(f"    Points: {len(sol_pendulum.t):,}")
    print(f"    Saved: {path}")

    # 4. Lotka-Volterra
    print("\n[4/4] LOTKA-VOLTERRA")
    def lotka_volterra(t, state, alpha=1.1, beta=0.4, delta=0.1, gamma=0.4):
        x, y = state
        return [alpha * x - beta * x * y, delta * x * y - gamma * y]

    sol_lv = solve_ivp(lotka_volterra, [0, 100], [10, 5],
                       t_eval=np.linspace(0, 100, 2000))
    trajectory_df = pl.DataFrame({
        't': sol_lv.t, 'prey': sol_lv.y[0], 'predator': sol_lv.y[1]
    })
    obs_df = create_observations(sol_lv.t, {'prey': sol_lv.y[0], 'predator': sol_lv.y[1]}, 'lotka')
    path = save_domain('lotka_volterra', trajectory_df, obs_df)
    print(f"    Points: {len(sol_lv.t):,}")
    print(f"    Saved: {path}")

    print("\n" + "=" * 60)
    print("SUCCESS: 4 domains created")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("DYNAMICAL SYSTEMS GENERATOR")
    print("=" * 60)
    generate_datasets()
