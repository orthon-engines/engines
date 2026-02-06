"""
Stage 09a: Cohort Thermodynamics Entry Point
============================================

Pure orchestration - computes thermodynamic-like quantities for cohorts.

Inputs:
    - state_geometry.parquet
    - cohort_evolution.parquet (optional)

Output:
    - cohort_thermodynamics.parquet

Computes thermodynamic analogs:
    - Entropy (from eigenvalue distribution)
    - Energy (from effective dimension)
    - Temperature (from velocity variance)
    - Free energy landscape
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def compute_entropy(eigenvalues: np.ndarray) -> float:
    """Compute entropy from eigenvalue distribution."""
    # Normalize to probabilities
    eig = np.array(eigenvalues)
    eig = eig[eig > 0]  # Positive eigenvalues only

    if len(eig) == 0:
        return np.nan

    # Normalize
    p = eig / np.sum(eig)

    # Shannon entropy
    entropy = -np.sum(p * np.log(p + 1e-10))

    return float(entropy)


def compute_effective_temperature(velocities: np.ndarray) -> float:
    """
    Compute effective temperature from velocity distribution.

    In statistical mechanics: kT ~ <v^2> / 2
    Here we use variance of velocities as temperature proxy.
    """
    v = np.array(velocities)
    v = v[~np.isnan(v)]

    if len(v) < 2:
        return np.nan

    # Variance as temperature proxy
    return float(np.var(v))


def run(
    state_geometry_path: str,
    cohort_evolution_path: Optional[str] = None,
    output_path: str = "cohort_thermodynamics.parquet",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute thermodynamic quantities for cohorts.

    Args:
        state_geometry_path: Path to state_geometry.parquet
        cohort_evolution_path: Path to cohort_evolution.parquet (optional)
        output_path: Output path for cohort_thermodynamics.parquet
        verbose: Print progress

    Returns:
        Cohort thermodynamics DataFrame
    """
    if verbose:
        print("=" * 70)
        print("STAGE 09a: COHORT THERMODYNAMICS")
        print("Computing thermodynamic analogs")
        print("=" * 70)

    # Load state geometry
    sg = pl.read_parquet(state_geometry_path)

    if verbose:
        print(f"Loaded state_geometry: {sg.shape}")

    # Check for required columns
    has_effective_dim = 'effective_dim' in sg.columns
    has_eigenvalues = any('eigenvalue' in c or 'eig_' in c for c in sg.columns)

    # Get eigenvalue columns
    eig_cols = [c for c in sg.columns if 'eigenvalue' in c or 'eig_' in c]

    # Check for cohort
    has_cohort = 'cohort' in sg.columns

    results = []

    if has_cohort:
        cohorts = sg['cohort'].unique().to_list()
        if verbose:
            print(f"Cohorts: {len(cohorts)}")

        for cohort in cohorts:
            cohort_data = sg.filter(pl.col('cohort') == cohort).sort('I')

            if len(cohort_data) == 0:
                continue

            # Compute entropy from eigenvalues
            if eig_cols:
                eigenvalues = np.concatenate([
                    cohort_data[c].to_numpy() for c in eig_cols
                ])
                entropy = compute_entropy(eigenvalues)
            else:
                entropy = np.nan

            # Energy proxy: mean effective dimension
            if has_effective_dim:
                energy = float(cohort_data['effective_dim'].mean())
                energy_std = float(cohort_data['effective_dim'].std())
            else:
                energy = np.nan
                energy_std = np.nan

            # Compute velocity from effective_dim changes
            if has_effective_dim:
                eff_dims = cohort_data['effective_dim'].to_numpy()
                velocities = np.diff(eff_dims)
                temperature = compute_effective_temperature(velocities)
            else:
                temperature = np.nan

            # Free energy = Energy - Temperature * Entropy
            if not np.isnan(energy) and not np.isnan(temperature) and not np.isnan(entropy):
                free_energy = energy - temperature * entropy
            else:
                free_energy = np.nan

            results.append({
                'cohort': cohort,
                'entropy': entropy,
                'energy': energy,
                'energy_std': energy_std,
                'temperature': temperature,
                'free_energy': free_energy,
                'n_samples': len(cohort_data),
            })
    else:
        # Global thermodynamics
        if eig_cols:
            eigenvalues = np.concatenate([
                sg[c].to_numpy() for c in eig_cols
            ])
            entropy = compute_entropy(eigenvalues)
        else:
            entropy = np.nan

        if has_effective_dim:
            energy = float(sg['effective_dim'].mean())
            energy_std = float(sg['effective_dim'].std())
            velocities = np.diff(sg['effective_dim'].to_numpy())
            temperature = compute_effective_temperature(velocities)
        else:
            energy = np.nan
            energy_std = np.nan
            temperature = np.nan

        if not np.isnan(energy) and not np.isnan(temperature) and not np.isnan(entropy):
            free_energy = energy - temperature * entropy
        else:
            free_energy = np.nan

        results.append({
            'cohort': 'global',
            'entropy': entropy,
            'energy': energy,
            'energy_std': energy_std,
            'temperature': temperature,
            'free_energy': free_energy,
            'n_samples': len(sg),
        })

    # Build DataFrame
    df = pl.DataFrame(results) if results else pl.DataFrame()

    # Write output
    if len(df) > 0:
        df.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        print(f"Shape: {df.shape}")

        if len(df) > 0 and 'entropy' in df.columns:
            print("\nThermodynamic summary:")
            for row in df.iter_rows(named=True):
                print(f"  {row['cohort']}:")
                print(f"    Entropy: {row['entropy']:.3f}" if not np.isnan(row['entropy']) else "    Entropy: N/A")
                print(f"    Energy: {row['energy']:.3f}" if not np.isnan(row['energy']) else "    Energy: N/A")
                print(f"    Temperature: {row['temperature']:.6f}" if not np.isnan(row['temperature']) else "    Temperature: N/A")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 09a: Cohort Thermodynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes thermodynamic analogs for cohorts:
  - Entropy (from eigenvalue distribution)
  - Energy (from effective dimension)
  - Temperature (from velocity variance)
  - Free energy (F = E - TS)

Interpretation:
  High entropy = more disorder/spread
  High energy = higher effective dimension
  High temperature = more dynamic change

Example:
  python -m prism.entry_points.stage_09a_cohort_thermodynamics \\
      state_geometry.parquet -o cohort_thermodynamics.parquet
"""
    )
    parser.add_argument('state_geometry', help='Path to state_geometry.parquet')
    parser.add_argument('--cohort-evolution', help='Path to cohort_evolution.parquet')
    parser.add_argument('-o', '--output', default='cohort_thermodynamics.parquet',
                        help='Output path (default: cohort_thermodynamics.parquet)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.state_geometry,
        args.cohort_evolution,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
