"""
PRISM Dynamic State Runner
==========================

Computes system-level energy state from signal and geometry field vectors.

Pipeline:
---------
signal_field.parquet + geometry_field.parquet → dynamic_state.py → state/system.parquet

Energy Decomposition:
--------------------
system_energy = signal_energy_sum + coupling_energy

Where:
- signal_energy_sum: Sum of individual signal energies (from gradient magnitudes)
- coupling_energy: Energy stored in pairwise relationships (from geometry field)

The coupling energy represents "hidden" energy stored in the relational structure.
When correlations break, this energy is released into the system.

Usage:
------
    python -m prism.entry_points.dynamic_state
    python -m prism.entry_points.dynamic_state --verbose
"""

import argparse
import logging
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from orthon._internal.db.parquet_store import (
    get_path,
    get_data_root,
    ensure_directory,
    OBSERVATIONS,
    SIGNALS,
    GEOMETRY,
    STATE,
    COHORTS,
)
from orthon._internal.db.polars_io import write_parquet_atomic
from orthon._internal.utils.memory import force_gc, get_memory_usage_mb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# ENERGY COMPUTATION
# =============================================================================

def compute_signal_energy(
    signal_field: pl.DataFrame,
    window_end: datetime,
) -> Dict[str, float]:
    """
    Compute aggregate signal energy for a window.

    Energy proxy = sum of squared gradient magnitudes.
    High gradients = high activity = high energy.

    Args:
        signal_field: DataFrame with signal field vectors
        window_end: The window timestamp

    Returns:
        Dict with signal energy metrics
    """
    # Filter to window
    window_data = signal_field.filter(pl.col('window_end') == window_end)

    if len(window_data) == 0:
        return {
            'signal_energy_sum': 0.0,
            'signal_energy_mean': 0.0,
            'signal_energy_std': 0.0,
            'n_signals': 0,
        }

    # Get gradient magnitudes
    if 'gradient_magnitude' in window_data.columns:
        grads = window_data['gradient_magnitude'].drop_nulls().to_numpy()
    elif 'gradient' in window_data.columns:
        grads = np.abs(window_data['gradient'].drop_nulls().to_numpy())
    else:
        return {
            'signal_energy_sum': 0.0,
            'signal_energy_mean': 0.0,
            'signal_energy_std': 0.0,
            'n_signals': 0,
        }

    if len(grads) == 0:
        return {
            'signal_energy_sum': 0.0,
            'signal_energy_mean': 0.0,
            'signal_energy_std': 0.0,
            'n_signals': 0,
        }

    # Energy = sum of squared gradients (kinetic energy analogy)
    energy_sum = float(np.sum(grads ** 2))
    energy_mean = float(np.mean(grads ** 2))
    energy_std = float(np.std(grads ** 2)) if len(grads) > 1 else 0.0

    return {
        'signal_energy_sum': energy_sum,
        'signal_energy_mean': energy_mean,
        'signal_energy_std': energy_std,
        'n_signals': len(grads),
    }


def compute_coupling_energy(
    geometry_field: pl.DataFrame,
    window_end: datetime,
) -> Dict[str, float]:
    """
    Compute coupling energy from geometry field (pairwise relationships).

    Coupling energy = energy stored in pairwise correlations.
    When correlations break, this energy is released.

    Args:
        geometry_field: DataFrame with geometry field vectors (from pairwise)
        window_end: The window timestamp

    Returns:
        Dict with coupling energy metrics
    """
    # Filter to window
    window_data = geometry_field.filter(pl.col('window_end') == window_end)

    if len(window_data) == 0:
        return {
            'coupling_energy_sum': 0.0,
            'coupling_energy_mean': 0.0,
            'coupling_energy_std': 0.0,
            'n_pairs': 0,
            'mean_correlation': None,
            'correlation_dispersion': None,
        }

    # Look for coupling_energy metric in geometry field
    if 'metric_name' in window_data.columns:
        # Long format - filter to coupling_energy
        coupling_rows = window_data.filter(pl.col('metric_name') == 'coupling_energy')
        if len(coupling_rows) > 0 and 'metric_value' in coupling_rows.columns:
            values = coupling_rows['metric_value'].drop_nulls().to_numpy()
        else:
            values = np.array([])

        # Also get correlation metrics
        corr_rows = window_data.filter(pl.col('metric_name') == 'pearson_r')
        if len(corr_rows) > 0 and 'metric_value' in corr_rows.columns:
            corrs = corr_rows['metric_value'].drop_nulls().to_numpy()
        else:
            corrs = np.array([])
    else:
        # Wide format - look for coupling_energy column
        if 'coupling_energy' in window_data.columns:
            values = window_data['coupling_energy'].drop_nulls().to_numpy()
        else:
            values = np.array([])

        if 'pearson_r' in window_data.columns:
            corrs = window_data['pearson_r'].drop_nulls().to_numpy()
        else:
            corrs = np.array([])

    # Compute coupling energy aggregates
    if len(values) > 0:
        coupling_sum = float(np.sum(values))
        coupling_mean = float(np.mean(values))
        coupling_std = float(np.std(values)) if len(values) > 1 else 0.0
    else:
        coupling_sum = 0.0
        coupling_mean = 0.0
        coupling_std = 0.0

    # Correlation statistics
    if len(corrs) > 0:
        mean_corr = float(np.mean(corrs))
        corr_disp = float(np.std(corrs)) if len(corrs) > 1 else 0.0
    else:
        mean_corr = None
        corr_disp = None

    return {
        'coupling_energy_sum': coupling_sum,
        'coupling_energy_mean': coupling_mean,
        'coupling_energy_std': coupling_std,
        'n_pairs': len(values),
        'mean_correlation': mean_corr,
        'correlation_dispersion': corr_disp,
    }


def compute_system_energy(
    signal_energy: Dict[str, float],
    coupling_energy: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute total system energy.

    system_energy = signal_energy + coupling_energy

    The coupling energy is "hidden" in the relational structure.
    Total energy should be approximately conserved (physics test).

    Args:
        signal_energy: From compute_signal_energy
        coupling_energy: From compute_coupling_energy

    Returns:
        Dict with system energy metrics
    """
    ind_sum = signal_energy.get('signal_energy_sum', 0.0)
    coup_sum = coupling_energy.get('coupling_energy_sum', 0.0)

    system_total = ind_sum + coup_sum

    # Energy partition
    if system_total > 0:
        signal_fraction = ind_sum / system_total
        coupling_fraction = coup_sum / system_total
    else:
        signal_fraction = 0.5
        coupling_fraction = 0.5

    return {
        'system_energy': system_total,
        'signal_fraction': signal_fraction,
        'coupling_fraction': coupling_fraction,
    }


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_dynamic_state(
    signal_field_path: Optional[Path] = None,
    geometry_field_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Compute system-level dynamic state from field vectors.

    Args:
        signal_field_path: Path to signal_field.parquet
        geometry_field_path: Path to geometry_field.parquet
        output_path: Path for output system.parquet
        verbose: Print progress

    Returns:
        DataFrame with system state metrics per window
    """
    ensure_directory()

    # Default paths
    if signal_field_path is None:
        signal_field_path = get_path(SIGNALS)
    if geometry_field_path is None:
        geometry_field_path = get_path(GEOMETRY)
    if output_path is None:
        output_path = get_path(STATE)

    if verbose:
        print("=" * 70)
        print("DYNAMIC STATE COMPUTATION")
        print("=" * 70)
        print(f"Signal field: {signal_field_path}")
        print(f"Geometry field: {geometry_field_path}")

    # Check inputs exist
    if not signal_field_path.exists():
        raise FileNotFoundError(f"Signal field not found: {signal_field_path}")

    has_geometry = geometry_field_path.exists()
    if not has_geometry:
        if verbose:
            print(f"[WARN] Geometry field not found: {geometry_field_path}")
            print("       Running without coupling energy (signal energy only)")

    # Load signal field (lazy for memory)
    signal_lazy = pl.scan_parquet(signal_field_path)
    windows = (
        signal_lazy
        .select('window_end')
        .unique()
        .sort('window_end')
        .collect()['window_end']
        .to_list()
    )

    if verbose:
        print(f"Windows to process: {len(windows)}")
        print()

    # Load geometry field if available
    if has_geometry:
        geometry_df = pl.read_parquet(geometry_field_path)
    else:
        geometry_df = None

    # Process window by window
    results = []
    start_mem = get_memory_usage_mb()

    for i, window_end in enumerate(windows):
        # Load signal data for this window
        ind_window = (
            pl.scan_parquet(signal_field_path)
            .filter(pl.col('window_end') == window_end)
            .collect()
        )

        # Compute signal energy
        ind_energy = compute_signal_energy(ind_window, window_end)

        # Compute coupling energy if geometry available
        if geometry_df is not None:
            coup_energy = compute_coupling_energy(geometry_df, window_end)
        else:
            coup_energy = {
                'coupling_energy_sum': 0.0,
                'coupling_energy_mean': 0.0,
                'coupling_energy_std': 0.0,
                'n_pairs': 0,
                'mean_correlation': None,
                'correlation_dispersion': None,
            }

        # Compute system energy
        sys_energy = compute_system_energy(ind_energy, coup_energy)

        # Build result row
        row = {
            'window_end': window_end,
            **ind_energy,
            **coup_energy,
            **sys_energy,
        }
        results.append(row)

        # Progress
        if verbose and (i + 1) % 10 == 0:
            mem = get_memory_usage_mb()
            print(f"  Window {i+1}/{len(windows)}: system_energy={sys_energy['system_energy']:.2f} [mem: {mem:.0f} MB]")

        # Cleanup
        del ind_window
        if (i + 1) % 50 == 0:
            force_gc()

    # Create output DataFrame
    state_df = pl.DataFrame(results)

    if verbose and len(state_df) > 0:
        print("\n" + "=" * 70)
        print("DYNAMIC STATE SUMMARY")
        print("=" * 70)
        print(f"Total windows: {len(state_df)}")

        if 'system_energy' in state_df.columns:
            mean_sys = state_df['system_energy'].mean()
            std_sys = state_df['system_energy'].std()
            print(f"System energy: mean={mean_sys:.2f}, std={std_sys:.2f}" if mean_sys else "System energy: N/A")

        if 'signal_fraction' in state_df.columns:
            mean_frac = state_df['signal_fraction'].mean()
            print(f"Mean signal fraction: {mean_frac:.2%}" if mean_frac else "Mean signal fraction: N/A")

        if 'mean_correlation' in state_df.columns:
            mean_corr = state_df['mean_correlation'].mean()
            print(f"Mean pairwise correlation: {mean_corr:.4f}" if mean_corr else "Mean pairwise correlation: N/A")

    # Save
    write_parquet_atomic(state_df, output_path)

    if verbose:
        print(f"\nSaved: {output_path}")
        end_mem = get_memory_usage_mb()
        print(f"Memory: {start_mem:.0f} → {end_mem:.0f} MB")

    return state_df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Dynamic State Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Computes system-level energy state from signal and geometry field vectors.

Energy Decomposition:
    system_energy = signal_energy_sum + coupling_energy_sum

Where:
    - signal_energy: Activity of individual signals (gradient magnitudes)
    - coupling_energy: Energy stored in pairwise relationships

Examples:
    python -m prism.entry_points.dynamic_state
    python -m prism.entry_points.dynamic_state --verbose
        """
    )
    parser.add_argument(
        '--signal-field',
        type=str,
        help='Input signal_field.parquet'
    )
    parser.add_argument(
        '--geometry-field',
        type=str,
        help='Input geometry_field.parquet'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output system.parquet'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Extra verbose output'
    )

    args = parser.parse_args()

    signal_path = Path(args.signal_field) if args.signal_field else None
    geometry_path = Path(args.geometry_field) if args.geometry_field else None
    output_path = Path(args.output) if args.output else None

    state_df = run_dynamic_state(
        signal_field_path=signal_path,
        geometry_field_path=geometry_path,
        output_path=output_path,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print("\n" + "=" * 70)
        print("DYNAMIC STATE COMPLETE")
        print("=" * 70)
        print(f"Output: {output_path or get_path(STATE)}")
        print(f"Rows: {len(state_df):,}")


if __name__ == '__main__':
    main()
