"""
PRISM Derivation Generator
==========================

CLI tool to generate mathematical derivation documents with actual data values.

Usage:
    python -m prism.derivations.generator --engine hurst --signal lorenz_x --window 47
    python -m prism.derivations.generator --all-engines --signal lorenz_x --window 47
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl

from orthon._internal.db.parquet_store import (
    get_path,
    get_data_root,
    OBSERVATIONS,
    SIGNALS,
    GEOMETRY,
    STATE,
    COHORTS,
)


# Registry of engines that support derivation
DERIVABLE_ENGINES = {
    'hurst': 'prism.engines.hurst',
    'lyapunov': 'prism.engines.lyapunov',
    'sample_entropy': 'prism.engines.entropy',
    'permutation_entropy': 'prism.engines.entropy',
    'dfa': 'prism.engines.characterize',
    'spectral_entropy': 'prism.engines.spectral',
    'garch': 'prism.engines.garch',
}


def get_signal_data(signal_id: str, window_idx: int = None) -> tuple:
    """
    Load signal data from observations parquet.

    Returns:
        tuple: (values array, window_start, window_end, window_id)
    """
    obs_path = get_path(OBSERVATIONS)

    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Observations not found: {obs_path}")

    obs = pl.read_parquet(obs_path)

    # Filter to signal
    ind_obs = obs.filter(pl.col('signal_id') == signal_id).sort('obs_date')

    if ind_obs.height == 0:
        raise ValueError(f"No observations found for signal: {signal_id}")

    values = ind_obs['value'].to_numpy()
    dates = ind_obs['obs_date'].to_list()

    # If window_idx specified, use windowed data
    if window_idx is not None:
        # Load window size from config
        try:
            from orthon._internal.utils.stride import load_stride_config
            config = load_stride_config()
            if hasattr(config, 'windows') and 'anchor' in config.windows:
                window_size = config.windows['anchor'].window_days
                stride_size = config.windows['anchor'].stride_days
            else:
                raise RuntimeError("No anchor window configured")
        except Exception as e:
            raise RuntimeError(f"No window configuration found in config/stride.yaml: {e}")

        if window_idx * stride_size + window_size > len(values):
            raise ValueError(f"Window {window_idx} exceeds data range")

        start_idx = window_idx * stride_size
        end_idx = start_idx + window_size

        values = values[start_idx:end_idx]
        window_start = str(dates[start_idx])
        window_end = str(dates[min(end_idx - 1, len(dates) - 1)])
        window_id = str(window_idx)
    else:
        window_start = str(dates[0])
        window_end = str(dates[-1])
        window_id = "full"

    return values, window_start, window_end, window_id


def generate_derivation(engine_name: str, signal_id: str,
                        window_idx: int = None, output_dir: str = None) -> str:
    """
    Generate a derivation document for the specified engine and data.

    Returns:
        str: Path to generated markdown file
    """
    # Load data
    values, window_start, window_end, window_id = get_signal_data(
        signal_id, window_idx
    )

    print(f"Loaded {len(values)} observations for {signal_id}")
    print(f"Window: {window_start} to {window_end}")

    # Get the derivation function
    if engine_name == 'hurst':
        from orthon._internal.engines.hurst import compute_hurst_with_derivation
        result, derivation = compute_hurst_with_derivation(
            values,
            signal_id=signal_id,
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
        )
    elif engine_name == 'lyapunov':
        from orthon._internal.engines.lyapunov import compute_lyapunov_with_derivation
        result, derivation = compute_lyapunov_with_derivation(
            values,
            signal_id=signal_id,
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
        )
    elif engine_name == 'sample_entropy':
        from orthon._internal.engines.entropy import compute_sample_entropy_with_derivation
        result, derivation = compute_sample_entropy_with_derivation(
            values,
            signal_id=signal_id,
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
        )
    elif engine_name == 'permutation_entropy':
        from orthon._internal.engines.entropy import compute_permutation_entropy_with_derivation
        result, derivation = compute_permutation_entropy_with_derivation(
            values,
            signal_id=signal_id,
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
        )
    elif engine_name == 'dfa':
        from orthon._internal.engines.characterize import compute_dfa_with_derivation
        result, derivation = compute_dfa_with_derivation(
            values,
            signal_id=signal_id,
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
        )
    elif engine_name == 'spectral_entropy':
        from orthon._internal.engines.spectral import compute_spectral_entropy_with_derivation
        result, derivation = compute_spectral_entropy_with_derivation(
            values,
            signal_id=signal_id,
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
        )
    elif engine_name == 'garch':
        from orthon._internal.engines.garch import compute_garch_with_derivation
        result, derivation = compute_garch_with_derivation(
            values,
            signal_id=signal_id,
            window_id=window_id,
            window_start=window_start,
            window_end=window_end,
        )
    else:
        raise NotImplementedError(f"Derivation not implemented for engine: {engine_name}")

    # Set data path for reproducibility
    derivation.data_path = get_path(OBSERVATIONS)

    # Generate markdown
    markdown = derivation.to_markdown()

    # Determine output path
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'docs' / 'derivations'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{engine_name}_{signal_id}_w{window_id}.md"
    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        f.write(markdown)

    print(f"Generated: {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate mathematical derivation documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate Hurst derivation for lorenz_x, window 47
    python -m prism.derivations.generator --engine hurst --signal lorenz_x --window 47

    # Generate derivation using full data series
    python -m prism.derivations.generator --engine hurst --signal lorenz_x

    # Specify output directory
    python -m prism.derivations.generator --engine hurst --signal lorenz_x --output docs/derivations/
"""
    )

    parser.add_argument('--engine', type=str, required=True,
                        choices=list(DERIVABLE_ENGINES.keys()),
                        help='Engine to generate derivation for')
    parser.add_argument('--signal', type=str, required=True,
                        help='Signal ID to use for derivation')
    parser.add_argument('--window', type=int, default=None,
                        help='Window index (0-based, uses 252d window with 21d stride)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for generated markdown')
    parser.add_argument('--all-engines', action='store_true',
                        help='Generate derivations for all available engines')

    args = parser.parse_args()

    if args.all_engines:
        for engine in DERIVABLE_ENGINES.keys():
            try:
                generate_derivation(engine, args.signal, args.window, args.output)
            except NotImplementedError as e:
                print(f"Skipping {engine}: {e}")
    else:
        generate_derivation(args.engine, args.signal, args.window, args.output)


if __name__ == '__main__':
    main()
