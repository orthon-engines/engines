"""
PRISM CLI

New architecture: Typology → Signal Vector → State Vector → Geometry

Usage:
    python -m prism <data_dir>                    # Run full pipeline
    python -m prism typology <data_dir>           # Run typology only
    python -m prism signal-vector <data_dir>      # Run signal vector (aggregate)
    python -m prism signal-vector-temporal <data_dir>  # Run signal vector (temporal)
    python -m prism state-vector <data_dir>       # Run state vector
    python -m prism dynamics <data_dir>           # Run dynamics
    python -m prism geometry <data_dir>           # Run geometry pipeline
    python -m prism --legacy <manifest.yaml>      # Run legacy 53-engine pipeline
"""

import sys
from pathlib import Path


def run_typology(data_dir: Path) -> dict:
    """Run typology engine."""
    from prism.engines.typology_engine import run_typology as run_typ

    obs_path = data_dir / 'observations.parquet'
    output_path = data_dir / 'typology.parquet'

    print(f"[TYPOLOGY] {obs_path} → {output_path}")
    result = run_typ(str(obs_path), str(output_path))
    return {'typology': output_path, 'rows': len(result)}


def run_signal_vector(data_dir: Path, temporal: bool = True) -> dict:
    """Run signal vector (temporal by default - includes I column)."""
    typology_path = data_dir / 'typology.parquet'
    obs_path = data_dir / 'observations.parquet'
    output_path = data_dir / 'signal_vector.parquet'

    if temporal:
        from prism.signal_vector_temporal import compute_signal_vector_temporal_sql
        print(f"[SIGNAL VECTOR] → {output_path}")
        result = compute_signal_vector_temporal_sql(
            str(obs_path), str(typology_path), str(output_path)
        )
    else:
        # Deprecated: aggregate mode loses I column
        from prism.signal_vector import run_signal_vector as run_sv
        print(f"[SIGNAL VECTOR (aggregate - deprecated)] → {output_path}")
        result = run_sv(str(data_dir))

    return {'signal_vector': output_path, 'rows': len(result)}


def run_state_vector(data_dir: Path) -> dict:
    """Run state vector."""
    from prism.engines.state_vector import compute_state_vector

    signal_vector_path = data_dir / 'signal_vector.parquet'
    typology_path = data_dir / 'typology.parquet'
    output_path = data_dir / 'state_vector.parquet'

    print(f"[STATE VECTOR] → {output_path}")
    result = compute_state_vector(
        str(signal_vector_path), str(typology_path), str(output_path)
    )
    return {'state_vector': output_path, 'rows': len(result)}


def run_geometry(data_dir: Path) -> dict:
    """Run geometry pipeline (signal_geometry + signal_pairwise)."""
    import polars as pl

    signal_vector_path = data_dir / 'signal_vector.parquet'
    state_vector_path = data_dir / 'state_vector.parquet'

    results = {}

    # Signal geometry
    print(f"[SIGNAL GEOMETRY]")
    from prism.engines.signal_geometry import compute_signal_geometry
    try:
        sig_geom = compute_signal_geometry(
            str(signal_vector_path), str(state_vector_path),
            str(data_dir / 'signal_geometry.parquet')
        )
        results['signal_geometry'] = len(sig_geom)
    except Exception as e:
        print(f"  Warning: {e}")
        results['signal_geometry'] = 0

    # Signal pairwise
    print(f"[SIGNAL PAIRWISE]")
    from prism.engines.signal_pairwise import compute_signal_pairwise
    try:
        sig_pair = compute_signal_pairwise(
            str(signal_vector_path), str(state_vector_path),
            str(data_dir / 'signal_pairwise.parquet')
        )
        results['signal_pairwise'] = len(sig_pair)
    except Exception as e:
        print(f"  Warning: {e}")
        results['signal_pairwise'] = 0

    return results


def run_dynamics(data_dir: Path) -> dict:
    """Run dynamics engines."""
    import polars as pl
    from prism.engines.dynamics_runner import run_dynamics
    from prism.engines.information_flow_runner import run_information_flow

    obs = pl.read_parquet(data_dir / 'observations.parquet')

    results = {}

    print(f"[DYNAMICS]")
    try:
        dyn = run_dynamics(obs, data_dir)
        results['dynamics'] = len(dyn) if len(dyn) > 0 else 0
    except Exception as e:
        print(f"  Warning: {e}")
        results['dynamics'] = 0

    print(f"[INFORMATION FLOW]")
    try:
        info = run_information_flow(obs, data_dir)
        results['information_flow'] = len(info) if len(info) > 0 else 0
    except Exception as e:
        print(f"  Warning: {e}")
        results['information_flow'] = 0

    return results


def run_sql(data_dir: Path) -> dict:
    """Run SQL engines."""
    from prism.sql_runner import SQLRunner

    obs_path = data_dir / 'observations.parquet'

    print(f"[SQL ENGINES]")
    runner = SQLRunner(obs_path, data_dir, engines=['zscore', 'statistics', 'correlation'])
    result = runner.run()

    return result


def run_full_pipeline(data_dir: Path) -> dict:
    """Run full new pipeline."""
    print("=" * 70)
    print("PRISM PIPELINE (New Architecture)")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print()

    results = {}

    # 1. Typology
    results['typology'] = run_typology(data_dir)

    # 2. Signal Vector (temporal - includes I column)
    results['signal_vector'] = run_signal_vector(data_dir, temporal=True)

    # 3. State Vector
    results['state_vector'] = run_state_vector(data_dir)

    # 4. Geometry
    results['geometry'] = run_geometry(data_dir)

    # 5. SQL
    results['sql'] = run_sql(data_dir)

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return results


def run_legacy(manifest_path: Path) -> dict:
    """Run legacy 53-engine pipeline."""
    from prism._legacy.runner import run
    return run(manifest_path)


def main():
    """PRISM CLI entry point."""

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg1 = sys.argv[1]

    # Legacy mode
    if arg1 == '--legacy':
        if len(sys.argv) < 3:
            print("Usage: python -m prism --legacy <manifest.yaml>")
            sys.exit(1)
        result = run_legacy(Path(sys.argv[2]))
        print(f"Results: {result}")
        return 0

    # Command mode
    commands = {
        'typology': lambda d: run_typology(d),
        'signal-vector': lambda d: run_signal_vector(d, temporal=False),
        'signal-vector-temporal': lambda d: run_signal_vector(d, temporal=True),
        'state-vector': lambda d: run_state_vector(d),
        'geometry': lambda d: run_geometry(d),
        'dynamics': lambda d: run_dynamics(d),
        'sql': lambda d: run_sql(d),
    }

    if arg1 in commands:
        if len(sys.argv) < 3:
            print(f"Usage: python -m prism {arg1} <data_dir>")
            sys.exit(1)
        data_dir = Path(sys.argv[2])
        result = commands[arg1](data_dir)
        print(f"Result: {result}")
        return 0

    # Default: full pipeline on data_dir
    data_dir = Path(arg1)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    result = run_full_pipeline(data_dir)
    print(f"\nResults: {result}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
