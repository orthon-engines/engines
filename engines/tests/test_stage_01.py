"""Smoke test for stage_01_signal_vector (signal_vector.py entry point)."""
import tempfile
from pathlib import Path
import numpy as np
import polars as pl


def create_synthetic_observations(n_signals: int = 3, n_samples: int = 500) -> pl.DataFrame:
    """Generate minimal observations.parquet."""
    rows = []
    for sig_id in range(n_signals):
        for i in range(n_samples):
            rows.append({
                'signal_id': f'sig_{sig_id}',
                'I': i,
                'value': np.sin(i * 0.1) + np.random.normal(0, 0.1),
            })
    return pl.DataFrame(rows)


def create_minimal_manifest(signal_ids: list) -> dict:
    """Create manifest that ORTHON would produce."""
    return {
        'version': '2.5',
        'system': {
            'window': 100,
            'stride': 50,
        },
        'cohorts': {
            'default': {
                sig_id: {
                    'engines': ['statistics', 'trend'],  # minimal set
                    'window_size': 100,
                    'stride': 50,
                }
                for sig_id in signal_ids
            }
        },
        'skip_signals': [],
    }


def test_stage_01_smoke():
    """End-to-end smoke test."""
    from engines.entry_points.signal_vector import run

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic data
        np.random.seed(42)
        obs = create_synthetic_observations(n_signals=3, n_samples=500)
        obs_path = tmpdir / 'observations.parquet'
        obs.write_parquet(obs_path)

        # Create manifest
        signal_ids = obs['signal_id'].unique().to_list()
        manifest = create_minimal_manifest(signal_ids)

        # Run stage 01
        output_path = tmpdir / 'signal_vector.parquet'
        result = run(
            observations_path=str(obs_path),
            output_path=str(output_path),
            manifest=manifest,
            verbose=True,
        )

        # Validate output
        assert output_path.exists(), "Output file not created"
        assert len(result) > 0, "No rows computed"
        assert 'signal_id' in result.columns, "Missing signal_id"
        assert 'I' in result.columns, "Missing I (window index)"

        print(f"✓ Stage 01 produced {len(result)} rows")
        print(f"  Columns: {result.columns}")


def test_engine_registry():
    """Verify engine registry works."""
    from engines.manifold.registry import get_registry

    r = get_registry()
    engines = r.list_engines()

    assert len(engines) > 10, f"Expected 10+ engines, got {len(engines)}"
    assert 'statistics' in engines, "Missing statistics engine"
    assert 'trend' in engines, "Missing trend engine"
    assert 'hurst' in engines, "Missing hurst engine"

    print(f"✓ Registry has {len(engines)} engines")


def test_core_engine_imports():
    """Verify core engines import without error."""
    from engines.manifold.signal import (
        statistics,
        memory,
        complexity,
        spectral,
        trend,
        hurst,
    )

    # Verify they have compute functions
    assert hasattr(statistics, 'compute'), "statistics missing compute()"
    assert hasattr(memory, 'compute'), "memory missing compute()"
    assert hasattr(complexity, 'compute'), "complexity missing compute()"
    assert hasattr(hurst, 'compute'), "hurst missing compute()"

    print("✓ Core engine imports OK")


if __name__ == '__main__':
    print("=== Running smoke tests ===\n")

    print("Test 1: Engine registry")
    test_engine_registry()

    print("\nTest 2: Core engine imports")
    test_core_engine_imports()

    print("\nTest 3: Stage 01 smoke test")
    test_stage_01_smoke()

    print("\n=== All tests passed ===")
