"""
Structure Engine Runner

Runs all structure engines and outputs to parquet.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time

from .covariance_engine import CovarianceEngine, CovarianceConfig
from .eigenvalue_engine import EigenvalueEngine, EigenvalueConfig
from .koopman_engine import KoopmanEngine, KoopmanConfig
from .spectral_engine import SpectralEngine, SpectralConfig
from .wavelet_engine import WaveletEngine, WaveletConfig


@dataclass
class StructureRunnerConfig:
    """Configuration for structure runner."""
    min_samples: int = 30
    sample_rate: float = 1.0
    run_covariance: bool = True
    run_eigenvalue: bool = True
    run_koopman: bool = True
    run_spectral: bool = True
    run_wavelet: bool = True
    output_dir: Optional[str] = None


class StructureRunner:
    """
    Run all structure engines on observation data.

    Takes observations parquet with columns:
    - unit_id: Entity identifier
    - signal_id: Signal name
    - index: Time/sequence index
    - value: Measurement value

    Outputs structure.parquet with one row per entity.
    """

    def __init__(self, config: Optional[StructureRunnerConfig] = None):
        self.config = config or StructureRunnerConfig()

        # Initialize engines
        self.engines = {}

        if self.config.run_covariance:
            self.engines['covariance'] = CovarianceEngine(
                CovarianceConfig(min_samples=self.config.min_samples)
            )

        if self.config.run_eigenvalue:
            self.engines['eigenvalue'] = EigenvalueEngine(
                EigenvalueConfig(min_samples=self.config.min_samples)
            )

        if self.config.run_koopman:
            self.engines['koopman'] = KoopmanEngine(
                KoopmanConfig(min_samples=self.config.min_samples)
            )

        if self.config.run_spectral:
            self.engines['spectral'] = SpectralEngine(
                SpectralConfig(
                    min_samples=self.config.min_samples,
                    sample_rate=self.config.sample_rate
                )
            )

        if self.config.run_wavelet:
            self.engines['wavelet'] = WaveletEngine(
                WaveletConfig(min_samples=self.config.min_samples)
            )

    def run(
        self,
        observations_path: str,
        output_path: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Run structure engines on observations.

        Parameters
        ----------
        observations_path : str
            Path to observations parquet
        output_path : str, optional
            Path to write output parquet

        Returns
        -------
        pl.DataFrame
            Structure metrics for all entities
        """
        print(f"Loading observations from {observations_path}")
        obs = pl.read_parquet(observations_path)

        # Get unique entities
        entities = obs.select('unit_id').unique().to_series().to_list()
        print(f"Found {len(entities)} entities")

        results = []
        start_time = time.time()

        for i, unit_id in enumerate(entities):
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                print(f"Processing entity {i+1}/{len(entities)}: {unit_id} ({elapsed:.1f}s)")

            # Extract signals for this entity
            entity_obs = obs.filter(pl.col('unit_id') == unit_id)
            signals = self._extract_signals(entity_obs)

            if not signals:
                continue

            # Run each engine
            entity_result = {'unit_id': unit_id}

            for engine_name, engine in self.engines.items():
                try:
                    result = engine.compute(signals, unit_id)
                    row = engine.to_parquet_row(result)

                    # Prefix columns with engine name (except unit_id)
                    for k, v in row.items():
                        if k != 'unit_id':
                            entity_result[f'{engine_name}_{k}'] = v
                except Exception as e:
                    print(f"  Warning: {engine_name} failed for {unit_id}: {e}")

            results.append(entity_result)

        # Create DataFrame
        if results:
            df = pl.DataFrame(results)
        else:
            df = pl.DataFrame({'unit_id': []})

        # Write output
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(output_path)
            print(f"Wrote {len(df)} rows to {output_path}")

        elapsed = time.time() - start_time
        print(f"Structure analysis complete in {elapsed:.1f}s")

        return df

    def _extract_signals(
        self,
        entity_obs: pl.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Extract signals from entity observations."""
        signals = {}

        # Get unique signal IDs
        signal_ids = entity_obs.select('signal_id').unique().to_series().to_list()

        for sig_id in signal_ids:
            sig_data = (
                entity_obs
                .filter(pl.col('signal_id') == sig_id)
                .sort('index')
                .select('value')
                .to_series()
                .to_numpy()
            )

            if len(sig_data) >= self.config.min_samples:
                signals[sig_id] = sig_data

        return signals


def run_structure_analysis(
    observations_path: str,
    output_path: str,
    config_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Convenience function to run structure analysis.

    Parameters
    ----------
    observations_path : str
        Path to observations parquet
    output_path : str
        Path to write structure.parquet
    config_path : str, optional
        Path to YAML config (not yet implemented)

    Returns
    -------
    pl.DataFrame
        Structure metrics
    """
    runner = StructureRunner()
    return runner.run(observations_path, output_path)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m prism.engines.signal.structure.structure_runner <observations.parquet> <output.parquet>")
        sys.exit(1)

    observations_path = sys.argv[1]
    output_path = sys.argv[2]

    run_structure_analysis(observations_path, output_path)
