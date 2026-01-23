"""
orthon/pipeline.py
Unified pipeline interface for Ørthon diagnostic system.

Usage:
    from orthon import Pipeline

    pipeline = Pipeline(
        input_file="data.csv",
        output_dir="./results",
        entity_col="unit_id",
        time_col="cycle"
    )

    # Run full pipeline
    results = pipeline.run()

    # Or run stages individually
    pipeline.run_vector()
    pipeline.run_geometry()
    pipeline.run_state()
    pipeline.run_cohort()
"""

from pathlib import Path
from typing import Optional, Dict, Any
import polars as pl

from orthon.config import engines, cohort, pipeline as pipeline_config


class Pipeline:
    """
    Unified Ørthon pipeline.

    geometry leads — ørthon
    """

    def __init__(
        self,
        input_file: str | Path,
        output_dir: str | Path = "./results",
        entity_col: str = "entity_id",
        time_col: str = "timestamp",
        config_overrides: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pipeline.

        Args:
            input_file: Path to input data (csv, parquet, txt)
            output_dir: Directory for output parquet files
            entity_col: Column name for entity identifier
            time_col: Column name for time/cycle index
            config_overrides: Optional dict to override config values
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.entity_col = entity_col
        self.time_col = time_col
        self.config_overrides = config_overrides or {}

        # Validate input exists
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configs
        self._engines_config = engines()
        self._cohort_config = cohort()
        self._pipeline_config = pipeline_config()

        # Track outputs
        self.outputs: Dict[str, Path] = {}
        self._data: Optional[pl.DataFrame] = None

    # ================================================================
    # DATA LOADING
    # ================================================================

    def load(self) -> pl.DataFrame:
        """Load input data into Polars DataFrame."""
        suffix = self.input_file.suffix.lower()

        if suffix == '.parquet':
            self._data = pl.read_parquet(self.input_file)
        elif suffix == '.csv':
            self._data = pl.read_csv(self.input_file)
        elif suffix == '.txt':
            # Assume tab-separated
            self._data = pl.read_csv(self.input_file, separator='\t')
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # Validate required columns
        missing = []
        if self.entity_col not in self._data.columns:
            missing.append(self.entity_col)
        if self.time_col not in self._data.columns:
            missing.append(self.time_col)

        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {self._data.columns}")

        return self._data

    @property
    def data(self) -> pl.DataFrame:
        """Get loaded data, loading if necessary."""
        if self._data is None:
            self.load()
        return self._data

    # ================================================================
    # PIPELINE STAGES
    # ================================================================

    def run_vector(self) -> Path:
        """
        Run Vector layer - compute behavioral metrics per signal.

        Output: vector.parquet
        """
        from orthon._internal.entry_points import signal_vector

        output_path = self.output_dir / "vector.parquet"

        # Save observations first if needed
        obs_path = self.output_dir / "observations.parquet"
        if not obs_path.exists():
            self.data.write_parquet(obs_path)

        signal_vector.run_vector(
            data_root=str(self.output_dir),
            entity_col=self.entity_col,
            time_col=self.time_col,
        )

        self.outputs['vector'] = output_path
        return output_path

    def run_geometry(self) -> Path:
        """
        Run Geometry layer - compute pairwise relationships.

        Requires: vector.parquet
        Output: geometry.parquet
        """
        from orthon._internal.entry_points import geometry

        if 'vector' not in self.outputs:
            raise RuntimeError("Must run vector layer first")

        output_path = self.output_dir / "geometry.parquet"

        geometry.run_geometry(
            data_root=str(self.output_dir),
            entity_col=self.entity_col,
            time_col=self.time_col,
        )

        self.outputs['geometry'] = output_path
        return output_path

    def run_state(self) -> Path:
        """
        Run State layer - regime detection, coherence tracking.

        Requires: geometry.parquet
        Output: state.parquet
        """
        from orthon._internal.entry_points import state

        if 'geometry' not in self.outputs:
            raise RuntimeError("Must run geometry layer first")

        output_path = self.output_dir / "state.parquet"

        state.run_state(
            data_root=str(self.output_dir),
            entity_col=self.entity_col,
            time_col=self.time_col,
        )

        self.outputs['state'] = output_path
        return output_path

    def run_cohort(self) -> Path:
        """
        Run Cohort discovery - group entities by behavior.

        Requires: vector.parquet, geometry.parquet, state.parquet
        Output: cohorts.parquet
        """
        from orthon.cohort import discover_cohorts

        # Check dependencies
        required = ['vector', 'geometry', 'state']
        missing = [r for r in required if r not in self.outputs]
        if missing:
            raise RuntimeError(f"Must run these layers first: {missing}")

        output_path = self.output_dir / "cohorts.parquet"

        result = discover_cohorts(
            vector_path=str(self.outputs['vector']),
            geometry_path=str(self.outputs['geometry']),
            state_path=str(self.outputs['state']),
        )

        # Save results
        result.save(str(output_path))

        self.outputs['cohorts'] = output_path
        return output_path

    # ================================================================
    # FULL PIPELINE
    # ================================================================

    def run(
        self,
        stages: Optional[list] = None,
        stop_after: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Run pipeline stages.

        Args:
            stages: List of stages to run. Default: all
                    Options: ['vector', 'geometry', 'state', 'cohorts']
            stop_after: Stop after this stage

        Returns:
            Dict mapping stage names to output paths
        """
        all_stages = ['vector', 'geometry', 'state', 'cohorts']

        if stages is None:
            stages = all_stages

        # Validate stages
        for s in stages:
            if s not in all_stages:
                raise ValueError(f"Unknown stage: {s}. Options: {all_stages}")

        # Ensure proper order
        stages = [s for s in all_stages if s in stages]

        # Load data first
        self.load()

        # Run stages
        for stage in stages:
            print(f"Running {stage}...")

            if stage == 'vector':
                self.run_vector()
            elif stage == 'geometry':
                self.run_geometry()
            elif stage == 'state':
                self.run_state()
            elif stage == 'cohorts':
                self.run_cohort()

            if stop_after and stage == stop_after:
                break

        return self.outputs

    # ================================================================
    # UTILITIES
    # ================================================================

    def summary(self) -> str:
        """Get pipeline summary."""
        lines = [
            "Ørthon Pipeline Summary",
            "=" * 40,
            f"Input: {self.input_file}",
            f"Output: {self.output_dir}",
            f"Entity column: {self.entity_col}",
            f"Time column: {self.time_col}",
            "",
            "Outputs:",
        ]

        if self.outputs:
            for stage, path in self.outputs.items():
                size = path.stat().st_size / 1024 if path.exists() else 0
                lines.append(f"  {stage}: {path.name} ({size:.1f} KB)")
        else:
            lines.append("  (none yet)")

        return "\n".join(lines)

    def list_outputs(self) -> Dict[str, Path]:
        """List all generated output files."""
        return self.outputs

    def get_output(self, stage: str) -> pl.DataFrame:
        """Load output parquet for a stage."""
        if stage not in self.outputs:
            raise ValueError(f"Stage '{stage}' not run yet. Available: {list(self.outputs.keys())}")
        return pl.read_parquet(self.outputs[stage])


# ================================================================
# CONVENIENCE FUNCTIONS
# ================================================================

def run(
    input_file: str | Path,
    output_dir: str | Path = "./results",
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
    stages: Optional[list] = None
) -> Dict[str, Path]:
    """
    Convenience function to run pipeline.

    Usage:
        from orthon import run
        results = run("data.csv", entity_col="unit_id", time_col="cycle")
    """
    pipeline = Pipeline(
        input_file=input_file,
        output_dir=output_dir,
        entity_col=entity_col,
        time_col=time_col
    )
    return pipeline.run(stages=stages)
