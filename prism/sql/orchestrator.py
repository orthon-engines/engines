"""
PRISM SQL Orchestrator

CANONICAL RULE: Orchestrators are PURE.

This main orchestrator:
  - Loads observations
  - Runs all SQL stages in order
  - Queries views by name
  - Exports to parquet

NO computation. NO inline SQL. NO business logic.
All logic lives in the SQL files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import duckdb

from .stages import (
    LoadStage,
    CalculusStage,
    StatisticsStage,
    ClassificationStage,
    TypologyStage,
    GeometryStage,
    DynamicsStage,
    CausalityStage,
    EntropyStage,
    PhysicsStage,
    ManifoldStage,
)


# Stage execution order
STAGES = [
    ('load', LoadStage),
    ('calculus', CalculusStage),
    ('statistics', StatisticsStage),
    ('classification', ClassificationStage),
    ('typology', TypologyStage),
    ('geometry', GeometryStage),
    ('dynamics', DynamicsStage),
    ('causality', CausalityStage),
    ('entropy', EntropyStage),
    ('physics', PhysicsStage),
    ('manifold', ManifoldStage),
]


class SQLOrchestrator:
    """
    Main SQL pipeline orchestrator.

    PURE PLUMBING ONLY:
      - load_observations()  : Load parquet into database
      - run_stage()          : Execute a single stage
      - run_all()            : Execute all stages in order
      - query()              : Query any view by name
      - export()             : Write views to parquet

    NO computation. NO inline SQL. NO business logic.
    """

    def __init__(self, db_path: str = ':memory:'):
        """
        Initialize orchestrator.

        Args:
            db_path: DuckDB database path (':memory:' for in-memory)
        """
        self.conn = duckdb.connect(db_path)
        self._stages: Dict[str, Any] = {}
        self._output_dir: Optional[Path] = None

        # Initialize all stage orchestrators
        for name, stage_class in STAGES:
            self._stages[name] = stage_class(self.conn)

    # ═══════════════════════════════════════════════════════════════════════
    # LOAD: Import data (no transformation)
    # ═══════════════════════════════════════════════════════════════════════

    def load_observations(self, path: str) -> int:
        """
        Load observations parquet.

        PURE: Just creates table from file. No transformation.

        Returns:
            Number of rows loaded
        """
        load_stage = self._stages['load']
        load_stage.load_observations(path)
        return load_stage.get_row_count()

    def load_primitives(self, path: str) -> None:
        """
        Load PRISM primitives (if available).

        PURE: Just creates table from file. No transformation.
        """
        self.conn.execute(f"CREATE OR REPLACE TABLE primitives AS SELECT * FROM '{path}'")

    # ═══════════════════════════════════════════════════════════════════════
    # RUN: Execute stages (no logic, just sequence)
    # ═══════════════════════════════════════════════════════════════════════

    def run_stage(self, stage_name: str) -> None:
        """
        Run a single stage.

        PURE: Just loads and executes SQL file.

        Args:
            stage_name: One of the stage names (load, calculus, etc.)
        """
        if stage_name not in self._stages:
            raise ValueError(f"Unknown stage: {stage_name}")
        self._stages[stage_name].run()

    def run_all(self, stop_after: Optional[str] = None) -> List[str]:
        """
        Run all stages in order.

        PURE: Just sequences stage execution. No logic.

        Args:
            stop_after: Optional stage name to stop after

        Returns:
            List of stages executed
        """
        executed = []
        for name, _ in STAGES:
            self.run_stage(name)
            executed.append(name)
            if name == stop_after:
                break
        return executed

    # ═══════════════════════════════════════════════════════════════════════
    # QUERY: Get data from views (no inline SQL)
    # ═══════════════════════════════════════════════════════════════════════

    def query(self, view_name: str):
        """
        Query a view by name.

        PURE: Just executes SELECT * FROM view_name.

        Args:
            view_name: Name of view to query

        Returns:
            DataFrame
        """
        return self.conn.execute(f"SELECT * FROM {view_name}").fetchdf()

    def get_stage(self, stage_name: str):
        """Get a stage orchestrator for direct access."""
        return self._stages.get(stage_name)

    def get_prism_work_order(self) -> list:
        """
        Get PRISM work order from typology stage.

        PURE: Just queries the view. Logic is in SQL.
        """
        return self._stages['typology'].get_prism_work_order()

    def get_system_summary(self) -> dict:
        """Get system-level summary."""
        df = self.query('v_system_summary')
        if len(df) > 0:
            return df.iloc[0].to_dict()
        return {}

    def get_alerts(self) -> list:
        """Get current alerts and anomalies."""
        return self.query('v_alerts').to_dict(orient='records')

    # ═══════════════════════════════════════════════════════════════════════
    # EXPORT: Write to parquet (no transformation)
    # ═══════════════════════════════════════════════════════════════════════

    def set_output_dir(self, path: str) -> None:
        """Set output directory for exports."""
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, view_name: str, filename: str) -> Path:
        """
        Export a view to parquet.

        PURE: Just COPY view TO file.

        Args:
            view_name: View to export
            filename: Output filename

        Returns:
            Path to exported file
        """
        if self._output_dir is None:
            raise RuntimeError("Output directory not set. Call set_output_dir() first.")

        output_path = self._output_dir / filename
        self.conn.execute(f"COPY {view_name} TO '{output_path}' (FORMAT PARQUET)")
        return output_path

    def export_all(self) -> Dict[str, Path]:
        """
        Export all standard outputs including engine tables.

        PURE: Just sequences exports. No logic.

        Returns:
            Dict mapping output name to path
        """
        # SQL view exports
        view_exports = {
            'signal_class': ('v_export_signal_class', 'signal_class.parquet'),
            'signal_typology': ('v_export_signal_typology', 'signal_typology.parquet'),
            'behavioral_geometry': ('v_export_behavioral_geometry', 'behavioral_geometry.parquet'),
            'dynamical_systems': ('v_export_dynamical_systems', 'dynamical_systems.parquet'),
            'causal_mechanics': ('v_export_causal_mechanics', 'causal_mechanics.parquet'),
        }

        # Engine table exports (created by Python engines)
        table_exports = {
            # Typology engines
            'hurst': ('t_hurst', 'hurst.parquet'),
            'garch': ('t_garch', 'garch.parquet'),
            # Geometry engines
            'lof_scores': ('t_lof_scores', 'lof_scores.parquet'),
            'clusters': ('t_clusters', 'clusters.parquet'),
            'pca': ('t_pca', 'pca.parquet'),
            # Dynamics engines
            'lyapunov': ('t_lyapunov', 'lyapunov.parquet'),
            'attractor': ('t_attractor', 'attractor.parquet'),
            'basin': ('t_basin', 'basin.parquet'),
            'dmd': ('t_dmd', 'dmd.parquet'),
            # Causality engines
            'granger': ('t_granger', 'granger.parquet'),
            'transfer_entropy': ('t_transfer_entropy', 'transfer_entropy.parquet'),
            'cointegration': ('t_cointegration', 'cointegration.parquet'),
            'causal_graph': ('t_causal_graph', 'causal_graph.parquet'),
            # Entropy engines
            'entropy': ('t_entropy', 'entropy.parquet'),
            'mutual_info': ('t_mutual_info', 'mutual_info.parquet'),
        }

        paths = {}

        # Export SQL views
        for name, (view, filename) in view_exports.items():
            try:
                paths[name] = self.export(view, filename)
            except Exception as e:
                print(f"Warning: Could not export {name}: {e}")

        # Export engine tables
        for name, (table, filename) in table_exports.items():
            try:
                # Check if table exists
                self.conn.execute(f"SELECT 1 FROM {table} LIMIT 0")
                paths[name] = self.export(table, filename)
            except Exception:
                # Table doesn't exist (engine didn't run or failed)
                pass

        return paths

    def export_manifold_json(self, filename: str = 'manifold.json') -> Path:
        """
        Export manifold data as JSON for viewer.

        PURE: Just queries view and writes JSON.
        """
        if self._output_dir is None:
            raise RuntimeError("Output directory not set.")

        manifold_data = self._stages['manifold'].get_manifold_json()
        output_path = self._output_dir / filename
        output_path.write_text(json.dumps(manifold_data, indent=2))
        return output_path

    def write_manifest(self) -> Path:
        """
        Write manifest.json with run metadata.

        PURE: Just records metadata. No logic.
        """
        if self._output_dir is None:
            raise RuntimeError("Output directory not set.")

        manifest = {
            'generated_at': datetime.now().isoformat(),
            'stages_executed': [name for name, _ in STAGES],
            'files': {}
        }

        # Record output files
        for f in self._output_dir.glob('*.parquet'):
            row_count = self.conn.execute(f"SELECT COUNT(*) FROM '{f}'").fetchone()[0]
            manifest['files'][f.name] = {
                'rows': row_count,
                'path': str(f)
            }

        manifest_path = self._output_dir / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest_path

    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATION
    # ═══════════════════════════════════════════════════════════════════════

    def validate_stage(self, stage_name: str) -> bool:
        """Check if a stage's views are all queryable."""
        return self._stages[stage_name].validate()

    def validate_all(self) -> Dict[str, bool]:
        """Validate all stages."""
        return {name: stage.validate() for name, stage in self._stages.items()}

    # ═══════════════════════════════════════════════════════════════════════
    # PIPELINE: Full run sequence
    # ═══════════════════════════════════════════════════════════════════════

    def run_pipeline(
        self,
        observations_path: str,
        output_dir: str,
        primitives_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full pipeline.

        PURE: Just sequences operations. No logic.

        Args:
            observations_path: Path to observations parquet
            output_dir: Directory for outputs
            primitives_path: Optional PRISM primitives parquet

        Returns:
            Dict with run results
        """
        # 1. Load observations
        n_rows = self.load_observations(observations_path)
        print(f"Loaded {n_rows:,} observations")

        # 2. Load PRISM primitives if provided
        if primitives_path:
            self.load_primitives(primitives_path)
            print("Loaded PRISM primitives")

        # 3. Run all stages
        print("\nExecuting stages:")
        for name, _ in STAGES:
            print(f"  {name}...", end=' ', flush=True)
            try:
                self.run_stage(name)
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
                raise

        # 4. Export
        self.set_output_dir(output_dir)
        paths = self.export_all()
        manifest_path = self.write_manifest()

        print(f"\nExported {len(paths)} files to {output_dir}")

        return {
            'status': 'complete',
            'input_rows': n_rows,
            'output_dir': str(output_dir),
            'files': [str(p) for p in paths.values()],
            'manifest': str(manifest_path),
        }


def main():
    """CLI entry point for full pipeline."""
    import sys

    if len(sys.argv) < 2:
        print("PRISM SQL Pipeline - Full Engine Mode")
        print("=" * 50)
        print("\nUsage: python -m prism.sql.orchestrator <input.parquet> [output_dir]")
        print("\nStages (runs ALL SQL + ALL Python engines):")
        for name, stage_class in STAGES:
            doc = stage_class.__doc__ or ''
            print(f"  {name}: {doc.strip().split(chr(10))[0]}")
        print("\nExamples:")
        print("  python -m prism.sql.orchestrator data/observations.parquet outputs/")
        print("  python -m prism.sql.orchestrator data/observations.parquet /Users/jasonrudder/Domains/cwru")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'outputs'

    # Check for --primitives flag
    primitives_path = None
    if '--primitives' in sys.argv:
        idx = sys.argv.index('--primitives')
        if idx + 1 < len(sys.argv):
            primitives_path = sys.argv[idx + 1]

    print("=" * 60)
    print("PRISM FULL PIPELINE")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    orchestrator = SQLOrchestrator()
    result = orchestrator.run_pipeline(input_path, output_dir, primitives_path)

    print("\n" + "=" * 60)
    print(f"COMPLETE: {len(result['files'])} files → {output_dir}")
    print("=" * 60)

    # List output files
    for f in sorted(Path(output_dir).glob('*.parquet')):
        try:
            count = orchestrator.conn.execute(f"SELECT COUNT(*) FROM '{f}'").fetchone()[0]
            print(f"  {f.name}: {count:,} rows")
        except Exception:
            print(f"  {f.name}: (unknown)")


if __name__ == '__main__':
    main()
