"""
PRISM SQL Runner

Executes engines from sql/ folder via DuckDB.
"""

import duckdb
from pathlib import Path
from typing import List, Dict, Any
import os


# Available SQL engines
SQL_ENGINES = [
    'zscore',
    'correlation',
    'statistics',
    'regime_assignment'
]


class SQLRunner:
    """
    Executes SQL-based engines via DuckDB.

    SQL engines are good for:
    - Simple aggregations (statistics)
    - Window functions (zscore, regimes)
    - Joins (correlation across signals)
    - Operations on large datasets
    """

    def __init__(
        self,
        observations_path: Path,
        output_dir: Path,
        engines: List[str],
        params: Dict[str, Any] = None
    ):
        self.observations_path = Path(observations_path)
        self.output_dir = Path(output_dir)
        self.engines = engines
        self.params = params or {}

        # SQL files location
        self.sql_dir = Path(__file__).parent / 'engines' / 'sql'

        # DuckDB connection
        self.conn = None

    def run(self) -> dict:
        """Execute ALL SQL engines. No exceptions."""
        engines_to_run = SQL_ENGINES  # ALL engines, always

        print(f"\n[SQL ENGINES] Running ALL: {engines_to_run}")

        # Connect to DuckDB
        self.conn = duckdb.connect(':memory:')
        self._configure_duckdb()

        # Load observations
        self._load_observations()

        # Optionally load Python outputs for reference
        self._load_python_outputs()

        # Run each engine
        results = {}
        for engine_name in engines_to_run:
            try:
                self._run_engine(engine_name)
                results[engine_name] = 'success'
            except Exception as e:
                print(f"  Error in {engine_name}: {e}")
                results[engine_name] = f'error: {e}'

        # Close connection
        self.conn.close()

        return {
            'engines_run': len(engines_to_run),
            'results': results
        }

    def _configure_duckdb(self):
        """Configure DuckDB for optimal performance."""
        cpu_count = os.cpu_count() or 4

        self.conn.execute(f"PRAGMA threads={cpu_count}")
        self.conn.execute("PRAGMA memory_limit='4GB'")

    def _load_observations(self):
        """Load observations into DuckDB."""
        self.conn.execute(f"""
            CREATE TABLE observations AS
            SELECT * FROM read_parquet('{self.observations_path}')
        """)

        count = self.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        n_entities = self.conn.execute("SELECT COUNT(DISTINCT unit_id) FROM observations").fetchone()[0]
        n_signals = self.conn.execute("SELECT COUNT(DISTINCT signal_id) FROM observations").fetchone()[0]

        print(f"  Loaded {count:,} observations ({n_entities} entities, {n_signals} signals)")

        # Create index for performance
        self.conn.execute("""
            CREATE INDEX idx_obs_entity_signal_I
            ON observations(unit_id, signal_id, I)
        """)

    def _load_python_outputs(self):
        """
        Load outputs from Python runner for SQL engines that need them.

        For example, regime_assignment might need primitives.parquet

        Note: These are optional reference tables. SQL engines should work
        on observations alone, but may use enriched data if available.
        """
        # Try to load primitives (optional)
        primitives_path = self.output_dir / 'primitives.parquet'
        try:
            if primitives_path.exists():
                self.conn.execute(f"""
                    CREATE TABLE primitives AS
                    SELECT * FROM read_parquet('{primitives_path}')
                """)
                print(f"  Loaded primitives.parquet for SQL reference")
        except Exception as e:
            print(f"  Note: Could not load primitives.parquet: {e}")

        # Try to load enriched observations (optional)
        obs_enriched_path = self.output_dir / 'observations_enriched.parquet'
        try:
            if obs_enriched_path.exists():
                self.conn.execute(f"""
                    CREATE TABLE observations_enriched AS
                    SELECT * FROM read_parquet('{obs_enriched_path}')
                """)
                print(f"  Loaded observations_enriched.parquet for SQL reference")
        except Exception as e:
            print(f"  Note: Could not load observations_enriched.parquet: {e}")

    def _run_engine(self, engine_name: str):
        """Run a single SQL engine."""
        sql_path = self.sql_dir / f'{engine_name}.sql'

        if not sql_path.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_path}")

        print(f"  Running {engine_name}...")

        # Read SQL
        sql = sql_path.read_text()

        # Substitute any parameters
        engine_params = self.params.get(engine_name, {})
        for key, value in engine_params.items():
            sql = sql.replace(f'${{{key}}}', str(value))

        # Strip trailing semicolon for COPY compatibility
        sql = sql.strip()
        if sql.endswith(';'):
            sql = sql[:-1].strip()

        # Export the results directly to parquet
        try:
            output_path = self.output_dir / f'{engine_name}.parquet'

            self.conn.execute(f"""
                COPY ({sql})
                TO '{output_path}' (FORMAT PARQUET)
            """)

            # Report
            count = self.conn.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
            print(f"    → {engine_name}.parquet: {count:,} rows")

        except Exception as e:
            raise RuntimeError(f"Could not run {engine_name}: {e}")
