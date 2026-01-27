"""
Base Stage Orchestrator

Each stage:
  1. Runs SQL file (for SQL-computable stuff)
  2. Calls Python engines (for real algorithms)
  3. Inserts results back to DuckDB
"""

from pathlib import Path
from typing import List
import duckdb
import numpy as np
import pandas as pd


class StageOrchestrator:
    """
    Base class for stage orchestrators.

    Each stage:
    1. Has a SQL file (sql/{stage_name}.sql)
    2. Creates views (v_*) from SQL
    3. Creates tables (t_*) from Python engines
    """

    # Override in subclass
    SQL_FILE: str = None
    VIEWS: List[str] = []
    TABLES: List[str] = []  # Engine-created tables
    DEPENDS_ON: List[str] = []

    # Subsampling for expensive engines
    MAX_SAMPLES = 5000

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        """
        Initialize with database connection.

        Args:
            conn: DuckDB connection (shared across all stages)
        """
        self.conn = conn
        self._sql_dir = Path(__file__).parent.parent / 'sql'
        self._loaded = False

    @property
    def sql_path(self) -> Path:
        """Path to this stage's SQL file."""
        if self.SQL_FILE is None:
            raise NotImplementedError(f"{self.__class__.__name__} must define SQL_FILE")
        return self._sql_dir / self.SQL_FILE

    def load_sql(self) -> str:
        """Load SQL from file. No modification."""
        return self.sql_path.read_text()

    def run(self) -> None:
        """
        Execute this stage: SQL first, then Python engines.
        """
        # 1. Run SQL
        sql = self.load_sql()
        try:
            self.conn.execute(sql)
        except Exception as e:
            raise RuntimeError(f"SQL execution failed in {self.SQL_FILE}: {e}") from e

        # 2. Run Python engines
        self._run_engines()

        self._loaded = True

    def _run_engines(self) -> None:
        """Override in subclass to call Python engines."""
        pass

    def _get_signal(self, signal_id: str) -> np.ndarray:
        """Get signal time series as numpy array."""
        df = self.conn.execute(f"""
            SELECT y FROM observations
            WHERE signal_id = '{signal_id}'
            ORDER BY I
        """).fetchdf()
        return df['y'].values if len(df) > 0 else np.array([])

    def _subsample(self, arr: np.ndarray, max_samples: int = None) -> np.ndarray:
        """Subsample array if too long."""
        max_samples = max_samples or self.MAX_SAMPLES
        if len(arr) > max_samples:
            indices = np.linspace(0, len(arr) - 1, max_samples, dtype=int)
            return arr[indices]
        return arr

    def _insert_df(self, table_name: str, df: pd.DataFrame) -> None:
        """Insert DataFrame as table."""
        self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")

    def get_views(self) -> List[str]:
        """Return list of views this stage creates."""
        return self.VIEWS.copy()

    def get_dependencies(self) -> List[str]:
        """Return list of views this stage depends on."""
        return self.DEPENDS_ON.copy()

    def validate(self) -> bool:
        """
        Validate all views exist.

        Returns True if all views are queryable.
        """
        for view in self.VIEWS:
            try:
                self.conn.execute(f"SELECT 1 FROM {view} LIMIT 0")
            except Exception:
                return False
        return True

    def query(self, view_name: str):
        """
        Query a view by name.

        Args:
            view_name: Name of view (must be in self.VIEWS)

        Returns:
            DataFrame
        """
        if view_name not in self.VIEWS:
            raise ValueError(f"View {view_name} not in {self.__class__.__name__}.VIEWS")
        return self.conn.execute(f"SELECT * FROM {view_name}").fetchdf()

    def __repr__(self):
        status = "loaded" if self._loaded else "not loaded"
        return f"<{self.__class__.__name__} [{status}] views={len(self.VIEWS)}>"
