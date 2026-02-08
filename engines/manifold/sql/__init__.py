"""
ENGINES SQL Engines - Pure SQL computations via DuckDB.

Fast primitives for statistics, z-scores, correlations, and regime assignment.

Available SQL engines:
- zscore: Z-score normalization and anomaly detection (sensitive to outliers)
- mad_anomaly: MAD-based anomaly detection (robust to outliers)
- statistics: Summary statistics per signal
- correlation: Correlation matrix

See docs/NORMALIZATION.md for guidance on zscore vs mad_anomaly.
"""

from pathlib import Path
from typing import List

SQL_DIR = Path(__file__).parent


def get_sql(name: str) -> str:
    """Load SQL file by name."""
    path = SQL_DIR / f"{name}.sql"
    if not path.exists():
        raise FileNotFoundError(f"SQL engine not found: {name}")
    return path.read_text()


def list_sql_engines() -> List[str]:
    """List available SQL engines."""
    return [p.stem for p in SQL_DIR.glob("*.sql")]


__all__ = ['get_sql', 'list_sql_engines', 'SQL_DIR']
