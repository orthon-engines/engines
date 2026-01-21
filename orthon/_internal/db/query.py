"""
PRISM Query Layer

Polars-based query utilities for Parquet files.

Key Functions:
    describe_file(file) - Get column information for a file
    file_stats(file) - Get basic statistics for a file

Note:
    SQL query functions have been removed. Use Polars DataFrame operations
    directly instead:

    >>> import polars as pl
    >>> from orthon._internal.db import read_file, get_path, OBSERVATIONS, SIGNALS
    >>>
    >>> # Read and filter
    >>> observations = read_file(OBSERVATIONS)
    >>> sensor_data = observations.filter(pl.col('signal_id') == 'SENSOR_01')
    >>>
    >>> # Aggregations
    >>> avg_by_signal = observations.group_by('signal_id').agg(
    ...     pl.col('value').mean().alias('avg_value')
    ... )
    >>>
    >>> # Joins
    >>> obs = pl.read_parquet(get_path(OBSERVATIONS))
    >>> cohorts = pl.read_parquet(get_path(COHORTS))
    >>> joined = obs.join(cohorts, on='entity_id')
"""

import polars as pl

from orthon._internal.db.parquet_store import get_path, OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS


def describe_file(file: str) -> pl.DataFrame:
    """
    Get column information for a parquet file.

    Args:
        file: File constant (OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS)

    Returns:
        DataFrame with column_name, column_type columns

    Example:
        >>> describe_file(OBSERVATIONS)
        shape: (4, 2)
        +--------------+-------------+
        | column_name  | column_type |
        | ---          | ---         |
        | str          | str         |
        +==============+=============+
        | entity_id    | Utf8        |
        | signal_id    | Utf8        |
        | timestamp    | Float64     |
        | value        | Float64     |
        +--------------+-------------+
    """
    path = get_path(file)
    if not path.exists():
        return pl.DataFrame({"column_name": [], "column_type": []})

    lf = pl.scan_parquet(path)
    schema_dict = lf.schema

    return pl.DataFrame(
        {
            "column_name": list(schema_dict.keys()),
            "column_type": [str(dt) for dt in schema_dict.values()],
        }
    )


def file_stats(file: str) -> dict:
    """
    Get basic statistics for a parquet file.

    Args:
        file: File constant (OBSERVATIONS, SIGNALS, GEOMETRY, STATE, COHORTS)

    Returns:
        Dict with row_count, column_count, file_size_bytes

    Example:
        >>> stats = file_stats(OBSERVATIONS)
        >>> print(stats)
        {'row_count': 50000, 'column_count': 4, 'file_size_bytes': 1234567}
    """
    path = get_path(file)

    if not path.exists():
        return {"row_count": 0, "column_count": 0, "file_size_bytes": 0}

    lf = pl.scan_parquet(path)
    row_count = lf.select(pl.len()).collect().item()
    column_count = len(lf.schema)
    file_size = path.stat().st_size

    return {
        "row_count": row_count,
        "column_count": column_count,
        "file_size_bytes": file_size,
    }


# Backwards compatible aliases
describe_table = describe_file
table_stats = file_stats


# CLI support
if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="PRISM Query Interface")
    parser.add_argument("--domain", help="Domain name (cmapss, tep, femto, etc.)")
    parser.add_argument("--describe", "-d", help="Describe file (observations, signals, geometry, state, cohorts)")
    parser.add_argument("--stats", help="Show stats for file")

    args = parser.parse_args()

    if args.domain:
        os.environ["PRISM_DOMAIN"] = args.domain

    FILE_MAP = {
        "observations": OBSERVATIONS,
        "signals": SIGNALS,
        "geometry": GEOMETRY,
        "state": STATE,
        "cohorts": COHORTS,
    }

    if args.describe:
        file = FILE_MAP.get(args.describe.lower())
        if not file:
            print(f"Error: Unknown file '{args.describe}'. Use: observations, signals, geometry, state, cohorts")
            sys.exit(1)
        result = describe_file(file)
        print(result)

    elif args.stats:
        file = FILE_MAP.get(args.stats.lower())
        if not file:
            print(f"Error: Unknown file '{args.stats}'. Use: observations, signals, geometry, state, cohorts")
            sys.exit(1)
        stats = file_stats(file)
        for k, v in stats.items():
            print(f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v}")

    else:
        parser.print_help()
