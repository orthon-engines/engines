"""
PRISM Loader - Universal data ingestion for industrial diagnostics.

Handles CSV, TXT, Parquet files and converts to standardized observations format.
Auto-detects or accepts explicit column mappings.

Usage:
    # Auto mode
    loader = Loader.from_file("train_FD001.txt")

    # Explicit mapping
    loader = Loader.from_file("data.csv",
        entity_col="unit_id",
        timestamp_col="cycle",
        sensor_cols=["temp_1", "temp_2", "pressure"],
    )

    # Save schema for reuse
    loader.save_schema("my_schema.yaml")

    # Load test data with same schema
    test_loader = Loader.from_file("test.csv", schema="my_schema.yaml")

    # Get standardized observations
    observations = loader.observations
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import polars as pl


class Loader:
    """
    Universal data loader that converts any tabular format to standardized observations.

    Observations format:
        entity_id: str   - Unique identifier for each unit/machine/bearing
        timestamp: float - Time index (cycles, seconds, etc.)
        signal: str      - Signal/sensor name
        value: float     - Signal value
    """

    # Common C-MAPSS patterns
    CMAPSS_PATTERN = re.compile(r'(train|test)_FD\d{3}', re.IGNORECASE)

    # Common column name patterns for auto-detection
    ENTITY_PATTERNS = ['unit_id', 'entity_id', 'engine_id', 'bearing_id', 'machine_id', 'id', 'unit']
    TIMESTAMP_PATTERNS = ['cycle', 'timestamp', 'time', 't', 'cycles', 'timestep']
    TARGET_PATTERNS = ['rul', 'target', 'remaining_useful_life', 'ttf', 'time_to_failure']

    def __init__(
        self,
        df: pl.DataFrame,
        schema: Dict[str, Any],
        source_path: Optional[str] = None,
    ):
        """
        Initialize loader with dataframe and schema.

        Use Loader.from_file() or Loader.from_dataframe() instead of direct init.
        """
        self._df = df
        self._schema = schema
        self._source_path = source_path
        self._observations: Optional[pl.DataFrame] = None

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        schema: Optional[Union[str, Path, Dict]] = None,
        entity_col: Optional[Union[str, int]] = None,
        timestamp_col: Optional[Union[str, int]] = None,
        sensor_cols: Optional[List[Union[str, int]]] = None,
        op_cols: Optional[List[Union[str, int]]] = None,
        target_col: Optional[Union[str, int]] = None,
        separator: Optional[str] = None,
        has_header: Optional[bool] = None,
    ) -> "Loader":
        """
        Load data from file (CSV, TXT, or Parquet).

        Args:
            path: Path to data file
            schema: Path to schema YAML, or schema dict (overrides other args)
            entity_col: Column name or index for entity ID
            timestamp_col: Column name or index for timestamp
            sensor_cols: List of column names or indices for sensors
            op_cols: List of column names or indices for operating conditions
            target_col: Column name or index for target (optional)
            separator: Field separator (auto-detected if None)
            has_header: Whether file has header row (auto-detected if None)

        Returns:
            Loader instance with data loaded and schema resolved
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Load schema from file if provided as path
        if isinstance(schema, (str, Path)):
            schema = cls._load_schema_file(schema)

        # Detect file type and load
        suffix = path.suffix.lower()

        if suffix == '.parquet':
            df = pl.read_parquet(path)
            has_header = True  # Parquet always has column names

        elif suffix in ['.csv', '.txt']:
            # Auto-detect separator and header
            if separator is None or has_header is None:
                detected_sep, detected_header = cls._detect_format(path)
                separator = separator or detected_sep
                has_header = has_header if has_header is not None else detected_header

            df = pl.read_csv(
                path,
                separator=separator,
                has_header=has_header,
                truncate_ragged_lines=True,
                infer_schema_length=10000,
            )
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .txt, or .parquet")

        # Build or merge schema
        resolved_schema = cls._resolve_schema(
            df=df,
            path=path,
            schema=schema,
            entity_col=entity_col,
            timestamp_col=timestamp_col,
            sensor_cols=sensor_cols,
            op_cols=op_cols,
            target_col=target_col,
            has_header=has_header,
        )

        return cls(df=df, schema=resolved_schema, source_path=str(path))

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        schema: Optional[Dict] = None,
        entity_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        sensor_cols: Optional[List[str]] = None,
        op_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
    ) -> "Loader":
        """
        Create loader from existing Polars DataFrame.
        """
        resolved_schema = cls._resolve_schema(
            df=df,
            path=None,
            schema=schema,
            entity_col=entity_col,
            timestamp_col=timestamp_col,
            sensor_cols=sensor_cols,
            op_cols=op_cols,
            target_col=target_col,
            has_header=True,
        )

        return cls(df=df, schema=resolved_schema)

    @property
    def observations(self) -> pl.DataFrame:
        """
        Get standardized observations DataFrame.

        Format: entity_id | timestamp | signal | value
        """
        if self._observations is None:
            self._observations = self._build_observations()
        return self._observations

    @property
    def schema(self) -> Dict[str, Any]:
        """Get the resolved schema."""
        return self._schema.copy()

    @property
    def raw(self) -> pl.DataFrame:
        """Get the raw loaded DataFrame."""
        return self._df

    @property
    def entity_col(self) -> str:
        """Get entity column name."""
        return self._schema['entity_col']

    @property
    def timestamp_col(self) -> str:
        """Get timestamp column name."""
        return self._schema['timestamp_col']

    @property
    def sensor_cols(self) -> List[str]:
        """Get sensor column names."""
        return self._schema.get('sensor_cols', [])

    @property
    def signal_cols(self) -> List[str]:
        """Get all signal columns (sensors + ops)."""
        return self.sensor_cols + self._schema.get('op_cols', [])

    def save_schema(self, path: Union[str, Path]) -> None:
        """
        Save schema to YAML file for reuse.

        Args:
            path: Output path for schema YAML
        """
        path = Path(path)

        with open(path, 'w') as f:
            yaml.dump(self._schema, f, default_flow_style=False, sort_keys=False)

        print(f"Schema saved: {path}")

    def summary(self) -> None:
        """Print summary of loaded data."""
        obs = self.observations

        print("=" * 60)
        print("LOADER SUMMARY")
        print("=" * 60)

        if self._source_path:
            print(f"Source: {self._source_path}")

        print(f"\nRaw shape: {self._df.shape[0]:,} rows × {self._df.shape[1]} columns")
        print(f"Observations: {obs.shape[0]:,} rows")

        print(f"\nSchema:")
        print(f"  Entity column: {self.entity_col}")
        print(f"  Timestamp column: {self.timestamp_col}")
        print(f"  Sensor columns: {len(self.sensor_cols)}")
        if self._schema.get('op_cols'):
            print(f"  Operating condition columns: {len(self._schema['op_cols'])}")
        if self._schema.get('target_col'):
            print(f"  Target column: {self._schema['target_col']}")

        print(f"\nEntities: {obs['entity_id'].n_unique()}")
        print(f"Signals: {obs['signal'].n_unique()}")
        print(f"Timestamp range: [{obs['timestamp'].min()}, {obs['timestamp'].max()}]")

        # Check for nulls
        null_count = obs['value'].null_count()
        if null_count > 0:
            print(f"\n  Null values: {null_count:,} ({100*null_count/len(obs):.1f}%)")

    # =========================================================================
    # Private methods
    # =========================================================================

    @staticmethod
    def _detect_format(path: Path) -> tuple:
        """
        Auto-detect separator and header from file.

        Returns:
            (separator, has_header)
        """
        with open(path, 'r') as f:
            first_lines = [f.readline() for _ in range(5)]

        first_line = first_lines[0]

        # Detect separator
        if '\t' in first_line:
            separator = '\t'
        elif ',' in first_line:
            separator = ','
        elif '  ' in first_line or first_line.strip().count(' ') > 3:
            separator = ' '
        else:
            separator = ','  # default

        # Detect header: if first row has mostly non-numeric values, it's a header
        parts = first_line.strip().split(separator) if separator != ' ' else first_line.split()
        numeric_count = sum(1 for p in parts if Loader._is_numeric(p.strip()))
        has_header = numeric_count < len(parts) / 2

        return separator, has_header

    @staticmethod
    def _is_numeric(s: str) -> bool:
        """Check if string is numeric."""
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _load_schema_file(path: Union[str, Path]) -> Dict:
        """Load schema from YAML file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")

        with open(path, 'r') as f:
            return yaml.safe_load(f)

    @classmethod
    def _resolve_schema(
        cls,
        df: pl.DataFrame,
        path: Optional[Path],
        schema: Optional[Dict],
        entity_col: Optional[Union[str, int]],
        timestamp_col: Optional[Union[str, int]],
        sensor_cols: Optional[List[Union[str, int]]],
        op_cols: Optional[List[Union[str, int]]],
        target_col: Optional[Union[str, int]],
        has_header: bool,
    ) -> Dict[str, Any]:
        """
        Resolve schema from explicit args, provided schema, or auto-detection.
        """
        columns = df.columns

        # Start with provided schema or empty
        resolved = schema.copy() if schema else {}

        # Helper to resolve column reference (name or index) to name
        def resolve_col(ref: Union[str, int, None], default: Optional[str] = None) -> Optional[str]:
            if ref is None:
                return default
            if isinstance(ref, int):
                if 0 <= ref < len(columns):
                    return columns[ref]
                raise ValueError(f"Column index {ref} out of range")
            return ref

        def resolve_cols(refs: Optional[List[Union[str, int]]]) -> Optional[List[str]]:
            if refs is None:
                return None
            result = []
            for ref in refs:
                if isinstance(ref, int):
                    if 0 <= ref < len(columns):
                        result.append(columns[ref])
                    else:
                        raise ValueError(f"Column index {ref} out of range")
                else:
                    result.append(ref)
            return result

        # Override with explicit args
        if entity_col is not None:
            resolved['entity_col'] = resolve_col(entity_col)
        if timestamp_col is not None:
            resolved['timestamp_col'] = resolve_col(timestamp_col)
        if sensor_cols is not None:
            resolved['sensor_cols'] = resolve_cols(sensor_cols)
        if op_cols is not None:
            resolved['op_cols'] = resolve_cols(op_cols)
        if target_col is not None:
            resolved['target_col'] = resolve_col(target_col)

        # Auto-detect if C-MAPSS format
        if path and cls.CMAPSS_PATTERN.search(path.stem):
            resolved = cls._apply_cmapss_schema(df, resolved, has_header)

        # Auto-detect remaining missing fields
        resolved = cls._auto_detect_schema(df, resolved, has_header)

        # Validate required fields
        if 'entity_col' not in resolved:
            raise ValueError("Could not determine entity column. Specify entity_col=...")
        if 'timestamp_col' not in resolved:
            raise ValueError("Could not determine timestamp column. Specify timestamp_col=...")
        if 'sensor_cols' not in resolved or not resolved['sensor_cols']:
            raise ValueError("Could not determine sensor columns. Specify sensor_cols=...")

        return resolved

    @classmethod
    def _apply_cmapss_schema(
        cls,
        df: pl.DataFrame,
        schema: Dict,
        has_header: bool,
    ) -> Dict:
        """
        Apply C-MAPSS schema conventions.

        C-MAPSS format: unit_id, cycle, op_1, op_2, op_3, s_1, s_2, ..., s_21
        """
        columns = df.columns
        n_cols = len(columns)

        # C-MAPSS has 26+ columns (unit, cycle, 3 ops, 21 sensors, possibly extra)
        if n_cols >= 26:
            # Build readable column names
            readable_names = (
                ['unit_id', 'cycle'] +
                [f'op_{i}' for i in range(1, 4)] +
                [f's_{i}' for i in range(1, 22)]
            )

            # Extend if there are extra columns
            if n_cols > 26:
                readable_names.extend([f'extra_{i}' for i in range(n_cols - 26)])

            schema['column_names'] = readable_names

            # Store with readable names
            if 'entity_col' not in schema:
                schema['entity_col'] = 'unit_id'
            if 'timestamp_col' not in schema:
                schema['timestamp_col'] = 'cycle'
            if 'op_cols' not in schema:
                schema['op_cols'] = [f'op_{i}' for i in range(1, 4)]
            if 'sensor_cols' not in schema:
                schema['sensor_cols'] = [f's_{i}' for i in range(1, 22)]

        return schema

    @classmethod
    def _auto_detect_schema(
        cls,
        df: pl.DataFrame,
        schema: Dict,
        has_header: bool,
    ) -> Dict:
        """
        Auto-detect schema fields by column names and data patterns.
        """
        columns = df.columns

        # Helper to find column by patterns
        def find_column(patterns: List[str]) -> Optional[str]:
            for col in columns:
                col_lower = col.lower()
                for pattern in patterns:
                    if pattern in col_lower:
                        return col
            return None

        # Auto-detect entity column
        if 'entity_col' not in schema:
            detected = find_column(cls.ENTITY_PATTERNS)
            if detected:
                schema['entity_col'] = detected
            elif not has_header:
                # Assume first column is entity if no header
                schema['entity_col'] = columns[0]

        # Auto-detect timestamp column
        if 'timestamp_col' not in schema:
            detected = find_column(cls.TIMESTAMP_PATTERNS)
            if detected:
                schema['timestamp_col'] = detected
            elif not has_header and len(columns) > 1:
                # Assume second column is timestamp if no header
                schema['timestamp_col'] = columns[1]

        # Auto-detect target column
        if 'target_col' not in schema:
            detected = find_column(cls.TARGET_PATTERNS)
            if detected:
                schema['target_col'] = detected

        # Auto-detect sensor columns: all numeric columns not already assigned
        if 'sensor_cols' not in schema:
            assigned = {
                schema.get('entity_col'),
                schema.get('timestamp_col'),
                schema.get('target_col'),
            }
            assigned.update(schema.get('op_cols', []))

            sensor_cols = []
            for col in columns:
                if col not in assigned:
                    # Check if column is numeric
                    dtype = df[col].dtype
                    if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                        sensor_cols.append(col)

            if sensor_cols:
                schema['sensor_cols'] = sensor_cols

        return schema

    def _build_observations(self) -> pl.DataFrame:
        """
        Convert raw data to standardized observations format.

        Output: entity_id | timestamp | signal | value
        """
        df = self._df
        schema = self._schema

        # Rename columns first if mapping provided
        if 'column_names' in schema:
            if len(schema['column_names']) <= len(df.columns):
                # Only rename as many columns as we have names for
                rename_map = dict(zip(df.columns[:len(schema['column_names'])], schema['column_names']))
                df = df.rename(rename_map)

        entity_col = schema['entity_col']
        timestamp_col = schema['timestamp_col']
        signal_cols = schema.get('sensor_cols', []) + schema.get('op_cols', [])

        # Melt from wide to long format
        observations = df.select(
            [entity_col, timestamp_col] + signal_cols
        ).unpivot(
            index=[entity_col, timestamp_col],
            on=signal_cols,
            variable_name='signal',
            value_name='value',
        ).rename({
            entity_col: 'entity_id',
            timestamp_col: 'timestamp',
        })

        # Ensure correct types
        observations = observations.with_columns([
            pl.col('entity_id').cast(pl.Utf8),
            pl.col('timestamp').cast(pl.Float64),
            pl.col('signal').cast(pl.Utf8),
            pl.col('value').cast(pl.Float64),
        ])

        # Sort for consistency
        observations = observations.sort(['entity_id', 'timestamp', 'signal'])

        return observations


# =============================================================================
# Convenience functions
# =============================================================================

def load(
    path: Union[str, Path],
    schema: Optional[Union[str, Path, Dict]] = None,
    **kwargs,
) -> Loader:
    """
    Load data from file. Shorthand for Loader.from_file().

    Args:
        path: Path to data file (CSV, TXT, or Parquet)
        schema: Optional schema file or dict
        **kwargs: Additional arguments passed to Loader.from_file()

    Returns:
        Loader instance
    """
    return Loader.from_file(path, schema=schema, **kwargs)


def load_cmapss(
    train_path: Union[str, Path],
    test_path: Optional[Union[str, Path]] = None,
    rul_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """
    Load C-MAPSS dataset with standard conventions.

    Args:
        train_path: Path to training file (e.g., train_FD001.txt)
        test_path: Optional path to test file
        rul_path: Optional path to RUL ground truth file

    Returns:
        Tuple of (train_loader, test_loader, rul_array) - test/rul are None if not provided
    """
    train_loader = Loader.from_file(train_path)

    test_loader = None
    if test_path:
        # Use same schema as train
        test_loader = Loader.from_file(test_path, schema=train_loader.schema)

    rul = None
    if rul_path:
        rul_path = Path(rul_path)
        with open(rul_path, 'r') as f:
            rul = [float(line.strip()) for line in f if line.strip()]

    return train_loader, test_loader, rul


if __name__ == "__main__":
    # Demo/test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python loader.py <data_file>")
        print("       python loader.py train_FD001.txt")
        sys.exit(1)

    path = sys.argv[1]
    loader = Loader.from_file(path)
    loader.summary()

    print("\nFirst 10 observations:")
    print(loader.observations.head(10))

    print("\nSchema:")
    print(yaml.dump(loader.schema, default_flow_style=False))
