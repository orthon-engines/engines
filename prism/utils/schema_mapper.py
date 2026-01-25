"""
Schema Mapper
=============

Maps arbitrary input columns to PRISM's standard schema.

Standard Schema:
    entity_id   | String   | The thing that fails (engine, bearing, unit)
    signal_id   | String   | The measurement (sensor_1, temp, vibration)
    timestamp   | Float64  | Time (cycles, seconds, etc.)
    value       | Float64  | Raw measurement value

This module handles:
    - Wide format (one column per signal) → Long format
    - Column renaming from domain-specific to standard
    - Type casting to required dtypes
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import polars as pl


@dataclass
class ColumnMapping:
    """Column mapping configuration."""
    # Entity identification
    entity_column: Optional[str] = None  # Column containing entity_id
    entity_default: str = "unit_1"  # Default if no entity column

    # Time
    time_column: str = "timestamp"  # Column containing time

    # Signals - EITHER wide format OR long format
    # Wide format: signal_columns = ['sensor_1', 'sensor_2', ...]
    # Long format: signal_column = 'signal_id', value_column = 'value'
    signal_columns: Optional[List[str]] = None  # Wide format: list of signal columns
    signal_column: Optional[str] = None  # Long format: column containing signal_id
    value_column: Optional[str] = None  # Long format: column containing value

    # Additional columns to preserve (e.g., labels, metadata)
    preserve_columns: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: Dict) -> 'ColumnMapping':
        """Create from dictionary (typically from YAML)."""
        return cls(
            entity_column=d.get('entity_column'),
            entity_default=d.get('entity_default', 'unit_1'),
            time_column=d.get('time_column', 'timestamp'),
            signal_columns=d.get('signal_columns'),
            signal_column=d.get('signal_column'),
            value_column=d.get('value_column'),
            preserve_columns=d.get('preserve_columns'),
        )

    def is_wide_format(self) -> bool:
        """Check if config specifies wide format (one column per signal)."""
        return self.signal_columns is not None

    def is_long_format(self) -> bool:
        """Check if config specifies long format (signal_id, value columns)."""
        return self.signal_column is not None and self.value_column is not None


class SchemaMapper:
    """
    Maps arbitrary input data to PRISM's standard schema.

    Never changes based on domain - all domain knowledge is in config.
    """

    def __init__(self, mapping: Union[ColumnMapping, Dict]):
        """
        Initialize with column mapping.

        Args:
            mapping: ColumnMapping object or dict from YAML config
        """
        if isinstance(mapping, dict):
            self.mapping = ColumnMapping.from_dict(mapping)
        else:
            self.mapping = mapping

    def to_observations(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert input DataFrame to observations schema.

        Args:
            df: Input DataFrame in any format

        Returns:
            DataFrame with columns: entity_id, signal_id, timestamp, value
        """
        if self.mapping.is_wide_format():
            return self._wide_to_long(df)
        elif self.mapping.is_long_format():
            return self._rename_long(df)
        else:
            # Try to auto-detect based on columns
            return self._auto_detect_and_convert(df)

    def _wide_to_long(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert wide format (one column per signal) to long format."""
        # Get entity column
        if self.mapping.entity_column and self.mapping.entity_column in df.columns:
            entity_col = self.mapping.entity_column
        else:
            # Add default entity
            df = df.with_columns(pl.lit(self.mapping.entity_default).alias("_entity"))
            entity_col = "_entity"

        # Get time column
        time_col = self.mapping.time_column
        if time_col not in df.columns:
            # Try common names
            for alias in ['time', 'cycle', 't', 'timestamp', 'obs_date']:
                if alias in df.columns:
                    time_col = alias
                    break

        # Filter signal columns to those that exist
        signal_cols = [c for c in self.mapping.signal_columns if c in df.columns]

        if not signal_cols:
            raise ValueError(f"No signal columns found. Available: {df.columns}")

        # Melt wide to long
        result = df.unpivot(
            index=[entity_col, time_col],
            on=signal_cols,
            variable_name="signal_id",
            value_name="value",
        )

        # Rename to standard schema
        result = result.rename({
            entity_col: "entity_id",
            time_col: "timestamp",
        })

        # Cast to final types
        result = result.select([
            pl.col("entity_id").cast(pl.Utf8),
            pl.col("signal_id").cast(pl.Utf8),
            pl.col("timestamp").cast(pl.Float64),
            pl.col("value").cast(pl.Float64),
        ])

        return result

    def _rename_long(self, df: pl.DataFrame) -> pl.DataFrame:
        """Rename columns in long format data."""
        mapping = self.mapping

        # Build rename dict
        renames = {
            mapping.signal_column: "signal_id",
            mapping.value_column: "value",
            mapping.time_column: "timestamp",
        }

        if mapping.entity_column:
            renames[mapping.entity_column] = "entity_id"

        # Apply renames (only for columns that exist)
        renames = {k: v for k, v in renames.items() if k in df.columns}
        df = df.rename(renames)

        # Add default entity if needed
        if "entity_id" not in df.columns:
            df = df.with_columns(pl.lit(mapping.entity_default).alias("entity_id"))

        # Cast to final types
        df = df.select([
            pl.col("entity_id").cast(pl.Utf8),
            pl.col("signal_id").cast(pl.Utf8),
            pl.col("timestamp").cast(pl.Float64),
            pl.col("value").cast(pl.Float64),
        ])

        return df

    def _auto_detect_and_convert(self, df: pl.DataFrame) -> pl.DataFrame:
        """Auto-detect format and convert."""
        # Check for standard long format columns
        long_cols = {'signal_id', 'value', 'timestamp'}
        if long_cols.issubset(set(df.columns)):
            # Already in correct format, just need entity
            if 'entity_id' not in df.columns:
                df = df.with_columns(pl.lit(self.mapping.entity_default).alias("entity_id"))
            return df.select([
                pl.col("entity_id").cast(pl.Utf8),
                pl.col("signal_id").cast(pl.Utf8),
                pl.col("timestamp").cast(pl.Float64),
                pl.col("value").cast(pl.Float64),
            ])

        # Check for common aliases
        alias_map = {
            'entity_id': ['unit_id', 'engine_id', 'bearing_id', 'run_id', 'id'],
            'signal_id': ['sensor_id', 'indicator_id', 'feature', 'column'],
            'timestamp': ['time', 'cycle', 't', 'obs_date', 'observed_at'],
            'value': ['reading', 'measurement', 'val'],
        }

        renames = {}
        for target, aliases in alias_map.items():
            if target not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        renames[alias] = target
                        break

        if renames:
            df = df.rename(renames)

        # If we still don't have signal_id/value, assume wide format
        if 'signal_id' not in df.columns or 'value' not in df.columns:
            # Assume all numeric columns (except time/entity) are signals
            time_col = 'timestamp' if 'timestamp' in df.columns else None
            entity_col = 'entity_id' if 'entity_id' in df.columns else None

            exclude = {time_col, entity_col} - {None}
            signal_cols = [c for c in df.columns
                          if c not in exclude
                          and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

            if signal_cols:
                self.mapping.signal_columns = signal_cols
                if time_col:
                    self.mapping.time_column = time_col
                return self._wide_to_long(df)

        # Final fallback
        if 'entity_id' not in df.columns:
            df = df.with_columns(pl.lit(self.mapping.entity_default).alias("entity_id"))

        required = ['entity_id', 'signal_id', 'timestamp', 'value']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Could not auto-detect columns. Missing: {missing}")

        return df.select([
            pl.col("entity_id").cast(pl.Utf8),
            pl.col("signal_id").cast(pl.Utf8),
            pl.col("timestamp").cast(pl.Float64),
            pl.col("value").cast(pl.Float64),
        ])
