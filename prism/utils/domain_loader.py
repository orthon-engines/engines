"""
Domain Loader
=============

Config-driven data loader that keeps code domain-agnostic.

All domain-specific knowledge lives in YAML config files:
    - Column mappings (what's the entity? what's the time?)
    - Signal definitions (which columns are sensors?)
    - Data format (wide vs long)

Usage:
    from prism.utils.domain_loader import DomainLoader

    # Load from config
    loader = DomainLoader('config/domains/cmapss.yaml')
    observations = loader.load_parquet('data/raw/train_FD001.parquet')

    # Or from dict
    config = {
        'entity_column': 'unit_id',
        'time_column': 'cycle',
        'signal_columns': ['sensor_1', 'sensor_2', ...],
    }
    loader = DomainLoader(config)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import polars as pl

from .schema_mapper import SchemaMapper, ColumnMapping


@dataclass
class DomainConfig:
    """
    Domain configuration.

    Contains ALL domain-specific knowledge - code never changes.
    """
    # Identity
    name: str = "unknown"
    description: str = ""
    source: str = ""  # e.g., "cmapss", "femto", "tep"

    # Column mapping
    entity_column: Optional[str] = None
    entity_default: str = "unit_1"
    time_column: str = "timestamp"
    signal_columns: Optional[List[str]] = None  # Wide format
    signal_column: Optional[str] = None  # Long format
    value_column: Optional[str] = None  # Long format

    # Signal metadata
    signal_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # e.g., {'sensor_1': {'unit': 'psi', 'description': 'Pressure'}}

    # Labels/targets (optional)
    label_column: Optional[str] = None
    label_mapping: Dict[str, Any] = field(default_factory=dict)

    # Data paths (optional, for fully self-contained configs)
    data_path: Optional[str] = None
    observations_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'DomainConfig':
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DomainConfig':
        """Create from dictionary."""
        return cls(
            name=d.get('name', d.get('domain', 'unknown')),
            description=d.get('description', ''),
            source=d.get('source', ''),
            entity_column=d.get('entity_column'),
            entity_default=d.get('entity_default', 'unit_1'),
            time_column=d.get('time_column', 'timestamp'),
            signal_columns=d.get('signal_columns'),
            signal_column=d.get('signal_column'),
            value_column=d.get('value_column'),
            signal_metadata=d.get('signal_metadata', {}),
            label_column=d.get('label_column'),
            label_mapping=d.get('label_mapping', {}),
            data_path=d.get('data_path'),
            observations_path=d.get('observations_path'),
        )

    def to_column_mapping(self) -> ColumnMapping:
        """Convert to ColumnMapping for SchemaMapper."""
        return ColumnMapping(
            entity_column=self.entity_column,
            entity_default=self.entity_default,
            time_column=self.time_column,
            signal_columns=self.signal_columns,
            signal_column=self.signal_column,
            value_column=self.value_column,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (for YAML serialization)."""
        return {
            'name': self.name,
            'description': self.description,
            'source': self.source,
            'entity_column': self.entity_column,
            'entity_default': self.entity_default,
            'time_column': self.time_column,
            'signal_columns': self.signal_columns,
            'signal_column': self.signal_column,
            'value_column': self.value_column,
            'signal_metadata': self.signal_metadata,
            'label_column': self.label_column,
            'label_mapping': self.label_mapping,
            'data_path': self.data_path,
            'observations_path': self.observations_path,
        }


class DomainLoader:
    """
    Domain-agnostic data loader.

    All domain-specific knowledge is in the config.
    Code NEVER changes for a new domain.
    """

    def __init__(self, config: Union[str, Path, Dict, DomainConfig]):
        """
        Initialize with config.

        Args:
            config: Path to YAML, dict, or DomainConfig
        """
        if isinstance(config, (str, Path)):
            self.config = DomainConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = DomainConfig.from_dict(config)
        else:
            self.config = config

        self._mapper = SchemaMapper(self.config.to_column_mapping())

    def load_parquet(self, path: Union[str, Path]) -> pl.DataFrame:
        """
        Load parquet file and convert to observations schema.

        Args:
            path: Path to parquet file

        Returns:
            DataFrame with standard observations schema
        """
        df = pl.read_parquet(path)
        return self._mapper.to_observations(df)

    def load_csv(
        self,
        path: Union[str, Path],
        separator: str = ',',
        has_header: bool = True,
    ) -> pl.DataFrame:
        """
        Load CSV file and convert to observations schema.

        Args:
            path: Path to CSV file
            separator: Column separator
            has_header: Whether file has header row

        Returns:
            DataFrame with standard observations schema
        """
        df = pl.read_csv(path, separator=separator, has_header=has_header)
        return self._mapper.to_observations(df)

    def load_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert existing DataFrame to observations schema.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with standard observations schema
        """
        return self._mapper.to_observations(df)

    def load_observations(self) -> Optional[pl.DataFrame]:
        """
        Load observations from configured path.

        Returns:
            DataFrame with standard observations schema, or None if no path configured
        """
        path = self.config.observations_path or self.config.data_path
        if not path:
            return None

        path = Path(path)
        if path.suffix == '.parquet':
            return self.load_parquet(path)
        elif path.suffix in ('.csv', '.txt'):
            return self.load_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def get_signal_metadata(self, signal_id: str) -> Dict[str, Any]:
        """Get metadata for a specific signal."""
        return self.config.signal_metadata.get(signal_id, {})

    def get_signals(self) -> List[str]:
        """Get list of signal IDs."""
        if self.config.signal_columns:
            return self.config.signal_columns
        return []

    @property
    def name(self) -> str:
        """Domain name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Domain description."""
        return self.config.description


def create_config_from_data(
    df: pl.DataFrame,
    name: str = "auto",
    entity_column: Optional[str] = None,
    time_column: Optional[str] = None,
) -> DomainConfig:
    """
    Auto-create domain config from DataFrame.

    Useful for quickly setting up a new domain.

    Args:
        df: Sample DataFrame
        name: Domain name
        entity_column: Entity column (auto-detected if None)
        time_column: Time column (auto-detected if None)

    Returns:
        DomainConfig with inferred settings
    """
    columns = df.columns

    # Auto-detect entity column
    if not entity_column:
        entity_aliases = ['unit_id', 'engine_id', 'bearing_id', 'entity_id', 'id', 'run_id']
        for alias in entity_aliases:
            if alias in columns:
                entity_column = alias
                break

    # Auto-detect time column
    if not time_column:
        time_aliases = ['timestamp', 'time', 'cycle', 't', 'obs_date']
        for alias in time_aliases:
            if alias in columns:
                time_column = alias
                break
        if not time_column:
            time_column = 'timestamp'

    # Determine signal columns (numeric columns that aren't entity/time)
    exclude = {entity_column, time_column} - {None}
    numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
    signal_columns = [
        c for c in columns
        if c not in exclude and df[c].dtype in numeric_types
    ]

    return DomainConfig(
        name=name,
        entity_column=entity_column,
        time_column=time_column or 'timestamp',
        signal_columns=signal_columns if signal_columns else None,
    )
