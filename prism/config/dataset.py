"""
PRISM Dataset Configuration
===========================

Axis-agnostic configuration for datasets.

PRISM's math works on any ordered dimension:
- Engine cycles (time)
- Sediment depth (space)
- Distance along pipe (space)
- Frame number (sequence)
- Base pair position (genomics)

PRISM doesn't care what the axis is called. It cares about order and change.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class DatasetConfig:
    """Dataset-specific configuration for axis-agnostic computation."""

    # What is the entity?
    entity_column: str = "entity_id"

    # What is the ordered dimension? (time, depth, distance, cycle, etc.)
    ordered_dimension: str = "timestamp"

    # What are the signals? (None = auto-detect numeric columns)
    signal_columns: Optional[List[str]] = None

    # Optional: domain name for interpretation
    domain: str = "generic"

    # Sample thresholds
    min_samples: int = 50
    min_samples_geometry: int = 30
    min_samples_dynamics: int = 100

    # Windowing (None = full signal)
    window_size: Optional[int] = None
    stride: Optional[int] = None

    # Engine selection (None = all enabled)
    engines: Optional[Dict[str, Dict[str, bool]]] = None

    # Domain metadata
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "DatasetConfig":
        """Load config from YAML file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Handle source mappings (electrochemistry style)
        source = data.get('source', {})

        return cls(
            entity_column=source.get('entity_col', data.get('entity_col', 'entity_id')),
            ordered_dimension=source.get('timestamp_col', data.get('time_col', 'timestamp')),
            signal_columns=source.get('signals', data.get('signal_columns')),
            domain=data.get('domain', 'generic'),
            min_samples=data.get('min_samples', 50),
            min_samples_geometry=data.get('min_samples_geometry', 30),
            min_samples_dynamics=data.get('min_samples_dynamics', 100),
            window_size=data.get('window_size'),
            stride=data.get('stride'),
            engines=data.get('engines'),
            metadata=data.get('metadata'),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'entity_column': self.entity_column,
            'ordered_dimension': self.ordered_dimension,
            'signal_columns': self.signal_columns,
            'domain': self.domain,
            'min_samples': self.min_samples,
            'min_samples_geometry': self.min_samples_geometry,
            'min_samples_dynamics': self.min_samples_dynamics,
            'window_size': self.window_size,
            'stride': self.stride,
            'engines': self.engines,
            'metadata': self.metadata,
        }


# Internal column names (used by all engines)
INTERNAL_ENTITY = '_entity'
INTERNAL_AXIS = '_axis'
INTERNAL_SIGNAL = '_signal'
INTERNAL_VALUE = '_value'


def normalize_columns(df, config: DatasetConfig):
    """
    Rename user columns to internal standard names.

    This allows engines to use consistent column names regardless
    of the source data's naming conventions.
    """
    import polars as pl

    renames = {}

    # Entity column
    if config.entity_column in df.columns:
        renames[config.entity_column] = INTERNAL_ENTITY
    elif 'entity_id' in df.columns:
        renames['entity_id'] = INTERNAL_ENTITY

    # Ordered dimension (axis)
    if config.ordered_dimension in df.columns:
        renames[config.ordered_dimension] = INTERNAL_AXIS
    elif 'timestamp' in df.columns:
        renames['timestamp'] = INTERNAL_AXIS

    # Signal column (if in long format)
    if 'signal_id' in df.columns:
        renames['signal_id'] = INTERNAL_SIGNAL

    # Value column (if in long format)
    if 'value' in df.columns:
        renames['value'] = INTERNAL_VALUE

    if renames:
        return df.rename(renames)
    return df


def denormalize_columns(df, config: DatasetConfig):
    """
    Rename internal columns back to user column names for output.
    """
    import polars as pl

    renames = {}

    if INTERNAL_ENTITY in df.columns:
        renames[INTERNAL_ENTITY] = config.entity_column

    if INTERNAL_AXIS in df.columns:
        renames[INTERNAL_AXIS] = config.ordered_dimension

    if INTERNAL_SIGNAL in df.columns:
        renames[INTERNAL_SIGNAL] = 'signal_id'

    if INTERNAL_VALUE in df.columns:
        renames[INTERNAL_VALUE] = 'value'

    if renames:
        return df.rename(renames)
    return df


# Preset configs for common domains
CMAPSS_CONFIG = DatasetConfig(
    entity_column="unit_id",
    ordered_dimension="cycle",
    signal_columns=["T2", "T24", "T30", "P2", "Ps30", "Nf", "Nc", "phi"],
    domain="turbofan",
    min_samples=50,
)

ELECTROCHEM_CONFIG = DatasetConfig(
    entity_column="Station",
    ordered_dimension="Sediment_depth",
    signal_columns=["O2", "Fe_II", "Org_Fe_III", "FeS_aq", "SH2S"],
    domain="electrochemistry",
    min_samples=20,
    min_samples_geometry=10,
)

FEMTO_CONFIG = DatasetConfig(
    entity_column="bearing_id",
    ordered_dimension="timestamp",
    signal_columns=["h_acc", "v_acc"],
    domain="bearing",
    min_samples=100,
)
