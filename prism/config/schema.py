"""
PRISM Config Schema — Shared between ORTHON and PRISM

ORTHON writes config.json, PRISM reads it.
This file should be identical in both repos.

Usage (ORTHON - writing):
    config = PrismConfig(
        sequence_column="timestamp",
        entities=["P-101", "P-102"],
        discipline="reaction",  # Optional, from dropdown
        ...
    )
    config.to_json("config.json")

Usage (PRISM - reading):
    config = PrismConfig.from_json("config.json")
    disc = config.get_effective_discipline()
    if disc:
        # Route to discipline-specific engines
    print(config.global_constants)

Note: Use 'discipline' field for discipline-specific engines.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
from pathlib import Path
import json


# =============================================================================
# DISCIPLINES
# =============================================================================

# Import the authoritative discipline registry
from prism.disciplines.registry import DISCIPLINES, list_disciplines

# Discipline type - all valid discipline names
DisciplineType = Optional[str]  # Validated against DISCIPLINES keys at runtime


# =============================================================================
# INDEX TYPES (ORTHON Canonical Spec v1.0.0)
# =============================================================================

IndexType = Literal["temporal", "spatial", "ordinal", "spectral", "continuous"]
MonotonicType = Literal["increasing", "decreasing", "none"]
BoundaryType = Literal["truncate", "pad", "error"]


class IndexConfig(BaseModel):
    """
    Index configuration - the foundational axis for all ORTHON/PRISM analysis.

    The index defines:
    - The independent variable (x-axis) for all signals
    - The unit of measurement for windows and stride
    - The join key across all analytical layers

    ORTHON Canonical Spec v1.0.0
    """
    # REQUIRED FIELDS
    column: str = Field(..., description="Column name in source data")
    type: IndexType = Field(..., description="Semantic type: temporal|spatial|ordinal|spectral|continuous")
    unit: str = Field(..., description="Physical or logical unit (e.g., 'seconds', 'meters', 'cycles')")

    # OPTIONAL FIELDS
    monotonic: MonotonicType = Field(default="increasing", description="Index direction")
    nullable: bool = Field(default=False, description="Allow null/NaN index values")
    display_name: Optional[str] = Field(default=None, description="Human-readable name")
    display_format: Optional[str] = Field(default=None, description="Format string for display")

    # VALIDATION
    min_value: Optional[float] = Field(default=None, description="Minimum valid index value")
    max_value: Optional[float] = Field(default=None, description="Maximum valid index value")


class WindowConfig(BaseModel):
    """
    Window/stride configuration for PRISM computation.

    CRITICAL: size and stride are in INDEX UNITS, not row counts.

    Window k contains all rows where:
        index_start + k*stride ≤ index_value < index_start + k*stride + size

    ORTHON Canonical Spec v1.0.0
    """
    size: float = Field(..., description="Window width in INDEX UNITS (not row count)")
    stride: float = Field(..., description="Step between windows in INDEX UNITS (not row count)")
    min_observations: int = Field(default=10, description="Minimum rows required per window")
    boundary: BoundaryType = Field(default="truncate", description="How to handle incomplete windows")

    # Optional metadata from ORTHON auto-detection
    auto_detected: Optional[bool] = Field(
        default=None,
        description="True if ORTHON auto-detected these values"
    )
    detection_method: Optional[str] = Field(
        default=None,
        description="Method used for auto-detection (e.g., 'sample_rate', 'domain_default', 'manual')"
    )


class SignalInfo(BaseModel):
    """Metadata for a single signal"""
    column: str = Field(..., description="Original column name in source data")
    signal_id: str = Field(..., description="Normalized signal identifier")
    unit: Optional[str] = Field(None, description="Unit string (e.g., 'psi', 'gpm', 'degF')")


class PrismConfig(BaseModel):
    """
    Configuration contract between ORTHON and PRISM.

    ORTHON produces this from user data.
    PRISM consumes this to run analysis.
    """

    # ==========================================================================
    # METADATA
    # ==========================================================================

    source_file: str = Field(
        default="",
        description="Original source file path"
    )
    created_at: str = Field(
        default="",
        description="ISO timestamp when config was created"
    )
    orthon_version: str = Field(
        default="0.1.0",
        description="ORTHON version that created this config"
    )

    # ==========================================================================
    # DISCIPLINE (OPTIONAL)
    # ==========================================================================

    discipline: DisciplineType = Field(
        default=None,
        description="Discipline for specialized engines. None = general/core engines only."
    )

    # ==========================================================================
    # SEQUENCE (X-AXIS)
    # ==========================================================================

    sequence_column: Optional[str] = Field(
        default=None,
        description="Column used as x-axis (time, depth, cycle, etc.). None = row index."
    )
    sequence_unit: Optional[str] = Field(
        default=None,
        description="Unit of sequence column (e.g., 's', 'm', 'ft', 'cycle')"
    )
    sequence_name: str = Field(
        default="index",
        description="Semantic name: 'time', 'depth', 'cycle', 'distance', or 'index'"
    )

    # ==========================================================================
    # ENTITIES
    # ==========================================================================

    entity_column: Optional[str] = Field(
        default=None,
        description="Column used for entity grouping. None = single entity."
    )
    entities: List[str] = Field(
        default=["default"],
        description="List of unique entity identifiers"
    )

    # ==========================================================================
    # CONSTANTS
    # ==========================================================================

    global_constants: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constants that apply to all entities (e.g., fluid_density)"
    )
    per_entity_constants: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Constants that vary by entity (e.g., pipe diameter)"
    )

    # ==========================================================================
    # SIGNALS
    # ==========================================================================

    signals: List[SignalInfo] = Field(
        default_factory=list,
        description="List of signals detected in data"
    )

    # ==========================================================================
    # WINDOW CONFIG (REQUIRED)
    # ==========================================================================

    window: WindowConfig = Field(
        ...,  # Required - no default
        description="Window/stride configuration. REQUIRED - PRISM will fail without this."
    )

    # ==========================================================================
    # STATS
    # ==========================================================================

    row_count: int = Field(
        default=0,
        description="Number of rows in source data"
    )
    observation_count: int = Field(
        default=0,
        description="Number of observations in observations.parquet"
    )

    # ==========================================================================
    # METHODS
    # ==========================================================================

    def to_json(self, path: Union[str, Path]) -> None:
        """Write config to JSON file"""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "PrismConfig":
        """Load config from JSON file"""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.model_validate(data)

    def get_constant(self, name: str, entity: Optional[str] = None) -> Optional[Any]:
        """
        Get a constant value, checking per-entity first, then global.

        Args:
            name: Constant name
            entity: Entity ID (optional, for per-entity lookup)

        Returns:
            Constant value or None
        """
        # Check per-entity first
        if entity and entity in self.per_entity_constants:
            if name in self.per_entity_constants[entity]:
                return self.per_entity_constants[entity][name]

        # Fall back to global
        return self.global_constants.get(name)

    def get_signal_unit(self, signal_id: str) -> Optional[str]:
        """Get unit for a signal by signal_id"""
        for sig in self.signals:
            if sig.signal_id == signal_id:
                return sig.unit
        return None

    def signal_ids(self) -> List[str]:
        """Get list of all signal IDs"""
        return [s.signal_id for s in self.signals]

    def get_effective_discipline(self) -> Optional[str]:
        """Get the effective discipline for specialized engine routing."""
        return self.discipline

    def get_discipline_info(self) -> Optional[Dict[str, Any]]:
        """Get discipline metadata if discipline is specified"""
        disc = self.get_effective_discipline()
        if disc and disc in DISCIPLINES:
            return DISCIPLINES[disc]
        return None

    def get_discipline_engines(self) -> List[str]:
        """Get list of discipline-specific engines to run"""
        info = self.get_discipline_info()
        if not info:
            return []
        engines = list(info.get("engines", []))
        # Also include subdiscipline engines
        for sub in info.get("subdisciplines", {}).values():
            engines.extend(sub.get("engines", []))
        return engines

    def summary(self) -> str:
        """Human-readable summary"""
        disc = self.get_effective_discipline()
        lines = [
            "PrismConfig Summary",
            "=" * 40,
            f"Source: {self.source_file}",
            f"Discipline: {disc or '(general/core only)'}",
            f"Sequence: {self.sequence_column or '(row index)'} [{self.sequence_unit or 'none'}]",
            f"Window: size={self.window.size}, stride={self.window.stride}, min_obs={self.window.min_observations}"
            + (f" (auto-detected via {self.window.detection_method})" if self.window.auto_detected else ""),
            f"Entities: {len(self.entities)} ({', '.join(self.entities[:3])}{'...' if len(self.entities) > 3 else ''})",
            f"Signals: {len(self.signals)}",
        ]

        for sig in self.signals[:5]:
            lines.append(f"  - {sig.signal_id} [{sig.unit or '?'}]")
        if len(self.signals) > 5:
            lines.append(f"  ... and {len(self.signals) - 5} more")

        if self.global_constants:
            lines.append(f"Global constants: {len(self.global_constants)}")
            for k, v in list(self.global_constants.items())[:3]:
                lines.append(f"  - {k}: {v}")

        return "\n".join(lines)
