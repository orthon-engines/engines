"""
PRISM Config Schema — Shared between ORTHON and PRISM

ORTHON writes config.json, PRISM reads it.
This file should be identical in both repos.

Usage (ORTHON - writing):
    config = PrismConfig(
        sequence_column="timestamp",
        entities=["P-101", "P-102"],
        domain="turbomachinery",  # Optional, from dropdown
        ...
    )
    config.to_json("config.json")

Usage (PRISM - reading):
    config = PrismConfig.from_json("config.json")
    if config.domain:
        # Route to domain-specific engines
    print(config.global_constants)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
from pathlib import Path
import json


# =============================================================================
# AVAILABLE DOMAINS
# =============================================================================

# Domain registry - ORTHON uses this for dropdown, PRISM uses for routing
DOMAINS = {
    "turbomachinery": {
        "name": "Turbomachinery",
        "description": "Gas turbines, compressors, turbofans (C-MAPSS, etc.)",
        "engines": ["compressor_efficiency", "turbine_efficiency", "polytropic_efficiency"],
    },
    "fluid": {
        "name": "Fluid Dynamics",
        "description": "CFD validation, PINN solutions, Navier-Stokes",
        "engines": ["vorticity", "divergence", "q_criterion"],
    },
    "battery": {
        "name": "Battery",
        "description": "Li-ion degradation, capacity fade (CALCE, etc.)",
        "engines": ["capacity_fade", "impedance", "soh"],
    },
    "bearing": {
        "name": "Bearings",
        "description": "Rotating machinery bearings (FEMTO, CWRU, etc.)",
        "engines": ["envelope_spectrum", "ball_pass_frequency"],
    },
    "chemical": {
        "name": "Chemical Process",
        "description": "Reaction kinetics, TEP, batch processes",
        "engines": ["reaction_rate", "yield", "selectivity"],
    },
}

DomainType = Optional[Literal[
    "turbomachinery",
    "fluid",
    "battery",
    "bearing",
    "chemical",
]]


class WindowConfig(BaseModel):
    """Window/stride configuration for PRISM computation"""
    size: int = Field(..., description="Window size in observations")
    stride: int = Field(..., description="Stride between windows in observations")
    min_samples: int = Field(default=50, description="Minimum samples required for computation")

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
    # DOMAIN (OPTIONAL)
    # ==========================================================================

    domain: DomainType = Field(
        default=None,
        description="Domain for specialized engines. None = general/core engines only."
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

    def get_domain_info(self) -> Optional[Dict[str, Any]]:
        """Get domain metadata if domain is specified"""
        if self.domain and self.domain in DOMAINS:
            return DOMAINS[self.domain]
        return None

    def get_domain_engines(self) -> List[str]:
        """Get list of domain-specific engines to run"""
        info = self.get_domain_info()
        return info["engines"] if info else []

    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            "PrismConfig Summary",
            "=" * 40,
            f"Source: {self.source_file}",
            f"Domain: {self.domain or '(general)'}",
            f"Sequence: {self.sequence_column or '(row index)'} [{self.sequence_unit or 'none'}]",
            f"Window: size={self.window.size}, stride={self.window.stride}, min_samples={self.window.min_samples}"
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
