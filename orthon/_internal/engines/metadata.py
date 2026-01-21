"""
Engine metadata contract.

Defines static, descriptive metadata for engines.
No logic, no behavior, no configuration.
"""

from dataclasses import dataclass
from typing import Literal, Set

EngineType = Literal["vector", "geometry", "state"]


@dataclass(frozen=True)
class EngineMetadata:
    """
    Static metadata describing an engine's capabilities.

    Attributes:
        name: Engine identifier (matches engine class name pattern)
        engine_type: 'vector' (single series) or 'geometry' (multi-series)
        description: Brief description of what the engine computes
        domains: Set of applicable domain tags (e.g., 'signal_topology', 'correlation')
        requires_window: Whether the engine needs a time window to operate
        deterministic: Whether outputs are reproducible given same inputs
    """
    name: str
    engine_type: EngineType
    description: str
    domains: Set[str]
    requires_window: bool
    deterministic: bool = True
