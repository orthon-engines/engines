"""
PRISM State Vector Base

Structural definition only. No computation. No interpretation.

IMMUTABILITY CLARIFICATION:
    The state layer is "conceptually experimental" but "operationally immutable."

    State construction logic may evolve, but:
        - Once a state vector is written to the DB for a given system_record_id,
          it is IMMUTABLE.
        - Changes to state logic result in:
            * A new system run
            * A new system_record_id
            * New state records
        - Never overwrites. Never updates. Never deletes.

    "Experimental" refers to CODE evolution, not DATA mutability.

Key invariants:
    - A state vector cannot exist without a system.
    - All persisted state data is immutable.
    - Changing logic creates new systems, not new truth.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class StateVector:
    """
    System-scoped signal state vector.

    This object describes the position of a single signal
    within the observed geometric field, scoped to a specific system run.

    It contains no interpretation, weighting, or dominance logic.

    Key invariant: Signals do not belong to systems. Signal records do.
    A state vector cannot exist without a system_record_id.

    Attributes:
        system_record_id: The system run this state belongs to (required)
        signal_name: The signal this state describes
        dimensions: Position coordinates (dimension -> value)
        observed_at: Timestamp of the observation window
        metadata: Optional additional context

    Invariants:
        - Cannot exist without a system_record_id
        - IMMUTABLE once persisted to results.state_vectors
        - Describes position, not verdict
        - Derived from persisted data only
        - Reproducible from DB alone
        - No cross-signal comparison
    """

    system_record_id: str  # Required - cannot exist without a system
    signal_name: str
    dimensions: Dict[str, float]
    observed_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
