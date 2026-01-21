"""
PRISM Comparison Utility

Canonical comparison logic for vector & geometry audit reconciliation.

This utility:
    - Centralizes tolerance thresholds
    - Prevents divergence between vector and geometry logic
    - Makes audit behavior explicit and reviewable
    - Avoids embedding policy in scripts or engines

Tolerance Policy:
    PASS_TOLERANCE (2%): Values within this are considered a MATCH
    WARN_TOLERANCE (5%): Values within this trigger WARNING_DRIFT
    Beyond WARN_TOLERANCE: ERROR_DRIFT
"""

from dataclasses import dataclass
from enum import Enum


class ComparisonStatus(str, Enum):
    MATCH = "MATCH"
    WARNING_DRIFT = "WARNING_DRIFT"
    ERROR_DRIFT = "ERROR_DRIFT"


@dataclass(frozen=True)
class ComparisonResult:
    status: ComparisonStatus
    relative_diff: float
    old_value: float
    new_value: float


# =============================================================================
# Tolerance Policy
# =============================================================================

PASS_TOLERANCE = 0.02   # 2%
WARN_TOLERANCE = 0.05   # 5%


def relative_diff(old: float, new: float) -> float:
    """
    Compute relative difference between two values.

    Rules:
    - old == 0 and new == 0 => 0.0
    - zero-crossing => infinite difference
    """
    if old == 0.0 and new == 0.0:
        return 0.0
    if old == 0.0 or new == 0.0:
        return float("inf")
    return abs(new - old) / abs(old)


def compare_values(old: float, new: float) -> ComparisonResult:
    """
    Compare two numeric values using PRISM tolerance rules.
    """
    diff = relative_diff(old, new)

    if diff <= PASS_TOLERANCE:
        status = ComparisonStatus.MATCH
    elif diff <= WARN_TOLERANCE:
        status = ComparisonStatus.WARNING_DRIFT
    else:
        status = ComparisonStatus.ERROR_DRIFT

    return ComparisonResult(
        status=status,
        relative_diff=diff,
        old_value=old,
        new_value=new,
    )
