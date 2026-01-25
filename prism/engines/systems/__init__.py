"""
PRISM Systems Engines

Fleet-level / cross-entity analysis.

Unlike other engines which operate per-entity, Systems engines
aggregate across ALL entities for each window.

Output is indexed by window only, not by entity.

Engines:
    fleet_status: Aggregate counts/means across entities
    entity_ranking: Rank entities by baseline_distance
    leading_indicator: First entities to show divergence
    correlated_trajectories: Entity pairs with similar trajectories
"""

from .fleet_status import compute as compute_fleet_status
from .entity_ranking import compute as compute_entity_ranking
from .leading_indicator import compute as compute_leading_indicator
from .correlated_trajectories import compute as compute_correlated_trajectories

__all__ = [
    'compute_fleet_status',
    'compute_entity_ranking',
    'compute_leading_indicator',
    'compute_correlated_trajectories',
]
