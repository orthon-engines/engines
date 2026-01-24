"""
Behavioral Geometry Layer
=========================

Orchestrates geometry engines to answer:
    "How do signals relate to each other?"

Sub-questions:
    - Correlation: Who moves with whom?
    - Clustering: What groups exist?
    - Network: What is the topology?
    - Causality: Who leads whom?
    - Decoupling: Are relationships breaking?

This is a PURE ORCHESTRATOR - no computation here.
All geometry calculations delegated to engines/geometry/.

Output:
    - GeometryVector: Numerical measurements for downstream layers
    - GeometryTypology: Classification for interpretation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class TopologyClass(Enum):
    """Network topology classification"""
    HIGHLY_CONNECTED = "highly_connected"   # Dense, most nodes connected
    MODULAR = "modular"                     # Clear clusters
    HIERARCHICAL = "hierarchical"           # Hub-and-spoke
    FRAGMENTED = "fragmented"               # Disconnected components
    SPARSE = "sparse"                       # Few connections


class StabilityClass(Enum):
    """Relationship stability classification"""
    STABLE = "stable"                       # Relationships holding
    WEAKENING = "weakening"                 # Correlations declining
    RESTRUCTURING = "restructuring"         # Clusters changing
    BREAKING = "breaking"                   # Active decoupling
    BROKEN = "broken"                       # Major relationships lost


class LeadershipClass(Enum):
    """Causal structure classification"""
    CLEAR_LEADER = "clear_leader"           # One dominant signal
    MULTIPLE_LEADERS = "multiple_leaders"   # Several leading signals
    BIDIRECTIONAL = "bidirectional"         # Mutual causality
    CONTEMPORANEOUS = "contemporaneous"     # Move together
    FRAGMENTED = "fragmented"               # No clear structure


# =============================================================================
# OUTPUT DATACLASSES
# =============================================================================

@dataclass
class GeometryVector:
    """
    Numerical measurements from behavioral geometry.
    This is the DATA output - consumed by downstream layers.
    """

    # === IDENTIFICATION ===
    timestamp: datetime = field(default_factory=datetime.now)
    n_signals: int = 0

    # === CORRELATION STRUCTURE ===
    mean_correlation: float = 0.0
    median_correlation: float = 0.0
    correlation_dispersion: float = 0.0
    n_strong_pairs: int = 0
    n_weak_pairs: int = 0

    # Eigenvalue analysis
    variance_explained_1: float = 0.0       # First PC
    effective_dimension: float = 1.0        # How many independent signals

    # === CLUSTERING ===
    n_clusters: int = 1
    silhouette_score: float = 0.0
    mean_internal_correlation: float = 0.0

    # === NETWORK TOPOLOGY ===
    network_density: float = 0.0
    mean_degree: float = 0.0
    transitivity: float = 0.0               # Clustering coefficient
    n_components: int = 1
    n_hubs: int = 0

    # === CAUSALITY ===
    n_causal_pairs: int = 0
    n_bidirectional: int = 0
    mean_lead_lag: float = 0.0              # Average optimal lag

    # === DECOUPLING ===
    n_decoupled_pairs: int = 0
    n_severe_decouplings: int = 0
    decoupling_rate: float = 0.0            # Fraction of pairs decoupled

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result


@dataclass
class GeometryTypology:
    """
    Classification from behavioral geometry.
    This is the INFORMATION output - human-interpretable.
    """

    # === CLASSIFICATIONS ===
    topology_class: TopologyClass = TopologyClass.SPARSE
    stability_class: StabilityClass = StabilityClass.STABLE
    leadership_class: LeadershipClass = LeadershipClass.CONTEMPORANEOUS

    # === CLUSTER INFO ===
    cluster_assignments: np.ndarray = None
    cluster_names: List[str] = None

    # === HUB IDENTIFICATION ===
    hub_indices: List[int] = field(default_factory=list)
    hub_names: List[str] = field(default_factory=list)

    # === LEADER IDENTIFICATION ===
    leader_indices: List[int] = field(default_factory=list)
    leader_names: List[str] = field(default_factory=list)

    # === DECOUPLING ALERTS ===
    decoupling_alerts: List[str] = field(default_factory=list)

    # === HUMAN-READABLE ===
    summary: str = ""
    alerts: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'topology_class': self.topology_class.value,
            'stability_class': self.stability_class.value,
            'leadership_class': self.leadership_class.value,
            'hub_indices': self.hub_indices,
            'hub_names': self.hub_names,
            'leader_indices': self.leader_indices,
            'leader_names': self.leader_names,
            'decoupling_alerts': self.decoupling_alerts,
            'summary': self.summary,
            'alerts': self.alerts,
            'confidence': self.confidence
        }


@dataclass
class BehavioralGeometryOutput:
    """Complete output from Behavioral Geometry layer"""
    vector: GeometryVector
    typology: GeometryTypology

    # Detailed results (optional, for deep inspection)
    correlation_matrix: Optional[np.ndarray] = None
    distance_matrix: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None
    adjacency_matrix: Optional[np.ndarray] = None


# =============================================================================
# LAYER IMPLEMENTATION
# =============================================================================

class BehavioralGeometryLayer:
    """
    Layer: How do signals relate to each other?

    Pure orchestrator. Calls geometry engines, classifies results.
    Contains ZERO computation - only coordination and classification logic.

    Answers:
        - Correlation: Who moves with whom?
        - Clustering: What groups exist?
        - Network: What is the topology?
        - Causality: Who leads whom?
        - Decoupling: Are relationships breaking?
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.correlation_threshold = self.config.get('correlation_threshold', 0.5)
        self.historical_window = self.config.get('historical_window', 100)
        self.recent_window = self.config.get('recent_window', 20)

    def analyze(
        self,
        signals: np.ndarray,
        signal_ids: Optional[List[str]] = None,
        previous: Optional[BehavioralGeometryOutput] = None
    ) -> BehavioralGeometryOutput:
        """
        Analyze relationships between multiple signals.

        Args:
            signals: 2D array (n_signals, n_observations)
            signal_ids: Optional signal identifiers
            previous: Previous output (for change detection)

        Returns:
            BehavioralGeometryOutput with vector and typology
        """
        # Import engines (deferred to avoid circular imports)
        from prism.engines.geometry import (
            bg_correlation as correlation,
            bg_distance as distance,
            bg_clustering as clustering,
            bg_network as network,
            bg_granger as granger,
            bg_lead_lag as lead_lag,
            bg_decoupling as decoupling
        )

        signals = np.asarray(signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_signals = signals.shape[0]

        if signal_ids is None:
            signal_ids = [f"signal_{i}" for i in range(n_signals)]

        # === CALL ALL ENGINES ===
        corr_result = correlation.compute_matrix(signals)
        dist_result = distance.compute_matrix(signals)

        cluster_result = clustering.compute_from_signals(signals)
        cluster_profiles = clustering.get_cluster_profiles(
            signals, cluster_result.labels, signal_ids
        )

        network_result = network.compute(
            corr_result.pearson_matrix,
            threshold=self.correlation_threshold
        )
        node_profiles = network.get_node_profiles(network_result, signal_ids)

        granger_result = granger.compute_matrix(signals)
        lead_lag_result = lead_lag.compute_matrix(signals)

        decoupling_result = decoupling.compute_matrix(
            signals, signal_ids,
            self.historical_window, self.recent_window
        )

        # === BUILD VECTOR ===
        vector = self._build_vector(
            n_signals, corr_result, cluster_result, cluster_profiles,
            network_result, granger_result, lead_lag_result, decoupling_result
        )

        # === CLASSIFY ===
        typology = self._classify(
            signal_ids, corr_result, cluster_result, network_result,
            node_profiles, granger_result, lead_lag_result, decoupling_result
        )

        return BehavioralGeometryOutput(
            vector=vector,
            typology=typology,
            correlation_matrix=corr_result.pearson_matrix,
            distance_matrix=dist_result.correlation_distance_matrix,
            cluster_labels=cluster_result.labels,
            adjacency_matrix=network_result.adjacency_matrix
        )

    def _build_vector(
        self, n_signals, corr, cluster, cluster_profiles,
        network, granger, lead_lag, decoupling
    ) -> GeometryVector:
        """Assemble GeometryVector from engine outputs."""

        # Mean internal correlation across clusters
        if cluster_profiles:
            mean_internal = np.mean([p.mean_internal_correlation for p in cluster_profiles])
        else:
            mean_internal = 0.0

        # Mean lead/lag
        mean_lag = np.mean(np.abs(lead_lag.lag_matrix[lead_lag.lag_matrix != 0])) \
            if np.any(lead_lag.lag_matrix != 0) else 0.0

        # Decoupling rate
        n_possible_pairs = n_signals * (n_signals - 1) / 2
        decoupling_rate = decoupling.n_decoupled_pairs / n_possible_pairs \
            if n_possible_pairs > 0 else 0.0

        return GeometryVector(
            timestamp=datetime.now(),
            n_signals=n_signals,

            # Correlation
            mean_correlation=corr.mean_correlation,
            median_correlation=corr.median_correlation,
            correlation_dispersion=corr.correlation_dispersion,
            n_strong_pairs=corr.n_strong_pairs,
            n_weak_pairs=corr.n_weak_pairs,
            variance_explained_1=corr.variance_explained_1,
            effective_dimension=corr.effective_dimension,

            # Clustering
            n_clusters=cluster.n_clusters,
            silhouette_score=cluster.silhouette_score,
            mean_internal_correlation=float(mean_internal),

            # Network
            network_density=network.density,
            mean_degree=network.mean_degree,
            transitivity=network.transitivity,
            n_components=network.n_components,
            n_hubs=len(network.hub_indices),

            # Causality
            n_causal_pairs=granger.n_causal_pairs,
            n_bidirectional=granger.n_bidirectional,
            mean_lead_lag=float(mean_lag),

            # Decoupling
            n_decoupled_pairs=decoupling.n_decoupled_pairs,
            n_severe_decouplings=decoupling.n_severe,
            decoupling_rate=float(decoupling_rate)
        )

    def _classify(
        self, signal_ids, corr, cluster, network,
        node_profiles, granger, lead_lag, decoupling
    ) -> GeometryTypology:
        """Convert measurements to classification."""

        # Topology class
        if network.density > 0.7:
            topology_class = TopologyClass.HIGHLY_CONNECTED
        elif cluster.n_clusters > 2 and cluster.silhouette_score > 0.3:
            topology_class = TopologyClass.MODULAR
        elif len(network.hub_indices) > 0 and network.density < 0.5:
            topology_class = TopologyClass.HIERARCHICAL
        elif network.n_components > 1:
            topology_class = TopologyClass.FRAGMENTED
        else:
            topology_class = TopologyClass.SPARSE

        # Stability class
        if decoupling.n_severe > 0:
            stability_class = StabilityClass.BREAKING
        elif decoupling.n_decoupled_pairs > len(signal_ids) // 2:
            stability_class = StabilityClass.BROKEN
        elif decoupling.n_decoupled_pairs > 0:
            stability_class = StabilityClass.WEAKENING
        else:
            stability_class = StabilityClass.STABLE

        # Leadership class
        if len(granger.top_causers) > 0 and granger.n_causal_pairs > 0:
            if granger.n_bidirectional > granger.n_causal_pairs // 2:
                leadership_class = LeadershipClass.BIDIRECTIONAL
            elif len(set(granger.top_causers)) == 1:
                leadership_class = LeadershipClass.CLEAR_LEADER
            else:
                leadership_class = LeadershipClass.MULTIPLE_LEADERS
        elif corr.mean_correlation > 0.5:
            leadership_class = LeadershipClass.CONTEMPORANEOUS
        else:
            leadership_class = LeadershipClass.FRAGMENTED

        # Hub names
        hub_names = [signal_ids[i] if i < len(signal_ids) else f"signal_{i}"
                     for i in network.hub_indices]

        # Leader names
        leader_names = [signal_ids[i] if i < len(signal_ids) else f"signal_{i}"
                        for i in granger.top_causers]

        # Generate summary and alerts
        summary, alerts = self._generate_summary(
            topology_class, stability_class, leadership_class,
            network, decoupling, hub_names, leader_names
        )

        # Add decoupling alerts
        alerts.extend(decoupling.alerts)

        # Confidence
        confidence = self._compute_confidence(corr, cluster, network)

        return GeometryTypology(
            topology_class=topology_class,
            stability_class=stability_class,
            leadership_class=leadership_class,
            cluster_assignments=cluster.labels,
            hub_indices=network.hub_indices,
            hub_names=hub_names,
            leader_indices=granger.top_causers,
            leader_names=leader_names,
            decoupling_alerts=decoupling.alerts,
            summary=summary,
            alerts=alerts,
            confidence=confidence
        )

    def _generate_summary(
        self, topology, stability, leadership,
        network, decoupling, hub_names, leader_names
    ) -> tuple:
        """Generate human-readable summary and alerts."""

        alerts = []

        # Stability alerts
        if stability == StabilityClass.BREAKING:
            alerts.append("ACTIVE DECOUPLING: Relationships breaking down")
        elif stability == StabilityClass.WEAKENING:
            alerts.append("Correlation structure weakening")

        # Topology alerts
        if topology == TopologyClass.FRAGMENTED:
            alerts.append("Network fragmented: disconnected components")

        # Leadership
        if leader_names:
            leaders_str = ", ".join(leader_names[:3])
            alerts.append(f"Leaders: {leaders_str}")

        # Hubs
        if hub_names:
            hubs_str = ", ".join(hub_names[:3])
            alerts.append(f"Hubs: {hubs_str}")

        summary = f"**{topology.value.replace('_', ' ').title()}** | {stability.value} | {leadership.value.replace('_', ' ')}"

        return summary, alerts

    def _compute_confidence(self, corr, cluster, network) -> float:
        """Compute overall classification confidence."""

        # Higher confidence when structure is clear
        corr_conf = min(corr.variance_explained_1 * 2, 1.0)
        cluster_conf = cluster.silhouette_score if cluster.silhouette_score > 0 else 0.3
        network_conf = network.transitivity if network.density > 0.1 else 0.3

        confidence = (corr_conf + cluster_conf + network_conf) / 3

        return float(np.clip(confidence, 0, 1))


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_geometry(
    signals: np.ndarray,
    signal_ids: Optional[List[str]] = None
) -> BehavioralGeometryOutput:
    """
    Convenience function for quick geometry analysis.

    Example:
        result = analyze_geometry(my_signals)
        print(result.typology.summary)
        print(result.typology.hub_names)
    """
    layer = BehavioralGeometryLayer()
    return layer.analyze(signals, signal_ids)


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("BEHAVIORAL GEOMETRY LAYER - DEMONSTRATION")
    print("=" * 70)

    n_obs = 200

    # Generate correlated signals
    base = np.cumsum(np.random.randn(n_obs))

    signals = np.array([
        base + np.random.randn(n_obs) * 0.5,           # High correlation with base
        base + np.random.randn(n_obs) * 0.5,           # High correlation with base
        -base + np.random.randn(n_obs) * 0.5,          # Negative correlation
        np.cumsum(np.random.randn(n_obs)),             # Independent
        np.cumsum(np.random.randn(n_obs)),             # Independent
    ])

    signal_ids = ["SPY", "QQQ", "TLT", "GLD", "BTC"]

    layer = BehavioralGeometryLayer()
    result = layer.analyze(signals, signal_ids)

    print(f"\nTopology: {result.typology.topology_class.value}")
    print(f"Stability: {result.typology.stability_class.value}")
    print(f"Leadership: {result.typology.leadership_class.value}")
    print(f"Confidence: {result.typology.confidence:.0%}")
    print()
    print(f"Summary: {result.typology.summary}")
    print()
    print("Metrics:")
    print(f"  Mean correlation: {result.vector.mean_correlation:.3f}")
    print(f"  Network density: {result.vector.network_density:.3f}")
    print(f"  Clusters: {result.vector.n_clusters}")
    print(f"  Effective dimension: {result.vector.effective_dimension:.1f}")
    print()
    if result.typology.alerts:
        print("Alerts:")
        for alert in result.typology.alerts:
            print(f"  {alert}")
