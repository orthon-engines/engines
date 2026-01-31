"""
Integration Engine

Combines all metrics into unified health assessment.
Computes composite health score, risk level, and recommendations.

Key insight: Health is the absence of multiple coincident degradation signals.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class IntegrationConfig:
    """Configuration for integration engine."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        'stability': 0.25,
        'predictability': 0.20,
        'physics': 0.25,
        'topology': 0.15,
        'causality': 0.15,
    })
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        'lyapunov_critical': 0.1,
        'determinism_low': 0.5,
        'csd_high': 2.0,
        'efficiency_low': 0.7,
        'balance_error': 10.0,
    })


class IntegrationEngine:
    """
    Master Integration Engine.

    Combines all engine outputs into unified health assessment.

    Inputs (optional parquets):
    - lyapunov.parquet: Stability metrics
    - attractor.parquet: Dimension metrics
    - recurrence.parquet: RQA metrics
    - bifurcation.parquet: CSD metrics
    - energy.parquet: Energy balance
    - constitutive.parquet: Material properties
    - topology.parquet: Topological metrics
    - causality_network.parquet: Causal network metrics

    Outputs:
    - health_score: 0-100 (higher is healthier)
    - risk_level: LOW, MODERATE, HIGH, CRITICAL
    - primary_concern: Top issue
    - recommendation: Action to take
    """

    ENGINE_TYPE = "advanced"

    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()

    def compute(
        self,
        metrics: Dict[str, Dict[str, Any]],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute health assessment from multiple metric sources.

        Parameters
        ----------
        metrics : dict
            Maps metric source name to dict of metric values
            e.g., {'lyapunov': {'lyapunov': 0.1, ...}, ...}
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Health assessment
        """
        concerns = []
        weights = self.config.weights
        thresholds = self.config.thresholds

        result = {'unit_id': unit_id}

        # Helper to get value
        def get_val(source: str, key: str):
            if source not in metrics or metrics[source] is None:
                return None
            return metrics[source].get(key)

        # ----- STABILITY SCORE -----
        stability_score = 0.0

        lyap = get_val('lyapunov', 'lyapunov')
        if lyap is not None and not np.isnan(lyap):
            result['lyapunov'] = float(lyap)
            if lyap > thresholds['lyapunov_critical']:
                stability_score += 0.6
                concerns.append(('Chaotic dynamics', 0.9))
            elif lyap > 0:
                stability_score += 0.3
                concerns.append(('Marginal stability', 0.5))

        eff_dim = get_val('attractor', 'effective_dimension')
        if eff_dim is not None and not np.isnan(eff_dim):
            result['effective_dimension'] = float(eff_dim)
            if eff_dim < 2:
                stability_score += 0.4
                concerns.append(('Dimension collapse', 0.7))

        dim_collapse = get_val('attractor', 'dimension_collapse')
        if dim_collapse:
            stability_score += 0.3
            concerns.append(('Attractor collapse', 0.8))

        result['stability_score'] = min(1.0, stability_score)

        # ----- PREDICTABILITY SCORE -----
        predictability_score = 0.0

        det = get_val('recurrence', 'determinism')
        if det is not None and not np.isnan(det):
            result['determinism'] = float(det)
            if det < thresholds['determinism_low']:
                predictability_score += 0.5
                concerns.append(('Low determinism', 0.6))

        csd = get_val('bifurcation', 'csd_score')
        approaching = get_val('bifurcation', 'approaching_bifurcation')
        if csd is not None and not np.isnan(csd):
            result['csd_score'] = float(csd)
            if csd > thresholds['csd_high']:
                predictability_score += 0.3
                concerns.append(('Critical slowing down', 0.7))
        if approaching:
            predictability_score += 0.4
            concerns.append(('Approaching bifurcation', 0.95))

        result['predictability_score'] = min(1.0, predictability_score)

        # ----- PHYSICS SCORE -----
        physics_score = 0.0

        efficiency = get_val('energy', 'efficiency')
        balance_res = get_val('energy', 'balance_residual_pct')
        if efficiency is not None and not np.isnan(efficiency):
            result['efficiency'] = float(efficiency)
            if efficiency < thresholds['efficiency_low']:
                physics_score += 0.3
                concerns.append(('Low efficiency', 0.5))
        if balance_res is not None and not np.isnan(balance_res):
            result['balance_residual_pct'] = float(balance_res)
            if abs(balance_res) > thresholds['balance_error']:
                physics_score += 0.5
                concerns.append(('Energy imbalance', 0.8))

        const_status = get_val('constitutive', 'status')
        if const_status in ['DEGRADED', 'DRIFTING']:
            physics_score += 0.4
            concerns.append(('Constitutive drift', 0.6))

        result['physics_score'] = min(1.0, physics_score)

        # ----- TOPOLOGY SCORE -----
        topology_score = 0.0

        frag = get_val('topology', 'fragmentation')
        topo_change = get_val('topology', 'topology_change')
        if frag:
            topology_score += 0.5
            concerns.append(('Attractor fragmentation', 0.7))
        if topo_change is not None and not np.isnan(topo_change) and topo_change > 0.5:
            topology_score += 0.3
            concerns.append(('Topology change', 0.5))

        result['topology_score'] = min(1.0, topology_score)

        # ----- CAUSALITY SCORE -----
        causality_score = 0.0

        n_loops = get_val('causality_network', 'n_feedback_loops')
        hierarchy = get_val('causality_network', 'hierarchy')
        if n_loops is not None and n_loops > 3:
            causality_score += 0.3
            concerns.append(('Many feedback loops', 0.4))
        if hierarchy is not None and not np.isnan(hierarchy) and hierarchy < 0.3:
            causality_score += 0.3
            concerns.append(('Low hierarchy', 0.4))

        result['causality_score'] = min(1.0, causality_score)

        # ----- COMPOSITE HEALTH SCORE -----
        weighted_risk = (
            weights['stability'] * result['stability_score'] +
            weights['predictability'] * result['predictability_score'] +
            weights['physics'] * result['physics_score'] +
            weights['topology'] * result['topology_score'] +
            weights['causality'] * result['causality_score']
        )

        # Health = 100 - risk*100
        health_score = max(0, min(100, 100 * (1 - weighted_risk)))
        result['health_score'] = float(health_score)

        # Risk level
        if health_score >= 80:
            risk_level = 'LOW'
        elif health_score >= 60:
            risk_level = 'MODERATE'
        elif health_score >= 40:
            risk_level = 'HIGH'
        else:
            risk_level = 'CRITICAL'
        result['risk_level'] = risk_level

        # Top concerns
        concerns.sort(key=lambda x: -x[1])
        result['primary_concern'] = concerns[0][0] if len(concerns) > 0 else 'None'
        result['secondary_concern'] = concerns[1][0] if len(concerns) > 1 else 'None'
        result['n_concerns'] = len(concerns)

        # Recommendation
        if risk_level == 'CRITICAL':
            rec = 'IMMEDIATE INSPECTION REQUIRED'
        elif risk_level == 'HIGH':
            rec = 'Schedule maintenance within 1 week'
        elif risk_level == 'MODERATE':
            rec = 'Monitor closely, plan maintenance'
        else:
            rec = 'Normal operation'
        result['recommendation'] = rec

        return result

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result when no data available."""
        return {
            'unit_id': unit_id,
            'stability_score': np.nan,
            'predictability_score': np.nan,
            'physics_score': np.nan,
            'topology_score': np.nan,
            'causality_score': np.nan,
            'health_score': np.nan,
            'risk_level': 'UNKNOWN',
            'primary_concern': 'Insufficient data',
            'secondary_concern': 'None',
            'n_concerns': 0,
            'recommendation': 'Collect more data',
        }


def run_integration_engine(
    parquet_paths: Dict[str, Path],
    config: IntegrationConfig,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Run integration engine on multiple parquet sources.

    Parameters
    ----------
    parquet_paths : dict
        Maps source name to parquet path:
        'lyapunov', 'attractor', 'recurrence', 'bifurcation',
        'energy', 'constitutive', 'topology', 'causality_network'
    config : IntegrationConfig
        Engine configuration
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        Health assessment for all entities
    """
    engine = IntegrationEngine(config)

    # Load all available data
    dfs = {}
    for name, path in parquet_paths.items():
        try:
            dfs[name] = pl.read_parquet(path)
        except Exception:
            dfs[name] = None

    # Get all unique entity IDs
    all_entities = set()
    for df in dfs.values():
        if df is not None and 'unit_id' in df.columns:
            all_entities.update(df['unit_id'].unique().to_list())

    results = []

    for unit_id in sorted(all_entities):
        # Gather metrics for this entity from all sources
        metrics = {}

        for source_name, df in dfs.items():
            if df is None or 'unit_id' not in df.columns:
                continue

            entity_rows = df.filter(pl.col('unit_id') == unit_id)
            if len(entity_rows) == 0:
                continue

            # Take first row (or could aggregate)
            row = entity_rows.to_dicts()[0]
            metrics[source_name] = row

        result = engine.compute(metrics, unit_id)
        results.append(result)

    df = pl.DataFrame(results) if results else pl.DataFrame({
        'unit_id': [], 'health_score': [], 'risk_level': [], 'recommendation': []
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)

    return df
