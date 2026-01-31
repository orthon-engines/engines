"""
Constitutive Engine

Detects and tracks constitutive relationships (linear laws).
Monitors coefficient drift for degradation detection.

Examples:
- Ohm's Law: V = IR (resistance change = corrosion)
- Hooke's Law: F = kx (stiffness change = fatigue)
- Darcy's Law: dP = kQ^2 (flow coefficient change = fouling)
- Heat Transfer: Q = hA*dT (heat transfer degradation)

Constitutive drift = degradation.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from ..primitives.pairwise import linear_regression
from ..primitives.tests import mann_kendall


@dataclass
class RelationshipDefinition:
    """Definition of a constitutive relationship."""
    name: str
    type: str = 'linear'  # 'linear' | 'quadratic' | 'ratio'
    independent: str = ''  # X signal
    dependent: str = ''  # Y signal
    expected_coefficient: Optional[float] = None
    tolerance_pct: float = 20.0  # % drift to flag as degraded


@dataclass
class ConstitutiveConfig:
    """Configuration for constitutive engine."""
    relationships: List[RelationshipDefinition] = field(default_factory=list)
    auto_detect: bool = False  # Auto-detect linear relationships
    r_squared_threshold: float = 0.7  # Min R^2 to consider valid
    drift_threshold_pct: float = 20.0  # % drift to flag as degraded
    signal_columns: Optional[List[str]] = None


class ConstitutiveEngine:
    """
    Constitutive Relationship Engine.

    Tracks physical law relationships and detects coefficient drift.

    Outputs:
    - relationship_name: Name of the relationship
    - coefficient: Fitted coefficient (slope)
    - intercept: Fitted intercept
    - r_squared: Coefficient of determination
    - coefficient_drift_pct: Drift from expected/baseline
    - trend: Mann-Kendall trend in coefficient over time
    - status: STABLE, DRIFTING, DEGRADED, WEAK_RELATIONSHIP
    """

    ENGINE_TYPE = "physics"

    def __init__(self, config: Optional[ConstitutiveConfig] = None):
        self.config = config or ConstitutiveConfig()
        self.coefficient_history: Dict[Tuple[str, str], List[float]] = {}

    def compute(
        self,
        signals: Dict[str, np.ndarray],
        unit_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Compute constitutive relationships for an entity.

        Parameters
        ----------
        signals : dict
            Dictionary mapping signal_id to numpy array of values
        unit_id : str
            Entity identifier

        Returns
        -------
        dict
            Constitutive relationship metrics
        """
        if not signals:
            return self._empty_result(unit_id)

        min_len = min(len(v) for v in signals.values())
        if min_len < 20:
            return self._empty_result(unit_id)

        # Auto-detect relationships if enabled
        relationships = list(self.config.relationships)
        if self.config.auto_detect and not relationships:
            relationships = self._auto_detect_relationships(signals)

        results_list = []

        for rel_def in relationships:
            if rel_def.independent not in signals or rel_def.dependent not in signals:
                continue

            try:
                x = np.asarray(signals[rel_def.independent])[:min_len]
                y = np.asarray(signals[rel_def.dependent])[:min_len]

                # Remove NaN
                valid = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[valid]
                y_clean = y[valid]

                if len(x_clean) < 10:
                    continue

                # Fit relationship
                if rel_def.type == 'linear':
                    slope, intercept, r2, p_value = linear_regression(x_clean, y_clean)
                elif rel_def.type == 'quadratic':
                    slope, intercept, r2, p_value = linear_regression(x_clean ** 2, y_clean)
                elif rel_def.type == 'ratio':
                    # y = k*x, so k = mean(y/x)
                    ratios = y_clean / (x_clean + 1e-10)
                    slope = float(np.nanmean(ratios))
                    intercept = 0.0
                    # R^2 approximation
                    y_pred = slope * x_clean
                    ss_res = np.sum((y_clean - y_pred) ** 2)
                    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                    r2 = 1 - ss_res / (ss_tot + 1e-10)
                    p_value = np.nan
                else:
                    continue

                # Track history for drift detection
                history_key = (unit_id, rel_def.name)
                if history_key not in self.coefficient_history:
                    self.coefficient_history[history_key] = []
                self.coefficient_history[history_key].append(slope)

                # Compute drift
                if rel_def.expected_coefficient is not None:
                    baseline = rel_def.expected_coefficient
                elif len(self.coefficient_history[history_key]) > 1:
                    baseline = self.coefficient_history[history_key][0]
                else:
                    baseline = slope

                coef_drift_pct = (slope - baseline) / (abs(baseline) + 1e-10) * 100

                # Trend test
                history = self.coefficient_history[history_key]
                if len(history) >= 4:
                    trend, p_trend, tau, trend_slope = mann_kendall(np.array(history))
                else:
                    trend = 'no trend'
                    p_trend = 1.0

                # Status determination
                if r2 < self.config.r_squared_threshold:
                    status = 'WEAK_RELATIONSHIP'
                elif abs(coef_drift_pct) > self.config.drift_threshold_pct:
                    status = 'DEGRADED'
                elif trend in ['increasing', 'decreasing'] and p_trend < 0.05:
                    status = 'DRIFTING'
                else:
                    status = 'STABLE'

                results_list.append({
                    'relationship_name': rel_def.name,
                    'independent_signal': rel_def.independent,
                    'dependent_signal': rel_def.dependent,
                    'relationship_type': rel_def.type,
                    'coefficient': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r2),
                    'expected_coefficient': rel_def.expected_coefficient,
                    'coefficient_drift_pct': float(coef_drift_pct),
                    'trend': trend,
                    'status': status,
                })

            except Exception:
                continue

        return {
            'unit_id': unit_id,
            'n_samples': min_len,
            'n_relationships': len(results_list),
            'relationships': results_list,
        }

    def _auto_detect_relationships(
        self,
        signals: Dict[str, np.ndarray]
    ) -> List[RelationshipDefinition]:
        """Auto-detect linear relationships between signals."""
        signal_names = list(signals.keys())
        relationships = []

        for i, x_name in enumerate(signal_names):
            for j, y_name in enumerate(signal_names):
                if i >= j:
                    continue

                try:
                    x = np.asarray(signals[x_name])
                    y = np.asarray(signals[y_name])

                    valid = ~(np.isnan(x) | np.isnan(y))
                    x_clean = x[valid]
                    y_clean = y[valid]

                    if len(x_clean) < 20:
                        continue

                    slope, intercept, r2, p_value = linear_regression(x_clean, y_clean)

                    if r2 >= self.config.r_squared_threshold:
                        relationships.append(RelationshipDefinition(
                            name=f'{x_name}_to_{y_name}',
                            type='linear',
                            independent=x_name,
                            dependent=y_name,
                        ))
                except Exception:
                    continue

        return relationships

    def _empty_result(self, unit_id: str) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        return {
            'unit_id': unit_id,
            'n_samples': 0,
            'n_relationships': 0,
            'relationships': [],
        }

    def to_parquet_rows(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert result to list of rows (one per relationship)."""
        rows = []
        for rel in result.get('relationships', []):
            row = {
                'unit_id': result['unit_id'],
                'n_samples': result['n_samples'],
                **rel,
            }
            rows.append(row)
        return rows


def run_constitutive_engine(
    observations: pl.DataFrame,
    config: ConstitutiveConfig,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Run constitutive engine on observations DataFrame.

    Parameters
    ----------
    observations : pl.DataFrame
        Observations with unit_id, signal_id, index, value
    config : ConstitutiveConfig
        Engine configuration
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        Constitutive relationship results
    """
    engine = ConstitutiveEngine(config)

    entities = observations.select('unit_id').unique().to_series().to_list()
    all_rows = []

    for unit_id in entities:
        entity_obs = observations.filter(pl.col('unit_id') == unit_id)

        signals = {}
        for sig_id in entity_obs.select('signal_id').unique().to_series().to_list():
            sig_data = (
                entity_obs
                .filter(pl.col('signal_id') == sig_id)
                .sort('index')
                .select('value')
                .to_series()
                .to_numpy()
            )
            signals[sig_id] = sig_data

        result = engine.compute(signals, unit_id)
        rows = engine.to_parquet_rows(result)
        all_rows.extend(rows)

    if all_rows:
        df = pl.DataFrame(all_rows)
    else:
        df = pl.DataFrame({
            'unit_id': [],
            'relationship_name': [],
            'coefficient': [],
            'r_squared': [],
            'status': [],
        })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)

    return df
