"""
Baseline Engine

Computes fleet-wide and per-entity baselines for all metrics.
Establishes what "normal" looks like for comparison.

Key insight: You can't detect anomalies without knowing what's normal.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class BaselineConfig:
    """Configuration for baseline engine."""
    baseline_windows: int = 10  # Number of initial windows for baseline
    percentiles: List[int] = field(default_factory=lambda: [5, 25, 50, 75, 95])
    group_by: str = 'fleet'  # 'fleet' | 'entity'
    min_samples: int = 3


class BaselineEngine:
    """
    Baseline Computation Engine.

    Computes statistical baselines for all numeric metrics.

    Outputs:
    - metric_source: Which engine produced the metric
    - metric_name: Name of the metric
    - unit_id: Entity or 'FLEET' for fleet-wide
    - mean, std, median, min, max: Basic statistics
    - p5, p25, p50, p75, p95: Percentiles
    - n_samples: Number of samples in baseline
    """

    ENGINE_TYPE = "statistics"

    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()

    def compute_baseline(
        self,
        data: np.ndarray,
        metric_name: str,
        source_name: str,
        unit_id: str = 'FLEET'
    ) -> Dict[str, Any]:
        """
        Compute baseline statistics for a metric.

        Parameters
        ----------
        data : np.ndarray
            Array of metric values
        metric_name : str
            Name of the metric
        source_name : str
            Name of the source engine
        unit_id : str
            Entity ID or 'FLEET'

        Returns
        -------
        dict
            Baseline statistics
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]

        if len(data) < self.config.min_samples:
            return self._empty_result(metric_name, source_name, unit_id)

        stats = {
            'metric_source': source_name,
            'metric_name': metric_name,
            'unit_id': unit_id,
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'n_samples': len(data),
        }

        # Add percentiles
        for p in self.config.percentiles:
            stats[f'p{p}'] = float(np.percentile(data, p))

        # Coefficient of variation
        if stats['mean'] != 0:
            stats['cv'] = float(stats['std'] / abs(stats['mean']))
        else:
            stats['cv'] = np.nan

        # Interquartile range
        stats['iqr'] = stats['p75'] - stats['p25']

        return stats

    def _empty_result(
        self,
        metric_name: str,
        source_name: str,
        unit_id: str
    ) -> Dict[str, Any]:
        """Return empty result for insufficient data."""
        result = {
            'metric_source': source_name,
            'metric_name': metric_name,
            'unit_id': unit_id,
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'n_samples': 0,
            'cv': np.nan,
            'iqr': np.nan,
        }
        for p in self.config.percentiles:
            result[f'p{p}'] = np.nan
        return result


def run_baseline_engine(
    parquet_paths: Dict[str, Path],
    config: BaselineConfig,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Compute baseline statistics for all metrics from all engines.

    Parameters
    ----------
    parquet_paths : dict
        Maps engine name to parquet path
    config : BaselineConfig
        Engine configuration
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        Baseline statistics for all metrics
    """
    engine = BaselineEngine(config)
    results = []

    for source_name, path in parquet_paths.items():
        try:
            df = pl.read_parquet(path)
        except Exception:
            continue

        # Filter to baseline windows if window_id exists
        if 'window_id' in df.columns:
            df_baseline = df.filter(pl.col('window_id') < config.baseline_windows)
        else:
            df_baseline = df

        if len(df_baseline) == 0:
            continue

        # Get numeric columns (metrics)
        numeric_cols = [
            c for c in df_baseline.columns
            if df_baseline[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            and c not in ['window_id']
        ]

        for metric in numeric_cols:
            if config.group_by == 'entity' and 'unit_id' in df_baseline.columns:
                # Per-entity baselines
                entities = df_baseline['unit_id'].unique().to_list()
                for unit_id in entities:
                    entity_data = df_baseline.filter(
                        pl.col('unit_id') == unit_id
                    )[metric].drop_nulls().to_numpy()

                    if len(entity_data) >= config.min_samples:
                        stats = engine.compute_baseline(
                            entity_data, metric, source_name, unit_id
                        )
                        results.append(stats)

            # Always compute fleet-wide baseline
            fleet_data = df_baseline[metric].drop_nulls().to_numpy()
            if len(fleet_data) >= config.min_samples:
                stats = engine.compute_baseline(
                    fleet_data, metric, source_name, 'FLEET'
                )
                results.append(stats)

    df_out = pl.DataFrame(results) if results else pl.DataFrame({
        'metric_source': [], 'metric_name': [], 'unit_id': [],
        'mean': [], 'std': [], 'n_samples': []
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.write_parquet(output_path)

    return df_out


def get_baseline_for_metric(
    baseline_df: pl.DataFrame,
    source: str,
    metric: str,
    unit_id: str = 'FLEET'
) -> Optional[Dict[str, Any]]:
    """
    Retrieve baseline statistics for a specific metric.

    Parameters
    ----------
    baseline_df : pl.DataFrame
        Baseline DataFrame
    source : str
        Source engine name
    metric : str
        Metric name
    unit_id : str
        Entity ID or 'FLEET'

    Returns
    -------
    dict or None
        Baseline statistics or None if not found
    """
    # Try entity-specific first
    filtered = baseline_df.filter(
        (pl.col('metric_source') == source) &
        (pl.col('metric_name') == metric) &
        (pl.col('unit_id') == unit_id)
    )

    if len(filtered) == 0 and unit_id != 'FLEET':
        # Fall back to fleet baseline
        filtered = baseline_df.filter(
            (pl.col('metric_source') == source) &
            (pl.col('metric_name') == metric) &
            (pl.col('unit_id') == 'FLEET')
        )

    if len(filtered) == 0:
        return None

    return filtered.to_dicts()[0]
