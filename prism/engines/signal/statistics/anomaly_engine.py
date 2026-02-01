"""
Anomaly Engine

Computes anomaly scores by comparing current values to baselines.
Uses z-scores, percentile rankings, and multi-metric fusion.

Key insight: Anomaly = significant deviation from established baseline.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .baseline_engine import get_baseline_for_metric


@dataclass
class AnomalyConfig:
    """Configuration for anomaly engine."""
    z_threshold: float = 2.0  # Z-score threshold for anomaly
    critical_threshold: float = 3.0  # Z-score for critical anomaly
    elevated_threshold: float = 1.5  # Z-score for elevated status


class AnomalyEngine:
    """
    Anomaly Detection Engine.

    Computes anomaly scores by comparing to baselines.

    Outputs:
    - unit_id, window_id: Location
    - metric_source, metric_name: Which metric
    - value: Current value
    - baseline_mean, baseline_std: Baseline statistics
    - z_score: Standard deviations from mean (computed, not classified)
    - percentile_rank: Position in baseline distribution

    NOTE: PRISM computes only. No is_anomaly or anomaly_severity classification.
    ORTHON interprets z_score to determine anomaly status.
    """

    ENGINE_TYPE = "statistics"

    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()

    def score_value(
        self,
        value: float,
        baseline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score a single value against baseline.

        Parameters
        ----------
        value : float
            Current metric value
        baseline : dict
            Baseline statistics (mean, std, percentiles)

        Returns
        -------
        dict
            Anomaly score details
        """
        # Return computed values only - no classification
        if baseline is None or np.isnan(value):
            return {
                'z_score': np.nan,
                'percentile_rank': np.nan,
            }

        mean = baseline.get('mean', 0)
        std = baseline.get('std', 1)

        if std == 0 or np.isnan(std):
            std = 1e-10

        # Z-score (computed value - ORTHON interprets)
        z = (value - mean) / std

        # Percentile rank (computed value)
        p5 = baseline.get('p5', mean - 2 * std)
        p95 = baseline.get('p95', mean + 2 * std)

        if p95 > p5:
            pct_rank = (value - p5) / (p95 - p5) * 100
            pct_rank = np.clip(pct_rank, 0, 100)
        else:
            pct_rank = 50.0

        return {
            'baseline_mean': float(mean),
            'baseline_std': float(std),
            'z_score': float(z),
            'percentile_rank': float(pct_rank),
        }


def run_anomaly_engine(
    parquet_paths: Dict[str, Path],
    baseline_path: Path,
    config: AnomalyConfig,
    output_path: Optional[Path] = None
) -> pl.DataFrame:
    """
    Compute anomaly scores for all metrics.

    Parameters
    ----------
    parquet_paths : dict
        Maps engine name to parquet path
    baseline_path : Path
        Path to baseline parquet
    config : AnomalyConfig
        Engine configuration
    output_path : Path, optional
        Path to write output parquet

    Returns
    -------
    pl.DataFrame
        Anomaly scores for all metrics
    """
    engine = AnomalyEngine(config)

    # Load baseline
    try:
        baseline_df = pl.read_parquet(baseline_path)
    except Exception:
        baseline_df = pl.DataFrame()

    results = []

    for source_name, path in parquet_paths.items():
        try:
            df = pl.read_parquet(path)
        except Exception:
            continue

        if 'unit_id' not in df.columns:
            continue

        # Get numeric columns
        numeric_cols = [
            c for c in df.columns
            if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            and c not in ['window_id']
        ]

        for row in df.iter_rows(named=True):
            unit_id = row.get('unit_id')
            window_id = row.get('window_id', 0)
            timestamp = row.get('timestamp_start')

            for metric in numeric_cols:
                value = row.get(metric)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue

                # Get baseline (try entity-specific, fall back to fleet)
                baseline = get_baseline_for_metric(
                    baseline_df, source_name, metric, unit_id
                )

                if baseline is None:
                    baseline = get_baseline_for_metric(
                        baseline_df, source_name, metric, 'FLEET'
                    )

                # Score the value
                score = engine.score_value(float(value), baseline)

                results.append({
                    'unit_id': unit_id,
                    'window_id': window_id,
                    'timestamp_start': timestamp,
                    'metric_source': source_name,
                    'metric_name': metric,
                    'value': float(value),
                    **score,
                })

    df_out = pl.DataFrame(results) if results else pl.DataFrame({
        'unit_id': [], 'window_id': [], 'metric_source': [],
        'metric_name': [], 'value': [], 'z_score': [],
        'z_score': [], 'percentile_rank': []
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.write_parquet(output_path)

    return df_out


def compute_composite_anomaly_score(
    anomaly_df: pl.DataFrame,
    unit_id: str,
    window_id: int = None,
    weights: Dict[str, float] = None
) -> float:
    """
    Compute composite anomaly score for entity/window.

    Combines z-scores across metrics with optional weighting.
    Returns score in [0, 100] where higher = more anomalous.

    Parameters
    ----------
    anomaly_df : pl.DataFrame
        Anomaly DataFrame
    unit_id : str
        Entity ID
    window_id : int, optional
        Window ID (if None, uses all windows)
    weights : dict, optional
        Metric weights

    Returns
    -------
    float
        Composite anomaly score (0-100)
    """
    filtered = anomaly_df.filter(pl.col('unit_id') == unit_id)

    if window_id is not None:
        filtered = filtered.filter(pl.col('window_id') == window_id)

    if len(filtered) == 0:
        return 0.0

    z_scores = filtered['z_score'].drop_nulls().to_numpy()

    if len(z_scores) == 0:
        return 0.0

    if weights:
        metric_names = filtered['metric_name'].to_list()
        w = np.array([weights.get(m, 1.0) for m in metric_names])
        weighted_z = np.abs(z_scores) * w[:len(z_scores)]
        composite = np.sqrt(np.mean(weighted_z ** 2))
    else:
        composite = np.sqrt(np.mean(z_scores ** 2))

    # Convert to 0-100 scale using CDF
    # z=2 → ~97.7, z=3 → ~99.9
    from scipy.stats import norm
    score = norm.cdf(composite) * 100

    return float(score)
