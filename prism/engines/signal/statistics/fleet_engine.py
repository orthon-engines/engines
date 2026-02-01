"""
Fleet Engine

Computes fleet-wide analytics: entity rankings, clustering,
cohort analysis, and comparative statistics.

Key insight: Individual health only makes sense in fleet context.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class FleetConfig:
    """Configuration for fleet engine."""
    n_clusters: int = 3
    ranking_metric: str = 'health_score'
    health_tiers: Dict[str, float] = None

    def __post_init__(self):
        if self.health_tiers is None:
            self.health_tiers = {
                'HEALTHY': 80,
                'MODERATE': 60,
                'AT_RISK': 40,
                'CRITICAL': 0,
            }


class FleetEngine:
    """
    Fleet Analytics Engine.

    Computes fleet-wide comparisons and rankings.

    Outputs:
    - Rankings: Entity rankings by health and other metrics
    - Clusters: Entity groupings by behavior similarity
    - Cohorts: Health tier groupings
    """

    ENGINE_TYPE = "statistics"

    def __init__(self, config: Optional[FleetConfig] = None):
        self.config = config or FleetConfig()

    def compute_rankings(
        self,
        health_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Compute entity rankings from health data.

        Parameters
        ----------
        health_df : pl.DataFrame
            Health DataFrame with unit_id, health_score, risk_level

        Returns
        -------
        pl.DataFrame
            Entity rankings
        """
        if len(health_df) == 0:
            return pl.DataFrame()

        # Compute per-entity statistics
        entity_stats = health_df.group_by('unit_id').agg([
            pl.col('health_score').mean().alias('avg_health'),
            pl.col('health_score').min().alias('min_health'),
            pl.col('health_score').max().alias('max_health'),
            pl.col('health_score').std().alias('health_volatility'),
            pl.col('health_score').last().alias('latest_health'),
            pl.col('window_id').max().alias('latest_window') if 'window_id' in health_df.columns else pl.lit(0).alias('latest_window'),
        ])

        # Count risk events
        if 'risk_level' in health_df.columns:
            risk_counts = health_df.group_by('unit_id').agg([
                pl.col('risk_level').filter(pl.col('risk_level') == 'CRITICAL').count().alias('critical_events'),
                pl.col('risk_level').filter(pl.col('risk_level') == 'HIGH').count().alias('high_events'),
            ])
            entity_stats = entity_stats.join(risk_counts, on='unit_id', how='left')
        else:
            entity_stats = entity_stats.with_columns([
                pl.lit(0).alias('critical_events'),
                pl.lit(0).alias('high_events'),
            ])

        # Add rankings
        entity_stats = entity_stats.with_columns([
            pl.col('avg_health').rank(descending=False).alias('health_rank'),
            (pl.col('critical_events') + pl.col('high_events')).rank(descending=True).alias('risk_rank'),
        ])

        # Add health tier
        tiers = self.config.health_tiers
        entity_stats = entity_stats.with_columns([
            pl.when(pl.col('avg_health') >= tiers['HEALTHY'])
            .then(pl.lit('HEALTHY'))
            .when(pl.col('avg_health') >= tiers['MODERATE'])
            .then(pl.lit('MODERATE'))
            .when(pl.col('avg_health') >= tiers['AT_RISK'])
            .then(pl.lit('AT_RISK'))
            .otherwise(pl.lit('CRITICAL'))
            .alias('health_tier')
        ])

        return entity_stats

    def compute_clusters(
        self,
        entity_stats: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Cluster entities by health behavior.

        Parameters
        ----------
        entity_stats : pl.DataFrame
            Entity statistics

        Returns
        -------
        pl.DataFrame
            Entity stats with cluster labels
        """
        if len(entity_stats) < self.config.n_clusters:
            return entity_stats.with_columns([
                pl.lit('CLUSTER_1').alias('cluster')
            ])

        # Feature matrix for clustering
        feature_cols = ['avg_health', 'health_volatility', 'critical_events']
        available_cols = [c for c in feature_cols if c in entity_stats.columns]

        if len(available_cols) < 2:
            return entity_stats.with_columns([
                pl.lit('CLUSTER_1').alias('cluster')
            ])

        features = entity_stats.select(available_cols).fill_null(0).to_numpy()

        # Normalize
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-10
        features_norm = (features - mean) / std

        # Simple k-means clustering
        try:
            from scipy.cluster.vq import kmeans2
            centroids, labels = kmeans2(features_norm, self.config.n_clusters, minit='++')

            # Name clusters by average health (1 = best, n = worst)
            cluster_health = []
            for c in range(self.config.n_clusters):
                mask = labels == c
                if np.any(mask):
                    cluster_health.append(np.mean(entity_stats['avg_health'].to_numpy()[mask]))
                else:
                    cluster_health.append(0)

            cluster_order = np.argsort(cluster_health)[::-1]
            cluster_names = {cluster_order[i]: f'CLUSTER_{i+1}' for i in range(self.config.n_clusters)}
            cluster_labels = [cluster_names[l] for l in labels]

        except Exception:
            cluster_labels = ['CLUSTER_1'] * len(entity_stats)

        return entity_stats.with_columns([
            pl.Series('cluster', cluster_labels)
        ])


def run_fleet_engine(
    health_path: Path,
    anomaly_path: Optional[Path],
    config: FleetConfig,
    output_path: Optional[Path] = None
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute fleet-wide analytics.

    Parameters
    ----------
    health_path : Path
        Path to integration engine health output
    anomaly_path : Path, optional
        Path to anomaly engine output
    config : FleetConfig
        Engine configuration
    output_path : Path, optional
        Base path for outputs

    Returns
    -------
    tuple of pl.DataFrame
        (rankings_df, cohorts_df)
    """
    engine = FleetEngine(config)

    # Load health data
    try:
        health_df = pl.read_parquet(health_path)
    except Exception:
        return pl.DataFrame(), pl.DataFrame()

    # Compute rankings
    rankings = engine.compute_rankings(health_df)

    if len(rankings) == 0:
        return pl.DataFrame(), pl.DataFrame()

    # Add clusters
    rankings = engine.compute_clusters(rankings)

    # Add high z-score counts if available (PRISM computes, ORTHON interprets)
    if anomaly_path:
        try:
            anomaly_df = pl.read_parquet(anomaly_path)
            # Count z-scores by magnitude (computed values, not classification)
            anomaly_summary = anomaly_df.group_by('unit_id').agg([
                (pl.col('z_score').abs() > 3.0).sum().alias('high_zscore_count'),
                (pl.col('z_score').abs() > 2.0).sum().alias('elevated_zscore_count'),
                pl.col('z_score').abs().max().alias('max_zscore'),
            ])
            rankings = rankings.join(anomaly_summary, on='unit_id', how='left')
            rankings = rankings.fill_null(0)
        except Exception:
            rankings = rankings.with_columns([
                pl.lit(0).alias('total_anomalies'),
                pl.lit(0).alias('critical_anomalies'),
                pl.lit(0).alias('warning_anomalies'),
            ])

    # Compute cohort statistics
    cohorts = rankings.group_by('health_tier').agg([
        pl.count().alias('n_entities'),
        pl.col('avg_health').mean().alias('cohort_avg_health'),
        pl.col('critical_events').sum().alias('total_critical_events'),
    ])

    # Write outputs
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rankings.write_parquet(output_path.parent / f"{output_path.stem}_rankings.parquet")
        cohorts.write_parquet(output_path.parent / f"{output_path.stem}_cohorts.parquet")

    return rankings, cohorts
