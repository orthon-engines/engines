"""
Orthon Feature Engineering Module
=================================

Heavy-duty feature engineering for time series and industrial data.

Components:
    - RollingFeatureEngine: Compute rolling window statistics
    - ClusterNormalizer: Normalize by cluster/regime
    - FeatureEngineeringPipeline: Combined pipeline

Quick Start:
    from orthon.features import (
        RollingFeatureEngine,
        ClusterNormalizer,
        FeatureEngineeringPipeline,
    )

    # Full pipeline
    pipeline = FeatureEngineeringPipeline(
        n_clusters=6,
        windows=[10, 20, 30],
        op_cols=['op_1', 'op_2'],
        signal_cols=['s11', 's12'],
    )
    train_feat = pipeline.fit_transform(train_df, entity_col='unit', time_col='cycle')
    test_feat = pipeline.transform(test_df)
"""

from orthon.features.rolling_features import (
    # Configuration
    RollingConfig,

    # Main engines
    RollingFeatureEngine,
    ClusterNormalizer,
    FeatureEngineeringPipeline,

    # Data classes
    ClusterBaseline,

    # Convenience functions
    compute_all_rolling_features,
    compute_cluster_normalized_features,
)

__all__ = [
    'RollingConfig',
    'RollingFeatureEngine',
    'ClusterNormalizer',
    'FeatureEngineeringPipeline',
    'ClusterBaseline',
    'compute_all_rolling_features',
    'compute_cluster_normalized_features',
]
