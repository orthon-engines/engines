"""
Heavy Rolling Window Statistics & Cluster Normalization
========================================================

Two powerful feature engineering techniques for time series:

1. ROLLING WINDOW STATISTICS
   - Compute statistical features over sliding windows
   - Captures dynamics: trends, volatility, acceleration

2. CLUSTER NORMALIZATION
   - Normalize by cluster/regime membership
   - Makes features comparable across operating conditions

Usage:
    from orthon.features.rolling_features import (
        RollingFeatureEngine,
        ClusterNormalizer,
        compute_all_rolling_features,
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# ROLLING WINDOW STATISTICS
# =============================================================================

@dataclass
class RollingConfig:
    """Configuration for rolling window features."""

    windows: List[int] = field(default_factory=lambda: [10, 20, 30, 50])

    # Which statistics to compute
    compute_mean: bool = True
    compute_std: bool = True
    compute_min: bool = True
    compute_max: bool = True
    compute_range: bool = True
    compute_slope: bool = True           # Linear trend
    compute_delta: bool = True           # Change from N ago
    compute_curvature: bool = True       # Acceleration (slope of slope)
    compute_skew: bool = True            # Distribution asymmetry
    compute_kurtosis: bool = True        # Tail heaviness
    compute_quantiles: bool = True       # q25, q50, q75
    compute_iqr: bool = True             # Interquartile range
    compute_cv: bool = True              # Coefficient of variation
    compute_zscore: bool = True          # Current value as z-score of window
    compute_momentum: bool = True        # Rate of change
    compute_volatility: bool = True      # Rolling std of returns
    compute_autocorr: bool = True        # Lag-1 autocorrelation
    compute_entropy: bool = False        # Approximate entropy (slow)


class RollingFeatureEngine:
    """
    Compute heavy rolling window statistics for time series.

    This computes ALL useful rolling statistics over multiple windows,
    giving you comprehensive temporal dynamics features.

    Example:
        engine = RollingFeatureEngine(windows=[10, 20, 30])
        features = engine.compute(signal, prefix='sensor1')
    """

    def __init__(
        self,
        windows: List[int] = None,
        config: RollingConfig = None,
    ):
        self.config = config or RollingConfig()
        if windows:
            self.config.windows = windows

    def compute(
        self,
        values: np.ndarray,
        prefix: str = '',
        return_df: bool = False,
    ) -> Union[Dict[str, np.ndarray], pd.DataFrame]:
        """
        Compute all rolling features for a single signal.

        Args:
            values: 1D array of signal values (time-ordered)
            prefix: Prefix for feature names (e.g., 'sensor1_')
            return_df: If True, return DataFrame instead of dict

        Returns:
            Dictionary or DataFrame with rolling features
        """
        n = len(values)
        features = {}

        # Add prefix separator if provided
        if prefix and not prefix.endswith('_'):
            prefix = prefix + '_'

        for W in self.config.windows:
            # Initialize arrays
            feat_mean = np.full(n, np.nan)
            feat_std = np.full(n, np.nan)
            feat_min = np.full(n, np.nan)
            feat_max = np.full(n, np.nan)
            feat_range = np.full(n, np.nan)
            feat_slope = np.full(n, np.nan)
            feat_delta = np.full(n, np.nan)
            feat_curv = np.full(n, np.nan)
            feat_skew = np.full(n, np.nan)
            feat_kurt = np.full(n, np.nan)
            feat_q25 = np.full(n, np.nan)
            feat_q50 = np.full(n, np.nan)
            feat_q75 = np.full(n, np.nan)
            feat_iqr = np.full(n, np.nan)
            feat_cv = np.full(n, np.nan)
            feat_zscore = np.full(n, np.nan)
            feat_momentum = np.full(n, np.nan)
            feat_volatility = np.full(n, np.nan)
            feat_autocorr = np.full(n, np.nan)

            for i in range(W - 1, n):
                window = values[i - W + 1 : i + 1]

                # Basic statistics
                if self.config.compute_mean:
                    feat_mean[i] = np.mean(window)

                if self.config.compute_std:
                    feat_std[i] = np.std(window)

                if self.config.compute_min:
                    feat_min[i] = np.min(window)

                if self.config.compute_max:
                    feat_max[i] = np.max(window)

                if self.config.compute_range:
                    feat_range[i] = np.max(window) - np.min(window)

                # Trend (slope via linear regression)
                if self.config.compute_slope:
                    x = np.arange(W)
                    try:
                        coeffs = np.polyfit(x, window, 1)
                        feat_slope[i] = coeffs[0]
                    except:
                        feat_slope[i] = 0

                # Delta (change from W steps ago)
                if self.config.compute_delta:
                    feat_delta[i] = values[i] - values[i - W + 1]

                # Curvature (acceleration)
                if self.config.compute_curvature and W >= 6:
                    mid = W // 2
                    slope1 = (window[mid] - window[0]) / mid if mid > 0 else 0
                    slope2 = (window[-1] - window[mid]) / (W - mid) if (W - mid) > 0 else 0
                    feat_curv[i] = slope2 - slope1

                # Distribution shape
                if self.config.compute_skew and W >= 8:
                    feat_skew[i] = stats.skew(window)

                if self.config.compute_kurtosis and W >= 8:
                    feat_kurt[i] = stats.kurtosis(window)

                # Quantiles
                if self.config.compute_quantiles:
                    feat_q25[i] = np.percentile(window, 25)
                    feat_q50[i] = np.percentile(window, 50)
                    feat_q75[i] = np.percentile(window, 75)

                if self.config.compute_iqr:
                    feat_iqr[i] = np.percentile(window, 75) - np.percentile(window, 25)

                # Coefficient of variation
                if self.config.compute_cv:
                    mean_val = np.mean(window)
                    if abs(mean_val) > 1e-10:
                        feat_cv[i] = np.std(window) / abs(mean_val)
                    else:
                        feat_cv[i] = 0

                # Z-score of current value within window
                if self.config.compute_zscore:
                    std_val = np.std(window)
                    if std_val > 1e-10:
                        feat_zscore[i] = (values[i] - np.mean(window)) / std_val
                    else:
                        feat_zscore[i] = 0

                # Momentum (rate of change)
                if self.config.compute_momentum and W >= 2:
                    feat_momentum[i] = (values[i] - values[i - 1]) if i > 0 else 0

                # Volatility (std of returns)
                if self.config.compute_volatility and W >= 3:
                    returns = np.diff(window)
                    feat_volatility[i] = np.std(returns)

                # Autocorrelation (lag-1)
                if self.config.compute_autocorr and W >= 4:
                    try:
                        autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]
                        feat_autocorr[i] = autocorr if not np.isnan(autocorr) else 0
                    except:
                        feat_autocorr[i] = 0

            # Store features with window suffix
            w_suffix = f'_{W}'

            if self.config.compute_mean:
                features[f'{prefix}mean{w_suffix}'] = feat_mean
            if self.config.compute_std:
                features[f'{prefix}std{w_suffix}'] = feat_std
            if self.config.compute_min:
                features[f'{prefix}min{w_suffix}'] = feat_min
            if self.config.compute_max:
                features[f'{prefix}max{w_suffix}'] = feat_max
            if self.config.compute_range:
                features[f'{prefix}range{w_suffix}'] = feat_range
            if self.config.compute_slope:
                features[f'{prefix}slope{w_suffix}'] = feat_slope
            if self.config.compute_delta:
                features[f'{prefix}delta{w_suffix}'] = feat_delta
            if self.config.compute_curvature:
                features[f'{prefix}curv{w_suffix}'] = feat_curv
            if self.config.compute_skew:
                features[f'{prefix}skew{w_suffix}'] = feat_skew
            if self.config.compute_kurtosis:
                features[f'{prefix}kurt{w_suffix}'] = feat_kurt
            if self.config.compute_quantiles:
                features[f'{prefix}q25{w_suffix}'] = feat_q25
                features[f'{prefix}q50{w_suffix}'] = feat_q50
                features[f'{prefix}q75{w_suffix}'] = feat_q75
            if self.config.compute_iqr:
                features[f'{prefix}iqr{w_suffix}'] = feat_iqr
            if self.config.compute_cv:
                features[f'{prefix}cv{w_suffix}'] = feat_cv
            if self.config.compute_zscore:
                features[f'{prefix}zscore{w_suffix}'] = feat_zscore
            if self.config.compute_momentum:
                features[f'{prefix}momentum{w_suffix}'] = feat_momentum
            if self.config.compute_volatility:
                features[f'{prefix}volatility{w_suffix}'] = feat_volatility
            if self.config.compute_autocorr:
                features[f'{prefix}autocorr{w_suffix}'] = feat_autocorr

        if return_df:
            return pd.DataFrame(features)
        return features

    def compute_multi_signal(
        self,
        data: pd.DataFrame,
        signal_cols: List[str],
        entity_col: str = None,
        sort_col: str = None,
    ) -> pd.DataFrame:
        """
        Compute rolling features for multiple signals.

        Args:
            data: DataFrame with signal columns
            signal_cols: List of column names to process
            entity_col: If provided, compute per-entity
            sort_col: Column to sort by (e.g., 'cycle', 'timestamp')

        Returns:
            DataFrame with original data + rolling features
        """
        result = data.copy()

        if entity_col:
            # Process per entity
            all_features = []

            for entity in data[entity_col].unique():
                entity_data = data[data[entity_col] == entity].copy()

                if sort_col:
                    entity_data = entity_data.sort_values(sort_col)

                for col in signal_cols:
                    values = entity_data[col].values
                    feats = self.compute(values, prefix=col)

                    for feat_name, feat_vals in feats.items():
                        entity_data[feat_name] = feat_vals

                all_features.append(entity_data)

            result = pd.concat(all_features, ignore_index=True)
        else:
            # Process entire dataset
            if sort_col:
                result = result.sort_values(sort_col)

            for col in signal_cols:
                values = result[col].values
                feats = self.compute(values, prefix=col)

                for feat_name, feat_vals in feats.items():
                    result[feat_name] = feat_vals

        return result


# =============================================================================
# CLUSTER NORMALIZATION
# =============================================================================

@dataclass
class ClusterBaseline:
    """Baseline statistics for a single cluster."""
    cluster_id: int
    signal_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_samples: int = 0


class ClusterNormalizer:
    """
    Normalize features by cluster/regime membership.

    This is essential for multi-condition datasets where raw values
    vary by operating condition. Normalizing by cluster makes
    degradation comparable across regimes.

    Process:
    1. Cluster data by operating conditions
    2. Compute baseline (healthy) statistics per cluster
    3. Normalize new observations by their cluster's baseline

    Example:
        normalizer = ClusterNormalizer(n_clusters=6)
        normalizer.fit(train_df, op_cols=['op_1', 'op_2'], signal_cols=['s11', 's12'])
        normalized = normalizer.transform(test_df)
    """

    def __init__(
        self,
        n_clusters: int = 6,
        healthy_pct: float = 0.20,
        use_median: bool = False,
        robust_std: bool = True,
    ):
        """
        Args:
            n_clusters: Number of operating regimes/clusters
            healthy_pct: Fraction of life considered "healthy" (0.0 to 1.0)
            use_median: Use median instead of mean for baseline
            robust_std: Use MAD-based robust std instead of regular std
        """
        self.n_clusters = n_clusters
        self.healthy_pct = healthy_pct
        self.use_median = use_median
        self.robust_std = robust_std

        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.baselines: Dict[int, ClusterBaseline] = {}
        self.op_cols: List[str] = []
        self.signal_cols: List[str] = []
        self.fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        op_cols: List[str],
        signal_cols: List[str],
        entity_col: str = None,
        time_col: str = None,
    ) -> 'ClusterNormalizer':
        """
        Learn cluster structure and baseline statistics.

        Args:
            data: Training DataFrame
            op_cols: Operating condition columns (for clustering)
            signal_cols: Signal columns to normalize
            entity_col: Entity/unit identifier column
            time_col: Time/cycle column (for computing life percentage)
        """
        self.op_cols = op_cols
        self.signal_cols = signal_cols

        df = data.copy()

        # Step 1: Cluster operating conditions
        print(f"[1/3] Clustering {len(df):,} samples into {self.n_clusters} regimes...")

        op_data = df[op_cols].values
        self.scaler = StandardScaler()
        op_scaled = self.scaler.fit_transform(op_data)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        cluster_labels = self.kmeans.fit_predict(op_scaled)
        df['_cluster'] = cluster_labels

        # Show cluster distribution
        for c in range(self.n_clusters):
            count = (cluster_labels == c).sum()
            center = self.scaler.inverse_transform([self.kmeans.cluster_centers_[c]])[0]
            print(f"  Cluster {c}: {count:,} samples, center={[f'{v:.3f}' for v in center]}")

        # Step 2: Compute life percentage per entity
        print(f"\n[2/3] Computing healthy baselines (first {self.healthy_pct:.0%} of life)...")

        if entity_col and time_col:
            # Per-entity life percentage
            df['_life_pct'] = df.groupby(entity_col)[time_col].transform(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
            )
        else:
            # Assume entire dataset spans full life
            df['_life_pct'] = np.linspace(0, 1, len(df))

        # Filter to healthy portion
        healthy_df = df[df['_life_pct'] <= self.healthy_pct]
        print(f"  Healthy samples: {len(healthy_df):,} ({len(healthy_df)/len(df):.1%} of total)")

        # Step 3: Compute per-cluster baselines
        print(f"\n[3/3] Computing per-cluster statistics for {len(signal_cols)} signals...")

        for cluster_id in range(self.n_clusters):
            cluster_healthy = healthy_df[healthy_df['_cluster'] == cluster_id]

            baseline = ClusterBaseline(
                cluster_id=cluster_id,
                n_samples=len(cluster_healthy),
            )

            for signal in signal_cols:
                if signal not in cluster_healthy.columns:
                    continue

                values = cluster_healthy[signal].dropna().values

                if len(values) < 5:
                    print(f"  WARNING: Cluster {cluster_id} has only {len(values)} healthy samples for {signal}")
                    continue

                if self.use_median:
                    center = np.median(values)
                else:
                    center = np.mean(values)

                if self.robust_std:
                    # MAD-based robust std (multiply by 1.4826 for normal equivalence)
                    spread = np.median(np.abs(values - np.median(values))) * 1.4826
                else:
                    spread = np.std(values)

                # Ensure non-zero spread
                spread = max(spread, 1e-10)

                baseline.signal_stats[signal] = {
                    'center': float(center),
                    'spread': float(spread),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values),
                }

            self.baselines[cluster_id] = baseline
            print(f"  Cluster {cluster_id}: {baseline.n_samples:,} samples, {len(baseline.signal_stats)} signals")

        self.fitted = True
        print("\n[OK] Fitting complete!")

        return self

    def transform(
        self,
        data: pd.DataFrame,
        add_cluster_id: bool = True,
        output_mode: str = 'zscore',
    ) -> pd.DataFrame:
        """
        Transform data using learned cluster baselines.

        Args:
            data: DataFrame to transform
            add_cluster_id: Add cluster assignment column
            output_mode:
                'zscore': (value - center) / spread
                'distance': |value - center| / spread (always positive)
                'percentile': Position relative to healthy range

        Returns:
            DataFrame with normalized features
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")

        df = data.copy()

        # Assign clusters
        op_data = df[self.op_cols].values
        op_scaled = self.scaler.transform(op_data)
        cluster_labels = self.kmeans.predict(op_scaled)

        if add_cluster_id:
            df['cluster_id'] = cluster_labels

        # Normalize each signal
        for signal in self.signal_cols:
            if signal not in df.columns:
                continue

            normalized = np.full(len(df), np.nan)

            for cluster_id in range(self.n_clusters):
                mask = cluster_labels == cluster_id

                if cluster_id not in self.baselines:
                    continue

                baseline = self.baselines[cluster_id]

                if signal not in baseline.signal_stats:
                    continue

                stats = baseline.signal_stats[signal]
                values = df.loc[mask, signal].values

                if output_mode == 'zscore':
                    # Standard z-score (can be negative)
                    normalized[mask] = (values - stats['center']) / stats['spread']

                elif output_mode == 'distance':
                    # Absolute distance (always positive)
                    normalized[mask] = np.abs(values - stats['center']) / stats['spread']

                elif output_mode == 'percentile':
                    # Position in healthy range [0, 1]
                    healthy_range = stats['max'] - stats['min']
                    if healthy_range > 1e-10:
                        normalized[mask] = (values - stats['min']) / healthy_range
                    else:
                        normalized[mask] = 0.5

            # Add normalized column
            df[f'{signal}_norm'] = normalized

        return df

    def compute_healthy_distance(
        self,
        data: pd.DataFrame,
        aggregate: bool = True,
    ) -> pd.DataFrame:
        """
        Compute healthy distance features (like we did for C-MAPSS).

        This is the key regime-aware feature: z-score distance from
        the healthy baseline of each signal's operating regime.

        Args:
            data: DataFrame to process
            aggregate: Add aggregate metrics (mean, max, std of distances)
        """
        df = self.transform(data, output_mode='distance')

        if aggregate:
            # Get all distance columns
            dist_cols = [c for c in df.columns if c.endswith('_norm')]

            if dist_cols:
                df['hd_mean'] = df[dist_cols].mean(axis=1)
                df['hd_max'] = df[dist_cols].max(axis=1)
                df['hd_std'] = df[dist_cols].std(axis=1)
                df['hd_sum'] = df[dist_cols].sum(axis=1)

                # Top-N distances
                for i, col in enumerate(df[dist_cols].values.argsort(axis=1)[:, -3:][:, ::-1].T):
                    df[f'hd_top{i+1}'] = df[dist_cols].values[np.arange(len(df)), col]

        return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_all_rolling_features(
    data: pd.DataFrame,
    signal_cols: List[str],
    windows: List[int] = None,
    entity_col: str = None,
    sort_col: str = None,
    config: RollingConfig = None,
) -> pd.DataFrame:
    """
    Convenience function to compute all rolling features.

    Example:
        df = compute_all_rolling_features(
            data=train_df,
            signal_cols=['s11', 's12', 's15'],
            windows=[10, 20, 30],
            entity_col='unit_id',
            sort_col='cycle',
        )
    """
    engine = RollingFeatureEngine(windows=windows, config=config)
    return engine.compute_multi_signal(
        data=data,
        signal_cols=signal_cols,
        entity_col=entity_col,
        sort_col=sort_col,
    )


def compute_cluster_normalized_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    op_cols: List[str],
    signal_cols: List[str],
    n_clusters: int = 6,
    entity_col: str = None,
    time_col: str = None,
    healthy_pct: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, ClusterNormalizer]:
    """
    Convenience function for cluster normalization pipeline.

    Example:
        train_norm, test_norm, normalizer = compute_cluster_normalized_features(
            train_data=train_df,
            test_data=test_df,
            op_cols=['op_1', 'op_2'],
            signal_cols=['s11', 's12', 's15'],
            n_clusters=6,
            entity_col='unit_id',
            time_col='cycle',
        )
    """
    normalizer = ClusterNormalizer(
        n_clusters=n_clusters,
        healthy_pct=healthy_pct,
    )

    normalizer.fit(
        data=train_data,
        op_cols=op_cols,
        signal_cols=signal_cols,
        entity_col=entity_col,
        time_col=time_col,
    )

    train_norm = normalizer.compute_healthy_distance(train_data)
    test_norm = normalizer.compute_healthy_distance(test_data)

    return train_norm, test_norm, normalizer


# =============================================================================
# COMBINED PIPELINE
# =============================================================================

class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline combining:
    1. Cluster normalization (regime-aware baselines)
    2. Heavy rolling window statistics

    This is the full pipeline that achieved our best C-MAPSS results.

    Example:
        pipeline = FeatureEngineeringPipeline(
            n_clusters=6,
            windows=[10, 20, 30],
            op_cols=['op_1', 'op_2'],
            signal_cols=['s11', 's12', 's15'],
        )

        train_features = pipeline.fit_transform(train_df, entity_col='unit', time_col='cycle')
        test_features = pipeline.transform(test_df)
    """

    def __init__(
        self,
        n_clusters: int = 6,
        windows: List[int] = None,
        op_cols: List[str] = None,
        signal_cols: List[str] = None,
        healthy_pct: float = 0.20,
        rolling_config: RollingConfig = None,
    ):
        self.n_clusters = n_clusters
        self.windows = windows or [10, 20, 30]
        self.op_cols = op_cols or []
        self.signal_cols = signal_cols or []
        self.healthy_pct = healthy_pct

        self.normalizer = ClusterNormalizer(
            n_clusters=n_clusters,
            healthy_pct=healthy_pct,
        )
        self.rolling_engine = RollingFeatureEngine(
            windows=self.windows,
            config=rolling_config,
        )

        self.fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        entity_col: str = None,
        time_col: str = None,
    ) -> 'FeatureEngineeringPipeline':
        """Fit the pipeline on training data."""
        print("=" * 70)
        print("FEATURE ENGINEERING PIPELINE - FIT")
        print("=" * 70)

        self.entity_col = entity_col
        self.time_col = time_col

        # Fit cluster normalizer
        self.normalizer.fit(
            data=data,
            op_cols=self.op_cols,
            signal_cols=self.signal_cols,
            entity_col=entity_col,
            time_col=time_col,
        )

        self.fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline."""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")

        print("\n" + "-" * 70)
        print("TRANSFORMING DATA")
        print("-" * 70)

        # Step 1: Cluster normalization
        print("\n[1/2] Computing cluster-normalized healthy distances...")
        df = self.normalizer.compute_healthy_distance(data)

        # Step 2: Rolling features on healthy distance
        print("\n[2/2] Computing rolling window statistics...")

        # Compute rolling features on the aggregate healthy distance
        if 'hd_mean' in df.columns:
            df = self.rolling_engine.compute_multi_signal(
                data=df,
                signal_cols=['hd_mean'],
                entity_col=self.entity_col if hasattr(self, 'entity_col') else None,
                sort_col=self.time_col if hasattr(self, 'time_col') else None,
            )

        # Also compute rolling on key normalized signals
        norm_cols = [c for c in df.columns if c.endswith('_norm')][:5]  # Top 5
        if norm_cols:
            df = self.rolling_engine.compute_multi_signal(
                data=df,
                signal_cols=norm_cols,
                entity_col=self.entity_col if hasattr(self, 'entity_col') else None,
                sort_col=self.time_col if hasattr(self, 'time_col') else None,
            )

        print(f"\n[OK] Generated {len(df.columns) - len(data.columns)} new features")

        return df

    def fit_transform(
        self,
        data: pd.DataFrame,
        entity_col: str = None,
        time_col: str = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(data, entity_col, time_col)
        return self.transform(data)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RollingConfig',
    'RollingFeatureEngine',
    'ClusterBaseline',
    'ClusterNormalizer',
    'FeatureEngineeringPipeline',
    'compute_all_rolling_features',
    'compute_cluster_normalized_features',
]
