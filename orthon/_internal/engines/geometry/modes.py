"""
prism/modules/modes.py

Discover behavioral modes from Laplace signatures.

NOMENCLATURE:
    - Domain:    Required. The complete system under analysis. NO DEFAULT.
    - Cohort:    Predefined physical/logical grouping (input)
    - Signal: A single signal topology (input)
    - Mode:      Discovered behavioral grouping (output)

Signals that share similar Laplace dynamics belong to the same MODE.
This is NOT the same as a cohort or "laplace_cohort" (deprecated term).

How Does an Signal Get a Mode Score?
---------------------------------------
1. Extract Laplace fingerprint (gradient/divergence statistics)
2. Cluster fingerprints using GMM (soft assignment)
3. Compute mode_id, mode_affinity, mode_entropy

Key Insight: Low affinity / high entropy = signal is changing modes = REGIME TRANSITION SIGNAL.
"""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# LAPLACE FINGERPRINT EXTRACTION
# =============================================================================

def extract_laplace_fingerprint(
    field_df: pl.DataFrame,
    signal_id: str,
    cohort_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract Laplace fingerprint for a single signal.

    Fingerprint features:
    - gradient_mean: Average rate of change
    - gradient_std: Volatility of change
    - gradient_skew: Asymmetry of changes
    - divergence_mean: Average field divergence
    - divergence_std: Stability of divergence
    - abs_divergence_mean: Magnitude of divergence

    Args:
        field_df: Laplace field DataFrame with gradient/divergence columns
        signal_id: Signal to extract fingerprint for
        cohort_id: Optional cohort filter

    Returns:
        Dictionary with fingerprint features, or None if insufficient data
    """
    # Filter to signal
    subset = field_df.filter(pl.col('signal_id') == signal_id)

    if cohort_id is not None and 'cohort_id' in field_df.columns:
        subset = subset.filter(pl.col('cohort_id') == cohort_id)

    if len(subset) < 5:
        return None

    # Extract gradient and divergence
    grad = subset['gradient'].drop_nulls().to_numpy()
    div = subset['divergence'].drop_nulls().to_numpy()

    if len(grad) < 5 or len(div) < 5:
        return None

    return {
        'signal_id': signal_id,
        'cohort_id': cohort_id,
        'gradient_mean': float(np.nanmean(grad)),
        'gradient_std': float(np.nanstd(grad)),
        'gradient_skew': float(pd.Series(grad).skew()) if len(grad) > 2 else 0.0,
        'divergence_mean': float(np.nanmean(div)),
        'divergence_std': float(np.nanstd(div)),
        'abs_divergence_mean': float(np.nanmean(np.abs(div))),
    }


def extract_cohort_fingerprints(
    field_df: pl.DataFrame,
    cohort_id: str,
    signals: List[str]
) -> pd.DataFrame:
    """
    Extract Laplace fingerprints for all signals in a cohort.

    Args:
        field_df: Laplace field DataFrame
        cohort_id: Cohort identifier
        signals: List of signal IDs in the cohort

    Returns:
        DataFrame with fingerprint features per signal
    """
    fingerprints = []

    for ind in signals:
        fp = extract_laplace_fingerprint(field_df, ind, cohort_id)
        if fp is not None:
            fingerprints.append(fp)

    if not fingerprints:
        return pd.DataFrame()

    return pd.DataFrame(fingerprints)


# =============================================================================
# MODE DISCOVERY (GMM CLUSTERING)
# =============================================================================

def find_optimal_modes(X: np.ndarray, max_modes: int = 10, min_modes: int = 2) -> int:
    """
    Find optimal number of modes using BIC (Bayesian Information Criterion).

    Args:
        X: Feature matrix (n_samples, n_features)
        max_modes: Maximum modes to consider
        min_modes: Minimum modes to consider

    Returns:
        Optimal number of modes
    """
    if len(X) < min_modes + 1:
        return min_modes

    best_bic = np.inf
    best_n = min_modes

    for n in range(min_modes, min(max_modes + 1, len(X))):
        try:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
            gmm.fit(X)
            bic = gmm.bic(X)

            if bic < best_bic:
                best_bic = bic
                best_n = n
        except Exception as e:
            logger.debug(f"GMM with n={n} failed: {e}")
            continue

    return best_n


def compute_mode_scores(mode_probabilities: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute mode scores from GMM probability matrix.

    Args:
        mode_probabilities: (n_signals, n_modes) probability matrix

    Returns:
        Dictionary with mode_id, mode_affinity, mode_entropy arrays
    """
    mode_id = np.argmax(mode_probabilities, axis=1)
    mode_affinity = np.max(mode_probabilities, axis=1)

    # Entropy: -Σ(p × log(p)) - measures uncertainty
    # Higher entropy = more uncertain about mode assignment
    mode_entropy = -np.sum(
        mode_probabilities * np.log(mode_probabilities + 1e-10),
        axis=1
    )

    return {
        'mode_id': mode_id,
        'mode_affinity': mode_affinity,
        'mode_entropy': mode_entropy,
    }


def discover_modes(
    field_df: pl.DataFrame,
    domain_id: str,
    cohort_id: str,
    signals: List[str],
    max_modes: int = 10,
) -> Optional[pd.DataFrame]:
    """
    Discover modes within a cohort from Laplace fingerprints.

    Uses Gaussian Mixture Model for soft clustering, providing:
    - mode_id: Primary mode assignment
    - mode_affinity: Confidence in assignment (0-1)
    - mode_entropy: Uncertainty of assignment (lower = more certain)

    Args:
        field_df: Laplace field DataFrame
        domain_id: Domain identifier
        cohort_id: Cohort identifier
        signals: List of signal IDs
        max_modes: Maximum number of modes to discover

    Returns:
        DataFrame with mode assignments, or None if insufficient data
    """
    if len(signals) < 3:
        logger.warning(f"Cohort {cohort_id}: insufficient signals ({len(signals)}) for mode discovery")
        return None

    # Extract fingerprints
    fp_df = extract_cohort_fingerprints(field_df, cohort_id, signals)

    if len(fp_df) < 3:
        logger.warning(f"Cohort {cohort_id}: insufficient fingerprints ({len(fp_df)})")
        return None

    # Feature columns for clustering
    feature_cols = [
        'gradient_mean', 'gradient_std', 'gradient_skew',
        'divergence_mean', 'divergence_std', 'abs_divergence_mean'
    ]

    X = fp_df[feature_cols].fillna(0).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal number of modes
    best_n = find_optimal_modes(X_scaled, max_modes)
    logger.info(f"Cohort {cohort_id}: discovered {best_n} modes from {len(fp_df)} signals")

    # Fit GMM
    gmm = GaussianMixture(
        n_components=best_n,
        covariance_type='full',
        random_state=42,
        max_iter=200
    )
    gmm.fit(X_scaled)

    # Get soft assignments
    probs = gmm.predict_proba(X_scaled)
    scores = compute_mode_scores(probs)

    # Build result DataFrame
    result = fp_df[['signal_id', 'cohort_id']].copy()
    result['domain_id'] = domain_id
    result['mode_id'] = scores['mode_id']
    result['mode_affinity'] = scores['mode_affinity']
    result['mode_entropy'] = scores['mode_entropy']
    result['n_modes'] = best_n

    # Include fingerprint for queryability
    for col in feature_cols:
        result[f'fingerprint_{col}'] = fp_df[col].values

    return result


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def run_modes(
    input_path: str,
    output_path: str,
    domain_id: str,
    cohort_ids: Optional[List[str]] = None,
    cohort_members: Optional[pl.DataFrame] = None,
    max_modes: int = 10,
) -> pd.DataFrame:
    """
    Run mode discovery on signal field data.

    Args:
        input_path: Path to signal_field.parquet
        output_path: Path to save cohort_modes.parquet
        domain_id: Domain identifier (REQUIRED)
        cohort_ids: Specific cohorts to process (None = all)
        cohort_members: DataFrame mapping cohort_id -> signal_id
        max_modes: Maximum modes per cohort

    Returns:
        Combined DataFrame of all mode assignments
    """
    logger.info(f"Loading signal field from {input_path}")

    # Use lazy scan to determine cohorts without loading full file
    lazy_field = pl.scan_parquet(input_path)

    # Determine cohorts and their signals
    if cohort_members is not None:
        if cohort_ids is None:
            cohort_ids = cohort_members['cohort_id'].unique().to_list()

        cohort_signal_map = {
            cid: cohort_members.filter(
                pl.col('cohort_id') == cid
            )['signal_id'].to_list()
            for cid in cohort_ids
        }
    elif 'cohort_id' in lazy_field.collect_schema().names():
        if cohort_ids is None:
            # Lazy scan for unique cohort_ids only
            cohort_ids = (
                lazy_field
                .select('cohort_id')
                .unique()
                .collect()
            )['cohort_id'].to_list()

        # Build signal map from lazy scans (one per cohort)
        cohort_signal_map = {}
        for cid in cohort_ids:
            cohort_signal_map[cid] = (
                lazy_field
                .filter(pl.col('cohort_id') == cid)
                .select('signal_id')
                .unique()
                .collect()
            )['signal_id'].to_list()
    else:
        # Single cohort = all signals (lazy scan for unique signal_ids)
        all_signals = (
            lazy_field
            .select('signal_id')
            .unique()
            .collect()
        )['signal_id'].to_list()
        cohort_ids = ['default']
        cohort_signal_map = {'default': all_signals}

    logger.info(f"Processing {len(cohort_ids)} cohorts in domain {domain_id}")

    # Process each cohort with per-cohort lazy loading
    all_modes = []
    for cohort_id in cohort_ids:
        signals = cohort_signal_map.get(cohort_id, [])

        if len(signals) < 3:
            continue

        # Lazy load only this cohort's data (filter pushdown)
        if 'cohort_id' in lazy_field.collect_schema().names():
            cohort_field_df = (
                lazy_field
                .filter(pl.col('cohort_id') == cohort_id)
                .collect()
            )
        else:
            # Single cohort case - filter by signals
            cohort_field_df = (
                lazy_field
                .filter(pl.col('signal_id').is_in(signals))
                .collect()
            )

        modes_df = discover_modes(
            cohort_field_df, domain_id, cohort_id, signals, max_modes
        )

        if modes_df is not None and len(modes_df) > 0:
            all_modes.append(modes_df)

    if all_modes:
        result = pd.concat(all_modes, ignore_index=True)
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(result)} mode assignments to {output_path}")
        return result
    else:
        logger.warning("No modes discovered")
        return pd.DataFrame()


# =============================================================================
# AFFINITY-WEIGHTED FEATURES (PR #6)
# =============================================================================

def compute_affinity_weighted_features(
    modes_df: pl.DataFrame,
    signal_metrics: pl.DataFrame,
    cohort_id: str,
) -> Dict[str, Any]:
    """
    Compute affinity-weighted mode features.

    High-affinity signals (strongly in their mode) dominate.
    Low-affinity signals (transitioning between modes) are downweighted.

    Problem solved: Binary mode assignment loses information. An signal
    with 0.51 affinity and one with 0.99 affinity are treated identically.
    This function weights by affinity to preserve that information.

    Args:
        modes_df: Mode assignments with mode_affinity scores
        signal_metrics: Per-signal characterization metrics (long format)
        cohort_id: Cohort to process

    Returns:
        Dictionary of affinity-weighted features
    """
    # Filter to cohort
    cohort_modes = modes_df.filter(pl.col('cohort_id') == cohort_id)

    if len(cohort_modes) == 0:
        return {'cohort_id': cohort_id}

    # Convert to pandas for easier manipulation
    modes_pd = cohort_modes.to_pandas()

    features = {'cohort_id': cohort_id}

    # Global affinity statistics
    features['aff_mean'] = float(modes_pd['mode_affinity'].mean())
    features['aff_std'] = float(modes_pd['mode_affinity'].std())
    features['aff_min'] = float(modes_pd['mode_affinity'].min())
    features['aff_range'] = float(modes_pd['mode_affinity'].max() - modes_pd['mode_affinity'].min())

    # Entropy statistics
    features['entropy_mean'] = float(modes_pd['mode_entropy'].mean())
    features['entropy_std'] = float(modes_pd['mode_entropy'].std())
    features['entropy_max'] = float(modes_pd['mode_entropy'].max())

    # Get fingerprint columns (from modes_df)
    fingerprint_cols = [c for c in modes_pd.columns if c.startswith('fingerprint_')]

    # Affinity-weighted aggregations per mode
    modes = sorted(modes_pd['mode_id'].unique())

    for mode_id in modes:
        mode_data = modes_pd[modes_pd['mode_id'] == mode_id]
        affinities = mode_data['mode_affinity'].values
        total_affinity = affinities.sum()

        if total_affinity == 0:
            continue

        # Affinity-weighted fingerprint features
        for col in fingerprint_cols:
            if col not in mode_data.columns:
                continue

            values = mode_data[col].values
            valid_mask = ~np.isnan(values)

            if valid_mask.sum() == 0:
                continue

            valid_values = values[valid_mask]
            valid_affinities = affinities[valid_mask]
            valid_total = valid_affinities.sum()

            if valid_total == 0:
                continue

            # Affinity-weighted mean
            weighted_mean = (valid_values * valid_affinities).sum() / valid_total
            short_name = col.replace('fingerprint_', '')
            features[f'm{mode_id}_{short_name}_wmean'] = float(weighted_mean)

            # Affinity-weighted variance
            if len(valid_values) > 1:
                weighted_var = (valid_affinities * (valid_values - weighted_mean)**2).sum() / valid_total
                features[f'm{mode_id}_{short_name}_wvar'] = float(weighted_var)

    # Cross-mode affinity contrast (compare weighted means between modes)
    for i, mode_i in enumerate(modes):
        for mode_j in modes[i+1:]:
            for col in fingerprint_cols:
                short_name = col.replace('fingerprint_', '')
                key_i = f'm{mode_i}_{short_name}_wmean'
                key_j = f'm{mode_j}_{short_name}_wmean'
                if key_i in features and key_j in features:
                    contrast = abs(features[key_i] - features[key_j])
                    features[f'contrast_{mode_i}_{mode_j}_{short_name}'] = float(contrast)

    # Transitioning signal analysis
    # High entropy = transitioning between modes = early warning
    median_entropy = modes_pd['mode_entropy'].median()
    high_entropy_mask = modes_pd['mode_entropy'] > median_entropy

    features['n_transitioning'] = int(high_entropy_mask.sum())
    features['transitioning_ratio'] = float(high_entropy_mask.mean())

    # Compare transitioning vs stable signals
    if high_entropy_mask.sum() > 0 and (~high_entropy_mask).sum() > 0:
        transitioning = modes_pd[high_entropy_mask]
        stable = modes_pd[~high_entropy_mask]

        for col in fingerprint_cols:
            if col not in modes_pd.columns:
                continue

            short_name = col.replace('fingerprint_', '')
            trans_mean = transitioning[col].mean()
            stable_mean = stable[col].mean()

            if not np.isnan(trans_mean) and not np.isnan(stable_mean):
                features[f'trans_{short_name}_mean'] = float(trans_mean)
                features[f'stable_{short_name}_mean'] = float(stable_mean)
                features[f'trans_contrast_{short_name}'] = float(trans_mean - stable_mean)

    # Mode concentration (ratio of actual modes used)
    n_modes = modes_pd['n_modes'].iloc[0] if 'n_modes' in modes_pd.columns else len(modes)
    features['n_modes_used'] = len(modes)
    features['n_modes_total'] = int(n_modes)
    features['mode_concentration'] = float(len(modes) / n_modes) if n_modes > 0 else 1.0

    return features


def compute_affinity_dynamics(
    modes_history: pl.DataFrame,
    cohort_id: str,
) -> Dict[str, Any]:
    """
    Track how affinity changes over time for each signal.

    Dropping affinity = signal leaving its mode = regime transition.

    Args:
        modes_history: Signal topology of mode assignments (multiple windows)
                      Must have 'window_end' or 'obs_date' column
        cohort_id: Cohort to analyze

    Returns:
        Affinity dynamics features
    """
    cohort_data = modes_history.filter(pl.col('cohort_id') == cohort_id)

    if len(cohort_data) < 2:
        return {'cohort_id': cohort_id}

    # Convert to pandas
    cohort_pd = cohort_data.to_pandas()

    # Determine time column
    time_col = 'window_end' if 'window_end' in cohort_pd.columns else 'obs_date'
    if time_col not in cohort_pd.columns:
        return {'cohort_id': cohort_id}

    features = {'cohort_id': cohort_id}

    # Per-signal affinity trajectory
    signals = cohort_pd['signal_id'].unique()

    affinity_trends = []
    mode_switches = []

    for ind_id in signals:
        ind_data = cohort_pd[cohort_pd['signal_id'] == ind_id].sort_values(time_col)

        if len(ind_data) < 2:
            continue

        affinities = ind_data['mode_affinity'].values
        modes = ind_data['mode_id'].values

        # Affinity trend (positive = strengthening, negative = weakening)
        if len(affinities) > 1:
            trend = np.polyfit(range(len(affinities)), affinities, 1)[0]
            affinity_trends.append(trend)

        # Count mode switches
        switches = np.sum(np.diff(modes) != 0)
        mode_switches.append(switches)

    if affinity_trends:
        features['aff_trend_mean'] = float(np.mean(affinity_trends))
        features['aff_trend_std'] = float(np.std(affinity_trends))
        features['aff_trend_min'] = float(np.min(affinity_trends))
        features['n_weakening'] = int(np.sum(np.array(affinity_trends) < -0.01))
        features['n_strengthening'] = int(np.sum(np.array(affinity_trends) > 0.01))

    if mode_switches:
        features['total_mode_switches'] = int(np.sum(mode_switches))
        features['signals_with_switches'] = int(np.sum(np.array(mode_switches) > 0))
        features['max_switches_per_signal'] = int(np.max(mode_switches))
        features['switch_ratio'] = float(np.mean(np.array(mode_switches) > 0))

    return features


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Discover modes from Laplace fingerprints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
NOMENCLATURE:
  Mode = Discovered behavioral grouping from Laplace dynamics
  This is NOT the same as a cohort (predefined grouping).

  The term "laplace_cohort" is DEPRECATED. Use "mode" instead.

OUTPUT:
  cohort_modes.parquet with columns:
    - domain_id, cohort_id, signal_id
    - mode_id: Primary mode assignment (0, 1, 2, ...)
    - mode_affinity: Confidence in assignment (0-1)
    - mode_entropy: Uncertainty (lower = more certain)
    - fingerprint_*: Laplace signature features

KEY INSIGHT:
  Low affinity / high entropy = REGIME TRANSITION SIGNAL
        '''
    )

    parser.add_argument('--input', required=True, help='signal_field.parquet')
    parser.add_argument('--output', required=True, help='cohort_modes.parquet')
    parser.add_argument('--domain', required=True, help='Domain ID (REQUIRED, no default)')
    parser.add_argument('--cohort', action='append', dest='cohorts', help='Cohort ID(s) to process')
    parser.add_argument('--max-modes', type=int, default=10, help='Maximum modes per cohort')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    run_modes(
        args.input,
        args.output,
        args.domain,
        args.cohorts,
        max_modes=args.max_modes
    )
