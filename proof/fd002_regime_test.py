#!/usr/bin/env python3
"""
FD002 Regime-Aware Test
=======================

Tests if regime-aware healthy baselines can close the FD002 gap.

Pipeline:
1. Load FD002 raw data
2. Cluster operating conditions into 6 regimes
3. Compute per-regime healthy baselines (first 20% of life)
4. Compute healthy_distance features (z-score from regime baseline)
5. Train XGBoost with regime-aware features
6. Compare to baseline

Key hypothesis: healthy_distance captures degradation with regime variation factored out.

Usage:
    python fd002_regime_test.py
    python fd002_regime_test.py --data-dir /path/to/cmapss
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not installed. Using sklearn GradientBoosting instead.")
    from sklearn.ensemble import GradientBoostingRegressor


# =============================================================================
# C-MAPSS DATA LOADING
# =============================================================================

COLUMNS = ['unit_id', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]

# Sensors known to be informative (drop near-constant: s1, s5, s6, s10, s16, s18, s19)
INFORMATIVE_SENSORS = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12',
                       's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

# Operating condition columns
OP_COLS = ['op_1', 'op_2', 'op_3']


def load_cmapss(path: str) -> pd.DataFrame:
    """Load C-MAPSS txt file."""
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)
    return df


def add_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """Add RUL column: max_cycle - current_cycle per unit, capped."""
    max_cycles = df.groupby('unit_id')['cycle'].max().rename('max_cycle')
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    if cap > 0:
        df['RUL'] = df['RUL'].clip(upper=cap)
    return df


def load_rul_file(path: str) -> np.ndarray:
    """Load ground truth RUL file."""
    with open(path, 'r') as f:
        return np.array([float(line.strip()) for line in f if line.strip()])


# =============================================================================
# REGIME LEARNING
# =============================================================================

@dataclass
class LearnedRegimes:
    """Container for learned regime structure."""

    n_regimes: int = 6
    regime_centers: np.ndarray = None
    regime_scaler: StandardScaler = None

    # Per-regime healthy baselines: regime_id -> signal -> {mean, std}
    healthy_baselines: Dict[int, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    # Signal names
    signal_names: List[str] = field(default_factory=list)

    # Regime assignment for training data
    train_regime_assignments: Dict[Tuple[int, int], int] = field(default_factory=dict)


def cluster_operating_conditions(df: pd.DataFrame, n_regimes: int = 6) -> Tuple[KMeans, StandardScaler, np.ndarray]:
    """
    Cluster operating conditions into regimes.

    FD002 has 6 operating conditions defined by (altitude, Mach, TRA).
    We use op_1 (altitude proxy) and op_2 (Mach proxy) for clustering.
    """
    op_data = df[['op_1', 'op_2']].values

    scaler = StandardScaler()
    op_scaled = scaler.fit_transform(op_data)

    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    regime_labels = kmeans.fit_predict(op_scaled)

    return kmeans, scaler, regime_labels


def compute_healthy_baselines(
    df: pd.DataFrame,
    regime_labels: np.ndarray,
    healthy_pct: float = 0.20,
    signal_cols: List[str] = None,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Compute healthy baseline statistics per regime.

    Uses first `healthy_pct` of each unit's life as "healthy" data.
    Aggregates across all units within each regime.
    """
    if signal_cols is None:
        signal_cols = INFORMATIVE_SENSORS

    # Add regime labels to dataframe
    df = df.copy()
    df['regime'] = regime_labels

    # Compute lifecycle percentage per unit
    df['life_pct'] = df.groupby('unit_id')['cycle'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )

    # Filter to healthy portion
    healthy_df = df[df['life_pct'] <= healthy_pct]

    # Compute per-regime, per-signal statistics
    baselines = {}

    for regime in range(df['regime'].max() + 1):
        regime_healthy = healthy_df[healthy_df['regime'] == regime]

        if len(regime_healthy) < 10:
            print(f"  WARNING: Regime {regime} has only {len(regime_healthy)} healthy samples")
            continue

        baselines[regime] = {}

        for signal in signal_cols:
            if signal in regime_healthy.columns:
                values = regime_healthy[signal].dropna()
                if len(values) > 0:
                    baselines[regime][signal] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()) + 1e-10,  # Avoid division by zero
                        'count': len(values),
                    }

    return baselines


def learn_regimes(train_df: pd.DataFrame, n_regimes: int = 6) -> LearnedRegimes:
    """
    Learn regime structure from training data.
    """
    print("\n" + "="*70)
    print("REGIME LEARNING")
    print("="*70)

    learned = LearnedRegimes(n_regimes=n_regimes)
    learned.signal_names = INFORMATIVE_SENSORS

    # Step 1: Cluster operating conditions
    print("\n[1/2] Clustering operating conditions...")
    kmeans, scaler, regime_labels = cluster_operating_conditions(train_df, n_regimes)

    learned.regime_centers = kmeans.cluster_centers_
    learned.regime_scaler = scaler

    # Show regime distribution
    unique, counts = np.unique(regime_labels, return_counts=True)
    print(f"  Found {n_regimes} regimes:")
    for r, c in zip(unique, counts):
        center = scaler.inverse_transform([kmeans.cluster_centers_[r]])[0]
        print(f"    Regime {r}: op_1={center[0]:.2f}, op_2={center[1]:.4f} ({c:,} samples)")

    # Store regime assignments
    for idx, (_, row) in enumerate(train_df.iterrows()):
        key = (int(row['unit_id']), int(row['cycle']))
        learned.train_regime_assignments[key] = int(regime_labels[idx])

    # Step 2: Compute healthy baselines per regime
    print("\n[2/2] Computing healthy baselines per regime...")
    learned.healthy_baselines = compute_healthy_baselines(
        train_df, regime_labels, healthy_pct=0.20, signal_cols=INFORMATIVE_SENSORS
    )

    for regime, signals in learned.healthy_baselines.items():
        print(f"    Regime {regime}: {len(signals)} signals with healthy baselines")

    print("\n  Learning complete!")

    return learned


# =============================================================================
# REGIME APPLICATION (FEATURE GENERATION)
# =============================================================================

def assign_regime(row: pd.Series, learned: LearnedRegimes) -> int:
    """Assign operating regime to a single observation."""
    op_values = np.array([[row['op_1'], row['op_2']]])
    op_scaled = learned.regime_scaler.transform(op_values)
    distances = np.linalg.norm(op_scaled - learned.regime_centers, axis=1)
    return int(np.argmin(distances))


def compute_healthy_distance(
    row: pd.Series,
    regime: int,
    learned: LearnedRegimes,
) -> Dict[str, float]:
    """
    Compute z-score distance from healthy baseline for each signal.

    This is the key regime-aware feature: how far is each signal from
    its expected value in this operating regime?
    """
    distances = {}

    if regime not in learned.healthy_baselines:
        # Fallback to regime 0 if unknown
        regime = 0

    baseline = learned.healthy_baselines.get(regime, {})

    for signal in learned.signal_names:
        if signal in baseline and signal in row.index:
            mean = baseline[signal]['mean']
            std = baseline[signal]['std']
            value = row[signal]

            # Z-score distance
            distances[f'hd_{signal}'] = abs(value - mean) / std
        else:
            distances[f'hd_{signal}'] = 0.0

    # Aggregate metrics
    hd_values = [v for k, v in distances.items() if k.startswith('hd_')]
    if hd_values:
        distances['hd_mean'] = np.mean(hd_values)
        distances['hd_max'] = np.max(hd_values)
        distances['hd_std'] = np.std(hd_values)
    else:
        distances['hd_mean'] = 0.0
        distances['hd_max'] = 0.0
        distances['hd_std'] = 0.0

    return distances


def apply_regimes(df: pd.DataFrame, learned: LearnedRegimes, verbose: bool = True) -> pd.DataFrame:
    """
    Apply learned regimes to generate features.

    For each observation:
    1. Assign regime based on operating conditions
    2. Compute healthy_distance (z-score from regime baseline)
    3. Add regime one-hot encoding
    """
    if verbose:
        print("\n" + "="*70)
        print("APPLYING REGIME FEATURES")
        print("="*70)

    # Assign regimes
    if verbose:
        print("\n  Assigning regimes...")

    regimes = []
    for _, row in df.iterrows():
        regimes.append(assign_regime(row, learned))

    df = df.copy()
    df['regime_id'] = regimes

    # Show distribution
    if verbose:
        regime_counts = df['regime_id'].value_counts().sort_index()
        for r, c in regime_counts.items():
            print(f"    Regime {r}: {c:,} samples")

    # Compute healthy distances
    if verbose:
        print("\n  Computing healthy distances...")

    hd_rows = []
    for _, row in df.iterrows():
        regime = row['regime_id']
        hd = compute_healthy_distance(row, regime, learned)
        hd_rows.append(hd)

    hd_df = pd.DataFrame(hd_rows)

    # Merge
    df = pd.concat([df.reset_index(drop=True), hd_df.reset_index(drop=True)], axis=1)

    # Add regime one-hot encoding
    for r in range(learned.n_regimes):
        df[f'regime_{r}'] = (df['regime_id'] == r).astype(int)

    if verbose:
        hd_cols = [c for c in df.columns if c.startswith('hd_')]
        print(f"\n  Generated {len(hd_cols)} healthy_distance features")
        print(f"  Mean hd_mean: {df['hd_mean'].mean():.3f}")
        print(f"  Max hd_max: {df['hd_max'].max():.3f}")

    return df


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def train_baseline_model(train_df: pd.DataFrame, cap_rul: int = 125):
    """Train baseline XGBoost on raw sensors only."""

    feature_cols = OP_COLS + [f's_{i}' for i in range(1, 22)]

    X = train_df[feature_cols].values
    y = train_df['RUL'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    return model, feature_cols, val_rmse


def train_regime_model(train_df: pd.DataFrame, learned: LearnedRegimes, cap_rul: int = 125):
    """Train XGBoost with regime-aware features."""

    # Apply regime features
    train_df = apply_regimes(train_df, learned, verbose=True)

    # Feature columns: raw sensors + regime features
    raw_cols = OP_COLS + [f's_{i}' for i in range(1, 22)]
    hd_cols = [c for c in train_df.columns if c.startswith('hd_')]
    regime_cols = [c for c in train_df.columns if c.startswith('regime_')]

    feature_cols = raw_cols + hd_cols + regime_cols

    # Remove any missing columns
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    X = train_df[feature_cols].fillna(0).values
    y = train_df['RUL'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    return model, feature_cols, val_rmse, train_df


def evaluate_on_test(
    model,
    test_df: pd.DataFrame,
    rul_actual: np.ndarray,
    feature_cols: List[str],
    learned: LearnedRegimes = None,
    cap_rul: int = 125,
) -> Dict[str, float]:
    """Evaluate model on test set."""

    # Apply regime features if learned structure provided
    if learned is not None:
        test_df = apply_regimes(test_df, learned, verbose=False)

    # Get last cycle per unit
    last_cycles = test_df.groupby('unit_id').last().reset_index()

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in last_cycles.columns:
            last_cycles[col] = 0.0

    X_test = last_cycles[feature_cols].fillna(0).values

    y_pred = model.predict(X_test)

    # Cap actual RUL for comparison
    rul_capped = np.clip(rul_actual, 0, cap_rul)

    rmse = np.sqrt(mean_squared_error(rul_capped, y_pred))
    mae = mean_absolute_error(rul_capped, y_pred)
    r2 = r2_score(rul_capped, y_pred)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred,
        'actual': rul_capped,
    }


def show_feature_importance(model, feature_cols: List[str], top_n: int = 20):
    """Show top feature importances."""

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("  Model does not have feature_importances_")
        return

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n  Top {top_n} features:")
    print("  " + "-"*50)

    for i, idx in enumerate(indices):
        name = feature_cols[idx]
        imp = importances[idx]

        # Categorize
        if name.startswith('hd_'):
            cat = "[HD]"  # Healthy distance
        elif name.startswith('regime_'):
            cat = "[REG]"  # Regime
        elif name.startswith('op_'):
            cat = "[OP]"  # Operating condition
        elif name.startswith('s_'):
            cat = "[RAW]"  # Raw sensor
        else:
            cat = "[?]"

        print(f"  {i+1:2d}. {name:<25} {imp:.4f}  {cat}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='FD002 Regime-Aware Test')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing C-MAPSS files')
    parser.add_argument('--cap-rul', type=int, default=125,
                        help='Cap RUL at this value')
    parser.add_argument('--n-regimes', type=int, default=6,
                        help='Number of operating regimes')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Find FD002 files
    train_path = data_dir / 'train_FD002.txt'
    test_path = data_dir / 'test_FD002.txt'
    rul_path = data_dir / 'RUL_FD002.txt'

    # Also check alternative locations
    if not train_path.exists():
        for alt_dir in [data_dir / 'CMAPSSData', data_dir / 'cmapss', Path('.')]:
            if (alt_dir / 'train_FD002.txt').exists():
                train_path = alt_dir / 'train_FD002.txt'
                test_path = alt_dir / 'test_FD002.txt'
                rul_path = alt_dir / 'RUL_FD002.txt'
                break

    if not train_path.exists():
        print(f"ERROR: Could not find train_FD002.txt")
        print(f"Searched in: {data_dir}")
        print(f"\nPlease download C-MAPSS data from NASA and specify --data-dir")
        return 1

    print("="*70)
    print("FD002 REGIME-AWARE TEST")
    print("="*70)
    print(f"\nData directory: {data_dir}")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")
    print(f"RUL: {rul_path}")

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    train_df = load_cmapss(str(train_path))
    train_df = add_rul(train_df, cap=args.cap_rul)

    test_df = load_cmapss(str(test_path))
    rul_actual = load_rul_file(str(rul_path))

    print(f"\nTrain: {len(train_df):,} rows, {train_df['unit_id'].nunique()} units")
    print(f"Test: {len(test_df):,} rows, {test_df['unit_id'].nunique()} units")
    print(f"RUL values: {len(rul_actual)}")

    # =========================================================================
    # BASELINE MODEL (raw sensors only)
    # =========================================================================
    print("\n" + "="*70)
    print("BASELINE MODEL (raw sensors + operating conditions)")
    print("="*70)

    baseline_model, baseline_cols, baseline_val_rmse = train_baseline_model(train_df, args.cap_rul)
    print(f"\n  Features: {len(baseline_cols)}")
    print(f"  Validation RMSE: {baseline_val_rmse:.2f}")

    baseline_results = evaluate_on_test(
        baseline_model, test_df, rul_actual, baseline_cols, learned=None, cap_rul=args.cap_rul
    )
    print(f"\n  TEST RMSE: {baseline_results['rmse']:.2f}")
    print(f"  TEST MAE: {baseline_results['mae']:.2f}")
    print(f"  TEST R²: {baseline_results['r2']:.3f}")

    # =========================================================================
    # LEARN REGIMES
    # =========================================================================
    learned = learn_regimes(train_df, n_regimes=args.n_regimes)

    # =========================================================================
    # REGIME-AWARE MODEL
    # =========================================================================
    print("\n" + "="*70)
    print("REGIME-AWARE MODEL (+ healthy_distance features)")
    print("="*70)

    regime_model, regime_cols, regime_val_rmse, train_with_features = train_regime_model(
        train_df.copy(), learned, args.cap_rul
    )
    print(f"\n  Features: {len(regime_cols)}")
    print(f"  Validation RMSE: {regime_val_rmse:.2f}")

    regime_results = evaluate_on_test(
        regime_model, test_df, rul_actual, regime_cols, learned=learned, cap_rul=args.cap_rul
    )
    print(f"\n  TEST RMSE: {regime_results['rmse']:.2f}")
    print(f"  TEST MAE: {regime_results['mae']:.2f}")
    print(f"  TEST R²: {regime_results['r2']:.3f}")

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (Regime-Aware Model)")
    print("="*70)
    show_feature_importance(regime_model, regime_cols, top_n=25)

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    print(f"\n{'Model':<30} {'Val RMSE':<12} {'Test RMSE':<12} {'Test MAE':<12}")
    print("-"*70)
    print(f"{'Baseline (raw sensors)':<30} {baseline_val_rmse:<12.2f} {baseline_results['rmse']:<12.2f} {baseline_results['mae']:<12.2f}")
    print(f"{'Regime-Aware (+hd)':<30} {regime_val_rmse:<12.2f} {regime_results['rmse']:<12.2f} {regime_results['mae']:<12.2f}")

    improvement = (baseline_results['rmse'] - regime_results['rmse']) / baseline_results['rmse'] * 100

    if improvement > 0:
        print(f"\n✓ Regime-aware model improves Test RMSE by {improvement:.1f}%")
    else:
        print(f"\n✗ Baseline wins by {-improvement:.1f}%")

    print(f"\n{'='*70}")
    print("BENCHMARKS")
    print("="*70)
    print(f"\n  FD002 Published SOTA: 11.46 RMSE")
    print(f"  Your current FD002:   16.80 RMSE (+47% gap)")
    print(f"  This test baseline:   {baseline_results['rmse']:.2f} RMSE")
    print(f"  This test regime:     {regime_results['rmse']:.2f} RMSE")

    new_gap = (regime_results['rmse'] - 11.46) / 11.46 * 100
    print(f"\n  New gap to SOTA:      {new_gap:.1f}%")

    # =========================================================================
    # WORST PREDICTIONS
    # =========================================================================
    print("\n" + "="*70)
    print("WORST PREDICTIONS (Regime-Aware Model)")
    print("="*70)

    errors = regime_results['actual'] - regime_results['predictions']
    abs_errors = np.abs(errors)
    worst_idx = np.argsort(abs_errors)[-10:][::-1]

    print(f"\n  {'Unit':<8} {'Actual':<10} {'Predicted':<12} {'Error':<10}")
    print("  " + "-"*45)
    for idx in worst_idx:
        print(f"  {idx+1:<8} {regime_results['actual'][idx]:<10.1f} "
              f"{regime_results['predictions'][idx]:<12.1f} {errors[idx]:+.1f}")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

    return 0


if __name__ == '__main__':
    exit(main())
