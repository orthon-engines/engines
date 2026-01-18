#!/usr/bin/env python3
"""
Compute PRISM features on full test unit trajectories.

For RUL prediction, we compute features on the FULL available trajectory
of each test unit, not sliding windows. This gives us one feature vector
per test unit that captures the behavioral state at cutoff.

This approach is simpler and more appropriate for RUL prediction:
- No window size constraints (handles short test units)
- Features describe the full degradation pattern visible at cutoff
- One-to-one mapping: unit -> feature vector -> RUL prediction
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Any

DATA_DIR = Path('/Users/jasonrudder/prism-mac/data')
CMAPSS_DIR = DATA_DIR / 'CMAPSSData'
PRISM_DIR = DATA_DIR / 'cmapss_fd001'

# Import PRISM engines
import sys
sys.path.insert(0, str(Path('/Users/jasonrudder/prism-mac')))

from prism.engines.realized_vol import compute_realized_vol
from prism.engines.hilbert import compute_hilbert
from prism.engines.hurst import compute_hurst
from prism.engines.entropy import compute_entropy
from prism.engines.garch import compute_garch
from prism.engines.rqa import compute_rqa
from prism.engines.spectral import compute_spectral
from prism.engines.wavelet import compute_wavelets
from prism.engines.lyapunov import compute_lyapunov

# Sensor column mapping (s1-s21)
SENSOR_COLS = [f's{i}' for i in range(1, 22)]

# Engine minimum observations
ENGINE_MIN_OBS = {
    'hurst': 20,
    'entropy': 30,
    'lyapunov': 30,
    'garch': 50,
    'spectral': 40,
    'wavelet': 40,
    'rqa': 30,
    'realized_vol': 15,
    'hilbert': 20,
}

# Core engines to run (skip conditional engines that need more data)
CORE_ENGINES = {
    'hurst': compute_hurst,
    'entropy': compute_entropy,
    'realized_vol': compute_realized_vol,
    'hilbert': compute_hilbert,
    'rqa': compute_rqa,
}

# Additional engines for longer trajectories
EXTENDED_ENGINES = {
    'garch': compute_garch,
    'spectral': compute_spectral,
    'wavelet': compute_wavelets,
    'lyapunov': compute_lyapunov,
}


def compute_trajectory_features(values: np.ndarray, engines: Dict = None) -> Dict[str, float]:
    """
    Compute PRISM features on a full trajectory.

    Args:
        values: Time series values (full trajectory)
        engines: Dict of engines to run (default: CORE_ENGINES)

    Returns:
        Dict of feature_name -> value
    """
    if engines is None:
        engines = CORE_ENGINES

    features = {}
    n = len(values)

    # Run each engine
    for engine_name, engine_func in engines.items():
        min_obs = ENGINE_MIN_OBS.get(engine_name, 15)

        if n < min_obs:
            continue

        try:
            try:
                metrics = engine_func(values, min_obs=min_obs)
            except TypeError:
                metrics = engine_func(values)

            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    try:
                        numeric_value = float(metric_value)
                        if np.isfinite(numeric_value):
                            features[f'{engine_name}_{metric_name}'] = numeric_value
                    except (TypeError, ValueError):
                        continue
        except Exception as e:
            # Skip failed engines
            continue

    return features


def compute_unit_features(unit_df: pl.DataFrame, use_extended: bool = False) -> Dict[str, float]:
    """
    Compute PRISM features for a single unit across all sensors.

    Args:
        unit_df: DataFrame with sensor columns (s1-s21)
        use_extended: Whether to use extended engines (for longer trajectories)

    Returns:
        Dict of feature_name -> value (aggregated across sensors)
    """
    n = len(unit_df)
    all_features = {}

    # Select engines based on trajectory length
    engines = CORE_ENGINES.copy()
    if use_extended and n >= 50:
        engines.update(EXTENDED_ENGINES)

    # Compute features for each sensor
    sensor_features = []
    for sensor in SENSOR_COLS:
        if sensor not in unit_df.columns:
            continue

        values = unit_df[sensor].to_numpy()

        # Clean NaN values
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() < 15:
            continue

        clean_values = values[valid_mask]
        features = compute_trajectory_features(clean_values, engines)

        if features:
            sensor_features.append(features)

    if not sensor_features:
        return {}

    # Aggregate across sensors: mean of each feature
    all_feature_names = set()
    for sf in sensor_features:
        all_feature_names.update(sf.keys())

    for feature_name in all_feature_names:
        values = [sf.get(feature_name) for sf in sensor_features if feature_name in sf]
        if values:
            all_features[feature_name] = np.mean(values)
            all_features[f'{feature_name}_std'] = np.std(values)

    return all_features


def compute_all_unit_features(split: str = 'test') -> pl.DataFrame:
    """
    Compute PRISM features for all units in a split.

    Args:
        split: 'train' or 'test'

    Returns:
        DataFrame with unit, features columns
    """
    # Load data
    if split == 'test':
        df = pl.read_parquet(CMAPSS_DIR / 'test_FD001.parquet')
    else:
        df = pl.read_parquet(CMAPSS_DIR / 'train_FD001.parquet')

    units = df['unit'].unique().sort().to_list()
    print(f"Processing {len(units)} {split} units...")

    # Compute features for each unit
    rows = []
    for unit in units:
        unit_df = df.filter(pl.col('unit') == unit)
        n_cycles = len(unit_df)

        # Use extended engines for longer trajectories
        use_extended = n_cycles >= 50

        features = compute_unit_features(unit_df, use_extended=use_extended)

        if features:
            row = {'unit': unit, 'n_cycles': n_cycles}
            row.update(features)
            rows.append(row)
            print(f"  Unit {unit}: {n_cycles} cycles, {len(features)} features")
        else:
            print(f"  Unit {unit}: {n_cycles} cycles - SKIPPED (no features)")

    # Create DataFrame
    result = pl.DataFrame(rows, infer_schema_length=None)
    print(f"\nComputed features for {len(result)} units")
    print(f"Total features: {len(result.columns) - 2}")

    return result


def load_rul_labels() -> pl.DataFrame:
    """Load RUL ground truth labels for test set."""
    rul_path = CMAPSS_DIR / 'RUL_FD001.txt'

    with open(rul_path) as f:
        rul_values = [int(line.strip()) for line in f]

    # Create DataFrame with unit numbers (1-indexed)
    return pl.DataFrame({
        'unit': list(range(1, len(rul_values) + 1)),
        'rul_actual': rul_values,
    })


def main():
    """Compute features for train and test sets."""
    output_dir = Path('/Users/jasonrudder/prism-mac/notebooks/ml_accelerator')

    print("=" * 80)
    print("PRISM FULL-TRAJECTORY FEATURE COMPUTATION")
    print("=" * 80)
    print()

    # Compute train features
    print("\n=== TRAIN SET ===")
    train_features = compute_all_unit_features('train')
    train_path = output_dir / 'train_trajectory_features.parquet'
    train_features.write_parquet(train_path)
    print(f"Saved to {train_path}")

    # Compute test features
    print("\n=== TEST SET ===")
    test_features = compute_all_unit_features('test')
    test_path = output_dir / 'test_trajectory_features.parquet'
    test_features.write_parquet(test_path)
    print(f"Saved to {test_path}")

    # Show feature overlap
    train_cols = set(train_features.columns) - {'unit', 'n_cycles'}
    test_cols = set(test_features.columns) - {'unit', 'n_cycles'}
    common_cols = train_cols & test_cols

    print(f"\n=== FEATURE SUMMARY ===")
    print(f"Train features: {len(train_cols)}")
    print(f"Test features: {len(test_cols)}")
    print(f"Common features: {len(common_cols)}")

    # Show sample features
    print("\n=== SAMPLE FEATURES ===")
    for col in sorted(common_cols)[:10]:
        train_val = train_features[col].mean()
        test_val = test_features[col].mean()
        print(f"  {col}: train={train_val:.4f}, test={test_val:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
