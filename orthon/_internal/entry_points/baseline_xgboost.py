"""
Baseline XGBoost on raw C-MAPSS data.

No PRISM features - just raw sensors + operational settings.
This establishes the baseline to beat.

Usage:
    python baseline_xgboost.py
    python baseline_xgboost.py --train data/train_FD001.txt --test data/test_FD001.txt --rul data/RUL_FD001.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


# C-MAPSS column names
COLUMNS = ['unit_id', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]


def load_cmapss(path: str) -> pd.DataFrame:
    """Load C-MAPSS txt file."""
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add RUL column: max_cycle - current_cycle per unit."""
    max_cycles = df.groupby('unit_id')['cycle'].max().rename('max_cycle')
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    return df


def load_rul_file(path: str) -> np.ndarray:
    """Load ground truth RUL file (one value per line)."""
    with open(path, 'r') as f:
        return np.array([float(line.strip()) for line in f if line.strip()])


def main():
    parser = argparse.ArgumentParser(description='Baseline XGBoost on raw C-MAPSS')
    parser.add_argument('--train', type=str, default='data/train_FD001.txt')
    parser.add_argument('--test', type=str, default='data/test_FD001.txt')
    parser.add_argument('--rul', type=str, default='data/RUL_FD001.txt')
    parser.add_argument('--cap-rul', type=int, default=125,
                        help='Cap RUL at this value (common practice, 0 to disable)')
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load training data
    # -------------------------------------------------------------------------
    print("Loading training data...")
    train_df = load_cmapss(args.train)
    train_df = add_rul(train_df)

    print(f"  {len(train_df):,} rows, {train_df['unit_id'].nunique()} units")

    # Cap RUL (common practice - early cycles don't degrade much)
    if args.cap_rul > 0:
        train_df['RUL'] = train_df['RUL'].clip(upper=args.cap_rul)
        print(f"  RUL capped at {args.cap_rul}")

    # -------------------------------------------------------------------------
    # Prepare features
    # -------------------------------------------------------------------------
    feature_cols = [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]

    X_train_full = train_df[feature_cols].values
    y_train_full = train_df['RUL'].values

    # Hold out 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print(f"Features: {len(feature_cols)} (3 op settings + 21 sensors)")

    # -------------------------------------------------------------------------
    # Train XGBoost
    # -------------------------------------------------------------------------
    print("\nTraining XGBoost...")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # Validation metrics
    # -------------------------------------------------------------------------
    y_val_pred = model.predict(X_val)

    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n" + "="*50)
    print("VALIDATION RESULTS (on held-out train data)")
    print("="*50)
    print(f"RMSE: {val_rmse:.4f}")
    print(f"MAE:  {val_mae:.4f}")
    print(f"R²:   {val_r2:.4f}")

    # -------------------------------------------------------------------------
    # Test set evaluation
    # -------------------------------------------------------------------------
    if Path(args.test).exists() and Path(args.rul).exists():
        print(f"\nLoading test data...")
        test_df = load_cmapss(args.test)
        rul_actual = load_rul_file(args.rul)

        print(f"  {len(test_df):,} rows, {test_df['unit_id'].nunique()} units")

        # For test, we predict RUL at the LAST cycle of each unit
        last_cycles = test_df.groupby('unit_id').last().reset_index()
        X_test = last_cycles[feature_cols].values

        if len(X_test) != len(rul_actual):
            print(f"  Warning: test units ({len(X_test)}) != RUL values ({len(rul_actual)})")
        else:
            y_test_pred = model.predict(X_test)

            # Cap predictions to match training
            if args.cap_rul > 0:
                rul_actual_capped = np.clip(rul_actual, 0, args.cap_rul)
            else:
                rul_actual_capped = rul_actual

            test_rmse = np.sqrt(mean_squared_error(rul_actual_capped, y_test_pred))
            test_mae = mean_absolute_error(rul_actual_capped, y_test_pred)
            test_r2 = r2_score(rul_actual_capped, y_test_pred)

            print(f"\n" + "="*50)
            print("TEST RESULTS (actual held-out test set)")
            print("="*50)
            print(f"RMSE: {test_rmse:.4f}")
            print(f"MAE:  {test_mae:.4f}")
            print(f"R²:   {test_r2:.4f}")

            # Show some predictions
            print(f"\nSample predictions (first 10 units):")
            print("-" * 40)
            for i in range(min(10, len(y_test_pred))):
                err = rul_actual_capped[i] - y_test_pred[i]
                print(f"  Unit {i+1:3d}: actual={rul_actual_capped[i]:6.1f}, "
                      f"pred={y_test_pred[i]:6.1f}, error={err:+6.1f}")

            # Worst predictions
            abs_errors = np.abs(rul_actual_capped - y_test_pred)
            worst_idx = np.argsort(abs_errors)[-5:][::-1]

            print(f"\nWorst predictions:")
            print("-" * 40)
            for i in worst_idx:
                err = rul_actual_capped[i] - y_test_pred[i]
                print(f"  Unit {i+1:3d}: actual={rul_actual_capped[i]:6.1f}, "
                      f"pred={y_test_pred[i]:6.1f}, error={err:+6.1f}")
    else:
        print(f"\nTest files not found, skipping test evaluation")
        print(f"  Expected: {args.test}")
        print(f"  Expected: {args.rul}")

    print(f"\n" + "="*50)
    print("BASELINE COMPLETE")
    print("="*50)
    print("\nThis is your baseline to beat with PRISM features.")


if __name__ == "__main__":
    main()
