"""
PRISM ML Predict — Run inference on test data using trained model.

Reads:  data/ml_model.pkl (trained model)
        data/test/ml_features.parquet (test features)
        data/test/RUL.txt (optional ground truth)
Writes: data/test/ml_predictions.parquet

Usage:
    python -m prism.entry_points.ml_predict
    python -m prism.entry_points.ml_predict --ground-truth data/test/RUL_FD001.txt
    python -m prism.entry_points.ml_predict --features path/to/ml_features.parquet
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from prism.db.parquet_store import (
    get_path,
    ML_FEATURES,
    ML_MODEL,
)


def load_ground_truth(path: str) -> np.ndarray:
    """
    Load RUL ground truth from C-MAPSS format (one value per line).
    """
    with open(path, 'r') as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return np.array(values)


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on test data using trained PRISM model'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to trained model (default: data/ml_model.pkl)'
    )
    parser.add_argument(
        '--features', type=str, default=None,
        help='Path to test ml_features.parquet (default: data/test/ml_features.parquet)'
    )
    parser.add_argument(
        '--ground-truth', type=str, default=None,
        help='Path to ground truth RUL file (optional, for evaluation)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save predictions (default: data/test/ml_predictions.parquet)'
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Resolve paths
    # -------------------------------------------------------------------------
    model_path = args.model or str(Path(get_path(ML_MODEL)).with_suffix('.pkl'))
    features_path = args.features or str(Path(get_path(ML_FEATURES)).parent / 'test' / 'ml_features.parquet')
    output_path = args.output or str(Path(features_path).parent / 'ml_predictions.parquet')

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    print("Loading model...")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run ml_train first:\n"
            "  python -m prism.entry_points.ml_train"
        )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"  Loaded: {model_path}")
    print(f"  Model type: {type(model).__name__}")

    # Load metadata if available
    metadata_path = Path(model_path).with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"  Training RMSE: {metadata.get('test_rmse', 'N/A')}")

    # -------------------------------------------------------------------------
    # Load test features
    # -------------------------------------------------------------------------
    print("\nLoading test features...")

    if not Path(features_path).exists():
        raise FileNotFoundError(
            f"Test features not found: {features_path}\n"
            "Run the pipeline on test data first, or specify --features path"
        )

    features = pl.read_parquet(features_path)
    print(f"  {len(features):,} entities, {len(features.columns)} columns")

    # -------------------------------------------------------------------------
    # Prepare features
    # -------------------------------------------------------------------------
    # Identify entity column
    entity_col = None
    for col in ['entity_id', 'engine_id', 'unit_id', 'bearing_id']:
        if col in features.columns:
            entity_col = col
            break

    if entity_col is None:
        entity_col = features.columns[0]

    # Feature columns = everything except entity and target (if present)
    exclude_cols = [entity_col, 'target']
    feature_cols = [c for c in features.columns if c not in exclude_cols]

    # Filter to only numeric columns that the model expects
    numeric_cols = [c for c in feature_cols
                    if features[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Boolean]]

    print(f"  Using {len(numeric_cols)} numeric features")

    X = features.select(numeric_cols).to_pandas()
    entity_ids = features[entity_col].to_list()

    # -------------------------------------------------------------------------
    # Run predictions
    # -------------------------------------------------------------------------
    print("\nRunning predictions...")

    predictions = model.predict(X)

    print(f"  Predicted {len(predictions)} values")
    print(f"  Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")

    # -------------------------------------------------------------------------
    # Build results DataFrame
    # -------------------------------------------------------------------------
    results = pl.DataFrame({
        entity_col: entity_ids,
        'predicted_rul': predictions,
    })

    # -------------------------------------------------------------------------
    # Evaluate against ground truth (if provided)
    # -------------------------------------------------------------------------
    if args.ground_truth:
        print(f"\nLoading ground truth: {args.ground_truth}")

        if not Path(args.ground_truth).exists():
            print(f"  Warning: Ground truth file not found")
        else:
            actual = load_ground_truth(args.ground_truth)

            if len(actual) != len(predictions):
                print(f"  Warning: Ground truth length ({len(actual)}) != predictions ({len(predictions)})")
            else:
                rmse = np.sqrt(mean_squared_error(actual, predictions))
                mae = mean_absolute_error(actual, predictions)
                r2 = r2_score(actual, predictions)

                print(f"\n" + "="*50)
                print("TEST SET EVALUATION")
                print("="*50)
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE:  {mae:.4f}")
                print(f"R²:   {r2:.4f}")

                # Add actuals to results
                results = results.with_columns(
                    pl.Series('actual_rul', actual),
                    pl.Series('error', actual - predictions),
                    pl.Series('abs_error', np.abs(actual - predictions)),
                )

                # Show worst predictions
                print(f"\nWorst Predictions (by absolute error):")
                print("-" * 50)
                worst = results.sort('abs_error', descending=True).head(5)
                for row in worst.iter_rows(named=True):
                    print(f"  {row[entity_col]}: actual={row['actual_rul']:.0f}, "
                          f"pred={row['predicted_rul']:.0f}, error={row['error']:.0f}")

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.write_parquet(output_path)
    print(f"\nPredictions saved: {output_path}")

    # Also save as CSV for easy viewing
    csv_path = Path(output_path).with_suffix('.csv')
    results.write_csv(csv_path)
    print(f"CSV saved: {csv_path}")

    print(f"\n" + "="*50)
    print("PREDICTION COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
