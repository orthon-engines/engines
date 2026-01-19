"""
PRISM ML Train — Train ML model on PRISM features.

Supports multiple frameworks: XGBoost, CatBoost, LightGBM, RandomForest, GradientBoosting.
User picks the framework, PRISM provides the features.

Reads:  data/ml_features.parquet
Writes: data/ml_model.pkl, data/ml_results.parquet, data/ml_importance.parquet

Usage:
    python -m prism.entry_points.ml_train
    python -m prism.entry_points.ml_train --model catboost
    python -m prism.entry_points.ml_train --model lightgbm
    python -m prism.entry_points.ml_train --tune
    python -m prism.entry_points.ml_train --testing
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from prism.db.parquet_store import (
    get_path,
    ML_FEATURES,
    ML_RESULTS,
    ML_IMPORTANCE,
    ML_MODEL,
)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODELS = {
    'xgboost': {
        'class': 'xgboost.XGBRegressor',
        'params': {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        },
        'tune_params': {
            'n_estimators': [100, 300, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
        },
    },
    'catboost': {
        'class': 'catboost.CatBoostRegressor',
        'params': {
            'iterations': 300,
            'depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': False,
        },
        'tune_params': {
            'iterations': [100, 300, 500],
            'depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
        },
    },
    'lightgbm': {
        'class': 'lightgbm.LGBMRegressor',
        'params': {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
        },
        'tune_params': {
            'n_estimators': [100, 300, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
        },
    },
    'randomforest': {
        'class': 'sklearn.ensemble.RandomForestRegressor',
        'params': {
            'n_estimators': 300,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1,
        },
        'tune_params': {
            'n_estimators': [100, 300, 500],
            'max_depth': [6, 10, 15],
            'min_samples_split': [2, 5, 10],
        },
    },
    'gradientboosting': {
        'class': 'sklearn.ensemble.GradientBoostingRegressor',
        'params': {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42,
        },
        'tune_params': {
            'n_estimators': [100, 300, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
        },
    },
}


def get_model(name: str):
    """
    Get model instance by name.
    
    Dynamically imports the model class to avoid requiring all dependencies.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(MODELS.keys())}")
    
    config = MODELS[name]
    class_path = config['class']
    params = config['params']
    
    # Dynamic import
    if name == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    
    elif name == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params)
    
    elif name == 'lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)
    
    elif name == 'randomforest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    
    elif name == 'gradientboosting':
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**params)
    
    else:
        raise ValueError(f"Model {name} not implemented")


def tune_model(name: str, X_train, y_train):
    """
    Run hyperparameter tuning with GridSearchCV.
    """
    from sklearn.model_selection import GridSearchCV
    
    config = MODELS[name]
    base_model = get_model(name)
    tune_params = config['tune_params']
    
    print(f"  Tuning {name} with {len(tune_params)} parameters...")
    
    grid_search = GridSearchCV(
        base_model,
        tune_params,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1,
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best RMSE: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def compute_feature_importance(
    model, 
    feature_names: list
) -> pl.DataFrame:
    """
    Extract feature importance from trained model.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return None
    
    importance_df = pl.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort('importance', descending=True)
    
    # Add cumulative importance
    total = importance_df['importance'].sum()
    importance_df = importance_df.with_columns(
        (pl.col('importance') / total * 100).alias('importance_pct'),
        (pl.col('importance').cum_sum() / total * 100).alias('cumulative_pct'),
    )
    
    return importance_df


def print_feature_importance(importance_df: pl.DataFrame, top_n: int = 15):
    """
    Pretty print feature importance.
    """
    print(f"\nTop {top_n} Features:")
    print("-" * 60)
    
    for row in importance_df.head(top_n).iter_rows(named=True):
        bar_len = int(row['importance_pct'] / 2)
        bar = "█" * bar_len
        print(f"  {row['feature'][:35]:<35} {row['importance_pct']:>5.1f}% {bar}")
    
    # Group by feature type
    print(f"\nFeature Group Importance:")
    print("-" * 40)
    
    groups = {}
    for row in importance_df.iter_rows(named=True):
        # Extract group from feature name (vector_, geometry_, state_)
        parts = row['feature'].split('_')
        if len(parts) >= 2:
            group = parts[0]
        else:
            group = 'other'
        
        groups[group] = groups.get(group, 0) + row['importance_pct']
    
    for group, pct in sorted(groups.items(), key=lambda x: -x[1]):
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"  {group:<15} {pct:>5.1f}% {bar}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train ML model on PRISM features'
    )
    parser.add_argument(
        '--model', choices=list(MODELS.keys()), default='xgboost',
        help='ML framework to use (default: xgboost)'
    )
    parser.add_argument(
        '--tune', action='store_true',
        help='Run hyperparameter tuning (slower, potentially better)'
    )
    parser.add_argument(
        '--split', type=float, default=0.8,
        help='Train/test split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--cv', type=int, default=None,
        help='Run N-fold cross-validation instead of train/test split'
    )
    parser.add_argument(
        '--testing', action='store_true',
        help='Enable test mode (uses smaller subset)'
    )
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # Load features
    # -------------------------------------------------------------------------
    print("Loading features...")
    
    features_path = get_path(ML_FEATURES)
    if not Path(features_path).exists():
        raise FileNotFoundError(
            "ml_features.parquet not found. Run ml_features first:\n"
            "  python -m prism.entry_points.ml_features --target RUL"
        )
    
    features = pl.read_parquet(features_path)
    print(f"  {len(features):,} entities, {len(features.columns)} columns")
    
    # -------------------------------------------------------------------------
    # Validate target exists
    # -------------------------------------------------------------------------
    if 'target' not in features.columns:
        raise ValueError(
            "No target column found. Run ml_features with --target:\n"
            "  python -m prism.entry_points.ml_features --target RUL"
        )
    
    # -------------------------------------------------------------------------
    # Prepare X and y
    # -------------------------------------------------------------------------
    # Identify entity column
    entity_col = None
    for col in ['entity_id', 'engine_id', 'unit_id', 'bearing_id']:
        if col in features.columns:
            entity_col = col
            break
    
    if entity_col is None:
        entity_col = features.columns[0]  # Assume first column is entity
    
    # Feature columns = everything except entity and target
    feature_cols = [c for c in features.columns if c not in [entity_col, 'target']]
    
    X = features.select(feature_cols).to_pandas()
    y = features['target'].to_pandas()
    
    print(f"\nDataset:")
    print(f"  Entities: {len(X):,}")
    print(f"  Features: {len(feature_cols):,}")
    print(f"  Target range: {y.min():.2f} - {y.max():.2f}")
    
    # -------------------------------------------------------------------------
    # Testing mode: use subset
    # -------------------------------------------------------------------------
    if args.testing:
        n_test = min(100, len(X))
        X = X.head(n_test)
        y = y.head(n_test)
        print(f"\n[TESTING] Using {n_test} samples")
    
    # -------------------------------------------------------------------------
    # Cross-validation mode
    # -------------------------------------------------------------------------
    if args.cv:
        print(f"\nRunning {args.cv}-fold cross-validation...")
        
        model = get_model(args.model)
        scores = cross_val_score(
            model, X, y, 
            cv=args.cv, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        rmse_scores = -scores
        print(f"\nCross-Validation Results ({args.model}):")
        print(f"  RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
        print(f"  Folds: {rmse_scores}")
        
        # Still train final model on all data
        print(f"\nTraining final model on all data...")
        model.fit(X, y)
        
    else:
        # -------------------------------------------------------------------------
        # Train/test split
        # -------------------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=args.split, random_state=42
        )
        
        print(f"\nTrain/Test Split:")
        print(f"  Train: {len(X_train):,} samples")
        print(f"  Test:  {len(X_test):,} samples")
        
        # -------------------------------------------------------------------------
        # Train model
        # -------------------------------------------------------------------------
        print(f"\nTraining {args.model}...")
        
        if args.tune:
            model = tune_model(args.model, X_train, y_train)
        else:
            model = get_model(args.model)
            model.fit(X_train, y_train)
        
        # -------------------------------------------------------------------------
        # Evaluate
        # -------------------------------------------------------------------------
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n" + "="*50)
        print(f"RESULTS: {args.model.upper()}")
        print(f"="*50)
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE:  {test_rmse:.4f}")
        print(f"Test MAE:   {test_mae:.4f}")
        print(f"Test R²:    {test_r2:.4f}")
        
        # Check for overfitting
        overfit_ratio = train_rmse / test_rmse if test_rmse > 0 else 0
        if overfit_ratio < 0.7:
            print(f"\n⚠️  Warning: Possible overfitting (train/test ratio: {overfit_ratio:.2f})")
        
        # -------------------------------------------------------------------------
        # Save predictions
        # -------------------------------------------------------------------------
        # Get entity IDs for test set
        test_indices = X_test.index.tolist()
        test_entities = features[entity_col].to_list()
        test_entity_ids = [test_entities[i] for i in test_indices]
        
        results = pl.DataFrame({
            entity_col: test_entity_ids,
            'actual': y_test.values,
            'predicted': y_pred_test,
            'error': y_test.values - y_pred_test,
            'abs_error': np.abs(y_test.values - y_pred_test),
        })
        
        results_path = get_path(ML_RESULTS)
        results.write_parquet(results_path)
        print(f"\nPredictions saved: {results_path}")
    
    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------
    importance_df = compute_feature_importance(model, feature_cols)
    
    if importance_df is not None:
        print_feature_importance(importance_df)
        
        importance_path = get_path(ML_IMPORTANCE)
        importance_df.write_parquet(importance_path)
        print(f"\nFeature importance saved: {importance_path}")
    
    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------
    model_path = Path(get_path(ML_MODEL)).with_suffix('.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved: {model_path}")
    
    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------
    metadata = {
        'model': args.model,
        'n_features': len(feature_cols),
        'n_entities': len(features),
        'tuned': args.tune,
    }
    
    if not args.cv:
        metadata.update({
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2),
        })
    
    metadata_path = Path(get_path(ML_MODEL)).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n" + "="*50)
    print("ML TRAINING COMPLETE")
    print(f"="*50)


if __name__ == "__main__":
    main()
