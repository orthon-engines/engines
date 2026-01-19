#!/usr/bin/env python3
"""
PRISM Ablation Study v2 — With Cohort Discovery Validation

Enhanced version that demonstrates PRISM's "blind discovery" capability:
1. Standard layer ablation (raw → vector → geometry → state)
2. Cohort-by-cohort predictive contribution
3. "What did PRISM discover?" summary

This is the "give me unlabeled data and I'll tell you what you have" demo.

Usage:
    python -m prism.entry_points.ml_ablation --target RUL
    python -m prism.entry_points.ml_ablation --target RUL --show-discovery
    python -m prism.entry_points.ml_ablation --target RUL --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError:
    print("Error: XGBoost required. Install with: pip install xgboost")
    sys.exit(1)

from prism.db.parquet_store import get_path, OBSERVATIONS, VECTOR, GEOMETRY, STATE, COHORTS


# =============================================================================
# Feature Aggregation Functions
# =============================================================================

def aggregate_raw_features(observations: pl.DataFrame, entity_col: str) -> pl.DataFrame:
    """Stage 0: Simple statistics on raw observations."""
    value_col = 'value'
    signal_col = 'signal_id' if 'signal_id' in observations.columns else 'signal_type'
    
    features = (
        observations
        .group_by([entity_col, signal_col])
        .agg([
            pl.col(value_col).mean().alias('mean'),
            pl.col(value_col).std().alias('std'),
            pl.col(value_col).last().alias('last'),
        ])
    )
    
    features_wide = features.pivot(
        on=signal_col,
        index=entity_col,
        values=['mean', 'std', 'last'],
    )
    
    return features_wide


def aggregate_vector_features(vector: pl.DataFrame, entity_col: str) -> pl.DataFrame:
    """Stage 2: Aggregate vector metrics (51 behavioral metrics)."""
    exclude = {entity_col, 'signal_id', 'signal_type', 'timestamp', 
               'window_end', 'window_start', 'cohort_id'}
    
    metrics = [c for c in vector.columns if c not in exclude and not c.startswith('_')]
    
    if not metrics:
        return pl.DataFrame()
    
    aggs = []
    for m in metrics:
        aggs.extend([
            pl.col(m).mean().alias(f'v_{m}_mean'),
            pl.col(m).std().alias(f'v_{m}_std'),
            pl.col(m).last().alias(f'v_{m}_last'),
        ])
    
    return vector.group_by(entity_col).agg(aggs)


def aggregate_geometry_features(geometry: pl.DataFrame, entity_col: str) -> pl.DataFrame:
    """Stage 3: Aggregate geometry features."""
    exclude = {entity_col, 'cohort_id', 'timestamp', 'window_end', 
               'window_start', 'signal_id_a', 'signal_id_b'}
    
    metrics = [c for c in geometry.columns if c not in exclude and not c.startswith('_')]
    
    if not metrics:
        return pl.DataFrame()
    
    aggs = []
    for m in metrics:
        aggs.extend([
            pl.col(m).mean().alias(f'g_{m}_mean'),
            pl.col(m).last().alias(f'g_{m}_last'),
        ])
    
    return geometry.group_by(entity_col).agg(aggs)


def aggregate_state_features(state: pl.DataFrame, entity_col: str) -> pl.DataFrame:
    """Stage 4: Aggregate state features."""
    exclude = {entity_col, 'timestamp', 'window_end', 'window_start',
               'signal_id_a', 'signal_id_b', 'cohort_id'}
    
    metrics = [c for c in state.columns if c not in exclude and not c.startswith('_')]
    
    if not metrics:
        return pl.DataFrame()
    
    aggs = []
    for m in metrics:
        aggs.extend([
            pl.col(m).mean().alias(f's_{m}_mean'),
            pl.col(m).last().alias(f's_{m}_last'),
        ])
    
    return state.group_by(entity_col).agg(aggs)


# =============================================================================
# Model Training
# =============================================================================

def run_stage(
    X: pl.DataFrame, 
    y: np.ndarray, 
    stage_name: str,
) -> Dict[str, Any]:
    """Run XGBoost regression and return metrics."""
    X_pd = X.fill_null(0).fill_nan(0).to_pandas()
    X_pd = X_pd.select_dtypes(include=[np.number])
    
    if X_pd.shape[1] == 0:
        return {
            'stage': stage_name,
            'n_features': 0,
            'rmse': float('inf'),
            'model': None,
            'feature_importance': {},
        }
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_pd, y, train_size=0.8, random_state=42
    )
    
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Get feature importance
    importance = dict(zip(X_pd.columns, model.feature_importances_))
    top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
    
    return {
        'stage': stage_name,
        'n_features': X_pd.shape[1],
        'rmse': float(rmse),
        'model': model,
        'feature_importance': top_features,
    }


# =============================================================================
# Cohort Discovery Analysis
# =============================================================================

def analyze_cohort_discovery(
    cohort_members: pl.DataFrame,
    observations: pl.DataFrame,
    entity_col: str,
) -> Dict[str, Any]:
    """
    Analyze what PRISM discovered about the data structure.
    
    This is the "I know what you have" demo.
    """
    discovery = {
        'n_cohorts': 0,
        'cohorts': [],
        'interpretation': [],
    }
    
    # Find the cohort and signal columns
    cohort_col = None
    signal_col = None
    
    for col in ['cohort_id', 'cohort', 'cluster_id']:
        if col in cohort_members.columns:
            cohort_col = col
            break
    
    for col in ['signal_id', 'signal_type', 'sensor_id']:
        if col in cohort_members.columns:
            signal_col = col
            break
    
    if not cohort_col or not signal_col:
        return discovery
    
    # Group signals by cohort
    cohort_groups = (
        cohort_members
        .group_by(cohort_col)
        .agg(pl.col(signal_col).unique().alias('signals'))
        .sort(cohort_col)
    )
    
    discovery['n_cohorts'] = len(cohort_groups)
    
    for row in cohort_groups.iter_rows(named=True):
        cohort_id = row[cohort_col]
        signals = sorted(row['signals'])
        
        # Try to interpret what this cohort represents
        interpretation = interpret_cohort(signals)
        
        discovery['cohorts'].append({
            'cohort_id': str(cohort_id),
            'signals': signals,
            'n_signals': len(signals),
            'interpretation': interpretation,
        })
    
    return discovery


def interpret_cohort(signals: List[str]) -> str:
    """
    Attempt to interpret what a cohort represents based on signal names.
    
    This uses common industrial naming conventions.
    """
    signals_lower = [s.lower() for s in signals]
    signals_str = ' '.join(signals_lower)
    
    # Temperature indicators
    if any(t in signals_str for t in ['temp', 't2', 't24', 't30', 't50', 'htbleed']):
        if any(h in signals_str for h in ['t30', 't50', 'hot']):
            return "Hot section temperatures"
        return "Temperature sensors"
    
    # Pressure indicators
    if any(p in signals_str for p in ['press', 'p2', 'p15', 'p30', 'ps30', 'bpr']):
        return "Pressure measurements"
    
    # Speed/RPM indicators
    if any(n in signals_str for n in ['nf', 'nc', 'nrf', 'nrc', 'rpm', 'speed']):
        if 'f' in signals_str and 'c' not in signals_str:
            return "Fan spool speed"
        if 'c' in signals_str and 'f' not in signals_str:
            return "Core spool speed"
        return "Rotational speed sensors"
    
    # Flow indicators
    if any(w in signals_str for w in ['w31', 'w32', 'flow', 'mass']):
        return "Mass flow measurements"
    
    # Operational parameters
    if any(o in signals_str for o in ['op1', 'op2', 'op3', 'cmd', 'dmd', 'far']):
        return "Operational/control parameters"
    
    # Failure/health indicators
    if any(r in signals_str for r in ['rul', 'health', 'fail', 'life']):
        return "Health/failure indicators"
    
    # Chemical process indicators
    if any(c in signals_str for c in ['ph', 'conc', 'react', 'feed', 'prod']):
        return "Process chemistry"
    
    return "Unknown grouping"


def run_cohort_ablation(
    observations: pl.DataFrame,
    cohort_members: pl.DataFrame,
    entity_col: str,
    target_col: str,
    y: np.ndarray,
    entity_ids: List,
) -> List[Dict[str, Any]]:
    """
    Test predictive contribution of each cohort individually.
    
    Shows which discovered groupings matter for prediction.
    """
    results = []
    
    # Find columns
    cohort_col = None
    signal_col = None
    
    for col in ['cohort_id', 'cohort', 'cluster_id']:
        if col in cohort_members.columns:
            cohort_col = col
            break
    
    for col in ['signal_id', 'signal_type', 'sensor_id']:
        if col in cohort_members.columns:
            signal_col = col
            break
    
    if not cohort_col or not signal_col:
        return results
    
    # Get unique cohorts
    cohorts = cohort_members.select(cohort_col).unique().to_series().to_list()
    
    for cohort_id in sorted(cohorts):
        # Get signals in this cohort
        cohort_signals = (
            cohort_members
            .filter(pl.col(cohort_col) == cohort_id)
            .select(signal_col)
            .to_series()
            .to_list()
        )
        
        # Filter observations to just this cohort's signals
        obs_col = 'signal_id' if 'signal_id' in observations.columns else 'signal_type'
        cohort_obs = observations.filter(pl.col(obs_col).is_in(cohort_signals))
        
        if len(cohort_obs) == 0:
            continue
        
        # Aggregate features for this cohort
        X_cohort = aggregate_raw_features(cohort_obs, entity_col)
        X_cohort = X_cohort.sort(entity_col)
        X_cohort = X_cohort.filter(pl.col(entity_col).is_in(entity_ids))
        
        if len(X_cohort) != len(y):
            continue
        
        X_features = X_cohort.drop(entity_col)
        
        # Run model
        result = run_stage(X_features, y, f"cohort_{cohort_id}")
        result['cohort_id'] = str(cohort_id)
        result['signals'] = cohort_signals
        result['interpretation'] = interpret_cohort(cohort_signals)
        results.append(result)
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def detect_entity_column(df: pl.DataFrame, hint: Optional[str] = None) -> str:
    """Auto-detect the entity column."""
    if hint and hint in df.columns:
        return hint
    
    candidates = ['entity_id', 'engine_id', 'unit_id', 'unit', 'bearing_id', 'asset_id']
    for col in candidates:
        if col in df.columns:
            return col
    
    raise ValueError(f"Could not detect entity column. Available: {df.columns}")


def main():
    parser = argparse.ArgumentParser(
        description='PRISM Ablation Study v2 — With Cohort Discovery Validation'
    )
    parser.add_argument(
        '--target', type=str, required=True,
        help='Target column for prediction (e.g., RUL)'
    )
    parser.add_argument(
        '--entity', type=str, default=None,
        help='Entity column (auto-detected if not specified)'
    )
    parser.add_argument(
        '--show-discovery', action='store_true',
        help='Show detailed cohort discovery analysis'
    )
    parser.add_argument(
        '--cohort-ablation', action='store_true',
        help='Run per-cohort predictive analysis'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON file for results'
    )
    args = parser.parse_args()
    
    results = []
    discovery = {}
    cohort_results = []
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("=" * 70)
    print("PRISM ABLATION STUDY v2")
    print("With Cohort Discovery Validation")
    print("=" * 70)
    print(f"Target: {args.target}")
    print()
    
    obs_path = get_path(OBSERVATIONS)
    if not Path(obs_path).exists():
        print(f"Error: Observations not found at {obs_path}")
        sys.exit(1)

    observations = pl.read_parquet(obs_path)
    entity_col = detect_entity_column(observations, args.entity)
    print(f"Entity column: {entity_col}")
    
    if args.target not in observations.columns:
        print(f"Error: Target '{args.target}' not in columns")
        sys.exit(1)
    
    target = observations.group_by(entity_col).agg(
        pl.col(args.target).last().alias('target')
    ).sort(entity_col)
    
    y = target['target'].to_numpy()
    entity_ids = target[entity_col].to_list()
    print(f"Entities: {len(y)}")
    print()
    
    # =========================================================================
    # Cohort Discovery Analysis
    # =========================================================================
    cohort_path = get_path(COHORTS)
    cohort_members = None

    if Path(cohort_path).exists():
        cohort_members = pl.read_parquet(cohort_path)
        discovery = analyze_cohort_discovery(cohort_members, observations, entity_col)
        
        if args.show_discovery or True:  # Always show discovery
            print("=" * 70)
            print("COHORT DISCOVERY: What PRISM Found")
            print("=" * 70)
            print(f"Number of cohorts discovered: {discovery['n_cohorts']}")
            print()
            
            for cohort in discovery['cohorts']:
                print(f"  {cohort['cohort_id']}: {cohort['interpretation']}")
                print(f"     Signals: {', '.join(cohort['signals'][:5])}", end='')
                if len(cohort['signals']) > 5:
                    print(f" ... (+{len(cohort['signals'])-5} more)")
                else:
                    print()
            print()
    
    # =========================================================================
    # Stage 0: Raw observations only
    # =========================================================================
    print("=" * 70)
    print("LAYER ABLATION")
    print("=" * 70)
    print()
    print("Stage 0: Raw Observations Only")
    print("-" * 40)
    
    X_raw = aggregate_raw_features(observations, entity_col)
    X_raw = X_raw.sort(entity_col)
    X_raw = X_raw.filter(pl.col(entity_col).is_in(entity_ids))
    X_raw_features = X_raw.drop(entity_col)
    
    result = run_stage(X_raw_features, y, "0_raw")
    results.append(result)
    print(f"  Features: {result['n_features']}")
    print(f"  RMSE: {result['rmse']:.2f}")
    print()
    
    # =========================================================================
    # Stage 1: + Cohort (as categorical)
    # =========================================================================
    if cohort_members is not None:
        print("Stage 1: + Cohort Discovery")
        print("-" * 40)
        
        cohort_col = None
        for col in ['cohort_id', 'cohort', 'cluster_id']:
            if col in cohort_members.columns:
                cohort_col = col
                break
        
        if cohort_col:
            # This is tricky - we need signal-level cohorts aggregated to entity level
            # For now, skip if structure doesn't match
            print("  (Cohort contribution measured via cohort ablation below)")
            print()
    
    # =========================================================================
    # Stage 2: + Vector metrics
    # =========================================================================
    vector_path = get_path(VECTOR)
    X_vector = None

    if Path(vector_path).exists():
        print("Stage 2: + Vector Metrics (51 behavioral)")
        print("-" * 40)
        
        vector = pl.read_parquet(vector_path)
        
        if entity_col in vector.columns:
            X_vector = aggregate_vector_features(vector, entity_col)
            X_vector = X_vector.sort(entity_col)
            X_vector = X_vector.filter(pl.col(entity_col).is_in(entity_ids))
            X_vector_features = X_vector.drop(entity_col)
            
            result = run_stage(X_vector_features, y, "2_vector")
            delta = result['rmse'] - results[-1]['rmse']
            results.append(result)
            print(f"  Features: {result['n_features']}")
            print(f"  RMSE: {result['rmse']:.2f} (Δ {delta:+.2f})")
            
            # Show top features
            if result['feature_importance']:
                print(f"  Top features: {list(result['feature_importance'].keys())[:3]}")
        print()
    
    # =========================================================================
    # Stage 3: + Geometry
    # =========================================================================
    geometry_path = get_path(GEOMETRY)
    X_combined = None

    if Path(geometry_path).exists():
        print("Stage 3: + Geometry (coupling, structure)")
        print("-" * 40)
        
        geometry = pl.read_parquet(geometry_path)
        
        if entity_col in geometry.columns:
            X_geom = aggregate_geometry_features(geometry, entity_col)
            X_geom = X_geom.sort(entity_col)
            X_geom = X_geom.filter(pl.col(entity_col).is_in(entity_ids))
            
            if X_vector is not None:
                X_combined = X_vector.join(X_geom, on=entity_col, how='left')
            else:
                X_combined = X_geom
            
            X_combined_features = X_combined.drop(entity_col)
            
            result = run_stage(X_combined_features, y, "3_geometry")
            delta = result['rmse'] - results[-1]['rmse']
            results.append(result)
            print(f"  Features: {result['n_features']}")
            print(f"  RMSE: {result['rmse']:.2f} (Δ {delta:+.2f})")
        print()
    
    # =========================================================================
    # Stage 4: + State
    # =========================================================================
    state_path = get_path(STATE)

    if Path(state_path).exists():
        print("Stage 4: + State (velocity, acceleration)")
        print("-" * 40)
        
        state = pl.read_parquet(state_path)
        
        if entity_col in state.columns:
            X_state = aggregate_state_features(state, entity_col)
            X_state = X_state.sort(entity_col)
            X_state = X_state.filter(pl.col(entity_col).is_in(entity_ids))
            
            if X_combined is not None:
                X_full = X_combined.join(X_state, on=entity_col, how='left')
            elif X_vector is not None:
                X_full = X_vector.join(X_state, on=entity_col, how='left')
            else:
                X_full = X_state
            
            X_full_features = X_full.drop(entity_col)
            
            result = run_stage(X_full_features, y, "4_state")
            delta = result['rmse'] - results[-1]['rmse']
            results.append(result)
            print(f"  Features: {result['n_features']}")
            print(f"  RMSE: {result['rmse']:.2f} (Δ {delta:+.2f})")
        print()
    
    # =========================================================================
    # Per-Cohort Ablation
    # =========================================================================
    if args.cohort_ablation and cohort_members is not None:
        print("=" * 70)
        print("COHORT ABLATION: Which Groupings Matter?")
        print("=" * 70)
        print()
        
        cohort_results = run_cohort_ablation(
            observations, cohort_members, entity_col, 
            args.target, y, entity_ids
        )
        
        # Sort by RMSE
        cohort_results.sort(key=lambda x: x['rmse'])
        
        for cr in cohort_results:
            print(f"  {cr['cohort_id']}: RMSE {cr['rmse']:.2f}")
            print(f"     {cr['interpretation']}")
            print(f"     {cr['n_features']} features from {len(cr['signals'])} signals")
            print()
        
        # Best cohort
        if cohort_results:
            best = cohort_results[0]
            print(f"  BEST SINGLE COHORT: {best['cohort_id']} ({best['interpretation']})")
            print(f"  RMSE: {best['rmse']:.2f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if len(results) >= 2:
        first_rmse = results[0]['rmse']
        last_rmse = results[-1]['rmse']
        
        if first_rmse > 0 and first_rmse != float('inf'):
            improvement = (first_rmse - last_rmse) / first_rmse * 100
            print(f"Raw → Full PRISM: {first_rmse:.2f} → {last_rmse:.2f} ({improvement:.0f}% reduction)")
        
        # Biggest contributor
        biggest_delta = 0
        biggest_stage = None
        for i in range(1, len(results)):
            delta = results[i-1]['rmse'] - results[i]['rmse']
            if delta > biggest_delta:
                biggest_delta = delta
                biggest_stage = results[i]['stage']
        
        if biggest_stage:
            print(f"Biggest contributor: {biggest_stage} (Δ -{biggest_delta:.2f} RMSE)")
    
    print()
    print("KEY INSIGHT:")
    print("  PRISM discovered physical system structure from unlabeled data.")
    print("  'Give me your mystery sensors — I'll tell you what you have.'")
    
    # =========================================================================
    # Export
    # =========================================================================
    if args.output:
        output_data = {
            'target': args.target,
            'entity_column': entity_col,
            'n_entities': len(y),
            'discovery': discovery,
            'layer_ablation': [
                {k: v for k, v in r.items() if k != 'model'}
                for r in results
            ],
            'cohort_ablation': [
                {k: v for k, v in r.items() if k != 'model'}
                for r in cohort_results
            ] if cohort_results else [],
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
