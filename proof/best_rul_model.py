"""
Best RUL Prediction Model - C-MAPSS FD001/FD002
================================================

Results achieved:
- FD001: 13.36 RMSE (SOTA: 10.82, gap: +23%)
- FD002: 15.04 RMSE (SOTA: 11.46, gap: +31%)

Key features:
- Regime-aware healthy distance (per operating condition)
- Rolling temporal features (windows: 10, 20, 30 cycles)
- Sensor slope/volatility trends
"""

import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CACHE_DIR = Path("/var/folders/2v/f2fc1dgd24x8rcn0l72b73sw0000gn/T/cmapss_data")
COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

SENSOR_COLS = [f's{i}' for i in range(1, 22)]
KEY_SENSORS = ['s11', 's12', 's15', 's7', 's2', 's3', 's4', 's9', 's14', 's17']

# Regimes per dataset
N_REGIMES = {
    'FD001': 1,  # Single operating condition
    'FD002': 6,  # Six operating conditions
}

# XGBoost parameters (tuned)
XGB_PARAMS = {
    'n_estimators': 800,
    'max_depth': 7,
    'learning_rate': 0.015,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 2,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0,
}

# Feature windows
WINDOWS = [10, 20, 30]

# ============================================================
# FEATURES USED (29 total)
# ============================================================

FEATURES = [
    # Base
    "cycle",
    "dist_top1", "dist_top2", "dist_top3",

    # HD temporal (per window)
    "hd_slope_10", "hd_slope_20", "hd_slope_30",
    "hd_delta_10", "hd_delta_20", "hd_delta_30",
    "hd_std_10", "hd_std_20",
    "hd_curv_20", "hd_curv_30",
    "hd_max_20", "hd_min_20",

    # Sensor slopes/deltas
    "s11_slope_20", "s11_delta_20", "s11_std_20",
    "s12_slope_20", "s12_delta_20",
    "s15_slope_20", "s15_delta_20",
    "s11_s12_corr",

    # Extras
    "cycle_sq", "cycle_log",
    "s11", "s12", "s15",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_data(subset):
    """Load C-MAPSS train/test data."""
    train_pdf = pd.read_csv(CACHE_DIR / f"train_{subset}.txt", sep=r'\s+', header=None, names=COLS)
    test_pdf = pd.read_csv(CACHE_DIR / f"test_{subset}.txt", sep=r'\s+', header=None, names=COLS)
    rul_true = np.loadtxt(CACHE_DIR / f"RUL_{subset}.txt")

    train_df = pl.from_pandas(train_pdf)
    test_df = pl.from_pandas(test_pdf)

    # Add RUL to train (capped at 125)
    max_cycles = train_df.group_by("unit").agg(pl.col("cycle").max().alias("max_cycle"))
    train_df = train_df.join(max_cycles, on="unit")
    train_df = train_df.with_columns(
        pl.when(pl.col("max_cycle") - pl.col("cycle") > 125)
        .then(125)
        .otherwise(pl.col("max_cycle") - pl.col("cycle"))
        .alias("RUL")
    )

    return train_df, test_df, rul_true


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def compute_regime_baselines(train_df, n_regimes):
    """Compute healthy baselines per operating regime."""
    # Cluster operating conditions
    op_data = train_df.select(["op1", "op2"]).to_numpy()
    regime_km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    regime_km.fit(op_data)

    train_df = train_df.with_columns(pl.Series("regime_id", regime_km.predict(op_data)))

    # Compute per-regime healthy baselines
    regime_baselines = {}
    for regime in range(n_regimes):
        regime_baselines[regime] = {}
        regime_data = train_df.filter(pl.col("regime_id") == regime)

        for unit in regime_data["unit"].unique().to_list()[:40]:
            unit_data = regime_data.filter(pl.col("unit") == unit)
            if len(unit_data) < 5:
                continue

            # First 20% of life = healthy
            healthy = unit_data.head(max(1, len(unit_data) // 5))

            for sig in SENSOR_COLS:
                vals = healthy[sig].drop_nulls().to_numpy()
                if len(vals) > 0:
                    if sig not in regime_baselines[regime]:
                        regime_baselines[regime][sig] = {'means': [], 'stds': []}
                    regime_baselines[regime][sig]['means'].append(np.mean(vals))
                    regime_baselines[regime][sig]['stds'].append(np.std(vals))

        # Aggregate
        for sig in list(regime_baselines[regime].keys()):
            stats = regime_baselines[regime][sig]
            if stats['means']:
                regime_baselines[regime][sig] = {
                    'mean': np.mean(stats['means']),
                    'std': np.mean(stats['stds']) + np.std(stats['means']) + 1e-10
                }

    return regime_km, regime_baselines


def compute_features(df, regime_km, regime_baselines):
    """Compute all features for a dataframe."""
    # Add regime
    op_vals = df.select(["op1", "op2"]).to_numpy()
    df = df.with_columns(pl.Series("regime_id", regime_km.predict(op_vals)))

    all_rows = []

    for unit in df["unit"].unique().to_list():
        unit_df = df.filter(pl.col("unit") == unit).sort("cycle")
        rows = unit_df.to_dicts()

        # History tracking
        hd_history = []
        sensor_hist = {s: [] for s in KEY_SENSORS}

        for i, row in enumerate(rows):
            regime = row['regime_id']
            baselines = regime_baselines.get(regime, {})

            # Compute healthy distances
            dists = []
            for sig in SENSOR_COLS:
                if sig in baselines:
                    val = row[sig]
                    if val is not None:
                        d = abs(val - baselines[sig]['mean']) / baselines[sig]['std']
                        dists.append((sig, d))
                        row[f'{sig}_dist'] = d

            dists.sort(key=lambda x: -x[1])
            for j, (sig, d) in enumerate(dists[:5]):
                row[f'dist_top{j+1}'] = d

            hd = np.mean([d for _, d in dists[:5]]) if dists else 0
            row['hd'] = hd
            hd_history.append(hd)

            # Track sensors
            for s in KEY_SENSORS:
                sensor_hist[s].append(row.get(s, 0) or 0)

            n = len(hd_history)

            # Rolling window features
            for W in WINDOWS:
                if n >= W:
                    recent = hd_history[-W:]
                    row[f'hd_mean_{W}'] = np.mean(recent)
                    row[f'hd_std_{W}'] = np.std(recent)
                    row[f'hd_max_{W}'] = np.max(recent)
                    row[f'hd_min_{W}'] = np.min(recent)
                    row[f'hd_delta_{W}'] = hd - hd_history[-W]

                    # Slope
                    x = np.arange(W)
                    coeffs = np.polyfit(x, recent, 1)
                    row[f'hd_slope_{W}'] = coeffs[0]

                    # Curvature
                    if W >= 10:
                        mid = W // 2
                        slope1 = (recent[mid] - recent[0]) / mid
                        slope2 = (recent[-1] - recent[mid]) / (W - mid)
                        row[f'hd_curv_{W}'] = slope2 - slope1
                else:
                    for feat in ['mean', 'std', 'max', 'min', 'delta', 'slope']:
                        row[f'hd_{feat}_{W}'] = 0
                    if W >= 10:
                        row[f'hd_curv_{W}'] = 0

            # Sensor rolling features
            for s in KEY_SENSORS[:5]:
                hist = sensor_hist[s]
                for W in [20]:
                    if len(hist) >= W:
                        recent = hist[-W:]
                        row[f'{s}_mean_{W}'] = np.mean(recent)
                        row[f'{s}_std_{W}'] = np.std(recent)
                        row[f'{s}_delta_{W}'] = hist[-1] - hist[-W]
                        row[f'{s}_slope_{W}'] = np.polyfit(np.arange(W), recent, 1)[0]
                    else:
                        row[f'{s}_mean_{W}'] = hist[-1] if hist else 0
                        row[f'{s}_std_{W}'] = 0
                        row[f'{s}_delta_{W}'] = 0
                        row[f'{s}_slope_{W}'] = 0

            # Cross-sensor correlation
            if n >= 10:
                s11_recent = sensor_hist['s11'][-10:]
                s12_recent = sensor_hist['s12'][-10:]
                if np.std(s11_recent) > 0 and np.std(s12_recent) > 0:
                    row['s11_s12_corr'] = np.corrcoef(s11_recent, s12_recent)[0, 1]
                else:
                    row['s11_s12_corr'] = 0
            else:
                row['s11_s12_corr'] = 0

            # Polynomial cycle features
            row['cycle_sq'] = row['cycle'] ** 2
            row['cycle_log'] = np.log1p(row['cycle'])

            all_rows.append(row)

    return pl.DataFrame(all_rows)


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_and_evaluate(subset):
    """Train model and evaluate on official test set."""
    print(f"\n{'='*60}")
    print(f"Processing {subset}")
    print('='*60)

    # Load data
    train_df, test_df, rul_true = load_data(subset)
    n_regimes = N_REGIMES[subset]

    print(f"Train: {len(train_df)} rows, {train_df['unit'].n_unique()} engines")
    print(f"Test: {len(test_df)} rows, {test_df['unit'].n_unique()} engines")

    # Compute baselines
    print("Computing regime baselines...")
    regime_km, regime_baselines = compute_regime_baselines(train_df, n_regimes)

    # Compute features
    print("Computing features...")
    train_feat = compute_features(train_df, regime_km, regime_baselines)
    test_feat = compute_features(test_df, regime_km, regime_baselines)

    # Get available features
    available = [f for f in FEATURES if f in train_feat.columns]
    print(f"Using {len(available)} features")

    # Prepare training data
    X_train = train_feat.select(available).to_numpy()
    y_train = train_feat["RUL"].to_numpy()
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Train model
    print("Training XGBoost...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train_s, y_train)

    # Prepare test data (last cycle per unit)
    test_last = test_feat.group_by("unit").agg(pl.col("cycle").max().alias("last_cycle"))
    test_final = test_feat.join(test_last, on="unit").filter(
        pl.col("cycle") == pl.col("last_cycle")
    ).sort("unit")

    X_test = test_final.select(available).to_numpy()
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_s = scaler.transform(X_test)

    # Predict
    y_pred = np.clip(model.predict(X_test_s), 0, 125)
    rul_capped = np.clip(rul_true, 0, 125)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(rul_capped, y_pred))
    sota = 10.82 if subset == "FD001" else 11.46
    gap = (rmse - sota) / sota * 100

    print(f"\n{subset} Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  SOTA: {sota}")
    print(f"  Gap:  +{gap:.0f}%")

    return rmse, model, scaler, available


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("BEST RUL MODEL - C-MAPSS")
    print("="*60)

    results = {}
    for subset in ["FD001", "FD002"]:
        rmse, model, scaler, features = train_and_evaluate(subset)
        results[subset] = {
            'rmse': rmse,
            'model': model,
            'scaler': scaler,
            'features': features,
        }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Dataset':<10} {'RMSE':>10} {'SOTA':>10} {'Gap':>10}")
    print("-"*40)
    print(f"{'FD001':<10} {results['FD001']['rmse']:>10.2f} {'10.82':>10} {'+' + str(int((results['FD001']['rmse']-10.82)/10.82*100)) + '%':>10}")
    print(f"{'FD002':<10} {results['FD002']['rmse']:>10.2f} {'11.46':>10} {'+' + str(int((results['FD002']['rmse']-11.46)/11.46*100)) + '%':>10}")
    print("="*60)
