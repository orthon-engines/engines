"""
Ørthon Cohort Discovery v3 - Correlation-Based Signal Matching

The key insight: Match signals by WHO THEY CORRELATE WITH, not their absolute values.

If signal X in Entity A correlates strongly with signals Y and Z,
then find the signal in Entity B that also correlates with its Y and Z equivalents.

This is structure-preserving matching - the correlation GRAPH is what matters.
"""

import polars as pl
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
import os

DATA_PATH = Path(os.environ.get("ORTHON_DATA_PATH", "data/battery"))


def load_observations():
    """Load observations data."""
    return pl.read_parquet(DATA_PATH / "observations.parquet")


def build_signal_matrix(obs: pl.DataFrame, entity_id: str) -> tuple:
    """
    Build time-aligned signal matrix for one entity.
    Returns: (signal_names, timestamps, value_matrix)
    """
    entity_data = obs.filter(pl.col("entity_id") == entity_id)

    # Pivot to wide format: rows = timestamps, cols = signals
    pivot = entity_data.pivot(
        values="value",
        index="timestamp",
        on="signal_id"
    ).sort("timestamp")

    timestamps = pivot["timestamp"].to_numpy()
    signal_names = [c for c in pivot.columns if c != "timestamp"]

    # Build matrix
    matrix = np.column_stack([
        pivot[s].to_numpy() for s in signal_names
    ])

    return signal_names, timestamps, matrix


def compute_correlation_fingerprint(matrix: np.ndarray) -> np.ndarray:
    """
    Compute correlation fingerprint for each signal.

    For signal i, fingerprint = correlation with all other signals.
    This captures the RELATIONAL structure, not absolute values.
    """
    # Handle NaN
    matrix_clean = np.nan_to_num(matrix, nan=0.0)

    # Z-score normalize each column (signal)
    scaler = StandardScaler()
    matrix_norm = scaler.fit_transform(matrix_clean)

    # Compute correlation matrix
    n_signals = matrix_norm.shape[1]
    corr = np.corrcoef(matrix_norm.T)

    # Handle NaN correlations
    corr = np.nan_to_num(corr, nan=0.0)

    # Each row of corr is the fingerprint for that signal
    return corr


def compute_lagged_correlation(matrix: np.ndarray, max_lag: int = 5) -> np.ndarray:
    """
    Compute lagged cross-correlation features.
    Captures temporal relationships - does A lead B or vice versa?
    """
    matrix_clean = np.nan_to_num(matrix, nan=0.0)
    n_signals = matrix_clean.shape[1]

    # Z-score normalize
    scaler = StandardScaler()
    matrix_norm = scaler.fit_transform(matrix_clean)

    # For each signal pair, compute correlation at different lags
    lag_features = []

    for i in range(n_signals):
        signal_lags = []
        for j in range(n_signals):
            if i == j:
                signal_lags.extend([1.0, 0])  # Self-correlation
            else:
                # Find best lag
                best_corr = 0
                best_lag = 0
                for lag in range(-max_lag, max_lag + 1):
                    if lag == 0:
                        c = np.corrcoef(matrix_norm[:, i], matrix_norm[:, j])[0, 1]
                    elif lag > 0:
                        c = np.corrcoef(matrix_norm[lag:, i], matrix_norm[:-lag, j])[0, 1]
                    else:
                        c = np.corrcoef(matrix_norm[:lag, i], matrix_norm[-lag:, j])[0, 1]

                    if not np.isnan(c) and abs(c) > abs(best_corr):
                        best_corr = c
                        best_lag = lag

                signal_lags.extend([best_corr, best_lag / max_lag])

        lag_features.append(signal_lags)

    return np.array(lag_features)


def compute_spectral_fingerprint(matrix: np.ndarray) -> np.ndarray:
    """
    Compute spectral (frequency) fingerprint for each signal.
    Captures periodicity patterns independent of amplitude.
    """
    matrix_clean = np.nan_to_num(matrix, nan=0.0)
    n_signals = matrix_clean.shape[1]

    spectral_features = []

    for i in range(n_signals):
        signal = matrix_clean[:, i]

        # Remove mean
        signal = signal - signal.mean()

        # FFT
        fft = np.abs(np.fft.rfft(signal))

        # Normalize to relative power
        if fft.sum() > 0:
            fft = fft / fft.sum()

        # Take first N frequency bins as features
        n_bins = min(20, len(fft))
        spectral_features.append(fft[:n_bins])

    # Pad to same length
    max_len = max(len(f) for f in spectral_features)
    spectral_features = [
        np.pad(f, (0, max_len - len(f))) for f in spectral_features
    ]

    return np.array(spectral_features)


def compute_shape_fingerprint(matrix: np.ndarray) -> np.ndarray:
    """
    Compute normalized shape fingerprint.
    Z-score the entire series, then sample at fixed percentiles.
    """
    matrix_clean = np.nan_to_num(matrix, nan=0.0)
    n_signals = matrix_clean.shape[1]
    n_samples = matrix_clean.shape[0]

    # Sample points (percentiles of the time series)
    percentiles = np.linspace(0, 100, 50)
    sample_indices = (percentiles / 100 * (n_samples - 1)).astype(int)

    shape_features = []

    for i in range(n_signals):
        signal = matrix_clean[:, i]

        # Z-score normalize
        if signal.std() > 0:
            signal_norm = (signal - signal.mean()) / signal.std()
        else:
            signal_norm = signal - signal.mean()

        # Sample at percentiles
        shape = signal_norm[sample_indices]
        shape_features.append(shape)

    return np.array(shape_features)


def match_signals_hungarian(fingerprints_a: np.ndarray, fingerprints_b: np.ndarray) -> list:
    """
    Use Hungarian algorithm for optimal 1-to-1 matching.
    Minimizes total distance between matched pairs.
    """
    # Compute distance matrix
    dist = cdist(fingerprints_a, fingerprints_b, metric='correlation')

    # Handle NaN
    dist = np.nan_to_num(dist, nan=1.0)

    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(dist)

    # Return matches with distances
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append((i, j, 1 - dist[i, j]))  # Convert distance to similarity

    return matches


def combined_fingerprint(matrix: np.ndarray) -> np.ndarray:
    """
    Combine multiple fingerprint types for robust matching.
    """
    # Correlation structure (most important)
    corr_fp = compute_correlation_fingerprint(matrix)

    # Shape (normalized time series)
    shape_fp = compute_shape_fingerprint(matrix)

    # Spectral (frequency content)
    # spectral_fp = compute_spectral_fingerprint(matrix)

    # Combine with weights
    # Correlation fingerprint is N x N, shape is N x 50
    # We'll use correlation as primary, shape as secondary

    combined = np.hstack([
        corr_fp * 2.0,      # Weight correlation higher
        shape_fp * 1.0,     # Shape
    ])

    return combined


def blind_match_entities(obs: pl.DataFrame, entity_a: str, entity_b: str) -> dict:
    """
    Blind signal matching between two entities using structural fingerprints.
    """
    print(f"\n{'='*70}")
    print(f"BLIND MATCHING: {entity_a} vs {entity_b}")
    print("="*70)

    # Build signal matrices
    names_a, ts_a, matrix_a = build_signal_matrix(obs, entity_a)
    names_b, ts_b, matrix_b = build_signal_matrix(obs, entity_b)

    print(f"\n  {entity_a}: {len(names_a)} signals, {len(ts_a)} timesteps")
    print(f"  {entity_b}: {len(names_b)} signals, {len(ts_b)} timesteps")

    # Compute combined fingerprints
    print("\n  Computing structural fingerprints...")
    fp_a = combined_fingerprint(matrix_a)
    fp_b = combined_fingerprint(matrix_b)

    # Match using Hungarian algorithm
    print("  Running optimal matching...")
    matches = match_signals_hungarian(fp_a, fp_b)

    # Evaluate
    print(f"\n  {'Signal A':<30} {'Matched B':<30} {'Correct?':<10} {'Similarity'}")
    print("  " + "-"*85)

    correct = 0
    results = {}

    for i, j, sim in sorted(matches, key=lambda x: names_a[x[0]]):
        name_a = names_a[i]
        name_b = names_b[j]
        is_match = "YES" if name_a == name_b else "NO"
        if name_a == name_b:
            correct += 1

        print(f"  {name_a:<30} {name_b:<30} {is_match:<10} {sim:.3f}")
        results[name_a] = {'matched_to': name_b, 'correct': name_a == name_b, 'similarity': sim}

    accuracy = correct / len(names_a) * 100
    print(f"\n  {'='*70}")
    print(f"  ACCURACY: {correct}/{len(names_a)} ({accuracy:.1f}%)")
    print("="*70)

    return results, accuracy


def iterative_matching(obs: pl.DataFrame, entities: list) -> dict:
    """
    Iterative refinement: Use matched signals to improve subsequent matches.

    1. Match entity 0 to entity 1
    2. Use consensus to match entity 2
    3. Continue building confidence
    """
    print("\n" + "="*70)
    print("ITERATIVE CROSS-ENTITY MATCHING")
    print("="*70)

    if len(entities) < 2:
        print("Need at least 2 entities")
        return {}

    # Start with first two entities
    all_results = {}

    for i in range(len(entities) - 1):
        entity_a = entities[i]
        entity_b = entities[i + 1]

        results, accuracy = blind_match_entities(obs, entity_a, entity_b)
        all_results[f"{entity_a}_to_{entity_b}"] = {
            'results': results,
            'accuracy': accuracy
        }

    return all_results


def main():
    print("="*70)
    print("ØRTHON COHORT DISCOVERY v3")
    print("Correlation-Based Structural Matching")
    print("="*70)
    print(f"\nData path: {DATA_PATH}")

    # Load data
    obs = load_observations()
    entities = obs["entity_id"].unique().sort().to_list()

    print(f"Entities: {entities}")
    print(f"Total observations: {len(obs):,}")

    if len(entities) >= 2:
        # Test blind matching between first two entities
        results, accuracy = blind_match_entities(obs, entities[0], entities[1])

        # If more entities, do iterative matching
        if len(entities) > 2:
            all_results = iterative_matching(obs, entities)
    else:
        print("Need at least 2 entities for cross-matching")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
