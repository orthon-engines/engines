"""
Regime Detector Engine

Detects behavioral regimes using Hidden Markov Model.

Definition:
    Learns latent regimes from observable features (baseline_distance, hd_slope)
    Regimes are ordered by baseline_distance: 0 = closest to baseline, N = furthest

Why HMM:
    - Learns regimes from data (no manual threshold)
    - Probabilistic (gives confidence)
    - Handles noisy transitions
    - Standard in prognostics literature

Config:
    n_regimes: int (default 3) - Number of HMM states
    regime_features: list (default ['baseline_distance', 'hd_slope'])

Output:
    regime: int - Regime label (0 = nominal, higher = more degraded)
    regime_prob: float - Probability of assigned regime
    regime_N_prob: float - Probability of each regime
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import polars as pl


def compute(
    df: pl.DataFrame = None,
    baseline_distance: np.ndarray = None,
    hd_slope: np.ndarray = None,
    config: Dict[str, Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Detect regimes using Hidden Markov Model.

    Args:
        df: DataFrame with baseline_distance, hd_slope columns
        baseline_distance: Alternative: array of baseline distances
        hd_slope: Alternative: array of hd_slope values
        config: Contains n_regimes, regime_features

    Returns:
        Dict with regime labels, probabilities, transition matrix
    """
    if config is None:
        config = {}

    n_regimes = config.get('n_regimes', 3)
    features = config.get('regime_features', ['baseline_distance', 'hd_slope'])

    # Get feature matrix
    if df is not None:
        X = _extract_features(df, features)
    elif baseline_distance is not None:
        if hd_slope is not None:
            X = np.column_stack([baseline_distance, hd_slope])
        else:
            X = np.asarray(baseline_distance).reshape(-1, 1)
    else:
        return _null_result("No data provided")

    if X is None or len(X) < n_regimes * 5:
        return _null_result(f"Insufficient data: {len(X) if X is not None else 0} < {n_regimes * 5}")

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Try to import hmmlearn
    try:
        from hmmlearn import hmm
    except ImportError:
        # Fallback to simple clustering if hmmlearn not available
        return _compute_fallback(X, n_regimes)

    # Fit HMM
    try:
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=42,
        )
        model.fit(X)

        # Predict
        regime_labels = model.predict(X)
        regime_probs = model.predict_proba(X)

        # Order regimes by mean of first feature (baseline_distance)
        # So regime 0 = closest to baseline (nominal)
        regime_means = [X[regime_labels == i, 0].mean() if np.sum(regime_labels == i) > 0 else np.inf
                        for i in range(n_regimes)]
        order = np.argsort(regime_means)
        label_map = {old: new for new, old in enumerate(order)}

        # Reorder labels and probabilities
        regime_labels = np.array([label_map[l] for l in regime_labels])
        regime_probs = regime_probs[:, order]

        # Reorder transition matrix
        transition_matrix = model.transmat_[order][:, order]

    except Exception as e:
        return _compute_fallback(X, n_regimes, error=str(e))

    # Build result
    result = {
        'regime': regime_labels.tolist(),
        'regime_prob': regime_probs.max(axis=1).tolist(),
        'n_regimes': n_regimes,
        'method': 'hmm',
    }

    # Add per-regime probabilities
    for i in range(n_regimes):
        result[f'regime_{i}_prob'] = regime_probs[:, i].tolist()

    # Add transition matrix as flattened
    result['transition_matrix'] = transition_matrix.tolist()

    # Summary statistics
    for i in range(n_regimes):
        count = np.sum(regime_labels == i)
        result[f'regime_{i}_count'] = int(count)
        result[f'regime_{i}_fraction'] = float(count / len(regime_labels))

    return result


def _compute_fallback(X: np.ndarray, n_regimes: int, error: str = None) -> Dict[str, Any]:
    """
    Fallback to simple quantile-based regime detection.

    Used when hmmlearn is not available or fails.
    """
    from scipy import stats

    # Use first feature (baseline_distance) for regime assignment
    values = X[:, 0]

    # Compute quantile boundaries
    quantiles = np.linspace(0, 1, n_regimes + 1)[1:-1]
    boundaries = np.quantile(values, quantiles)

    # Assign regimes based on quantiles
    regime_labels = np.digitize(values, boundaries)

    # Compute "probabilities" based on distance to boundaries
    # This is a rough approximation - not true probabilities
    regime_probs = np.zeros((len(values), n_regimes))
    for i, v in enumerate(values):
        regime_probs[i, regime_labels[i]] = 0.8  # Assigned regime gets 0.8
        remaining = 0.2 / (n_regimes - 1) if n_regimes > 1 else 0
        for j in range(n_regimes):
            if j != regime_labels[i]:
                regime_probs[i, j] = remaining

    result = {
        'regime': regime_labels.tolist(),
        'regime_prob': regime_probs.max(axis=1).tolist(),
        'n_regimes': n_regimes,
        'method': 'quantile_fallback',
    }

    if error:
        result['hmm_error'] = error

    for i in range(n_regimes):
        result[f'regime_{i}_prob'] = regime_probs[:, i].tolist()
        count = np.sum(regime_labels == i)
        result[f'regime_{i}_count'] = int(count)
        result[f'regime_{i}_fraction'] = float(count / len(regime_labels))

    return result


def _extract_features(df: pl.DataFrame, features: List[str]) -> Optional[np.ndarray]:
    """Extract feature matrix from DataFrame."""
    available = [f for f in features if f in df.columns]

    if not available:
        return None

    return df.select(available).to_numpy()


def _null_result(reason: str) -> Dict[str, Any]:
    """Return null result with reason."""
    return {
        'regime': None,
        'regime_prob': None,
        'n_regimes': None,
        'method': None,
        'regime_detector_error': reason,
    }
