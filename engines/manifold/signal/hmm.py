"""
Hidden Markov Model Engine
==========================

Probabilistic regime detection via Gaussian HMM.

Fits HMM with BIC-based state selection (2 to min(max_states, n//10)),
identifying the most likely number of hidden regimes and their transitions.

Outputs:
    - hmm_n_states: number of hidden states selected by BIC
    - hmm_current_state: most likely state at final observation
    - hmm_current_state_prob: posterior probability of current state
    - hmm_state_entropy: entropy of stationary distribution (regime diversity)
    - hmm_transition_rate: fraction of state transitions (instability)
    - hmm_bic: BIC of best model
    - hmm_log_likelihood: log-likelihood of best model

Layer: Dynamics (per-signal)
Min samples: 30

Requires: hmmlearn (optional dependency)
    pip install hmmlearn>=0.3.0
"""

import numpy as np
from typing import Dict

# Optional import with graceful fallback
try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


def compute(y: np.ndarray, max_states: int = 5) -> Dict[str, float]:
    """
    Fit Gaussian HMM and extract regime metrics.

    Args:
        y: 1D signal array
        max_states: Maximum number of hidden states to consider

    Returns:
        dict with hmm_* prefixed keys
    """
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 30:
        return _empty_result()

    if not _HMM_AVAILABLE:
        return _empty_result()

    X = y.reshape(-1, 1)

    # BIC-based state selection
    best_bic = np.inf
    best_model = None
    best_n = 2

    max_k = min(max_states, max(2, n // 10))

    for k in range(2, max_k + 1):
        try:
            model = GaussianHMM(
                n_components=k,
                covariance_type='full',
                n_iter=100,
                random_state=42,
                verbose=False,
            )
            model.fit(X)
            ll = model.score(X)

            # BIC = -2*LL + k*log(n) where k = free parameters
            # Free params: k-1 (initial probs) + k*(k-1) (transition) + k*2 (means + variances)
            n_params = (k - 1) + k * (k - 1) + k * 2
            bic = -2 * ll + n_params * np.log(n)

            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = k
        except Exception:
            continue

    if best_model is None:
        return _empty_result()

    # Decode state sequence
    try:
        states = best_model.predict(X)
        state_probs = best_model.predict_proba(X)
    except Exception:
        return _empty_result()

    # Current state (last observation)
    current_state = int(states[-1])
    current_state_prob = float(state_probs[-1, current_state])

    # Stationary distribution entropy
    try:
        transmat = best_model.transmat_
        # Stationary distribution: eigenvector of transmat^T with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.abs(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        # Entropy of stationary distribution
        stationary_safe = stationary[stationary > 1e-10]
        state_entropy = float(-np.sum(stationary_safe * np.log(stationary_safe)))
    except Exception:
        state_entropy = np.nan

    # Transition rate: fraction of time steps with state change
    transitions = np.sum(np.diff(states) != 0)
    transition_rate = float(transitions / max(len(states) - 1, 1))

    # Log-likelihood
    log_likelihood = float(best_model.score(X))

    return {
        'hmm_n_states': int(best_n),
        'hmm_current_state': current_state,
        'hmm_current_state_prob': current_state_prob,
        'hmm_state_entropy': state_entropy,
        'hmm_transition_rate': transition_rate,
        'hmm_bic': float(best_bic),
        'hmm_log_likelihood': log_likelihood,
    }


def _empty_result() -> Dict[str, float]:
    """Return NaN result when HMM cannot be computed."""
    return {
        'hmm_n_states': np.nan,
        'hmm_current_state': np.nan,
        'hmm_current_state_prob': np.nan,
        'hmm_state_entropy': np.nan,
        'hmm_transition_rate': np.nan,
        'hmm_bic': np.nan,
        'hmm_log_likelihood': np.nan,
    }
