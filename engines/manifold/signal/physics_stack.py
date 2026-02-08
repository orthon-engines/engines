"""
Physics Stack Engine

Computes metrics for all physics layers:
- L1: State (phase space position)
- L2: Coherence (signal coupling)
- L3: Mechanics (energy proxy)
- L4: Thermodynamics (dissipation)

ENGINES computes. Orthon interprets.
"""

import numpy as np
import pandas as pd
from scipy.linalg import inv
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================
# METRIC DETECTION
# ============================================================

EXCLUDE_COLUMNS = {'unit_id', 'signal_id', 'I', 'value', 'unit', 'timestamp'}

PREFERRED_STATE_METRICS = [
    'rolling_hurst', 'rolling_entropy', 'rolling_kurtosis',
    'rolling_rms', 'rolling_std', 'rolling_mean', 'rolling_volatility',
    'rolling_skewness', 'rolling_crest_factor', 'rolling_range',
    'rolling_pulsation', 'rolling_lyapunov',
    'dy', 'd2y', 'd3y', 'curvature',
    'z_score',
]


def get_available_metrics(df: pd.DataFrame) -> List[str]:
    """Auto-detect available numeric metrics in dataframe."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available = [c for c in numeric_cols if c not in EXCLUDE_COLUMNS]

    preferred = [c for c in PREFERRED_STATE_METRICS if c in available]
    others = [c for c in available if c not in PREFERRED_STATE_METRICS]

    return preferred + others


# ============================================================
# L1: STATE DISTANCE
# ============================================================

def compute_baseline(
    vectors: np.ndarray,
    n_baseline: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute baseline mean and inverse covariance."""
    n_obs, n_metrics = vectors.shape
    n_baseline = min(n_baseline, max(10, n_obs // 2))

    baseline = vectors[:n_baseline].copy()

    valid_cols = []
    for i in range(n_metrics):
        col = baseline[:, i]
        valid_count = np.sum(~np.isnan(col))
        if valid_count >= n_baseline * 0.5:
            valid_cols.append(i)

    if len(valid_cols) < 2:
        return None, None, 0

    baseline = baseline[:, valid_cols]

    col_means = np.nanmean(baseline, axis=0)
    for i in range(baseline.shape[1]):
        mask = np.isnan(baseline[:, i])
        baseline[mask, i] = col_means[i]

    col_std = np.std(baseline, axis=0)
    varying_cols = col_std > 1e-10

    if np.sum(varying_cols) < 2:
        return None, None, 0

    baseline = baseline[:, varying_cols]
    valid_cols = [valid_cols[i] for i, v in enumerate(varying_cols) if v]

    mean = np.mean(baseline, axis=0)
    cov = np.cov(baseline, rowvar=False)

    if cov.ndim == 0:
        cov = np.array([[max(cov, 1e-6)]])
    else:
        cov += np.eye(cov.shape[0]) * 1e-6

    try:
        cov_inv = inv(cov)
    except:
        try:
            cov_inv = np.linalg.pinv(cov)
        except:
            return None, None, 0

    return mean, cov_inv, len(valid_cols)


def compute_state_distance(
    obs_enriched: pd.DataFrame,
    unit_id: str,
    I: np.ndarray,
    n_baseline: int = 100
) -> Dict[str, np.ndarray]:
    """Compute state distance and velocity using available metrics."""
    n = len(I)

    entity_data = obs_enriched[obs_enriched['unit_id'] == unit_id]

    if entity_data.empty:
        return {
            'state_distance': np.full(n, np.nan),
            'state_velocity': np.full(n, np.nan),
            'state_acceleration': np.full(n, np.nan),
            'n_metrics_used': 0,
        }

    metrics = get_available_metrics(entity_data)

    if len(metrics) < 2:
        return {
            'state_distance': np.full(n, np.nan),
            'state_velocity': np.full(n, np.nan),
            'state_acceleration': np.full(n, np.nan),
            'n_metrics_used': 0,
        }

    metrics_by_I = entity_data.groupby('I')[metrics].mean().reset_index()
    metrics_by_I = metrics_by_I.sort_values('I')

    metric_I = metrics_by_I['I'].values
    vectors = metrics_by_I[metrics].values

    baseline_mean, baseline_cov_inv, n_metrics_used = compute_baseline(vectors, n_baseline)

    if baseline_mean is None:
        return {
            'state_distance': np.full(n, np.nan),
            'state_velocity': np.full(n, np.nan),
            'state_acceleration': np.full(n, np.nan),
            'n_metrics_used': 0,
        }

    state_distance = np.zeros(len(metric_I))

    for i in range(len(metric_I)):
        vec = vectors[i, :n_metrics_used].copy()

        nan_mask = np.isnan(vec)
        if nan_mask.all():
            state_distance[i] = np.nan
            continue
        vec[nan_mask] = baseline_mean[nan_mask]

        diff = vec - baseline_mean
        try:
            state_distance[i] = np.sqrt(np.clip(
                np.dot(np.dot(diff, baseline_cov_inv), diff),
                0, None
            ))
        except:
            state_distance[i] = np.nan

    if len(metric_I) > 1:
        state_velocity = np.gradient(state_distance, metric_I)
        state_acceleration = np.gradient(state_velocity, metric_I)
    else:
        state_velocity = np.zeros_like(state_distance)
        state_acceleration = np.zeros_like(state_distance)

    if not np.array_equal(metric_I, I):
        state_distance = np.interp(I, metric_I, state_distance)
        state_velocity = np.interp(I, metric_I, state_velocity)
        state_acceleration = np.interp(I, metric_I, state_acceleration)

    return {
        'state_distance': state_distance,
        'state_velocity': state_velocity,
        'state_acceleration': state_acceleration,
        'n_metrics_used': n_metrics_used,
    }


# ============================================================
# L2: COHERENCE (Eigenvalue-Based)
# ============================================================

def compute_coherence(
    obs_enriched: pd.DataFrame,
    unit_id: str,
    I: np.ndarray,
    window: int = 50
) -> Dict[str, np.ndarray]:
    """
    Compute coherence using eigenvalue decomposition of correlation matrix.

    This replaces mean(|correlation|) with spectral coherence:
    - spectral_coherence = λ₁ / Σλᵢ (variance explained by dominant mode)
    - effective_dim = participation ratio (number of independent modes)
    - eigenvalue_entropy = disorder of eigenvalue distribution

    Why eigenvalues?
    - Captures STRUCTURE, not just average strength
    - Detects when signals fragment into independent modes
    - More sensitive to structural decoupling
    """
    n = len(I)

    entity_data = obs_enriched[obs_enriched['unit_id'] == unit_id]
    signals = entity_data['signal_id'].unique()
    n_signals = len(signals)

    if n_signals < 2:
        return {
            'coherence': np.full(n, np.nan),
            'coherence_velocity': np.full(n, np.nan),
            'effective_dim': np.full(n, np.nan),
            'eigenvalue_entropy': np.full(n, np.nan),
            'n_pairs': 0,
            'n_signals': n_signals,
        }

    n_pairs = n_signals * (n_signals - 1) // 2

    try:
        pivoted = entity_data.pivot_table(
            index='I',
            columns='signal_id',
            values='value',
            aggfunc='mean'
        ).sort_index()
    except Exception:
        return {
            'coherence': np.full(n, np.nan),
            'coherence_velocity': np.full(n, np.nan),
            'effective_dim': np.full(n, np.nan),
            'eigenvalue_entropy': np.full(n, np.nan),
            'n_pairs': n_pairs,
            'n_signals': n_signals,
        }

    pivot_I = pivoted.index.values
    n_pivot = len(pivot_I)

    # Initialize output arrays
    coherence = np.full(n_pivot, np.nan)
    effective_dim = np.full(n_pivot, np.nan)
    eigenvalue_entropy = np.full(n_pivot, np.nan)

    # Compute rolling eigenvalue-based coherence
    for i in range(window, n_pivot):
        window_data = pivoted.iloc[i-window:i].values.copy()  # Copy to avoid read-only

        # Skip if too many NaNs
        nan_fraction = np.isnan(window_data).sum() / window_data.size
        if nan_fraction > 0.5:
            continue

        # Fill NaNs with column mean for this window
        col_means = np.nanmean(window_data, axis=0)
        for col in range(window_data.shape[1]):
            mask = np.isnan(window_data[:, col])
            window_data[mask, col] = col_means[col]

        try:
            # Compute correlation matrix
            R = np.corrcoef(window_data, rowvar=False)

            # Handle NaN in correlation matrix
            if np.any(np.isnan(R)):
                R = np.nan_to_num(R, nan=0.0)
                np.fill_diagonal(R, 1.0)

            # Eigenvalue decomposition
            eigenvalues = np.linalg.eigvalsh(R)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

            total_variance = np.sum(eigenvalues)

            if total_variance > 0:
                # Spectral coherence: fraction of variance in first eigenvalue
                coherence[i] = eigenvalues[0] / total_variance

                # Effective dimensionality (participation ratio)
                effective_dim[i] = total_variance**2 / np.sum(eigenvalues**2)

                # Eigenvalue entropy (normalized)
                p = eigenvalues / total_variance
                p = p[p > 1e-10]  # Avoid log(0)
                if len(p) > 1:
                    eigenvalue_entropy[i] = -np.sum(p * np.log(p)) / np.log(len(eigenvalues))
                else:
                    eigenvalue_entropy[i] = 0.0

        except Exception:
            continue

    # Fill early values with first valid
    first_valid_idx = window
    if first_valid_idx < n_pivot:
        if not np.isnan(coherence[first_valid_idx]):
            coherence[:first_valid_idx] = coherence[first_valid_idx]
        if not np.isnan(effective_dim[first_valid_idx]):
            effective_dim[:first_valid_idx] = effective_dim[first_valid_idx]
        if not np.isnan(eigenvalue_entropy[first_valid_idx]):
            eigenvalue_entropy[:first_valid_idx] = eigenvalue_entropy[first_valid_idx]

    # Compute velocity
    coherence_clean = np.nan_to_num(coherence, nan=0)
    if n_pivot > 1:
        coherence_velocity = np.gradient(coherence_clean, pivot_I)
    else:
        coherence_velocity = np.zeros(n_pivot)

    # Interpolate to requested I values
    if not np.array_equal(pivot_I, I):
        coherence = np.interp(I, pivot_I, coherence)
        coherence_velocity = np.interp(I, pivot_I, coherence_velocity)
        effective_dim = np.interp(I, pivot_I, effective_dim)
        eigenvalue_entropy = np.interp(I, pivot_I, eigenvalue_entropy)

    return {
        'coherence': coherence,                    # λ₁/Σλ (0 to 1, higher = more coupled)
        'coherence_velocity': coherence_velocity,  # d(coherence)/dt
        'effective_dim': effective_dim,            # Participation ratio (1 to N)
        'eigenvalue_entropy': eigenvalue_entropy,  # Normalized entropy (0 to 1)
        'n_pairs': n_pairs,
        'n_signals': n_signals,
    }


# ============================================================
# L3: MECHANICS
# ============================================================

def compute_energy(
    obs_enriched: pd.DataFrame,
    unit_id: str,
    I: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute energy proxy and flow metrics."""
    n = len(I)

    entity_data = obs_enriched[obs_enriched['unit_id'] == unit_id]
    signals = entity_data['signal_id'].unique()

    if len(signals) == 0:
        return {
            'energy_proxy': np.full(n, np.nan),
            'energy_velocity': np.full(n, np.nan),
            'energy_flow_asymmetry': np.full(n, np.nan),
        }

    has_dy = 'dy' in entity_data.columns

    energy_records = []

    for signal in signals:
        signal_data = entity_data[entity_data['signal_id'] == signal].sort_values('I')

        y = signal_data['value'].values
        signal_I = signal_data['I'].values

        if has_dy:
            dy = signal_data['dy'].fillna(0).values
        else:
            dy = np.gradient(y, signal_I) if len(signal_I) > 1 else np.zeros_like(y)

        signal_energy = y**2 + dy**2

        for i_val, e in zip(signal_I, signal_energy):
            energy_records.append({'I': i_val, 'signal': signal, 'energy': e})

    if not energy_records:
        return {
            'energy_proxy': np.full(n, np.nan),
            'energy_velocity': np.full(n, np.nan),
            'energy_flow_asymmetry': np.full(n, np.nan),
        }

    energy_df = pd.DataFrame(energy_records)

    energy_by_I = energy_df.groupby('I').agg({
        'energy': ['sum', list]
    }).reset_index()
    energy_by_I.columns = ['I', 'total_energy', 'energy_list']
    energy_by_I = energy_by_I.sort_values('I')

    energy_I = energy_by_I['I'].values
    energy_proxy = energy_by_I['total_energy'].values

    if len(energy_I) > 1:
        energy_velocity = np.gradient(energy_proxy, energy_I)
    else:
        energy_velocity = np.zeros_like(energy_proxy)

    def gini(values):
        values = np.array(values)
        values = values[~np.isnan(values)]
        if len(values) == 0 or np.sum(values) == 0:
            return 0.0
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n

    energy_flow_asymmetry = np.array([
        gini(el) for el in energy_by_I['energy_list'].values
    ])

    if not np.array_equal(energy_I, I):
        energy_proxy = np.interp(I, energy_I, energy_proxy)
        energy_velocity = np.interp(I, energy_I, energy_velocity)
        energy_flow_asymmetry = np.interp(I, energy_I, energy_flow_asymmetry)

    return {
        'energy_proxy': energy_proxy,
        'energy_velocity': energy_velocity,
        'energy_flow_asymmetry': energy_flow_asymmetry,
    }


# ============================================================
# L4: THERMODYNAMICS
# ============================================================

def compute_thermodynamics(
    energy_proxy: np.ndarray,
    obs_enriched: pd.DataFrame,
    unit_id: str,
    I: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute thermodynamic metrics."""
    n = len(I)

    total_energy = energy_proxy.copy()

    if len(I) > 1:
        energy_change = np.gradient(total_energy, I)
        dissipation_rate = np.maximum(0, -energy_change)
    else:
        dissipation_rate = np.zeros(n)

    entity_data = obs_enriched[obs_enriched['unit_id'] == unit_id]

    # Check for rolling entropy columns
    entropy_col = None
    for col in ['rolling_sample_entropy', 'rolling_entropy', 'rolling_permutation_entropy']:
        if col in entity_data.columns:
            entropy_col = col
            break

    if entropy_col:
        entropy_by_I = entity_data.groupby('I')[entropy_col].mean().reset_index()
        entropy_by_I = entropy_by_I.sort_values('I')

        entropy_I = entropy_by_I['I'].values
        entropy_vals = entropy_by_I[entropy_col].values

        if len(entropy_I) > 1:
            entropy_change = np.gradient(entropy_vals, entropy_I)
            entropy_production = np.maximum(0, entropy_change)
        else:
            entropy_production = np.zeros(len(entropy_I))

        if not np.array_equal(entropy_I, I):
            entropy_production = np.interp(I, entropy_I, entropy_production)
    else:
        entropy_production = np.full(n, np.nan)

    return {
        'total_energy': total_energy,
        'dissipation_rate': dissipation_rate,
        'entropy_production': entropy_production,
    }


# ============================================================
# MAIN ENTRY POINTS
# ============================================================

def compute_physics_for_entity(
    obs_enriched: pd.DataFrame,
    unit_id: str,
    n_baseline: int = 100,
    coherence_window: int = 50
) -> pd.DataFrame:
    """Compute all physics layers for a single entity."""
    entity_data = obs_enriched[obs_enriched['unit_id'] == unit_id]

    if entity_data.empty:
        return pd.DataFrame()

    I = np.sort(entity_data['I'].unique())
    n = len(I)

    if n < 10:
        return pd.DataFrame()

    state = compute_state_distance(obs_enriched, unit_id, I, n_baseline)
    coherence = compute_coherence(obs_enriched, unit_id, I, coherence_window)
    energy = compute_energy(obs_enriched, unit_id, I)
    thermo = compute_thermodynamics(energy['energy_proxy'], obs_enriched, unit_id, I)

    result = pd.DataFrame({
        'unit_id': unit_id,
        'I': I,
        # L1: State
        'state_distance': state['state_distance'],
        'state_velocity': state['state_velocity'],
        'state_acceleration': state['state_acceleration'],
        'n_metrics_used': state['n_metrics_used'],
        # L2: Coherence (eigenvalue-based)
        'coherence': coherence['coherence'],
        'coherence_velocity': coherence['coherence_velocity'],
        'effective_dim': coherence['effective_dim'],
        'eigenvalue_entropy': coherence['eigenvalue_entropy'],
        'n_pairs': coherence['n_pairs'],
        'n_signals': coherence['n_signals'],
        # L3: Mechanics
        'energy_proxy': energy['energy_proxy'],
        'energy_velocity': energy['energy_velocity'],
        'energy_flow_asymmetry': energy['energy_flow_asymmetry'],
        # L4: Thermodynamics
        'total_energy': thermo['total_energy'],
        'dissipation_rate': thermo['dissipation_rate'],
        'entropy_production': thermo['entropy_production'],
    })

    return result


def compute_physics_for_all_entities(
    obs_enriched: pd.DataFrame,
    n_baseline: int = 100,
    coherence_window: int = 50
) -> pd.DataFrame:
    """Compute physics stack for all entities."""
    entities = obs_enriched['unit_id'].unique()

    results = []
    for unit_id in entities:
        try:
            result = compute_physics_for_entity(
                obs_enriched, unit_id, n_baseline, coherence_window
            )
            if not result.empty:
                results.append(result)
        except Exception as e:
            print(f"  Warning: physics computation failed for {unit_id}: {e}")

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


def compute(obs_enriched: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Main compute function for engine interface.

    Called by ManifestRunner when physics is enabled.
    """
    return compute_physics_for_all_entities(
        obs_enriched,
        n_baseline=params.get('n_baseline', 100),
        coherence_window=params.get('coherence_window', 50),
    )
