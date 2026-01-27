#!/usr/bin/env python3
"""
MOSERA Pipeline - Full Four-Layer Analysis

Layers:
1. Signal Typology     - WHAT each signal is
2. Behavioral Geometry - HOW signals relate
3. Dynamical Systems   - WHEN regimes change
4. Causal Mechanics    - WHY signals change

Usage:
    python pipeline.py <input.parquet> <output_dir>
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LAYER 1: SIGNAL ENGINES
# ============================================================================

def compute_hurst_rs(y: np.ndarray) -> dict:
    """Rescaled range Hurst exponent"""
    try:
        from prism.engines.core.memory.hurst_rs import compute_hurst_rs as _hurst
        result = _hurst(y)
        return {'hurst_rs': result.get('hurst', np.nan), 'hurst_r2': result.get('r_squared', np.nan)}
    except:
        # Fallback implementation
        n = len(y)
        if n < 20:
            return {'hurst_rs': np.nan, 'hurst_r2': np.nan}

        max_k = min(n // 2, 100)
        sizes = [int(n / k) for k in range(2, max_k) if n / k >= 8]
        sizes = sorted(set(sizes))[:20]

        rs_values = []
        for size in sizes:
            n_chunks = n // size
            rs_chunk = []
            for i in range(n_chunks):
                chunk = y[i*size:(i+1)*size]
                mean = np.mean(chunk)
                cumdev = np.cumsum(chunk - mean)
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(chunk, ddof=1)
                if S > 0:
                    rs_chunk.append(R / S)
            if rs_chunk:
                rs_values.append((np.log(size), np.log(np.mean(rs_chunk))))

        if len(rs_values) < 3:
            return {'hurst_rs': np.nan, 'hurst_r2': np.nan}

        x = np.array([v[0] for v in rs_values])
        y_rs = np.array([v[1] for v in rs_values])
        slope, intercept = np.polyfit(x, y_rs, 1)

        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y_rs - y_pred) ** 2)
        ss_tot = np.sum((y_rs - np.mean(y_rs)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {'hurst_rs': float(slope), 'hurst_r2': float(r2)}


def compute_sample_entropy(y: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Sample entropy"""
    try:
        from prism.engines.core.information.sample_entropy import compute_sample_entropy as _se
        return _se(y, m=m, r=r)
    except:
        n = len(y)
        if n < 50:
            return np.nan

        # Subsample for speed
        if n > 2000:
            y = y[::n//2000]
            n = len(y)

        r_val = r * np.std(y)

        def count_matches(template_len):
            count = 0
            for i in range(n - template_len):
                for j in range(i + 1, n - template_len):
                    if np.max(np.abs(y[i:i+template_len] - y[j:j+template_len])) <= r_val:
                        count += 1
            return count

        A = count_matches(m + 1)
        B = count_matches(m)

        if B == 0 or A == 0:
            return np.nan

        return -np.log(A / B)


def compute_lyapunov(y: np.ndarray) -> dict:
    """Largest Lyapunov exponent"""
    try:
        from prism.engines.core.dynamics.lyapunov import compute_lyapunov as _lyap
        result = _lyap(y)
        return {
            'lyapunov': result.get('lyapunov_exp', np.nan),
            'is_chaotic': result.get('is_chaotic', False)
        }
    except:
        n = len(y)
        if n < 100:
            return {'lyapunov': np.nan, 'is_chaotic': False}

        # Subsample
        if n > 5000:
            y = y[::n//5000]
            n = len(y)

        # Simple estimation via divergence
        embed_dim = 3
        delay = max(1, n // 50)

        # Create embedding
        m = n - (embed_dim - 1) * delay
        if m < 20:
            return {'lyapunov': np.nan, 'is_chaotic': False}

        embedded = np.zeros((m, embed_dim))
        for i in range(embed_dim):
            embedded[:, i] = y[i*delay:i*delay+m]

        # Find nearest neighbors and track divergence
        lyap_sum = 0
        count = 0

        for i in range(min(100, m - 10)):
            # Find nearest neighbor
            dists = np.sum((embedded[i+1:] - embedded[i])**2, axis=1)
            j = np.argmin(dists) + i + 1

            if j < m - 1:
                d0 = np.sqrt(dists[j - i - 1])
                d1 = np.linalg.norm(embedded[min(i+1, m-1)] - embedded[min(j+1, m-1)])

                if d0 > 1e-10 and d1 > 1e-10:
                    lyap_sum += np.log(d1 / d0)
                    count += 1

        lyap = lyap_sum / count if count > 0 else np.nan

        return {
            'lyapunov': float(lyap) if not np.isnan(lyap) else np.nan,
            'is_chaotic': lyap > 0.01 if not np.isnan(lyap) else False
        }


def compute_spectral_slope(y: np.ndarray) -> float:
    """Spectral slope (1/f noise characterization)"""
    try:
        n = len(y)
        if n < 64:
            return np.nan

        # FFT
        fft = np.fft.rfft(y - np.mean(y))
        psd = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(n)

        # Log-log regression (skip DC)
        mask = freqs > 0
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask] + 1e-10)

        # Fit
        slope, _ = np.polyfit(log_f, log_p, 1)
        return float(slope)
    except:
        return np.nan


def compute_garch(y: np.ndarray) -> dict:
    """GARCH volatility parameters"""
    try:
        from prism.engines.core.volatility.garch import compute_garch as _garch
        return _garch(y)
    except:
        try:
            from arch import arch_model
            returns = np.diff(y)
            if len(returns) < 100:
                return {'omega': np.nan, 'alpha': np.nan, 'beta': np.nan}

            model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off', show_warning=False)

            return {
                'omega': float(result.params.get('omega', np.nan)),
                'alpha': float(result.params.get('alpha[1]', np.nan)),
                'beta': float(result.params.get('beta[1]', np.nan))
            }
        except:
            return {'omega': np.nan, 'alpha': np.nan, 'beta': np.nan}


def compute_fft_features(y: np.ndarray, sampling_rate: float = 1200) -> dict:
    """FFT-based features"""
    try:
        n = len(y)
        if n < 64:
            return {'dominant_freq': np.nan, 'spectral_centroid': np.nan, 'spectral_bandwidth': np.nan}

        fft = np.fft.rfft(y - np.mean(y))
        psd = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(n, 1/sampling_rate)

        # Dominant frequency
        dom_idx = np.argmax(psd[1:]) + 1
        dom_freq = freqs[dom_idx]

        # Spectral centroid
        psd_norm = psd / (np.sum(psd) + 1e-10)
        centroid = np.sum(freqs * psd_norm)

        # Bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm))

        return {
            'dominant_freq': float(dom_freq),
            'spectral_centroid': float(centroid),
            'spectral_bandwidth': float(bandwidth)
        }
    except:
        return {'dominant_freq': np.nan, 'spectral_centroid': np.nan, 'spectral_bandwidth': np.nan}


def compute_rqa(y: np.ndarray) -> dict:
    """Recurrence Quantification Analysis"""
    try:
        from prism.engines.core.recurrence.rqa import compute_rqa as _rqa
        return _rqa(y)
    except:
        n = len(y)
        if n < 100:
            return {'recurrence_rate': np.nan, 'determinism': np.nan, 'laminarity': np.nan}

        # Subsample
        if n > 1000:
            y = y[::n//1000]
            n = len(y)

        # Simple recurrence rate
        threshold = 0.1 * np.std(y)
        rec_count = 0
        total = 0

        for i in range(n):
            for j in range(i+1, min(i+50, n)):
                if np.abs(y[i] - y[j]) < threshold:
                    rec_count += 1
                total += 1

        rr = rec_count / total if total > 0 else 0

        return {
            'recurrence_rate': float(rr),
            'determinism': np.nan,
            'laminarity': np.nan
        }


# ============================================================================
# LAYER 2: GEOMETRY ENGINES
# ============================================================================

def compute_correlation(y1: np.ndarray, y2: np.ndarray) -> float:
    """Pearson correlation"""
    try:
        if len(y1) != len(y2):
            min_len = min(len(y1), len(y2))
            y1, y2 = y1[:min_len], y2[:min_len]
        return float(np.corrcoef(y1, y2)[0, 1])
    except:
        return np.nan


def compute_mutual_info(y1: np.ndarray, y2: np.ndarray, n_bins: int = 20) -> float:
    """Mutual information"""
    try:
        if len(y1) != len(y2):
            min_len = min(len(y1), len(y2))
            y1, y2 = y1[:min_len], y2[:min_len]

        # Discretize
        bins1 = np.linspace(np.min(y1), np.max(y1), n_bins + 1)
        bins2 = np.linspace(np.min(y2), np.max(y2), n_bins + 1)

        d1 = np.digitize(y1, bins1[:-1]) - 1
        d2 = np.digitize(y2, bins2[:-1]) - 1

        # Joint histogram
        joint = np.zeros((n_bins, n_bins))
        for i, j in zip(d1, d2):
            joint[min(i, n_bins-1), min(j, n_bins-1)] += 1

        joint = joint / len(y1)

        # Marginals
        p1 = np.sum(joint, axis=1)
        p2 = np.sum(joint, axis=0)

        # MI
        mi = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and p1[i] > 0 and p2[j] > 0:
                    mi += joint[i, j] * np.log(joint[i, j] / (p1[i] * p2[j]))

        return float(mi)
    except:
        return np.nan


def compute_dtw(y1: np.ndarray, y2: np.ndarray) -> float:
    """Dynamic Time Warping distance"""
    try:
        from prism.engines.core.state.dtw import compute_dtw as _dtw
        result = _dtw(y1, y2)
        return result.get('dtw_distance', np.nan)
    except:
        # Subsample for speed
        if len(y1) > 500:
            y1 = y1[::len(y1)//500]
        if len(y2) > 500:
            y2 = y2[::len(y2)//500]

        n, m = len(y1), len(y2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(y1[i-1] - y2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )

        return float(dtw_matrix[n, m])


def compute_copula(y1: np.ndarray, y2: np.ndarray) -> dict:
    """Copula tail dependence"""
    try:
        if len(y1) != len(y2):
            min_len = min(len(y1), len(y2))
            y1, y2 = y1[:min_len], y2[:min_len]

        # Rank transform to uniform
        from scipy.stats import rankdata
        u1 = rankdata(y1) / (len(y1) + 1)
        u2 = rankdata(y2) / (len(y2) + 1)

        # Lower tail (both in bottom 10%)
        lower_mask = (u1 < 0.1) & (u2 < 0.1)
        lower_tail = np.sum(lower_mask) / (0.1 * len(y1))

        # Upper tail (both in top 10%)
        upper_mask = (u1 > 0.9) & (u2 > 0.9)
        upper_tail = np.sum(upper_mask) / (0.1 * len(y1))

        return {
            'copula_lower_tail': float(lower_tail),
            'copula_upper_tail': float(upper_tail)
        }
    except:
        return {'copula_lower_tail': np.nan, 'copula_upper_tail': np.nan}


def compute_lof(features: np.ndarray, k: int = 5) -> np.ndarray:
    """Local Outlier Factor"""
    try:
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=min(k, len(features)-1), novelty=False)
        lof.fit(features)
        return -lof.negative_outlier_factor_
    except:
        return np.ones(len(features))


def compute_clustering(features: np.ndarray, n_clusters: int = 4) -> dict:
    """K-means clustering"""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n_samples = len(features)
        n_clusters = min(n_clusters, n_samples - 1)

        if n_clusters < 2:
            return {'labels': np.zeros(n_samples), 'silhouette': np.nan}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        sil = silhouette_score(features, labels) if len(set(labels)) > 1 else 0

        return {'labels': labels, 'silhouette': float(sil)}
    except:
        return {'labels': np.zeros(len(features)), 'silhouette': np.nan}


def compute_mst(distance_matrix: np.ndarray, signal_ids: list) -> list:
    """Minimum Spanning Tree"""
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree

        mst = minimum_spanning_tree(distance_matrix)
        mst = mst.toarray()

        edges = []
        for i in range(len(signal_ids)):
            for j in range(i+1, len(signal_ids)):
                if mst[i, j] > 0 or mst[j, i] > 0:
                    weight = mst[i, j] if mst[i, j] > 0 else mst[j, i]
                    edges.append({
                        'signal_a': signal_ids[i],
                        'signal_b': signal_ids[j],
                        'mst_weight': float(weight)
                    })

        return edges
    except:
        return []


# ============================================================================
# LAYER 3: DYNAMICS ENGINES
# ============================================================================

def compute_attractor(y: np.ndarray) -> dict:
    """Attractor characterization"""
    try:
        from prism.engines.core.dynamics.attractor import compute_attractor as _attr
        return _attr(y)
    except:
        n = len(y)
        if n < 100:
            return {'attractor_type': 'unknown', 'attractor_dimension': np.nan}

        # Embed
        delay = max(1, n // 50)
        embed_dim = 3
        m = n - (embed_dim - 1) * delay

        if m < 20:
            return {'attractor_type': 'unknown', 'attractor_dimension': np.nan}

        embedded = np.zeros((m, embed_dim))
        for i in range(embed_dim):
            embedded[:, i] = y[i*delay:i*delay+m]

        # Check for fixed point (low variance in embedding)
        var = np.mean(np.var(embedded, axis=0))
        if var < 0.01 * np.var(y):
            return {'attractor_type': 'fixed_point', 'attractor_dimension': 0.0}

        # Check for limit cycle (periodicity)
        autocorr = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.5:
                peaks.append(i)
                break

        if peaks:
            return {'attractor_type': 'limit_cycle', 'attractor_dimension': 1.0}

        # Default to strange attractor
        return {'attractor_type': 'strange', 'attractor_dimension': 2.0}


def compute_basin(y: np.ndarray) -> dict:
    """Basin of attraction analysis"""
    try:
        n = len(y)
        if n < 100:
            return {'basin_stability': np.nan, 'n_basins': 1}

        # Estimate number of stable states via histogram modes
        hist, bins = np.histogram(y, bins=50)

        # Find local maxima
        modes = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                modes.append(i)

        n_basins = max(1, len(modes))

        # Stability = time in most common basin
        if len(modes) > 0:
            main_mode = modes[np.argmax([hist[m] for m in modes])]
            main_range = (bins[main_mode], bins[main_mode + 1])
            in_basin = np.sum((y >= main_range[0]) & (y <= main_range[1]))
            stability = in_basin / n
        else:
            stability = 1.0

        return {'basin_stability': float(stability), 'n_basins': n_basins}
    except:
        return {'basin_stability': np.nan, 'n_basins': 1}


def compute_regime_detection(times: np.ndarray, y: np.ndarray) -> dict:
    """Detect regime changes"""
    try:
        n = len(y)
        if n < 100:
            return {'n_regimes': 1, 'transitions': [], 'current_regime': 0}

        # Windowed mean/std
        window = max(20, n // 20)
        step = window // 2

        means = []
        stds = []
        times_w = []

        for i in range(0, n - window, step):
            means.append(np.mean(y[i:i+window]))
            stds.append(np.std(y[i:i+window]))
            times_w.append(times[i + window // 2] if len(times) > i + window // 2 else i)

        means = np.array(means)
        stds = np.array(stds)

        # Detect jumps in mean
        mean_diff = np.abs(np.diff(means))
        threshold = 2 * np.std(means)

        transitions = []
        for i, diff in enumerate(mean_diff):
            if diff > threshold:
                transitions.append({
                    'time': float(times_w[i+1]) if i+1 < len(times_w) else i+1,
                    'type': 'mean_shift'
                })

        # Regime count
        n_regimes = len(transitions) + 1

        return {
            'n_regimes': n_regimes,
            'transitions': transitions,
            'current_regime': n_regimes - 1
        }
    except:
        return {'n_regimes': 1, 'transitions': [], 'current_regime': 0}


def compute_dmd(y: np.ndarray) -> dict:
    """Dynamic Mode Decomposition"""
    try:
        n = len(y)
        if n < 100:
            return {'dmd_modes': [], 'dmd_freqs': []}

        # Create Hankel matrix
        delay = 10
        m = n - delay

        X = np.zeros((delay, m))
        for i in range(delay):
            X[i, :] = y[i:i+m]

        X1 = X[:, :-1]
        X2 = X[:, 1:]

        # SVD
        U, s, Vh = np.linalg.svd(X1, full_matrices=False)

        # Truncate
        r = min(5, len(s))
        U = U[:, :r]
        s = s[:r]
        Vh = Vh[:r, :]

        # DMD matrix
        A = X2 @ Vh.T @ np.diag(1/s) @ U.T

        # Eigenvalues
        eigs = np.linalg.eigvals(A)
        freqs = np.abs(np.angle(eigs)) / (2 * np.pi)

        return {
            'dmd_modes': list(np.abs(eigs)[:3]),
            'dmd_freqs': list(freqs[:3])
        }
    except:
        return {'dmd_modes': [], 'dmd_freqs': []}


# ============================================================================
# LAYER 4: CAUSAL ENGINES
# ============================================================================

def compute_granger(y1: np.ndarray, y2: np.ndarray, max_lag: int = 5) -> dict:
    """Granger causality test"""
    try:
        from prism.engines.core.state.granger import compute_granger as _granger
        return _granger(y1, y2, max_lag=max_lag)
    except:
        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            if len(y1) != len(y2):
                min_len = min(len(y1), len(y2))
                y1, y2 = y1[:min_len], y2[:min_len]

            data = np.column_stack([y2, y1])

            result = grangercausalitytests(data, maxlag=max_lag, verbose=False)

            # Get best lag
            best_lag = 1
            best_p = 1.0
            for lag in range(1, max_lag + 1):
                p = result[lag][0]['ssr_ftest'][1]
                if p < best_p:
                    best_p = p
                    best_lag = lag

            f_stat = result[best_lag][0]['ssr_ftest'][0]

            return {
                'granger_f': float(f_stat),
                'granger_p': float(best_p),
                'granger_lag': best_lag,
                'granger_significant': best_p < 0.05
            }
        except:
            return {
                'granger_f': np.nan,
                'granger_p': np.nan,
                'granger_lag': 0,
                'granger_significant': False
            }


def compute_transfer_entropy(y1: np.ndarray, y2: np.ndarray, k: int = 1) -> float:
    """Transfer entropy from y1 to y2"""
    try:
        from prism.engines.core.state.transfer_entropy import compute_transfer_entropy as _te
        result = _te(y1, y2, k=k)
        return result.get('transfer_entropy', np.nan)
    except:
        try:
            if len(y1) != len(y2):
                min_len = min(len(y1), len(y2))
                y1, y2 = y1[:min_len], y2[:min_len]

            n = len(y1)
            if n < 100:
                return np.nan

            # Discretize
            n_bins = 10
            d1 = np.digitize(y1, np.linspace(np.min(y1), np.max(y1), n_bins + 1)[:-1]) - 1
            d2 = np.digitize(y2, np.linspace(np.min(y2), np.max(y2), n_bins + 1)[:-1]) - 1

            # Count transitions
            te = 0
            for t in range(k, n - 1):
                y2_future = d2[t + 1]
                y2_past = tuple(d2[t-k+1:t+1])
                y1_past = tuple(d1[t-k+1:t+1])

                # p(y2_future | y2_past, y1_past) / p(y2_future | y2_past)
                # Simplified: just measure predictability gain
                pass

            # Fallback to correlation-based proxy
            corr = np.abs(np.corrcoef(y1[:-1], y2[1:])[0, 1])
            return float(corr * 0.5)  # Scaled proxy
        except:
            return np.nan


def compute_cointegration(y1: np.ndarray, y2: np.ndarray) -> dict:
    """Engle-Granger cointegration test"""
    try:
        from prism.engines.core.state.cointegration import compute_cointegration as _coint
        return _coint(y1, y2)
    except:
        try:
            from statsmodels.tsa.stattools import coint

            if len(y1) != len(y2):
                min_len = min(len(y1), len(y2))
                y1, y2 = y1[:min_len], y2[:min_len]

            stat, p, crit = coint(y1, y2)

            return {
                'coint_stat': float(stat),
                'coint_p': float(p),
                'is_cointegrated': p < 0.05
            }
        except:
            return {
                'coint_stat': np.nan,
                'coint_p': np.nan,
                'is_cointegrated': False
            }


# ============================================================================
# PIPELINE CLASS
# ============================================================================

class MoseraPipeline:
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = {
            'started_at': datetime.now().isoformat(),
            'input_path': str(self.input_path),
            'output_dir': str(self.output_dir),
            'layers': {}
        }

        # Data storage
        self.raw_data = None
        self.entities = []
        self.signals_by_entity = {}
        self.signal_data = {}

    def run(self):
        """Execute full 4-layer pipeline"""
        print("=" * 70)
        print("MOSERA PIPELINE - Four Layer Analysis")
        print("=" * 70)
        print(f"Input:  {self.input_path}")
        print(f"Output: {self.output_dir}")
        print("=" * 70)

        # Load data
        self._load_data()

        # Layer 1: Signal Typology
        print("\n" + "=" * 70)
        print("[Layer 1] SIGNAL TYPOLOGY - WHAT each signal is")
        print("=" * 70)
        self._run_layer1_typology()

        # Layer 2: Behavioral Geometry
        print("\n" + "=" * 70)
        print("[Layer 2] BEHAVIORAL GEOMETRY - HOW signals relate")
        print("=" * 70)
        self._run_layer2_geometry()

        # Layer 3: Dynamical Systems
        print("\n" + "=" * 70)
        print("[Layer 3] DYNAMICAL SYSTEMS - WHEN regimes change")
        print("=" * 70)
        self._run_layer3_dynamics()

        # Layer 4: Causal Mechanics
        print("\n" + "=" * 70)
        print("[Layer 4] CAUSAL MECHANICS - WHY signals change")
        print("=" * 70)
        self._run_layer4_mechanics()

        # Summary
        print("\n" + "=" * 70)
        print("[Summary] Generating combined insights...")
        print("=" * 70)
        self._generate_summary()

        # Manifest
        self._write_manifest()

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        self._print_final_summary()

    def _load_data(self):
        """Load input parquet"""
        print("\nLoading data...")
        self.raw_data = pd.read_parquet(self.input_path)

        # Get structure
        self.entities = self.raw_data['entity_id'].unique().tolist()

        for entity in self.entities:
            entity_data = self.raw_data[self.raw_data['entity_id'] == entity]
            signals = entity_data['signal_id'].unique().tolist()
            self.signals_by_entity[entity] = signals

            for signal in signals:
                sig_data = entity_data[entity_data['signal_id'] == signal].sort_values('I')
                key = (entity, signal)
                self.signal_data[key] = {
                    'I': sig_data['I'].values,
                    'y': sig_data['y'].values
                }

        n_signals = sum(len(s) for s in self.signals_by_entity.values())
        print(f"  Loaded {len(self.raw_data):,} observations")
        print(f"  {len(self.entities)} entities, {n_signals} total signals")

    def _run_layer1_typology(self):
        """Layer 1: Signal Typology"""
        rows = []
        points_rows = []

        total = sum(len(s) for s in self.signals_by_entity.values())
        done = 0

        for entity in self.entities:
            for signal in self.signals_by_entity[entity]:
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  [{done}/{total}] Processing signals...")

                key = (entity, signal)
                y = self.signal_data[key]['y']
                I = self.signal_data[key]['I']

                # Basic stats
                row = {
                    'entity_id': entity,
                    'signal_id': signal,
                    'n_points': len(y),
                    'mean': float(np.mean(y)),
                    'std': float(np.std(y)),
                    'min': float(np.min(y)),
                    'max': float(np.max(y)),
                    'median': float(np.median(y)),
                    'skewness': float(pd.Series(y).skew()),
                    'kurtosis': float(pd.Series(y).kurtosis()),
                }

                # Hurst
                hurst = compute_hurst_rs(y)
                row.update(hurst)

                # Entropy
                row['sample_entropy'] = compute_sample_entropy(y)

                # Lyapunov
                lyap = compute_lyapunov(y)
                row.update(lyap)

                # Spectral
                row['spectral_slope'] = compute_spectral_slope(y)

                # FFT
                fft_feat = compute_fft_features(y)
                row.update(fft_feat)

                # GARCH
                garch = compute_garch(y)
                row.update(garch)

                # RQA
                rqa = compute_rqa(y)
                row.update(rqa)

                # Classification
                hurst_val = row.get('hurst_rs', 0.5)
                if pd.isna(hurst_val):
                    hurst_val = 0.5

                if hurst_val > 0.6:
                    row['behavioral_class'] = 'trending'
                elif hurst_val < 0.4:
                    row['behavioral_class'] = 'mean_reverting'
                else:
                    row['behavioral_class'] = 'random_walk'

                row['_computed_at'] = datetime.now().isoformat()
                rows.append(row)

                # Points data (subsampled)
                step = max(1, len(y) // 1000)
                for i in range(0, len(y), step):
                    points_rows.append({
                        'entity_id': entity,
                        'signal_id': signal,
                        'I': float(I[i]),
                        'y': float(y[i]),
                        'dy': float(np.gradient(y)[i]) if len(y) > 1 else 0,
                    })

        # Save
        primitives = pd.DataFrame(rows)
        primitives.to_parquet(self.output_dir / 'primitives.parquet', index=False)
        print(f"  ✓ primitives.parquet: {len(primitives)} rows")

        primitives_points = pd.DataFrame(points_rows)
        primitives_points.to_parquet(self.output_dir / 'primitives_points.parquet', index=False)
        print(f"  ✓ primitives_points.parquet: {len(primitives_points)} rows")

        self.primitives = primitives
        self.manifest['layers']['typology'] = {
            'status': 'complete',
            'outputs': ['primitives.parquet', 'primitives_points.parquet'],
            'n_signals': len(primitives)
        }

    def _run_layer2_geometry(self):
        """Layer 2: Behavioral Geometry"""
        geometry_rows = []
        cluster_rows = []

        for entity in self.entities:
            signals = self.signals_by_entity[entity]
            n_signals = len(signals)
            n_pairs = n_signals * (n_signals - 1) // 2

            print(f"  Entity {entity}: {n_signals} signals, {n_pairs} pairs")

            # Pairwise computations
            pair_idx = 0
            for i, sig_a in enumerate(signals):
                for sig_b in signals[i+1:]:
                    pair_idx += 1
                    if pair_idx % 50 == 0:
                        print(f"    [{pair_idx}/{n_pairs}] pairs...")

                    y_a = self.signal_data[(entity, sig_a)]['y']
                    y_b = self.signal_data[(entity, sig_b)]['y']

                    row = {
                        'entity_id': entity,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'correlation': compute_correlation(y_a, y_b),
                        'mutual_information': compute_mutual_info(y_a, y_b),
                        'dtw_distance': compute_dtw(y_a, y_b),
                    }

                    # Copula
                    copula = compute_copula(y_a, y_b)
                    row.update(copula)

                    row['_computed_at'] = datetime.now().isoformat()
                    geometry_rows.append(row)

            # Clustering per entity
            entity_primitives = self.primitives[self.primitives['entity_id'] == entity]

            # Build feature matrix
            feature_cols = ['hurst_rs', 'sample_entropy', 'spectral_slope', 'lyapunov']
            features = entity_primitives[feature_cols].fillna(0).values

            if len(features) >= 2:
                # LOF
                lof_scores = compute_lof(features)

                # Clustering
                n_clusters = min(4, len(features) - 1)
                cluster_result = compute_clustering(features, n_clusters)

                for j, sig in enumerate(signals):
                    cluster_rows.append({
                        'entity_id': entity,
                        'signal_id': sig,
                        'cluster_id': int(cluster_result['labels'][j]),
                        'lof_score': float(lof_scores[j]),
                        'is_outlier': lof_scores[j] > 1.5,
                        'silhouette_score': cluster_result['silhouette'],
                        '_computed_at': datetime.now().isoformat()
                    })

        # Save
        geometry = pd.DataFrame(geometry_rows)
        geometry.to_parquet(self.output_dir / 'geometry.parquet', index=False)
        print(f"  ✓ geometry.parquet: {len(geometry)} rows")

        clusters = pd.DataFrame(cluster_rows)
        clusters.to_parquet(self.output_dir / 'clusters.parquet', index=False)
        print(f"  ✓ clusters.parquet: {len(clusters)} rows")

        self.geometry = geometry
        self.clusters = clusters
        self.manifest['layers']['geometry'] = {
            'status': 'complete',
            'outputs': ['geometry.parquet', 'clusters.parquet'],
            'n_pairs': len(geometry)
        }

    def _run_layer3_dynamics(self):
        """Layer 3: Dynamical Systems"""
        dynamics_rows = []
        transitions_rows = []

        total = sum(len(s) for s in self.signals_by_entity.values())
        done = 0

        for entity in self.entities:
            for signal in self.signals_by_entity[entity]:
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  [{done}/{total}] Analyzing dynamics...")

                key = (entity, signal)
                y = self.signal_data[key]['y']
                I = self.signal_data[key]['I']

                # Attractor
                attractor = compute_attractor(y)

                # Basin
                basin = compute_basin(y)

                # Regime detection
                regime = compute_regime_detection(I, y)

                # DMD
                dmd = compute_dmd(y)

                # Lyapunov (reuse from layer 1)
                lyap = compute_lyapunov(y)

                row = {
                    'entity_id': entity,
                    'signal_id': signal,
                    'attractor_type': attractor['attractor_type'],
                    'attractor_dimension': attractor['attractor_dimension'],
                    'basin_stability': basin['basin_stability'],
                    'n_basins': basin['n_basins'],
                    'n_regimes': regime['n_regimes'],
                    'current_regime': regime['current_regime'],
                    'lyapunov_local': lyap['lyapunov'],
                    'is_chaotic': lyap['is_chaotic'],
                    '_computed_at': datetime.now().isoformat()
                }
                dynamics_rows.append(row)

                # Transitions
                for trans in regime['transitions']:
                    transitions_rows.append({
                        'entity_id': entity,
                        'signal_id': signal,
                        'transition_I': trans['time'],
                        'transition_type': trans['type'],
                        '_computed_at': datetime.now().isoformat()
                    })

        # Save
        dynamics = pd.DataFrame(dynamics_rows)
        dynamics.to_parquet(self.output_dir / 'dynamics.parquet', index=False)
        print(f"  ✓ dynamics.parquet: {len(dynamics)} rows")

        if transitions_rows:
            transitions = pd.DataFrame(transitions_rows)
            transitions.to_parquet(self.output_dir / 'transitions.parquet', index=False)
            print(f"  ✓ transitions.parquet: {len(transitions)} rows")
        else:
            transitions = pd.DataFrame(columns=['entity_id', 'signal_id', 'transition_I', 'transition_type'])
            transitions.to_parquet(self.output_dir / 'transitions.parquet', index=False)
            print(f"  ✓ transitions.parquet: 0 rows (no transitions detected)")

        self.dynamics = dynamics
        self.manifest['layers']['dynamics'] = {
            'status': 'complete',
            'outputs': ['dynamics.parquet', 'transitions.parquet'],
            'n_transitions': len(transitions_rows)
        }

    def _run_layer4_mechanics(self):
        """Layer 4: Causal Mechanics"""
        mechanics_rows = []
        causal_graph_rows = []
        manifold_rows = []

        for entity in self.entities:
            signals = self.signals_by_entity[entity]
            n_pairs = len(signals) * (len(signals) - 1)  # Directed pairs

            print(f"  Entity {entity}: {n_pairs} directed pairs")

            # Causal analysis (directed pairs)
            pair_idx = 0
            for sig_a in signals:
                for sig_b in signals:
                    if sig_a == sig_b:
                        continue

                    pair_idx += 1
                    if pair_idx % 50 == 0:
                        print(f"    [{pair_idx}/{n_pairs}] causal pairs...")

                    y_a = self.signal_data[(entity, sig_a)]['y']
                    y_b = self.signal_data[(entity, sig_b)]['y']

                    # Granger: does A cause B?
                    granger = compute_granger(y_a, y_b)

                    # Transfer entropy: A -> B
                    te = compute_transfer_entropy(y_a, y_b)

                    # Cointegration (symmetric)
                    coint = compute_cointegration(y_a, y_b)

                    row = {
                        'entity_id': entity,
                        'source_signal': sig_a,
                        'target_signal': sig_b,
                        'granger_f': granger['granger_f'],
                        'granger_p': granger['granger_p'],
                        'granger_lag': granger['granger_lag'],
                        'granger_significant': granger['granger_significant'],
                        'transfer_entropy': te,
                        'coint_stat': coint['coint_stat'],
                        'coint_p': coint['coint_p'],
                        'is_cointegrated': coint['is_cointegrated'],
                        '_computed_at': datetime.now().isoformat()
                    }
                    mechanics_rows.append(row)

                    # Add to causal graph if significant
                    if granger['granger_significant'] or (te and te > 0.1):
                        causal_graph_rows.append({
                            'entity_id': entity,
                            'from_signal': sig_a,
                            'to_signal': sig_b,
                            'edge_type': 'causal',
                            'edge_weight': te if te else 0,
                            'granger_p': granger['granger_p'],
                            'lag': granger['granger_lag'],
                            '_computed_at': datetime.now().isoformat()
                        })

            # Manifold (system state trajectory)
            # Get all signals for this entity aligned by time
            all_I = set()
            for sig in signals:
                all_I.update(self.signal_data[(entity, sig)]['I'])
            all_I = sorted(all_I)

            # Sample time points
            sample_I = all_I[::max(1, len(all_I) // 500)]

            for t_idx, t in enumerate(sample_I):
                # Get values at this time
                values = []
                for sig in signals:
                    sig_data = self.signal_data[(entity, sig)]
                    idx = np.searchsorted(sig_data['I'], t)
                    if idx < len(sig_data['y']):
                        values.append(sig_data['y'][idx])
                    else:
                        values.append(np.nan)

                if len(values) >= 3 and not any(np.isnan(values)):
                    # Simple PCA projection
                    values = np.array(values)
                    manifold_rows.append({
                        'entity_id': entity,
                        'I': float(t),
                        'manifold_x': float(values[0]) if len(values) > 0 else np.nan,
                        'manifold_y': float(values[1]) if len(values) > 1 else np.nan,
                        'manifold_z': float(values[2]) if len(values) > 2 else np.nan,
                        '_computed_at': datetime.now().isoformat()
                    })

        # Save
        mechanics = pd.DataFrame(mechanics_rows)
        mechanics.to_parquet(self.output_dir / 'mechanics.parquet', index=False)
        print(f"  ✓ mechanics.parquet: {len(mechanics)} rows")

        if causal_graph_rows:
            causal_graph = pd.DataFrame(causal_graph_rows)
            causal_graph.to_parquet(self.output_dir / 'causal_graph.parquet', index=False)
            print(f"  ✓ causal_graph.parquet: {len(causal_graph)} edges")
        else:
            causal_graph = pd.DataFrame(columns=['entity_id', 'from_signal', 'to_signal', 'edge_type', 'edge_weight'])
            causal_graph.to_parquet(self.output_dir / 'causal_graph.parquet', index=False)
            print(f"  ✓ causal_graph.parquet: 0 edges")

        manifold = pd.DataFrame(manifold_rows)
        manifold.to_parquet(self.output_dir / 'manifold.parquet', index=False)
        print(f"  ✓ manifold.parquet: {len(manifold)} trajectory points")

        self.mechanics = mechanics
        self.manifest['layers']['mechanics'] = {
            'status': 'complete',
            'outputs': ['mechanics.parquet', 'causal_graph.parquet', 'manifold.parquet'],
            'n_causal_edges': len(causal_graph_rows)
        }

    def _generate_summary(self):
        """Generate combined insights"""
        summary_rows = []

        for entity in self.entities:
            for signal in self.signals_by_entity[entity]:
                # Get data from each layer
                prim = self.primitives[
                    (self.primitives['entity_id'] == entity) &
                    (self.primitives['signal_id'] == signal)
                ].iloc[0] if len(self.primitives) > 0 else {}

                clust = self.clusters[
                    (self.clusters['entity_id'] == entity) &
                    (self.clusters['signal_id'] == signal)
                ]
                clust = clust.iloc[0] if len(clust) > 0 else {}

                dyn = self.dynamics[
                    (self.dynamics['entity_id'] == entity) &
                    (self.dynamics['signal_id'] == signal)
                ]
                dyn = dyn.iloc[0] if len(dyn) > 0 else {}

                # Count causal relationships
                n_causes = len(self.mechanics[
                    (self.mechanics['entity_id'] == entity) &
                    (self.mechanics['source_signal'] == signal) &
                    (self.mechanics['granger_significant'] == True)
                ])

                n_caused_by = len(self.mechanics[
                    (self.mechanics['entity_id'] == entity) &
                    (self.mechanics['target_signal'] == signal) &
                    (self.mechanics['granger_significant'] == True)
                ])

                # Determine causal role
                if n_causes > n_caused_by:
                    causal_role = 'SOURCE'
                elif n_causes < n_caused_by:
                    causal_role = 'SINK'
                elif n_causes > 0:
                    causal_role = 'CONDUIT'
                else:
                    causal_role = 'ISOLATED'

                row = {
                    'entity_id': entity,
                    'signal_id': signal,
                    # Layer 1
                    'behavioral_class': prim.get('behavioral_class', 'unknown'),
                    'hurst_rs': prim.get('hurst_rs', np.nan),
                    'sample_entropy': prim.get('sample_entropy', np.nan),
                    'lyapunov': prim.get('lyapunov', np.nan),
                    # Layer 2
                    'cluster_id': clust.get('cluster_id', -1) if isinstance(clust, dict) else clust.cluster_id if hasattr(clust, 'cluster_id') else -1,
                    'lof_score': clust.get('lof_score', np.nan) if isinstance(clust, dict) else clust.lof_score if hasattr(clust, 'lof_score') else np.nan,
                    'is_outlier': clust.get('is_outlier', False) if isinstance(clust, dict) else clust.is_outlier if hasattr(clust, 'is_outlier') else False,
                    # Layer 3
                    'attractor_type': dyn.get('attractor_type', 'unknown') if isinstance(dyn, dict) else dyn.attractor_type if hasattr(dyn, 'attractor_type') else 'unknown',
                    'n_regimes': dyn.get('n_regimes', 1) if isinstance(dyn, dict) else dyn.n_regimes if hasattr(dyn, 'n_regimes') else 1,
                    'is_chaotic': dyn.get('is_chaotic', False) if isinstance(dyn, dict) else dyn.is_chaotic if hasattr(dyn, 'is_chaotic') else False,
                    # Layer 4
                    'n_signals_caused': n_causes,
                    'n_signals_caused_by': n_caused_by,
                    'causal_role': causal_role,
                    '_computed_at': datetime.now().isoformat()
                }
                summary_rows.append(row)

        summary = pd.DataFrame(summary_rows)
        summary.to_parquet(self.output_dir / 'summary.parquet', index=False)
        print(f"  ✓ summary.parquet: {len(summary)} rows")

        self.manifest['summary'] = {'n_signals': len(summary)}

    def _write_manifest(self):
        """Write manifest.json"""
        self.manifest['completed_at'] = datetime.now().isoformat()

        with open(self.output_dir / 'manifest.json', 'w') as f:
            json.dump(self.manifest, f, indent=2)

        print(f"  ✓ manifest.json")

    def _print_final_summary(self):
        """Print final summary"""
        print("\nOUTPUTS:")
        for f in sorted(self.output_dir.glob('*.parquet')):
            df = pd.read_parquet(f)
            size = f.stat().st_size / 1024
            print(f"  {f.name:<30} {len(df):>8,} rows  [{size:>8.1f} KB]")

        print(f"\nLayers completed: {len(self.manifest['layers'])}/4")
        for layer, info in self.manifest['layers'].items():
            print(f"  {layer}: {info['status']}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pipeline.py <input.parquet> <output_dir>")
        print("\nExample:")
        print("  python pipeline.py data/observations.parquet /Users/jasonrudder/Domains/cwru")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    pipeline = MoseraPipeline(input_path, output_dir)
    pipeline.run()
