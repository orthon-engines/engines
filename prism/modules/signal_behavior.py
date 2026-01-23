"""
prism/modules/signal_behavior.py - Signal Behavior Computation

Runs all computation engines and produces raw behavioral metrics.
This is the ENGINE layer - it computes measurements without interpretation.

Engines:
    CORE ENGINES (run on all signals):
        - hurst: Long-range dependence and memory
        - entropy: Information content and complexity
        - rqa: Recurrence quantification (phase space)
        - realized_vol: Short-window volatility

    CONDITIONAL ENGINES (based on characterization):
        - spectral: Frequency domain analysis
        - wavelet: Multi-resolution analysis
        - garch: Volatility clustering
        - lyapunov: Chaos and sensitivity

    DISCONTINUITY ENGINES:
        - break_detector: Detect structural breaks
        - heaviside: Persistent level shifts
        - dirac: Transient impulses

    TRANSFORM ENGINES:
        - hilbert: Amplitude, phase, instantaneous frequency
        - derivatives: Rate of change features
        - statistical: Basic statistical baseline

Usage:
    from prism.modules.signal_behavior import compute_engines, compute_all_metrics

    metrics = compute_all_metrics(signal_values, engines=['hurst', 'entropy', 'rqa'])
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# ENGINE CONFIGURATION
# =============================================================================

# Minimum observations required per engine
ENGINE_MIN_OBS = {
    'hurst': 20,
    'entropy': 30,
    'lyapunov': 30,
    'garch': 50,
    'spectral': 40,
    'wavelet': 40,
    'rqa': 30,
    'realized_vol': 15,
    'hilbert': 20,
    'derivatives': 10,
    'statistical': 5,
    'break_detector': 50,
    'heaviside': 50,
    'dirac': 50,
}

# Engines by category
CORE_ENGINES = ['hurst', 'entropy', 'rqa', 'realized_vol', 'statistical']
CONDITIONAL_ENGINES = ['spectral', 'wavelet', 'garch', 'lyapunov']
DISCONTINUITY_ENGINES = ['break_detector', 'heaviside', 'dirac']
TRANSFORM_ENGINES = ['hilbert', 'derivatives']

ALL_ENGINES = CORE_ENGINES + CONDITIONAL_ENGINES + DISCONTINUITY_ENGINES + TRANSFORM_ENGINES


# =============================================================================
# INDIVIDUAL ENGINE COMPUTATIONS
# =============================================================================

def compute_hurst(values: np.ndarray) -> Dict[str, float]:
    """Compute Hurst exponent and related metrics."""
    metrics = {}

    try:
        from nolds import hurst_rs
        h = hurst_rs(values)
        metrics['hurst_exponent'] = float(h) if not np.isnan(h) else np.nan
    except Exception:
        metrics['hurst_exponent'] = np.nan

    return metrics


def compute_entropy(values: np.ndarray) -> Dict[str, float]:
    """Compute entropy measures."""
    metrics = {}

    try:
        from antropy import sample_entropy, perm_entropy, spectral_entropy

        se = sample_entropy(values)
        metrics['sample_entropy'] = float(se) if not np.isnan(se) else np.nan

        pe = perm_entropy(values, normalize=True)
        metrics['permutation_entropy'] = float(pe) if not np.isnan(pe) else np.nan

        spe = spectral_entropy(values, sf=1.0, normalize=True)
        metrics['spectral_entropy'] = float(spe) if not np.isnan(spe) else np.nan
    except Exception:
        metrics['sample_entropy'] = np.nan
        metrics['permutation_entropy'] = np.nan
        metrics['spectral_entropy'] = np.nan

    return metrics


def compute_rqa(values: np.ndarray, embedding_dim: int = 2, time_delay: int = 1) -> Dict[str, float]:
    """Compute Recurrence Quantification Analysis metrics."""
    metrics = {}

    try:
        from pyrqa.time_series import TimeSeries
        from pyrqa.settings import Settings
        from pyrqa.computation import RQAComputation

        # Normalize for RQA
        vals = (values - np.mean(values)) / (np.std(values) + 1e-10)

        ts = TimeSeries(vals, embedding_dimension=embedding_dim, time_delay=time_delay)
        settings = Settings(
            ts,
            analysis_type='Classic',
            similarity_measure='EuclideanMetric',
            theiler_corrector=1,
            neighbourhood='FixedRadius',
            radius=0.1
        )
        computation = RQAComputation.create(settings)
        result = computation.run()

        metrics['recurrence_rate'] = float(result.recurrence_rate)
        metrics['determinism'] = float(result.determinism)
        metrics['laminarity'] = float(result.laminarity)
        metrics['rqa_entropy'] = float(result.entropy_diagonal_lines) if hasattr(result, 'entropy_diagonal_lines') else np.nan
        metrics['avg_diagonal'] = float(result.average_diagonal_line) if hasattr(result, 'average_diagonal_line') else np.nan
        metrics['max_diagonal'] = float(result.longest_diagonal_line) if hasattr(result, 'longest_diagonal_line') else np.nan
        metrics['avg_vertical'] = float(result.average_vertical_line) if hasattr(result, 'average_vertical_line') else np.nan
    except Exception:
        metrics['recurrence_rate'] = np.nan
        metrics['determinism'] = np.nan
        metrics['laminarity'] = np.nan
        metrics['rqa_entropy'] = np.nan
        metrics['avg_diagonal'] = np.nan
        metrics['max_diagonal'] = np.nan
        metrics['avg_vertical'] = np.nan

    return metrics


def compute_realized_vol(values: np.ndarray) -> Dict[str, float]:
    """Compute realized volatility and distribution metrics."""
    metrics = {}

    try:
        returns = np.diff(values) / (np.abs(values[:-1]) + 1e-10)

        # Realized volatility
        metrics['realized_volatility'] = float(np.sqrt(np.sum(returns ** 2)))

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1e-10)
        metrics['max_drawdown'] = float(np.max(drawdown))

        # Return distribution
        metrics['return_mean'] = float(np.mean(returns))
        metrics['return_std'] = float(np.std(returns))
        metrics['return_skew'] = float(_skewness(returns))
        metrics['return_kurtosis'] = float(_kurtosis(returns))
    except Exception:
        metrics['realized_volatility'] = np.nan
        metrics['max_drawdown'] = np.nan
        metrics['return_mean'] = np.nan
        metrics['return_std'] = np.nan
        metrics['return_skew'] = np.nan
        metrics['return_kurtosis'] = np.nan

    return metrics


def compute_spectral(values: np.ndarray) -> Dict[str, float]:
    """Compute spectral/frequency domain features."""
    metrics = {}

    try:
        fft = np.fft.rfft(values)
        freqs = np.fft.rfftfreq(len(values))
        power = np.abs(fft) ** 2

        # Dominant frequency
        if len(power) > 1:
            metrics['dominant_freq'] = float(freqs[np.argmax(power[1:]) + 1])
        else:
            metrics['dominant_freq'] = 0.0

        # Spectral centroid
        total_power = np.sum(power) + 1e-10
        metrics['spectral_centroid'] = float(np.sum(freqs * power) / total_power)

        # Spectral bandwidth
        centroid = metrics['spectral_centroid']
        metrics['spectral_bandwidth'] = float(
            np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power)
        )

        # Spectral slope (log-log regression)
        positive_mask = (freqs > 0) & (power > 0)
        if np.sum(positive_mask) > 2:
            log_freqs = np.log10(freqs[positive_mask])
            log_power = np.log10(power[positive_mask])
            coeffs = np.polyfit(log_freqs, log_power, 1)
            metrics['spectral_slope'] = float(coeffs[0])

            # R-squared
            pred = np.polyval(coeffs, log_freqs)
            ss_res = np.sum((log_power - pred) ** 2)
            ss_tot = np.sum((log_power - np.mean(log_power)) ** 2)
            metrics['spectral_slope_r2'] = float(1 - ss_res / (ss_tot + 1e-10))
        else:
            metrics['spectral_slope'] = np.nan
            metrics['spectral_slope_r2'] = np.nan

        # Low/high frequency ratio
        mid_idx = len(freqs) // 2
        low_power = np.sum(power[:mid_idx])
        high_power = np.sum(power[mid_idx:])
        metrics['spectral_low_high_ratio'] = float(low_power / (high_power + 1e-10))
    except Exception:
        metrics['dominant_freq'] = np.nan
        metrics['spectral_centroid'] = np.nan
        metrics['spectral_bandwidth'] = np.nan
        metrics['spectral_slope'] = np.nan
        metrics['spectral_slope_r2'] = np.nan
        metrics['spectral_low_high_ratio'] = np.nan

    return metrics


def compute_wavelet(values: np.ndarray) -> Dict[str, float]:
    """Compute wavelet-based features."""
    metrics = {}

    try:
        import pywt

        # Multi-level wavelet decomposition
        max_level = min(5, pywt.dwt_max_level(len(values), 'db4'))
        coeffs = pywt.wavedec(values, 'db4', level=max_level)

        # Energy at each level
        energies = [np.sum(c ** 2) for c in coeffs]
        total_energy = sum(energies) + 1e-10

        # Relative energy proportions
        for i, energy in enumerate(energies):
            metrics[f'wavelet_energy_level_{i}'] = float(energy / total_energy)

        # Dominant level (highest energy detail)
        if len(energies) > 1:
            detail_energies = energies[1:]  # Skip approximation
            metrics['wavelet_dominant_level'] = int(np.argmax(detail_energies) + 1)
        else:
            metrics['wavelet_dominant_level'] = 0
    except Exception:
        for i in range(6):
            metrics[f'wavelet_energy_level_{i}'] = np.nan
        metrics['wavelet_dominant_level'] = np.nan

    return metrics


def compute_garch(values: np.ndarray) -> Dict[str, float]:
    """Compute GARCH volatility model parameters."""
    metrics = {}

    try:
        from arch import arch_model

        returns = np.diff(values) / (np.abs(values[:-1]) + 1e-10)

        if len(returns) > 10 and np.std(returns) > 1e-10:
            model = arch_model(returns * 100, vol='GARCH', p=1, q=1, rescale=False)
            res = model.fit(disp='off', show_warning=False)

            metrics['garch_omega'] = float(res.params.get('omega', np.nan))
            metrics['garch_alpha'] = float(res.params.get('alpha[1]', np.nan))
            metrics['garch_beta'] = float(res.params.get('beta[1]', np.nan))

            # Persistence
            alpha = metrics['garch_alpha']
            beta = metrics['garch_beta']
            if not np.isnan(alpha) and not np.isnan(beta):
                metrics['garch_persistence'] = float(alpha + beta)
            else:
                metrics['garch_persistence'] = np.nan

            # Unconditional volatility
            omega = metrics['garch_omega']
            persistence = metrics['garch_persistence']
            if not np.isnan(omega) and not np.isnan(persistence) and persistence < 1:
                metrics['garch_unconditional_vol'] = float(np.sqrt(omega / (1 - persistence)))
            else:
                metrics['garch_unconditional_vol'] = np.nan
        else:
            raise ValueError("Insufficient data for GARCH")
    except Exception:
        metrics['garch_omega'] = np.nan
        metrics['garch_alpha'] = np.nan
        metrics['garch_beta'] = np.nan
        metrics['garch_persistence'] = np.nan
        metrics['garch_unconditional_vol'] = np.nan

    return metrics


def compute_lyapunov(values: np.ndarray) -> Dict[str, float]:
    """Compute Lyapunov exponent."""
    metrics = {}

    try:
        from nolds import lyap_r

        le = lyap_r(values)
        metrics['lyapunov_exponent'] = float(le) if not np.isnan(le) else np.nan

        # Embedding dimension estimate
        # Simple heuristic: use false nearest neighbors or default
        metrics['lyapunov_embedding_dim'] = 2  # Default
    except Exception:
        metrics['lyapunov_exponent'] = np.nan
        metrics['lyapunov_embedding_dim'] = np.nan

    return metrics


def compute_hilbert(values: np.ndarray) -> Dict[str, float]:
    """Compute Hilbert transform features."""
    metrics = {}

    try:
        from scipy.signal import hilbert

        analytic = hilbert(values)
        amplitude = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))

        metrics['hilbert_amplitude_mean'] = float(np.mean(amplitude))
        metrics['hilbert_amplitude_std'] = float(np.std(amplitude))
        metrics['hilbert_phase_mean'] = float(np.mean(phase))
        metrics['hilbert_phase_std'] = float(np.std(phase))

        # Instantaneous frequency
        inst_freq = np.diff(phase) / (2 * np.pi)
        metrics['hilbert_inst_freq_mean'] = float(np.mean(inst_freq))
        metrics['hilbert_inst_freq_std'] = float(np.std(inst_freq))
    except Exception:
        metrics['hilbert_amplitude_mean'] = np.nan
        metrics['hilbert_amplitude_std'] = np.nan
        metrics['hilbert_phase_mean'] = np.nan
        metrics['hilbert_phase_std'] = np.nan
        metrics['hilbert_inst_freq_mean'] = np.nan
        metrics['hilbert_inst_freq_std'] = np.nan

    return metrics


def compute_derivatives(values: np.ndarray) -> Dict[str, float]:
    """Compute derivative/rate of change features."""
    metrics = {}

    try:
        # First derivative
        d1 = np.diff(values)
        metrics['derivative_mean'] = float(np.mean(d1))
        metrics['derivative_std'] = float(np.std(d1))
        metrics['derivative_max'] = float(np.max(np.abs(d1)))

        # Second derivative (acceleration)
        if len(d1) > 1:
            d2 = np.diff(d1)
            metrics['acceleration_mean'] = float(np.mean(d2))
            metrics['acceleration_std'] = float(np.std(d2))
        else:
            metrics['acceleration_mean'] = np.nan
            metrics['acceleration_std'] = np.nan
    except Exception:
        metrics['derivative_mean'] = np.nan
        metrics['derivative_std'] = np.nan
        metrics['derivative_max'] = np.nan
        metrics['acceleration_mean'] = np.nan
        metrics['acceleration_std'] = np.nan

    return metrics


def compute_statistical(values: np.ndarray) -> Dict[str, float]:
    """Compute statistical baseline metrics."""
    metrics = {}

    try:
        metrics['stat_mean'] = float(np.mean(values))
        metrics['stat_std'] = float(np.std(values))
        metrics['stat_min'] = float(np.min(values))
        metrics['stat_max'] = float(np.max(values))
        metrics['stat_range'] = float(np.max(values) - np.min(values))
        metrics['stat_skew'] = float(_skewness(values))
        metrics['stat_kurtosis'] = float(_kurtosis(values))

        # Z-score of last value
        std = metrics['stat_std']
        if std > 1e-10:
            metrics['stat_zscore'] = float((values[-1] - metrics['stat_mean']) / std)
        else:
            metrics['stat_zscore'] = 0.0
    except Exception:
        metrics['stat_mean'] = np.nan
        metrics['stat_std'] = np.nan
        metrics['stat_min'] = np.nan
        metrics['stat_max'] = np.nan
        metrics['stat_range'] = np.nan
        metrics['stat_skew'] = np.nan
        metrics['stat_kurtosis'] = np.nan
        metrics['stat_zscore'] = np.nan

    return metrics


def compute_break_detector(values: np.ndarray) -> Dict[str, Any]:
    """Compute break detection metrics."""
    metrics = {}

    try:
        from prism.engines.break_detector import compute_breaks, get_break_metrics

        breaks = compute_breaks(values)
        break_metrics = get_break_metrics(breaks, len(values))

        metrics['break_detected'] = len(breaks) > 0
        metrics['break_n_breaks'] = len(breaks)
        metrics['break_mean_interval'] = break_metrics.get('mean_interval', np.nan)
        metrics['break_interval_cv'] = break_metrics.get('interval_cv', np.nan)
        metrics['break_dominant_period'] = break_metrics.get('dominant_period', np.nan)
        metrics['break_is_accelerating'] = break_metrics.get('is_accelerating', False)
    except Exception:
        metrics['break_detected'] = False
        metrics['break_n_breaks'] = 0
        metrics['break_mean_interval'] = np.nan
        metrics['break_interval_cv'] = np.nan
        metrics['break_dominant_period'] = np.nan
        metrics['break_is_accelerating'] = False

    return metrics


def compute_heaviside(values: np.ndarray) -> Dict[str, Any]:
    """Compute Heaviside (step function) detection metrics."""
    metrics = {}

    try:
        from prism.engines.heaviside import get_heaviside_metrics

        h_metrics = get_heaviside_metrics(values)

        metrics['heaviside_detected'] = h_metrics.get('detected', False)
        metrics['heaviside_n_steps'] = h_metrics.get('n_steps', 0)
        metrics['heaviside_max_magnitude'] = h_metrics.get('max_magnitude', np.nan)
        metrics['heaviside_mean_magnitude'] = h_metrics.get('mean_magnitude', np.nan)
        metrics['heaviside_up_ratio'] = h_metrics.get('up_ratio', np.nan)
    except Exception:
        metrics['heaviside_detected'] = False
        metrics['heaviside_n_steps'] = 0
        metrics['heaviside_max_magnitude'] = np.nan
        metrics['heaviside_mean_magnitude'] = np.nan
        metrics['heaviside_up_ratio'] = np.nan

    return metrics


def compute_dirac(values: np.ndarray) -> Dict[str, Any]:
    """Compute Dirac (impulse) detection metrics."""
    metrics = {}

    try:
        from prism.engines.dirac import get_dirac_metrics

        d_metrics = get_dirac_metrics(values)

        metrics['dirac_detected'] = d_metrics.get('detected', False)
        metrics['dirac_n_impulses'] = d_metrics.get('n_impulses', 0)
        metrics['dirac_max_magnitude'] = d_metrics.get('max_magnitude', np.nan)
        metrics['dirac_mean_half_life'] = d_metrics.get('mean_half_life', np.nan)
        metrics['dirac_up_ratio'] = d_metrics.get('up_ratio', np.nan)
    except Exception:
        metrics['dirac_detected'] = False
        metrics['dirac_n_impulses'] = 0
        metrics['dirac_max_magnitude'] = np.nan
        metrics['dirac_mean_half_life'] = np.nan
        metrics['dirac_up_ratio'] = np.nan

    return metrics


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return float((np.sum((x - mean) ** 3) / n) / (std ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return float((np.sum((x - mean) ** 4) / n) / (std ** 4) - 3)


# =============================================================================
# MAIN COMPUTATION FUNCTIONS
# =============================================================================

# Engine dispatch table
ENGINE_FUNCTIONS = {
    'hurst': compute_hurst,
    'entropy': compute_entropy,
    'rqa': compute_rqa,
    'realized_vol': compute_realized_vol,
    'spectral': compute_spectral,
    'wavelet': compute_wavelet,
    'garch': compute_garch,
    'lyapunov': compute_lyapunov,
    'hilbert': compute_hilbert,
    'derivatives': compute_derivatives,
    'statistical': compute_statistical,
    'break_detector': compute_break_detector,
    'heaviside': compute_heaviside,
    'dirac': compute_dirac,
}


def compute_engines(
    values: np.ndarray,
    engines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute metrics for specified engines.

    Args:
        values: Signal values (numpy array)
        engines: List of engine names to run (None = core engines only)

    Returns:
        Dictionary of all computed metrics
    """
    if engines is None:
        engines = CORE_ENGINES

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    all_metrics = {}

    for engine in engines:
        if engine not in ENGINE_FUNCTIONS:
            continue

        # Check minimum observations
        min_obs = ENGINE_MIN_OBS.get(engine, 10)
        if n < min_obs:
            continue

        # Run engine
        try:
            metrics = ENGINE_FUNCTIONS[engine](values)
            all_metrics.update(metrics)
        except Exception as e:
            # Log but continue
            pass

    return all_metrics


def compute_all_metrics(
    values: np.ndarray,
    characterization: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute all applicable metrics based on characterization.

    This is the main entry point for comprehensive signal behavior analysis.

    Args:
        values: Signal values
        characterization: Characterization result (determines which engines to run)

    Returns:
        Dictionary of all computed metrics
    """
    # Always run core engines
    engines_to_run = list(CORE_ENGINES)

    # Add conditional engines based on characterization
    if characterization:
        # Spectral/wavelet if periodic
        if characterization.get('ax_periodicity', 0) > 0.3:
            engines_to_run.extend(['spectral', 'wavelet'])

        # GARCH if volatile
        if characterization.get('ax_volatility', 0) > 0.3:
            engines_to_run.append('garch')

        # Lyapunov if complex
        if characterization.get('ax_complexity', 0) > 0.3:
            engines_to_run.append('lyapunov')

        # Discontinuity engines if breaks detected
        if characterization.get('n_breaks', 0) > 0 or characterization.get('has_steps') or characterization.get('has_impulses'):
            engines_to_run.extend(['heaviside', 'dirac'])

    # Always run break detector, hilbert, derivatives
    engines_to_run.extend(['break_detector', 'hilbert', 'derivatives'])

    # Deduplicate while preserving order
    seen = set()
    unique_engines = []
    for e in engines_to_run:
        if e not in seen:
            seen.add(e)
            unique_engines.append(e)

    return compute_engines(values, unique_engines)


def get_engine_list_for_characterization(characterization: Dict[str, Any]) -> List[str]:
    """
    Determine which engines to run based on characterization.

    Args:
        characterization: Characterization result dictionary

    Returns:
        List of engine names to run
    """
    engines = list(CORE_ENGINES)

    # Conditional based on 6-axis scores
    if characterization.get('ax_periodicity', 0) > 0.3:
        engines.extend(['spectral', 'wavelet'])

    if characterization.get('ax_volatility', 0) > 0.3:
        engines.append('garch')

    if characterization.get('ax_complexity', 0) > 0.3:
        engines.append('lyapunov')

    # Always include transform and break detection
    engines.extend(['hilbert', 'derivatives', 'break_detector'])

    # Heaviside/Dirac if pre-characterization detected breaks
    if characterization.get('n_breaks', 0) > 0:
        engines.extend(['heaviside', 'dirac'])

    return list(set(engines))
