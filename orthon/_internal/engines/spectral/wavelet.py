"""
prism/modules/wavelet_microscope.py

Wavelet-based frequency analysis for degradation detection.
Identifies which frequency bands show earliest signal degradation.

PR #6: Wavelet Microscope

Key insight: Degradation often appears first at specific frequencies
before manifesting in aggregate metrics like SNR.

The Problem:
signal_to_noise_std has 60.8% importance - by far the dominant feature. But:
- Which sensors show the most SNR volatility?
- At what frequencies does SNR degrade first?
- Is there a characteristic "failure frequency band"?

The Solution:
Wavelet decomposition to identify frequency-specific degradation patterns.
"""

import numpy as np
import polars as pl
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Check for pywt availability
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    logger.warning("PyWavelets not installed. Install with: pip install PyWavelets")


@dataclass
class WaveletBand:
    """Represents a wavelet decomposition band."""
    level: int
    name: str  # 'detail' or 'approx'
    frequency_range: Tuple[float, float]  # Normalized frequency range
    coefficients: np.ndarray
    energy: float
    snr: float


def compute_wavelet_decomposition(
    signal: np.ndarray,
    wavelet: str = 'db4',
    max_level: int = None,
) -> List[WaveletBand]:
    """
    Decompose signal into frequency bands using wavelets.

    Args:
        signal: 1D signal topology
        wavelet: Wavelet family (default: Daubechies 4)
        max_level: Maximum decomposition level (default: auto)

    Returns:
        List of WaveletBand objects from high to low frequency
    """
    if not HAS_PYWT:
        raise ImportError("PyWavelets required. Install with: pip install PyWavelets")

    if len(signal) < 4:
        return []

    if max_level is None:
        max_level = pywt.dwt_max_level(len(signal), wavelet)
        max_level = max(1, min(max_level, 6))  # Cap at 6 levels

    # Multilevel decomposition
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    except Exception as e:
        logger.debug(f"Wavelet decomposition failed: {e}")
        return []

    bands = []

    # Approximation coefficients (lowest frequency)
    approx = coeffs[0]
    approx_var = np.var(approx) if len(approx) > 1 else 1e-10
    approx_energy = np.sum(approx**2)
    approx_snr = np.mean(approx**2) / (approx_var + 1e-10)

    bands.append(WaveletBand(
        level=max_level,
        name='approx',
        frequency_range=(0, 1 / (2**max_level)),
        coefficients=approx,
        energy=float(approx_energy),
        snr=float(approx_snr),
    ))

    # Detail coefficients (high to low frequency)
    for level, detail in enumerate(coeffs[1:], 1):
        if len(detail) == 0:
            continue

        detail_var = np.var(detail) if len(detail) > 1 else 1e-10
        detail_energy = np.sum(detail**2)
        detail_snr = np.mean(detail**2) / (detail_var + 1e-10)

        # Frequency range for this detail level
        freq_low = 1 / (2**(max_level - level + 2))
        freq_high = 1 / (2**(max_level - level + 1))

        bands.append(WaveletBand(
            level=max_level - level + 1,
            name=f'detail_{level}',
            frequency_range=(freq_low, freq_high),
            coefficients=detail,
            energy=float(detail_energy),
            snr=float(detail_snr),
        ))

    return bands


def compute_band_snr_evolution(
    observations: pl.DataFrame,
    signal_id: str,
    window_size: int = 21,
    step_size: int = 7,
    wavelet: str = 'db4',
) -> pl.DataFrame:
    """
    Track SNR evolution per frequency band over time.

    Args:
        observations: Signal data with 'signal_id', 'obs_date', 'value'
        signal_id: Signal to analyze
        window_size: Rolling window size
        step_size: Step between windows
        wavelet: Wavelet family

    Returns:
        DataFrame with band SNR per window
    """
    if not HAS_PYWT:
        return pl.DataFrame()

    ind_data = observations.filter(
        pl.col('signal_id') == signal_id
    ).sort('obs_date')

    if len(ind_data) == 0:
        return pl.DataFrame()

    values = ind_data['value'].to_numpy()
    dates = ind_data['obs_date'].to_list()

    if len(values) < window_size:
        return pl.DataFrame()

    results = []

    for start in range(0, len(values) - window_size + 1, step_size):
        end = start + window_size
        window = values[start:end]
        window_end = dates[end - 1]

        # Handle NaN values
        if np.any(np.isnan(window)):
            continue

        # Decompose window
        bands = compute_wavelet_decomposition(window, wavelet)

        if len(bands) == 0:
            continue

        row = {
            'signal_id': signal_id,
            'window_end': window_end,
            'window_start_idx': start,
            'total_energy': sum(b.energy for b in bands),
        }

        for band in bands:
            row[f'{band.name}_energy'] = band.energy
            row[f'{band.name}_snr'] = band.snr
            row[f'{band.name}_freq_low'] = band.frequency_range[0]
            row[f'{band.name}_freq_high'] = band.frequency_range[1]

        # Energy distribution across bands
        total = row['total_energy']
        for band in bands:
            row[f'{band.name}_energy_ratio'] = band.energy / (total + 1e-10)

        results.append(row)

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)


def identify_degradation_band(
    band_evolution: pl.DataFrame,
    threshold_pct: float = 0.2,
) -> Dict[str, Any]:
    """
    Identify which frequency band shows earliest degradation.

    Degradation signature: SNR drops while energy increases (noise injection)

    Args:
        band_evolution: Output from compute_band_snr_evolution
        threshold_pct: Minimum change to consider significant

    Returns:
        Dictionary with degradation band analysis
    """
    if len(band_evolution) < 3:
        return {}

    # Get band columns
    snr_cols = [c for c in band_evolution.columns if c.endswith('_snr')]
    energy_cols = [c for c in band_evolution.columns
                   if c.endswith('_energy') and c != 'total_energy']

    if not snr_cols:
        return {}

    results = {
        'signal_id': band_evolution['signal_id'][0],
    }

    # Compare early vs late windows
    n_windows = len(band_evolution)
    early_windows = band_evolution.head(max(1, n_windows // 3))
    late_windows = band_evolution.tail(max(1, n_windows // 3))

    degradation_scores = {}

    for snr_col in snr_cols:
        band_name = snr_col.replace('_snr', '')
        energy_col = f'{band_name}_energy'

        if energy_col not in band_evolution.columns:
            continue

        # SNR change (negative = degradation)
        early_snr = early_windows[snr_col].mean()
        late_snr = late_windows[snr_col].mean()

        if early_snr is None or late_snr is None:
            continue

        snr_change = (late_snr - early_snr) / (abs(early_snr) + 1e-10)

        # Energy change (positive = noise injection)
        early_energy = early_windows[energy_col].mean()
        late_energy = late_windows[energy_col].mean()

        if early_energy is None or late_energy is None:
            continue

        energy_change = (late_energy - early_energy) / (abs(early_energy) + 1e-10)

        # Degradation score: SNR drops AND energy increases
        # Negative SNR change * positive energy change = positive score
        degradation_score = max(0, -snr_change) * max(0, energy_change)

        degradation_scores[band_name] = {
            'snr_change': float(snr_change),
            'energy_change': float(energy_change),
            'degradation_score': float(degradation_score),
        }

        results[f'{band_name}_snr_change'] = float(snr_change)
        results[f'{band_name}_energy_change'] = float(energy_change)
        results[f'{band_name}_degradation_score'] = float(degradation_score)

    # Identify worst band
    if degradation_scores:
        worst_band = max(degradation_scores.keys(),
                        key=lambda k: degradation_scores[k]['degradation_score'])
        results['worst_degradation_band'] = worst_band
        results['worst_degradation_score'] = degradation_scores[worst_band]['degradation_score']
        results['worst_snr_change'] = degradation_scores[worst_band]['snr_change']
        results['worst_energy_change'] = degradation_scores[worst_band]['energy_change']

    return results


def run_wavelet_microscope(
    observations: pl.DataFrame,
    cohort_id: str,
    signal_ids: Optional[List[str]] = None,
    top_n_snr_variance: int = 5,
    window_size: int = 21,
) -> pl.DataFrame:
    """
    Run wavelet microscope on high-SNR-variance sensors.

    Args:
        observations: Raw observation data
        cohort_id: Cohort to analyze
        signal_ids: Specific signals (default: auto-select by variance)
        top_n_snr_variance: Number of top variance signals to analyze
        window_size: Window size for wavelet decomposition

    Returns:
        DataFrame with wavelet degradation analysis per signal
    """
    if not HAS_PYWT:
        logger.warning("PyWavelets not available, skipping wavelet analysis")
        return pl.DataFrame()

    # Filter to cohort signals
    if 'cohort_id' in observations.columns:
        cohort_obs = observations.filter(pl.col('cohort_id') == cohort_id)
    else:
        # Infer from signal_id pattern (e.g., u001_s1 -> u001)
        cohort_obs = observations.filter(
            pl.col('signal_id').str.starts_with(cohort_id + '_')
        )

    if len(cohort_obs) == 0:
        return pl.DataFrame()

    # Auto-select high variance signals if not specified
    if signal_ids is None:
        # Compute variance per signal
        var_stats = cohort_obs.group_by('signal_id').agg([
            pl.col('value').std().alias('std'),
            pl.col('value').count().alias('n'),
        ]).filter(pl.col('n') >= window_size)

        if len(var_stats) == 0:
            return pl.DataFrame()

        # Get top N by variance
        signal_ids = (
            var_stats
            .sort('std', descending=True)
            .head(top_n_snr_variance)
            ['signal_id']
            .to_list()
        )

    logger.info(f"Analyzing {len(signal_ids)} signals with wavelet microscope")

    results = []

    for ind_id in signal_ids:
        # Compute band evolution
        band_evo = compute_band_snr_evolution(
            cohort_obs, ind_id, window_size=window_size
        )

        if len(band_evo) == 0:
            continue

        # Identify degradation band
        deg_analysis = identify_degradation_band(band_evo)
        if deg_analysis:
            deg_analysis['cohort_id'] = cohort_id
            results.append(deg_analysis)

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results)


def extract_wavelet_features(
    wavelet_results: pl.DataFrame,
    cohort_id: str,
) -> Dict[str, Any]:
    """
    Extract cohort-level features from wavelet analysis.

    Args:
        wavelet_results: Output from run_wavelet_microscope
        cohort_id: Cohort identifier

    Returns:
        Dictionary of wavelet-derived features
    """
    if len(wavelet_results) == 0:
        return {'cohort_id': cohort_id}

    cohort_data = wavelet_results.filter(pl.col('cohort_id') == cohort_id)

    if len(cohort_data) == 0:
        return {'cohort_id': cohort_id}

    features = {'cohort_id': cohort_id}

    # Aggregate degradation scores across signals
    if 'worst_degradation_score' in cohort_data.columns:
        features['wavelet_max_degradation'] = float(cohort_data['worst_degradation_score'].max())
        features['wavelet_mean_degradation'] = float(cohort_data['worst_degradation_score'].mean())
        features['wavelet_n_degrading'] = int((cohort_data['worst_degradation_score'] > 0.1).sum())

    if 'worst_snr_change' in cohort_data.columns:
        features['wavelet_worst_snr_change'] = float(cohort_data['worst_snr_change'].min())
        features['wavelet_mean_snr_change'] = float(cohort_data['worst_snr_change'].mean())

    if 'worst_energy_change' in cohort_data.columns:
        features['wavelet_max_energy_change'] = float(cohort_data['worst_energy_change'].max())
        features['wavelet_mean_energy_change'] = float(cohort_data['worst_energy_change'].mean())

    # Most common degradation band
    if 'worst_degradation_band' in cohort_data.columns:
        bands = cohort_data['worst_degradation_band'].to_list()
        if bands:
            from collections import Counter
            band_counts = Counter(bands)
            features['wavelet_dominant_band'] = band_counts.most_common(1)[0][0]
            features['wavelet_dominant_band_count'] = band_counts.most_common(1)[0][1]

    # Band-specific aggregates
    snr_change_cols = [c for c in cohort_data.columns if c.endswith('_snr_change')]
    for col in snr_change_cols:
        band = col.replace('_snr_change', '')
        vals = cohort_data[col].drop_nulls()
        if len(vals) > 0:
            features[f'wavelet_{band}_snr_change_mean'] = float(vals.mean())
            features[f'wavelet_{band}_snr_change_min'] = float(vals.min())

    return features


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Wavelet microscope for degradation detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
WHAT IT DOES:
  Identifies which frequency bands show earliest signal degradation.

  Degradation signature: SNR drops while energy increases (noise injection)

OUTPUT:
  wavelet_analysis.parquet with columns:
    - signal_id, cohort_id
    - {band}_snr_change: SNR change from early to late lifecycle
    - {band}_energy_change: Energy change from early to late lifecycle
    - {band}_degradation_score: Combined degradation metric
    - worst_degradation_band: Which frequency band degraded most
    - worst_degradation_score: How severe the degradation was

KEY INSIGHT:
  High-frequency bands often degrade first as vibrations increase,
  followed by mid-frequency resonance, then low-frequency structural failure.
        '''
    )

    parser.add_argument('--input', required=True, help='observations.parquet')
    parser.add_argument('--output', required=True, help='wavelet_analysis.parquet')
    parser.add_argument('--domain', required=True, help='Domain ID (required)')
    parser.add_argument('--cohort', action='append', help='Cohort ID(s)')
    parser.add_argument('--all-cohorts', action='store_true', help='Process all cohorts')
    parser.add_argument('--top-n', type=int, default=5, help='Top N high-variance signals')
    parser.add_argument('--window-size', type=int, default=21, help='Wavelet window size')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    if not HAS_PYWT:
        logger.error("PyWavelets required. Install with: pip install PyWavelets")
        exit(1)

    # Use lazy scan for cohort discovery (no full file load)
    lazy_obs = pl.scan_parquet(args.input)

    # Get cohorts from lazy scan
    if args.all_cohorts:
        schema = lazy_obs.collect_schema()
        if 'cohort_id' in schema.names():
            cohorts = lazy_obs.select('cohort_id').unique().collect()['cohort_id'].to_list()
        else:
            # Infer from signal_id pattern
            ids = lazy_obs.select('signal_id').unique().collect()['signal_id'].to_list()
            cohorts = list(set(i.rsplit('_', 1)[0] for i in ids if '_' in i))
    else:
        cohorts = args.cohort or []

    all_results = []
    for cohort_id in cohorts:
        logger.info(f"Processing cohort: {cohort_id}")

        # Lazy load only this cohort's data (filter pushdown)
        schema = lazy_obs.collect_schema()
        if 'cohort_id' in schema.names():
            cohort_obs = lazy_obs.filter(pl.col('cohort_id') == cohort_id).collect()
        else:
            # Filter by signal_id pattern
            cohort_obs = lazy_obs.filter(
                pl.col('signal_id').str.starts_with(cohort_id + '_')
            ).collect()

        result = run_wavelet_microscope(
            cohort_obs, cohort_id,
            top_n_snr_variance=args.top_n,
            window_size=args.window_size
        )
        if len(result) > 0:
            all_results.append(result)

    if all_results:
        final = pl.concat(all_results)
        final.write_parquet(args.output)
        logger.info(f"Saved {len(final)} rows to {args.output}")
    else:
        logger.warning("No wavelet results generated")
