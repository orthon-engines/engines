"""
PRISM Domain Clock
==================

Auto-detect domain frequency from the fastest-changing signal.
The domain's clock is set by its fastest signal - all other signals
are captured relative to this timescale.

Key insight: If 10 balls bounce at different rates (20x/min vs 20x/sec),
the domain frequency is 20x/sec. The slow ball still contributes valid
observations - it just appears "static" within fast windows.

Usage:
    from prism.core.domain_clock import DomainClock
    
    clock = DomainClock(min_cycles=3)
    domain_info = clock.characterize(observations_df)
    
    # Now use domain_info['window_samples'] for all computations
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.signal import welch
from scipy.fft import fft, fftfreq
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DomainInfo:
    """Domain timing characteristics."""
    domain_frequency: float          # Hz - set by fastest signal
    fastest_signal: str              # Which signal sets the clock
    window_samples: int              # Samples per window (for vector layer)
    window_duration: float           # Seconds per window
    min_window_samples: int          # Hard floor for statistical validity
    signal_frequencies: Dict[str, float]  # Per-signal characteristic frequencies
    frequency_ratios: Dict[str, float]    # Ratio to domain frequency
    sampling_info: Dict[str, Any]    # Sampling rate info
    

class DomainClock:
    """
    Detect domain frequency from fastest-changing signal.
    
    The domain clock determines:
    - Window size for vector computations
    - Stride for rolling windows
    - Laplace s-values for cross-signal comparison
    
    Args:
        min_cycles: Minimum cycles of fastest signal per window (default: 3)
        min_samples: Hard floor for statistical validity (default: 20)
        max_samples: Hard ceiling to prevent memory issues (default: 1000)
    """
    
    def __init__(
        self, 
        min_cycles: int = 3,
        min_samples: int = 20,
        max_samples: int = 1000,
    ):
        self.min_cycles = min_cycles
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.domain_info: Optional[DomainInfo] = None
        
    def characterize(
        self, 
        observations: pl.DataFrame,
        signal_col: str = 'signal_id',
        value_col: str = 'value',
        time_col: str = 'timestamp',
    ) -> DomainInfo:
        """
        Scan all signals and detect domain frequency.
        
        Args:
            observations: DataFrame with signal observations
            signal_col: Column name for signal ID
            value_col: Column name for values
            time_col: Column name for timestamps
            
        Returns:
            DomainInfo with timing characteristics
        """
        signal_ids = observations[signal_col].unique().to_list()
        
        print(f"[DomainClock] Scanning {len(signal_ids)} signals...")
        
        signal_frequencies = {}
        sampling_rates = {}
        
        for signal_id in signal_ids:
            signal_data = observations.filter(pl.col(signal_col) == signal_id)
            
            values = signal_data[value_col].to_numpy()
            
            # Handle different timestamp types
            time_data = signal_data[time_col]
            timestamps = self._to_numeric_time(time_data)
            
            if len(values) < self.min_samples:
                continue
                
            # Estimate characteristic frequency
            freq, sampling_rate = self._estimate_frequency(values, timestamps)
            
            if freq > 0:
                signal_frequencies[signal_id] = freq
                sampling_rates[signal_id] = sampling_rate
        
        if not signal_frequencies:
            raise ValueError("No signals with enough data to estimate frequency")
        
        # Domain clock = fastest signal
        fastest_signal = max(signal_frequencies, key=signal_frequencies.get)
        domain_frequency = signal_frequencies[fastest_signal]
        
        # Compute window size
        # Window must capture min_cycles of fastest signal
        avg_sampling_rate = np.mean(list(sampling_rates.values()))
        
        if domain_frequency > 0:
            samples_per_cycle = avg_sampling_rate / domain_frequency
            window_samples = int(self.min_cycles * samples_per_cycle)
        else:
            window_samples = self.min_samples
        
        # Enforce bounds
        window_samples = max(window_samples, self.min_samples)
        window_samples = min(window_samples, self.max_samples)
        
        # Window duration
        window_duration = window_samples / avg_sampling_rate if avg_sampling_rate > 0 else 0
        
        # Frequency ratios (how much slower than domain clock)
        frequency_ratios = {
            sid: domain_frequency / max(f, 1e-10)
            for sid, f in signal_frequencies.items()
        }
        
        self.domain_info = DomainInfo(
            domain_frequency=domain_frequency,
            fastest_signal=fastest_signal,
            window_samples=window_samples,
            window_duration=window_duration,
            min_window_samples=self.min_samples,
            signal_frequencies=signal_frequencies,
            frequency_ratios=frequency_ratios,
            sampling_info={
                'avg_sampling_rate': avg_sampling_rate,
                'sampling_rates': sampling_rates,
            }
        )
        
        self._print_summary()
        
        return self.domain_info
    
    def _to_numeric_time(self, time_data: pl.Series) -> np.ndarray:
        """Convert various timestamp formats to numeric seconds."""
        dtype = time_data.dtype

        if dtype == pl.Float64 or dtype == pl.Int64:
            # Already numeric
            return time_data.to_numpy().astype(float)

        elif dtype == pl.Datetime or str(dtype).startswith('Datetime'):
            # Convert datetime to seconds since start using numpy
            timestamps = time_data.to_numpy()
            if len(timestamps) > 0:
                # numpy datetime64 - convert to float seconds via int64
                start = timestamps[0]
                # timedelta64 to seconds
                diffs = (timestamps - start).astype('timedelta64[ns]').astype(np.float64) / 1e9
                return diffs
            return np.array([])

        elif dtype == pl.Date:
            # Convert date to seconds using numpy
            dates = time_data.to_numpy()
            if len(dates) > 0:
                start = dates[0]
                # timedelta64 to days, then to seconds
                diffs = (dates - start).astype('timedelta64[D]').astype(np.float64) * 86400
                return diffs
            return np.array([])

        else:
            # Try to cast to float
            try:
                return time_data.cast(pl.Float64).to_numpy()
            except:
                # Fallback: use index as time
                return np.arange(len(time_data), dtype=float)
    
    def _estimate_frequency(
        self, 
        values: np.ndarray, 
        timestamps: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate characteristic frequency of a signal.
        
        Uses multiple methods and takes the maximum (most dynamic estimate):
        1. Spectral analysis (dominant frequency)
        2. Autocorrelation decay time
        3. Zero-crossing rate
        
        Returns:
            (characteristic_frequency, sampling_rate)
        """
        if len(values) < self.min_samples or len(timestamps) < 2:
            return 0.0, 0.0
        
        # Clean data
        mask = np.isfinite(values) & np.isfinite(timestamps)
        values = values[mask]
        timestamps = timestamps[mask]
        
        if len(values) < self.min_samples:
            return 0.0, 0.0
        
        # Estimate sampling rate
        dt = np.median(np.diff(timestamps))
        if dt <= 0:
            dt = 1.0
        sampling_rate = 1.0 / dt
        
        # Detrend and normalize
        values = values - np.mean(values)
        std = np.std(values)
        if std > 0:
            values = values / std
        
        freq_estimates = []
        
        # Method 1: Spectral - dominant frequency
        try:
            freqs, psd = welch(values, fs=sampling_rate, nperseg=min(256, len(values)//2))
            if len(psd) > 0 and np.max(psd) > 0:
                # Find peak, excluding DC (index 0)
                psd_no_dc = psd[1:] if len(psd) > 1 else psd
                freqs_no_dc = freqs[1:] if len(freqs) > 1 else freqs
                
                peak_idx = np.argmax(psd_no_dc)
                spectral_freq = freqs_no_dc[peak_idx]
                
                if spectral_freq > 0:
                    freq_estimates.append(spectral_freq)
        except Exception:
            pass
        
        # Method 2: Autocorrelation decay time
        try:
            n = len(values)
            acf = np.correlate(values, values, mode='full')
            acf = acf[n-1:]  # Keep positive lags only
            acf = acf / acf[0] if acf[0] != 0 else acf
            
            # Find first crossing below 1/e
            decay_threshold = 1.0 / np.e
            decay_idx = np.where(acf < decay_threshold)[0]
            
            if len(decay_idx) > 0 and decay_idx[0] > 0:
                decorr_time = decay_idx[0] * dt
                acf_freq = 1.0 / decorr_time
                freq_estimates.append(acf_freq)
        except Exception:
            pass
        
        # Method 3: Zero-crossing rate (for oscillatory signals)
        try:
            zero_crossings = np.where(np.diff(np.sign(values)))[0]
            if len(zero_crossings) > 1:
                total_time = timestamps[-1] - timestamps[0]
                if total_time > 0:
                    zcr_freq = len(zero_crossings) / (2 * total_time)
                    freq_estimates.append(zcr_freq)
        except Exception:
            pass
        
        # Method 4: Variance of first derivative (activity measure)
        try:
            dv = np.diff(values) / dt
            activity = np.var(dv)
            # High activity = fast changes
            # Heuristic: freq ~ sqrt(activity)
            activity_freq = np.sqrt(activity) / (2 * np.pi)
            if activity_freq > 0 and np.isfinite(activity_freq):
                freq_estimates.append(activity_freq)
        except Exception:
            pass
        
        # Take maximum (most dynamic estimate)
        if freq_estimates:
            characteristic_freq = max(freq_estimates)
        else:
            # Fallback: Nyquist / 10
            characteristic_freq = sampling_rate / 20
        
        return characteristic_freq, sampling_rate
    
    def _print_summary(self):
        """Print domain characterization summary."""
        if self.domain_info is None:
            return
            
        info = self.domain_info
        
        print()
        print("=" * 60)
        print("DOMAIN CLOCK CHARACTERIZATION")
        print("=" * 60)
        print(f"Domain frequency:    {info.domain_frequency:.6f} Hz")
        print(f"Set by signal:       {info.fastest_signal}")
        print(f"Window samples:      {info.window_samples}")
        print(f"Window duration:     {info.window_duration:.4f} seconds")
        print()
        
        # Top 5 fastest signals
        sorted_signals = sorted(
            info.signal_frequencies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        print("Top 5 fastest signals:")
        for signal_id, freq in sorted_signals:
            ratio = info.frequency_ratios[signal_id]
            print(f"  {signal_id[:40]:40s}  {freq:.6f} Hz  (ratio: {ratio:.1f}x)")
        
        print()
        print("=" * 60)
    
    def get_window_config(self) -> Dict[str, Any]:
        """Get window configuration for signal_vector."""
        if self.domain_info is None:
            raise ValueError("Must call characterize() first")
        
        info = self.domain_info
        
        # Stride = 1 cycle of domain frequency (or 1/3 of window)
        stride_samples = max(1, info.window_samples // 3)
        
        return {
            'window_samples': info.window_samples,
            'stride_samples': stride_samples,
            'min_samples': info.min_window_samples,
            'domain_frequency': info.domain_frequency,
            'fastest_signal': info.fastest_signal,
        }
    
    def get_laplace_s_values(self, n_values: int = 50) -> np.ndarray:
        """
        Get Laplace s-values spanning the domain's frequency range.
        
        The s-values span from below the slowest signal to above
        the fastest, enabling cross-signal comparison without
        time-domain interpolation.
        """
        if self.domain_info is None:
            raise ValueError("Must call characterize() first")
        
        info = self.domain_info
        
        slowest = min(info.signal_frequencies.values())
        fastest = info.domain_frequency
        
        # Span from 1/10 of slowest to 10x fastest
        s_min = slowest / 10
        s_max = fastest * 10
        
        # Logarithmically spaced for multi-scale coverage
        return np.logspace(np.log10(s_min), np.log10(s_max), n_values)
    
    def to_parquet_metadata(self) -> Dict[str, Any]:
        """Export domain info for storage alongside results."""
        if self.domain_info is None:
            return {}
        
        info = self.domain_info
        
        return {
            'domain_frequency': info.domain_frequency,
            'fastest_signal': info.fastest_signal,
            'window_samples': info.window_samples,
            'window_duration': info.window_duration,
            'n_signals_characterized': len(info.signal_frequencies),
            'avg_sampling_rate': info.sampling_info['avg_sampling_rate'],
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def auto_detect_window(
    observations: pl.DataFrame,
    min_cycles: int = 3,
    min_samples: int = 20,
) -> Dict[str, Any]:
    """
    One-liner to detect domain frequency and get window config.
    
    Usage:
        config = auto_detect_window(observations_df)
        window_samples = config['window_samples']
    """
    clock = DomainClock(min_cycles=min_cycles, min_samples=min_samples)
    clock.characterize(observations)
    return clock.get_window_config()


def characterize_domain(
    observations: pl.DataFrame,
    signal_col: str = 'signal_id',
    value_col: str = 'value',
    time_col: str = 'timestamp',
) -> DomainInfo:
    """
    Full domain characterization.
    
    Returns DomainInfo dataclass with all timing characteristics.
    """
    clock = DomainClock()
    return clock.characterize(
        observations,
        signal_col=signal_col,
        value_col=value_col,
        time_col=time_col,
    )


# =============================================================================
# INTEGRATION WITH SIGNAL_VECTOR
# =============================================================================

def integrate_with_signal_vector():
    """
    Shows how to integrate DomainClock with signal_vector entry point.
    
    The key changes to signal_vector.py:
    
    1. At start of run(), call DomainClock.characterize(observations)
    2. Replace hardcoded WINDOW_CONFIGS with domain_info
    3. Store domain_info in output parquet metadata
    """
    
    example = '''
    # In signal_vector.py, replace:
    
    WINDOW_CONFIGS = [
        {"name": "30d", "days": 30, "stride": 7},
        {"name": "90d", "days": 90, "stride": 14},
    ]
    
    # With:
    
    from prism.core.domain_clock import DomainClock
    
    def compute_adaptive_window_config(observations: pl.DataFrame) -> List[Dict]:
        """Auto-detect window from domain frequency."""
        
        clock = DomainClock(min_cycles=3, min_samples=20)
        domain_info = clock.characterize(observations)
        
        window_samples = domain_info.window_samples
        stride_samples = window_samples // 3
        
        # Convert to days if needed (for compatibility)
        # Or better: work directly in samples
        
        return [{
            "name": "adaptive",
            "samples": window_samples,
            "stride_samples": stride_samples,
            "domain_info": domain_info,
        }]
    
    # Then in main run():
    
    def run(domain: str, ...):
        observations = load_observations(domain)
        
        # Auto-detect instead of hardcoded
        window_configs = compute_adaptive_window_config(observations)
        
        for config in window_configs:
            process_window(config, observations, ...)
    '''
    
    return example


if __name__ == "__main__":
    # Test with synthetic data
    import polars as pl
    
    # Create test data with different frequencies
    np.random.seed(42)
    n = 1000
    t = np.linspace(0, 10, n)  # 10 seconds
    
    # Fast signal: 5 Hz oscillation
    fast_signal = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(n)
    
    # Medium signal: 1 Hz oscillation
    medium_signal = np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(n)
    
    # Slow signal: 0.1 Hz oscillation  
    slow_signal = np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(n)
    
    # Create DataFrame
    observations = pl.DataFrame({
        'signal_id': ['fast'] * n + ['medium'] * n + ['slow'] * n,
        'value': np.concatenate([fast_signal, medium_signal, slow_signal]),
        'obs_date': np.concatenate([t, t, t]),
    })
    
    # Characterize
    clock = DomainClock(min_cycles=3, min_samples=20)
    info = clock.characterize(observations)
    
    print(f"\nExpected: fast signal at ~5 Hz")
    print(f"Detected: {info.fastest_signal} at {info.domain_frequency:.2f} Hz")
    print(f"Window: {info.window_samples} samples")
    
    # Get Laplace s-values
    s_values = clock.get_laplace_s_values()
    print(f"\nLaplace s-values: {len(s_values)} values from {s_values[0]:.4f} to {s_values[-1]:.4f}")
