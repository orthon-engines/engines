use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

// ── Internal: shared Welch PSD computation ──────────────────────────────

/// Compute Welch PSD internally (no PyO3 overhead).
/// Returns (freqs, psd_values).
fn welch_psd_internal(y: &[f64], fs: f64, nperseg: Option<usize>) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let seg_len = nperseg.unwrap_or(n.min(256)).min(n);
    if seg_len < 2 {
        return (vec![], vec![]);
    }

    let overlap = seg_len / 2;
    let n_freqs = seg_len / 2 + 1;

    let mut psd_accum = vec![0.0; n_freqs];
    let mut n_segments = 0;

    let mut planner = FftPlanner::<f64>::new();
    let fft_algo = planner.plan_fft_forward(seg_len);

    // Hann window
    let window: Vec<f64> = (0..seg_len)
        .map(|i| {
            0.5 * (1.0
                - (2.0 * std::f64::consts::PI * i as f64 / (seg_len - 1) as f64).cos())
        })
        .collect();
    let window_power: f64 = window.iter().map(|w| w * w).sum::<f64>();

    let step = if overlap >= seg_len { 1 } else { seg_len - overlap };
    let mut start = 0;
    while start + seg_len <= n {
        let segment = &y[start..start + seg_len];

        // Detrend: remove segment mean (matches scipy detrend='constant')
        let seg_mean: f64 = segment.iter().sum::<f64>() / seg_len as f64;

        let mut buffer: Vec<Complex<f64>> = segment
            .iter()
            .zip(window.iter())
            .map(|(&v, &w)| Complex::new((v - seg_mean) * w, 0.0))
            .collect();

        fft_algo.process(&mut buffer);

        for i in 0..n_freqs {
            psd_accum[i] += buffer[i].norm_sqr();
        }

        n_segments += 1;
        start += step;
    }

    if n_segments == 0 {
        return (vec![], vec![]);
    }

    // Normalize: match scipy.signal.welch scaling
    let scale = 1.0 / (fs * window_power * n_segments as f64);
    for p in psd_accum.iter_mut() {
        *p *= scale;
    }
    // Double non-DC, non-Nyquist bins for one-sided spectrum
    if n_freqs > 2 {
        for p in psd_accum[1..n_freqs - 1].iter_mut() {
            *p *= 2.0;
        }
    }

    let freqs: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * fs / seg_len as f64)
        .collect();

    (freqs, psd_accum)
}

// ── PyO3 exports ────────────────────────────────────────────────────────

/// FFT of a real signal. Returns magnitude spectrum.
#[pyfunction]
pub fn fft<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let y = signal.as_slice()?;
    let n = y.len();

    let mut planner = FftPlanner::<f64>::new();
    let fft_algo = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = y.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft_algo.process(&mut buffer);

    let half = n / 2 + 1;
    let magnitudes: Vec<f64> = buffer[..half].iter().map(|c| c.norm()).collect();
    Ok(PyArray1::from_vec(py, magnitudes))
}

/// Power spectral density via Welch's method.
/// Matches: manifold.primitives.individual.spectral.psd
#[pyfunction]
#[pyo3(signature = (signal, fs=1.0, nperseg=None, method="welch"))]
pub fn psd<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    fs: f64,
    nperseg: Option<usize>,
    method: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let y = signal.as_slice()?;
    let (freqs, psd_vals) = welch_psd_internal(y, fs, nperseg);

    if freqs.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Signal too short for PSD computation",
        ));
    }

    Ok((
        PyArray1::from_vec(py, freqs),
        PyArray1::from_vec(py, psd_vals),
    ))
}

/// Dominant frequency (peak of PSD).
/// Matches: manifold.primitives.individual.spectral.dominant_frequency
#[pyfunction]
#[pyo3(signature = (signal, fs=1.0))]
pub fn dominant_frequency(signal: PyReadonlyArray1<f64>, fs: f64) -> PyResult<f64> {
    let y = signal.as_slice()?;
    if y.len() < 4 {
        return Ok(0.0);
    }

    let (freqs, psd_vals) = welch_psd_internal(y, fs, None);
    if freqs.is_empty() {
        return Ok(0.0);
    }

    // Find peak (skip DC at index 0)
    let mut max_power = 0.0f64;
    let mut max_idx = if freqs.len() > 1 { 1 } else { 0 };
    for i in 1..psd_vals.len() {
        if psd_vals[i] > max_power {
            max_power = psd_vals[i];
            max_idx = i;
        }
    }

    Ok(freqs[max_idx])
}

/// Spectral centroid — center of mass of PSD.
/// Matches: manifold.primitives.individual.spectral.spectral_centroid
#[pyfunction]
#[pyo3(signature = (signal, fs=1.0))]
pub fn spectral_centroid(signal: PyReadonlyArray1<f64>, fs: f64) -> PyResult<f64> {
    let y = signal.as_slice()?;
    if y.len() < 4 {
        return Ok(0.0);
    }

    let (freqs, psd_vals) = welch_psd_internal(y, fs, None);
    let total: f64 = psd_vals.iter().sum();

    if total <= 0.0 {
        return Ok(0.0);
    }

    let centroid: f64 = freqs
        .iter()
        .zip(psd_vals.iter())
        .map(|(&f, &p)| f * p)
        .sum::<f64>()
        / total;

    Ok(centroid)
}

/// Spectral bandwidth.
/// Matches: manifold.primitives.individual.spectral.spectral_bandwidth
#[pyfunction]
#[pyo3(signature = (signal, fs=1.0, p=2))]
pub fn spectral_bandwidth(signal: PyReadonlyArray1<f64>, fs: f64, p: i32) -> PyResult<f64> {
    let y = signal.as_slice()?;
    if y.len() < 4 {
        return Ok(0.0);
    }

    let (freqs, psd_vals) = welch_psd_internal(y, fs, None);
    let total: f64 = psd_vals.iter().sum();

    if total <= 0.0 {
        return Ok(0.0);
    }

    // Centroid
    let centroid: f64 = freqs
        .iter()
        .zip(psd_vals.iter())
        .map(|(&f, &p)| f * p)
        .sum::<f64>()
        / total;

    // Bandwidth
    let bw: f64 = freqs
        .iter()
        .zip(psd_vals.iter())
        .map(|(&f, &pw)| pw * (f - centroid).abs().powi(p))
        .sum::<f64>()
        / total;

    Ok(bw.powf(1.0 / p as f64))
}

/// Spectral entropy — flatness of PSD as Shannon entropy.
/// Matches: manifold.primitives.individual.spectral.spectral_entropy
#[pyfunction]
#[pyo3(signature = (signal, fs=1.0, normalize=true))]
pub fn spectral_entropy(signal: PyReadonlyArray1<f64>, fs: f64, normalize: bool) -> PyResult<f64> {
    let y = signal.as_slice()?;
    if y.len() < 4 {
        return Ok(0.0);
    }

    let (_freqs, psd_vals) = welch_psd_internal(y, fs, None);
    let total: f64 = psd_vals.iter().sum();

    if total <= 0.0 {
        return Ok(0.0);
    }

    // Shannon entropy in bits (log2)
    let mut entropy = 0.0;
    for &p in &psd_vals {
        let prob = p / total;
        if prob > 0.0 {
            entropy -= prob * prob.log2();
        }
    }

    if normalize {
        let max_entropy = (psd_vals.len() as f64).log2();
        if max_entropy > 0.0 {
            entropy /= max_entropy;
        }
    }

    Ok(entropy)
}

/// Band power — sum of PSD in a frequency band.
/// Matches: manifold.core.signal.frequency_bands.compute (single band)
#[pyfunction]
#[pyo3(signature = (signal, low, high, fs=1.0))]
pub fn band_power(signal: PyReadonlyArray1<f64>, low: f64, high: f64, fs: f64) -> PyResult<f64> {
    let y = signal.as_slice()?;
    if y.len() < 4 {
        return Ok(f64::NAN);
    }

    let (freqs, psd_vals) = welch_psd_internal(y, fs, None);

    let mut power = 0.0;
    for (&f, &p) in freqs.iter().zip(psd_vals.iter()) {
        if f >= low && f <= high {
            power += p;
        }
    }

    Ok(power)
}

/// ACF decay characteristics: acf_lag1, acf_lag10, acf_half_life.
/// Uses FFT-based autocorrelation then extracts lag values.
/// Matches: manifold.core.signal.memory.compute_acf_decay
#[pyfunction]
#[pyo3(signature = (signal, max_lag=50))]
pub fn acf_decay(signal: PyReadonlyArray1<f64>, max_lag: usize) -> PyResult<(f64, f64, f64)> {
    let y = signal.as_slice()?;
    let n = y.len();

    if n < 4 {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let ml = max_lag.min(n / 2);
    if ml < 2 {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    // Compute ACF via FFT (matches numpy correlate 'full' mode)
    let mean_val: f64 = y.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = y.iter().map(|&v| v - mean_val).collect();

    // FFT-based autocorrelation
    let fft_len = (2 * n).next_power_of_two();
    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(fft_len);
    let fft_inv = planner.plan_fft_inverse(fft_len);

    let mut buffer: Vec<Complex<f64>> = centered
        .iter()
        .map(|&v| Complex::new(v, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)).take(fft_len - n))
        .collect();

    fft_fwd.process(&mut buffer);

    // Power spectrum (|X|^2)
    for c in buffer.iter_mut() {
        *c = Complex::new(c.norm_sqr(), 0.0);
    }

    fft_inv.process(&mut buffer);

    // Normalize: autocorrelation at lag 0 = variance * n
    let acf0 = buffer[0].re / fft_len as f64;
    if acf0 <= 0.0 {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    // Normalized ACF values
    let acf_lag1 = if ml >= 1 {
        buffer[1].re / fft_len as f64 / acf0
    } else {
        f64::NAN
    };

    let acf_lag10 = if ml >= 10 {
        buffer[10].re / fft_len as f64 / acf0
    } else {
        f64::NAN
    };

    // Half-life: first lag where ACF < 0.5
    let mut half_life = f64::NAN;
    for lag in 0..=ml {
        let acf_val = buffer[lag].re / fft_len as f64 / acf0;
        if acf_val < 0.5 {
            half_life = lag as f64;
            break;
        }
    }

    Ok((acf_lag1, acf_lag10, half_life))
}

/// Signal-to-noise ratio via spectral method.
/// Matches: manifold.core.signal.snr.compute
#[pyfunction]
pub fn snr(signal: PyReadonlyArray1<f64>) -> PyResult<(f64, f64, f64, f64)> {
    let y = signal.as_slice()?;
    let n = y.len();

    if n < 8 {
        return Ok((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
    }

    // Remove DC
    let mean_val: f64 = y.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = y.iter().map(|&v| v - mean_val).collect();

    // rfft
    let mut planner = FftPlanner::<f64>::new();
    let fft_algo = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = centered
        .iter()
        .map(|&v| Complex::new(v, 0.0))
        .collect();

    fft_algo.process(&mut buffer);

    // Power spectrum, exclude DC
    let n_freqs = n / 2 + 1;
    let power: Vec<f64> = buffer[1..n_freqs].iter().map(|c| c.norm_sqr()).collect();

    if power.is_empty() {
        return Ok((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
    }

    let total_power: f64 = power.iter().sum();
    if total_power <= 0.0 {
        return Ok((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
    }

    // Noise floor: median power
    let mut sorted = power.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let noise_floor = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    // Signal: bins above 3x noise floor
    let signal_threshold = noise_floor * 3.0;
    let mut signal_power = 0.0;
    let mut noise_power = 0.0;
    for &p in &power {
        if p > signal_threshold {
            signal_power += p;
        } else {
            noise_power += p;
        }
    }

    if noise_power == 0.0 {
        noise_power = 1e-10;
    }

    let snr_linear = signal_power / noise_power;
    let snr_db = if snr_linear > 0.0 {
        10.0 * snr_linear.log10()
    } else {
        f64::NEG_INFINITY
    };

    Ok((snr_db, snr_linear, signal_power, noise_power))
}
