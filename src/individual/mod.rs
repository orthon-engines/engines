use pyo3::prelude::*;

pub mod hurst;
pub mod entropy;
pub mod spectral;
pub mod statistics;
pub mod derivatives;
pub mod stationarity;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // hurst
    m.add_function(wrap_pyfunction!(hurst::hurst_exponent, m)?)?;

    // entropy
    m.add_function(wrap_pyfunction!(entropy::sample_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::permutation_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(entropy::approximate_entropy, m)?)?;

    // spectral
    m.add_function(wrap_pyfunction!(spectral::fft, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::psd, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::dominant_frequency, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::spectral_centroid, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::spectral_bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::spectral_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::band_power, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::acf_decay, m)?)?;
    m.add_function(wrap_pyfunction!(spectral::snr, m)?)?;

    // statistics
    m.add_function(wrap_pyfunction!(statistics::mean, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::std, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::variance, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::skewness, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::kurtosis, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::rms, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::peak_to_peak, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::crest_factor, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::zero_crossings, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::mean_crossings, m)?)?;

    // stationarity
    m.add_function(wrap_pyfunction!(stationarity::adf_test, m)?)?;

    // derivatives
    m.add_function(wrap_pyfunction!(derivatives::first_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(derivatives::second_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(derivatives::gradient, m)?)?;
    m.add_function(wrap_pyfunction!(derivatives::finite_difference, m)?)?;
    m.add_function(wrap_pyfunction!(derivatives::velocity, m)?)?;
    m.add_function(wrap_pyfunction!(derivatives::acceleration, m)?)?;
    m.add_function(wrap_pyfunction!(derivatives::jerk, m)?)?;

    Ok(())
}
