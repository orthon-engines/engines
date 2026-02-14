use pyo3::prelude::*;

pub mod correlation;
pub mod causality;
pub mod distance;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // correlation
    m.add_function(wrap_pyfunction!(correlation::correlation, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::covariance, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::cross_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::lag_at_max_xcorr, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::spearman_rho, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::kendall_tau, m)?)?;
    m.add_function(wrap_pyfunction!(correlation::partial_correlation, m)?)?;

    // causality
    m.add_function(wrap_pyfunction!(causality::granger_causality, m)?)?;
    m.add_function(wrap_pyfunction!(causality::convergent_cross_mapping, m)?)?;

    // distance
    m.add_function(wrap_pyfunction!(distance::dynamic_time_warping, m)?)?;
    m.add_function(wrap_pyfunction!(distance::euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(distance::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(distance::manhattan_distance, m)?)?;

    Ok(())
}
