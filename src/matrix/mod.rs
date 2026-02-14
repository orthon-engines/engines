use pyo3::prelude::*;

pub mod covariance;
pub mod decomposition;
pub mod graph;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // covariance
    m.add_function(wrap_pyfunction!(covariance::covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(covariance::correlation_matrix, m)?)?;

    // decomposition
    m.add_function(wrap_pyfunction!(decomposition::eigendecomposition, m)?)?;
    m.add_function(wrap_pyfunction!(decomposition::condition_number, m)?)?;
    m.add_function(wrap_pyfunction!(decomposition::effective_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(decomposition::explained_variance_ratio, m)?)?;

    // graph
    m.add_function(wrap_pyfunction!(graph::laplacian_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(graph::laplacian_eigenvalues, m)?)?;

    Ok(())
}
