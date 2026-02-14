use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use super::decomposition::jacobi_eigh;


/// Compute graph Laplacian from adjacency matrix.
/// Matches: manifold.primitives.matrix.graph.laplacian_matrix
///
/// Unnormalized: L = D - A
/// Normalized: L_norm = I - D^{-1/2} A D^{-1/2}
#[pyfunction]
#[pyo3(signature = (adjacency, normalized=false))]
pub fn laplacian_matrix<'py>(
    py: Python<'py>,
    adjacency: PyReadonlyArray2<f64>,
    normalized: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let arr = adjacency.as_array();
    let n = arr.nrows();

    // Compute degrees: sum of |adjacency| per row
    let mut degrees = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            let v = arr[[i, j]];
            if !v.is_nan() {
                degrees[i] += v.abs();
            }
        }
    }

    let mut lap = vec![vec![0.0f64; n]; n];

    if normalized {
        // D^{-1/2}
        let d_inv_sqrt: Vec<f64> = degrees.iter().map(|&d| {
            if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }
        }).collect();

        // L_norm = I - D^{-1/2} A D^{-1/2}
        for i in 0..n {
            for j in 0..n {
                let val = d_inv_sqrt[i] * arr[[i, j]] * d_inv_sqrt[j];
                if i == j {
                    lap[i][j] = 1.0 - val;
                } else {
                    lap[i][j] = -val;
                }
            }
        }
    } else {
        // L = D - A
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    lap[i][j] = degrees[i] - arr[[i, j]];
                } else {
                    lap[i][j] = -arr[[i, j]];
                }
            }
        }
    }

    Ok(PyArray2::from_vec2(py, &lap)?)
}


/// Compute eigenvalues of graph Laplacian.
/// Matches: manifold.primitives.matrix.graph.laplacian_eigenvalues
///
/// Returns eigenvalues sorted ascending.
#[pyfunction]
#[pyo3(signature = (laplacian, k=None))]
pub fn laplacian_eigenvalues<'py>(
    py: Python<'py>,
    laplacian: PyReadonlyArray2<f64>,
    k: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = laplacian.as_array();
    let n = arr.nrows();

    // Convert to Vec<Vec<f64>>
    let mut mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            mat[i][j] = arr[[i, j]];
        }
    }

    let (mut eigenvalues, _) = jacobi_eigh(&mat);

    // Sort ascending (Laplacian eigenvalues are conventionally ascending)
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if let Some(k) = k {
        eigenvalues.truncate(k);
    }

    Ok(PyArray1::from_vec(py, eigenvalues))
}
