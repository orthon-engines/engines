use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Jacobi eigenvalue algorithm for symmetric matrices.
/// Returns (eigenvalues, eigenvectors) — NOT sorted (caller sorts).
pub fn jacobi_eigh(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();

    // Work copy
    let mut a: Vec<Vec<f64>> = matrix.to_vec();

    // Eigenvectors accumulator (starts as identity)
    let mut v: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0usize;
        let mut q = 1usize;

        for i in 0..n {
            for j in i + 1..n {
                let abs_val = a[i][j].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Compute rotation angle
        let diff = a[p][p] - a[q][q];
        let theta = if diff.abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / diff).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply Givens rotation: A' = G^T A G
        // First, compute new diagonal elements
        let app = cos_t * cos_t * a[p][p] + 2.0 * sin_t * cos_t * a[p][q] + sin_t * sin_t * a[q][q];
        let aqq = sin_t * sin_t * a[p][p] - 2.0 * sin_t * cos_t * a[p][q] + cos_t * cos_t * a[q][q];

        // Update off-diagonal elements
        // a[p][q] and a[q][p] become 0 (by construction)
        let mut new_p_row = vec![0.0; n];
        let mut new_q_row = vec![0.0; n];

        for i in 0..n {
            if i != p && i != q {
                new_p_row[i] = cos_t * a[p][i] + sin_t * a[q][i];
                new_q_row[i] = -sin_t * a[p][i] + cos_t * a[q][i];
            }
        }

        // Write back
        for i in 0..n {
            if i != p && i != q {
                a[p][i] = new_p_row[i];
                a[i][p] = new_p_row[i];
                a[q][i] = new_q_row[i];
                a[i][q] = new_q_row[i];
            }
        }
        a[p][p] = app;
        a[q][q] = aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        // Update eigenvectors: V' = V * G
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = cos_t * vip + sin_t * viq;
            v[i][q] = -sin_t * vip + cos_t * viq;
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();

    (eigenvalues, v)
}


/// Eigendecomposition of a square matrix.
/// Matches: manifold.primitives.individual.geometry.eigendecomposition
///
/// For symmetric matrices, uses Jacobi iteration (eigh).
/// Returns (eigenvalues, eigenvectors) sorted by |eigenvalue| descending.
#[pyfunction]
#[pyo3(signature = (matrix, symmetric=true, sort_descending=true))]
pub fn eigendecomposition<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<f64>,
    symmetric: bool,
    sort_descending: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let arr = matrix.as_array();
    let n = arr.nrows();

    if n != arr.ncols() {
        return Err(pyo3::exceptions::PyValueError::new_err("Matrix must be square"));
    }

    // Check for NaN
    for i in 0..n {
        for j in 0..n {
            if arr[[i, j]].is_nan() {
                let nan_vals: Vec<f64> = vec![f64::NAN; n];
                let nan_vecs: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; n];
                return Ok((
                    PyArray1::from_vec(py, nan_vals),
                    PyArray2::from_vec2(py, &nan_vecs)?,
                ));
            }
        }
    }

    // Convert to Vec<Vec<f64>>
    let mut mat: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            mat[i][j] = arr[[i, j]];
        }
    }

    let (mut eigenvalues, mut eigenvectors) = if symmetric {
        jacobi_eigh(&mat)
    } else {
        // For non-symmetric, symmetrize and compute
        // (This is an approximation — real non-symmetric eigendecomp needs QR iteration)
        // The hot path always uses symmetric=True so this is a fallback.
        let mut sym = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                sym[i][j] = 0.5 * (mat[i][j] + mat[j][i]);
            }
        }
        jacobi_eigh(&sym)
    };

    if sort_descending {
        // Sort by |eigenvalue| descending
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b].abs().partial_cmp(&eigenvalues[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_vals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
        let sorted_vecs: Vec<Vec<f64>> = (0..n)
            .map(|row| indices.iter().map(|&col| eigenvectors[row][col]).collect())
            .collect();

        eigenvalues = sorted_vals;
        eigenvectors = sorted_vecs;
    }

    Ok((
        PyArray1::from_vec(py, eigenvalues),
        PyArray2::from_vec2(py, &eigenvectors)?,
    ))
}


/// Compute condition number of a matrix.
/// Matches: manifold.primitives.individual.geometry.condition_number
///
/// Returns ratio of largest to smallest singular value.
#[pyfunction]
#[pyo3(signature = (matrix,))]
pub fn condition_number<'py>(
    _py: Python<'py>,
    matrix: PyReadonlyArray2<f64>,
) -> PyResult<f64> {
    let arr = matrix.as_array();
    let m = arr.nrows();
    let n = arr.ncols();

    // Compute A^T A
    let mut ata = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += arr[[k, i]] * arr[[k, j]];
            }
            ata[i][j] = sum;
            ata[j][i] = sum;
        }
    }

    // Eigenvalues of A^T A = squared singular values
    let (eigenvalues, _) = jacobi_eigh(&ata);

    let mut sv: Vec<f64> = eigenvalues.iter()
        .map(|&e| if e > 1e-24 { e.sqrt() } else { 0.0 })
        .filter(|&s| s > 1e-12)
        .collect();

    if sv.is_empty() {
        return Ok(f64::INFINITY);
    }

    sv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    Ok(sv[0] / sv[sv.len() - 1])
}


/// Compute effective dimension from eigenvalues via participation ratio.
/// Matches: manifold.primitives.individual.geometry.effective_dimension
///
/// PR = (Σ|λᵢ|)² / (Σλᵢ²)
#[pyfunction]
#[pyo3(signature = (eigenvalues, method="participation_ratio"))]
pub fn effective_dimension<'py>(
    _py: Python<'py>,
    eigenvalues: numpy::PyReadonlyArray1<f64>,
    method: &str,
) -> PyResult<f64> {
    let arr = eigenvalues.as_array();
    let vals: Vec<f64> = arr.iter()
        .map(|&v| v.abs())
        .filter(|&v| v > 1e-12)
        .collect();

    if vals.is_empty() {
        return Ok(0.0);
    }

    match method {
        "participation_ratio" => {
            let sum: f64 = vals.iter().sum();
            let sum_sq: f64 = vals.iter().map(|v| v * v).sum();
            if sum_sq == 0.0 { Ok(0.0) } else { Ok(sum * sum / sum_sq) }
        }
        "normalized_entropy" => {
            let total: f64 = vals.iter().sum();
            let probs: Vec<f64> = vals.iter().map(|v| v / total).collect();
            let entropy: f64 = -probs.iter().map(|p| p * (p + 1e-12).log2()).sum::<f64>();
            Ok(2.0f64.powf(entropy))
        }
        "inverse_participation" => {
            let total: f64 = vals.iter().sum();
            let probs: Vec<f64> = vals.iter().map(|v| v / total).collect();
            let ipr: f64 = probs.iter().map(|p| p * p).sum();
            if ipr == 0.0 { Ok(0.0) } else { Ok(1.0 / ipr) }
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown method: {}", method))),
    }
}


/// Explained variance ratio from eigenvalues.
/// Matches: manifold.primitives.individual.geometry.explained_variance_ratio
#[pyfunction]
#[pyo3(signature = (eigenvalues,))]
pub fn explained_variance_ratio<'py>(
    py: Python<'py>,
    eigenvalues: numpy::PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = eigenvalues.as_array();
    let n = arr.len();
    let abs_vals: Vec<f64> = arr.iter().map(|v| v.abs()).collect();
    let total: f64 = abs_vals.iter().sum();

    let result: Vec<f64> = if total == 0.0 {
        vec![0.0; n]
    } else {
        abs_vals.iter().map(|v| v / total).collect()
    };

    Ok(PyArray1::from_vec(py, result))
}
