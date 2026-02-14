use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Dynamic time warping distance.
/// Matches: manifold.primitives.pairwise.distance.dynamic_time_warping
///
/// Uses squared cost and returns sqrt of accumulated distance
/// to match the Python implementation.
#[pyfunction]
#[pyo3(signature = (sig_a, sig_b, window=None, return_path=false))]
pub fn dynamic_time_warping(
    sig_a: PyReadonlyArray1<f64>,
    sig_b: PyReadonlyArray1<f64>,
    window: Option<usize>,
    return_path: bool,
) -> PyResult<f64> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len();
    let m = b.len();

    if n == 0 || m == 0 {
        return Ok(f64::INFINITY);
    }

    let w = window.unwrap_or(n.max(m));

    // DP table
    let mut dtw = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dtw[0][0] = 0.0;

    for i in 1..=n {
        let j_start = if i > w { i - w } else { 1 };
        let j_end = (i + w).min(m);
        for j in j_start..=j_end {
            let cost = (a[i - 1] - b[j - 1]).powi(2);
            dtw[i][j] = cost
                + dtw[i - 1][j]
                    .min(dtw[i][j - 1])
                    .min(dtw[i - 1][j - 1]);
        }
    }

    Ok(dtw[n][m].sqrt())
}

/// Euclidean distance between two signals.
/// Matches: manifold.primitives.pairwise.distance.euclidean_distance
#[pyfunction]
#[pyo3(signature = (sig_a, sig_b, normalized=false))]
pub fn euclidean_distance(
    sig_a: PyReadonlyArray1<f64>,
    sig_b: PyReadonlyArray1<f64>,
    normalized: bool,
) -> PyResult<f64> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len().min(b.len());

    if n == 0 {
        return Ok(0.0);
    }

    let dist: f64 = a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    if normalized {
        Ok(dist / (n as f64).sqrt())
    } else {
        Ok(dist)
    }
}

/// Cosine similarity between two signals.
/// Matches: manifold.primitives.pairwise.distance.cosine_similarity
#[pyfunction]
pub fn cosine_similarity(
    sig_a: PyReadonlyArray1<f64>,
    sig_b: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len().min(b.len());

    if n == 0 {
        return Ok(0.0);
    }

    let dot: f64 = a[..n].iter().zip(b[..n].iter()).map(|(a, b)| a * b).sum();
    let norm_a: f64 = a[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b[..n].iter().map(|x| x * x).sum::<f64>().sqrt();

    let denom = norm_a * norm_b;
    if denom < 1e-15 {
        return Ok(0.0);
    }

    Ok(dot / denom)
}

/// Manhattan distance between two signals.
/// Matches: manifold.primitives.pairwise.distance.manhattan_distance
#[pyfunction]
#[pyo3(signature = (sig_a, sig_b, normalized=false))]
pub fn manhattan_distance(
    sig_a: PyReadonlyArray1<f64>,
    sig_b: PyReadonlyArray1<f64>,
    normalized: bool,
) -> PyResult<f64> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len().min(b.len());

    if n == 0 {
        return Ok(0.0);
    }

    let dist: f64 = a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    if normalized {
        Ok(dist / n as f64)
    } else {
        Ok(dist)
    }
}
