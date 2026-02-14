use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Pearson correlation coefficient.
/// Matches: manifold.primitives.pairwise.correlation.correlation
#[pyfunction]
pub fn correlation(sig_a: PyReadonlyArray1<f64>, sig_b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len().min(b.len());
    if n < 2 {
        return Ok(f64::NAN);
    }

    let mean_a: f64 = a[..n].iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b[..n].iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        return Ok(0.0);
    }

    Ok(cov / denom)
}

/// Covariance between two signals.
/// Matches: manifold.primitives.pairwise.correlation.covariance
#[pyfunction]
#[pyo3(signature = (sig_a, sig_b, ddof=0))]
pub fn covariance(
    sig_a: PyReadonlyArray1<f64>,
    sig_b: PyReadonlyArray1<f64>,
    ddof: usize,
) -> PyResult<f64> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len().min(b.len());
    if n <= ddof {
        return Ok(f64::NAN);
    }

    let mean_a: f64 = a[..n].iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b[..n].iter().sum::<f64>() / n as f64;

    let cov: f64 = a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(a, b)| (a - mean_a) * (b - mean_b))
        .sum::<f64>()
        / (n - ddof) as f64;

    Ok(cov)
}

/// Cross-correlation function.
/// Matches: manifold.primitives.pairwise.correlation.cross_correlation
#[pyfunction]
#[pyo3(signature = (sig_a, sig_b, max_lag=None, normalize=true))]
pub fn cross_correlation<'py>(
    py: Python<'py>,
    sig_a: PyReadonlyArray1<f64>,
    sig_b: PyReadonlyArray1<f64>,
    max_lag: Option<usize>,
    normalize: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len().min(b.len());
    let ml = max_lag.unwrap_or(n - 1);

    let mean_a: f64 = a[..n].iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b[..n].iter().sum::<f64>() / n as f64;

    let norm = if normalize {
        let var_a: f64 = a[..n].iter().map(|x| (x - mean_a).powi(2)).sum::<f64>();
        let var_b: f64 = b[..n].iter().map(|x| (x - mean_b).powi(2)).sum::<f64>();
        (var_a * var_b).sqrt()
    } else {
        n as f64
    };

    let mut result = Vec::with_capacity(2 * ml + 1);

    for lag in (-(ml as isize))..=(ml as isize) {
        let mut sum = 0.0;
        for i in 0..n {
            let j = i as isize + lag;
            if j >= 0 && (j as usize) < n {
                sum += (a[i] - mean_a) * (b[j as usize] - mean_b);
            }
        }
        result.push(if norm > 1e-15 { sum / norm } else { 0.0 });
    }

    Ok(PyArray1::from_vec(py, result))
}

/// Lag at maximum cross-correlation.
/// Matches: manifold.primitives.pairwise.correlation.lag_at_max_xcorr
#[pyfunction]
#[pyo3(signature = (sig_a, sig_b, max_lag=None))]
pub fn lag_at_max_xcorr(
    sig_a: PyReadonlyArray1<f64>,
    sig_b: PyReadonlyArray1<f64>,
    max_lag: Option<usize>,
) -> PyResult<isize> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len().min(b.len());
    let ml = max_lag.unwrap_or(n - 1);

    let mean_a: f64 = a[..n].iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b[..n].iter().sum::<f64>() / n as f64;

    let mut best_lag: isize = 0;
    let mut best_val = f64::NEG_INFINITY;

    for lag in (-(ml as isize))..=(ml as isize) {
        let mut sum = 0.0;
        for i in 0..n {
            let j = i as isize + lag;
            if j >= 0 && (j as usize) < n {
                sum += (a[i] - mean_a) * (b[j as usize] - mean_b);
            }
        }
        if sum > best_val {
            best_val = sum;
            best_lag = lag;
        }
    }

    Ok(best_lag)
}

/// Spearman rank correlation.
/// Matches: manifold.primitives.pairwise.correlation (spearman_rho)
///
/// Computes Pearson correlation of ranks.
#[pyfunction]
pub fn spearman_rho(sig_a: PyReadonlyArray1<f64>, sig_b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len().min(b.len());
    if n < 3 {
        return Ok(f64::NAN);
    }

    // Compute ranks via argsort(argsort(x))
    fn ranks(data: &[f64], n: usize) -> Vec<f64> {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| data[i].partial_cmp(&data[j]).unwrap_or(std::cmp::Ordering::Equal));
        let mut r = vec![0.0f64; n];
        for (rank, &idx) in indices.iter().enumerate() {
            r[idx] = rank as f64;
        }
        r
    }

    let ra = ranks(a, n);
    let rb = ranks(b, n);

    // Pearson correlation of ranks
    let mean_a: f64 = ra.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rb.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = ra[i] - mean_a;
        let db = rb[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        return Ok(0.0);
    }

    Ok(cov / denom)
}

/// Kendall's tau rank correlation.
/// Matches: manifold.primitives.pairwise (kendall_tau)
///
/// O(n²) direct comparison of concordant/discordant pairs.
#[pyfunction]
pub fn kendall_tau(sig_a: PyReadonlyArray1<f64>, sig_b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len().min(b.len());
    if n < 3 {
        return Ok(0.0);
    }

    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;

    for i in 0..(n - 1) {
        for j in (i + 1)..n {
            let dx = a[j] - a[i];
            let dy = b[j] - b[i];
            let product = dx * dy;
            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            }
            // ties (product == 0) are ignored (tau-a)
        }
    }

    let denom = concordant + discordant;
    if denom == 0 {
        return Ok(0.0);
    }

    Ok((concordant - discordant) as f64 / denom as f64)
}

/// Partial correlation (controlling for other variables).
/// Matches: manifold.primitives.pairwise.correlation.partial_correlation
#[pyfunction]
pub fn partial_correlation(
    sig_a: PyReadonlyArray1<f64>,
    sig_b: PyReadonlyArray1<f64>,
    controls: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let z = controls.as_slice()?;
    let n = a.len().min(b.len()).min(z.len());

    if n < 3 {
        return Ok(f64::NAN);
    }

    // Regress a and b on z, compute correlation of residuals
    let mean_z: f64 = z[..n].iter().sum::<f64>() / n as f64;
    let var_z: f64 = z[..n].iter().map(|x| (x - mean_z).powi(2)).sum::<f64>();

    let residual = |signal: &[f64]| -> Vec<f64> {
        let mean_s: f64 = signal[..n].iter().sum::<f64>() / n as f64;
        let cov_sz: f64 = signal[..n]
            .iter()
            .zip(z[..n].iter())
            .map(|(s, z)| (s - mean_s) * (z - mean_z))
            .sum();
        let beta = if var_z > 1e-15 { cov_sz / var_z } else { 0.0 };
        signal[..n]
            .iter()
            .zip(z[..n].iter())
            .map(|(s, z)| s - mean_s - beta * (z - mean_z))
            .collect()
    };

    let res_a = residual(a);
    let res_b = residual(b);

    // Correlation of residuals
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..n {
        cov += res_a[i] * res_b[i];
        va += res_a[i] * res_a[i];
        vb += res_b[i] * res_b[i];
    }
    let denom = (va * vb).sqrt();
    if denom < 1e-15 {
        return Ok(0.0);
    }
    Ok(cov / denom)
}
