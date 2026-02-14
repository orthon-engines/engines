use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Time-delay embedding.
/// Returns (n_points, dimension) array.
/// Matches: manifold.primitives.embedding.delay.time_delay_embedding
#[pyfunction]
#[pyo3(signature = (signal, dimension=3, delay=1))]
pub fn time_delay_embedding<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dimension: usize,
    delay: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let y = signal.as_slice()?;
    let n = y.len();

    let n_points = n.saturating_sub((dimension - 1) * delay);
    if n_points == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!(
                "Signal too short for embedding: {} samples, need at least {}",
                n,
                (dimension - 1) * delay + 1
            ),
        ));
    }

    let mut result = vec![vec![0.0f64; dimension]; n_points];
    for i in 0..n_points {
        for d in 0..dimension {
            result[i][d] = y[i + d * delay];
        }
    }

    Ok(PyArray2::from_vec2(py, &result)?)
}

/// Compute lagged mutual information between signal and its time-shifted copy.
fn lagged_mi(y: &[f64], lag: usize, n_bins: usize, min_val: f64, range: f64) -> f64 {
    let n_pairs = y.len() - lag;

    let bin_idx = |v: f64| -> usize {
        let b = ((v - min_val) / range * n_bins as f64) as usize;
        b.min(n_bins - 1)
    };

    let mut joint = vec![vec![0usize; n_bins]; n_bins];
    let mut margin_a = vec![0usize; n_bins];
    let mut margin_b = vec![0usize; n_bins];

    for i in 0..n_pairs {
        let bi = bin_idx(y[i]);
        let bj = bin_idx(y[i + lag]);
        joint[bi][bj] += 1;
        margin_a[bi] += 1;
        margin_b[bj] += 1;
    }

    let total = n_pairs as f64;
    let mut mi = 0.0;
    for bi in 0..n_bins {
        for bj in 0..n_bins {
            if joint[bi][bj] > 0 {
                let pij = joint[bi][bj] as f64 / total;
                let pi = margin_a[bi] as f64 / total;
                let pj = margin_b[bj] as f64 / total;
                mi += pij * (pij / (pi * pj)).ln();
            }
        }
    }
    mi
}

/// Estimate optimal delay via AMI (Average Mutual Information).
/// Returns the lag at first minimum of AMI.
/// Matches: manifold.primitives.embedding.delay.optimal_delay
#[pyfunction]
#[pyo3(signature = (signal, max_lag=None, method="mutual_info"))]
pub fn optimal_delay(
    signal: PyReadonlyArray1<f64>,
    max_lag: Option<usize>,
    method: &str,
) -> PyResult<usize> {
    let y = signal.as_slice()?;
    let n = y.len();
    let ml = max_lag.unwrap_or(n / 4).min(n / 2);

    if n < 4 {
        return Ok(1);
    }

    match method {
        "autocorr" => {
            // First zero crossing of autocorrelation
            // Match Python: acf = np.mean(x[:-lag]*x[lag:]) / np.var(x)
            let mean_val: f64 = y.iter().sum::<f64>() / n as f64;
            let var: f64 = y.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / n as f64;

            if var < 1e-15 {
                return Ok(1);
            }

            for lag in 1..ml {
                let n_pairs = (n - lag) as f64;
                let acf: f64 = y[..n - lag]
                    .iter()
                    .zip(y[lag..].iter())
                    .map(|(a, b)| (a - mean_val) * (b - mean_val))
                    .sum::<f64>()
                    / n_pairs
                    / var;
                if acf <= 0.0 {
                    return Ok(lag);
                }
            }
            Ok(ml)
        }
        "autocorr_e" => {
            // First 1/e decay of autocorrelation
            // Match Python: acf = np.mean(x[:-lag]*x[lag:]) / np.var(x)
            let mean_val: f64 = y.iter().sum::<f64>() / n as f64;
            let var: f64 = y.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / n as f64;

            if var < 1e-15 {
                return Ok(1);
            }

            let threshold = 1.0 / std::f64::consts::E;

            for lag in 1..ml {
                let n_pairs = (n - lag) as f64;
                let acf: f64 = y[..n - lag]
                    .iter()
                    .zip(y[lag..].iter())
                    .map(|(a, b)| (a - mean_val) * (b - mean_val))
                    .sum::<f64>()
                    / n_pairs
                    / var;
                if acf <= threshold {
                    return Ok(lag);
                }
            }
            Ok(ml)
        }
        _ => {
            // Mutual information first minimum (matches Python exactly)
            let n_bins = 16usize;
            let min_val = y.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = max_val - min_val;

            if range < 1e-15 {
                return Ok(1);
            }

            // Compute MI for all lags
            let mut mi_values: Vec<f64> = Vec::with_capacity(ml);
            for lag in 1..ml {
                let mi = lagged_mi(y, lag, n_bins, min_val, range);
                mi_values.push(mi);

                // Check for local minimum using 3-point window (matches Python)
                let len = mi_values.len();
                if len >= 3 {
                    if mi_values[len - 2] < mi_values[len - 1]
                        && mi_values[len - 2] < mi_values[len - 3]
                    {
                        return Ok(lag - 1);
                    }
                }
            }

            // Fallback: first significant drop (MI < 0.5 * initial_MI)
            if !mi_values.is_empty() {
                let initial_mi = mi_values[0];
                for (i, &mi) in mi_values.iter().enumerate() {
                    if mi < 0.5 * initial_mi {
                        return Ok(i + 1);
                    }
                }
            }

            Ok((ml / 4).max(1))
        }
    }
}

/// Estimate optimal embedding dimension via FNN or Cao's method.
/// Matches: manifold.primitives.embedding.delay.optimal_dimension
#[pyfunction]
#[pyo3(signature = (signal, delay=None, max_dim=10, method="fnn", threshold=0.01))]
pub fn optimal_dimension(
    signal: PyReadonlyArray1<f64>,
    delay: Option<usize>,
    max_dim: usize,
    method: &str,
    threshold: f64,
) -> PyResult<usize> {
    let y = signal.as_slice()?;
    let n = y.len();
    let tau = delay.unwrap_or(1);

    if n < (max_dim + 1) * tau + 2 {
        return Ok(2);
    }

    match method {
        "cao" => {
            // Cao's method: E1(d) ratio
            let mut prev_e = 0.0;

            for dim in 1..=max_dim {
                let n_points = n - dim * tau;
                if n_points < 10 {
                    return Ok(dim.max(2));
                }

                // Embed in dim-1 space for NN search (matches Python)
                let prev_dim = if dim > 1 { dim - 1 } else { 1 };
                let n_prev = n - (prev_dim) * tau;
                let n_pts = n_points.min(n_prev);

                let embed_prev: Vec<Vec<f64>> = (0..n_pts)
                    .map(|i| (0..prev_dim).map(|d| y[i + d * tau]).collect())
                    .collect();

                // For each point, find nearest neighbor in prev_dim embedding
                let n_check = n_pts.min(500);
                let mut e_sum = 0.0;
                let mut count = 0;

                for i in 0..n_check {
                    let mut best_dist = f64::INFINITY;
                    let mut best_j = 0;
                    for j in 0..n_pts {
                        if i == j {
                            continue;
                        }
                        let dist: f64 = embed_prev[i]
                            .iter()
                            .zip(embed_prev[j].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        if dist < best_dist {
                            best_dist = dist;
                            best_j = j;
                        }
                    }

                    if best_dist > 1e-10 {
                        // Distance in higher dimension (dim)
                        let embed_i: Vec<f64> = (0..dim).map(|d| y[i + d * tau]).collect();
                        let embed_j: Vec<f64> = (0..dim).map(|d| y[best_j + d * tau]).collect();
                        let r_d1: f64 = embed_i
                            .iter()
                            .zip(embed_j.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        e_sum += r_d1 / best_dist;
                        count += 1;
                    }
                }

                let e = if count > 0 { e_sum / count as f64 } else { 1.0 };

                if dim > 1 && prev_e > 1e-15 {
                    let e1 = e / prev_e;
                    if (e1 - 1.0).abs() < threshold {
                        return Ok(dim);
                    }
                }
                prev_e = e;
            }
            Ok(max_dim)
        }
        _ => {
            // False Nearest Neighbors (matches Python: threshold ratio = 10)
            for dim in 1..max_dim {
                let n_points = n - (dim + 1) * tau;
                if n_points < 10 {
                    return Ok(dim.max(2));
                }

                let embed: Vec<Vec<f64>> = (0..n_points)
                    .map(|i| (0..dim).map(|d| y[i + d * tau]).collect())
                    .collect();

                let mut fnn_count = 0;
                let mut total = 0;
                let n_check = n_points.min(1000);

                for i in 0..n_check {
                    let mut best_dist = f64::INFINITY;
                    let mut best_j = 0;
                    for j in 0..n_points {
                        if i == j {
                            continue;
                        }
                        let dist: f64 = embed[i]
                            .iter()
                            .zip(embed[j].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        if dist < best_dist {
                            best_dist = dist;
                            best_j = j;
                        }
                    }

                    if best_dist > 1e-10 {
                        // Distance in higher dimension
                        let r_d1 = (y[i + dim * tau] - y[best_j + dim * tau]).abs();
                        if r_d1 / best_dist > 10.0 {
                            fnn_count += 1;
                        }
                        total += 1;
                    }
                }

                let fnn_ratio = if total > 0 {
                    fnn_count as f64 / total as f64
                } else {
                    0.0
                };

                if fnn_ratio < threshold {
                    return Ok(dim + 1);
                }
            }
            Ok(max_dim)
        }
    }
}
