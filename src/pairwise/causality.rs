use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Solve Ax = b via Gaussian elimination with partial pivoting.
/// A is n×n, b is n×1. Returns solution x, or None if singular.
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    // Augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None; // Singular
        }
        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

/// Compute RSS from OLS regression: X * beta ≈ y.
/// X is (n_obs rows, n_cols columns stored as row-major Vec<Vec<f64>>).
fn ols_rss(x_rows: &[Vec<f64>], y: &[f64]) -> f64 {
    let n_obs = x_rows.len();
    let n_cols = x_rows[0].len();

    // Compute X^T X
    let mut xtx = vec![vec![0.0; n_cols]; n_cols];
    for i in 0..n_cols {
        for j in i..n_cols {
            let mut sum = 0.0;
            for t in 0..n_obs {
                sum += x_rows[t][i] * x_rows[t][j];
            }
            xtx[i][j] = sum;
            xtx[j][i] = sum;
        }
    }

    // Compute X^T y
    let mut xty = vec![0.0; n_cols];
    for i in 0..n_cols {
        let mut sum = 0.0;
        for t in 0..n_obs {
            sum += x_rows[t][i] * y[t];
        }
        xty[i] = sum;
    }

    // Solve (X^T X) beta = X^T y
    let beta = match solve_linear(&xtx, &xty) {
        Some(b) => b,
        None => return f64::INFINITY,
    };

    // RSS = sum((y - X*beta)^2)
    let mut rss = 0.0;
    for t in 0..n_obs {
        let mut pred = 0.0;
        for j in 0..n_cols {
            pred += x_rows[t][j] * beta[j];
        }
        rss += (y[t] - pred).powi(2);
    }

    rss
}


/// Granger causality test using proper OLS regression.
/// Returns (f_statistic, p_value, optimal_lag).
/// Matches: manifold.primitives.pairwise.causality.granger_causality
#[pyfunction]
#[pyo3(signature = (source, target, max_lag=5))]
pub fn granger_causality(
    source: PyReadonlyArray1<f64>,
    target: PyReadonlyArray1<f64>,
    max_lag: usize,
) -> PyResult<(f64, f64, usize)> {
    let x = source.as_slice()?;
    let y = target.as_slice()?;
    let n = x.len().min(y.len());

    if n < max_lag + 10 {
        return Ok((0.0, 1.0, 1));
    }

    // Step 1: Find optimal lag via AIC (matches Python)
    // All lags use the same observation window: y[max_lag..]
    let n_obs = n - max_lag;
    let y_vec: Vec<f64> = y[max_lag..n].to_vec();

    let mut best_aic = f64::INFINITY;
    let mut optimal_lag = 1usize;

    for lag in 1..=max_lag {
        // Restricted model: intercept + y_{t-1}..y_{t-lag}
        let n_cols = 1 + lag;
        if n_obs < n_cols + 2 {
            continue;
        }

        let mut x_rows: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
        for t in 0..n_obs {
            let mut row = vec![1.0]; // intercept
            for l in 1..=lag {
                row.push(y[max_lag + t - l]);
            }
            x_rows.push(row);
        }

        let rss = ols_rss(&x_rows, &y_vec);
        if rss <= 0.0 || rss.is_infinite() {
            continue;
        }

        let aic = (n_obs as f64) * (rss / n_obs as f64 + 1e-10).ln() + 2.0 * (lag + 1) as f64;
        if aic < best_aic {
            best_aic = aic;
            optimal_lag = lag;
        }
    }

    let lag = optimal_lag;

    // Step 2: Restricted model at optimal lag
    let n_cols_r = 1 + lag;
    let mut x_r: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
    for t in 0..n_obs {
        let mut row = vec![1.0];
        for l in 1..=lag {
            row.push(y[max_lag + t - l]);
        }
        x_r.push(row);
    }
    let rss_r = ols_rss(&x_r, &y_vec);

    // Step 3: Unrestricted model: restricted + x_{t-1}..x_{t-lag}
    let mut x_u: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
    for t in 0..n_obs {
        let mut row = x_r[t].clone();
        for l in 1..=lag {
            row.push(x[max_lag + t - l]);
        }
        x_u.push(row);
    }
    let rss_u = ols_rss(&x_u, &y_vec);

    let n_cols_u = x_u[0].len();
    let df1 = lag as f64;
    let df2 = (n_obs as f64) - (n_cols_u as f64);

    if df2 <= 0.0 || rss_u <= 0.0 {
        return Ok((0.0, 1.0, lag));
    }

    let f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2);

    // p-value from F-distribution via statrs
    let p_value = if f_stat > 0.0 && df1 > 0.0 && df2 > 0.0 {
        match FisherSnedecor::new(df1, df2) {
            Ok(f_dist) => 1.0 - f_dist.cdf(f_stat),
            Err(_) => 1.0,
        }
    } else {
        1.0
    };

    Ok((f_stat, p_value, lag))
}


/// Convergent cross mapping.
/// Returns (rho_a_causes_b, rho_b_causes_a).
/// Matches: manifold.primitives.pairwise.causality.convergent_cross_mapping
#[pyfunction]
#[pyo3(signature = (sig_a, sig_b, embedding_dim=3, tau=1, library_size=None))]
pub fn convergent_cross_mapping(
    sig_a: PyReadonlyArray1<f64>,
    sig_b: PyReadonlyArray1<f64>,
    embedding_dim: usize,
    tau: usize,
    library_size: Option<usize>,
) -> PyResult<(f64, f64)> {
    let a = sig_a.as_slice()?;
    let b = sig_b.as_slice()?;
    let n = a.len().min(b.len());

    let n_points = n.saturating_sub((embedding_dim - 1) * tau);
    let lib_size = library_size.unwrap_or(n_points);

    if n_points < embedding_dim + 2 {
        return Ok((f64::NAN, 1.0));
    }

    // Embed signal a
    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..embedding_dim).map(|d| a[i + d * tau]).collect())
        .collect();

    // For each point, find embedding_dim+1 nearest neighbors, predict b
    let k = embedding_dim + 1;
    let mut predictions = Vec::with_capacity(n_points);
    let mut actuals = Vec::with_capacity(n_points);

    for i in 0..n_points.min(lib_size) {
        let mut dists: Vec<(usize, f64)> = (0..n_points)
            .filter(|&j| j != i)
            .map(|j| {
                let d: f64 = embedded[i]
                    .iter()
                    .zip(embedded[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (j, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let neighbors: Vec<(usize, f64)> = dists.into_iter().take(k).collect();

        if neighbors.is_empty() {
            continue;
        }

        // Distance-weighted prediction
        let min_dist = neighbors[0].1.max(1e-15);
        let weights: Vec<f64> = neighbors
            .iter()
            .map(|&(_, d)| (-d / min_dist).exp())
            .collect();
        let w_sum: f64 = weights.iter().sum();

        if w_sum > 0.0 {
            let pred: f64 = neighbors
                .iter()
                .zip(weights.iter())
                .map(|(&(j, _), &w)| w * b[j])
                .sum::<f64>()
                / w_sum;

            predictions.push(pred);
            actuals.push(b[i]);
        }
    }

    // Correlation between predicted and actual
    if predictions.len() < 3 {
        return Ok((f64::NAN, 1.0));
    }

    let n_pred = predictions.len();
    let mean_p: f64 = predictions.iter().sum::<f64>() / n_pred as f64;
    let mean_a: f64 = actuals.iter().sum::<f64>() / n_pred as f64;

    let mut cov = 0.0;
    let mut var_p = 0.0;
    let mut var_a = 0.0;
    for i in 0..n_pred {
        let dp = predictions[i] - mean_p;
        let da = actuals[i] - mean_a;
        cov += dp * da;
        var_p += dp * dp;
        var_a += da * da;
    }

    let denom = (var_p * var_a).sqrt();
    let corr = if denom > 1e-15 { cov / denom } else { 0.0 };

    // Approximate p-value
    let t_stat = corr * ((n_pred - 2) as f64).sqrt() / (1.0 - corr * corr).max(1e-15).sqrt();
    let p_value = t_to_p_approx(t_stat, (n_pred - 2) as f64);

    Ok((corr, p_value))
}


/// Approximate t-distribution p-value (two-tailed).
fn t_to_p_approx(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 1.0;
    }
    let z = t.abs();
    let p = (-0.5 * z * z).exp() * 0.4;
    (2.0 * p).min(1.0).max(0.001)
}
