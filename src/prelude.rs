//! Internal helpers shared across modules. Re-exported from the crate root
//! for downstream convenience.

use nalgebra::{DMatrix, DVector};

use crate::{PortfolioError, Result};

/// Compute simple period-over-period returns from a `T x N` price matrix.
///
/// Rows are time, columns are assets. The result has shape `(T-1) x N`.
/// Any non-finite entry (NaN or +/- inf) is preserved — callers handle
/// missing data upstream as PyPortfolioOpt does.
pub fn returns_from_prices(prices: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let (rows, cols) = prices.shape();
    if rows < 2 {
        return Err(PortfolioError::InvalidArgument(
            "need at least two rows of prices to compute returns".into(),
        ));
    }
    let mut out = DMatrix::<f64>::zeros(rows - 1, cols);
    for j in 0..cols {
        for i in 1..rows {
            let prev = prices[(i - 1, j)];
            let curr = prices[(i, j)];
            out[(i - 1, j)] = curr / prev - 1.0;
        }
    }
    Ok(out)
}

/// Log returns: `ln(p_t / p_{t-1})`.
pub fn log_returns_from_prices(prices: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let (rows, cols) = prices.shape();
    if rows < 2 {
        return Err(PortfolioError::InvalidArgument(
            "need at least two rows of prices to compute log returns".into(),
        ));
    }
    let mut out = DMatrix::<f64>::zeros(rows - 1, cols);
    for j in 0..cols {
        for i in 1..rows {
            out[(i - 1, j)] = (prices[(i, j)] / prices[(i - 1, j)]).ln();
        }
    }
    Ok(out)
}

/// Column-wise mean of an `T x N` matrix → length-N vector.
pub fn column_means(m: &DMatrix<f64>) -> DVector<f64> {
    let (rows, cols) = m.shape();
    let mut out = DVector::<f64>::zeros(cols);
    if rows == 0 {
        return out;
    }
    for j in 0..cols {
        let mut acc = 0.0;
        for i in 0..rows {
            acc += m[(i, j)];
        }
        out[j] = acc / rows as f64;
    }
    out
}

/// Sample covariance with Bessel's correction (divisor `T-1`).
///
/// Operates on a `T x N` returns matrix and returns an `N x N` covariance.
pub fn sample_covariance(returns: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let (rows, cols) = returns.shape();
    if rows < 2 {
        return Err(PortfolioError::InvalidArgument(
            "need at least two observations for sample covariance".into(),
        ));
    }
    let means = column_means(returns);
    let mut centered = returns.clone();
    for j in 0..cols {
        let mu = means[j];
        for i in 0..rows {
            centered[(i, j)] -= mu;
        }
    }
    let cov = centered.transpose() * &centered / (rows as f64 - 1.0);
    Ok(cov)
}

/// Validate that a square covariance matrix has the expected dimension.
pub(crate) fn assert_square(m: &DMatrix<f64>, label: &str) -> Result<usize> {
    let (r, c) = m.shape();
    if r != c {
        return Err(PortfolioError::DimensionMismatch(format!(
            "{label}: expected square matrix, got {r}x{c}"
        )));
    }
    Ok(r)
}

/// Round entries of `weights` whose absolute value is below `cutoff` to
/// zero, renormalise the remainder so they sum to the original total,
/// and optionally round to `rounding` decimal places. Mirrors
/// PyPortfolioOpt's `base_optimizer.BaseOptimizer.clean_weights`.
pub fn clean_weights(
    weights: &DVector<f64>,
    cutoff: f64,
    rounding: Option<u32>,
) -> DVector<f64> {
    let mut cleaned = weights.clone();
    for v in cleaned.iter_mut() {
        if v.abs() < cutoff {
            *v = 0.0;
        }
    }
    let total: f64 = cleaned.iter().sum();
    if total.abs() > 1e-12 {
        for v in cleaned.iter_mut() {
            *v /= total;
        }
    }
    if let Some(places) = rounding {
        let factor = 10f64.powi(places as i32);
        for v in cleaned.iter_mut() {
            *v = (*v * factor).round() / factor;
        }
    }
    cleaned
}

/// Symmetrise a matrix in-place (`(A + A^T) / 2`). Useful after numerical
/// operations that produce minute asymmetry.
pub fn symmetrise(m: &mut DMatrix<f64>) {
    let n = m.nrows();
    debug_assert_eq!(m.nrows(), m.ncols());
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (m[(i, j)] + m[(j, i)]);
            m[(i, j)] = avg;
            m[(j, i)] = avg;
        }
    }
}
