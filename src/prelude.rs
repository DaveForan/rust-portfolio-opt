//! Internal helpers shared across modules. Re-exported from the crate root
//! for downstream convenience.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};

use crate::{PortfolioError, Result};

/// A length-N vector of floats paired with N ticker labels. Mirrors a
/// pandas `Series` indexed by ticker name. Returned by every estimator
/// that has a `_labeled` companion (e.g.
/// [`crate::expected_returns::mean_historical_return_labeled`]) so callers
/// can keep ticker order alongside the numerical values.
#[derive(Debug, Clone)]
pub struct LabeledVector {
    pub values: DVector<f64>,
    pub tickers: Vec<String>,
}

impl LabeledVector {
    pub fn new(values: DVector<f64>, tickers: Vec<String>) -> Result<Self> {
        if values.len() != tickers.len() {
            return Err(PortfolioError::DimensionMismatch(format!(
                "LabeledVector: values len {} ≠ tickers len {}",
                values.len(),
                tickers.len()
            )));
        }
        Ok(Self { values, tickers })
    }

    /// Look up a single value by ticker name.
    pub fn get(&self, ticker: &str) -> Option<f64> {
        self.tickers
            .iter()
            .position(|t| t == ticker)
            .map(|i| self.values[i])
    }

    /// Convert to a `BTreeMap<String, f64>` ordered alphabetically by
    /// ticker, matching PyPortfolioOpt's `OrderedDict` style output.
    pub fn to_map(&self) -> BTreeMap<String, f64> {
        self.tickers
            .iter()
            .zip(self.values.iter())
            .map(|(t, v)| (t.clone(), *v))
            .collect()
    }
}

/// A square N×N matrix paired with N ticker labels (used for both row
/// and column indexing — the same labelling pandas applies to a
/// covariance/correlation `DataFrame`).
#[derive(Debug, Clone)]
pub struct LabeledMatrix {
    pub values: DMatrix<f64>,
    pub tickers: Vec<String>,
}

impl LabeledMatrix {
    pub fn new(values: DMatrix<f64>, tickers: Vec<String>) -> Result<Self> {
        let (r, c) = values.shape();
        if r != c {
            return Err(PortfolioError::DimensionMismatch(format!(
                "LabeledMatrix: expected square matrix, got {r}x{c}"
            )));
        }
        if r != tickers.len() {
            return Err(PortfolioError::DimensionMismatch(format!(
                "LabeledMatrix: matrix is {r}x{r} but tickers len is {}",
                tickers.len()
            )));
        }
        Ok(Self { values, tickers })
    }

    /// Look up `[row_ticker, col_ticker]`.
    pub fn get(&self, row_ticker: &str, col_ticker: &str) -> Option<f64> {
        let r = self.tickers.iter().position(|t| t == row_ticker)?;
        let c = self.tickers.iter().position(|t| t == col_ticker)?;
        Some(self.values[(r, c)])
    }
}

/// Convert a `(values, tickers)` pair into a ticker-keyed `BTreeMap`,
/// matching PyPortfolioOpt's habit of returning weights as an
/// `OrderedDict[str, float]`. Errors if the lengths disagree.
pub fn to_weight_map(values: &DVector<f64>, tickers: &[String]) -> Result<BTreeMap<String, f64>> {
    if values.len() != tickers.len() {
        return Err(PortfolioError::DimensionMismatch(format!(
            "to_weight_map: values len {} ≠ tickers len {}",
            values.len(),
            tickers.len()
        )));
    }
    Ok(tickers
        .iter()
        .zip(values.iter())
        .map(|(t, v)| (t.clone(), *v))
        .collect())
}

/// Validate that two ticker label vectors agree in length and order.
pub(crate) fn assert_tickers_match(a: &[String], b: &[String], label: &str) -> Result<()> {
    if a.len() != b.len() {
        return Err(PortfolioError::DimensionMismatch(format!(
            "{label}: ticker counts disagree ({} vs {})",
            a.len(),
            b.len()
        )));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            return Err(PortfolioError::InvalidArgument(format!(
                "{label}: ticker mismatch at position {i}: '{x}' vs '{y}'"
            )));
        }
    }
    Ok(())
}

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
pub fn clean_weights(weights: &DVector<f64>, cutoff: f64, rounding: Option<u32>) -> DVector<f64> {
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
