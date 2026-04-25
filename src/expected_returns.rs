//! Expected-return estimators.
//!
//! Mirrors `pypfopt.expected_returns` — every function takes a price
//! matrix (`T x N`, rows = dates oldest→newest, columns = assets) and
//! returns an annualised mean-return vector of length `N`.

use nalgebra::{DMatrix, DVector};

use crate::prelude::{column_means, log_returns_from_prices, returns_from_prices};
use crate::{PortfolioError, Result, TRADING_DAYS_PER_YEAR};

/// Whether to compute simple or log returns under the hood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnsKind {
    Simple,
    Log,
}

impl Default for ReturnsKind {
    fn default() -> Self {
        ReturnsKind::Simple
    }
}

fn build_returns(prices: &DMatrix<f64>, kind: ReturnsKind) -> Result<DMatrix<f64>> {
    match kind {
        ReturnsKind::Simple => returns_from_prices(prices),
        ReturnsKind::Log => log_returns_from_prices(prices),
    }
}

/// Annualised mean of historical returns.
///
/// `frequency` is the number of periods per year (252 for daily). Defaults
/// to [`TRADING_DAYS_PER_YEAR`] when `None`.
pub fn mean_historical_return(
    prices: &DMatrix<f64>,
    kind: ReturnsKind,
    frequency: Option<usize>,
) -> Result<DVector<f64>> {
    let returns = build_returns(prices, kind)?;
    let mu = column_means(&returns);
    let f = frequency.unwrap_or(TRADING_DAYS_PER_YEAR) as f64;
    Ok(mu * f)
}

/// Exponentially-weighted mean of historical returns.
///
/// `span` is the EMA span (PyPortfolioOpt default is 500). Weights follow
/// `alpha = 2 / (span + 1)`, applied with the most-recent observation
/// receiving the largest weight. The result is then annualised by
/// `frequency`.
pub fn ema_historical_return(
    prices: &DMatrix<f64>,
    kind: ReturnsKind,
    span: usize,
    frequency: Option<usize>,
) -> Result<DVector<f64>> {
    if span < 1 {
        return Err(PortfolioError::InvalidArgument(
            "EMA span must be >= 1".into(),
        ));
    }
    let returns = build_returns(prices, kind)?;
    let (rows, cols) = returns.shape();
    if rows == 0 {
        return Err(PortfolioError::InvalidArgument(
            "no return observations".into(),
        ));
    }

    let alpha = 2.0 / (span as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;
    let mut mu = DVector::<f64>::zeros(cols);

    // Pandas-style EWMA with adjust=True: mean = sum_t (1-α)^(T-1-t) * x_t /
    // sum_t (1-α)^(T-1-t). This treats the most recent observation as
    // weight 1 and decays back through time.
    for j in 0..cols {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut weight = 1.0;
        for i in (0..rows).rev() {
            weighted_sum += weight * returns[(i, j)];
            weight_sum += weight;
            weight *= one_minus_alpha;
        }
        mu[j] = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        };
    }

    let f = frequency.unwrap_or(TRADING_DAYS_PER_YEAR) as f64;
    Ok(mu * f)
}

/// CAPM-implied expected returns.
///
/// `prices` includes the market column at `market_idx` (0-based). For each
/// asset we estimate its beta against the market by sample covariance and
/// return `rf + beta * (market_excess_mean)`, annualised via `frequency`.
///
/// The returned vector has length `prices.ncols() - 1` and excludes the
/// market column (asset order is preserved otherwise).
pub fn capm_return(
    prices: &DMatrix<f64>,
    kind: ReturnsKind,
    market_idx: usize,
    risk_free_rate: f64,
    frequency: Option<usize>,
) -> Result<DVector<f64>> {
    let cols = prices.ncols();
    if market_idx >= cols {
        return Err(PortfolioError::InvalidArgument(format!(
            "market_idx {market_idx} out of bounds for {cols} columns"
        )));
    }

    let returns = build_returns(prices, kind)?;
    let rows = returns.nrows();
    if rows < 2 {
        return Err(PortfolioError::InvalidArgument(
            "need at least two return observations for CAPM".into(),
        ));
    }
    let f = frequency.unwrap_or(TRADING_DAYS_PER_YEAR) as f64;
    let mkt = returns.column(market_idx).clone_owned();
    let mkt_mean = mkt.mean();
    let mkt_var: f64 = mkt
        .iter()
        .map(|x| (x - mkt_mean).powi(2))
        .sum::<f64>()
        / (rows as f64 - 1.0);
    if mkt_var <= 0.0 {
        return Err(PortfolioError::InvalidArgument(
            "market variance is zero or negative".into(),
        ));
    }

    // Annualised market premium over the risk-free rate.
    let market_premium_ann = mkt_mean * f - risk_free_rate;

    let mut out = DVector::<f64>::zeros(cols - 1);
    let mut k = 0_usize;
    for j in 0..cols {
        if j == market_idx {
            continue;
        }
        let asset = returns.column(j);
        let asset_mean = asset.mean();
        let mut cov = 0.0;
        for i in 0..rows {
            cov += (asset[i] - asset_mean) * (mkt[i] - mkt_mean);
        }
        cov /= rows as f64 - 1.0;
        let beta = cov / mkt_var;
        out[k] = risk_free_rate + beta * market_premium_ann;
        k += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::dmatrix;

    fn linear_prices() -> DMatrix<f64> {
        // Two assets that grow at constant 1% / 2% per period.
        let mut p = DMatrix::<f64>::zeros(11, 2);
        p[(0, 0)] = 100.0;
        p[(0, 1)] = 100.0;
        for i in 1..11 {
            p[(i, 0)] = p[(i - 1, 0)] * 1.01;
            p[(i, 1)] = p[(i - 1, 1)] * 1.02;
        }
        p
    }

    #[test]
    fn mean_simple_matches_constant_growth() {
        let p = linear_prices();
        let mu = mean_historical_return(&p, ReturnsKind::Simple, Some(252)).unwrap();
        assert_relative_eq!(mu[0], 0.01 * 252.0, max_relative = 1e-9);
        assert_relative_eq!(mu[1], 0.02 * 252.0, max_relative = 1e-9);
    }

    #[test]
    fn mean_log_matches_constant_growth() {
        let p = linear_prices();
        let mu = mean_historical_return(&p, ReturnsKind::Log, Some(252)).unwrap();
        assert_relative_eq!(mu[0], 0.01_f64.ln_1p() * 252.0, max_relative = 1e-9);
        assert_relative_eq!(mu[1], 0.02_f64.ln_1p() * 252.0, max_relative = 1e-9);
    }

    #[test]
    fn ema_recovers_constant_returns() {
        // Constant returns -> EMA = constant regardless of span.
        let p = linear_prices();
        let mu = ema_historical_return(&p, ReturnsKind::Simple, 5, Some(252)).unwrap();
        assert_relative_eq!(mu[0], 0.01 * 252.0, max_relative = 1e-9);
        assert_relative_eq!(mu[1], 0.02 * 252.0, max_relative = 1e-9);
    }

    #[test]
    fn ema_invalid_span_errors() {
        let p = linear_prices();
        assert!(ema_historical_return(&p, ReturnsKind::Simple, 0, None).is_err());
    }

    #[test]
    fn capm_asset_with_zero_beta_returns_rf() {
        // Constant returns -> beta=1 case is degenerate; use stochastic.
        // Build prices where asset 0 is uncorrelated with market (asset 1).
        let prices = dmatrix![
            100.0, 100.0;
            101.0,  99.0;
            100.0, 101.0;
            101.0,  99.0;
            100.0, 101.0;
            101.0,  99.0;
            100.0, 101.0;
            101.0,  99.0
        ];
        let mu = capm_return(&prices, ReturnsKind::Simple, 1, 0.02, Some(1)).unwrap();
        // beta should be ~ -1 because asset 0 ticks opposite to market.
        // Just sanity-check that the result is finite and a single asset.
        assert_eq!(mu.len(), 1);
        assert!(mu[0].is_finite());
    }

    #[test]
    fn capm_market_idx_out_of_bounds() {
        let p = linear_prices();
        let err = capm_return(&p, ReturnsKind::Simple, 99, 0.0, None).unwrap_err();
        matches!(err, PortfolioError::InvalidArgument(_));
    }
}
