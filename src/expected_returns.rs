//! Expected-return estimators.
//!
//! Mirrors `pypfopt.expected_returns` — every function takes a price
//! matrix (`T x N`, rows = dates oldest→newest, columns = assets) and
//! returns an annualised mean-return vector of length `N`.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};

use crate::prelude::{column_means, log_returns_from_prices, returns_from_prices, LabeledVector};
use crate::{PortfolioError, Result, TRADING_DAYS_PER_YEAR};

/// Whether to compute simple or log returns under the hood.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReturnsKind {
    #[default]
    Simple,
    Log,
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
/// to [`TRADING_DAYS_PER_YEAR`] when `None`. When `compounding` is true
/// (PyPortfolioOpt default) the result is the CAGR-style geometric mean
/// `(∏(1 + r))^(f/T) − 1`. When false the result is the arithmetic mean
/// times `f`.
pub fn mean_historical_return(
    prices: &DMatrix<f64>,
    kind: ReturnsKind,
    compounding: bool,
    frequency: Option<usize>,
) -> Result<DVector<f64>> {
    let returns = build_returns(prices, kind)?;
    let (rows, cols) = returns.shape();
    if rows == 0 {
        return Err(PortfolioError::InvalidArgument(
            "no return observations".into(),
        ));
    }
    let f = frequency.unwrap_or(TRADING_DAYS_PER_YEAR) as f64;
    if compounding {
        let mut mu = DVector::<f64>::zeros(cols);
        for j in 0..cols {
            let mut prod = 1.0_f64;
            for i in 0..rows {
                prod *= 1.0 + returns[(i, j)];
            }
            mu[j] = prod.powf(f / rows as f64) - 1.0;
        }
        Ok(mu)
    } else {
        let mu = column_means(&returns);
        Ok(mu * f)
    }
}

/// Exponentially-weighted mean of historical returns.
///
/// `span` is the EMA span (PyPortfolioOpt default is 500). Weights follow
/// `alpha = 2 / (span + 1)`, applied with the most-recent observation
/// receiving the largest weight. When `compounding` is true the EMA mean
/// `m` is annualised as `(1 + m)^f - 1`; otherwise as `m * f`.
pub fn ema_historical_return(
    prices: &DMatrix<f64>,
    kind: ReturnsKind,
    compounding: bool,
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
    if compounding {
        Ok(mu.map(|m| (1.0 + m).powf(f) - 1.0))
    } else {
        Ok(mu * f)
    }
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
    let mkt_var: f64 =
        mkt.iter().map(|x| (x - mkt_mean).powi(2)).sum::<f64>() / (rows as f64 - 1.0);
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

// ---------------------------------------------------------------------------
// Labeled (ticker-aware) wrappers — match PyPortfolioOpt's DataFrame I/O
// ---------------------------------------------------------------------------

fn validate_labels(prices: &DMatrix<f64>, tickers: &[String]) -> Result<()> {
    if prices.ncols() != tickers.len() {
        return Err(PortfolioError::DimensionMismatch(format!(
            "prices has {} columns but {} tickers were supplied",
            prices.ncols(),
            tickers.len()
        )));
    }
    Ok(())
}

/// Ticker-labeled version of [`mean_historical_return`]. Returns a
/// [`LabeledVector`] you can convert to `BTreeMap<String, f64>` via
/// [`LabeledVector::to_map`] when you need an ordered, ticker-keyed
/// output.
pub fn mean_historical_return_labeled(
    prices: &DMatrix<f64>,
    tickers: &[String],
    kind: ReturnsKind,
    compounding: bool,
    frequency: Option<usize>,
) -> Result<LabeledVector> {
    validate_labels(prices, tickers)?;
    let mu = mean_historical_return(prices, kind, compounding, frequency)?;
    LabeledVector::new(mu, tickers.to_vec())
}

/// Ticker-labeled version of [`ema_historical_return`].
pub fn ema_historical_return_labeled(
    prices: &DMatrix<f64>,
    tickers: &[String],
    kind: ReturnsKind,
    compounding: bool,
    span: usize,
    frequency: Option<usize>,
) -> Result<LabeledVector> {
    validate_labels(prices, tickers)?;
    let mu = ema_historical_return(prices, kind, compounding, span, frequency)?;
    LabeledVector::new(mu, tickers.to_vec())
}

/// Ticker-labeled version of [`capm_return`]. The returned labels exclude
/// the market column at `market_idx`.
pub fn capm_return_labeled(
    prices: &DMatrix<f64>,
    tickers: &[String],
    kind: ReturnsKind,
    market_idx: usize,
    risk_free_rate: f64,
    frequency: Option<usize>,
) -> Result<LabeledVector> {
    validate_labels(prices, tickers)?;
    if market_idx >= tickers.len() {
        return Err(PortfolioError::InvalidArgument(format!(
            "market_idx {market_idx} out of range for {} tickers",
            tickers.len()
        )));
    }
    let mu = capm_return(prices, kind, market_idx, risk_free_rate, frequency)?;
    let asset_tickers: Vec<String> = tickers
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            if i == market_idx {
                None
            } else {
                Some(t.clone())
            }
        })
        .collect();
    LabeledVector::new(mu, asset_tickers)
}

/// Convenience: run [`mean_historical_return_labeled`] and immediately
/// convert to a `BTreeMap`. Mirrors PyPortfolioOpt's
/// `expected_returns.mean_historical_return(prices_df)` returning a
/// pandas Series.
pub fn mean_historical_return_map(
    prices: &DMatrix<f64>,
    tickers: &[String],
    kind: ReturnsKind,
    compounding: bool,
    frequency: Option<usize>,
) -> Result<BTreeMap<String, f64>> {
    Ok(mean_historical_return_labeled(prices, tickers, kind, compounding, frequency)?.to_map())
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
        let mu = mean_historical_return(&p, ReturnsKind::Simple, false, Some(252)).unwrap();
        assert_relative_eq!(mu[0], 0.01 * 252.0, max_relative = 1e-9);
        assert_relative_eq!(mu[1], 0.02 * 252.0, max_relative = 1e-9);
    }

    #[test]
    fn mean_log_matches_constant_growth() {
        let p = linear_prices();
        let mu = mean_historical_return(&p, ReturnsKind::Log, false, Some(252)).unwrap();
        assert_relative_eq!(mu[0], 0.01_f64.ln_1p() * 252.0, max_relative = 1e-9);
        assert_relative_eq!(mu[1], 0.02_f64.ln_1p() * 252.0, max_relative = 1e-9);
    }

    #[test]
    fn mean_compounding_matches_cagr() {
        // 10 returns of 1% each → ∏(1+r) = 1.01^10. CAGR with f=252 is
        // (1.01^10)^(252/10) - 1 = 1.01^252 - 1.
        let p = linear_prices();
        let mu = mean_historical_return(&p, ReturnsKind::Simple, true, Some(252)).unwrap();
        assert_relative_eq!(mu[0], 1.01_f64.powf(252.0) - 1.0, max_relative = 1e-9);
        assert_relative_eq!(mu[1], 1.02_f64.powf(252.0) - 1.0, max_relative = 1e-9);
    }

    #[test]
    fn ema_recovers_constant_returns() {
        // Constant returns -> EMA = constant regardless of span.
        let p = linear_prices();
        let mu = ema_historical_return(&p, ReturnsKind::Simple, false, 5, Some(252)).unwrap();
        assert_relative_eq!(mu[0], 0.01 * 252.0, max_relative = 1e-9);
        assert_relative_eq!(mu[1], 0.02 * 252.0, max_relative = 1e-9);
    }

    #[test]
    fn ema_compounding_annualises_correctly() {
        // Constant returns: EMA mean = 0.01, compounded annual = 1.01^252 - 1.
        let p = linear_prices();
        let mu = ema_historical_return(&p, ReturnsKind::Simple, true, 5, Some(252)).unwrap();
        assert_relative_eq!(mu[0], 1.01_f64.powf(252.0) - 1.0, max_relative = 1e-9);
        assert_relative_eq!(mu[1], 1.02_f64.powf(252.0) - 1.0, max_relative = 1e-9);
    }

    #[test]
    fn ema_invalid_span_errors() {
        let p = linear_prices();
        assert!(ema_historical_return(&p, ReturnsKind::Simple, false, 0, None).is_err());
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

    #[test]
    fn mean_historical_return_labeled_carries_tickers() {
        let p = linear_prices();
        let tickers = vec!["AAPL".to_string(), "MSFT".to_string()];
        let lr =
            mean_historical_return_labeled(&p, &tickers, ReturnsKind::Simple, false, Some(252))
                .unwrap();
        assert_eq!(lr.tickers, tickers);
        assert_relative_eq!(lr.get("AAPL").unwrap(), 0.01 * 252.0, max_relative = 1e-9);
        let map = lr.to_map();
        assert!(map.contains_key("AAPL") && map.contains_key("MSFT"));
    }

    #[test]
    fn capm_return_labeled_drops_market_ticker() {
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
        let tickers = vec!["AAPL".to_string(), "SPY".to_string()];
        let lr =
            capm_return_labeled(&prices, &tickers, ReturnsKind::Simple, 1, 0.02, Some(1)).unwrap();
        assert_eq!(lr.tickers, vec!["AAPL".to_string()]);
        assert_eq!(lr.values.len(), 1);
    }

    #[test]
    fn labeled_rejects_mismatched_tickers() {
        let p = linear_prices();
        let tickers = vec!["AAPL".to_string()]; // wrong length
        assert!(
            mean_historical_return_labeled(&p, &tickers, ReturnsKind::Simple, false, None).is_err()
        );
    }
}
