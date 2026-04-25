//! Discrete allocation: convert continuous portfolio weights into integer
//! share counts given a total budget.
//!
//! Mirrors `pypfopt.discrete_allocation.DiscreteAllocation`. Two methods
//! are provided:
//!
//! - **Greedy** — floors every position to whole shares, then uses
//!   remaining cash to buy additional shares in decreasing order of
//!   fractional-allocation size. Fast, O(n log n).
//! - **Rounded** — simply rounds each allocation to the nearest whole
//!   share (may violate the budget slightly).

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};

use crate::{PortfolioError, Result};

/// Extract the most recent (last-row) price for each asset from a price
/// matrix, mirroring `pypfopt.discrete_allocation.get_latest_prices`.
/// The price matrix is `T x N` (rows = dates oldest→newest, columns =
/// assets); the result is an `N`-vector of the latest closes.
pub fn get_latest_prices(prices: &DMatrix<f64>) -> Result<DVector<f64>> {
    let rows = prices.nrows();
    if rows == 0 {
        return Err(PortfolioError::InvalidArgument(
            "price matrix has no rows".into(),
        ));
    }
    Ok(prices.row(rows - 1).transpose().into_owned())
}

/// Converts continuous portfolio weights into integer share counts.
///
/// # Example
/// ```
/// use rust_portfolio_opt::discrete_allocation::DiscreteAllocation;
/// use nalgebra::DVector;
///
/// let weights = DVector::from_vec(vec![0.6, 0.4]);
/// let prices  = DVector::from_vec(vec![100.0, 50.0]);
/// let da = DiscreteAllocation::new(weights, prices, 10_000.0).unwrap();
/// let (shares, leftover) = da.greedy_portfolio().unwrap();
/// assert!(leftover >= 0.0);
/// ```
pub struct DiscreteAllocation {
    /// Continuous weights (must sum to ≤ 1; negative weights represent
    /// short positions).
    pub weights: DVector<f64>,
    /// Latest market price per asset (same ordering).
    pub prices: DVector<f64>,
    /// Total cash budget.
    pub total_portfolio_value: f64,
    /// Fraction of the portfolio dedicated to short positions (e.g. 0.3 →
    /// 130/30). When `None`, derived from the weights as
    /// `sum(-w for w in weights if w < 0)`.
    pub short_ratio: Option<f64>,
}

impl DiscreteAllocation {
    pub fn new(
        weights: DVector<f64>,
        prices: DVector<f64>,
        total_portfolio_value: f64,
    ) -> Result<Self> {
        let n = weights.len();
        if prices.len() != n {
            return Err(PortfolioError::DimensionMismatch(format!(
                "weights has length {} but prices has length {}",
                n,
                prices.len()
            )));
        }
        if total_portfolio_value <= 0.0 {
            return Err(PortfolioError::InvalidArgument(
                "total_portfolio_value must be positive".into(),
            ));
        }
        for (i, &p) in prices.iter().enumerate() {
            if p <= 0.0 {
                return Err(PortfolioError::InvalidArgument(format!(
                    "price[{i}] = {p} is non-positive"
                )));
            }
        }
        Ok(Self {
            weights,
            prices,
            total_portfolio_value,
            short_ratio: None,
        })
    }

    /// Override the short ratio (e.g. 0.3 for a 130/30 portfolio).
    pub fn with_short_ratio(mut self, short_ratio: f64) -> Result<Self> {
        if short_ratio < 0.0 {
            return Err(PortfolioError::InvalidArgument(
                "short_ratio must be non-negative".into(),
            ));
        }
        self.short_ratio = Some(short_ratio);
        Ok(self)
    }

    fn effective_short_ratio(&self) -> f64 {
        if let Some(r) = self.short_ratio {
            return r;
        }
        self.weights.iter().filter(|w| **w < 0.0).map(|w| -w).sum()
    }

    /// Greedy allocation with the default options (no reinvestment of
    /// short proceeds, silent). Shortcut for
    /// `greedy_portfolio_with_options(false, false)`.
    pub fn greedy_portfolio(&self) -> Result<(HashMap<usize, i64>, f64)> {
        self.greedy_portfolio_with_options(false, false)
    }

    /// Greedy allocation with full options. When `reinvest` is true the
    /// proceeds from short positions are added to the long-side budget;
    /// when false (PyPortfolioOpt default) shorts and longs are budgeted
    /// independently. `verbose` is a no-op kept for API symmetry.
    ///
    /// Returns `(allocation, leftover)` where `allocation` maps asset
    /// index → number of shares (signed: positive = long, negative = short)
    /// and `leftover` is remaining unallocated cash.
    pub fn greedy_portfolio_with_options(
        &self,
        reinvest: bool,
        _verbose: bool,
    ) -> Result<(HashMap<usize, i64>, f64)> {
        let n = self.weights.len();
        let has_shorts = self.weights.iter().any(|w| *w < 0.0);
        let short_ratio = self.effective_short_ratio();

        // When there are short positions, follow PyPortfolioOpt's split:
        // long side gets `tpv` (or `tpv * (1 + short_ratio)` if reinvest
        // is set), short side gets `tpv * short_ratio` of "borrow".
        let (long_budget, short_budget) = if has_shorts && short_ratio > 0.0 {
            let short_val = self.total_portfolio_value * short_ratio;
            let long_val = if reinvest {
                self.total_portfolio_value + short_val
            } else {
                self.total_portfolio_value
            };
            (long_val, short_val)
        } else {
            (self.total_portfolio_value, 0.0)
        };

        // Long-side allocation.
        let long_indices: Vec<usize> = (0..n).filter(|&i| self.weights[i] > 0.0).collect();
        let long_total: f64 = long_indices.iter().map(|&i| self.weights[i]).sum();
        let mut shares = vec![0_i64; n];
        let mut leftover_long = long_budget;
        if long_total > 0.0 && long_budget > 0.0 {
            let (s, lo) = greedy_one_side(
                &long_indices,
                &self.weights,
                &self.prices,
                long_budget,
                long_total,
                false,
            );
            for (i, sh) in s {
                shares[i] = sh;
            }
            leftover_long = lo;
        }

        // Short-side allocation (if any).
        let short_indices: Vec<usize> = (0..n).filter(|&i| self.weights[i] < 0.0).collect();
        let short_total: f64 = short_indices.iter().map(|&i| -self.weights[i]).sum();
        let mut leftover_short = short_budget;
        if has_shorts && short_total > 0.0 && short_budget > 0.0 {
            let (s, lo) = greedy_one_side(
                &short_indices,
                &self.weights,
                &self.prices,
                short_budget,
                short_total,
                true,
            );
            for (i, sh) in s {
                shares[i] = sh;
            }
            leftover_short = lo;
        }

        let allocation: HashMap<usize, i64> = shares
            .iter()
            .enumerate()
            .filter(|(_, &s)| s != 0)
            .map(|(i, &s)| (i, s))
            .collect();

        Ok((allocation, leftover_long + leftover_short))
    }

    /// Rounded allocation — each position is simply rounded to the nearest
    /// whole share. May slightly overspend; use `greedy_portfolio` when the
    /// budget must not be exceeded.
    pub fn rounded_portfolio(&self) -> Result<(HashMap<usize, i64>, f64)> {
        let n = self.weights.len();
        let mut allocation = HashMap::new();
        let mut spent = 0.0_f64;
        for i in 0..n {
            let ideal = self.weights[i] * self.total_portfolio_value / self.prices[i];
            let s = ideal.round() as i64;
            if s != 0 {
                allocation.insert(i, s);
                spent += s as f64 * self.prices[i];
            }
        }
        let leftover = self.total_portfolio_value - spent;
        Ok((allocation, leftover))
    }

    /// Total market value of an allocation.
    pub fn allocation_value(&self, allocation: &HashMap<usize, i64>) -> f64 {
        allocation
            .iter()
            .map(|(&i, &s)| s as f64 * self.prices[i])
            .sum()
    }
}

/// Greedy single-side allocator. `weights` is the *full* weight vector;
/// only entries listed in `indices` are allocated. When `is_short` is
/// true the absolute weight is used for sizing and the resulting share
/// counts are negated.
fn greedy_one_side(
    indices: &[usize],
    weights: &DVector<f64>,
    prices: &DVector<f64>,
    budget: f64,
    weight_total: f64,
    is_short: bool,
) -> (Vec<(usize, i64)>, f64) {
    // Ideal continuous count, scaled so the side's weights are
    // re-normalised to sum to 1 within the side.
    let mut ideal = vec![0.0_f64; indices.len()];
    for (k, &i) in indices.iter().enumerate() {
        let w = weights[i].abs() / weight_total;
        ideal[k] = w * budget / prices[i];
    }
    let mut shares: Vec<i64> = ideal.iter().map(|&x| x.trunc() as i64).collect();
    let spent: f64 = (0..indices.len())
        .map(|k| shares[k] as f64 * prices[indices[k]])
        .sum();
    let mut remaining = budget - spent;

    let mut fracs: Vec<(usize, f64)> = (0..indices.len())
        .map(|k| (k, ideal[k] - ideal[k].trunc()))
        .collect();
    fracs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (k, _) in &fracs {
        let i = indices[*k];
        let price = prices[i];
        if remaining >= price - 1e-9 {
            shares[*k] += 1;
            remaining -= price;
        }
    }

    let mut out = Vec::with_capacity(indices.len());
    for (k, &i) in indices.iter().enumerate() {
        let s = if is_short { -shares[k] } else { shares[k] };
        if s != 0 {
            out.push((i, s));
        }
    }
    (out, remaining)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::dmatrix;

    #[test]
    fn get_latest_prices_returns_last_row() {
        let prices = dmatrix![
            100.0, 200.0;
            101.0, 199.0;
            102.0, 198.0
        ];
        let latest = get_latest_prices(&prices).unwrap();
        assert_eq!(latest.len(), 2);
        assert_relative_eq!(latest[0], 102.0);
        assert_relative_eq!(latest[1], 198.0);
    }

    #[test]
    fn greedy_does_not_exceed_budget() {
        let weights = DVector::from_vec(vec![0.6, 0.3, 0.1]);
        let prices = DVector::from_vec(vec![100.0, 50.0, 25.0]);
        let budget = 10_000.0;
        let da = DiscreteAllocation::new(weights, prices, budget).unwrap();
        let (alloc, leftover) = da.greedy_portfolio().unwrap();
        assert!(
            leftover >= -1e-9,
            "leftover should be non-negative, got {leftover}"
        );
        let spent = da.allocation_value(&alloc);
        assert!(
            spent <= budget + 1e-9,
            "spent {spent} exceeds budget {budget}"
        );
    }

    #[test]
    fn greedy_allocates_close_to_target() {
        let weights = DVector::from_vec(vec![0.5, 0.5]);
        let prices = DVector::from_vec(vec![100.0, 100.0]);
        let budget = 10_000.0;
        let da = DiscreteAllocation::new(weights, prices, budget).unwrap();
        let (alloc, leftover) = da.greedy_portfolio().unwrap();
        // 50 shares of each asset, no leftover.
        assert_eq!(*alloc.get(&0).unwrap_or(&0), 50);
        assert_eq!(*alloc.get(&1).unwrap_or(&0), 50);
        assert_relative_eq!(leftover, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn greedy_handles_fractional_shares() {
        // weights do not divide evenly — check total ≤ budget.
        let weights = DVector::from_vec(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        let prices = DVector::from_vec(vec![7.0, 11.0, 13.0]);
        let budget = 1_000.0;
        let da = DiscreteAllocation::new(weights, prices, budget).unwrap();
        let (alloc, leftover) = da.greedy_portfolio().unwrap();
        assert!(leftover >= -1e-9);
        let spent = da.allocation_value(&alloc);
        assert!(spent <= budget + 1e-9);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let w = DVector::from_vec(vec![0.5, 0.5]);
        let p = DVector::from_vec(vec![100.0]);
        assert!(DiscreteAllocation::new(w, p, 1000.0).is_err());
    }

    #[test]
    fn non_positive_price_errors() {
        let w = DVector::from_vec(vec![1.0]);
        let p = DVector::from_vec(vec![0.0]);
        assert!(DiscreteAllocation::new(w, p, 1000.0).is_err());
    }

    #[test]
    fn rounded_portfolio_allocates_nearest_share() {
        let weights = DVector::from_vec(vec![0.5, 0.5]);
        let prices = DVector::from_vec(vec![100.0, 100.0]);
        let budget = 10_100.0; // 50.5 shares each → rounds to 51
        let da = DiscreteAllocation::new(weights, prices, budget).unwrap();
        let (alloc, _) = da.rounded_portfolio().unwrap();
        assert_eq!(*alloc.get(&0).unwrap_or(&0), 51);
        assert_eq!(*alloc.get(&1).unwrap_or(&0), 51);
    }

    #[test]
    fn greedy_handles_signed_weights_with_shorts() {
        // 130/30: 130% long, 30% short. Long weights sum to 1.3, short to 0.3.
        let weights = DVector::from_vec(vec![0.7, 0.6, -0.3]);
        let prices = DVector::from_vec(vec![100.0, 200.0, 50.0]);
        let da = DiscreteAllocation::new(weights, prices, 10_000.0).unwrap();
        let (alloc, _leftover) = da.greedy_portfolio().unwrap();
        // Shorts are signed negative.
        assert!(alloc.get(&2).copied().unwrap_or(0) < 0);
        // Longs are positive.
        assert!(alloc.get(&0).copied().unwrap_or(0) > 0);
        assert!(alloc.get(&1).copied().unwrap_or(0) > 0);
    }

    #[test]
    fn zero_weight_assets_are_absent_from_allocation() {
        let weights = DVector::from_vec(vec![1.0, 0.0]);
        let prices = DVector::from_vec(vec![50.0, 200.0]);
        let da = DiscreteAllocation::new(weights, prices, 5_000.0).unwrap();
        let (alloc, _) = da.greedy_portfolio().unwrap();
        assert!(!alloc.contains_key(&1));
    }
}
