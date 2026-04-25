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

use nalgebra::DVector;

use crate::{PortfolioError, Result};

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
        Ok(Self { weights, prices, total_portfolio_value })
    }

    /// Greedy allocation.
    ///
    /// Returns `(allocation, leftover)` where `allocation` maps asset
    /// index → number of shares (signed: positive = long, negative = short)
    /// and `leftover` is remaining unallocated cash.
    pub fn greedy_portfolio(&self) -> Result<(HashMap<usize, i64>, f64)> {
        let n = self.weights.len();

        // Ideal (continuous) share count for each asset.
        let mut ideal = vec![0.0_f64; n];
        for i in 0..n {
            ideal[i] = self.weights[i] * self.total_portfolio_value / self.prices[i];
        }

        // Floor toward zero for every position.
        let mut shares: Vec<i64> = ideal.iter().map(|&x| x.trunc() as i64).collect();
        let spent: f64 = (0..n)
            .map(|i| shares[i] as f64 * self.prices[i])
            .sum();
        let mut remaining = self.total_portfolio_value - spent;

        // Sort assets by fractional remainder (descending) to decide
        // where to allocate leftover cash.
        let mut fracs: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let frac = ideal[i] - ideal[i].trunc();
                // For short positions the "fractional" shortfall is 1 - |frac|
                // so we weight by how "close" they are to needing one more share.
                let priority = if ideal[i] >= 0.0 { frac } else { 1.0 + frac };
                (i, priority)
            })
            .collect();
        fracs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Greedily buy/sell one additional share per asset if affordable.
        for (i, _) in &fracs {
            let price = self.prices[*i];
            if ideal[*i] >= 0.0 {
                // Long: buy one more share if we have cash.
                if remaining >= price - 1e-9 {
                    shares[*i] += 1;
                    remaining -= price;
                }
            } else {
                // Short: the floor already rounds toward zero; there is
                // no "buy more" action needed.
            }
        }

        let allocation: HashMap<usize, i64> = shares
            .iter()
            .enumerate()
            .filter(|(_, &s)| s != 0)
            .map(|(i, &s)| (i, s))
            .collect();

        Ok((allocation, remaining))
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn greedy_does_not_exceed_budget() {
        let weights = DVector::from_vec(vec![0.6, 0.3, 0.1]);
        let prices = DVector::from_vec(vec![100.0, 50.0, 25.0]);
        let budget = 10_000.0;
        let da = DiscreteAllocation::new(weights, prices, budget).unwrap();
        let (alloc, leftover) = da.greedy_portfolio().unwrap();
        assert!(leftover >= -1e-9, "leftover should be non-negative, got {leftover}");
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
    fn zero_weight_assets_are_absent_from_allocation() {
        let weights = DVector::from_vec(vec![1.0, 0.0]);
        let prices = DVector::from_vec(vec![50.0, 200.0]);
        let da = DiscreteAllocation::new(weights, prices, 5_000.0).unwrap();
        let (alloc, _) = da.greedy_portfolio().unwrap();
        assert!(!alloc.contains_key(&1));
    }
}
