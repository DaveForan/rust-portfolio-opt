//! Hierarchical Risk Parity (López de Prado, 2016).
//!
//! Mirrors `pypfopt.hierarchical_portfolio.HRPOpt`. The algorithm:
//!
//! 1. Convert the asset correlation matrix into a distance metric.
//! 2. Build a single-linkage agglomerative hierarchy over those
//!    distances.
//! 3. Reorder the assets so similar (correlated) ones sit next to each
//!    other in the leaf order ("quasi-diagonalisation").
//! 4. Recursively bisect the ordered set, distributing risk inversely
//!    to each side's volatility ("recursive bisection").

use nalgebra::{DMatrix, DVector};

use crate::prelude::{returns_from_prices, sample_covariance};
use crate::risk_models::cov_to_corr;
use crate::{PortfolioError, Result};

/// Linkage method used during HRP's agglomerative clustering. Mirrors
/// scipy's `linkage(method=...)`: `single` = nearest neighbour,
/// `complete` = farthest neighbour, `average` = unweighted mean
/// (UPGMA). Default is [`LinkageMethod::Single`] (PyPortfolioOpt's
/// default).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LinkageMethod {
    #[default]
    Single,
    Complete,
    Average,
}

pub struct HRPOpt {
    /// Returns matrix used to derive the covariance and clustering.
    /// Empty if [`Self::from_cov_matrix`] was used.
    pub returns: DMatrix<f64>,
    /// Override for the covariance matrix (set by `from_cov_matrix`).
    pub cov_override: Option<DMatrix<f64>>,
    pub linkage: LinkageMethod,
    pub weights: Option<DVector<f64>>,
}

impl HRPOpt {
    pub fn from_returns(returns: DMatrix<f64>) -> Result<Self> {
        if returns.nrows() < 2 || returns.ncols() < 2 {
            return Err(PortfolioError::InvalidArgument(
                "HRP needs at least 2 observations and 2 assets".into(),
            ));
        }
        Ok(Self {
            returns,
            cov_override: None,
            linkage: LinkageMethod::default(),
            weights: None,
        })
    }

    pub fn from_prices(prices: &DMatrix<f64>) -> Result<Self> {
        Self::from_returns(returns_from_prices(prices)?)
    }

    /// Build an HRP optimiser directly from a covariance matrix. The
    /// `returns` field is left empty; only optimisation works.
    pub fn from_cov_matrix(cov: DMatrix<f64>) -> Result<Self> {
        crate::prelude::assert_square(&cov, "HRPOpt::from_cov_matrix")?;
        if cov.nrows() < 2 {
            return Err(PortfolioError::InvalidArgument(
                "HRP needs at least 2 assets".into(),
            ));
        }
        Ok(Self {
            returns: DMatrix::<f64>::zeros(0, cov.nrows()),
            cov_override: Some(cov),
            linkage: LinkageMethod::default(),
            weights: None,
        })
    }

    /// Override the linkage method used during clustering.
    pub fn with_linkage(mut self, method: LinkageMethod) -> Self {
        self.linkage = method;
        self
    }

    fn cov(&self) -> Result<DMatrix<f64>> {
        if let Some(c) = &self.cov_override {
            return Ok(c.clone());
        }
        sample_covariance(&self.returns)
    }

    pub fn optimize(&mut self) -> Result<DVector<f64>> {
        let cov = self.cov()?;
        let corr = cov_to_corr(&cov)?;
        let dist = correlation_distance(&corr);
        let order = quasi_diagonalise_with(&dist, self.linkage);
        let weights_in_order = recursive_bisection(&cov, &order);

        let n = order.len();
        let mut w = DVector::<f64>::zeros(n);
        for (rank, &asset) in order.iter().enumerate() {
            w[asset] = weights_in_order[rank];
        }
        self.weights = Some(w.clone());
        Ok(w)
    }

    /// `(annualised return, annualised vol, Sharpe)` from the cached
    /// HRP weights and a caller-supplied expected-return vector. The
    /// internal covariance (computed from `self.returns`) is annualised
    /// by `frequency` (defaults to 252) so the result is comparable to
    /// the caller's annualised expected returns.
    pub fn portfolio_performance(
        &self,
        expected_returns: &DVector<f64>,
        risk_free_rate: f64,
        frequency: Option<usize>,
    ) -> Result<(f64, f64, f64)> {
        let w = self.weights.as_ref().ok_or_else(|| {
            PortfolioError::InvalidArgument(
                "no weights yet — call optimize before portfolio_performance".into(),
            )
        })?;
        if expected_returns.len() != w.len() {
            return Err(PortfolioError::DimensionMismatch(format!(
                "expected_returns length {} ≠ weight length {}",
                expected_returns.len(),
                w.len()
            )));
        }
        let f = frequency.unwrap_or(crate::TRADING_DAYS_PER_YEAR) as f64;
        // If the user supplied a precomputed cov (cov_override) they're
        // expected to have annualised it themselves; otherwise we
        // annualise the per-period sample cov.
        let cov = if self.cov_override.is_some() {
            self.cov()?
        } else {
            self.cov()? * f
        };
        let ret = expected_returns.dot(w);
        let var = (w.transpose() * &cov * w)[(0, 0)];
        let vol = var.max(0.0).sqrt();
        let sharpe = if vol > 0.0 {
            (ret - risk_free_rate) / vol
        } else {
            0.0
        };
        Ok((ret, vol, sharpe))
    }

    /// Round weights below `cutoff` to zero, renormalise, optionally
    /// round to `rounding` decimal places.
    pub fn clean_weights(&self, cutoff: f64, rounding: Option<u32>) -> Result<DVector<f64>> {
        let w = self.weights.as_ref().ok_or_else(|| {
            PortfolioError::InvalidArgument(
                "no weights yet — call optimize before clean_weights".into(),
            )
        })?;
        Ok(crate::prelude::clean_weights(w, cutoff, rounding))
    }
}

/// `d_ij = sqrt(0.5 * (1 - ρ_ij))`.
pub fn correlation_distance(corr: &DMatrix<f64>) -> DMatrix<f64> {
    let n = corr.nrows();
    let mut d = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let v = (0.5 * (1.0 - corr[(i, j)])).max(0.0);
            d[(i, j)] = v.sqrt();
        }
    }
    d
}

/// Single-linkage agglomerative clustering. Returns the leaf order as a
/// flat list of asset indices. Equivalent to
/// `quasi_diagonalise_with(dist, LinkageMethod::Single)`.
pub fn quasi_diagonalise(dist: &DMatrix<f64>) -> Vec<usize> {
    quasi_diagonalise_with(dist, LinkageMethod::Single)
}

/// Agglomerative clustering with the chosen [`LinkageMethod`].
pub fn quasi_diagonalise_with(dist: &DMatrix<f64>, method: LinkageMethod) -> Vec<usize> {
    let n = dist.nrows();
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while clusters.len() > 1 {
        let mut best_i = 0_usize;
        let mut best_j = 1_usize;
        let mut best_d = f64::INFINITY;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let d_ij = cluster_distance(dist, &clusters[i], &clusters[j], method);
                if d_ij < best_d {
                    best_d = d_ij;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        let merged_tail = clusters.remove(best_j);
        clusters[best_i].extend(merged_tail);
    }
    clusters.into_iter().next().unwrap_or_default()
}

fn cluster_distance(dist: &DMatrix<f64>, a: &[usize], b: &[usize], method: LinkageMethod) -> f64 {
    match method {
        LinkageMethod::Single => {
            let mut best = f64::INFINITY;
            for &x in a {
                for &y in b {
                    if dist[(x, y)] < best {
                        best = dist[(x, y)];
                    }
                }
            }
            best
        }
        LinkageMethod::Complete => {
            let mut worst = f64::NEG_INFINITY;
            for &x in a {
                for &y in b {
                    if dist[(x, y)] > worst {
                        worst = dist[(x, y)];
                    }
                }
            }
            worst
        }
        LinkageMethod::Average => {
            let mut acc = 0.0;
            let mut count = 0_usize;
            for &x in a {
                for &y in b {
                    acc += dist[(x, y)];
                    count += 1;
                }
            }
            if count == 0 {
                f64::INFINITY
            } else {
                acc / count as f64
            }
        }
    }
}

/// Recursive bisection: split the order in two, weight inversely to
/// each cluster's variance, recurse.
pub fn recursive_bisection(cov: &DMatrix<f64>, order: &[usize]) -> Vec<f64> {
    let n = order.len();
    let mut w = vec![1.0_f64; n];

    let mut stack: Vec<(usize, usize)> = vec![(0, n)];
    while let Some((start, end)) = stack.pop() {
        if end - start <= 1 {
            continue;
        }
        let mid = (start + end) / 2;
        let left = &order[start..mid];
        let right = &order[mid..end];
        let var_left = cluster_variance(cov, left);
        let var_right = cluster_variance(cov, right);
        let inv_left = 1.0 / var_left;
        let inv_right = 1.0 / var_right;
        let alpha = inv_left / (inv_left + inv_right);
        for v in w.iter_mut().take(mid).skip(start) {
            *v *= alpha;
        }
        for v in w.iter_mut().take(end).skip(mid) {
            *v *= 1.0 - alpha;
        }
        stack.push((start, mid));
        stack.push((mid, end));
    }
    w
}

fn cluster_variance(cov: &DMatrix<f64>, members: &[usize]) -> f64 {
    let n = members.len();
    if n == 0 {
        return 1e-12;
    }
    let mut iv = DVector::<f64>::zeros(n);
    let mut total = 0.0;
    for (k, &i) in members.iter().enumerate() {
        let v = cov[(i, i)].max(1e-12);
        iv[k] = 1.0 / v;
        total += iv[k];
    }
    if total > 0.0 {
        iv /= total;
    }
    let mut sub = DMatrix::<f64>::zeros(n, n);
    for (a, &i) in members.iter().enumerate() {
        for (b, &j) in members.iter().enumerate() {
            sub[(a, b)] = cov[(i, j)];
        }
    }
    let var = (iv.transpose() * &sub * &iv)[(0, 0)];
    var.max(1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn synthetic_returns() -> DMatrix<f64> {
        // 0 and 1 are highly correlated; 2 and 3 are highly correlated; 4 is independent.
        let rows = 200;
        let mut state = 42_u64;
        let next = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 33) as u32 as f64) / (u32::MAX as f64) - 0.5
        };
        let mut r = DMatrix::<f64>::zeros(rows, 5);
        for t in 0..rows {
            let a = next(&mut state) * 0.02;
            let b = next(&mut state) * 0.02;
            r[(t, 0)] = a + 0.001 * next(&mut state);
            r[(t, 1)] = a + 0.001 * next(&mut state);
            r[(t, 2)] = b + 0.001 * next(&mut state);
            r[(t, 3)] = b + 0.001 * next(&mut state);
            r[(t, 4)] = next(&mut state) * 0.02;
        }
        r
    }

    #[test]
    fn distance_is_zero_on_diagonal() {
        let r = synthetic_returns();
        let cov = sample_covariance(&r).unwrap();
        let corr = cov_to_corr(&cov).unwrap();
        let d = correlation_distance(&corr);
        for i in 0..5 {
            assert_relative_eq!(d[(i, i)], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn quasi_diag_keeps_correlated_pairs_adjacent() {
        let r = synthetic_returns();
        let cov = sample_covariance(&r).unwrap();
        let corr = cov_to_corr(&cov).unwrap();
        let d = correlation_distance(&corr);
        let order = quasi_diagonalise(&d);
        assert_eq!(order.len(), 5);
        let pos = |a: usize| order.iter().position(|&x| x == a).unwrap();
        let p01 = (pos(0) as i32 - pos(1) as i32).abs();
        let p23 = (pos(2) as i32 - pos(3) as i32).abs();
        assert_eq!(p01, 1, "0 and 1 must be adjacent in HRP order");
        assert_eq!(p23, 1, "2 and 3 must be adjacent in HRP order");
    }

    #[test]
    fn weights_sum_to_one() {
        let r = synthetic_returns();
        let mut hrp = HRPOpt::from_returns(r).unwrap();
        let w = hrp.optimize().unwrap();
        let total: f64 = w.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn weights_are_long_only() {
        let r = synthetic_returns();
        let mut hrp = HRPOpt::from_returns(r).unwrap();
        let w = hrp.optimize().unwrap();
        for v in w.iter() {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn linkage_methods_all_produce_valid_weights() {
        let r = synthetic_returns();
        for method in [
            LinkageMethod::Single,
            LinkageMethod::Complete,
            LinkageMethod::Average,
        ] {
            let mut hrp = HRPOpt::from_returns(r.clone())
                .unwrap()
                .with_linkage(method);
            let w = hrp.optimize().unwrap();
            let total: f64 = w.iter().sum();
            assert_relative_eq!(total, 1.0, epsilon = 1e-9);
            for v in w.iter() {
                assert!(*v >= 0.0);
            }
        }
    }

    #[test]
    fn from_cov_matrix_round_trips() {
        let r = synthetic_returns();
        let cov = sample_covariance(&r).unwrap();
        let mut hrp = HRPOpt::from_cov_matrix(cov).unwrap();
        let w = hrp.optimize().unwrap();
        let total: f64 = w.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn portfolio_performance_runs() {
        let r = synthetic_returns();
        let mut hrp = HRPOpt::from_returns(r).unwrap();
        hrp.optimize().unwrap();
        let mu = DVector::from_vec(vec![0.05, 0.06, 0.04, 0.05, 0.03]);
        let (ret, vol, _sh) = hrp.portfolio_performance(&mu, 0.0, None).unwrap();
        assert!(ret.is_finite());
        assert!(vol > 0.0);
    }
}
