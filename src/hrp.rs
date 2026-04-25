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

pub struct HRPOpt {
    pub returns: DMatrix<f64>,
}

impl HRPOpt {
    pub fn from_returns(returns: DMatrix<f64>) -> Result<Self> {
        if returns.nrows() < 2 || returns.ncols() < 2 {
            return Err(PortfolioError::InvalidArgument(
                "HRP needs at least 2 observations and 2 assets".into(),
            ));
        }
        Ok(Self { returns })
    }

    pub fn from_prices(prices: &DMatrix<f64>) -> Result<Self> {
        Self::from_returns(returns_from_prices(prices)?)
    }

    pub fn optimize(&self) -> Result<DVector<f64>> {
        let cov = sample_covariance(&self.returns)?;
        let corr = cov_to_corr(&cov)?;
        let dist = correlation_distance(&corr);
        let order = quasi_diagonalise(&dist);
        let weights_in_order = recursive_bisection(&cov, &order);

        let n = order.len();
        let mut w = DVector::<f64>::zeros(n);
        for (rank, &asset) in order.iter().enumerate() {
            w[asset] = weights_in_order[rank];
        }
        Ok(w)
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
/// flat list of asset indices.
pub fn quasi_diagonalise(dist: &DMatrix<f64>) -> Vec<usize> {
    let n = dist.nrows();
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while clusters.len() > 1 {
        let mut best_i = 0_usize;
        let mut best_j = 1_usize;
        let mut best_d = f64::INFINITY;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let mut d_ij = f64::INFINITY;
                for &a in &clusters[i] {
                    for &b in &clusters[j] {
                        if dist[(a, b)] < d_ij {
                            d_ij = dist[(a, b)];
                        }
                    }
                }
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
        for k in start..mid {
            w[k] *= alpha;
        }
        for k in mid..end {
            w[k] *= 1.0 - alpha;
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
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
        let hrp = HRPOpt::from_returns(r).unwrap();
        let w = hrp.optimize().unwrap();
        let total: f64 = w.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn weights_are_long_only() {
        let r = synthetic_returns();
        let hrp = HRPOpt::from_returns(r).unwrap();
        let w = hrp.optimize().unwrap();
        for v in w.iter() {
            assert!(*v >= 0.0);
        }
    }
}
