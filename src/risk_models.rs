//! Covariance / risk-matrix estimators.
//!
//! Mirrors `pypfopt.risk_models`. Inputs are price matrices (`T x N`); the
//! returns are derived internally via simple percent change. Outputs are
//! annualised `N x N` covariance matrices.

use nalgebra::{DMatrix, DVector};

use crate::prelude::{column_means, returns_from_prices, sample_covariance, symmetrise};
use crate::{PortfolioError, Result, TRADING_DAYS_PER_YEAR};

/// Plain sample covariance, annualised.
pub fn sample_cov(prices: &DMatrix<f64>, frequency: Option<usize>) -> Result<DMatrix<f64>> {
    let returns = returns_from_prices(prices)?;
    let cov = sample_covariance(&returns)?;
    let f = frequency.unwrap_or(TRADING_DAYS_PER_YEAR) as f64;
    Ok(cov * f)
}

/// Semi-covariance: covariance computed only over downside deviations
/// below `benchmark` (per-period).
pub fn semicovariance(
    prices: &DMatrix<f64>,
    benchmark: f64,
    frequency: Option<usize>,
) -> Result<DMatrix<f64>> {
    let returns = returns_from_prices(prices)?;
    let (rows, cols) = returns.shape();
    if rows < 2 {
        return Err(PortfolioError::InvalidArgument(
            "need at least two observations for semicovariance".into(),
        ));
    }
    // Drop everything above the benchmark to zero (PyPortfolioOpt's
    // implementation), then compute the population covariance with divisor
    // `T` (matches pypfopt — note: NOT T-1).
    let mut downside = returns.clone();
    for j in 0..cols {
        for i in 0..rows {
            let v = downside[(i, j)] - benchmark;
            downside[(i, j)] = v.min(0.0);
        }
    }
    let cov = downside.transpose() * &downside / (rows as f64);
    let f = frequency.unwrap_or(TRADING_DAYS_PER_YEAR) as f64;
    Ok(cov * f)
}

/// Exponentially-weighted covariance with span `span`. Most recent
/// observations carry the largest weight (`adjust=True` semantics).
pub fn exp_cov(
    prices: &DMatrix<f64>,
    span: usize,
    frequency: Option<usize>,
) -> Result<DMatrix<f64>> {
    if span < 1 {
        return Err(PortfolioError::InvalidArgument("span must be >= 1".into()));
    }
    let returns = returns_from_prices(prices)?;
    let (rows, cols) = returns.shape();
    if rows < 2 {
        return Err(PortfolioError::InvalidArgument(
            "need at least two observations for exp_cov".into(),
        ));
    }

    let alpha = 2.0 / (span as f64 + 1.0);
    let one_minus_alpha = 1.0 - alpha;

    // Pre-compute weights so the reduction cost is O(T) per pass.
    let mut weights = vec![0.0_f64; rows];
    let mut w = 1.0;
    for i in (0..rows).rev() {
        weights[i] = w;
        w *= one_minus_alpha;
    }
    let weight_sum: f64 = weights.iter().sum();

    // Weighted column means.
    let mut means = DVector::<f64>::zeros(cols);
    for j in 0..cols {
        let mut acc = 0.0;
        for i in 0..rows {
            acc += weights[i] * returns[(i, j)];
        }
        means[j] = acc / weight_sum;
    }

    let mut cov = DMatrix::<f64>::zeros(cols, cols);
    for j in 0..cols {
        for k in 0..=j {
            let mut acc = 0.0;
            for i in 0..rows {
                acc += weights[i] * (returns[(i, j)] - means[j]) * (returns[(i, k)] - means[k]);
            }
            let v = acc / weight_sum;
            cov[(j, k)] = v;
            cov[(k, j)] = v;
        }
    }
    let f = frequency.unwrap_or(TRADING_DAYS_PER_YEAR) as f64;
    Ok(cov * f)
}

/// Convert a covariance matrix to a correlation matrix.
pub fn cov_to_corr(cov: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let n = crate::prelude::assert_square(cov, "cov_to_corr")?;
    let mut std = DVector::<f64>::zeros(n);
    for i in 0..n {
        let v = cov[(i, i)];
        if v < 0.0 {
            return Err(PortfolioError::InvalidArgument(format!(
                "negative variance at index {i}"
            )));
        }
        std[i] = v.sqrt();
    }
    let mut corr = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if std[i] == 0.0 || std[j] == 0.0 {
                corr[(i, j)] = 0.0;
            } else {
                corr[(i, j)] = cov[(i, j)] / (std[i] * std[j]);
            }
        }
    }
    // Ensure unit diagonal exactly.
    for i in 0..n {
        corr[(i, i)] = 1.0;
    }
    Ok(corr)
}

/// Reconstruct a covariance matrix from a correlation matrix and a
/// vector of standard deviations.
pub fn corr_to_cov(corr: &DMatrix<f64>, std: &DVector<f64>) -> Result<DMatrix<f64>> {
    let n = crate::prelude::assert_square(corr, "corr_to_cov")?;
    if std.len() != n {
        return Err(PortfolioError::DimensionMismatch(format!(
            "std has length {}, expected {n}",
            std.len()
        )));
    }
    let mut cov = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            cov[(i, j)] = corr[(i, j)] * std[i] * std[j];
        }
    }
    Ok(cov)
}

// ---------------------------------------------------------------------------
// Covariance shrinkage
// ---------------------------------------------------------------------------

/// Shrinkage estimators that pull a sample covariance toward a structured
/// target. Construct from a price matrix, then call one of the methods to
/// get a numerical estimate.
pub struct CovarianceShrinkage {
    /// Period-over-period returns (`T x N`).
    pub returns: DMatrix<f64>,
    /// Annualisation factor.
    pub frequency: f64,
}

impl CovarianceShrinkage {
    pub fn new(prices: &DMatrix<f64>, frequency: Option<usize>) -> Result<Self> {
        let returns = returns_from_prices(prices)?;
        Ok(Self {
            returns,
            frequency: frequency.unwrap_or(TRADING_DAYS_PER_YEAR) as f64,
        })
    }

    /// Number of observations.
    fn t(&self) -> usize {
        self.returns.nrows()
    }

    /// Number of assets.
    fn n(&self) -> usize {
        self.returns.ncols()
    }

    /// The constant-correlation target: average pairwise correlation
    /// blended with the original variances. This is PyPortfolioOpt's
    /// default `shrinkage_target`.
    fn constant_correlation_target(&self, sample: &DMatrix<f64>) -> DMatrix<f64> {
        let n = self.n();
        let mut std = DVector::<f64>::zeros(n);
        for i in 0..n {
            std[i] = sample[(i, i)].sqrt();
        }
        let mut sum_corr = 0.0;
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if std[i] > 0.0 && std[j] > 0.0 {
                    sum_corr += sample[(i, j)] / (std[i] * std[j]);
                    count += 1;
                }
            }
        }
        let avg_corr = if count > 0 { sum_corr / count as f64 } else { 0.0 };
        let mut target = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    target[(i, j)] = sample[(i, i)];
                } else {
                    target[(i, j)] = avg_corr * std[i] * std[j];
                }
            }
        }
        target
    }

    /// Apply a fixed shrinkage intensity `delta` toward the
    /// constant-correlation target. Used internally by the named methods.
    fn shrink_toward(&self, sample: &DMatrix<f64>, target: &DMatrix<f64>, delta: f64) -> DMatrix<f64> {
        let mut shrunk = (1.0 - delta) * sample + delta * target;
        symmetrise(&mut shrunk);
        shrunk
    }

    /// Sample covariance (centered, divisor `T-1`), un-annualised.
    fn sample_cov_raw(&self) -> Result<DMatrix<f64>> {
        sample_covariance(&self.returns)
    }

    /// Centered returns and column means (helper used by the optimal
    /// intensity estimators).
    fn centered(&self) -> (DMatrix<f64>, DVector<f64>) {
        let means = column_means(&self.returns);
        let (rows, cols) = self.returns.shape();
        let mut centered = self.returns.clone();
        for j in 0..cols {
            for i in 0..rows {
                centered[(i, j)] -= means[j];
            }
        }
        (centered, means)
    }

    /// Ledoit-Wolf shrinkage with the constant-correlation target
    /// (Ledoit & Wolf 2003). Returns annualised covariance.
    pub fn ledoit_wolf(&self) -> Result<DMatrix<f64>> {
        let t = self.t() as f64;
        if t < 2.0 {
            return Err(PortfolioError::InvalidArgument(
                "need at least two observations for Ledoit-Wolf".into(),
            ));
        }

        let sample = self.sample_cov_raw()?;
        let target = self.constant_correlation_target(&sample);

        // Closed-form optimal intensity:
        //
        //   δ* = max(0, min(1, (π - ρ) / γ / T))
        //
        // where π = sum of variances of S_ij, γ = ||S - F||_F^2,
        // ρ = sum_i AsyVar(S_ii) + sum_{i≠j} (avg_corr/2) * (sqrt(s_jj/s_ii) * π_{ii,ij}
        //     + sqrt(s_ii/s_jj) * π_{jj,ij})
        //
        // This follows the PyPortfolioOpt implementation, which in turn
        // tracks the Ledoit-Wolf 2003 paper.
        let n = self.n();
        let (centered, _) = self.centered();

        // π_ij = (1/T) sum_t ((x_it - μ_i)(x_jt - μ_j) - s_ij)^2
        let mut pi_mat = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for tt in 0..self.t() {
                    let prod = centered[(tt, i)] * centered[(tt, j)];
                    let dev = prod - sample[(i, j)];
                    acc += dev * dev;
                }
                pi_mat[(i, j)] = acc / t;
            }
        }
        let pi: f64 = pi_mat.iter().sum();

        // γ = sum_{ij} (S_ij - F_ij)^2
        let mut gamma = 0.0;
        for i in 0..n {
            for j in 0..n {
                let d = sample[(i, j)] - target[(i, j)];
                gamma += d * d;
            }
        }

        // ρ = sum_i π_ii + (avg_corr / 2) * sum_{i≠j} (
        //     sqrt(s_jj/s_ii) * θ_iijj + sqrt(s_ii/s_jj) * θ_jjij ),
        // where θ_iikj = (1/T) sum_t ( (x_it - μ_i)^2 - s_ii ) * ( (x_kt - μ_k)(x_jt - μ_j) - s_kj )
        let mut std = DVector::<f64>::zeros(n);
        for i in 0..n {
            std[i] = sample[(i, i)].sqrt();
        }
        let mut sum_corr = 0.0;
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if std[i] > 0.0 && std[j] > 0.0 {
                    sum_corr += sample[(i, j)] / (std[i] * std[j]);
                    count += 1;
                }
            }
        }
        let avg_corr = if count > 0 { sum_corr / count as f64 } else { 0.0 };

        let theta = |i: usize, j: usize, k: usize, l: usize| -> f64 {
            let mut acc = 0.0;
            for tt in 0..self.t() {
                let a = centered[(tt, i)] * centered[(tt, j)] - sample[(i, j)];
                let b = centered[(tt, k)] * centered[(tt, l)] - sample[(k, l)];
                acc += a * b;
            }
            acc / t
        };

        let mut rho_diag = 0.0;
        for i in 0..n {
            rho_diag += pi_mat[(i, i)];
        }
        let mut rho_off = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if std[i] == 0.0 || std[j] == 0.0 {
                    continue;
                }
                let term = (std[j] / std[i]) * theta(i, i, i, j)
                    + (std[i] / std[j]) * theta(j, j, i, j);
                rho_off += 0.5 * avg_corr * term;
            }
        }
        let rho = rho_diag + rho_off;

        let kappa = if gamma > 0.0 { (pi - rho) / gamma } else { 0.0 };
        let delta = (kappa / t).clamp(0.0, 1.0);

        let shrunk = self.shrink_toward(&sample, &target, delta);
        Ok(shrunk * self.frequency)
    }

    /// Oracle Approximating Shrinkage (Chen et al., 2010) with the
    /// scaled-identity target. Returns annualised covariance.
    pub fn oracle_approximating(&self) -> Result<DMatrix<f64>> {
        let t = self.t() as f64;
        let n = self.n() as f64;
        if t < 2.0 {
            return Err(PortfolioError::InvalidArgument(
                "need at least two observations for oracle shrinkage".into(),
            ));
        }
        let sample = self.sample_cov_raw()?;
        let trace_s: f64 = (0..self.n()).map(|i| sample[(i, i)]).sum();
        let mu = trace_s / n;
        let target = DMatrix::<f64>::identity(self.n(), self.n()) * mu;

        let trace_s2: f64 = sample.iter().map(|x| x * x).sum();
        let trace_s_sq = trace_s.powi(2);

        // ρ = ((1 - 2/N) * trace(S²) + trace(S)²)
        //   / ((T + 1 - 2/N) * (trace(S²) - trace(S)² / N))
        let num = (1.0 - 2.0 / n) * trace_s2 + trace_s_sq;
        let denom = (t + 1.0 - 2.0 / n) * (trace_s2 - trace_s_sq / n);
        let rho = if denom > 0.0 {
            (num / denom).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let shrunk = self.shrink_toward(&sample, &target, rho);
        Ok(shrunk * self.frequency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_prices(seed: u64) -> DMatrix<f64> {
        // Deterministic pseudo-random walk for two correlated assets.
        let mut state = seed;
        let next = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 33) as u32 as f64) / (u32::MAX as f64) - 0.5
        };
        let rows = 200;
        let mut p = DMatrix::<f64>::zeros(rows, 3);
        p[(0, 0)] = 100.0;
        p[(0, 1)] = 50.0;
        p[(0, 2)] = 20.0;
        for i in 1..rows {
            let shock = next(&mut state) * 0.02;
            p[(i, 0)] = p[(i - 1, 0)] * (1.0 + shock + 0.0005);
            p[(i, 1)] = p[(i - 1, 1)] * (1.0 + 0.7 * shock + next(&mut state) * 0.005);
            p[(i, 2)] = p[(i - 1, 2)] * (1.0 + next(&mut state) * 0.01);
        }
        p
    }

    #[test]
    fn sample_cov_is_symmetric_and_psd_ish() {
        let p = make_prices(7);
        let cov = sample_cov(&p, Some(252)).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(cov[(i, j)], cov[(j, i)], max_relative = 1e-12);
            }
            // Variance is positive
            assert!(cov[(i, i)] > 0.0);
        }
    }

    #[test]
    fn semicov_matches_pypfopt_formula() {
        // Diagonal of `semicovariance(p, target)` is the population mean of
        // `min(r - target, 0)²`, annualised. Reproduce the formula directly
        // and compare element-wise.
        let p = make_prices(11);
        let semi = semicovariance(&p, 0.0, Some(252)).unwrap();
        let returns = returns_from_prices(&p).unwrap();
        let (t, n) = returns.shape();
        for i in 0..n {
            let mut acc = 0.0;
            for k in 0..t {
                let d = (returns[(k, i)] - 0.0).min(0.0);
                acc += d * d;
            }
            let expected = acc / (t as f64) * 252.0;
            assert_relative_eq!(semi[(i, i)], expected, max_relative = 1e-12);
            // And it must be non-negative (downside variance is a real moment).
            assert!(semi[(i, i)] >= 0.0);
        }
    }

    #[test]
    fn exp_cov_recovers_sample_for_huge_span() {
        // As span → ∞ all weights are ~equal so exp_cov ≈ sample_cov
        // (modulo divisor convention: exp_cov uses weight-sum, sample
        // uses T-1). For a moderately large span the variances will be
        // close on the same order of magnitude.
        let p = make_prices(3);
        let cov = sample_cov(&p, Some(252)).unwrap();
        let ewma = exp_cov(&p, 10_000, Some(252)).unwrap();
        for i in 0..3 {
            // Within 5% of the sample covariance for a non-extreme span.
            assert!((ewma[(i, i)] - cov[(i, i)]).abs() / cov[(i, i)] < 0.05);
        }
    }

    #[test]
    fn cov_corr_round_trip() {
        let p = make_prices(13);
        let cov = sample_cov(&p, Some(252)).unwrap();
        let corr = cov_to_corr(&cov).unwrap();
        let std = DVector::from_iterator(3, (0..3).map(|i| cov[(i, i)].sqrt()));
        let cov2 = corr_to_cov(&corr, &std).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(cov[(i, j)], cov2[(i, j)], max_relative = 1e-9);
            }
        }
    }

    #[test]
    fn corr_diagonal_is_one() {
        let p = make_prices(17);
        let cov = sample_cov(&p, Some(252)).unwrap();
        let corr = cov_to_corr(&cov).unwrap();
        for i in 0..3 {
            assert_relative_eq!(corr[(i, i)], 1.0, max_relative = 1e-12);
        }
    }

    #[test]
    fn ledoit_wolf_runs_and_is_symmetric() {
        let p = make_prices(19);
        let shrunk = CovarianceShrinkage::new(&p, Some(252))
            .unwrap()
            .ledoit_wolf()
            .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(shrunk[(i, j)], shrunk[(j, i)], max_relative = 1e-9);
            }
            assert!(shrunk[(i, i)] > 0.0);
        }
    }

    #[test]
    fn oracle_runs_and_is_symmetric() {
        let p = make_prices(23);
        let shrunk = CovarianceShrinkage::new(&p, Some(252))
            .unwrap()
            .oracle_approximating()
            .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(shrunk[(i, j)], shrunk[(j, i)], max_relative = 1e-9);
            }
            assert!(shrunk[(i, i)] > 0.0);
        }
    }
}
