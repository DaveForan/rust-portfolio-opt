//! Black-Litterman: blend an equilibrium prior with investor views.
//!
//! Mirrors `pypfopt.black_litterman`. The flow:
//!
//! 1. Estimate the market-implied risk aversion δ from the market portfolio.
//! 2. Derive the equilibrium prior `Π = δ · Σ · w_mkt`.
//! 3. Express investor views as a linear system `P · μ = Q ± Ω`.
//! 4. Combine prior + views into the posterior mean `bl_returns` and
//!    covariance `bl_cov`.
//!
//! Both absolute views ("asset *i* will return *q*") and relative views
//! ("asset *i* will outperform asset *j* by *q*") are expressed by
//! filling in rows of the picking matrix `P`.

use nalgebra::{DMatrix, DVector};

use crate::prelude::assert_square;
use crate::{PortfolioError, Result};

/// Recover the implied excess returns under market equilibrium.
///
/// `Π = δ · Σ · w_mkt`, where `δ` is the market-implied risk aversion and
/// `w_mkt` are market-cap weights.
pub fn market_implied_prior_returns(
    market_caps: &DVector<f64>,
    risk_aversion: f64,
    cov: &DMatrix<f64>,
    risk_free_rate: f64,
) -> Result<DVector<f64>> {
    let n = assert_square(cov, "market_implied_prior_returns cov")?;
    if market_caps.len() != n {
        return Err(PortfolioError::DimensionMismatch(format!(
            "market_caps length {} ≠ {n}",
            market_caps.len()
        )));
    }
    let total: f64 = market_caps.iter().sum();
    if total <= 0.0 {
        return Err(PortfolioError::InvalidArgument(
            "market caps must sum to a positive value".into(),
        ));
    }
    let w_mkt = market_caps / total;
    let pi = (cov * &w_mkt) * risk_aversion;
    let mut out = pi;
    for i in 0..n {
        out[i] += risk_free_rate;
    }
    Ok(out)
}

/// Implied risk aversion from the market portfolio:
///
/// `δ = (E[R_m] - r_f) / σ_m²`
pub fn market_implied_risk_aversion(
    market_excess_return: f64,
    market_variance: f64,
) -> Result<f64> {
    if market_variance <= 0.0 {
        return Err(PortfolioError::InvalidArgument(
            "market variance must be positive".into(),
        ));
    }
    Ok(market_excess_return / market_variance)
}

/// Black-Litterman posterior over expected returns and covariance.
pub struct BlackLittermanModel {
    pub cov: DMatrix<f64>,
    pub pi: DVector<f64>,
    /// Picking matrix `P` (k x n): each row is one view.
    pub p: DMatrix<f64>,
    /// View returns (length k).
    pub q: DVector<f64>,
    /// Diagonal of the view-uncertainty matrix Ω (length k).
    pub omega_diag: DVector<f64>,
    /// Prior scaling factor τ (typically 0.025–0.05).
    pub tau: f64,
}

impl BlackLittermanModel {
    /// Build the model. If `omega_diag` is `None`, He & Litterman's
    /// proportional rule is used: `Ω = diag(P · (τΣ) · Pᵀ)`.
    pub fn new(
        cov: DMatrix<f64>,
        pi: DVector<f64>,
        p: DMatrix<f64>,
        q: DVector<f64>,
        omega_diag: Option<DVector<f64>>,
        tau: f64,
    ) -> Result<Self> {
        let n = assert_square(&cov, "BlackLittermanModel cov")?;
        if pi.len() != n {
            return Err(PortfolioError::DimensionMismatch(format!(
                "pi length {} ≠ cov dim {n}",
                pi.len()
            )));
        }
        if p.ncols() != n {
            return Err(PortfolioError::DimensionMismatch(format!(
                "P column count {} ≠ {n}",
                p.ncols()
            )));
        }
        let k = p.nrows();
        if q.len() != k {
            return Err(PortfolioError::DimensionMismatch(format!(
                "q length {} ≠ P row count {k}",
                q.len()
            )));
        }
        if tau <= 0.0 {
            return Err(PortfolioError::InvalidArgument("tau must be > 0".into()));
        }

        let omega_diag = match omega_diag {
            Some(o) => {
                if o.len() != k {
                    return Err(PortfolioError::DimensionMismatch(format!(
                        "omega_diag length {} ≠ k = {k}",
                        o.len()
                    )));
                }
                o
            }
            None => {
                let tau_sigma = &cov * tau;
                let prod = &p * &tau_sigma * p.transpose();
                let mut diag = DVector::<f64>::zeros(k);
                for i in 0..k {
                    diag[i] = prod[(i, i)].max(1e-12);
                }
                diag
            }
        };

        Ok(Self { cov, pi, p, q, omega_diag, tau })
    }

    fn omega_matrix(&self) -> DMatrix<f64> {
        let k = self.omega_diag.len();
        let mut o = DMatrix::<f64>::zeros(k, k);
        for i in 0..k {
            o[(i, i)] = self.omega_diag[i];
        }
        o
    }

    /// Posterior mean: `μ_BL = (τ Σ)⁻¹ + Pᵀ Ω⁻¹ P)⁻¹  ((τ Σ)⁻¹ Π + Pᵀ Ω⁻¹ Q)`.
    pub fn bl_returns(&self) -> Result<DVector<f64>> {
        let tau_sigma = &self.cov * self.tau;
        let omega = self.omega_matrix();

        let tau_sigma_inv = tau_sigma
            .clone()
            .try_inverse()
            .ok_or_else(|| PortfolioError::Singular("τΣ is singular".into()))?;
        let omega_inv = omega
            .try_inverse()
            .ok_or_else(|| PortfolioError::Singular("Ω is singular".into()))?;

        let lhs = &tau_sigma_inv + self.p.transpose() * &omega_inv * &self.p;
        let rhs = &tau_sigma_inv * &self.pi + self.p.transpose() * &omega_inv * &self.q;
        let lhs_inv = lhs
            .try_inverse()
            .ok_or_else(|| PortfolioError::Singular("BL master matrix is singular".into()))?;
        Ok(lhs_inv * rhs)
    }

    /// Posterior covariance: `Σ_BL = Σ + ((τΣ)⁻¹ + Pᵀ Ω⁻¹ P)⁻¹`.
    pub fn bl_cov(&self) -> Result<DMatrix<f64>> {
        let tau_sigma = &self.cov * self.tau;
        let omega = self.omega_matrix();
        let tau_sigma_inv = tau_sigma
            .clone()
            .try_inverse()
            .ok_or_else(|| PortfolioError::Singular("τΣ is singular".into()))?;
        let omega_inv = omega
            .try_inverse()
            .ok_or_else(|| PortfolioError::Singular("Ω is singular".into()))?;
        let m = (&tau_sigma_inv + self.p.transpose() * &omega_inv * &self.p)
            .try_inverse()
            .ok_or_else(|| PortfolioError::Singular("BL master matrix is singular".into()))?;
        Ok(&self.cov + m)
    }
}

/// Build a row of the picking matrix encoding an absolute view on a
/// single asset. `n` is the total number of assets.
pub fn absolute_view(n: usize, asset_idx: usize) -> Result<DVector<f64>> {
    if asset_idx >= n {
        return Err(PortfolioError::InvalidArgument(format!(
            "asset_idx {asset_idx} out of range for n = {n}"
        )));
    }
    let mut row = DVector::<f64>::zeros(n);
    row[asset_idx] = 1.0;
    Ok(row)
}

/// Build a row encoding a relative view: `asset_long` outperforms
/// `asset_short` by some amount (the magnitude lives in `Q`).
pub fn relative_view(n: usize, asset_long: usize, asset_short: usize) -> Result<DVector<f64>> {
    if asset_long >= n || asset_short >= n || asset_long == asset_short {
        return Err(PortfolioError::InvalidArgument(
            "invalid relative view indices".into(),
        ));
    }
    let mut row = DVector::<f64>::zeros(n);
    row[asset_long] = 1.0;
    row[asset_short] = -1.0;
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn cov_3() -> DMatrix<f64> {
        DMatrix::from_row_slice(
            3,
            3,
            &[
                0.04, 0.01, 0.005,
                0.01, 0.09, 0.02,
                0.005, 0.02, 0.16,
            ],
        )
    }

    #[test]
    fn implied_risk_aversion_basic() {
        let delta = market_implied_risk_aversion(0.08, 0.04).unwrap();
        // (0.08 - 0) / 0.04 = 2.0
        assert_relative_eq!(delta, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn implied_prior_returns_match_formula() {
        let cov = cov_3();
        let caps = DVector::from_vec(vec![100.0, 200.0, 100.0]);
        let pi = market_implied_prior_returns(&caps, 2.0, &cov, 0.0).unwrap();
        // Manually:
        let total = 400.0;
        let w = DVector::from_vec(vec![100.0 / total, 200.0 / total, 100.0 / total]);
        let expected = (&cov * &w) * 2.0;
        for i in 0..3 {
            assert_relative_eq!(pi[i], expected[i], max_relative = 1e-12);
        }
    }

    #[test]
    fn bl_with_no_views_recovers_prior() {
        let cov = cov_3();
        let pi = DVector::from_vec(vec![0.05, 0.07, 0.12]);
        // Zero views: P is 0×3, Q is empty.
        let p = DMatrix::<f64>::zeros(0, 3);
        let q = DVector::<f64>::zeros(0);
        let omega = Some(DVector::<f64>::zeros(0));
        let bl = BlackLittermanModel::new(cov.clone(), pi.clone(), p, q, omega, 0.05).unwrap();

        // BL with no views returns the prior verbatim.
        let mu_bl = bl.bl_returns().unwrap();
        for i in 0..3 {
            assert_relative_eq!(mu_bl[i], pi[i], max_relative = 1e-9);
        }
        // BL covariance with no views = Σ + τΣ.
        let sigma_bl = bl.bl_cov().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(
                    sigma_bl[(i, j)],
                    cov[(i, j)] * (1.0 + 0.05),
                    max_relative = 1e-9
                );
            }
        }
    }

    #[test]
    fn bl_pulls_returns_toward_view() {
        let cov = cov_3();
        let pi = DVector::from_vec(vec![0.05, 0.07, 0.12]);
        // View: asset 0 returns 0.20 (way above prior 0.05)
        let row = absolute_view(3, 0).unwrap();
        let p = DMatrix::from_rows(&[row.transpose()]);
        let q = DVector::from_vec(vec![0.20]);
        let bl = BlackLittermanModel::new(cov, pi.clone(), p, q, None, 0.05).unwrap();
        let mu = bl.bl_returns().unwrap();
        // Posterior for asset 0 should land between prior (0.05) and view (0.20).
        assert!(mu[0] > pi[0]);
        assert!(mu[0] < 0.20);
    }

    #[test]
    fn relative_view_increases_long_decreases_short() {
        let cov = cov_3();
        let pi = DVector::from_vec(vec![0.05, 0.05, 0.05]);
        // Asset 0 outperforms asset 1 by 0.05.
        let row = relative_view(3, 0, 1).unwrap();
        let p = DMatrix::from_rows(&[row.transpose()]);
        let q = DVector::from_vec(vec![0.05]);
        let bl = BlackLittermanModel::new(cov, pi.clone(), p, q, None, 0.05).unwrap();
        let mu = bl.bl_returns().unwrap();
        assert!(mu[0] > pi[0]);
        assert!(mu[1] < pi[1]);
    }

    #[test]
    fn invalid_relative_view_indices() {
        assert!(relative_view(3, 0, 0).is_err());
        assert!(relative_view(3, 5, 1).is_err());
    }
}
