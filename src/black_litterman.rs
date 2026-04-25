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

use crate::prelude::{assert_square, returns_from_prices};
use crate::{PortfolioError, Result, TRADING_DAYS_PER_YEAR};

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

/// Implied risk aversion computed directly from a market price series.
/// Mirrors `pypfopt.black_litterman.market_implied_risk_aversion(prices,
/// frequency, risk_free_rate)`.
///
/// `market_prices` is a column matrix (or single-column `DMatrix`) of
/// market prices (e.g. SPY closes) in chronological order.
pub fn market_implied_risk_aversion_from_prices(
    market_prices: &DMatrix<f64>,
    frequency: Option<usize>,
    risk_free_rate: f64,
) -> Result<f64> {
    if market_prices.ncols() != 1 {
        return Err(PortfolioError::InvalidArgument(
            "market_prices must have exactly one column".into(),
        ));
    }
    let returns = returns_from_prices(market_prices)?;
    let t = returns.nrows();
    if t < 2 {
        return Err(PortfolioError::InvalidArgument(
            "need at least two observations for implied risk aversion".into(),
        ));
    }
    let f = frequency.unwrap_or(TRADING_DAYS_PER_YEAR) as f64;
    let mean: f64 = returns.iter().sum::<f64>() / t as f64;
    let var: f64 = returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / (t as f64 - 1.0);
    market_implied_risk_aversion(mean * f - risk_free_rate, var * f)
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
    /// Risk aversion δ used by [`Self::bl_weights`] when no override is
    /// supplied. Defaults to 1.0; mirrors PyPortfolioOpt's default.
    pub risk_aversion: f64,
    /// Cached posterior weights from the most recent `bl_weights` call.
    pub weights: Option<DVector<f64>>,
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

        Ok(Self {
            cov,
            pi,
            p,
            q,
            omega_diag,
            tau,
            risk_aversion: 1.0,
            weights: None,
        })
    }

    /// Override the default risk-aversion δ used by [`Self::bl_weights`].
    pub fn with_risk_aversion(mut self, delta: f64) -> Self {
        self.risk_aversion = delta;
        self
    }

    /// Idzorek's confidence-based view uncertainty. Given per-view
    /// confidences in `[0, 1]` returns the diagonal of Ω. Mirrors
    /// `BlackLittermanModel.idzorek_method` in PyPortfolioOpt.
    pub fn idzorek_omega(
        view_confidences: &DVector<f64>,
        cov_matrix: &DMatrix<f64>,
        p: &DMatrix<f64>,
        tau: f64,
    ) -> Result<DVector<f64>> {
        let k = p.nrows();
        if view_confidences.len() != k {
            return Err(PortfolioError::DimensionMismatch(format!(
                "view_confidences length {} ≠ k = {k}",
                view_confidences.len()
            )));
        }
        let mut out = DVector::<f64>::zeros(k);
        for i in 0..k {
            let conf = view_confidences[i];
            if !(0.0..=1.0).contains(&conf) {
                return Err(PortfolioError::InvalidArgument(
                    "view confidences must be in [0, 1]".into(),
                ));
            }
            if conf == 0.0 {
                out[i] = 1e6;
                continue;
            }
            let alpha = (1.0 - conf) / conf;
            let row = p.row(i).transpose();
            let v = (row.transpose() * cov_matrix * &row)[(0, 0)];
            out[i] = tau * alpha * v;
        }
        Ok(out)
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

    /// Posterior-implied weights `w = (δ Σ)⁻¹ μ_BL`, normalised to sum
    /// to 1. Caches the result on `self.weights`.
    pub fn bl_weights(&mut self, risk_aversion: Option<f64>) -> Result<DVector<f64>> {
        let delta = risk_aversion.unwrap_or(self.risk_aversion);
        if delta <= 0.0 {
            return Err(PortfolioError::InvalidArgument(
                "risk_aversion must be positive".into(),
            ));
        }
        let mu = self.bl_returns()?;
        let scaled_cov = &self.cov * delta;
        let inv = scaled_cov
            .try_inverse()
            .ok_or_else(|| PortfolioError::Singular("δΣ is singular".into()))?;
        let raw = inv * mu;
        let total: f64 = raw.iter().sum();
        if total.abs() < 1e-12 {
            return Err(PortfolioError::OptimisationFailed(
                "BL implied weights sum to zero".into(),
            ));
        }
        let normalised = raw / total;
        self.weights = Some(normalised.clone());
        Ok(normalised)
    }

    /// `(annualised return, annualised vol, Sharpe)` using the cached
    /// posterior weights. Call [`Self::bl_weights`] first.
    pub fn portfolio_performance(&self, risk_free_rate: f64) -> Result<(f64, f64, f64)> {
        let w = self.weights.as_ref().ok_or_else(|| {
            PortfolioError::InvalidArgument(
                "no weights yet — call bl_weights before portfolio_performance".into(),
            )
        })?;
        let mu = self.bl_returns()?;
        let ret = mu.dot(w);
        let var = (w.transpose() * &self.cov * w)[(0, 0)];
        let vol = var.max(0.0).sqrt();
        let sharpe = if vol > 0.0 { (ret - risk_free_rate) / vol } else { 0.0 };
        Ok((ret, vol, sharpe))
    }

    /// Round weights below `cutoff` to zero, renormalise, optionally
    /// round to `rounding` decimal places. Mirrors PyPortfolioOpt's
    /// `clean_weights`.
    pub fn clean_weights(&self, cutoff: f64, rounding: Option<u32>) -> Result<DVector<f64>> {
        let w = self.weights.as_ref().ok_or_else(|| {
            PortfolioError::InvalidArgument(
                "no weights yet — call bl_weights before clean_weights".into(),
            )
        })?;
        Ok(crate::prelude::clean_weights(w, cutoff, rounding))
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

    #[test]
    fn bl_weights_normalised_and_finite() {
        let cov = cov_3();
        let pi = DVector::from_vec(vec![0.05, 0.07, 0.12]);
        let row = absolute_view(3, 0).unwrap();
        let p = DMatrix::from_rows(&[row.transpose()]);
        let q = DVector::from_vec(vec![0.20]);
        let mut bl = BlackLittermanModel::new(cov, pi, p, q, None, 0.05).unwrap();
        let w = bl.bl_weights(Some(2.0)).unwrap();
        let total: f64 = w.iter().sum();
        assert_relative_eq!(total, 1.0, max_relative = 1e-9);
        for v in w.iter() {
            assert!(v.is_finite());
        }
        let (ret, vol, _sharpe) = bl.portfolio_performance(0.0).unwrap();
        assert!(ret.is_finite());
        assert!(vol > 0.0);
    }

    #[test]
    fn idzorek_omega_grows_with_low_confidence() {
        let cov = cov_3();
        let row = absolute_view(3, 0).unwrap();
        let p = DMatrix::from_rows(&[row.transpose()]);
        let confidences = DVector::from_vec(vec![0.5]);
        let omega = BlackLittermanModel::idzorek_omega(&confidences, &cov, &p, 0.05).unwrap();
        assert_eq!(omega.len(), 1);
        // alpha = 1 (since 1-0.5)/0.5; omega = tau * alpha * P Σ Pᵀ
        // = 0.05 * 1 * 0.04 = 0.002
        assert_relative_eq!(omega[0], 0.05 * 0.04, max_relative = 1e-12);
        // Lower confidence ⇒ higher uncertainty.
        let low = DVector::from_vec(vec![0.1]);
        let omega_low = BlackLittermanModel::idzorek_omega(&low, &cov, &p, 0.05).unwrap();
        assert!(omega_low[0] > omega[0]);
        // Zero confidence sentinel.
        let zero = DVector::from_vec(vec![0.0]);
        let omega_zero = BlackLittermanModel::idzorek_omega(&zero, &cov, &p, 0.05).unwrap();
        assert!(omega_zero[0] >= 1e6);
    }

    #[test]
    fn risk_aversion_from_prices_matches_formula() {
        // Build a deterministic price series with known mean/var.
        let mut prices = DMatrix::<f64>::zeros(252, 1);
        prices[(0, 0)] = 100.0;
        for i in 1..252 {
            // Alternate +1% / -0.5% so mean and variance are nonzero.
            let r = if i % 2 == 0 { 0.01 } else { -0.005 };
            prices[(i, 0)] = prices[(i - 1, 0)] * (1.0 + r);
        }
        let delta = market_implied_risk_aversion_from_prices(&prices, Some(252), 0.0).unwrap();
        assert!(delta.is_finite());
    }
}
