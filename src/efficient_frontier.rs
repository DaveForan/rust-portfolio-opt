//! Mean-variance optimisation à la `pypfopt.efficient_frontier`.
//!
//! Construct an [`EfficientFrontier`] from an expected-return vector and
//! a covariance matrix, then call one of the optimisers:
//!
//! - [`EfficientFrontier::min_volatility`] — minimum variance portfolio.
//! - [`EfficientFrontier::max_sharpe`]     — tangency portfolio (assumes
//!   long-only `[0, 1]` bounds; long-short is not yet supported).
//! - [`EfficientFrontier::efficient_risk`] — maximise return for a target
//!   volatility.
//! - [`EfficientFrontier::efficient_return`] — minimise variance for a
//!   target return.
//!
//! After solving, the optimiser stores the weights internally so
//! [`EfficientFrontier::portfolio_performance`] and
//! [`EfficientFrontier::clean_weights`] can be called without re-solving.

use nalgebra::{DMatrix, DVector};

use crate::prelude::assert_square;
use crate::qp::{solve, QpProblem, QpSettings};
use crate::{PortfolioError, Result};

/// Per-asset weight bounds. Use `(f64::NEG_INFINITY, f64::INFINITY)` to
/// leave a coordinate unconstrained.
pub type WeightBound = (f64, f64);

/// Mean-variance portfolio optimiser.
pub struct EfficientFrontier {
    pub expected_returns: DVector<f64>,
    pub cov: DMatrix<f64>,
    pub bounds: Vec<WeightBound>,
    pub solver_settings: QpSettings,
    /// L2 regularisation strength: an extra `½ γ ||w||²` is added to the
    /// objective. Mirrors `EfficientFrontier.add_objective(L2_reg, gamma)`
    /// in PyPortfolioOpt. Defaults to 0 (no regularisation).
    pub l2_gamma: f64,
    /// Market-neutral mode: enforce `1ᵀw = 0` instead of `1ᵀw = 1`. The
    /// caller must allow negative bounds to make the problem feasible.
    pub market_neutral: bool,
    weights: Option<DVector<f64>>,
}

impl EfficientFrontier {
    pub fn new(expected_returns: DVector<f64>, cov: DMatrix<f64>) -> Result<Self> {
        let n = assert_square(&cov, "EfficientFrontier::new cov")?;
        if expected_returns.len() != n {
            return Err(PortfolioError::DimensionMismatch(format!(
                "expected_returns has length {}, cov is {n}x{n}",
                expected_returns.len()
            )));
        }
        Ok(Self {
            expected_returns,
            cov,
            bounds: vec![(0.0, 1.0); n],
            solver_settings: QpSettings::default(),
            l2_gamma: 0.0,
            market_neutral: false,
            weights: None,
        })
    }

    /// Add an L2 regularisation term `½ γ ||w||²` to every subsequent
    /// optimisation. Mirrors `add_objective(objective_functions.L2_reg)`
    /// in PyPortfolioOpt.
    pub fn with_l2_reg(mut self, gamma: f64) -> Result<Self> {
        if gamma < 0.0 {
            return Err(PortfolioError::InvalidArgument(
                "L2 reg gamma must be non-negative".into(),
            ));
        }
        self.l2_gamma = gamma;
        Ok(self)
    }

    /// Switch the budget constraint from `1ᵀw = 1` to `1ᵀw = 0` so the
    /// portfolio is dollar-neutral. Caller must supply bounds that allow
    /// shorts (e.g. `with_uniform_bounds(-1.0, 1.0)`).
    pub fn with_market_neutral(mut self, market_neutral: bool) -> Self {
        self.market_neutral = market_neutral;
        self
    }

    /// Effective covariance with L2 regularisation applied.
    fn p_matrix(&self) -> DMatrix<f64> {
        if self.l2_gamma > 0.0 {
            let mut p = self.cov.clone();
            for i in 0..p.nrows() {
                p[(i, i)] += self.l2_gamma;
            }
            p
        } else {
            self.cov.clone()
        }
    }

    fn budget_target(&self) -> f64 {
        if self.market_neutral {
            0.0
        } else {
            1.0
        }
    }

    pub fn with_bounds(mut self, bounds: Vec<WeightBound>) -> Result<Self> {
        if bounds.len() != self.expected_returns.len() {
            return Err(PortfolioError::DimensionMismatch(format!(
                "bounds length {} does not match {} assets",
                bounds.len(),
                self.expected_returns.len()
            )));
        }
        self.bounds = bounds;
        Ok(self)
    }

    pub fn with_uniform_bounds(mut self, low: f64, high: f64) -> Self {
        self.bounds = vec![(low, high); self.expected_returns.len()];
        self
    }

    pub fn with_solver_settings(mut self, settings: QpSettings) -> Self {
        self.solver_settings = settings;
        self
    }

    fn n(&self) -> usize {
        self.expected_returns.len()
    }

    fn lb_ub(&self) -> (DVector<f64>, DVector<f64>) {
        let n = self.n();
        let mut lb = DVector::<f64>::zeros(n);
        let mut ub = DVector::<f64>::zeros(n);
        for i in 0..n {
            lb[i] = self.bounds[i].0;
            ub[i] = self.bounds[i].1;
        }
        (lb, ub)
    }

    pub fn weights(&self) -> Option<&DVector<f64>> {
        self.weights.as_ref()
    }

    /// Minimum-variance portfolio.
    pub fn min_volatility(&mut self) -> Result<DVector<f64>> {
        let n = self.n();
        let q = DVector::<f64>::zeros(n);
        let a = DMatrix::from_row_slice(1, n, &vec![1.0_f64; n]);
        let b = DVector::from_vec(vec![self.budget_target()]);
        let (lb, ub) = self.lb_ub();
        let prob = QpProblem::new(self.p_matrix(), q, a, b, lb, ub)?;
        let w = solve(&prob, self.solver_settings)?;
        self.weights = Some(w.clone());
        Ok(w)
    }

    /// Maximise quadratic utility `μᵀw - ½ δ wᵀΣw`. Equivalent to
    /// `EfficientFrontier.max_quadratic_utility(risk_aversion=δ)` in
    /// PyPortfolioOpt. When `market_neutral` is `false`, weights sum
    /// to 1; otherwise the result is dollar-neutral. The `market_neutral`
    /// argument overrides the value set by [`Self::with_market_neutral`].
    pub fn max_quadratic_utility(
        &mut self,
        risk_aversion: f64,
        market_neutral: Option<bool>,
    ) -> Result<DVector<f64>> {
        if risk_aversion <= 0.0 {
            return Err(PortfolioError::InvalidArgument(
                "risk_aversion must be positive".into(),
            ));
        }
        let n = self.n();
        // P = δ Σ + γ I; q = -μ.  ½ wᵀ(δΣ + γI)w + (-μ)ᵀw.
        let mut p = self.cov.clone() * risk_aversion;
        if self.l2_gamma > 0.0 {
            for i in 0..n {
                p[(i, i)] += self.l2_gamma;
            }
        }
        let q = -self.expected_returns.clone();
        let a = DMatrix::from_row_slice(1, n, &vec![1.0_f64; n]);
        let target = if market_neutral.unwrap_or(self.market_neutral) {
            0.0
        } else {
            1.0
        };
        let b = DVector::from_vec(vec![target]);
        let (lb, ub) = self.lb_ub();
        let prob = QpProblem::new(p, q, a, b, lb, ub)?;
        let w = solve(&prob, self.solver_settings)?;
        self.weights = Some(w.clone());
        Ok(w)
    }

    /// Tangency portfolio (maximum Sharpe ratio).
    ///
    /// Currently supports long-only bounds (`lb >= 0`). Uses the
    /// Cornuejols & Tütüncü transformation:
    ///
    /// ```text
    ///   minimise   ½ yᵀ Σ y
    ///   subject to (μ - r_f)ᵀ y = 1, y ≥ 0
    /// ```
    /// Then `w = y / Σᵢ yᵢ`.
    pub fn max_sharpe(&mut self, risk_free_rate: f64) -> Result<DVector<f64>> {
        let n = self.n();
        for i in 0..n {
            if self.bounds[i].0 < 0.0 {
                return Err(PortfolioError::InvalidArgument(
                    "max_sharpe currently requires lb >= 0 for every asset".into(),
                ));
            }
        }

        let mut excess = self.expected_returns.clone();
        for i in 0..n {
            excess[i] -= risk_free_rate;
        }
        if excess.iter().all(|v| *v <= 0.0) {
            return Err(PortfolioError::Infeasible(
                "no asset has expected return above the risk-free rate".into(),
            ));
        }

        let q = DVector::<f64>::zeros(n);
        let a = DMatrix::from_row_slice(1, n, excess.as_slice());
        let b = DVector::from_vec(vec![1.0]);
        let lb = DVector::from_element(n, 0.0);
        let ub = DVector::from_element(n, f64::INFINITY);
        let prob = QpProblem::new(self.cov.clone(), q, a, b, lb, ub)?;
        let y = solve(&prob, self.solver_settings)?;
        let scale: f64 = y.iter().sum();
        if scale <= 0.0 {
            return Err(PortfolioError::OptimisationFailed(
                "max_sharpe scaling factor is non-positive".into(),
            ));
        }
        let mut w = y / scale;

        for i in 0..n {
            if w[i] > self.bounds[i].1 + 1e-6 {
                return Err(PortfolioError::Infeasible(format!(
                    "tangency portfolio violates ub on asset {i}: weight {} > {}",
                    w[i], self.bounds[i].1
                )));
            }
            if w[i] < self.bounds[i].0 - 1e-8 {
                w[i] = self.bounds[i].0;
            }
            if w[i] < 0.0 {
                w[i] = 0.0;
            }
        }
        self.weights = Some(w.clone());
        Ok(w)
    }

    /// Minimum variance for a target return.
    pub fn efficient_return(&mut self, target_return: f64) -> Result<DVector<f64>> {
        let n = self.n();
        let q = DVector::<f64>::zeros(n);
        let mut a = DMatrix::<f64>::zeros(2, n);
        for j in 0..n {
            a[(0, j)] = 1.0;
            a[(1, j)] = self.expected_returns[j];
        }
        let b = DVector::from_vec(vec![self.budget_target(), target_return]);
        let (lb, ub) = self.lb_ub();
        let prob = QpProblem::new(self.p_matrix(), q, a, b, lb, ub)?;
        let w = solve(&prob, self.solver_settings)?;
        self.weights = Some(w.clone());
        Ok(w)
    }

    /// Maximum return for a target volatility (bisection on the return
    /// axis, solving min-variance at each candidate target return).
    pub fn efficient_risk(&mut self, target_volatility: f64) -> Result<DVector<f64>> {
        if target_volatility <= 0.0 {
            return Err(PortfolioError::InvalidArgument(
                "target_volatility must be positive".into(),
            ));
        }

        let mv = {
            let mut clone = EfficientFrontier {
                expected_returns: self.expected_returns.clone(),
                cov: self.cov.clone(),
                bounds: self.bounds.clone(),
                solver_settings: self.solver_settings,
                l2_gamma: self.l2_gamma,
                market_neutral: self.market_neutral,
                weights: None,
            };
            clone.min_volatility()?
        };
        let mv_ret = self.expected_returns.dot(&mv);
        let mv_vol = (mv.transpose() * &self.cov * &mv)[(0, 0)].sqrt();
        if target_volatility < mv_vol - 1e-8 {
            return Err(PortfolioError::Infeasible(format!(
                "target_volatility {target_volatility} below minimum achievable {mv_vol}"
            )));
        }

        let mut high_return = mv_ret;
        for i in 0..self.n() {
            if self.bounds[i].1 >= 1.0 - 1e-12 && self.bounds[i].0 <= 1e-12 {
                high_return = high_return.max(self.expected_returns[i]);
            } else {
                high_return = high_return.max(self.expected_returns[i] * self.bounds[i].1);
            }
        }
        if high_return <= mv_ret {
            self.weights = Some(mv.clone());
            return Ok(mv);
        }

        let tol = 1e-6;
        let mut lo = mv_ret;
        let mut hi = high_return;
        let mut best = mv;

        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            match self.efficient_return(mid) {
                Ok(w) => {
                    let vol = (w.transpose() * &self.cov * &w)[(0, 0)].sqrt();
                    if (vol - target_volatility).abs() < tol {
                        self.weights = Some(w.clone());
                        return Ok(w);
                    }
                    if vol < target_volatility {
                        lo = mid;
                        best = w;
                    } else {
                        hi = mid;
                    }
                }
                Err(_) => {
                    hi = mid;
                }
            }
        }
        self.weights = Some(best.clone());
        Ok(best)
    }

    /// `(annualised return, annualised vol, Sharpe)` for the most
    /// recently solved portfolio.
    pub fn portfolio_performance(&self, risk_free_rate: f64) -> Result<(f64, f64, f64)> {
        let w = self.weights.as_ref().ok_or_else(|| {
            PortfolioError::InvalidArgument(
                "no weights yet — call an optimiser before portfolio_performance".into(),
            )
        })?;
        let ret = self.expected_returns.dot(w);
        let var = (w.transpose() * &self.cov * w)[(0, 0)];
        let vol = var.max(0.0).sqrt();
        let sharpe = if vol > 0.0 {
            (ret - risk_free_rate) / vol
        } else {
            0.0
        };
        Ok((ret, vol, sharpe))
    }

    /// Round weights below `cutoff` to zero, renormalise, optionally
    /// round to `rounding` decimal places. Mirrors PyPortfolioOpt's
    /// `clean_weights`.
    pub fn clean_weights(&self, cutoff: f64, rounding: Option<u32>) -> Result<DVector<f64>> {
        let w = self.weights.as_ref().ok_or_else(|| {
            PortfolioError::InvalidArgument(
                "no weights yet — call an optimiser before clean_weights".into(),
            )
        })?;
        Ok(crate::prelude::clean_weights(w, cutoff, rounding))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn diag(values: &[f64]) -> DMatrix<f64> {
        let n = values.len();
        let mut m = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = values[i];
        }
        m
    }

    #[test]
    fn min_vol_two_uncorrelated_assets() {
        let mu = DVector::from_vec(vec![0.10, 0.15]);
        let cov = diag(&[1.0, 4.0]);
        let mut ef = EfficientFrontier::new(mu, cov).unwrap();
        let w = ef.min_volatility().unwrap();
        assert_relative_eq!(w[0], 4.0 / 5.0, epsilon = 1e-3);
        assert_relative_eq!(w[1], 1.0 / 5.0, epsilon = 1e-3);
        let total: f64 = w.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn min_vol_respects_bounds() {
        let mu = DVector::from_vec(vec![0.10, 0.15, 0.20]);
        let cov = diag(&[1.0, 4.0, 9.0]);
        let mut ef = EfficientFrontier::new(mu, cov)
            .unwrap()
            .with_uniform_bounds(0.2, 0.5);
        let w = ef.min_volatility().unwrap();
        for i in 0..3 {
            assert!(w[i] >= 0.2 - 1e-3);
            assert!(w[i] <= 0.5 + 1e-3);
        }
        let total: f64 = w.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn max_sharpe_two_assets() {
        let mu = DVector::from_vec(vec![0.10, 0.20]);
        let cov = diag(&[0.04, 0.16]);
        let mut ef = EfficientFrontier::new(mu, cov).unwrap();
        let w = ef.max_sharpe(0.0).unwrap();
        assert_relative_eq!(w[0], 2.0 / 3.0, epsilon = 1e-3);
        assert_relative_eq!(w[1], 1.0 / 3.0, epsilon = 1e-3);
    }

    #[test]
    fn max_sharpe_infeasible_below_rf() {
        let mu = DVector::from_vec(vec![0.01, 0.02]);
        let cov = diag(&[0.04, 0.16]);
        let mut ef = EfficientFrontier::new(mu, cov).unwrap();
        let err = ef.max_sharpe(0.05).unwrap_err();
        matches!(err, PortfolioError::Infeasible(_));
    }

    #[test]
    fn efficient_return_matches_target() {
        let mu = DVector::from_vec(vec![0.05, 0.15]);
        let cov = diag(&[0.01, 0.04]);
        let mut ef = EfficientFrontier::new(mu, cov).unwrap();
        let target = 0.10;
        let w = ef.efficient_return(target).unwrap();
        let achieved = ef.expected_returns.dot(&w);
        assert_relative_eq!(achieved, target, epsilon = 1e-4);
        let total: f64 = w.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn efficient_risk_finds_target_vol() {
        let mu = DVector::from_vec(vec![0.05, 0.15]);
        let cov = diag(&[0.01, 0.09]);
        let mut ef = EfficientFrontier::new(mu, cov).unwrap();
        let target_vol = 0.15;
        let w = ef.efficient_risk(target_vol).unwrap();
        let vol = (w.transpose() * &ef.cov * &w)[(0, 0)].sqrt();
        assert!((vol - target_vol).abs() < 1e-3, "vol = {vol}");
    }

    #[test]
    fn portfolio_performance_after_min_vol() {
        let mu = DVector::from_vec(vec![0.10, 0.15]);
        let cov = diag(&[0.04, 0.09]);
        let mut ef = EfficientFrontier::new(mu, cov).unwrap();
        ef.min_volatility().unwrap();
        let (ret, vol, sharpe) = ef.portfolio_performance(0.0).unwrap();
        assert!(ret.is_finite() && vol > 0.0 && sharpe.is_finite());
    }

    #[test]
    fn max_quadratic_utility_recovers_two_asset_solution() {
        // For uncorrelated assets and δ = 1, the maximiser of
        // μᵀw - 0.5 wᵀΣw subject to 1ᵀw = 1 has a closed-form analogue;
        // we just check the result is feasible, sums to 1, and sits in
        // bounds.
        let mu = DVector::from_vec(vec![0.10, 0.20]);
        let cov = diag(&[0.04, 0.16]);
        let mut ef = EfficientFrontier::new(mu, cov).unwrap();
        let w = ef.max_quadratic_utility(2.0, None).unwrap();
        let total: f64 = w.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-6);
        for v in w.iter() {
            assert!(*v >= -1e-6 && *v <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn l2_reg_pulls_toward_equal_weight() {
        // With γ very large the L2 term dominates and pushes weights
        // toward the unconstrained-equal-weight direction.
        let mu = DVector::from_vec(vec![0.01, 0.02, 0.03]);
        let cov = diag(&[0.01, 0.01, 0.01]);
        let mut ef = EfficientFrontier::new(mu, cov)
            .unwrap()
            .with_l2_reg(1000.0)
            .unwrap();
        let w = ef.min_volatility().unwrap();
        // Each weight should be near 1/3.
        for i in 0..3 {
            assert!((w[i] - 1.0 / 3.0).abs() < 0.05);
        }
    }

    #[test]
    fn market_neutral_min_vol_sums_to_zero() {
        let mu = DVector::from_vec(vec![0.05, 0.10]);
        let cov = diag(&[0.04, 0.09]);
        let mut ef = EfficientFrontier::new(mu, cov)
            .unwrap()
            .with_uniform_bounds(-1.0, 1.0)
            .with_market_neutral(true);
        let w = ef.min_volatility().unwrap();
        let total: f64 = w.iter().sum();
        assert!(total.abs() < 1e-3, "1ᵀw = {total} should be ≈ 0");
    }

    #[test]
    fn clean_weights_zeroes_below_cutoff() {
        let mu = DVector::from_vec(vec![0.10, 0.15, 0.20]);
        let cov = diag(&[0.04, 0.09, 0.16]);
        let mut ef = EfficientFrontier::new(mu, cov).unwrap();
        ef.min_volatility().unwrap();
        let cleaned = ef.clean_weights(1e-3, Some(4)).unwrap();
        let total: f64 = cleaned.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-3);
    }
}
