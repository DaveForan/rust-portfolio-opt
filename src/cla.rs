//! Critical Line Algorithm (CLA) for exact mean-variance efficient frontier
//! tracing.
//!
//! Mirrors `pypfopt.cla.CLA`. The algorithm (Markowitz 1956, revisited by
//! López de Prado & Schachermayer 2015) traces every "corner portfolio" on
//! the efficient frontier, i.e. the vertices at which the active-constraint
//! set changes.
//!
//! # Algorithm overview
//!
//! Assets are either **free** (`lb < w < ub`) or **bound** (`w = lb` or
//! `w = ub`). The portfolio on each frontier *segment* satisfies the KKT
//! conditions
//!
//! ```text
//!   (Σ w)_i = λ μ_i + γ    for free i
//!   (Σ w)_i ≤ λ μ_i + γ    for i at lb
//!   (Σ w)_i ≥ λ μ_i + γ    for i at ub
//! ```
//!
//! Within a segment the free weights are an affine function of λ:
//! `w_F(λ) = λ α_F + β_F`.  Corner portfolios occur when
//!
//! * a free asset hits one of its bounds, or
//! * a bound asset's KKT multiplier changes sign.
//!
//! The frontier runs from λ → +∞ (maximum-return portfolio) down to λ = 0
//! (minimum-variance portfolio).

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};

use crate::prelude::{
    assert_square, assert_tickers_match, to_weight_map, LabeledMatrix, LabeledVector,
};
use crate::{PortfolioError, Result};

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// Exact efficient-frontier tracer.
pub struct CLA {
    pub mu: DVector<f64>,
    pub sigma: DMatrix<f64>,
    /// Per-asset lower bounds (default 0).
    pub lb: DVector<f64>,
    /// Per-asset upper bounds (default 1).
    pub ub: DVector<f64>,
    /// Optional ticker labels (one per asset). Required by `_labeled`
    /// companion methods.
    pub tickers: Option<Vec<String>>,
    n: usize,
    /// Corner portfolios ordered from high-λ (high return) to low-λ (min var).
    corner_weights: Vec<DVector<f64>>,
    corner_lambdas: Vec<f64>,
    /// Weights from the most recent objective call.
    weights: Option<DVector<f64>>,
}

impl CLA {
    /// Create a new CLA with default long-only bounds (0 ≤ w ≤ 1).
    pub fn new(mu: DVector<f64>, sigma: DMatrix<f64>) -> Result<Self> {
        let n = assert_square(&sigma, "CLA::new sigma")?;
        if mu.len() != n {
            return Err(PortfolioError::DimensionMismatch(format!(
                "mu length {} ≠ sigma dim {n}",
                mu.len()
            )));
        }
        if n == 0 {
            return Err(PortfolioError::InvalidArgument(
                "need at least one asset".into(),
            ));
        }
        Ok(Self {
            mu,
            sigma,
            lb: DVector::zeros(n),
            ub: DVector::from_element(n, 1.0),
            tickers: None,
            n,
            corner_weights: Vec::new(),
            corner_lambdas: Vec::new(),
            weights: None,
        })
    }

    /// Build a CLA from labeled inputs. Ticker vectors on `mu` and
    /// `sigma` must agree.
    pub fn from_labeled(mu: LabeledVector, sigma: LabeledMatrix) -> Result<Self> {
        assert_tickers_match(&mu.tickers, &sigma.tickers, "CLA::from_labeled")?;
        let tickers = mu.tickers.clone();
        let mut cla = Self::new(mu.values, sigma.values)?;
        cla.tickers = Some(tickers);
        Ok(cla)
    }

    /// Attach ticker labels.
    pub fn with_tickers(mut self, tickers: Vec<String>) -> Result<Self> {
        if tickers.len() != self.n {
            return Err(PortfolioError::DimensionMismatch(format!(
                "with_tickers: {} tickers but {} assets",
                tickers.len(),
                self.n
            )));
        }
        self.tickers = Some(tickers);
        Ok(self)
    }

    fn require_tickers(&self, label: &str) -> Result<&Vec<String>> {
        self.tickers.as_ref().ok_or_else(|| {
            PortfolioError::InvalidArgument(format!(
                "{label}: no tickers attached — call with_tickers or from_labeled first"
            ))
        })
    }

    pub fn with_uniform_bounds(mut self, lb: f64, ub: f64) -> Result<Self> {
        if lb > ub {
            return Err(PortfolioError::InvalidArgument(format!(
                "lb ({lb}) > ub ({ub})"
            )));
        }
        self.lb = DVector::from_element(self.n, lb);
        self.ub = DVector::from_element(self.n, ub);
        Ok(self)
    }

    pub fn weights(&self) -> Option<&DVector<f64>> {
        self.weights.as_ref()
    }

    // -----------------------------------------------------------------------
    // Main entry points
    // -----------------------------------------------------------------------

    /// Minimum-variance portfolio (last corner on the frontier).
    pub fn min_vol(&mut self) -> Result<DVector<f64>> {
        self.ensure_frontier()?;
        let w = self
            .corner_weights
            .last()
            .ok_or_else(|| PortfolioError::OptimisationFailed("no corner portfolios".into()))?
            .clone();
        self.weights = Some(w.clone());
        Ok(w)
    }

    /// Maximum Sharpe-ratio portfolio.
    ///
    /// Scans every corner portfolio and the segments between adjacent
    /// corners to find the exact tangency point.
    pub fn max_sharpe(&mut self, risk_free_rate: f64) -> Result<DVector<f64>> {
        self.ensure_frontier()?;
        let nc = self.corner_weights.len();
        if nc == 0 {
            return Err(PortfolioError::OptimisationFailed(
                "no corner portfolios".into(),
            ));
        }

        let sharpe = |w: &DVector<f64>| {
            let ret = self.mu.dot(w);
            let var = (w.transpose() * &self.sigma * w)[(0, 0)];
            let vol = var.max(0.0).sqrt();
            if vol < 1e-12 {
                f64::NEG_INFINITY
            } else {
                (ret - risk_free_rate) / vol
            }
        };

        // Evaluate all corner portfolios.
        let mut best_sharpe = f64::NEG_INFINITY;
        let mut best_w = self.corner_weights[0].clone();
        for w in &self.corner_weights {
            let s = sharpe(w);
            if s > best_sharpe {
                best_sharpe = s;
                best_w = w.clone();
            }
        }

        // Also check the analytic maximum *within* each segment.  Between
        // two adjacent corners the Sharpe is a quasi-concave function of α
        // (a scalar mixing parameter), so a simple golden-section search
        // over α ∈ [0, 1] suffices.
        for i in 0..nc.saturating_sub(1) {
            let w1 = &self.corner_weights[i];
            let w2 = &self.corner_weights[i + 1];
            let w_opt = golden_section_max_sharpe(w1, w2, &self.mu, &self.sigma, risk_free_rate);
            let s = sharpe(&w_opt);
            if s > best_sharpe {
                best_sharpe = s;
                best_w = w_opt;
            }
        }

        if best_sharpe == f64::NEG_INFINITY {
            return Err(PortfolioError::Infeasible(
                "no portfolio has positive Sharpe ratio".into(),
            ));
        }
        self.weights = Some(best_w.clone());
        Ok(best_w)
    }

    /// Sample `points` (return, vol) pairs spanning the efficient frontier.
    pub fn efficient_frontier(&mut self, points: usize) -> Result<Vec<(f64, f64)>> {
        self.ensure_frontier()?;
        if points < 2 {
            return Err(PortfolioError::InvalidArgument(
                "need at least 2 points".into(),
            ));
        }
        let nc = self.corner_weights.len();
        if nc == 0 {
            return Ok(Vec::new());
        }

        let corner_perf: Vec<(f64, f64)> = self
            .corner_weights
            .iter()
            .map(|w| portfolio_perf(w, &self.mu, &self.sigma))
            .collect();

        // Span of returns across the frontier.
        let ret_min = corner_perf.last().unwrap().0;
        let ret_max = corner_perf.first().unwrap().0;
        if (ret_max - ret_min).abs() < 1e-12 {
            return Ok(vec![corner_perf[0]; points]);
        }

        let mut result = Vec::with_capacity(points);
        for k in 0..points {
            // Walk high → low to match the corner-portfolio ordering.
            let target_ret = ret_max - (ret_max - ret_min) * (k as f64 / (points - 1) as f64);
            let w = self.portfolio_at_return(target_ret, &corner_perf)?;
            let (r, v) = portfolio_perf(&w, &self.mu, &self.sigma);
            result.push((r, v));
        }
        Ok(result)
    }

    /// `(return, vol, Sharpe)` for the most recently solved portfolio.
    pub fn portfolio_performance(&self, risk_free_rate: f64) -> Result<(f64, f64, f64)> {
        let w = self.weights.as_ref().ok_or_else(|| {
            PortfolioError::InvalidArgument("no weights — call min_vol or max_sharpe first".into())
        })?;
        let (ret, vol) = portfolio_perf(w, &self.mu, &self.sigma);
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

    // -----------------------------------------------------------------------
    // Labeled (ticker-keyed) companions
    // -----------------------------------------------------------------------

    /// Ticker-keyed minimum-variance weights.
    pub fn min_vol_labeled(&mut self) -> Result<BTreeMap<String, f64>> {
        let w = self.min_vol()?;
        let tickers = self.require_tickers("min_vol_labeled")?;
        to_weight_map(&w, tickers)
    }

    /// Ticker-keyed tangency weights.
    pub fn max_sharpe_labeled(&mut self, risk_free_rate: f64) -> Result<BTreeMap<String, f64>> {
        let w = self.max_sharpe(risk_free_rate)?;
        let tickers = self.require_tickers("max_sharpe_labeled")?;
        to_weight_map(&w, tickers)
    }

    /// Ticker-keyed cleaned weights.
    pub fn clean_weights_labeled(
        &self,
        cutoff: f64,
        rounding: Option<u32>,
    ) -> Result<BTreeMap<String, f64>> {
        let w = self.clean_weights(cutoff, rounding)?;
        let tickers = self.require_tickers("clean_weights_labeled")?;
        to_weight_map(&w, tickers)
    }

    // -----------------------------------------------------------------------
    // Frontier computation
    // -----------------------------------------------------------------------

    fn ensure_frontier(&mut self) -> Result<()> {
        if self.corner_weights.is_empty() {
            self.compute_frontier()?;
        }
        Ok(())
    }

    fn compute_frontier(&mut self) -> Result<()> {
        // ---- Initialisation ------------------------------------------------
        // Start with the asset that has the highest expected return at its
        // upper bound.  All other assets start at their lower bounds.
        let i_max = argmax(&self.mu);
        let mut free: Vec<usize> = vec![i_max];
        let mut w_bound = self.lb.clone();
        // The free asset absorbs the remaining budget.
        let budget_lower: f64 = self.lb.iter().sum();
        w_bound[i_max] = 0.0; // it is free; don't count it in w_bound
        let w_free_init = (1.0 - budget_lower).clamp(self.lb[i_max], self.ub[i_max]);
        let mut w_current = w_bound.clone();
        w_current[i_max] = w_free_init;

        let mut lambda = f64::INFINITY;

        // Limit iterations to prevent infinite loops.
        for _iter in 0..self.n * self.n + 20 {
            // Solve KKT for current free set.
            let kkt = match self.solve_kkt(&free, &w_current) {
                Ok(k) => k,
                Err(_) => break,
            };

            // Find next transition.
            let (t_lambda, t_event) = self.find_transition(&free, &w_current, &kkt, lambda)?;

            if t_lambda < -1e-10 {
                // No more valid transitions; record current and stop.
                let w_corner = self.evaluate_corner(&free, &w_current, &kkt, 0.0);
                self.push_corner(w_corner, 0.0);
                break;
            }

            let w_corner = self.evaluate_corner(&free, &w_current, &kkt, t_lambda);
            self.push_corner(w_corner.clone(), t_lambda);

            if t_lambda <= 1e-10 {
                break;
            }

            // Update state for next segment.
            match t_event {
                TransitionEvent::FreeHitsLower(k) => {
                    let asset = free[k];
                    w_current[asset] = self.lb[asset];
                    free.remove(k);
                }
                TransitionEvent::FreeHitsUpper(k) => {
                    let asset = free[k];
                    w_current[asset] = self.ub[asset];
                    free.remove(k);
                }
                TransitionEvent::BoundBecomeFree(i) => {
                    free.push(i);
                    // w_current[i] stays at its current bound value until
                    // recalculated from the next KKT.
                }
                TransitionEvent::None => break,
            }

            lambda = t_lambda;
            w_current = w_corner;
        }

        // Always ensure we have at least the min-variance endpoint.
        if self.corner_weights.is_empty() {
            // Fallback: run the unconstrained min-variance via KKT.
            if let Ok(w) = self.min_variance_fallback() {
                let lam = 0.0;
                self.push_corner(w, lam);
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // KKT system: given free set F, solve for w_F(λ) = λ α_F + β_F
    // -----------------------------------------------------------------------

    fn solve_kkt(&self, free: &[usize], w: &DVector<f64>) -> Result<KktSolution> {
        let k = free.len();
        if k == 0 {
            return Ok(KktSolution {
                alpha_f: DVector::zeros(0),
                beta_f: DVector::zeros(0),
                alpha_gamma: 0.0,
                beta_gamma: 0.0,
                free: vec![],
            });
        }

        // Build the (k+1) × (k+1) KKT matrix M:
        //   M = [ Σ_{FF}   1_F ]
        //       [ 1_F'     0   ]
        let mut m = DMatrix::<f64>::zeros(k + 1, k + 1);
        for (a, &i) in free.iter().enumerate() {
            for (b, &j) in free.iter().enumerate() {
                m[(a, b)] = self.sigma[(i, j)];
            }
            m[(a, k)] = 1.0;
            m[(k, a)] = 1.0;
        }
        // Small regularisation for near-singular cases.
        for a in 0..k {
            m[(a, a)] += 1e-12;
        }

        let lu = m.lu();
        if lu.determinant().abs() < 1e-20 {
            return Err(PortfolioError::Singular("KKT matrix is singular".into()));
        }

        // RHS for α: solve M [α_F; α_γ] = [μ_F; 0]
        let mut rhs_alpha = DVector::<f64>::zeros(k + 1);
        for (a, &i) in free.iter().enumerate() {
            rhs_alpha[a] = self.mu[i];
        }
        let sol_a = lu
            .solve(&rhs_alpha)
            .ok_or_else(|| PortfolioError::Singular("KKT α solve failed".into()))?;

        // RHS for β: solve M [β_F; β_γ] = [-Σ_{FB} w_B; 1 - Σ_b w_b]
        let mut rhs_beta = DVector::<f64>::zeros(k + 1);
        let mut sum_wb = 0.0_f64;
        for j in 0..self.n {
            if free.contains(&j) {
                continue;
            }
            sum_wb += w[j];
            for (a, &i) in free.iter().enumerate() {
                rhs_beta[a] -= self.sigma[(i, j)] * w[j];
            }
        }
        rhs_beta[k] = 1.0 - sum_wb;

        let sol_b = lu
            .solve(&rhs_beta)
            .ok_or_else(|| PortfolioError::Singular("KKT β solve failed".into()))?;

        Ok(KktSolution {
            alpha_f: sol_a.rows(0, k).clone_owned(),
            beta_f: sol_b.rows(0, k).clone_owned(),
            alpha_gamma: sol_a[k],
            beta_gamma: sol_b[k],
            free: free.to_vec(),
        })
    }

    // -----------------------------------------------------------------------
    // Evaluate the portfolio at a specific λ on the current segment.
    // -----------------------------------------------------------------------

    fn evaluate_corner(
        &self,
        free: &[usize],
        w_current: &DVector<f64>,
        kkt: &KktSolution,
        lambda: f64,
    ) -> DVector<f64> {
        let mut w = w_current.clone();
        for (k, &i) in free.iter().enumerate() {
            w[i] = if lambda.is_finite() {
                lambda * kkt.alpha_f[k] + kkt.beta_f[k]
            } else {
                kkt.beta_f[k]
            };
            // Clamp to bounds.
            w[i] = w[i].clamp(self.lb[i] - 1e-9, self.ub[i] + 1e-9);
        }
        w
    }

    // -----------------------------------------------------------------------
    // Find the next transition (largest λ < current_lambda where something
    // changes).
    // -----------------------------------------------------------------------

    fn find_transition(
        &self,
        free: &[usize],
        w: &DVector<f64>,
        kkt: &KktSolution,
        current_lambda: f64,
    ) -> Result<(f64, TransitionEvent)> {
        const EPS: f64 = 1e-10;
        let current_finite = if current_lambda.is_infinite() {
            1e15_f64
        } else {
            current_lambda
        };

        let mut best_lambda = -1.0_f64;
        let mut best_event = TransitionEvent::None;

        let try_update = |candidate: f64,
                          ev: TransitionEvent,
                          best_l: &mut f64,
                          best_e: &mut TransitionEvent| {
            if candidate >= 0.0 && candidate < current_finite - EPS && candidate > *best_l + EPS {
                *best_l = candidate;
                *best_e = ev;
            }
        };

        // --- Free assets hitting their bounds ---
        for (k, &i) in free.iter().enumerate() {
            let af = kkt.alpha_f[k];
            let bf = kkt.beta_f[k];
            if af.abs() > EPS {
                // Hits lower bound: af * λ + bf = lb  →  λ = (lb - bf) / af
                let lam_lb = (self.lb[i] - bf) / af;
                if lam_lb > 0.0 {
                    try_update(
                        lam_lb,
                        TransitionEvent::FreeHitsLower(k),
                        &mut best_lambda,
                        &mut best_event,
                    );
                }
                // Hits upper bound: af * λ + bf = ub  →  λ = (ub - bf) / af
                let lam_ub = (self.ub[i] - bf) / af;
                if lam_ub > 0.0 {
                    try_update(
                        lam_ub,
                        TransitionEvent::FreeHitsUpper(k),
                        &mut best_lambda,
                        &mut best_event,
                    );
                }
            }
        }

        // --- Bound assets wanting to join the free set ---
        // The KKT multiplier for bound asset i is:
        //   z_i(λ) = λ μ_i − (Σ w(λ))_i − γ(λ)
        //          = λ (μ_i − a_i − α_γ) − b_i − β_γ
        // where a_i = Σ_{iF} α_F, b_i = Σ_{iF} β_F + Σ_{iB} w_B.
        for i in 0..self.n {
            if free.contains(&i) {
                continue;
            }
            let a_i: f64 = free
                .iter()
                .enumerate()
                .map(|(k, &j)| self.sigma[(i, j)] * kkt.alpha_f[k])
                .sum();
            let b_free: f64 = free
                .iter()
                .enumerate()
                .map(|(k, &j)| self.sigma[(i, j)] * kkt.beta_f[k])
                .sum();
            let b_bound: f64 = (0..self.n)
                .filter(|&j| !free.contains(&j))
                .map(|j| self.sigma[(i, j)] * w[j])
                .sum();
            let b_i = b_free + b_bound;

            let slope = self.mu[i] - a_i - kkt.alpha_gamma;
            let intercept = -(b_i + kkt.beta_gamma);

            if slope.abs() < EPS {
                continue;
            }

            let lam_z = -intercept / slope; // z_i(lam_z) = 0
            if lam_z <= 0.0 {
                continue;
            }

            // Verify the sign flip is the right direction for the constraint.
            // With z_i = λμ_i − (Σw)_i − γ:
            //   at lower bound z_i = −ν_l ≤ 0; asset rejoins when z_i flips
            //     positive as λ decreases.
            //   at upper bound z_i =  ν_u ≥ 0; asset rejoins when z_i flips
            //     negative as λ decreases.
            let z_at_current = slope * current_finite + intercept;
            let is_at_lower = (w[i] - self.lb[i]).abs() < 1e-8;
            let is_at_upper = (w[i] - self.ub[i]).abs() < 1e-8;
            let valid = if is_at_lower {
                z_at_current <= EPS
            } else if is_at_upper {
                z_at_current >= -EPS
            } else {
                false
            };

            if valid {
                try_update(
                    lam_z,
                    TransitionEvent::BoundBecomeFree(i),
                    &mut best_lambda,
                    &mut best_event,
                );
            }
        }

        Ok((best_lambda, best_event))
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn push_corner(&mut self, w: DVector<f64>, lambda: f64) {
        self.corner_weights.push(w);
        self.corner_lambdas.push(lambda);
    }

    fn min_variance_fallback(&self) -> Result<DVector<f64>> {
        // Unconstrained min variance via KKT (no bounds, sum=1):
        // [Σ  1] [w]   [0]
        // [1' 0] [γ] = [1]
        let n = self.n;
        let mut m = DMatrix::<f64>::zeros(n + 1, n + 1);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = self.sigma[(i, j)];
            }
            m[(i, n)] = 1.0;
            m[(n, i)] = 1.0;
        }
        let mut rhs = DVector::<f64>::zeros(n + 1);
        rhs[n] = 1.0;
        let sol = m
            .lu()
            .solve(&rhs)
            .ok_or_else(|| PortfolioError::Singular("min-variance fallback singular".into()))?;
        let mut w = sol.rows(0, n).clone_owned();
        // Project to bounds.
        for i in 0..n {
            w[i] = w[i].clamp(self.lb[i], self.ub[i]);
        }
        // Renormalise.
        let s: f64 = w.iter().sum();
        if s > 1e-12 {
            w /= s;
        }
        Ok(w)
    }

    /// Interpolate the portfolio at a given target return within the
    /// pre-computed corner portfolios.
    fn portfolio_at_return(&self, target_ret: f64, perf: &[(f64, f64)]) -> Result<DVector<f64>> {
        let nc = self.corner_weights.len();
        // Perf is ordered high→low return.
        for i in 0..nc.saturating_sub(1) {
            let (r1, _) = perf[i];
            let (r2, _) = perf[i + 1];
            if target_ret >= r2 - 1e-12 && target_ret <= r1 + 1e-12 {
                let span = r1 - r2;
                let alpha = if span.abs() < 1e-12 {
                    0.5
                } else {
                    (target_ret - r2) / span
                };
                let w =
                    alpha * &self.corner_weights[i] + (1.0 - alpha) * &self.corner_weights[i + 1];
                return Ok(w);
            }
        }
        // Closest corner.
        let best = perf
            .iter()
            .enumerate()
            .min_by(|(_, (r1, _)), (_, (r2, _))| {
                (r1 - target_ret)
                    .abs()
                    .partial_cmp(&(r2 - target_ret).abs())
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(self.corner_weights[best].clone())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Efficient-frontier segment KKT solution.
struct KktSolution {
    alpha_f: DVector<f64>,
    beta_f: DVector<f64>,
    alpha_gamma: f64,
    beta_gamma: f64,
    #[allow(dead_code)]
    free: Vec<usize>,
}

#[derive(Debug, Clone)]
enum TransitionEvent {
    FreeHitsLower(usize),   // index *into* free vec
    FreeHitsUpper(usize),   // index *into* free vec
    BoundBecomeFree(usize), // asset index
    None,
}

fn argmax(v: &DVector<f64>) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn portfolio_perf(w: &DVector<f64>, mu: &DVector<f64>, sigma: &DMatrix<f64>) -> (f64, f64) {
    let ret = mu.dot(w);
    let var = (w.transpose() * sigma * w)[(0, 0)];
    (ret, var.max(0.0).sqrt())
}

/// Golden-section search for the maximum-Sharpe mixing parameter α ∈ [0,1]
/// between two corner portfolios `w1` and `w2`.
fn golden_section_max_sharpe(
    w1: &DVector<f64>,
    w2: &DVector<f64>,
    mu: &DVector<f64>,
    sigma: &DMatrix<f64>,
    rf: f64,
) -> DVector<f64> {
    const GOLDEN: f64 = 0.618_033_988_7;
    const ITERS: usize = 50;
    const TOL: f64 = 1e-8;

    let sharpe = |alpha: f64| {
        let w = alpha * w1 + (1.0 - alpha) * w2;
        let ret = mu.dot(&w);
        let var = (w.transpose() * sigma * &w)[(0, 0)];
        let vol = var.max(0.0).sqrt();
        if vol < 1e-12 {
            f64::NEG_INFINITY
        } else {
            (ret - rf) / vol
        }
    };

    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut x1 = hi - GOLDEN * (hi - lo);
    let mut x2 = lo + GOLDEN * (hi - lo);
    let mut f1 = sharpe(x1);
    let mut f2 = sharpe(x2);

    for _ in 0..ITERS {
        if (hi - lo).abs() < TOL {
            break;
        }
        if f1 < f2 {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + GOLDEN * (hi - lo);
            f2 = sharpe(x2);
        } else {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - GOLDEN * (hi - lo);
            f1 = sharpe(x1);
        }
    }

    let alpha = 0.5 * (lo + hi);
    alpha * w1 + (1.0 - alpha) * w2
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn diag(v: &[f64]) -> DMatrix<f64> {
        let n = v.len();
        let mut m = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = v[i];
        }
        m
    }

    fn two_asset_setup() -> CLA {
        // Two uncorrelated assets: σ₁² = 0.01, σ₂² = 0.04
        // μ = [0.10, 0.15]
        // Min-variance: w = [4/5, 1/5]
        let mu = DVector::from_vec(vec![0.10, 0.15]);
        let cov = diag(&[0.01, 0.04]);
        CLA::new(mu, cov).unwrap()
    }

    #[test]
    fn min_vol_two_uncorrelated() {
        let mut cla = two_asset_setup();
        let w = cla.min_vol().unwrap();
        assert_relative_eq!(w[0], 4.0 / 5.0, epsilon = 1e-2);
        assert_relative_eq!(w[1], 1.0 / 5.0, epsilon = 1e-2);
        let total: f64 = w.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn min_vol_weights_are_long_only() {
        let mut cla = two_asset_setup();
        let w = cla.min_vol().unwrap();
        for v in w.iter() {
            assert!(*v >= -1e-6, "negative weight {v}");
        }
    }

    #[test]
    fn min_vol_weights_sum_to_one() {
        let mu = DVector::from_vec(vec![0.05, 0.10, 0.15]);
        let cov = diag(&[0.01, 0.04, 0.09]);
        let mut cla = CLA::new(mu, cov).unwrap();
        let w = cla.min_vol().unwrap();
        let total: f64 = w.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-2);
    }

    #[test]
    fn max_sharpe_returns_finite_weights() {
        let mut cla = two_asset_setup();
        let w = cla.max_sharpe(0.0).unwrap();
        for v in w.iter() {
            assert!(v.is_finite());
        }
        let total: f64 = w.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-2);
    }

    #[test]
    fn max_sharpe_beats_min_vol_sharpe() {
        let mut cla = two_asset_setup();
        let w_ms = cla.max_sharpe(0.0).unwrap();
        let s_ms = {
            let (r, v) = portfolio_perf(&w_ms, &cla.mu, &cla.sigma);
            r / v
        };
        let mut cla2 = two_asset_setup();
        let w_mv = cla2.min_vol().unwrap();
        let s_mv = {
            let (r, v) = portfolio_perf(&w_mv, &cla2.mu, &cla2.sigma);
            r / v
        };
        assert!(
            s_ms >= s_mv - 1e-4,
            "max_sharpe Sharpe {s_ms} < min_vol Sharpe {s_mv}"
        );
    }

    #[test]
    fn efficient_frontier_is_sorted_by_return() {
        let mu = DVector::from_vec(vec![0.05, 0.10, 0.15]);
        let cov = diag(&[0.01, 0.04, 0.09]);
        let mut cla = CLA::new(mu, cov).unwrap();
        let pts = cla.efficient_frontier(10).unwrap();
        assert_eq!(pts.len(), 10);
        for i in 1..pts.len() {
            assert!(
                pts[i].0 <= pts[i - 1].0 + 1e-8,
                "returns not sorted descending"
            );
        }
    }

    #[test]
    fn portfolio_performance_consistent() {
        let mut cla = two_asset_setup();
        cla.min_vol().unwrap();
        let (ret, vol, sharpe) = cla.portfolio_performance(0.0).unwrap();
        assert!(ret.is_finite() && vol > 0.0 && sharpe.is_finite());
        assert_relative_eq!(sharpe, ret / vol, max_relative = 1e-9);
    }

    #[test]
    fn corner_portfolios_are_non_empty() {
        let mut cla = two_asset_setup();
        cla.ensure_frontier().unwrap();
        assert!(!cla.corner_weights.is_empty());
    }

    #[test]
    fn min_vol_labeled_returns_ticker_keyed_weights() {
        let mu = DVector::from_vec(vec![0.10, 0.15]);
        let cov = diag(&[0.01, 0.04]);
        let lv = LabeledVector::new(mu, vec!["AAPL".into(), "MSFT".into()]).unwrap();
        let lm = LabeledMatrix::new(cov, vec!["AAPL".into(), "MSFT".into()]).unwrap();
        let mut cla = CLA::from_labeled(lv, lm).unwrap();
        let w = cla.min_vol_labeled().unwrap();
        assert!(w.contains_key("AAPL") && w.contains_key("MSFT"));
        let total: f64 = w.values().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-2);
    }

    #[test]
    fn cla_labeled_methods_error_without_tickers() {
        let mut cla = two_asset_setup();
        assert!(cla.min_vol_labeled().is_err());
    }
}
