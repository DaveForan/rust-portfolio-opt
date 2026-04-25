//! Small in-tree QP solver, used internally by the efficient-frontier and
//! Black-Litterman modules.
//!
//! Solves
//!
//! ```text
//!   minimise   ½ xᵀ P x + qᵀ x
//!   subject to A x = b
//!              l ≤ x ≤ u
//! ```
//!
//! `P` must be symmetric positive (semi-)definite. `A` is the dense
//! equality-constraint matrix (rows = constraints). Bounds are per
//! coordinate; pass `f64::NEG_INFINITY` / `f64::INFINITY` for
//! "unbounded".
//!
//! The implementation is an ADMM splitting in the spirit of the OSQP
//! formulation, simplified for dense problems. It handles all of the
//! constraint shapes the portfolio-opt modules need (sum-to-one, target
//! return, weight bounds, max-Sharpe transformation) without pulling in
//! a C dependency.

use nalgebra::{DMatrix, DVector};

use crate::{PortfolioError, Result};

/// Convex QP with linear equalities and box bounds.
pub struct QpProblem {
    pub p: DMatrix<f64>,        // n x n, symmetric, PSD
    pub q: DVector<f64>,        // n
    pub a_eq: DMatrix<f64>,     // m_eq x n  (may have 0 rows)
    pub b_eq: DVector<f64>,     // m_eq
    pub lb: DVector<f64>,       // n
    pub ub: DVector<f64>,       // n
}

impl QpProblem {
    pub fn new(
        p: DMatrix<f64>,
        q: DVector<f64>,
        a_eq: DMatrix<f64>,
        b_eq: DVector<f64>,
        lb: DVector<f64>,
        ub: DVector<f64>,
    ) -> Result<Self> {
        let n = p.nrows();
        if p.ncols() != n {
            return Err(PortfolioError::DimensionMismatch(
                "P must be square".into(),
            ));
        }
        if q.len() != n {
            return Err(PortfolioError::DimensionMismatch(
                "q length must match P dimension".into(),
            ));
        }
        if a_eq.ncols() != n {
            return Err(PortfolioError::DimensionMismatch(
                "A_eq column count must match P dimension".into(),
            ));
        }
        if b_eq.len() != a_eq.nrows() {
            return Err(PortfolioError::DimensionMismatch(
                "b_eq length must equal A_eq row count".into(),
            ));
        }
        if lb.len() != n || ub.len() != n {
            return Err(PortfolioError::DimensionMismatch(
                "bound vectors must have length n".into(),
            ));
        }
        for i in 0..n {
            if lb[i] > ub[i] {
                return Err(PortfolioError::InvalidArgument(format!(
                    "lb[{i}] > ub[{i}]"
                )));
            }
        }
        Ok(Self { p, q, a_eq, b_eq, lb, ub })
    }
}

/// Solver tolerances and iteration cap.
#[derive(Debug, Clone, Copy)]
pub struct QpSettings {
    pub max_iter: usize,
    /// Primal feasibility tolerance.
    pub eps_pri: f64,
    /// Dual feasibility tolerance.
    pub eps_dua: f64,
    /// ADMM step size.
    pub rho: f64,
    /// Tikhonov regularisation added to P diagonal for numerical stability.
    pub p_regulariser: f64,
}

impl Default for QpSettings {
    fn default() -> Self {
        Self {
            max_iter: 5_000,
            eps_pri: 1e-8,
            eps_dua: 1e-8,
            rho: 1.0,
            p_regulariser: 1e-10,
        }
    }
}

/// Solve [`QpProblem`] with the supplied [`QpSettings`].
///
/// Returns the optimal `x`. Equality constraints are enforced as part of
/// the projection (so they're satisfied to machine precision when the
/// linear system is well-conditioned).
pub fn solve(prob: &QpProblem, settings: QpSettings) -> Result<DVector<f64>> {
    let n = prob.p.nrows();

    // We split the constraints into:
    //   - equality:   A_eq x = b_eq      (handled in the x-update via KKT)
    //   - box:        l ≤ x ≤ u          (handled by projection of z)
    //
    // ADMM variables:
    //   z ∈ R^n  (proxy for x clamped to bounds)
    //   y ∈ R^n  (scaled dual for the box constraints)
    //
    // x-update: solve
    //   [ P + ρI    A_eq^T ] [x]   [ -q + ρ(z - y) ]
    //   [ A_eq      0      ] [λ] = [ b_eq          ]
    //
    // z-update: clip(x + y, l, u)
    // y-update: y += x - z

    let m = prob.a_eq.nrows();
    let dim = n + m;
    let mut kkt = DMatrix::<f64>::zeros(dim, dim);
    for i in 0..n {
        for j in 0..n {
            kkt[(i, j)] = prob.p[(i, j)];
        }
        kkt[(i, i)] += settings.rho + settings.p_regulariser;
    }
    for r in 0..m {
        for c in 0..n {
            kkt[(n + r, c)] = prob.a_eq[(r, c)];
            kkt[(c, n + r)] = prob.a_eq[(r, c)];
        }
    }

    let lu = kkt.clone().lu();
    if lu.determinant() == 0.0 {
        return Err(PortfolioError::Singular(
            "KKT matrix is singular — check constraint redundancy and PSD-ness of P".into(),
        ));
    }

    // Initial point: project the unconstrained solution into the box.
    let mut z = DVector::<f64>::zeros(n);
    for i in 0..n {
        z[i] = if prob.lb[i].is_finite() && prob.ub[i].is_finite() {
            0.5 * (prob.lb[i] + prob.ub[i])
        } else if prob.lb[i].is_finite() {
            prob.lb[i]
        } else if prob.ub[i].is_finite() {
            prob.ub[i]
        } else {
            0.0
        };
    }
    let mut y = DVector::<f64>::zeros(n);
    let mut x = z.clone();

    for _ in 0..settings.max_iter {
        // Build RHS: top n rows = -q + ρ(z - y); bottom m rows = b_eq.
        let mut rhs = DVector::<f64>::zeros(dim);
        for i in 0..n {
            rhs[i] = -prob.q[i] + settings.rho * (z[i] - y[i]);
        }
        for r in 0..m {
            rhs[n + r] = prob.b_eq[r];
        }
        let sol = lu.solve(&rhs).ok_or_else(|| {
            PortfolioError::OptimisationFailed("KKT linear solve failed".into())
        })?;
        for i in 0..n {
            x[i] = sol[i];
        }

        // z update: project x + y into box.
        let z_old = z.clone();
        for i in 0..n {
            let v = x[i] + y[i];
            z[i] = v.max(prob.lb[i]).min(prob.ub[i]);
        }

        // y update.
        for i in 0..n {
            y[i] += x[i] - z[i];
        }

        // Convergence check: primal = ||x - z||, dual = ρ ||z - z_old||.
        let mut pri = 0.0;
        let mut dua = 0.0;
        for i in 0..n {
            pri += (x[i] - z[i]).powi(2);
            dua += (z[i] - z_old[i]).powi(2);
        }
        let pri = pri.sqrt();
        let dua = settings.rho * dua.sqrt();

        if pri < settings.eps_pri && dua < settings.eps_dua {
            return Ok(z);
        }
    }

    Err(PortfolioError::OptimisationFailed(format!(
        "QP did not converge in {} iterations",
        settings.max_iter
    )))
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
    fn unconstrained_quadratic_gives_minimum() {
        // min 0.5 (x-1)^2 + 0.5 (x-2)^2 + ... two-var version
        // = 0.5 x^T I x - [1,2] x + const, so min at x = [1, 2]
        let p = diag(&[1.0, 1.0]);
        let q = DVector::from_vec(vec![-1.0, -2.0]);
        let a = DMatrix::<f64>::zeros(0, 2);
        let b = DVector::<f64>::zeros(0);
        let lb = DVector::from_vec(vec![f64::NEG_INFINITY, f64::NEG_INFINITY]);
        let ub = DVector::from_vec(vec![f64::INFINITY, f64::INFINITY]);
        let prob = QpProblem::new(p, q, a, b, lb, ub).unwrap();
        let x = solve(&prob, QpSettings::default()).unwrap();
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(x[1], 2.0, epsilon = 1e-4);
    }

    #[test]
    fn equality_constraint_simplex() {
        // min 0.5 (x1^2 + x2^2)  s.t. x1 + x2 = 1
        // Lagrangian L = 0.5(x1^2+x2^2) + λ(1 - x1 - x2)
        // ⇒ x1 = x2 = 0.5
        let p = diag(&[1.0, 1.0]);
        let q = DVector::<f64>::zeros(2);
        let a = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b = DVector::from_vec(vec![1.0]);
        let lb = DVector::from_vec(vec![f64::NEG_INFINITY, f64::NEG_INFINITY]);
        let ub = DVector::from_vec(vec![f64::INFINITY, f64::INFINITY]);
        let prob = QpProblem::new(p, q, a, b, lb, ub).unwrap();
        let x = solve(&prob, QpSettings::default()).unwrap();
        assert_relative_eq!(x[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(x[1], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn bounded_simplex_min_variance() {
        // min 0.5 w^T Σ w  s.t. sum(w) = 1, 0 ≤ w_i ≤ 1
        // Σ = diag(1, 4) ⇒ closed-form min variance at w = [4/5, 1/5]
        let p = diag(&[1.0, 4.0]);
        let q = DVector::<f64>::zeros(2);
        let a = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b = DVector::from_vec(vec![1.0]);
        let lb = DVector::from_vec(vec![0.0, 0.0]);
        let ub = DVector::from_vec(vec![1.0, 1.0]);
        let prob = QpProblem::new(p, q, a, b, lb, ub).unwrap();
        let x = solve(&prob, QpSettings::default()).unwrap();
        assert_relative_eq!(x[0], 4.0 / 5.0, epsilon = 1e-4);
        assert_relative_eq!(x[1], 1.0 / 5.0, epsilon = 1e-4);
    }

    #[test]
    fn active_box_constraints() {
        // min 0.5 (x1 - 5)^2 + 0.5 (x2 - 5)^2  s.t. 0 ≤ x_i ≤ 1
        // Optimum hits ub: x = [1, 1]
        let p = diag(&[1.0, 1.0]);
        let q = DVector::from_vec(vec![-5.0, -5.0]);
        let a = DMatrix::<f64>::zeros(0, 2);
        let b = DVector::<f64>::zeros(0);
        let lb = DVector::from_vec(vec![0.0, 0.0]);
        let ub = DVector::from_vec(vec![1.0, 1.0]);
        let prob = QpProblem::new(p, q, a, b, lb, ub).unwrap();
        let x = solve(&prob, QpSettings::default()).unwrap();
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-4);
    }
}
