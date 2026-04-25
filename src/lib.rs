//! Pure-Rust port of [PyPortfolioOpt][pyo].
//!
//! The crate mirrors PyPortfolioOpt's module structure:
//!
//! - [`expected_returns`] — historical mean / EMA / CAPM return estimators.
//! - [`risk_models`] — sample / EWMA / semi-covariance + Ledoit-Wolf and
//!   oracle-approximating shrinkage; cov ↔ corr helpers.
//! - [`efficient_frontier`] — mean-variance optimisation: minimum variance,
//!   tangency (max Sharpe), efficient risk / return, with weight bounds.
//! - [`black_litterman`] — equilibrium prior + view-blended posterior.
//! - [`hrp`] — hierarchical risk parity (correlation distance, single
//!   linkage, recursive bisection).
//! - [`cla`] — Markowitz's Critical Line Algorithm.
//! - [`discrete_allocation`] — convert continuous weights to integer share
//!   counts under a budget.
//!
//! All matrix / vector inputs and outputs use [`nalgebra`]'s `DMatrix` /
//! `DVector` so they slot into broader nalgebra-based pipelines.
//!
//! Each estimator and optimiser also has a `_labeled` companion that
//! accepts ticker names alongside the prices and returns
//! `BTreeMap<String, f64>` (ordered by ticker), mirroring how
//! PyPortfolioOpt accepts a `pandas.DataFrame` and returns an
//! `OrderedDict`. See [`LabeledVector`] / [`LabeledMatrix`] for the
//! intermediate types.
//!
//! Every estimator and optimiser keeps the same default annualisation
//! convention as PyPortfolioOpt: 252 trading days unless explicitly
//! overridden by the caller.
//!
//! [pyo]: https://github.com/PyPortfolio/PyPortfolioOpt

pub mod black_litterman;
pub mod cla;
pub mod discrete_allocation;
pub mod efficient_frontier;
pub mod expected_returns;
pub mod hrp;
pub mod risk_models;

mod prelude;
mod qp;

pub use prelude::*;

/// Crate-wide error type. Most fallible operations return [`PortfolioError`]
/// rather than panicking so callers can decide how to recover from
/// misconfigured inputs (mismatched shapes, infeasible optimisation, etc.).
#[derive(Debug, thiserror::Error)]
pub enum PortfolioError {
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("matrix is not positive (semi-)definite: {0}")]
    NotPositiveDefinite(String),

    #[error("optimisation failed: {0}")]
    OptimisationFailed(String),

    #[error("infeasible problem: {0}")]
    Infeasible(String),

    #[error("singular system: {0}")]
    Singular(String),
}

/// Convenience alias used by every module in the crate.
pub type Result<T> = std::result::Result<T, PortfolioError>;

/// Default annualisation factor — 252 trading days per year, matching
/// PyPortfolioOpt's defaults.
pub const TRADING_DAYS_PER_YEAR: usize = 252;
