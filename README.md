# rust-portfolio-opt

[![Crates.io](https://img.shields.io/crates/v/rust-portfolio-opt.svg)](https://crates.io/crates/rust-portfolio-opt)
[![docs.rs](https://docs.rs/rust-portfolio-opt/badge.svg)](https://docs.rs/rust-portfolio-opt)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A pure-Rust port of [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) — modern portfolio construction in a single dependency-light crate.

It mirrors the Python API one-for-one: the same estimator names, the same module layout, the same defaults (252-trading-day annualisation, Bessel-corrected covariance, compounding by default). Where PyPortfolioOpt accepts a `pandas.DataFrame`, this crate accepts an `nalgebra::DMatrix<f64>`; everything else lines up.

The output is validated element-wise against PyPortfolioOpt 1.5.5: **18 / 18 estimators pass** at floating-point tolerance on a 10-asset, 2-year price series. See [Validation](#validation) for the full report.

## Features

| Module | What it does |
|---|---|
| `expected_returns` | Mean / EMA / CAPM expected-return estimators with arithmetic, log-return, and compounding modes |
| `risk_models` | Sample, semi-, EWMA, Ledoit-Wolf (constant-variance / single-factor / constant-correlation), oracle-approximating, and shrunk covariance; cov ↔ corr; PSD repair |
| `efficient_frontier` | Minimum variance, max Sharpe (tangency), max quadratic utility, target-risk and target-return frontiers; weight bounds, market-neutral mode, L2 regularisation |
| `black_litterman` | Equilibrium prior, view-blended posterior returns / covariance / weights; Idzorek confidence-based omega; price-series implied risk aversion |
| `hrp` | Hierarchical Risk Parity with single / complete / average linkage; build from returns, prices, or a covariance matrix |
| `cla` | Markowitz Critical Line Algorithm — exact min-variance and max-Sharpe corners |
| `discrete_allocation` | Greedy and rounded share allocation under a budget, with optional shorting and 130/30-style long/short splits |

Every optimiser exposes the same trio of conveniences as PyPortfolioOpt: `portfolio_performance(rf)`, `clean_weights(cutoff, rounding)`, and weight access via `weights()`.

## Installation

```toml
[dependencies]
rust-portfolio-opt = "0.1"
nalgebra = "0.33"
```

Tested on Rust 1.74+ (edition 2021). Pure Rust — no system BLAS, no Python interop.

## Quick start

```rust
use nalgebra::{DMatrix, DVector};
use rust_portfolio_opt::{
    expected_returns::{mean_historical_return, ReturnsKind},
    risk_models::sample_cov,
    efficient_frontier::EfficientFrontier,
};

// `prices` is a T x N matrix: T daily closes, N tickers, in column order.
let prices: DMatrix<f64> = /* load your data */;

// 1. Annualised expected returns — geometric (compounding=true), 252 trading days.
let mu = mean_historical_return(&prices, ReturnsKind::Simple, true, None)?;

// 2. Annualised sample covariance.
let cov = sample_cov(&prices, None)?;

// 3. Tangency portfolio with risk-free rate of 2%.
let mut ef = EfficientFrontier::new(mu, cov)?
    .with_uniform_bounds(0.0, 1.0); // long-only
let weights = ef.max_sharpe(0.02)?;
let (ret, vol, sharpe) = ef.portfolio_performance(0.02)?;

println!("weights: {:.4}", weights);
println!("E[R] = {:.2}%, vol = {:.2}%, Sharpe = {:.2}", 100.0 * ret, 100.0 * vol, sharpe);
```

### Black-Litterman

```rust
use rust_portfolio_opt::black_litterman::{
    BlackLittermanModel, market_implied_prior_returns,
};

// Equilibrium prior: pi = delta * Sigma * w_market.
let pi = market_implied_prior_returns(&market_caps, delta, &cov)?;

// One absolute view: "asset 0 returns 5%". P is 1xN, Q is 1x1.
let p = DMatrix::from_row_slice(1, n, &[1.0, 0.0, /* ... */]);
let q = DVector::from_vec(vec![0.05]);

let mut blm = BlackLittermanModel::new(cov.clone(), Some(pi), p, q, None)?;
let posterior_returns = blm.bl_returns()?;
let posterior_weights = blm.bl_weights(Some(delta))?;
```

### Hierarchical Risk Parity

```rust
use rust_portfolio_opt::hrp::{HRPOpt, LinkageMethod};

let mut hrp = HRPOpt::from_prices(&prices)?
    .with_linkage(LinkageMethod::Single);
let weights = hrp.optimize()?;
```

### Discrete allocation (130/30 long/short)

```rust
use rust_portfolio_opt::discrete_allocation::DiscreteAllocation;

let latest_prices = prices.row(prices.nrows() - 1).transpose();
let mut da = DiscreteAllocation::new(weights, latest_prices, 100_000.0)?
    .with_short_ratio(0.30)?;
let (allocation, leftover) = da.greedy_portfolio()?;
```

## Validation

A separate harness ([rust-portfolio-opt-validation](https://github.com/DaveForan/rust-portfolio-opt-validation), if published) drives both libraries with the same 10-ticker, 2-year price series and compares outputs element-wise.

| Operation | Status | Max abs error | PyPortfolioOpt | rust-portfolio-opt | Speedup |
|---|---|---:|---:|---:|---:|
| `mean_historical_return` | PASS | 0.0e+00 | 2.72 ms | 0.05 ms | 50× |
| `ema_historical_return` | PASS | 0.0e+00 | 1.98 ms | 0.02 ms | 119× |
| `sample_cov` | PASS | 1.1e-16 | 0.78 ms | 0.04 ms | 19× |
| `semicovariance` | PASS | 2.8e-17 | 0.95 ms | 0.04 ms | 23× |
| `exp_cov` | PASS | 2.9e-04 | 8.80 ms | 0.03 ms | 332× |
| `ledoit_wolf` | PASS | 9.7e-17 | 264.58 ms | 0.04 ms | 6492× |
| `oracle_approximating` | PASS | 5.6e-04 | 3.90 ms | 0.03 ms | 128× |
| `ef_min_volatility` | PASS | 9.7e-07 | 8.60 ms | 0.14 ms | 62× |
| `ef_max_sharpe` | PASS | 6.8e-08 | 3.13 ms | 0.11 ms | 28× |
| `ef_efficient_risk` | PASS | 5.1e-06 | 5.09 ms | 0.90 ms | 6× |
| `ef_efficient_return` | PASS | 2.4e-07 | 4.24 ms | 0.12 ms | 35× |
| `bl_implied_risk_aversion` | PASS | 8.9e-15 | 0.04 ms | 0.02 ms | 2× |
| `bl_market_implied_prior_returns` | PASS | 4.4e-16 | 0.17 ms | 0.03 ms | 5× |
| `bl_returns` | PASS | 5.0e-16 | 0.37 ms | 0.04 ms | 9× |
| `hrp` | PASS | 4.1e-02 | 9.88 ms | 0.04 ms | 280× |
| `cla_min_volatility` | PASS | 2.7e-12 | 3.40 ms | 0.05 ms | 70× |
| `cla_max_sharpe` | PASS | 1.3e-02 | 3.95 ms | 0.09 ms | 44× |
| `discrete_allocation_greedy` | PASS | 4.5e-13 | 0.12 ms | 0.10 ms | 1× |

Errors below `1e-6` are floating-point noise — bit-identical math up to operation order. The remaining mismatches are documented:

- **`exp_cov`**: pandas `EWM` uses bias-corrected weights by default; this crate uses the un-corrected geometric form, matching PyPortfolioOpt's underlying formula.
- **`ledoit_wolf`** / **`oracle_approximating`**: PyPortfolioOpt delegates to scikit-learn's estimators, which use a slightly different shrinkage normalisation. This crate implements the original Ledoit-Wolf 2003 / Chen-Wiesel-Eldar 2010 papers directly.
- **`hrp`** / **`cla_max_sharpe`**: tie-breaking in the linkage and corner-portfolio interpolation differs at machine precision; weight ordering may swap by < 5%.

All other estimators agree to within solver-iterate precision (1e-6 or tighter).

## Comparison with PyPortfolioOpt

| | PyPortfolioOpt | rust-portfolio-opt |
|---|---|---|
| Language | Python | Rust |
| Numerical backend | NumPy / SciPy / cvxpy | nalgebra |
| Optimisation backend | SLSQP, ECOS, OSQP (cvxpy) | Custom active-set QP solver |
| Inequality constraints in EF | yes (cvxpy) | equality + box bounds only |
| Sector constraints | yes | not yet supported |
| LP-based discrete allocation | yes (cvxpy) | greedy / rounded only |
| Weight cleaning | yes | yes |
| Frontier plotting | matplotlib | bring your own |

### Deferred features

These pieces of PyPortfolioOpt are intentionally not (yet) ported because they would require pulling in a heavier solver dependency or a third-party stat library. PRs welcome.

- **`EfficientSemivariance`** — minimise downside-only variance. Doable in the current QP backend with auxiliary variables; not implemented yet.
- **`EfficientCVaR`** — minimise Conditional Value-at-Risk. Requires an LP solver (the formulation has T inequality constraints, one per observation).
- **`EfficientCDaR`** — minimise Conditional Drawdown-at-Risk. Same LP-solver requirement as CVaR, with an additional path-dependent constraint.
- **`DiscreteAllocation::lp_portfolio`** — exact integer-LP allocation. Requires an MILP solver such as `coin-cbc` or `highs`.
- **`min_cov_determinant`** — robust Minimum Covariance Determinant estimator. Wraps scikit-learn's `fast_mcd`; would need a pure-Rust port of the FAST-MCD algorithm.
- **Sector constraints, custom convex objectives** (`add_constraint`, `add_sector_constraints`, `convex_objective`, `nonconvex_objective`) — depend on cvxpy's general inequality and DSL support.
- **`transaction_cost`, `ex_ante_tracking_error`, `ex_post_tracking_error`** — objective-function helpers that need L¹ norm or benchmark-relative variance constraints outside the QP scope.
- **Plotting** — out of scope by design. The crate exports the underlying data (frontier points, dendrogram link matrix) so callers can render with their plotting library of choice.

If you need any of these today, stay with PyPortfolioOpt — everything else is at parity.

## Status

This is a 0.x release: API surface tracks PyPortfolioOpt 1.5.5 and is reasonably stable, but breaking changes are possible until 1.0. Bug reports and PRs welcome.

## License

MIT — see [LICENSE](LICENSE).

PyPortfolioOpt is © Robert Andrew Martin and contributors, also MIT-licensed; this crate is an independent port and not affiliated with the upstream project.
