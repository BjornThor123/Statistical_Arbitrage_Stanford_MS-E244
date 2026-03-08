# SSVI Stat-Arb Pipeline

This repository now includes an SSVI-first implementation of the original project proposal:

1. Fit option-implied volatility surfaces with SSVI.
2. Extract skew (slope of implied volatility vs log-moneyness).
3. Regress stock skew on sector skew to isolate idiosyncratic residual.
4. Trade mean reversion in residual z-scores.
5. Apply transaction costs and aggregate portfolio PnL.

## Files

- `src/strategies/ssvi/model.py`
  - SSVI total-variance parameterization.
  - Pure-NumPy calibration (`calibrate_ssvi_surface`) with constrained random search.
  - Surface methods for total variance, implied volatility, and skew.

- `src/strategies/ssvi/pipeline.py`
  - Loads options history from a zip file by date range.
  - Estimates forwards per `(date, exdate)` via put-call parity linear fit.
  - Cleans panel and computes `(k, t, sigma)`.
  - Fits one SSVI surface per date and returns daily skew series.

- `src/common/skew_signal_backtest.py`
  - Rolling regression of stock skew on sector skew.
  - Residual z-score signal and position logic.
  - Transaction-cost-aware backtest.
  - Portfolio aggregation and performance summary.

- `src/strategies/ssvi/run.py`
  - End-to-end runner for stock + sector option histories (zip or DuckDB).
  - Produces CSV/JSON artifacts under `src/strategies/ssvi/results/current`.

## Run

```bash
.venv/bin/python -m src.strategies.ssvi.run \
  --start-date 2007-01-01 \
  --end-date 2008-12-31
```

## Main Outputs

- `src/strategies/ssvi/results/current/skew_series/*.csv`
- `src/strategies/ssvi/results/current/signals/*.csv`
- `src/strategies/ssvi/results/current/backtests/*.csv`
- `src/strategies/ssvi/results/current/portfolio_backtest.csv`
- `src/strategies/ssvi/results/current/summary.json`

## Tune Strategy

```bash
.venv/bin/python -m src.strategies.ssvi.tune
```

## Generate Report

```bash
.venv/bin/python -m src.strategies.ssvi.report
```

Then build the single dashboard:

```bash
.venv/bin/python -m src.strategies.ssvi.dashboard
```

This creates:

- `src/strategies/ssvi/results/current/report_metrics.json`
- `src/strategies/ssvi/results/current/report.md`
- `src/strategies/ssvi/results/current/report_figures/*.png`
- `src/strategies/ssvi/results/current/dashboard.html`

## Notes

- Calibration uses only `numpy/pandas` (no `scipy` dependency).
- SSVI constraints are enforced via penalties and conservative bounds in calibration.
- Transaction cost model is implemented as:
  - `total_cost = 2 * (half_spread + impact + commission) + hedge_drag`
  - with hedge drag proportional to turnover and position magnitude.
