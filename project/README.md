# Statistical Arbitrage — IV Skew Pairs Trading

## Setup

Python 3.10+ required.

Install dependencies:

```bash
uv pip install -r requirements.txt 
```

Install the project package (from the `project/` directory):

```bash
pip install -e .
```

## Data

Place your raw data files in the `data/` directory with this structure:

```
data/
  equities/
    data/         # CRSP daily CSV files (*.csv.gz)
    metadata/     # crsp_data_dictionary.csv
  options/
    data/         # OptionMetrics files (*.zip or *.csv)
    metadata/     # options-data-dictionary.csv
  risk_free/
    data/         # yield_panel_daily_frequency_monthly_maturity.csv
```

## Step 1 — Build the DuckDB database

The `DataLoader` reads all raw CSVs, applies type casts from metadata dictionaries, and loads everything into a single DuckDB file (`data/market_data.duckdb`).

Run from the repository root (one level above `project/`):

```bash
python -m src.data_loader
```

This builds five tables in order:

1. `options` — raw options data with a `source_label` column per ticker
2. `equities` — CRSP daily equity data with a `source_label` column per sector
3. `risk_free` — wide-format yield panel
4. `rf_long` — unpivoted (date, maturity_months, rate)
5. `options_enriched` — options joined with spot prices and risk-free rates, plus computed fields (strike, mid_price, tte, log_moneyness, forward_price, etc.)

If a table already exists it is skipped. To rebuild, delete `data/market_data.duckdb` and run again.


## Step 2 — Extract implied-volatility skew

This reads from `options_enriched`, filters by the date range and max TTE in `config.py`, imputes missing implied volatilities via Black-Scholes, and computes a skew measure for each (ticker, date).

Run from the repository root:

```bash
python -m src.data_cleaning.extract_skew
```

Outputs:
- `data/skew.parquet` — long-format skew (date, ticker, skew)
- `data/cleaned_options.parquet` — cleaned options used downstream

This step is slow because it runs Black-Scholes inversion on every row with a missing IV. Run it once.

## Step 3 — Run the strategy and backtest

Two strategy variants are available:

**Stock vs Stock pairs (primary):**

```bash
python -m src.pairs_trading_skew
```

Trades all N*(N-1)/2 pairs of individual stocks. Signals are based on pairwise skew spread z-scores.

**Stock vs Sector pairs:**

```bash
python -m src.sector_pairs_trading_skew
```

Trades each stock against the sector ETF (XLF). Same signal logic but with a single hedge leg.

Both scripts load `data/skew.parquet` and `data/cleaned_options.parquet`, run the strategy pipeline, backtest, print performance metrics, and save plots to `plots/`.

## Configuration

All parameters live in `src/config.py`. Key settings:

| Parameter | Default | What it controls |
|---|---|---|
| `start_date` / `end_date` | 2015-01-01 / 2020-12-31 | Data window |
| `tte_target` | 15 | Target days-to-expiry for option selection |
| `delta_target` | 0.25 | Target delta for risk-reversal legs |
| `skew_method` | `"direct"` | How skew is measured (direct, polynomial, naive, vega_hedged, gamma_hedged, logmoneyness) |
| `entry_threshold_pct` | 0.975 | Percentile z-score threshold for entry |
| `estimation_window` | 60 | Rolling window for beta and spread stats |
| `transaction_cost_bps` | 20 | Option transaction cost in basis points |

## Script Overview

### Core pipeline (run these in order)

| Script | What it does |
|---|---|
| `src/data_loader.py` | Ingests raw CSVs into DuckDB, builds enriched options table |
| `src/data_cleaning/extract_skew.py` | Computes IV skew per (ticker, date) from options_enriched |
| `src/pairs_trading_skew.py` | Stock-vs-stock pairs trading strategy and backtest |
| `src/sector_pairs_trading_skew.py` | Stock-vs-sector (XLF) pairs trading strategy and backtest |

### Analysis and utilities

| Script | What it does |
|---|---|
| `src/config.py` | Central configuration (parameters, paths, column lists) |
| `src/run.py` | Sensitivity analysis — sweeps delta, TTE, transaction cost, entry threshold, and skew method |
| `src/cointegration_test.py` | Engle-Granger and Johansen cointegration tests on skew series |
| `src/drill_analytics.py` | Detailed P&L attribution plots from `data/pnl_drill.parquet` |
| `src/data_explorer.py` | EDA suite: missing values, distributions, correlations |
| `src/volatility_surface_generator.py` | Constructs and visualizes implied volatility surfaces |
| `src/utils/black_scholes.py` | Black-Scholes pricing, delta, and IV imputation |

### Running analysis scripts

Cointegration tests:

```bash
python -m src.cointegration_test
python -m src.cointegration_test --method johansen
```

Sensitivity analysis:

```bash
python -m src.run
```

Drill analytics (after running the backtest):

```bash
python -m src.drill_analytics
python -m src.drill_analytics --drill data/pnl_drill.parquet --out plots/drill_analytics
```
