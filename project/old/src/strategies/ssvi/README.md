# SSVI Skew Trading

## Core Layout

```text
project/src/
  common/
    skew_signal_backtest.py    # shared backtest/signal engine
  strategies/
    ssvi/
      model.py
      pipeline.py
      run.py
      dashboard.py
      results/
        current/
      _archive/                # non-core modules and old artifacts
```

## Setup

Run from `project`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run Strategy

Defaults target this repository's data path (`project/data/options/data/*.zip`).

```bash
python -m src.strategies.ssvi.run
```

### Run Strategy From DuckDB

```bash
python -m src.strategies.ssvi.run \
  --data-source duckdb \
  --db-path data/market_data.duckdb \
  --db-table options_enriched \
  --sector-ticker XLF
```

Optional: restrict universe with `--stock-tickers C GS JPM`.

### Fast Smoke Run

```bash
python -m src.strategies.ssvi.run \
  --start-date 2006-01-03 \
  --end-date 2006-03-31 \
  --calibration-backend cpu
```

## Build Dashboard

```bash
python -m src.strategies.ssvi.dashboard
```

Open `src/strategies/ssvi/results/current/dashboard.html`.
