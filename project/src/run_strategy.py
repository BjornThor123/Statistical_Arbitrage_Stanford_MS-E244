import pandas as pd
from typing import Callable
from src.config import get_config
from src.construct_skew import extract_skew_df, compute_idiosyncratic_skew

config = get_config()


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_strategy(
    df: pd.DataFrame,
    compute_signals: Callable,
    tte_days: int = 15,
    estimation_window: int = 60,
    z_threshold: float = 1.0,
    sector_ticker: str = config.sector_ticker,
) -> dict:
    """
    Orchestrate the full idiosyncratic skew arbitrage pipeline:

      1. Extract skew β for every ticker and date.
      2. Pivot to wide format (date × ticker).
      3. Isolate idiosyncratic skew via rolling OLS against the sector ETF.
      4. Generate z-score-based trading signals.

    Returns
    -------
    dict with keys:
      skew_df    – raw skew (long format, columns [skew, ticker])
      skew_pivot – raw skew (wide format, date × ticker)
      resid_df   – idiosyncratic residuals (wide, sector ETF excluded)
      z_scores   – rolling z-scores (wide)
      signals    – trading signals +1 / 0 / -1 (wide)
    """
    print("Step 1/4  Extracting skew...")
    skew_df = extract_skew_df(df, tte_days=tte_days)

    skew_pivot = (
        skew_df.reset_index()
        .pivot(index='date', columns='ticker', values='skew')
    )
    skew_pivot.columns.name = None

    print("Step 2/4  Isolating idiosyncratic skew (rolling OLS vs XLF)...")
    resid_df = compute_idiosyncratic_skew(
        skew_pivot, sector_ticker=sector_ticker, estimation_window=estimation_window
    )

    print("Step 3/4  Computing z-scores and signals...")
    signals, z_scores = compute_signals(
        resid_df, z_threshold=z_threshold, signal_window=estimation_window
    )

    # Extract daily spot prices (one observation per ticker per date is enough)
    spot_prices = (
        df.groupby(["date", "ticker"])["spot_price"]
        .first()
        .unstack("ticker")
    )
    spot_prices.index = pd.to_datetime(spot_prices.index)
    # Drop the sector ETF so signals and spot_prices have the same columns
    stock_tickers = [t for t in spot_prices.columns if t != sector_ticker]
    spot_prices = spot_prices[stock_tickers]

    print("Done.")
    return {
        "skew_df":     skew_df,
        "skew_pivot":  skew_pivot,
        "resid_df":    resid_df,
        "z_scores":    z_scores,
        "signals":     signals,
        "spot_prices": spot_prices,
    }