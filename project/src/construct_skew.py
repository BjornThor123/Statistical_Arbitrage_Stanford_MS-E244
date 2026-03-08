import pandas as pd
import numpy as np
from src.config import get_config

config = get_config()


# ── Idiosyncratic skew via rolling OLS ──────────────────────────────────────

def compute_idiosyncratic_skew(
    skew_pivot: pd.DataFrame,
    sector_ticker: str = config.sector_ticker,
    estimation_window: int = 60,
) -> pd.DataFrame:
    """
    Isolate idiosyncratic skew for each stock via rolling OLS against the
    sector ETF (XLF):

        skew_{i,t} = α_i + β_i · skew_{XLF,t} + ε_{i,t}

    Parameters are estimated over a rolling `estimation_window`-day window
    so only past data is used, avoiding look-ahead bias.

    Parameters
    ----------
    skew_pivot : wide DataFrame (date × ticker) of raw skew values.
    sector_ticker : column name of the sector ETF in skew_pivot.
    estimation_window : rolling window length in trading days.

    Returns
    -------
    DataFrame of residuals ε_{i,t}, excluding the sector ETF column.
    """
    if sector_ticker not in skew_pivot.columns:
        raise ValueError(
            f"Sector ticker '{sector_ticker}' not found in skew_pivot. "
            f"Available tickers: {list(skew_pivot.columns)}"
        )

    sec_skew = skew_pivot[sector_ticker]
    stock_tickers = [t for t in skew_pivot.columns if t != sector_ticker]

    resid_df = pd.DataFrame(np.nan, index=skew_pivot.index, columns=stock_tickers)

    for ticker in stock_tickers:
        stock_skew = skew_pivot[ticker]

        # Vectorised rolling OLS:  β = cov(y,x) / var(x),  α = ȳ - β·x̄
        rolling_cov   = stock_skew.rolling(estimation_window).cov(sec_skew)
        rolling_var   = sec_skew.rolling(estimation_window).var()
        rolling_beta  = rolling_cov / rolling_var
        rolling_alpha = (
            stock_skew.rolling(estimation_window).mean()
            - rolling_beta * sec_skew.rolling(estimation_window).mean()
        )

        resid_df[ticker] = stock_skew - (rolling_alpha + rolling_beta * sec_skew)

    return resid_df
