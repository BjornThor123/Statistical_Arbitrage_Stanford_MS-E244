import pandas as pd
import numpy as np
from src.config import get_config

config = get_config()


# ── Skew extraction ──────────────────────────────────────────────────────────

def extract_skew_df(df, tte_days=15, min_points=3) -> pd.DataFrame:
    """
    For each ticker and date, fit:
        IV = α + β·log(K/F) + γ·log(K/F)²
    using OTM calls and puts. Return β (the skew slope) at the given tte_days
    via linear interpolation between adjacent maturities when needed.

    Returns a DataFrame with columns [skew, ticker] indexed by date.
    """

    def _compute_beta(slice_df):
        if len(slice_df) < min_points:
            return np.nan
        x = slice_df['log_moneyness'].values
        iv = slice_df['impl_volatility'].values
        coeff = np.polyfit(x, iv, deg=2)
        return coeff[1]  # [γ, β, α] → coeff[1] is β (skew)

    def _run_regression(day_df):
        available_ttes = np.sort(day_df['tte_days'].unique())

        if tte_days in available_ttes:
            return _compute_beta(day_df.loc[day_df['tte_days'] == tte_days])

        ttes_below = available_ttes[available_ttes < tte_days]
        ttes_above = available_ttes[available_ttes > tte_days]

        if len(ttes_below) == 0 or len(ttes_above) == 0:
            return np.nan

        tau_l, tau_u = ttes_below.max(), ttes_above.min()
        beta_l = _compute_beta(day_df.loc[day_df['tte_days'] == tau_l])
        beta_u = _compute_beta(day_df.loc[day_df['tte_days'] == tau_u])

        if np.isnan(beta_l) or np.isnan(beta_u):
            return np.nan

        w = (tte_days - tau_l) / (tau_u - tau_l)
        return beta_l + w * (beta_u - beta_l)

    tickers = df['ticker'].unique()
    skew_dfs = []

    for ticker in tickers:
        otm_filter = (
            (df['ticker'] == ticker) &
            (
                ((df['log_moneyness'] >= 0) & (df['cp_flag'] == 'C')) |
                ((df['log_moneyness'] < 0)  & (df['cp_flag'] == 'P'))
            )
        )
        df_stock = df.loc[otm_filter]

        skew_series = df_stock.groupby('date')[
            ['log_moneyness', 'impl_volatility', 'tte_days']
        ].apply(_run_regression)

        skew_sub_df = skew_series.rename('skew').to_frame()
        skew_sub_df['ticker'] = ticker
        skew_dfs.append(skew_sub_df)

    return pd.concat(skew_dfs)


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