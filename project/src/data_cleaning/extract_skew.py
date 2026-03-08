"""
Data cleaning script: extract implied-volatility skew from raw options data.

Run once (slow — imputes missing IVs via Black-Scholes):
    python -m src.data_cleaning.extract_skew

Output: config.skew_path  (long-format parquet: date, ticker, skew)
"""

import numpy as np
import pandas as pd

from src.config import get_config
from src.data_loader import DataLoader
from src.utils.black_scholes import impute_impl_vol_bs

config = get_config()


def extract_skew_df(df, tte_days=15, min_points=3, verbose=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each ticker and date, fit:
        IV = α + β·log(K/F) + γ·log(K/F)²
    using OTM calls and puts. Return β (the skew slope) at the given tte_days
    via linear interpolation between adjacent maturities when needed.

    Returns (skew_df, cleaned_options_df) where:
      - skew_df has columns [skew, ticker] indexed by date
      - cleaned_options_df is the concatenation of all cleaned per-ticker option slices
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
    cleaned_stock_dfs = []

    for ticker in tickers:
        otm_filters = (
            (df['ticker'] == ticker) &
            (
                ((df['log_moneyness'] >= 0) & (df['cp_flag'] == 'C')) |
                ((df['log_moneyness'] < 0)  & (df['cp_flag'] == 'P'))
            )
        )
        df_stock = df.loc[otm_filters]

        if verbose: print(f'Imputing missing implied volatilities for {ticker}')
        df_stock = impute_impl_vol_bs(df_stock)
        if verbose: print(f'Done imputing missing implied volatilities for {ticker}')

        clean_filters = (
            (df_stock['tte_days'] >= 4) &
            (df_stock['impl_volatility'] <= df_stock['impl_volatility'].quantile(q=0.9))
        )
        df_stock = df_stock.loc[clean_filters]
        cleaned_stock_dfs.append(df_stock)

        skew_series = df_stock.groupby('date')[
            ['log_moneyness', 'impl_volatility', 'tte_days']
        ].apply(_run_regression)

        skew_sub_df = skew_series.rename('skew').to_frame()
        skew_sub_df['ticker'] = ticker
        skew_dfs.append(skew_sub_df)

    return pd.concat(skew_dfs), pd.concat(cleaned_stock_dfs)


def main():
    loader = DataLoader(data_path=config.data_path)

    query = (
        f"SELECT {', '.join(config.relevant_option_columns)} FROM options_enriched"
        f" WHERE date >= '{config.start_date}' AND date <= '{config.end_date}'"
        f" AND tte_days <= {config.max_tte}"
    )
    df = loader.query(query)

    skew_df, cleaned_options_df = extract_skew_df(df, tte_days=15)

    out_path = config.skew_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skew_df.to_parquet(out_path)
    print(f"Skew saved → {out_path.resolve()}")

    cleaned_path = config.cleaned_options_path
    cleaned_options_df.to_parquet(cleaned_path)
    print(f"Cleaned options saved → {cleaned_path.resolve()}")


if __name__ == "__main__":
    main()
