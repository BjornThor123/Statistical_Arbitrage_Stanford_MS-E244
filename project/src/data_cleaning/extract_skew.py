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


def extract_skew_df(
    df,
    tte_days: int = 15,
    min_points: int = 3,
    skew_method: str = "direct",
    delta_target: float = 0.25,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each ticker and date, compute an IV skew measure.

    Two methods are supported (controlled by skew_method):

    "direct" (default)
        skew = IV_put(Δ≈delta_target) − IV_call(Δ≈delta_target)
        Selects the OTM put and call closest to delta_target at the expiry
        nearest to tte_days. Positive when puts are expensive — consistent
        with the risk-reversal instrument actually traded.

    "polynomial"
        Fits  IV = α + β·log(K/F) + γ·log(K/F)²  across all OTM options,
        interpolating linearly between adjacent maturities when tte_days is
        not available exactly. Returns −β so that the sign convention matches
        "direct" (high = puts expensive; β is negative in equity markets).

    Both methods share the same data-cleaning pipeline (imputation, tte ≥ 4,
    IV ≤ 90th-percentile clip).

    Returns (skew_df, cleaned_options_df) where:
      - skew_df has columns [skew, ticker] indexed by date
      - cleaned_options_df is the concatenation of all cleaned per-ticker slices
    """

    # ── Polynomial helpers ────────────────────────────────────────────────────

    def _compute_beta(slice_df):
        if len(slice_df) < min_points:
            return np.nan
        x = slice_df['log_moneyness'].values
        iv = slice_df['impl_volatility'].values
        coeff = np.polyfit(x, iv, deg=2)
        return coeff[1]  # [γ, β, α] → coeff[1] is β (skew slope)

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

    # ── Direct 25Δ helper ─────────────────────────────────────────────────────

    def _compute_direct_skew(day_df):
        """Return IV_put(Δ≈delta_target) − IV_call(Δ≈delta_target)."""
        available_ttes = day_df['tte_days'].unique()
        closest_tte = available_ttes[
            np.argmin(np.abs(available_ttes - tte_days))
        ]
        slice_df = day_df[day_df['tte_days'] == closest_tte]

        calls = slice_df[(slice_df['cp_flag'] == 'C') & (slice_df['log_moneyness'] >= 0)].copy()
        puts  = slice_df[(slice_df['cp_flag'] == 'P') & (slice_df['log_moneyness'] <  0)].copy()

        if calls.empty or puts.empty:
            return np.nan

        calls['delta_dist'] = (calls['delta'] - delta_target).abs()
        puts['delta_dist']  = (puts['delta'].abs() - delta_target).abs()

        best_call_iv = calls.loc[calls['delta_dist'].idxmin(), 'impl_volatility']
        best_put_iv  = puts.loc[puts['delta_dist'].idxmin(),  'impl_volatility']

        if np.isnan(best_call_iv) or np.isnan(best_put_iv):
            return np.nan

        # Positive when puts expensive (matches: high spread → sell puts → long RR)
        return float(best_put_iv) - float(best_call_iv)

    # ── Per-ticker loop ───────────────────────────────────────────────────────

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

        if skew_method == "direct":
            skew_series = df_stock.groupby('date').apply(_compute_direct_skew)
        else:  # polynomial
            # Negate β: in equity markets β < 0 (puts expensive), so −β > 0,
            # giving the same sign convention as the direct method.
            skew_series = -df_stock.groupby('date')[
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

    skew_df, cleaned_options_df = extract_skew_df(
        df,
        tte_days=config.tte_target,
        skew_method=config.skew_method,
        delta_target=config.delta_target,
    )

    out_path = config.skew_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skew_df.to_parquet(out_path)
    print(f"Skew saved → {out_path.resolve()}")

    cleaned_path = config.cleaned_options_path
    cleaned_options_df.to_parquet(cleaned_path)
    print(f"Cleaned options saved → {cleaned_path.resolve()}")


if __name__ == "__main__":
    main()
