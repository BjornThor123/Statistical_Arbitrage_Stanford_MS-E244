import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


def _bs_price(S: float, K: float, r: float, T: float, sigma: float, cp: str) -> float:
    """Black-Scholes price for a European option."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if cp == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _implied_vol_single(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    cp: str,
    vol_lo: float = 1e-4,
    vol_hi: float = 20.0,
) -> float:
    """Implied vol for one option via Brent's method. Returns NaN on failure."""
    if np.isnan(market_price) or market_price <= 0:
        return np.nan
    if np.isnan(S) or S <= 0 or np.isnan(K) or K <= 0 or T <= 0:
        return np.nan

    # Arbitrage floor: price must exceed discounted intrinsic value
    disc = np.exp(-r * T)
    intrinsic = max(S - K * disc, 0) if cp == "C" else max(K * disc - S, 0)
    if market_price <= intrinsic:
        return np.nan

    try:
        f_lo = _bs_price(S, K, r, T, vol_lo, cp) - market_price
        f_hi = _bs_price(S, K, r, T, vol_hi, cp) - market_price
        if f_lo * f_hi > 0:
            return np.nan
        return brentq(
            lambda sigma: _bs_price(S, K, r, T, sigma, cp) - market_price,
            vol_lo,
            vol_hi,
            xtol=1e-6,
            maxiter=200,
        )
    except (ValueError, RuntimeError):
        return np.nan


def impute_impl_vol_bs(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes implied volatility using Black-Scholes inversion for rows where
    impl_volatility is NaN.

    Uses Brent's method to find the volatility σ such that the BS price
    equals the observed mid_price. Only rows with a valid mid_price,
    spot_price, strike, tte, and cp_flag are filled.

    Parameters
    ----------
    df : DataFrame with columns impl_volatility, mid_price, spot_price,
         strike, risk_free_rate, tte, cp_flag.

    Returns
    -------
    Copy of df with NaN impl_volatility filled where possible.
    """
    df = df.copy()
    mask = df["impl_volatility"].isna()
    df["imputed_bs"] = False

    if not mask.any():
        return df

    rows = df.loc[mask]
    imputed = rows.apply(
        lambda row: _implied_vol_single(
            market_price=row["mid_price"],
            S=row["spot_price"],
            K=row["strike"],
            r=row["risk_free_rate"] if pd.notna(row["risk_free_rate"]) else 0.0,
            T=row["tte"],
            cp=row["cp_flag"],
        ),
        axis=1,
    )
    df.loc[mask, "impl_volatility"] = imputed
    # Only mark as imputed where the value was actually filled (not still NaN)
    df.loc[mask & df["impl_volatility"].notna(), "imputed_bs"] = True
    return df
