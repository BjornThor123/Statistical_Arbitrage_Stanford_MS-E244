from __future__ import annotations

from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from interfaces import SkewOutput

try:
    import plotly.graph_objects as go
except Exception:
    go = None


def _stack_skew_output(skew: SkewOutput, tickers: List[str] | None = None) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    keep = set(tickers) if tickers is not None else None
    for t, df in skew.skew_by_ticker.items():
        if keep is not None and t not in keep:
            continue
        if df is None or df.empty:
            continue
        x = df.copy()
        if "ticker" not in x.columns:
            x["ticker"] = t
        x["date"] = pd.to_datetime(x["date"])
        rows.append(x)
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "skew", "n_points", "rmse_implied_volatility"])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def _safe_autocorr(s: pd.Series, lag: int) -> float:
    if len(s) <= lag:
        return np.nan
    return float(s.autocorr(lag=lag))


def _half_life_from_acf1(acf1: float) -> float:
    if not np.isfinite(acf1) or acf1 <= 0.0 or acf1 >= 1.0:
        return np.nan
    return float(np.log(0.5) / np.log(acf1))


def summarize_skew_quality(skew: SkewOutput, tickers: List[str] | None = None) -> pd.DataFrame:
    """Return per-ticker summary and time-series diagnostics from SkewOutput."""
    df = _stack_skew_output(skew, tickers=tickers)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "n_dates",
                "mean_skew",
                "std_skew",
                "mean_abs_skew",
                "acf_lag1",
                "acf_lag5",
                "half_life_days",
                "mean_abs_delta",
                "sign_flip_rate",
                "mean_n_points",
                "mean_rmse_implied_volatility",
            ]
        )
    rows = []
    for t, g in df.groupby("ticker", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        s = g["skew"].astype(float)
        ds = s.diff().dropna()
        signs = np.sign(s.to_numpy(dtype=np.float64))
        sign_flips = float(np.mean(signs[1:] * signs[:-1] < 0.0)) if len(signs) > 1 else np.nan
        acf1 = _safe_autocorr(s, lag=1)
        acf5 = _safe_autocorr(s, lag=5)
        rows.append(
            {
                "ticker": t,
                "n_dates": int(g["date"].nunique()),
                "mean_skew": float(s.mean()),
                "std_skew": float(s.std()),
                "mean_abs_skew": float(np.mean(np.abs(s.to_numpy(dtype=np.float64)))),
                "acf_lag1": acf1,
                "acf_lag5": acf5,
                "half_life_days": _half_life_from_acf1(acf1),
                "mean_abs_delta": float(np.mean(np.abs(ds.to_numpy(dtype=np.float64)))) if len(ds) else np.nan,
                "sign_flip_rate": sign_flips,
                "mean_n_points": float(g["n_points"].mean()) if "n_points" in g.columns else np.nan,
                "mean_rmse_implied_volatility": float(g["rmse_implied_volatility"].mean())
                if "rmse_implied_volatility" in g.columns
                else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    return out


def plot_skew_analytics(
    skew: SkewOutput,
    tickers: List[str] | None = None,
    engine: Literal["plotly", "matplotlib"] = "plotly",
) -> pd.DataFrame:
    """Interactive skew visualization from run_skew output only.

    Returns the per-ticker summary diagnostics table.
    """
    df = _stack_skew_output(skew, tickers=tickers)
    if df.empty:
        raise ValueError("No skew rows available for requested tickers.")

    summary = summarize_skew_quality(skew, tickers=tickers)
    print("Skew quality summary:")
    print(summary.round(6))

    if engine == "plotly":
        if go is None:
            raise ImportError("plotly is not installed. Install plotly or use engine='matplotlib'.")
        fig = go.Figure()
        for t, g in df.groupby("ticker", sort=True):
            g = g.sort_values("date")
            custom_cols = []
            if "n_points" in g.columns:
                custom_cols.append(g["n_points"].to_numpy(dtype=np.float64))
            else:
                custom_cols.append(np.full(len(g), np.nan))
            if "rmse_implied_volatility" in g.columns:
                custom_cols.append(g["rmse_implied_volatility"].to_numpy(dtype=np.float64))
            else:
                custom_cols.append(np.full(len(g), np.nan))
            custom = np.column_stack(custom_cols)
            fig.add_trace(
                go.Scatter(
                    x=g["date"],
                    y=g["skew"],
                    mode="lines+markers",
                    name=str(t),
                    customdata=custom,
                    hovertemplate=(
                        "Ticker=" + str(t) + "<br>Date=%{x|%Y-%m-%d}<br>Skew=%{y:.6f}<br>"
                        "n_points=%{customdata[0]:.0f}<br>RMSE=%{customdata[1]:.6f}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title="Skew Time Series",
            xaxis_title="Date",
            yaxis_title="Skew",
            hovermode="x unified",
        )
        fig.show()
        return summary

    fig, ax = plt.subplots(figsize=(12, 6))
    for t, g in df.groupby("ticker", sort=True):
        g = g.sort_values("date")
        ax.plot(g["date"], g["skew"], marker="o", linewidth=1.8, markersize=3.5, label=str(t))
    ax.set_title("Skew Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Skew")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=2)
    plt.show()
    return summary
