from __future__ import annotations

from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from interfaces import SkewOutput

"""
Skew analytics.

plot_skew_analytics(skew)     — time-series of skew per ticker, faceted by tenor.
summarize_skew_quality(skew)  — per-ticker, per-tenor diagnostics: ACF, half-life, RMSE.
"""

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None

_TENOR_DASHES = {0: "solid", 1: "dash", 2: "dot", 3: "dashdot"}


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
    """Return per-ticker, per-tenor summary diagnostics from SkewOutput."""
    df = _stack_skew_output(skew, tickers=tickers)
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "tenor_days", "n_dates", "mean_skew", "std_skew", "mean_abs_skew",
            "acf_lag1", "acf_lag5", "half_life_days", "mean_abs_delta", "sign_flip_rate",
            "mean_n_points", "mean_rmse_implied_volatility",
        ])
    has_tenor = "tenor_days" in df.columns
    group_cols = ["ticker", "tenor_days"] if has_tenor else ["ticker"]
    rows = []
    for keys, g in df.groupby(group_cols, sort=True):
        if not has_tenor:
            ticker = keys
            tenor = np.nan
        else:
            ticker, tenor = keys
        g = g.sort_values("date").reset_index(drop=True)
        s = g["skew"].astype(float)
        ds = s.diff().dropna()
        signs = np.sign(s.to_numpy(dtype=np.float64))
        sign_flips = float(np.mean(signs[1:] * signs[:-1] < 0.0)) if len(signs) > 1 else np.nan
        acf1 = _safe_autocorr(s, lag=1)
        acf5 = _safe_autocorr(s, lag=5)
        rows.append({
            "ticker": ticker,
            "tenor_days": tenor,
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
        })
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def plot_skew_analytics(
    skew: SkewOutput,
    tickers: List[str] | None = None,
    engine: Literal["plotly", "matplotlib"] = "plotly",
) -> pd.DataFrame:
    """Skew time-series visualization, faceted by tenor when multi-tenor data is present.

    Returns the per-ticker/tenor summary diagnostics table.
    """
    df = _stack_skew_output(skew, tickers=tickers)
    if df.empty:
        raise ValueError("No skew rows available for requested tickers.")

    summary = summarize_skew_quality(skew, tickers=tickers)
    print("Skew quality summary:")
    print(summary.round(6))

    has_tenor = "tenor_days" in df.columns
    tenors = sorted(df["tenor_days"].unique().tolist()) if has_tenor else [None]
    n_tenors = len(tenors)

    if engine == "plotly":
        if go is None or make_subplots is None:
            raise ImportError("plotly is not installed. Install plotly or use engine='matplotlib'.")

        fig = make_subplots(
            rows=n_tenors,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"{td}d skew" if td is not None else "Skew" for td in tenors],
            vertical_spacing=0.06,
        )
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        active_tickers = sorted(df["ticker"].unique().tolist())

        for row_idx, td in enumerate(tenors, start=1):
            sub = df[df["tenor_days"] == td] if has_tenor else df
            for ti, t in enumerate(active_tickers):
                g = sub[sub["ticker"] == t].sort_values("date")
                if g.empty:
                    continue
                color = colors[ti % len(colors)]
                custom = np.column_stack([
                    g["n_points"].to_numpy(dtype=np.float64) if "n_points" in g.columns else np.full(len(g), np.nan),
                    g["rmse_implied_volatility"].to_numpy(dtype=np.float64) if "rmse_implied_volatility" in g.columns else np.full(len(g), np.nan),
                ])
                fig.add_trace(
                    go.Scatter(
                        x=g["date"],
                        y=g["skew"],
                        mode="lines+markers",
                        name=str(t),
                        legendgroup=str(t),
                        showlegend=(row_idx == 1),
                        line=dict(color=color),
                        customdata=custom,
                        hovertemplate=(
                            f"{t}<br>%{{x|%Y-%m-%d}}<br>Skew=%{{y:.6f}}<br>"
                            "n_points=%{customdata[0]:.0f}<br>RMSE=%{customdata[1]:.6f}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=1,
                )

        fig.update_layout(
            title="Skew Term Structure",
            hovermode="x unified",
            height=300 * n_tenors + 100,
        )
        for row_idx in range(1, n_tenors + 1):
            fig.update_yaxes(title_text="skew", row=row_idx, col=1)
        fig.show()
        return summary

    # matplotlib fallback
    fig, axes = plt.subplots(n_tenors, 1, figsize=(12, 4 * n_tenors), sharex=True)
    if n_tenors == 1:
        axes = [axes]
    active_tickers = sorted(df["ticker"].unique().tolist())
    prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, td in zip(axes, tenors):
        sub = df[df["tenor_days"] == td] if has_tenor else df
        for ti, t in enumerate(active_tickers):
            g = sub[sub["ticker"] == t].sort_values("date")
            if g.empty:
                continue
            ax.plot(g["date"], g["skew"], marker="o", linewidth=1.8, markersize=3.5,
                    label=str(t), color=prop_colors[ti % len(prop_colors)])
        ax.set_title(f"{td}d skew" if td is not None else "Skew")
        ax.set_ylabel("Skew")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", ncol=2)

    plt.tight_layout()
    plt.show()
    return summary
