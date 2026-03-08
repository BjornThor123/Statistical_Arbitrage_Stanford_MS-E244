from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from interfaces import BacktestOutput

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None


def _stack_ticker_frames(backtest: BacktestOutput) -> pd.DataFrame:
    parts = []
    for t, df in backtest.by_ticker.items():
        if df is None or df.empty:
            continue
        x = df.copy()
        if "ticker" not in x.columns:
            x["ticker"] = t
        x["date"] = pd.to_datetime(x["date"])
        parts.append(x)
    if not parts:
        return pd.DataFrame(columns=["date", "ticker"])
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def summarize_backtest(backtest: BacktestOutput) -> pd.DataFrame:
    p = backtest.portfolio.copy()
    if p.empty:
        return pd.DataFrame(
            [
                {
                    "n_days": 0.0,
                    "avg_daily_net_pnl": 0.0,
                    "std_daily_net_pnl": 0.0,
                    "daily_sharpe_like": 0.0,
                    "daily_sortino_like": 0.0,
                    "final_cum_net_pnl": 0.0,
                    "max_drawdown": 0.0,
                    "hit_rate": 0.0,
                    "avg_daily_turnover": 0.0,
                }
            ]
        )
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").reset_index(drop=True)
    if "portfolio_cum_net_pnl" not in p.columns:
        p["portfolio_cum_net_pnl"] = p["portfolio_net_pnl"].cumsum()
    eq = p["portfolio_cum_net_pnl"].to_numpy(dtype=np.float64)
    dd = eq - np.maximum.accumulate(eq)
    r = p["portfolio_net_pnl"].to_numpy(dtype=np.float64)
    mu = float(np.mean(r))
    sd = float(np.std(r))
    downside = float(np.std(np.minimum(r, 0.0)))
    t = _stack_ticker_frames(backtest)
    avg_turnover = float(t["turnover"].mean()) if "turnover" in t.columns and not t.empty else np.nan
    return pd.DataFrame(
        [
            {
                "n_days": float(len(p)),
                "avg_daily_net_pnl": mu,
                "std_daily_net_pnl": sd,
                "daily_sharpe_like": float(mu / sd * np.sqrt(252.0)) if sd > 1e-12 else 0.0,
                "daily_sortino_like": float(mu / downside * np.sqrt(252.0)) if downside > 1e-12 else 0.0,
                "final_cum_net_pnl": float(eq[-1]),
                "max_drawdown": float(np.min(dd)) if len(dd) else 0.0,
                "hit_rate": float(np.mean(r > 0.0)),
                "avg_daily_turnover": avg_turnover,
                "total_option_pnl": float(p["portfolio_option_pnl"].sum()) if "portfolio_option_pnl" in p.columns else np.nan,
                "total_hedge_pnl": float(p["portfolio_hedge_pnl"].sum()) if "portfolio_hedge_pnl" in p.columns else np.nan,
                "total_cost": float(p["portfolio_cost"].sum()) if "portfolio_cost" in p.columns else np.nan,
            }
        ]
    )


def pnl_attribution(backtest: BacktestOutput) -> pd.DataFrame:
    """PnL attribution by component and ticker."""
    t = _stack_ticker_frames(backtest)
    if t.empty:
        return pd.DataFrame(columns=["ticker", "option_pnl", "hedge_pnl", "cost", "net_pnl", "recon_error"])
    if "gross_option_pnl" not in t.columns:
        t["gross_option_pnl"] = np.nan
    if "gross_hedge_pnl" not in t.columns:
        t["gross_hedge_pnl"] = np.nan
    if "actual_dollar_vega" not in t.columns:
        t["actual_dollar_vega"] = np.nan
    out = (
        t.groupby("ticker", as_index=False)
        .agg(
            option_pnl=("gross_option_pnl", "sum"),
            hedge_pnl=("gross_hedge_pnl", "sum"),
            cost=("cost", "sum"),
            net_pnl=("net_pnl", "sum"),
            trades=("trade_executed", "sum"),
            avg_abs_position=("position", lambda s: float(np.mean(np.abs(s)))),
            avg_actual_dollar_vega=("actual_dollar_vega", "mean"),
        )
        .sort_values("net_pnl")
        .reset_index(drop=True)
    )
    out["recon_error"] = out["option_pnl"] + out["hedge_pnl"] - out["cost"] - out["net_pnl"]
    return out


def plot_backtest_analytics(backtest: BacktestOutput, engine: Literal["plotly", "matplotlib"] = "plotly") -> pd.DataFrame:
    p = backtest.portfolio.copy()
    if p.empty:
        raise ValueError("Backtest portfolio is empty.")
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").reset_index(drop=True)
    if "portfolio_cum_net_pnl" not in p.columns:
        p["portfolio_cum_net_pnl"] = p["portfolio_net_pnl"].cumsum()
    p["drawdown"] = p["portfolio_cum_net_pnl"] - p["portfolio_cum_net_pnl"].cummax()
    if "portfolio_option_pnl" in p.columns:
        p["cum_option_pnl"] = p["portfolio_option_pnl"].cumsum()
    else:
        p["cum_option_pnl"] = np.nan
    if "portfolio_hedge_pnl" in p.columns:
        p["cum_hedge_pnl"] = p["portfolio_hedge_pnl"].cumsum()
    else:
        p["cum_hedge_pnl"] = np.nan
    p["cum_cost"] = p["portfolio_cost"].cumsum() if "portfolio_cost" in p.columns else np.nan
    p["rolling_sharpe_63d"] = (
        p["portfolio_net_pnl"].rolling(63).mean()
        / p["portfolio_net_pnl"].rolling(63).std().replace(0.0, np.nan)
        * np.sqrt(252.0)
    )

    t = _stack_ticker_frames(backtest)
    if t.empty:
        raise ValueError("Backtest by_ticker is empty.")
    if "residual" not in t.columns:
        t["residual"] = np.nan
    if "turnover" not in t.columns:
        t["turnover"] = np.nan
    if "n_contracts" not in t.columns:
        t["n_contracts"] = np.nan
    if "hedge_shares" not in t.columns:
        t["hedge_shares"] = np.nan

    residual_summary = (
        t.groupby("ticker", as_index=False)
        .agg(n_obs=("residual", "count"), mean_residual=("residual", "mean"), std_residual=("residual", "std"))
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    attribution = pnl_attribution(backtest)
    total_option = float(np.nansum(p.get("portfolio_option_pnl", 0.0)))
    total_hedge = float(np.nansum(p.get("portfolio_hedge_pnl", 0.0)))
    total_skew_vega = float(np.nansum(p.get("portfolio_skew_vega_pnl", 0.0)))
    total_level_vega = float(np.nansum(p.get("portfolio_level_vega_pnl", 0.0)))
    total_gamma = float(np.nansum(p.get("portfolio_gamma_pnl", 0.0)))
    total_theta = float(np.nansum(p.get("portfolio_theta_pnl", 0.0)))
    total_unexplained = float(np.nansum(p.get("portfolio_unexplained_pnl", 0.0)))
    total_cost = float(np.nansum(p.get("portfolio_cost", 0.0)))
    total_net = float(np.nansum(p.get("portfolio_net_pnl", 0.0)))
    denom = max(abs(total_net), 1e-12)
    component_table = pd.DataFrame(
        [
            {"component": "option_leg", "total_pnl": total_option, "vs_net_multiple": total_option / denom},
            {"component": "hedge_leg", "total_pnl": total_hedge, "vs_net_multiple": total_hedge / denom},
            {"component": "skew_vega", "total_pnl": total_skew_vega, "vs_net_multiple": total_skew_vega / denom},
            {"component": "level_vega", "total_pnl": total_level_vega, "vs_net_multiple": total_level_vega / denom},
            {"component": "gamma", "total_pnl": total_gamma, "vs_net_multiple": total_gamma / denom},
            {"component": "theta", "total_pnl": total_theta, "vs_net_multiple": total_theta / denom},
            {"component": "unexplained", "total_pnl": total_unexplained, "vs_net_multiple": total_unexplained / denom},
            {"component": "cost", "total_pnl": -total_cost, "vs_net_multiple": (-total_cost) / denom},
            {"component": "net", "total_pnl": total_net, "vs_net_multiple": 1.0},
        ]
    )
    worst_days = p.sort_values("portfolio_net_pnl").head(10).copy()
    worst_days_cols = ["date", "portfolio_net_pnl", "portfolio_option_pnl", "portfolio_hedge_pnl", "portfolio_cost"]
    worst_days = worst_days[[c for c in worst_days_cols if c in worst_days.columns]]

    summary = summarize_backtest(backtest).round(6)
    print("Backtest summary:")
    print(summary)
    print("PnL component totals:")
    print(component_table.round(6))
    print("Worst 10 days (PnL drivers):")
    print(worst_days.round(6))
    print("Residual summary:")
    print(residual_summary.round(6))
    print("PnL attribution by ticker:")
    print(attribution.round(6))
    max_recon_error = float(np.nanmax(np.abs(attribution["recon_error"]))) if not attribution.empty else 0.0
    if max_recon_error > 1e-6:
        print(f"WARNING: ticker attribution does not reconcile exactly. max_abs_recon_error={max_recon_error:.6f}")

    if engine == "plotly":
        if go is None or make_subplots is None:
            raise ImportError("plotly is not installed. Install plotly or use engine='matplotlib'.")
        fig = make_subplots(
            rows=5,
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.05,
            horizontal_spacing=0.06,
            subplot_titles=(
                "Cumulative Net PnL", "Drawdown",
                "Daily Net/Gross/Option/Hedge/Cost", "Rolling Sharpe (63d)",
                "Residual by Ticker", "Position by Ticker",
                "Contracts by Ticker", "Hedge Shares by Ticker",
                "Turnover by Ticker", "Ticker Net PnL Attribution",
            ),
        )
        fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_cum_net_pnl"], mode="lines", name="Cum Net"), row=1, col=1)
        if "cum_option_pnl" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=p["cum_option_pnl"], mode="lines", name="Cum Option"), row=1, col=1)
        if "cum_hedge_pnl" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=p["cum_hedge_pnl"], mode="lines", name="Cum Hedge"), row=1, col=1)
        if "cum_cost" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=-p["cum_cost"], mode="lines", name="-Cum Cost"), row=1, col=1)
        fig.add_trace(go.Scatter(x=p["date"], y=p["drawdown"], mode="lines", name="Drawdown", fill="tozeroy"), row=1, col=2)
        fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_net_pnl"], mode="lines", name="Net"), row=2, col=1)
        fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_gross_pnl"], mode="lines", name="Gross"), row=2, col=1)
        if "portfolio_option_pnl" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_option_pnl"], mode="lines", name="Option"), row=2, col=1)
        if "portfolio_hedge_pnl" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_hedge_pnl"], mode="lines", name="Hedge"), row=2, col=1)
        if "portfolio_skew_vega_pnl" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_skew_vega_pnl"], mode="lines", name="Skew Vega"), row=2, col=1)
        if "portfolio_level_vega_pnl" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_level_vega_pnl"], mode="lines", name="Level Vega"), row=2, col=1)
        if "portfolio_gamma_pnl" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_gamma_pnl"], mode="lines", name="Gamma"), row=2, col=1)
        if "portfolio_theta_pnl" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_theta_pnl"], mode="lines", name="Theta"), row=2, col=1)
        if "portfolio_unexplained_pnl" in p.columns:
            fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_unexplained_pnl"], mode="lines", name="Unexplained"), row=2, col=1)
        fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_cost"], mode="lines", name="Cost"), row=2, col=1)
        fig.add_trace(go.Scatter(x=p["date"], y=p["rolling_sharpe_63d"], mode="lines", name="Sharpe63d"), row=2, col=2)

        for k, g in t.groupby("ticker", sort=True):
            gs = g.sort_values("date")
            fig.add_trace(go.Scatter(x=gs["date"], y=gs["residual"], mode="lines", name=f"Residual {k}"), row=3, col=1)
            fig.add_trace(go.Scatter(x=gs["date"], y=gs["position"], mode="lines", name=f"Pos {k}"), row=3, col=2)
            fig.add_trace(go.Scatter(x=gs["date"], y=gs["n_contracts"], mode="lines", name=f"Contracts {k}"), row=4, col=1)
            fig.add_trace(go.Scatter(x=gs["date"], y=gs["hedge_shares"], mode="lines", name=f"Hedge {k}"), row=4, col=2)
            fig.add_trace(go.Scatter(x=gs["date"], y=gs["turnover"], mode="lines", name=f"Turnover {k}"), row=5, col=1)

        y_col = "net_pnl" if "net_pnl" in attribution.columns else attribution.columns[-1]
        fig.add_trace(
            go.Bar(x=attribution["ticker"], y=attribution[y_col], name="Ticker Net PnL"),
            row=5,
            col=2,
        )
        fig.update_layout(height=1600, title="Backtest Risk & Performance Dashboard", hovermode="x unified")
        fig.show()
        return summary

    fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex="col")
    axes = axes.reshape(5, 2)
    axes[0, 0].plot(p["date"], p["portfolio_cum_net_pnl"], label="Cum Net")
    if "cum_option_pnl" in p.columns:
        axes[0, 0].plot(p["date"], p["cum_option_pnl"], label="Cum Option")
    if "cum_hedge_pnl" in p.columns:
        axes[0, 0].plot(p["date"], p["cum_hedge_pnl"], label="Cum Hedge")
    if "cum_cost" in p.columns:
        axes[0, 0].plot(p["date"], -p["cum_cost"], label="-Cum Cost")
    axes[0, 0].legend()
    axes[0, 0].set_title("Cumulative PnL Decomposition")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(p["date"], p["drawdown"]); axes[0, 1].fill_between(p["date"], p["drawdown"], 0.0, alpha=0.3); axes[0, 1].set_title("Drawdown"); axes[0, 1].grid(alpha=0.25)
    axes[1, 0].plot(p["date"], p["portfolio_net_pnl"], label="Net")
    axes[1, 0].plot(p["date"], p["portfolio_gross_pnl"], label="Gross")
    if "portfolio_option_pnl" in p.columns:
        axes[1, 0].plot(p["date"], p["portfolio_option_pnl"], label="Option")
    if "portfolio_hedge_pnl" in p.columns:
        axes[1, 0].plot(p["date"], p["portfolio_hedge_pnl"], label="Hedge")
    if "portfolio_skew_vega_pnl" in p.columns:
        axes[1, 0].plot(p["date"], p["portfolio_skew_vega_pnl"], label="SkewVega")
    if "portfolio_level_vega_pnl" in p.columns:
        axes[1, 0].plot(p["date"], p["portfolio_level_vega_pnl"], label="LevelVega")
    if "portfolio_gamma_pnl" in p.columns:
        axes[1, 0].plot(p["date"], p["portfolio_gamma_pnl"], label="Gamma")
    if "portfolio_theta_pnl" in p.columns:
        axes[1, 0].plot(p["date"], p["portfolio_theta_pnl"], label="Theta")
    if "portfolio_unexplained_pnl" in p.columns:
        axes[1, 0].plot(p["date"], p["portfolio_unexplained_pnl"], label="Unexplained")
    axes[1, 0].plot(p["date"], p["portfolio_cost"], label="Cost")
    axes[1, 0].legend(); axes[1, 0].set_title("Daily PnL"); axes[1, 0].grid(alpha=0.25)
    axes[1, 1].plot(p["date"], p["rolling_sharpe_63d"]); axes[1, 1].set_title("Rolling Sharpe (63d)"); axes[1, 1].grid(alpha=0.25)
    for k, g in t.groupby("ticker", sort=True):
        gs = g.sort_values("date")
        axes[2, 0].plot(gs["date"], gs["residual"], label=str(k))
        axes[2, 1].plot(gs["date"], gs["position"], label=str(k))
        axes[3, 0].plot(gs["date"], gs["n_contracts"], label=str(k))
        axes[3, 1].plot(gs["date"], gs["hedge_shares"], label=str(k))
        axes[4, 0].plot(gs["date"], gs["turnover"], label=str(k))
    axes[2, 0].set_title("Residual by Ticker"); axes[2, 0].grid(alpha=0.25); axes[2, 0].legend(ncol=2, fontsize=8)
    axes[2, 1].set_title("Position by Ticker"); axes[2, 1].grid(alpha=0.25); axes[2, 1].legend(ncol=2, fontsize=8)
    axes[3, 0].set_title("Contracts by Ticker"); axes[3, 0].grid(alpha=0.25); axes[3, 0].legend(ncol=2, fontsize=8)
    axes[3, 1].set_title("Hedge Shares by Ticker"); axes[3, 1].grid(alpha=0.25); axes[3, 1].legend(ncol=2, fontsize=8)
    axes[4, 0].set_title("Turnover by Ticker"); axes[4, 0].grid(alpha=0.25); axes[4, 0].legend(ncol=2, fontsize=8)
    axes[4, 1].bar(attribution["ticker"], attribution["net_pnl"]); axes[4, 1].set_title("Ticker Net PnL Attribution"); axes[4, 1].grid(alpha=0.25)
    plt.tight_layout()
    plt.show()
    return summary
