from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from interfaces import WalkForwardOutput

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None


def summarize_walk_forward(wf: WalkForwardOutput) -> pd.DataFrame:
    p = wf.portfolio.copy()
    if p.empty:
        return pd.DataFrame(
            [
                {
                    "n_days": 0.0,
                    "avg_daily_net_pnl": 0.0,
                    "std_daily_net_pnl": 0.0,
                    "daily_sharpe_like": 0.0,
                    "final_cum_net_pnl": 0.0,
                    "max_drawdown": 0.0,
                    "hit_rate": 0.0,
                    "n_windows": 0.0,
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
    return pd.DataFrame(
        [
            {
                "n_days": float(len(p)),
                "avg_daily_net_pnl": mu,
                "std_daily_net_pnl": sd,
                "daily_sharpe_like": float(mu / sd * np.sqrt(252.0)) if sd > 1e-12 else 0.0,
                "final_cum_net_pnl": float(eq[-1]),
                "max_drawdown": float(np.min(dd)) if len(dd) else 0.0,
                "hit_rate": float(np.mean(r > 0.0)),
                "n_windows": float(len(wf.windows)),
            }
        ]
    )


def plot_walk_forward_analytics(
    wf: WalkForwardOutput,
    engine: Literal["plotly", "matplotlib"] = "plotly",
) -> pd.DataFrame:
    p = wf.portfolio.copy()
    if p.empty:
        raise ValueError("Walk-forward portfolio is empty.")
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").reset_index(drop=True)
    if "portfolio_cum_net_pnl" not in p.columns:
        p["portfolio_cum_net_pnl"] = p["portfolio_net_pnl"].cumsum()
    p["drawdown"] = p["portfolio_cum_net_pnl"] - p["portfolio_cum_net_pnl"].cummax()

    w = wf.windows.copy()
    if not w.empty:
        for c in ["train_start", "train_end", "test_start", "test_end"]:
            if c in w.columns:
                w[c] = pd.to_datetime(w[c])

    summary = summarize_walk_forward(wf).round(6)
    print("Walk-forward summary:")
    print(summary)
    if not w.empty:
        print("Walk-forward windows:")
        cols = [
            c
            for c in [
                "window_id",
                "selected_candidate_id",
                "train_score",
                "test_score",
                "test_sharpe",
                "test_final_cum_net_pnl",
            ]
            if c in w.columns
        ]
        print(w[cols].round(6))

    if engine == "plotly":
        if go is None or make_subplots is None:
            raise ImportError("plotly is not installed. Install plotly or use engine='matplotlib'.")
        fig = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=False,
            vertical_spacing=0.10,
            subplot_titles=(
                "OOS Cumulative Net PnL",
                "OOS Drawdown",
                "Window Train/Test Scores",
                "Selected Candidate by Window",
            ),
        )
        fig.add_trace(go.Scatter(x=p["date"], y=p["portfolio_cum_net_pnl"], mode="lines", name="OOS Cum Net"), row=1, col=1)
        fig.add_trace(go.Scatter(x=p["date"], y=p["drawdown"], mode="lines", fill="tozeroy", name="Drawdown"), row=1, col=2)
        if not w.empty:
            if "test_start" in w.columns and "train_score" in w.columns:
                fig.add_trace(go.Scatter(x=w["test_start"], y=w["train_score"], mode="lines+markers", name="Train Score"), row=2, col=1)
            if "test_start" in w.columns and "test_score" in w.columns:
                fig.add_trace(go.Scatter(x=w["test_start"], y=w["test_score"], mode="lines+markers", name="Test Score"), row=2, col=1)
            if "test_start" in w.columns and "selected_candidate_id" in w.columns:
                fig.add_trace(go.Bar(x=w["test_start"], y=w["selected_candidate_id"], name="Selected Candidate"), row=2, col=2)
        fig.update_layout(height=850, width=1200, title="Walk-Forward Analytics", template="plotly_white")
        fig.show()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes[0, 0].plot(p["date"], p["portfolio_cum_net_pnl"])
        axes[0, 0].set_title("OOS Cumulative Net PnL")
        axes[0, 0].grid(alpha=0.25)
        axes[0, 1].plot(p["date"], p["drawdown"])
        axes[0, 1].fill_between(p["date"], p["drawdown"], 0.0, alpha=0.25)
        axes[0, 1].set_title("OOS Drawdown")
        axes[0, 1].grid(alpha=0.25)
        if not w.empty and "test_start" in w.columns:
            if "train_score" in w.columns:
                axes[1, 0].plot(w["test_start"], w["train_score"], marker="o", label="Train")
            if "test_score" in w.columns:
                axes[1, 0].plot(w["test_start"], w["test_score"], marker="o", label="Test")
            axes[1, 0].set_title("Window Scores")
            axes[1, 0].grid(alpha=0.25)
            axes[1, 0].legend()
            if "selected_candidate_id" in w.columns:
                axes[1, 1].bar(w["test_start"], w["selected_candidate_id"])
        axes[1, 1].set_title("Selected Candidate by Window")
        axes[1, 1].grid(alpha=0.25)
        plt.tight_layout()
        plt.show()
    return summary
