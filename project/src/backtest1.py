"""
Backtest for the idiosyncratic skew arbitrage strategy.

P&L model
---------
The signal determines a directional position in the underlying stock each day:

    pnl[i, t] = signal[i, t-1] × spot_return[i, t]

where spot_return = (price_t - price_{t-1}) / price_{t-1}.

Signal conventions
  +1 (sell skew): stock skew too high → sell puts → net long the underlying
  -1 (buy skew) : stock skew too low  → buy puts  → net short the underlying

Capital is allocated equally across all active positions each day, capped at
max_position_frac per stock. Transaction costs are charged whenever a position
opens, closes, or reverses.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd
import seaborn as sns


PLOT_DIR = Path("project/plots")


# ── Portfolio simulation ──────────────────────────────────────────────────────

def compute_portfolio_returns(
    signals: pd.DataFrame,
    spot_prices: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    max_position_frac: float = 0.20,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Simulate daily portfolio P&L using spot-price returns as the P&L driver.

    Parameters
    ----------
    signals              : (+1 / 0 / -1) DataFrame, date × ticker.
    spot_prices          : daily closing spot prices, date × ticker.
    initial_capital      : starting portfolio value in $.
    max_position_frac    : maximum fraction of portfolio per single position.
    transaction_cost_bps : one-way cost in basis points, charged on signal changes.

    Returns
    -------
    DataFrame with columns: gross_returns, net_returns, portfolio_value,
    transaction_cost, active_positions, n_trades, cumulative_gross, cumulative_net.
    """
    common_idx  = signals.index.intersection(spot_prices.index)
    signals     = signals.reindex(common_idx)
    spot_prices = spot_prices.reindex(common_idx)

    # Daily spot returns — the natural P&L unit, already in % terms
    spot_returns = spot_prices.pct_change()

    # P&L per stock: hold position signal[t-1], realise return[t]
    lagged_signal = signals.shift(1)
    raw_pnl       = lagged_signal * spot_returns

    # Equal-weight allocation across active positions, capped per stock
    n_active = (lagged_signal != 0).sum(axis=1)
    weight   = np.minimum(1.0 / n_active.replace(0, np.nan), max_position_frac)

    gross_returns = raw_pnl.multiply(weight, axis=0).sum(axis=1)

    # Transaction cost: charged when a position opens, closes, or reverses
    signal_changes = (signals != signals.shift(1)) & (signals != 0)
    n_trades = signal_changes.sum(axis=1)
    txn_cost = (n_trades / n_active.clip(lower=1)) * (transaction_cost_bps / 10_000)

    net_returns = gross_returns - txn_cost

    cumulative_gross = (1 + gross_returns).cumprod()
    cumulative_net   = (1 + net_returns).cumprod()

    return pd.DataFrame({
        "gross_returns":    gross_returns,
        "net_returns":      net_returns,
        "portfolio_value":  initial_capital * cumulative_net,
        "transaction_cost": txn_cost,
        "active_positions": n_active,
        "n_trades":         n_trades,
        "cumulative_gross": cumulative_gross,
        "cumulative_net":   cumulative_net,
    })


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(metrics_df: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """
    Compute standard performance metrics from a daily-returns DataFrame.

    Parameters
    ----------
    metrics_df     : output of compute_portfolio_returns.
    risk_free_rate : annualised risk-free rate (default 0).

    Returns
    -------
    dict of scalar performance statistics.
    """
    rfr_daily = risk_free_rate / 252
    results   = {}

    for label, col in [("Gross", "gross_returns"), ("Net", "net_returns")]:
        r = metrics_df[col].dropna()

        total_ret = (1 + r).prod() - 1
        ann_ret   = (1 + total_ret) ** (252 / len(r)) - 1
        ann_vol   = r.std() * np.sqrt(252)
        sharpe    = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

        cum      = (1 + r).cumprod()
        drawdown = (cum / cum.cummax()) - 1
        max_dd   = drawdown.min()
        calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

        results[f"{label} Total Return"]    = total_ret
        results[f"{label} Ann. Return"]     = ann_ret
        results[f"{label} Ann. Volatility"] = ann_vol
        results[f"{label} Sharpe Ratio"]    = sharpe
        results[f"{label} Max Drawdown"]    = max_dd
        results[f"{label} Calmar Ratio"]    = calmar
        results[f"{label} Win Rate"]        = (r > rfr_daily).mean()
        results[f"{label} Skewness"]        = r.skew()
        results[f"{label} Kurtosis"]        = r.kurtosis()

    results["Avg Active Positions"]   = metrics_df["active_positions"].mean()
    results["Avg Daily Trades"]       = metrics_df["n_trades"].mean()
    results["Total Transaction Cost"] = metrics_df["transaction_cost"].sum()

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    metrics_df: pd.DataFrame,
    z_scores: pd.DataFrame,
    signals: pd.DataFrame,
    plot_dir: Path | str = PLOT_DIR,
) -> None:
    """
    Produce and save all backtest plots to plot_dir.

    Saved files
    -----------
    cumulative_returns.png  – gross vs net cumulative P&L
    drawdown.png            – net drawdown over time
    active_positions.png    – number of active positions per day
    z_scores.png            – idiosyncratic z-scores with entry thresholds
    signals.png             – signal heatmap (date × ticker)
    monthly_returns.png     – monthly net-return heatmap
    annual_returns.png      – annual net-return bar chart
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cumulative returns ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    (metrics_df["cumulative_gross"] - 1).plot(ax=ax, label="Gross", linewidth=1.5)
    (metrics_df["cumulative_net"]   - 1).plot(ax=ax, label="Net",   linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "cumulative_returns.png", dpi=150)
    plt.close(fig)

    # 2. Drawdown ───────────────────────────────────────────────────────────
    cum      = metrics_df["cumulative_net"]
    drawdown = (cum / cum.cummax()) - 1
    fig, ax  = plt.subplots(figsize=(12, 4))
    drawdown.plot(ax=ax, color="red", linewidth=1)
    ax.fill_between(drawdown.index, drawdown, 0, alpha=0.25, color="red")
    ax.yaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    ax.set_title("Drawdown (Net)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "drawdown.png", dpi=150)
    plt.close(fig)

    # 3. Active positions ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    metrics_df["active_positions"].plot(ax=ax, color="steelblue")
    ax.set_title("Number of Active Positions")
    ax.set_ylabel("Count")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "active_positions.png", dpi=150)
    plt.close(fig)

    # 4. Z-scores per ticker ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in z_scores.columns:
        z_scores[ticker].dropna().plot(ax=ax, label=ticker, alpha=0.8, linewidth=1)
    ax.axhline( 1, color="grey", linewidth=0.8, linestyle="--", label="±1σ threshold")
    ax.axhline(-1, color="grey", linewidth=0.8, linestyle="--")
    ax.axhline( 0, color="black", linewidth=0.8)
    ax.set_title("Idiosyncratic Skew Z-Scores")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "z_scores.png", dpi=150)
    plt.close(fig)

    # 5. Signal heatmap ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, max(4, len(signals.columns) * 0.6)))
    step = max(1, len(signals) // 20)
    sns.heatmap(
        signals.T.astype(float),
        ax=ax,
        cmap="RdYlGn",
        center=0,
        cbar_kws={"ticks": [-1, 0, 1], "label": "Signal"},
        xticklabels=step,
        yticklabels=True,
    )
    tick_locs = range(0, len(signals), step)
    ax.set_xticks(list(tick_locs))
    ax.set_xticklabels(
        [signals.index[i].strftime("%Y-%m-%d") for i in tick_locs],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_title("Trading Signals (+1 sell skew / -1 buy skew)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(plot_dir / "signals.png", dpi=150)
    plt.close(fig)

    # 6. Monthly returns heatmap ────────────────────────────────────────────
    try:
        monthly = (
            metrics_df["net_returns"]
            .resample("ME")
            .apply(lambda x: (1 + x).prod() - 1)
        )
        monthly_table = pd.DataFrame({
            "month": monthly.index.month,
            "year":  monthly.index.year,
            "ret":   monthly.values,
        }).pivot_table(index="month", columns="year", values="ret")

        fig, ax = plt.subplots(figsize=(max(8, len(monthly_table.columns) * 1.2), 6))
        sns.heatmap(
            monthly_table.T, annot=True, fmt=".1%",
            cmap="RdYlGn", center=0, ax=ax,
        )
        ax.set_title("Monthly Net Returns")
        fig.tight_layout()
        fig.savefig(plot_dir / "monthly_returns.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass  # skip if insufficient data

    # 7. Annual returns ─────────────────────────────────────────────────────
    annual = (
        metrics_df["net_returns"]
        .resample("YE")
        .apply(lambda x: (1 + x).prod() - 1)
    )
    annual.index = annual.index.year
    colors = ["green" if r > 0 else "red" for r in annual]
    fig, ax = plt.subplots(figsize=(max(6, len(annual) * 1.2), 4))
    annual.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
    ax.yaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    ax.set_title("Annual Net Returns")
    ax.set_xlabel("Year")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "annual_returns.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved → {plot_dir.resolve()}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_backtest(
    signals: pd.DataFrame,
    spot_prices: pd.DataFrame,
    z_scores: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    max_position_frac: float = 0.20,
    transaction_cost_bps: float = 5.0,
    risk_free_rate: float = 0.0,
    plot_dir: Path | str = PLOT_DIR,
) -> dict:
    """
    Run the full backtest pipeline: simulate returns → metrics → plots.

    Parameters
    ----------
    signals              : wide DataFrame of signals (+1 / 0 / -1), date × ticker.
    spot_prices          : daily spot prices, date × ticker.
    z_scores             : rolling z-scores (used for the z-score diagnostic plot).
    initial_capital      : starting portfolio value in $.
    max_position_frac    : max fraction of portfolio allocated per stock.
    transaction_cost_bps : one-way transaction cost in basis points.
    risk_free_rate       : annualised risk-free rate for Sharpe calculation.
    plot_dir             : directory where plots are saved.

    Returns
    -------
    dict with keys: metrics_df, metrics, plot_dir.
    """
    print("Running backtest...")

    metrics_df = compute_portfolio_returns(
        signals, spot_prices,
        initial_capital=initial_capital,
        max_position_frac=max_position_frac,
        transaction_cost_bps=transaction_cost_bps,
    )

    metrics = compute_metrics(metrics_df, risk_free_rate=risk_free_rate)

    print("\nPerformance Summary")
    print("─" * 46)
    for k, v in metrics.items():
        if isinstance(v, float):
            if any(kw in k for kw in ("Return", "Drawdown", "Rate", "Volatility")):
                print(f"  {k:<40} {v:>8.2%}")
            else:
                print(f"  {k:<40} {v:>8.4f}")
        else:
            print(f"  {k:<40} {v}")

    plot_results(metrics_df, z_scores, signals, plot_dir=plot_dir)

    return {"metrics_df": metrics_df, "metrics": metrics, "plot_dir": str(plot_dir)}
