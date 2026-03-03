"""
Idiosyncratic skew arbitrage strategy — spot-return P&L model.

Pipeline
--------
1. Extract skew β for every ticker and date.
2. Pivot to wide format (date × ticker).
3. Isolate idiosyncratic skew via rolling OLS against the sector ETF.
4. Generate z-score-based trading signals.
5. Simulate daily P&L using spot-price returns as the P&L driver.

P&L model
---------
    pnl[i, t] = signal[i, t-1] × spot_return[i, t]

Signal conventions
  +1 (sell skew): stock skew too high → sell puts → net long the underlying
  -1 (buy skew) : stock skew too low  → buy puts  → net short the underlying
   0  flat

Capital is allocated equally across all active positions each day, capped at
max_position_frac per stock. Transaction costs are charged whenever a position
opens, closes, or reverses.

To implement a strategy that trades options instead of (or in addition to) the
underlying, create a new strategy file following the same layout but replace
compute_portfolio_returns with your own P&L function.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import get_config
from src.construct_skew import extract_skew_df, compute_idiosyncratic_skew
from src.data_loader import DataLoader


config = get_config()



# ── Signal construction ───────────────────────────────────────────────────────

def compute_signals(
    resid_df: pd.DataFrame,
    entry_threshold: float = config.entry_threshold,
    exit_threshold: float = config.exit_threshold,
    signal_window: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalise idiosyncratic residuals into rolling z-scores, then apply a
    two-threshold (hysteresis) rule to reduce signal churn:

        z_{i,t} = (ε_{i,t} - μ_ε) / σ_ε   (rolling over signal_window days)

    Signal conventions
    ------------------
    +1  sell skew : z crosses above +entry_threshold (held until z < +exit_threshold)
    -1  buy skew  : z crosses below -entry_threshold (held until z > -exit_threshold)
     0  flat

    The exit_threshold < entry_threshold band prevents the signal from flipping on
    every small z-score oscillation near the entry boundary (hysteresis).

    Returns
    -------
    (signals, z_scores) – both DataFrames indexed like resid_df.
    """
    rolling_mean = resid_df.rolling(signal_window).mean()
    rolling_std  = resid_df.rolling(signal_window).std()
    z_scores     = (resid_df - rolling_mean) / rolling_std

    # Stateful hysteresis: must loop over time (each row depends on the previous
    # signal), but np.select vectorises across all tickers within each row.
    z_arr = z_scores.to_numpy()
    sig   = np.zeros_like(z_arr, dtype=int)

    for t in range(1, len(z_arr)):
        z    = z_arr[t]
        prev = sig[t - 1]
        nan  = np.isnan(z)

        flat  = (prev == 0) & ~nan
        long  = (prev == 1) & ~nan
        short = (prev == -1) & ~nan

        sig[t] = np.select(
            [
                nan,
                flat  & (z >  entry_threshold),          # enter long
                flat  & (z < -entry_threshold),          # enter short
                long  & (z >= exit_threshold),           # hold long
                long  & (z < -entry_threshold),          # flip to short
                short & (z <= -exit_threshold),          # hold short
                short & (z >  entry_threshold),          # flip to long
            ],
            [0, 1, -1, 1, -1, -1, 1],
            default=0,                                   # exit (flat + no entry)
        )

    signals = pd.DataFrame(sig, index=resid_df.index, columns=resid_df.columns)
    return signals, z_scores


# ── Strategy pipeline ─────────────────────────────────────────────────────────

def run_strategy(
    df: pd.DataFrame,
    tte_days: int = 15,
    estimation_window: int = 60,
    entry_threshold: float = config.entry_threshold,
    exit_threshold: float = config.exit_threshold,
    sector_ticker: str = config.sector_ticker,
) -> dict:
    """
    Orchestrate the full idiosyncratic skew arbitrage pipeline:

      1. Extract skew β for every ticker and date.
      2. Pivot to wide format (date × ticker).
      3. Isolate idiosyncratic skew via rolling OLS against the sector ETF.
      4. Generate z-score-based trading signals.
      5. Extract daily spot prices aligned to the signal dates.

    Returns
    -------
    dict with keys:
      skew_df    – raw skew (long format, columns [skew, ticker])
      skew_pivot – raw skew (wide format, date × ticker)
      resid_df   – idiosyncratic residuals (wide, sector ETF excluded)
      z_scores   – rolling z-scores (wide)
      signals    – trading signals +1 / 0 / -1 (wide)
      spot_prices – daily closing prices (wide, sector ETF excluded)
    """
    print("Step 1/4  Extracting skew...")
    skew_df = extract_skew_df(df, tte_days=tte_days)

    skew_pivot = (
        skew_df.reset_index()
        .pivot(index="date", columns="ticker", values="skew")
    )
    skew_pivot.columns.name = None

    print("Step 2/4  Isolating idiosyncratic skew (rolling OLS vs XLF)...")
    resid_df = compute_idiosyncratic_skew(
        skew_pivot, sector_ticker=sector_ticker, estimation_window=estimation_window
    )

    print("Step 3/4  Computing z-scores and signals...")
    signals, z_scores = compute_signals(
        resid_df,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        signal_window=estimation_window,
    )

    # Extract daily spot prices (one observation per ticker per date)
    spot_prices = (
        df.groupby(["date", "ticker"])["spot_price"]
        .first()
        .unstack("ticker")
    )
    spot_prices.index = pd.to_datetime(spot_prices.index)
    # Drop sector ETF so signals and spot_prices share the same columns
    stock_tickers = [t for t in spot_prices.columns if t != sector_ticker]
    spot_prices = spot_prices[stock_tickers]

    print("Done.")
    return {
        "skew_df":     skew_df,
        "skew_pivot":  skew_pivot,
        "resid_df":    resid_df,
        "z_scores":    z_scores,
        "signals":     signals,
        "spot_prices": spot_prices,
    }


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

    # Daily spot returns — the natural P&L unit
    spot_returns = spot_prices.pct_change()

    # P&L per stock: hold position signal[t-1], realise return[t]
    lagged_signal = signals.shift(1)
    raw_pnl       = lagged_signal * spot_returns

    # Equal-weight allocation across active positions, capped per stock
    n_active = (lagged_signal != 0).sum(axis=1)
    weight   = np.minimum(1.0 / n_active.replace(0, np.nan), max_position_frac)

    gross_returns = raw_pnl.multiply(weight, axis=0).sum(axis=1)

    # Transaction cost: charged on opens, closes, and reversals.
    # Use today's active count as the denominator so that re-entries from a flat
    # period (where yesterday's n_active=0 would otherwise blow up the cost) are
    # correctly scaled.  Also charge for closes (signal → 0), not just openings.
    prev_signal    = signals.shift(1).fillna(0)
    signal_changes = (signals != prev_signal) & ((signals != 0) | (prev_signal != 0))
    n_trades       = signal_changes.sum(axis=1)
    n_active_today = (signals != 0).sum(axis=1)
    txn_cost = (n_trades / n_active_today.clip(lower=1)) * (transaction_cost_bps / 10_000)

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
    plot_dir: Path | str = config.plot_dir / "simple_strat",
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


# ── Backtest orchestration ────────────────────────────────────────────────────

def run_backtest(
    signals: pd.DataFrame,
    spot_prices: pd.DataFrame,
    z_scores: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    max_position_frac: float = 0.20,
    transaction_cost_bps: float = 5.0,
    risk_free_rate: float = 0.0,
    plot_dir: Path | str = config.plot_dir,
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


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    loader = DataLoader(data_path=config.data_path)

    query = (
        f"SELECT {', '.join(config.relevant_option_columns)} FROM options_enriched"
        f" WHERE date >= '{config.start_date}' AND date <= '{config.end_date}'"
        f" AND tte_days <= {config.max_tte}"
    )
    df = loader.query(query)

    results = run_strategy(df)

    print("\nSignal counts per ticker:")
    print(
        results["signals"]
        .apply(lambda col: col.value_counts())
        .T.fillna(0).astype(int)
    )

    backtest_results = run_backtest(
        signals=results["signals"],
        spot_prices=results["spot_prices"],
        z_scores=results["z_scores"],
        plot_dir=config.plot_dir / "simple_strat",
    )

    return {**results, **backtest_results}


if __name__ == "__main__":
    main()
