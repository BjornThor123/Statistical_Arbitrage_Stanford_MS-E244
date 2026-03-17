"""
Drill analytics — report-quality plots from pnl_drill.parquet.

Reads data/pnl_drill.parquet (produced by calculate_portfolio_returns_drill)
and generates a full suite of attribution, decomposition, and regime plots
suitable for the final report.

Usage
-----
    python -m src.drill_analytics
    python -m src.drill_analytics --drill data/pnl_drill.parquet --out plots/drill_analytics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Plot style ────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PCT  = mpl_ticker.PercentFormatter(xmax=1, decimals=1)
PCT2 = mpl_ticker.PercentFormatter(xmax=1, decimals=2)

_SECTION_COLORS = {
    "gross": "#2196F3",
    "net":   "#4CAF50",
    "cost":  "#F44336",
    "i_leg": "#1976D2",
    "j_leg": "#FF9800",
    "opt":   "#7B1FA2",
    "hedge": "#009688",
}


# ── Data loading & derived quantities ─────────────────────────────────────────

def load_drill(path: Path) -> pd.DataFrame:
    drill = pd.read_parquet(path)
    drill["date"] = pd.to_datetime(drill["date"])
    return drill


def build_derived(drill: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    drill : original DataFrame augmented with weighted cost columns.
    daily : one row per date with portfolio-level aggregates reconstructed
            from the drill file.
    """
    # Weighted costs (match portfolio-level formula: txn_pair * weight)
    drill = drill.copy()
    drill["wtd_txn_bidask"] = drill["txn_bidask"] * drill["weight"]
    drill["wtd_txn_hedge"]  = drill["txn_hedge"]  * drill["weight"]
    drill["wtd_txn_total"]  = drill["txn_total"]  * drill["weight"]

    # Weighted leg-level P&L
    drill["wtd_i_opt"]   = drill["i_opt_ret"]   * drill["weight"]
    drill["wtd_i_hedge"] = drill["i_hedge_ret"]  * drill["weight"]
    drill["wtd_j_opt"]   = drill["j_opt_ret"]    * drill["weight"]
    drill["wtd_j_hedge"] = drill["j_hedge_ret"]  * drill["weight"]

    # Spot returns for regime detection
    drill["spot_ret_i"] = drill["d_spot_i"] / drill["spot_i"]
    drill["spot_ret_j"] = drill["d_spot_j"] / drill["spot_j"]

    daily = (
        drill.groupby("date")
        .agg(
            gross_ret   =("pair_wtd_gross",  "sum"),
            net_ret     =("pair_wtd_net",    "sum"),
            txn_bidask  =("wtd_txn_bidask",  "sum"),
            txn_hedge   =("wtd_txn_hedge",   "sum"),
            txn_total   =("wtd_txn_total",   "sum"),
            n_active    =("signal",          lambda x: (x != 0).sum()),
            n_trades    =("rr_trades",       "sum"),
            i_opt_ret   =("wtd_i_opt",       "sum"),
            i_hedge_ret =("wtd_i_hedge",     "sum"),
            j_opt_ret   =("wtd_j_opt",       "sum"),
            j_hedge_ret =("wtd_j_hedge",     "sum"),
            # Spot trend proxy: cross-sectional mean of daily spot returns
            spot_ret_cs =("spot_ret_i",      "mean"),
        )
    )

    # Cumulative series
    daily["cum_gross"] = (1 + daily["gross_ret"]).cumprod()
    daily["cum_net"]   = (1 + daily["net_ret"]).cumprod()

    # Rolling Sharpe at two horizons
    win = 42
    daily["roll_vol"]   = daily["net_ret"].rolling(win, min_periods=win // 2).std() * np.sqrt(252)
    daily["roll_sharpe"] = (
        daily["net_ret"].rolling(win, min_periods=win // 2).mean()
        / daily["net_ret"].rolling(win, min_periods=win // 2).std()
        * np.sqrt(252)
    )

    win252 = 252
    daily["roll1y_ret"] = (
        daily["net_ret"]
        .rolling(win252, min_periods=win252 // 2)
        .apply(lambda x: (1 + x).prod() - 1, raw=True)
    )
    daily["roll1y_sharpe"] = (
        daily["net_ret"].rolling(win252, min_periods=win252 // 2).mean()
        / daily["net_ret"].rolling(win252, min_periods=win252 // 2).std()
        * np.sqrt(252)
    )

    # Market return: equal-weighted mean of unique ticker daily spot returns
    # (deduplicate tickers that appear in multiple pairs)
    _i = drill[["date", "ticker_i", "spot_ret_i"]].rename(
        columns={"ticker_i": "ticker", "spot_ret_i": "ret"})
    _j = drill[["date", "ticker_j", "spot_ret_j"]].rename(
        columns={"ticker_j": "ticker", "spot_ret_j": "ret"})
    mkt_ret = (
        pd.concat([_i, _j])
        .drop_duplicates(subset=["date", "ticker"])
        .groupby("date")["ret"]
        .mean()
    )
    daily["mkt_ret"] = mkt_ret

    # Vol regime: terciles of rolling MARKET realized vol (not strategy vol)
    daily["market_vol"] = daily["mkt_ret"].rolling(win, min_periods=win // 2).std() * np.sqrt(252)
    vol_q = daily["market_vol"].quantile([1/3, 2/3]).values
    daily["vol_regime"] = pd.cut(
        daily["market_vol"],
        bins=[-np.inf, vol_q[0], vol_q[1], np.inf],
        labels=["Low Vol", "Mid Vol", "High Vol"],
    )

    # Spot trend regime: 21-day rolling direction of the stock basket
    daily["spot_trend"] = np.sign(
        daily["mkt_ret"].rolling(21, min_periods=10).mean()
    ).map({1.0: "Up Market", -1.0: "Down Market", 0.0: "Flat"})

    daily["year"]  = daily.index.year
    daily["month"] = daily.index.month

    return drill, daily


# ── Shared: trade run computation ─────────────────────────────────────────────

def _compute_trade_runs(drill: pd.DataFrame) -> pd.DataFrame:
    """
    Identify consecutive runs of non-zero signal per pair.

    Returns a DataFrame with columns:
        pair, direction, holding_days, trade_pnl
    where trade_pnl is the sum of pair_wtd_net over the run.
    """
    runs = []
    for pair, sub in drill.sort_values("date").groupby("pair"):
        sub = sub.set_index("date").sort_index()
        s   = sub["signal"]
        pnl = sub["pair_wtd_net"]
        in_pos  = False
        run_len = 0
        run_dir = 0
        run_pnl = 0.0

        for sig_val, pnl_val in zip(s, pnl):
            if pd.isna(sig_val) or sig_val == 0:
                if in_pos:
                    runs.append((pair, run_dir, run_len, run_pnl))
                    in_pos  = False
                    run_len = 0
                    run_pnl = 0.0
            else:
                sig_int  = int(sig_val)
                pnl_val  = pnl_val if not pd.isna(pnl_val) else 0.0
                if not in_pos:
                    in_pos  = True
                    run_len = 1
                    run_dir = sig_int
                    run_pnl = pnl_val
                elif sig_int == run_dir:
                    run_len += 1
                    run_pnl += pnl_val
                else:               # direction flip
                    runs.append((pair, run_dir, run_len, run_pnl))
                    run_len = 1
                    run_dir = sig_int
                    run_pnl = pnl_val
        if in_pos:
            runs.append((pair, run_dir, run_len, run_pnl))

    return pd.DataFrame(runs, columns=["pair", "direction", "holding_days", "trade_pnl"])


# ── Section A: Portfolio summary ──────────────────────────────────────────────

def plot_portfolio_summary(daily: pd.DataFrame, plot_dir: Path) -> None:
    """A1–A4: cumulative, distribution (log scale), rolling Sharpe, monthly heatmap."""

    # A1. Cumulative returns
    fig, ax = plt.subplots(figsize=(12, 5))
    (daily["cum_gross"] - 1).plot(ax=ax, label="Gross", color=_SECTION_COLORS["gross"], lw=1.8)
    (daily["cum_net"]   - 1).plot(ax=ax, label="Net",   color=_SECTION_COLORS["net"],   lw=1.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Cumulative Returns (reconstructed from drill)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "A1_cumulative_returns.png", dpi=150)
    plt.close(fig)

    # A1b. Drawdown
    dd = (daily["cum_net"] / daily["cum_net"].cummax()) - 1
    fig, ax = plt.subplots(figsize=(12, 3.5))
    dd.plot(ax=ax, color=_SECTION_COLORS["cost"], lw=1)
    ax.fill_between(dd.index, dd, 0, alpha=0.25, color=_SECTION_COLORS["cost"])
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Net Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "A1b_drawdown.png", dpi=150)
    plt.close(fig)

    # A2. Daily return distribution — log y-scale to reveal fat tails (excess kurtosis)
    from scipy.stats import norm
    r = daily["net_ret"].dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    _, bins, _ = ax.hist(r, bins=200, density=True, alpha=0.65,
                         color=_SECTION_COLORS["net"], label="Net daily ret")
    x = np.linspace(r.min(), r.max(), 500)
    ax.plot(x, norm.pdf(x, r.mean(), r.std()), color="black", lw=1.5, label="Normal fit")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(PCT2)
    ax.set_title("Daily Net Return Distribution (log density — shows fat tails)")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    stats_txt = (
        f"Mean: {r.mean():.3%}  Std: {r.std():.3%}\n"
        f"Skew: {r.skew():.2f}  Kurt: {r.kurtosis():.1f}"
    )
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(plot_dir / "A2_return_distribution.png", dpi=150)
    plt.close(fig)

    # A3. Rolling 42-day Sharpe + volatility
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    daily["roll_sharpe"].plot(ax=ax1, color=_SECTION_COLORS["net"], lw=1.4)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axhline(1, color="grey", lw=0.8, ls="--", label="Sharpe = 1")
    ax1.set_title("Rolling 42-Day Sharpe Ratio")
    ax1.set_ylabel("Sharpe")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    daily["roll_vol"].plot(ax=ax2, color=_SECTION_COLORS["cost"], lw=1.4)
    ax2.yaxis.set_major_formatter(PCT)
    ax2.set_title("Rolling 42-Day Annualised Volatility")
    ax2.set_ylabel("Ann. Vol")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "A3_rolling_sharpe_vol.png", dpi=150)
    plt.close(fig)

    # A4. Monthly P&L heatmap
    try:
        monthly = (
            daily["net_ret"]
            .resample("ME")
            .apply(lambda x: (1 + x).prod() - 1)
        )
        tbl = pd.DataFrame({
            "year":  monthly.index.year,
            "month": monthly.index.month,
            "ret":   monthly.values,
        }).pivot_table(index="month", columns="year", values="ret")

        fig, ax = plt.subplots(figsize=(max(8, len(tbl.columns) * 1.2), 6))
        sns.heatmap(tbl.T, annot=True, fmt=".1%", cmap="RdYlGn", center=0, ax=ax,
                    linewidths=0.5)
        ax.set_title("Monthly Net Returns")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        fig.tight_layout()
        fig.savefig(plot_dir / "A4_monthly_heatmap.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


# ── Section B: Pair attribution ───────────────────────────────────────────────

def plot_pair_attribution(drill: pd.DataFrame, plot_dir: Path) -> None:
    """B1–B3: total P&L per pair, top/bottom pair curves, active count."""

    # Aggregate per pair
    pair_pnl = (
        drill.groupby("pair")[["pair_wtd_net", "pair_wtd_gross", "wtd_txn_total"]]
        .sum()
        .sort_values("pair_wtd_net")
    )

    # B1. Horizontal bar: net P&L per pair
    fig, ax = plt.subplots(figsize=(9, max(5, len(pair_pnl) * 0.4)))
    colors = [_SECTION_COLORS["net"] if v >= 0 else _SECTION_COLORS["cost"]
              for v in pair_pnl["pair_wtd_net"]]
    pair_pnl["pair_wtd_net"].plot(kind="barh", ax=ax, color=colors, edgecolor="white", lw=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(PCT2)
    ax.set_title("Total Net P&L by Pair (sum of weighted daily returns)")
    ax.set_xlabel("Net P&L (fraction of capital)")
    ax.set_ylabel("Pair")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(plot_dir / "B1_pnl_by_pair.png", dpi=150)
    plt.close(fig)

    # B2. Cumulative P&L for top-5 and bottom-5 pairs
    n_show = min(5, len(pair_pnl) // 2)
    top_pairs    = pair_pnl.index[-n_show:]
    bottom_pairs = pair_pnl.index[:n_show]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, group, title, cmap_name in [
        (axes[0], top_pairs,    f"Top {n_show} Pairs",    "Blues"),
        (axes[1], bottom_pairs, f"Bottom {n_show} Pairs", "Reds"),
    ]:
        cmap = plt.get_cmap(cmap_name)
        for idx, pair in enumerate(group):
            sub = drill[drill["pair"] == pair].set_index("date")["pair_wtd_net"].sort_index()
            sub.cumsum().plot(ax=ax, label=pair,
                              color=cmap(0.4 + 0.6 * idx / max(n_show - 1, 1)),
                              lw=1.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.yaxis.set_major_formatter(PCT2)
        ax.set_title(f"Cumulative Net P&L — {title}")
        ax.set_xlabel("Date")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "B2_top_bottom_pairs_cumulative.png", dpi=150)
    plt.close(fig)

    # B3. Active pair count over time
    active_by_day = (
        drill[drill["signal"] != 0]
        .groupby("date")["pair"]
        .nunique()
    )
    fig, ax = plt.subplots(figsize=(12, 3.5))
    active_by_day.plot(ax=ax, color=_SECTION_COLORS["gross"], lw=1, alpha=0.8)
    active_by_day.rolling(21).mean().plot(ax=ax, color="black", lw=1.5,
                                          ls="--", label="21d MA")
    ax.set_title("Active Pair Count Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("# Active Pairs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "B3_active_pair_count.png", dpi=150)
    plt.close(fig)


# ── Section C: Leg decomposition ─────────────────────────────────────────────

def plot_leg_decomposition(drill: pd.DataFrame, daily: pd.DataFrame,
                           plot_dir: Path) -> None:
    """C1–C3: option vs delta hedge, i-leg vs j-leg, annual bar breakdown."""

    # C1. Cumulative: option vs delta hedge (weighted, portfolio-level)
    fig, ax = plt.subplots(figsize=(12, 5))
    opt_cum   = (daily["i_opt_ret"]   + daily["j_opt_ret"]).cumsum()
    hedge_cum = (daily["i_hedge_ret"] + daily["j_hedge_ret"]).cumsum()
    cost_cum  = -daily["txn_total"].cumsum()

    opt_cum.plot(ax=ax,   label="Option P&L",           color=_SECTION_COLORS["opt"],   lw=1.5)
    hedge_cum.plot(ax=ax, label="Delta-hedge P&L",       color=_SECTION_COLORS["hedge"], lw=1.5)
    cost_cum.plot(ax=ax,  label="Transaction costs (−)", color=_SECTION_COLORS["cost"],
                  lw=1.5, ls="--")
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(PCT2)
    ax.set_title("Cumulative P&L Decomposition: Option vs Delta Hedge vs Costs")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "C1_option_vs_hedge_cumulative.png", dpi=150)
    plt.close(fig)

    # C2. Cumulative: i-leg vs j-leg
    i_cum = (daily["i_opt_ret"] + daily["i_hedge_ret"]).cumsum()
    j_cum = (daily["j_opt_ret"] + daily["j_hedge_ret"]).cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    i_cum.plot(ax=ax, label="i-leg (long RR leg)",    color=_SECTION_COLORS["i_leg"], lw=1.5)
    j_cum.plot(ax=ax, label="j-leg (short β·RR leg)", color=_SECTION_COLORS["j_leg"], lw=1.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(PCT2)
    ax.set_title("Cumulative P&L: i-Leg vs j-Leg")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "C2_i_leg_vs_j_leg_cumulative.png", dpi=150)
    plt.close(fig)

    # C3. Annual stacked bar: additive decomposition (simple sum preserves opt+hedge+cost=net)
    daily2 = daily.copy()
    daily2["Option P&L"]        = daily2["i_opt_ret"]  + daily2["j_opt_ret"]
    daily2["Delta-hedge P&L"]   = daily2["i_hedge_ret"] + daily2["j_hedge_ret"]
    daily2["Transaction costs"] = -daily2["txn_total"]   # always ≤ 0

    cols = ["Option P&L", "Delta-hedge P&L", "Transaction costs"]
    # Simple sum: additively correct (opt + hedge + cost = net each day)
    annual = daily2[cols].resample("YE").sum()
    annual.index = annual.index.year

    colors = [_SECTION_COLORS["opt"], _SECTION_COLORS["hedge"], _SECTION_COLORS["cost"]]
    fig, ax = plt.subplots(figsize=(max(7, len(annual) * 1.3), 5))
    annual.plot(
        kind="bar", ax=ax, stacked=True,
        color=colors, edgecolor="white", lw=0.5,
    )
    # Explicitly pair each handle to its label to avoid matplotlib reverse-ordering bug
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors]
    ax.legend(handles, cols)
    ax.yaxis.set_major_formatter(PCT)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Annual P&L Decomposition: Option / Delta-Hedge / Transaction Costs\n"
                 "(simple sum — components add up to net return each day)")
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "C3_annual_leg_decomposition.png", dpi=150)
    plt.close(fig)


# ── Section D: Ticker attribution ─────────────────────────────────────────────

def plot_ticker_attribution(drill: pd.DataFrame, plot_dir: Path) -> None:
    """D1–D2: P&L per ticker across all pairs it appears in."""

    # Compute per-ticker net P&L (option + hedge) for each leg role
    i_contrib = (
        drill.groupby("ticker_i")[["wtd_i_opt", "wtd_i_hedge", "wtd_txn_total"]]
        .sum()
        .rename(columns={"wtd_i_opt": "opt", "wtd_i_hedge": "hedge",
                         "wtd_txn_total": "cost"})
    )
    i_contrib["cost"] *= -0.5   # half cost attributed to each leg
    i_contrib["net"]  = i_contrib["opt"] + i_contrib["hedge"] + i_contrib["cost"]

    j_contrib = (
        drill.groupby("ticker_j")[["wtd_j_opt", "wtd_j_hedge", "wtd_txn_total"]]
        .sum()
        .rename(columns={"wtd_j_opt": "opt", "wtd_j_hedge": "hedge",
                         "wtd_txn_total": "cost"})
    )
    j_contrib["cost"] *= -0.5
    j_contrib["net"]  = j_contrib["opt"] + j_contrib["hedge"] + j_contrib["cost"]

    # Combine: index = ticker, columns = i_net / j_net / total_net
    tickers = sorted(set(i_contrib.index) | set(j_contrib.index))
    ticker_df = pd.DataFrame(index=tickers)
    ticker_df["As i-leg"] = i_contrib["net"].reindex(tickers).fillna(0)
    ticker_df["As j-leg"] = j_contrib["net"].reindex(tickers).fillna(0)
    ticker_df["Total"]    = ticker_df["As i-leg"] + ticker_df["As j-leg"]
    ticker_df = ticker_df.sort_values("Total")

    # D1. Stacked bar: i-leg and j-leg contribution per ticker
    fig, ax = plt.subplots(figsize=(10, max(5, len(ticker_df) * 0.5)))
    ticker_df[["As i-leg", "As j-leg"]].plot(
        kind="barh", ax=ax, stacked=True,
        color=[_SECTION_COLORS["i_leg"], _SECTION_COLORS["j_leg"]],
        edgecolor="white", lw=0.5,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(PCT2)
    ax.set_title("Net P&L Attribution by Ticker\n(i-leg = traded directly; j-leg = hedge leg)")
    ax.set_xlabel("Net P&L (fraction of capital)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(plot_dir / "D1_pnl_by_ticker.png", dpi=150)
    plt.close(fig)

    # D2. Ticker × year net P&L heatmap (option + hedge, both legs)
    try:
        drill2 = drill.copy()
        drill2["year"] = drill2["date"].dt.year

        i_yr = (
            drill2.groupby(["ticker_i", "year"])[["wtd_i_opt", "wtd_i_hedge"]]
            .sum()
            .assign(pnl=lambda d: d["wtd_i_opt"] + d["wtd_i_hedge"])
            ["pnl"]
            .reset_index()
            .rename(columns={"ticker_i": "ticker"})
        )
        j_yr = (
            drill2.groupby(["ticker_j", "year"])[["wtd_j_opt", "wtd_j_hedge"]]
            .sum()
            .assign(pnl=lambda d: d["wtd_j_opt"] + d["wtd_j_hedge"])
            ["pnl"]
            .reset_index()
            .rename(columns={"ticker_j": "ticker"})
        )
        ticker_yr = (
            pd.concat([i_yr, j_yr])
            .groupby(["ticker", "year"])["pnl"]
            .sum()
            .unstack("year")
        )

        fig, ax = plt.subplots(figsize=(max(8, len(ticker_yr.columns) * 1.2),
                                        max(4, len(ticker_yr) * 0.5)))
        sns.heatmap(ticker_yr, annot=True, fmt=".1%", cmap="RdYlGn", center=0, ax=ax,
                    linewidths=0.5)
        ax.set_title("Net P&L by Ticker × Year (option + delta-hedge, weighted)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Ticker")
        fig.tight_layout()
        fig.savefig(plot_dir / "D2_ticker_year_heatmap.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass


# ── Section E: Signal & trade analysis ───────────────────────────────────────

def plot_signal_analysis(drill: pd.DataFrame, plot_dir: Path) -> None:
    """E1–E4: holding periods, per-trade P&L distribution, beta distribution."""

    # Compute trade runs — shared by E1 and E2
    runs_df = _compute_trade_runs(drill)

    # E1. Holding period distribution per pair × direction run
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = range(1, min(runs_df["holding_days"].max() + 2, 121))
    ax.hist(runs_df[runs_df["direction"] ==  1]["holding_days"], bins=bins,
            alpha=0.6, label="Long (+1)",  color=_SECTION_COLORS["net"])
    ax.hist(runs_df[runs_df["direction"] == -1]["holding_days"], bins=bins,
            alpha=0.6, label="Short (−1)", color=_SECTION_COLORS["cost"])
    med = runs_df["holding_days"].median()
    ax.axvline(med, color="black", lw=1.2, ls="--", label=f"Median = {med:.0f}d")
    ax.set_title("Holding Period Distribution")
    ax.set_xlabel("Days in position")
    ax.set_ylabel("Number of trades")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "E1_holding_periods.png", dpi=150)
    plt.close(fig)

    # E2. Per-trade net P&L distribution (long vs short)
    # trade_pnl = cumulative pair_wtd_net over the holding period
    long_pnl  = runs_df[runs_df["direction"] ==  1]["trade_pnl"]
    short_pnl = runs_df[runs_df["direction"] == -1]["trade_pnl"]

    clip_lo, clip_hi = -0.08, 0.08  # clip tails for legible display
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(long_pnl.clip(clip_lo, clip_hi),  bins=60, alpha=0.6,
            label="Long (+1)",  color=_SECTION_COLORS["net"])
    ax.hist(short_pnl.clip(clip_lo, clip_hi), bins=60, alpha=0.6,
            label="Short (−1)", color=_SECTION_COLORS["cost"])
    ax.axvline(0,                color="black",                lw=0.8)
    ax.axvline(long_pnl.median(),  color=_SECTION_COLORS["net"],  lw=1.2, ls="--",
               label=f"Long median {long_pnl.median():.3%}")
    ax.axvline(short_pnl.median(), color=_SECTION_COLORS["cost"], lw=1.2, ls="--",
               label=f"Short median {short_pnl.median():.3%}")
    ax.xaxis.set_major_formatter(PCT2)
    ax.set_title("Per-Trade Net P&L Distribution\n"
                 "(cumulative pair_wtd_net over holding period; tails clipped at ±8%)")
    ax.set_xlabel("Trade Net Return (portfolio-weighted)")
    ax.set_ylabel("Number of trades")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "E2_trade_pnl_distribution.png", dpi=150)
    plt.close(fig)

    # E4. Beta distribution over time (clipped to ±4 for readability)
    active = drill[drill["signal"].notna() & (drill["signal"] != 0)].copy()
    active_betas = active[active["beta"].notna() & (active["beta"].abs() < 4)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(active_betas["beta"], bins=80, color=_SECTION_COLORS["gross"], alpha=0.7,
            edgecolor="white", lw=0.3)
    ax.axvline(active_betas["beta"].median(), color="black", lw=1.2, ls="--",
               label=f"Median β = {active_betas['beta'].median():.2f}")
    ax.set_title("Distribution of Pair Betas (active positions, clipped to ±4)")
    ax.set_xlabel("β (skew-space hedge ratio)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "E4_beta_distribution.png", dpi=150)
    plt.close(fig)


# ── Section F: Regime analysis ────────────────────────────────────────────────

def plot_regime_analysis(daily: pd.DataFrame, plot_dir: Path) -> None:
    """F1–F5: vol-regime performance bars, boxplot, rolling 1-year, annual Sharpe, up/down."""

    regime_colors = {"Low Vol": "#4CAF50", "Mid Vol": "#FFC107", "High Vol": "#F44336"}

    # F1. Performance by vol regime: gross & net ann. return, gross & net Sharpe
    stats_rows = []
    for regime in ["Low Vol", "Mid Vol", "High Vol"]:
        mask = daily["vol_regime"] == regime
        r = daily.loc[mask, "net_ret"].dropna()
        g = daily.loc[mask, "gross_ret"].dropna()
        if len(r) < 10:
            continue
        net_ann    = r.mean() * 252
        net_vol    = r.std() * np.sqrt(252)
        net_sharpe = net_ann / net_vol if net_vol > 0 else np.nan
        gross_ann    = g.mean() * 252
        gross_vol    = g.std() * np.sqrt(252)
        gross_sharpe = gross_ann / gross_vol if gross_vol > 0 else np.nan
        stats_rows.append({
            "Regime": regime,
            "Gross Ann. Return": gross_ann, "Net Ann. Return": net_ann,
            "Gross Sharpe": gross_sharpe,   "Net Sharpe": net_sharpe,
        })

    stats_df = pd.DataFrame(stats_rows).set_index("Regime")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Ann. Return (gross vs net side-by-side)
    ax = axes[0]
    x = np.arange(len(stats_df))
    w = 0.35
    gross_vals = stats_df["Gross Ann. Return"]
    net_vals   = stats_df["Net Ann. Return"]
    ax.bar(x - w/2, gross_vals, w, label="Gross",
           color=_SECTION_COLORS["gross"], edgecolor="black", lw=0.5)
    ax.bar(x + w/2, net_vals,   w, label="Net",
           color=_SECTION_COLORS["net"],   edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df.index, rotation=30)
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Annualised Return")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Sharpe (gross vs net side-by-side)
    ax = axes[1]
    gross_s = stats_df["Gross Sharpe"]
    net_s   = stats_df["Net Sharpe"]
    ax.bar(x - w/2, gross_s, w, label="Gross",
           color=_SECTION_COLORS["gross"], edgecolor="black", lw=0.5)
    ax.bar(x + w/2, net_s,   w, label="Net",
           color=_SECTION_COLORS["net"],   edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df.index, rotation=30)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.yaxis.set_major_formatter(mpl_ticker.FormatStrFormatter("%.2f"))
    ax.set_title("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Strategy Performance by Market Volatility Regime (stock basket realized vol)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(plot_dir / "F1_performance_by_vol_regime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # F2. Box plot: daily net returns by vol regime
    regime_data = daily.dropna(subset=["vol_regime", "net_ret"])
    fig, ax = plt.subplots(figsize=(8, 5))
    regime_data.boxplot(
        column="net_ret", by="vol_regime",
        ax=ax, notch=False,
        boxprops=dict(color="steelblue"),
        medianprops=dict(color="red", lw=2),
        whiskerprops=dict(color="steelblue"),
        capprops=dict(color="steelblue"),
        flierprops=dict(marker=".", ms=3, alpha=0.4),
    )
    ax.yaxis.set_major_formatter(PCT2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Daily Net Returns by Volatility Regime")
    ax.set_xlabel("Vol Regime")
    ax.set_ylabel("Net Return")
    plt.suptitle("")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "F2_returns_by_vol_regime.png", dpi=150)
    plt.close(fig)

    # F3. Rolling 1-year net return + Sharpe (medium-term stability)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    daily["roll1y_ret"].plot(ax=ax1, color=_SECTION_COLORS["net"], lw=1.4)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.yaxis.set_major_formatter(PCT)
    ax1.set_title("Rolling 1-Year Net Return")
    ax1.set_ylabel("1-Year Return")
    ax1.grid(True, alpha=0.3)

    daily["roll1y_sharpe"].plot(ax=ax2, color=_SECTION_COLORS["gross"], lw=1.4)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhline(1, color="grey",  lw=0.8, ls="--", label="Sharpe = 1")
    ax2.set_title("Rolling 1-Year Sharpe Ratio")
    ax2.set_ylabel("Sharpe")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "F3_rolling_1y_return_sharpe.png", dpi=150)
    plt.close(fig)

    # F4. Annual Sharpe bar (gross & net side-by-side)
    _sharpe_fn = lambda r: (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan
    annual_net_sharpe = daily["net_ret"].groupby(daily.index.year).apply(_sharpe_fn)
    annual_gross_sharpe = daily["gross_ret"].groupby(daily.index.year).apply(_sharpe_fn)

    years = annual_net_sharpe.index
    x = np.arange(len(years))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(years) * 1.2), 4.5))
    ax.bar(x - w/2, annual_gross_sharpe, w, label="Gross",
           color=_SECTION_COLORS["gross"], edgecolor="black", lw=0.5)
    ax.bar(x + w/2, annual_net_sharpe,   w, label="Net",
           color=_SECTION_COLORS["net"],   edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="grey", lw=0.8, ls="--", label="Sharpe = 1")
    ax.set_title("Annual Sharpe Ratio (Gross vs Net)")
    ax.set_xlabel("Year")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "F4_annual_sharpe.png", dpi=150)
    plt.close(fig)

    # F5. Up-market vs down-market: gross & net ann. return + Sharpe
    trend_data = daily.dropna(subset=["spot_trend", "net_ret"])
    trend_data = trend_data[trend_data["spot_trend"] != "Flat"]
    if not trend_data.empty:
        trend_stats = []
        for trend in ["Down Market", "Up Market"]:
            mask = trend_data["spot_trend"] == trend
            r = trend_data.loc[mask, "net_ret"].dropna()
            g = trend_data.loc[mask, "gross_ret"].dropna()
            if len(r) < 10:
                continue
            trend_stats.append({
                "Trend": trend,
                "Gross Ann. Return": g.mean() * 252,
                "Net Ann. Return":   r.mean() * 252,
                "Gross Sharpe": (g.mean() / g.std() * np.sqrt(252)) if g.std() > 0 else np.nan,
                "Net Sharpe":   (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
            })
        ts_df = pd.DataFrame(trend_stats).set_index("Trend")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        x = np.arange(len(ts_df))
        w = 0.35

        ax = axes[0]
        ax.bar(x - w/2, ts_df["Gross Ann. Return"], w, label="Gross",
               color=_SECTION_COLORS["gross"], edgecolor="black", lw=0.5)
        ax.bar(x + w/2, ts_df["Net Ann. Return"],   w, label="Net",
               color=_SECTION_COLORS["net"],   edgecolor="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ts_df.index)
        ax.axhline(0, color="black", lw=0.8)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_title("Annualised Return")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1]
        ax.bar(x - w/2, ts_df["Gross Sharpe"], w, label="Gross",
               color=_SECTION_COLORS["gross"], edgecolor="black", lw=0.5)
        ax.bar(x + w/2, ts_df["Net Sharpe"],   w, label="Net",
               color=_SECTION_COLORS["net"],   edgecolor="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ts_df.index)
        ax.axhline(0, color="black", lw=0.8)
        ax.yaxis.set_major_formatter(mpl_ticker.FormatStrFormatter("%.2f"))
        ax.set_title("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Strategy Performance by Spot Market Direction\n"
                     "(21-day rolling mean of stock basket return)",
                     fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(plot_dir / "F5_returns_up_vs_down_market.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── Section G: Cost analysis ──────────────────────────────────────────────────

def plot_cost_analysis(daily: pd.DataFrame, plot_dir: Path) -> None:
    """G1–G3: cumulative cost breakdown, annual cost fraction, gross vs cost scatter."""

    # G1. Cumulative costs: bid-ask vs hedge
    fig, ax = plt.subplots(figsize=(12, 5))
    daily["txn_bidask"].cumsum().plot(ax=ax, label="Bid-ask",    color=_SECTION_COLORS["cost"],  lw=1.5)
    daily["txn_hedge"].cumsum().plot(ax=ax,  label="Delta hedge", color=_SECTION_COLORS["hedge"], lw=1.5)
    daily["txn_total"].cumsum().plot(ax=ax,  label="Total",       color="black", lw=1.8, ls="--")
    ax.yaxis.set_major_formatter(PCT2)
    ax.set_title("Cumulative Transaction Cost Breakdown")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "G1_cumulative_cost_breakdown.png", dpi=150)
    plt.close(fig)

    # G2. Annual cost fraction: txn_cost / |gross return| per year
    annual_cost  = daily["txn_total"].resample("YE").sum()
    annual_gross = daily["gross_ret"].resample("YE").sum()
    # Positive = costs ate into gross; cap display at 300% for legibility
    cost_frac = (annual_cost / annual_gross.abs()).clip(0, 3)
    cost_frac.index = cost_frac.index.year

    fig, ax = plt.subplots(figsize=(max(7, len(cost_frac) * 1.2), 4))
    cost_frac.plot(kind="bar", ax=ax,
                   color=_SECTION_COLORS["cost"], edgecolor="black")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="50% of gross eaten by costs")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":",  label="100% (net = 0)")
    ax.yaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    ax.set_title("Annual Transaction Cost / |Gross Return|\n"
                 "(values > 100% mean costs exceed gross — net negative year)")
    ax.set_xlabel("Year")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "G2_annual_cost_fraction.png", dpi=150)
    plt.close(fig)

    # G3. Scatter: daily gross return vs transaction cost
    fig, ax = plt.subplots(figsize=(7, 5))
    active_days = daily[(daily["n_trades"] > 0) & daily["gross_ret"].notna()]
    ax.scatter(active_days["gross_ret"], active_days["txn_total"],
               alpha=0.3, s=12, color=_SECTION_COLORS["gross"])
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(PCT2)
    ax.yaxis.set_major_formatter(PCT2)
    ax.set_title("Daily Gross Return vs Transaction Cost\n(active trading days only)")
    ax.set_xlabel("Gross Return")
    ax.set_ylabel("Transaction Cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "G3_gross_vs_cost_scatter.png", dpi=150)
    plt.close(fig)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(daily: pd.DataFrame) -> None:
    r    = daily["net_ret"].dropna()
    gross= daily["gross_ret"].dropna()

    total_net   = (1 + r).prod() - 1
    total_gross = (1 + gross).prod() - 1
    ann_net     = (1 + total_net)  ** (252 / len(r)) - 1
    ann_gross   = (1 + total_gross)** (252 / len(gross)) - 1
    sharpe      = r.mean() / r.std() * np.sqrt(252)
    cum_net     = (1 + r).cumprod()
    max_dd      = ((cum_net / cum_net.cummax()) - 1).min()
    calmar      = ann_net / abs(max_dd) if max_dd != 0 else float("nan")
    avg_cost    = daily["txn_total"].mean() * 252

    rows = [
        ("Gross Total Return",    total_gross,  ".2%"),
        ("Net Total Return",      total_net,    ".2%"),
        ("Gross Ann. Return",     ann_gross,    ".2%"),
        ("Net Ann. Return",       ann_net,      ".2%"),
        ("Ann. Volatility",       r.std() * np.sqrt(252), ".2%"),
        ("Sharpe Ratio",          sharpe,       ".3f"),
        ("Max Drawdown",          max_dd,       ".2%"),
        ("Calmar Ratio",          calmar,       ".3f"),
        ("Win Rate",              (r > 0).mean(),".2%"),
        ("Skewness",              r.skew(),     ".3f"),
        ("Kurtosis",              r.kurtosis(), ".3f"),
        ("Ann. Transaction Cost", avg_cost,     ".4%"),
    ]

    print("\n" + "─" * 48)
    print(f"  {'Drill Analytics — Performance Summary':^44}")
    print("─" * 48)
    for label, val, fmt in rows:
        print(f"  {label:<36} {val:{fmt}}")
    print("─" * 48)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Drill analytics from pnl_drill.parquet")
    parser.add_argument("--drill", default="data/pnl_drill.parquet",
                        help="Path to pnl_drill.parquet")
    parser.add_argument("--out",   default="plots/drill_analytics",
                        help="Output directory for plots")
    args = parser.parse_args()

    drill_path = Path(args.drill)
    plot_dir   = Path(args.out)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading drill data from {drill_path} ...")
    drill = load_drill(drill_path)
    print(f"  {len(drill):,} rows  |  {drill['pair'].nunique()} pairs  "
          f"|  {drill['date'].nunique()} dates")

    print("Building daily aggregates and regime labels ...")
    drill, daily = build_derived(drill)

    print_summary(daily)

    print("\nGenerating plots ...")

    print("  [A] Portfolio summary ...")
    plot_portfolio_summary(daily, plot_dir)

    print("  [B] Pair attribution ...")
    plot_pair_attribution(drill, plot_dir)

    print("  [C] Leg decomposition ...")
    plot_leg_decomposition(drill, daily, plot_dir)

    print("  [D] Ticker attribution ...")
    plot_ticker_attribution(drill, plot_dir)

    print("  [E] Signal & trade analysis ...")
    plot_signal_analysis(drill, plot_dir)

    print("  [F] Regime analysis ...")
    plot_regime_analysis(daily, plot_dir)

    print("  [G] Cost analysis ...")
    plot_cost_analysis(daily, plot_dir)

    total = sum(1 for _ in plot_dir.glob("*.png"))
    print(f"\nDone — {total} plots saved to {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
