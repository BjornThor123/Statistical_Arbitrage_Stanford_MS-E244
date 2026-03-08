"""
Option skew arbitrage strategy — risk reversal P&L model.

Pipeline
--------
1. Load pre-computed skew from parquet (run src.data_cleaning.extract_skew first).
2. For each (ticker, date), select OTM put and call legs (~25-delta, ~15 tte).
3. Construct daily risk-reversal series: RR = call_mid - put_mid.
4. Signal drives direction:
     +1  long  RR  (sold put, bought call) — bet that high skew reverts down
     -1  short RR  (bought put, sold call) — bet that low skew reverts up
5. Daily P&L has two components:
     (a) Option leg P&L : signal * Δ(call_mid - put_mid) / spot
     (b) Delta-hedge P&L: neutralises directional exposure daily
6. Mid price assumed tradable (optimistic; ignores bid-ask spread).

Notes
-----
We normalize P&L by the spot price so returns are dimensionless and
comparable across stocks with different price levels.  One unit of risk
reversal is one option contract (100 shares).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import get_config
from src.construct_skew import compute_idiosyncratic_skew
from src.data_loader import DataLoader

config = get_config()



# ── Option leg selection ──────────────────────────────────────────────────────

def select_risk_reversal_legs(
    df: pd.DataFrame,
    tte_target: int = 15,
    delta_target: float = 0.25,
) -> pd.DataFrame:
    """
    For each (ticker, date), select the OTM put and call legs of a risk
    reversal targeting ~delta_target delta and ~tte_target days to expiry.

    Selection logic (per ticker-date):
      1. OTM call: cp_flag='C', log_moneyness >= 0
         → pick the tte_days closest to tte_target, break ties by delta
           closest to +delta_target.
      2. OTM put : cp_flag='P', log_moneyness < 0
         → pick the tte_days closest to tte_target, break ties by |delta|
           closest to delta_target.

    Returns
    -------
    DataFrame indexed by (date, ticker) with columns:
        call_mid, call_delta, call_tte,
        put_mid,  put_delta,  put_tte,
        spot_price, contract_size,
        rr_value   (= call_mid - put_mid),
        net_delta  (= call_delta - put_delta; put_delta is negative)
    """
    records = []

    for (ticker, date), group in df.groupby(["ticker", "date"]):
        calls = group[(group["cp_flag"] == "C") & (group["log_moneyness"] >= 0)].copy()
        puts  = group[(group["cp_flag"] == "P") & (group["log_moneyness"] <  0)].copy()

        if calls.empty or puts.empty:
            continue

        # Step 1: filter to closest tte_days
        calls["tte_dist"] = (calls["tte_days"] - tte_target).abs()
        puts["tte_dist"]  = (puts["tte_days"]  - tte_target).abs()

        calls = calls[calls["tte_dist"] == calls["tte_dist"].min()]
        puts  = puts[puts["tte_dist"]   == puts["tte_dist"].min()]

        # Step 2: among same tte bucket, pick closest delta
        calls["delta_dist"] = (calls["delta"] - delta_target).abs()
        puts["delta_dist"]  = (puts["delta"].abs() - delta_target).abs()

        best_call = calls.loc[calls["delta_dist"].idxmin()]
        best_put  = puts.loc[puts["delta_dist"].idxmin()]

        # Skip if mid prices are missing
        if pd.isna(best_call["mid_price"]) or pd.isna(best_put["mid_price"]):
            continue

        contract_size = float(group["contract_size"].iloc[0]) if "contract_size" in group.columns else 100.0
        spot = float(group["spot_price"].iloc[0])

        # net_delta of a LONG risk reversal (long call + short put)
        # call_delta > 0, put_delta < 0  →  net = call_delta - put_delta > 0
        net_delta = float(best_call["delta"]) - float(best_put["delta"])

        records.append({
            "date":          date,
            "ticker":        ticker,
            "call_mid":      float(best_call["mid_price"]),
            "call_delta":    float(best_call["delta"]),
            "call_tte":      float(best_call["tte_days"]),
            "put_mid":       float(best_put["mid_price"]),
            "put_delta":     float(best_put["delta"]),
            "put_tte":       float(best_put["tte_days"]),
            "spot_price":    spot,
            "contract_size": contract_size,
            "rr_value":      float(best_call["mid_price"]) - float(best_put["mid_price"]),
            "net_delta":     net_delta,
        })

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index(["date", "ticker"])


# ── Signal construction ───────────────────────────────────────────────────────

def compute_signals(
    resid_df: pd.DataFrame,
    entry_threshold: float = config.entry_threshold,
    exit_threshold: float = config.exit_threshold,
    signal_window: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert idiosyncratic skew residuals into rolling z-scores, then apply a
    two-threshold (hysteresis) rule to reduce signal churn.

    +1  sell skew : z crosses above +entry_threshold (held until z < +exit_threshold)
                    trade: long RR (sell put, buy call)
    -1  buy  skew : z crosses below -entry_threshold (held until z > -exit_threshold)
                    trade: short RR (buy put, sell call)
     0  flat

    Returns (signals, z_scores).
    """
    rolling_mean = resid_df.rolling(signal_window).mean()
    rolling_std  = resid_df.rolling(signal_window).std()
    z_scores     = (resid_df - rolling_mean) / rolling_std

    z_arr = z_scores.to_numpy()
    sig   = np.zeros_like(z_arr, dtype=int)

    for t in range(1, len(z_arr)):
        z    = z_arr[t]
        prev = sig[t - 1]
        nan  = np.isnan(z)

        flat  = (prev == 0)  & ~nan
        long  = (prev == 1)  & ~nan
        short = (prev == -1) & ~nan

        sig[t] = np.select(
            [
                nan,
                flat  & (z >  entry_threshold),   # enter long
                flat  & (z < -entry_threshold),   # enter short
                long  & (z >= exit_threshold),    # hold long
                long  & (z < -entry_threshold),   # flip to short
                short & (z <= -exit_threshold),   # hold short
                short & (z >  entry_threshold),   # flip to long
            ],
            [0, 1, -1, 1, -1, -1, 1],
            default=0,                            # exit (flat + no entry)
        )

    signals = pd.DataFrame(sig, index=resid_df.index, columns=resid_df.columns)
    return signals, z_scores


# ── Strategy pipeline ─────────────────────────────────────────────────────────

def run_strategy(
    df: pd.DataFrame,
    tte_days: int = 15,
    delta_target: float = 0.25,
    estimation_window: int = 60,
    entry_threshold: float = config.entry_threshold,
    exit_threshold: float = config.exit_threshold,
    sector_ticker: str = config.sector_ticker,
    skew_path: Path | str = config.skew_path,
) -> dict:
    """
    Full strategy pipeline:

    1. Load pre-computed skew from parquet (produced by src.data_cleaning.extract_skew).
    2. Compute idiosyncratic skew residuals vs sector ETF.
    3. Generate z-score signals.
    4. Select risk-reversal legs (OTM put + OTM call) per ticker/date.

    Returns
    -------
    dict with keys:
        skew_df, skew_pivot, resid_df, z_scores, signals, rr_legs
    """
    print("Step 1/4  Loading skew from parquet...")
    skew_df = pd.read_parquet(skew_path)

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
        resid_df, entry_threshold=entry_threshold, exit_threshold=exit_threshold,
        signal_window=estimation_window,
    )

    print("Step 4/4  Selecting risk-reversal option legs...")
    rr_legs = select_risk_reversal_legs(df, tte_target=tte_days, delta_target=delta_target)

    print("Done.")
    return {
        "skew_df":    skew_df,
        "skew_pivot": skew_pivot,
        "resid_df":   resid_df,
        "z_scores":   z_scores,
        "signals":    signals,
        "rr_legs":    rr_legs,
    }


# ── Portfolio simulation ──────────────────────────────────────────────────────

def compute_portfolio_returns(
    signals: pd.DataFrame,
    rr_legs: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    max_position_frac: float = 0.20,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Simulate daily P&L from delta-hedged risk reversals.

    P&L model (per position, per day)
    ----------------------------------
    Let RR_t = call_mid_t - put_mid_t  (risk-reversal market value).

    Option leg P&L (per dollar of notional):
        option_ret_{i,t} = signal_{i,t-1} * (RR_{i,t} - RR_{i,t-1}) / spot_{i,t-1}

    Delta-hedge P&L (we short net_delta shares when signal=+1):
        hedge_ret_{i,t} = -signal_{i,t-1} * net_delta_{i,t-1}
                          * (spot_{i,t} - spot_{i,t-1}) / spot_{i,t-1}

    Total daily position return:
        pos_ret_{i,t} = option_ret_{i,t} + hedge_ret_{i,t}

    Portfolio return = equal-weight across active positions, capped at
    max_position_frac per stock.

    Parameters
    ----------
    signals              : (+1 / 0 / -1) wide DataFrame, date × ticker.
    rr_legs              : output of select_risk_reversal_legs (multi-indexed).
    initial_capital      : starting portfolio value in $.
    max_position_frac    : max fraction of portfolio per position.
    transaction_cost_bps : one-way cost in bps on signal changes (use 0 for
                           mid-price baseline since bid-ask already ignored).

    Returns
    -------
    DataFrame with gross/net returns and cumulative performance.
    """
    # Pivot rr_legs to wide format
    rr_pivot    = rr_legs["rr_value"].unstack("ticker")
    spot_pivot  = rr_legs["spot_price"].unstack("ticker")
    delta_pivot = rr_legs["net_delta"].unstack("ticker")

    rr_pivot.index    = pd.to_datetime(rr_pivot.index)
    spot_pivot.index  = pd.to_datetime(spot_pivot.index)
    delta_pivot.index = pd.to_datetime(delta_pivot.index)

    # Align all DataFrames to a common index and ticker set
    stock_tickers = [t for t in signals.columns if t != config.sector_ticker]
    common_tickers = [t for t in stock_tickers if t in rr_pivot.columns]
    common_idx = (
        signals.index
        .intersection(rr_pivot.index)
        .intersection(spot_pivot.index)
    )

    signals    = signals.reindex(common_idx)[common_tickers]
    rr_pivot   = rr_pivot.reindex(common_idx)[common_tickers]
    spot_pivot = spot_pivot.reindex(common_idx)[common_tickers]
    delta_pivot = delta_pivot.reindex(common_idx)[common_tickers]

    # Lagged signal (decided on day t-1, executed on day t)
    lagged_signal = signals.shift(1)

    # Option leg daily return: signal * ΔRR / spot_prev
    delta_rr   = rr_pivot.diff()
    spot_prev  = spot_pivot.shift(1)
    option_ret = lagged_signal * delta_rr / spot_prev

    # Delta-hedge daily return: -signal * net_delta_prev * Δspot / spot_prev
    delta_spot = spot_pivot.diff()
    hedge_ret  = -lagged_signal * delta_pivot.shift(1) * delta_spot / spot_prev

    pos_ret = option_ret + hedge_ret

    # Equal-weight across active positions, capped per stock
    n_active = (lagged_signal != 0).sum(axis=1)
    weight   = np.minimum(1.0 / n_active.replace(0, np.nan), max_position_frac)

    gross_returns = pos_ret.multiply(weight, axis=0).sum(axis=1)

    # Transaction cost on signal changes
    signal_changes = (signals != signals.shift(1)) & (signals != 0)
    n_trades = signal_changes.sum(axis=1)
    txn_cost = (n_trades / n_active.clip(lower=1)) * (transaction_cost_bps / 10_000)

    net_returns = gross_returns - txn_cost

    cumulative_gross = (1 + gross_returns).cumprod()
    cumulative_net   = (1 + net_returns).cumprod()

    return pd.DataFrame({
        "gross_returns":    gross_returns,
        "net_returns":      net_returns,
        "option_ret":       pos_ret.multiply(weight, axis=0).sum(axis=1),
        "hedge_ret":        hedge_ret.multiply(weight, axis=0).sum(axis=1),
        "portfolio_value":  initial_capital * cumulative_net,
        "transaction_cost": txn_cost,
        "active_positions": n_active,
        "n_trades":         n_trades,
        "cumulative_gross": cumulative_gross,
        "cumulative_net":   cumulative_net,
    })


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(metrics_df: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """Compute standard performance metrics from a daily-returns DataFrame."""
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
    plot_dir: Path | str = config.plot_dir / "option_skew",
) -> None:
    """Produce and save backtest plots to plot_dir."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cumulative returns
    fig, ax = plt.subplots(figsize=(12, 5))
    (metrics_df["cumulative_gross"] - 1).plot(ax=ax, label="Gross (option + delta hedge)", linewidth=1.5)
    (metrics_df["cumulative_net"]   - 1).plot(ax=ax, label="Net",  linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    ax.set_title("Cumulative Returns — Option Risk Reversal Strategy")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "cumulative_returns.png", dpi=150)
    plt.close(fig)

    # 2. Option vs delta-hedge P&L decomposition
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics_df["option_ret"].cumsum().plot(ax=ax, label="Option legs (cumulative)", linewidth=1.2)
    metrics_df["hedge_ret"].cumsum().plot(ax=ax, label="Delta hedge (cumulative)", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("P&L Decomposition: Option Legs vs Delta Hedge")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "pnl_decomposition.png", dpi=150)
    plt.close(fig)

    # 3. Drawdown
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

    # 4. Z-scores
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

    # 5. Signal heatmap
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
    ax.set_title("Trading Signals (+1 long RR / -1 short RR)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(plot_dir / "signals.png", dpi=150)
    plt.close(fig)

    # 6. Monthly returns
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
        pass

    # 7. Annual returns
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
    rr_legs: pd.DataFrame,
    z_scores: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    max_position_frac: float = 0.20,
    transaction_cost_bps: float = 0.0,
    risk_free_rate: float = 0.0,
    plot_dir: Path | str = config.plot_dir / "option_skew",
) -> dict:
    """Run the full backtest: simulate returns → metrics → plots."""
    print("Running backtest...")

    metrics_df = compute_portfolio_returns(
        signals, rr_legs,
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
        rr_legs=results["rr_legs"],
        z_scores=results["z_scores"],
        plot_dir=config.plot_dir / "option_skew",
    )

    return {**results, **backtest_results}


if __name__ == "__main__":
    main()
