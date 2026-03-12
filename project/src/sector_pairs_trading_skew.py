"""
Skew pairs trading strategy — stock vs sector risk reversal spread.

Thesis
------
A stock's implied vol skew is mispriced relative to the sector ETF (XLF).
Rather than betting on mean-reversion of the idiosyncratic component alone,
we trade the spread DIRECTLY:

    Long stock RR  +  Short β * sector RR   (when stock skew is high vs sector)
    Short stock RR +  Long  β * sector RR   (when stock skew is low  vs sector)

This makes the position explicitly delta-neutral in skew-space: P&L comes
from convergence of the stock-sector skew spread, not from the absolute level
of either leg. Sector-wide vol regime shifts are hedged out in the option
positions themselves, not just in the signal.

Spread definition
-----------------
    spread_{i,t} = skew_{i,t} - β_i * skew_{XLF,t}

where β_i is estimated via rolling OLS (60-day window) so no look-ahead.
We trade when this spread deviates significantly from its rolling mean.

P&L model (per pair, per day)
------------------------------
Stock leg (long RR when signal=+1):
    stock_ret_{i,t} = signal_{i,t-1} * Δ(RR_stock_{i,t}) / spot_stock_{i,t-1}

Sector leg (short β * sector RR when signal=+1):
    sector_ret_{i,t} = -signal_{i,t-1} * β_i * Δ(RR_sector_{t}) / spot_sector_{t-1}

Each leg is also delta-hedged daily.

Delta hedge (stock leg):
    stock_hedge_{i,t} = -signal_{i,t-1} * net_delta_stock_{i,t-1}
                        * Δspot_stock_{i,t} / spot_stock_{i,t-1}

Delta hedge (sector leg):
    sector_hedge_{i,t} = signal_{i,t-1} * β_i * net_delta_sector_{t-1}
                         * Δspot_sector_{t} / spot_sector_{t-1}

Total pair return:
    pair_ret_{i,t} = stock_ret + sector_ret + stock_hedge + sector_hedge

Notes
-----
- Mid prices assumed tradable (optimistic; ignores bid-ask spread, but 
accounts for it in transaction costs).
- β is recomputed each day from the rolling OLS, consistent with the signal.
- One unit = one option contract (100 shares) on each leg.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import get_config

config = get_config()


# ── Option leg selection ──────────────────────────────────────────────────────

def select_risk_reversal_legs(
    df: pd.DataFrame,
    tte_target: int = config.tte_target,
    delta_target: float = config.delta_target,
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
            print(f"Missing data for {ticker} on {date}")
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
            print(f"Missing mid price for {ticker} on {date}")
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
            "call_spread":   float(best_call["spread"]),
            "put_mid":       float(best_put["mid_price"]),
            "put_delta":     float(best_put["delta"]),
            "put_tte":       float(best_put["tte_days"]),
            "put_spread":    float(best_put["spread"]),
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


# ── Rolling beta extraction ───────────────────────────────────────────────────

def compute_rolling_betas(
    skew_pivot: pd.DataFrame,
    sector_ticker: str = config.sector_ticker,
    estimation_window: int = config.estimation_window,
) -> pd.DataFrame:
    """
    Return a DataFrame of rolling OLS betas (date × stock_ticker).

    β_{i,t} = cov(skew_i, skew_sector) / var(skew_sector)
    over the trailing `estimation_window` days.
    """
    sec_skew = skew_pivot[sector_ticker]
    stock_tickers = [t for t in skew_pivot.columns if t != sector_ticker]

    betas = pd.DataFrame(np.nan, index=skew_pivot.index, columns=stock_tickers)
    

    for ticker in stock_tickers:
        rolling_cov = skew_pivot[ticker].rolling(estimation_window).cov(sec_skew)
        rolling_var = sec_skew.rolling(estimation_window).var()
        betas[ticker] = rolling_cov / rolling_var

    return betas


# ── Signal construction ───────────────────────────────────────────────────────

def compute_spread_signals(
    skew_pivot: pd.DataFrame,
    betas: pd.DataFrame,
    sector_ticker: str = config.sector_ticker,
    signal_window: int = config.signal_window,
    entry_threshold_mode: str = config.entry_threshold_mode,
    entry_threshold: float = config.entry_threshold,
    entry_threshold_pct: float = config.entry_threshold_pct,
    exit_threshold: float = config.exit_threshold,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute the stock-sector skew spread and generate trading signals.

    Spread:  spread_{i,t} = skew_i - β_i * skew_sector

    Z-score: z_{i,t} = (spread - rolling_mean) / rolling_std

    Entry threshold modes
    --------------------
    "absolute"   : enter when z > entry_threshold (or z < -entry_threshold).
                   Fixed cutoff, same for all tickers and dates.
    "percentile" : at each date t, the upper cutoff is the entry_threshold_pct
                   quantile of the z-score's own expanding history (all
                   observations up to and including t).  The lower cutoff is
                   the symmetric (1 − pct) quantile.  min_periods is set to
                   signal_window so the threshold only activates once the
                   rolling z-score is meaningful — no lookahead bias.

    Signal:
        +1  stock skew high vs sector → sell stock skew, buy sector skew
            (long stock RR, short β * sector RR)
        -1  stock skew low  vs sector → buy  stock skew, sell sector skew
            (short stock RR, long β * sector RR)
         0  flat

    Returns (signals, z_scores, spread_df).
    """
    sec_skew = skew_pivot[sector_ticker]
    stock_tickers = [t for t in skew_pivot.columns if t != sector_ticker]

    # Spread: stock skew minus beta-scaled sector skew
    spread_df = pd.DataFrame(index=skew_pivot.index, columns=stock_tickers, dtype=float)
    for ticker in stock_tickers:
        spread_df[ticker] = skew_pivot[ticker] - betas[ticker] * sec_skew

    # Z-score the spread
    rolling_mean = spread_df.rolling(signal_window).mean()
    rolling_std  = spread_df.rolling(signal_window).std()
    z_scores     = (spread_df - rolling_mean) / rolling_std

    # ── Entry thresholds ──────────────────────────────────────────────────────
    if entry_threshold_mode == "percentile":
        # Expanding quantile computed from history up to t only (no lookahead).
        # min_periods matches signal_window so thresholds activate together with z-scores.
        upper_thresh = z_scores.expanding(min_periods=signal_window).quantile(entry_threshold_pct)
        lower_thresh = z_scores.expanding(min_periods=signal_window).quantile(1.0 - entry_threshold_pct)
        up_arr = upper_thresh.to_numpy()
        lo_arr = lower_thresh.to_numpy()
    elif entry_threshold_mode == "absolute":
        # Broadcast scalar thresholds to arrays for uniform loop logic below
        up_arr = np.full_like(z_scores.to_numpy(), fill_value=entry_threshold)
        lo_arr = np.full_like(z_scores.to_numpy(), fill_value=-entry_threshold)
    else:
        raise ValueError(f"Unknown entry_threshold_mode: {entry_threshold_mode!r}. Use 'absolute' or 'percentile'.")

    # ── Hysteresis signal loop ────────────────────────────────────────────────
    z_arr = z_scores.to_numpy()
    sig   = np.zeros_like(z_arr, dtype=int)

    for t in range(1, len(z_arr)):
        z    = z_arr[t]
        prev = sig[t - 1]
        z_up = up_arr[t]
        z_lo = lo_arr[t]

        # Treat as NaN if z-score or either threshold is unavailable
        nan   = np.isnan(z) | np.isnan(z_up) | np.isnan(z_lo)

        flat  = (prev == 0)  & ~nan
        long  = (prev == 1)  & ~nan
        short = (prev == -1) & ~nan

        sig[t] = np.select(
            [
                nan,
                flat  & (z >  z_up),
                flat  & (z <  z_lo),
                long  & (z >= exit_threshold),
                long  & (z <  z_lo),
                short & (z <= -exit_threshold),
                short & (z >  z_up),
            ],
            [0, 1, -1, 1, -1, -1, 1],
            default=0,
        )

    signals = pd.DataFrame(sig, index=spread_df.index, columns=stock_tickers)
    return signals, z_scores, spread_df


# ── Strategy pipeline ─────────────────────────────────────────────────────────

def run_strategy(
    df: pd.DataFrame,
    tte_days: int = config.tte_target,
    delta_target: float = config.delta_target,
    estimation_window: int = config.estimation_window,
    entry_threshold_mode: str = config.entry_threshold_mode,
    entry_threshold: float = config.entry_threshold,
    entry_threshold_pct: float = config.entry_threshold_pct,
    exit_threshold: float = config.exit_threshold,
    sector_ticker: str = config.sector_ticker,
    skew_path: Path | str = config.skew_path,
) -> dict:
    """
    Full pairs-trading strategy pipeline.

    1. Load pre-computed skew from parquet.
    2. Estimate rolling OLS betas (stock vs sector skew).
    3. Compute skew spread and z-score signals.
    4. Select risk-reversal legs for both the stock and the sector ETF.

    Returns
    -------
    dict with keys:
        skew_pivot, betas, spread_df, z_scores, signals,
        stock_rr_legs, sector_rr_legs
    """
    print("Step 1/4  Loading skew from parquet...")
    skew_df = pd.read_parquet(skew_path)
    skew_pivot = (
        skew_df.reset_index()
        .pivot(index="date", columns="ticker", values="skew")
    )
    skew_pivot.columns.name = None

    print("Step 2/4  Computing rolling betas (stock vs sector skew)...")
    betas = compute_rolling_betas(
        skew_pivot, sector_ticker=sector_ticker, estimation_window=estimation_window
    )

    print("Step 3/4  Computing spread z-scores and signals...")
    signals, z_scores, spread_df = compute_spread_signals(
        skew_pivot, betas,
        sector_ticker=sector_ticker,
        signal_window=estimation_window,
        entry_threshold_mode=entry_threshold_mode,
        entry_threshold=entry_threshold,
        entry_threshold_pct=entry_threshold_pct,
        exit_threshold=exit_threshold,
    )

    print("Step 4/4  Selecting risk-reversal legs (stock + sector)...")
    # Stock legs: all tickers except the sector ETF
    stock_df = df[df["ticker"] != sector_ticker]
    stock_rr_legs = select_risk_reversal_legs(
        stock_df, tte_target=tte_days, delta_target=delta_target
    )

    # Sector leg: XLF options
    sector_df = df[df["ticker"] == sector_ticker]
    sector_rr_legs = select_risk_reversal_legs(
        sector_df, tte_target=tte_days, delta_target=delta_target
    )

    print("Done.")
    return {
        "skew_pivot":     skew_pivot,
        "betas":          betas,
        "spread_df":      spread_df,
        "z_scores":       z_scores,
        "signals":        signals,
        "stock_rr_legs":  stock_rr_legs,
        "sector_rr_legs": sector_rr_legs,
    }


# ── Portfolio simulation ──────────────────────────────────────────────────────

def compute_portfolio_returns(
    signals: pd.DataFrame,
    betas: pd.DataFrame,
    stock_rr_legs: pd.DataFrame,
    sector_rr_legs: pd.DataFrame,
    sector_ticker: str = config.sector_ticker,
    initial_capital: float = config.initial_capital,
    max_position_frac: float = config.max_position_frac,
    transaction_cost_bps: float = config.transaction_cost_bps,
) -> pd.DataFrame:
    """
    Simulate daily P&L from the skew spread pairs trade.

    For each stock i with signal s_{i,t-1}:

      Stock leg option return:
          s_{i,t-1} * Δ(RR_stock_i) / spot_stock_i_{t-1}

      Sector leg option return  (β-scaled, opposite direction):
          -s_{i,t-1} * β_{i,t-1} * Δ(RR_sector) / spot_sector_{t-1}

      Stock delta hedge:
          -s_{i,t-1} * net_delta_stock_{i,t-1} * Δspot_stock_i / spot_stock_i_{t-1}

      Sector delta hedge:
          s_{i,t-1} * β_{i,t-1} * net_delta_sector_{t-1} * Δspot_sector / spot_sector_{t-1}

    Portfolio: equal-weight across active pairs, capped at max_position_frac.
    """
    # ── Pivot stock legs ──────────────────────────────────────────────────────
    stock_rr   = stock_rr_legs["rr_value"].unstack("ticker")
    stock_spot = stock_rr_legs["spot_price"].unstack("ticker")
    stock_delta= stock_rr_legs["net_delta"].unstack("ticker")

    stock_rr.index    = pd.to_datetime(stock_rr.index)
    stock_spot.index  = pd.to_datetime(stock_spot.index)
    stock_delta.index = pd.to_datetime(stock_delta.index)

    # ── Sector leg (single series for XLF) ───────────────────────────────────
    if sector_rr_legs.empty:
        raise ValueError("No sector RR legs found — check XLF option data.")

    # sector_rr_legs is multi-indexed (date, ticker); XLF is the only ticker
    sec_rr_series    = sector_rr_legs["rr_value"].xs(sector_ticker, level="ticker")
    sec_spot_series  = sector_rr_legs["spot_price"].xs(sector_ticker, level="ticker")
    sec_delta_series = sector_rr_legs["net_delta"].xs(sector_ticker, level="ticker")

    sec_rr_series.index    = pd.to_datetime(sec_rr_series.index)
    sec_spot_series.index  = pd.to_datetime(sec_spot_series.index)
    sec_delta_series.index = pd.to_datetime(sec_delta_series.index)

    # ── Common index & tickers ────────────────────────────────────────────────
    common_tickers = [t for t in signals.columns if t in stock_rr.columns]
    common_idx = (
        signals.index
        .intersection(stock_rr.index)
        .intersection(stock_spot.index)
        .intersection(sec_rr_series.index)
        .intersection(sec_spot_series.index)
    )

    signals     = signals.reindex(common_idx)[common_tickers]
    betas_align = betas.reindex(common_idx)[common_tickers]
    stock_rr    = stock_rr.reindex(common_idx)[common_tickers]
    stock_spot  = stock_spot.reindex(common_idx)[common_tickers]
    stock_delta = stock_delta.reindex(common_idx)[common_tickers]

    sec_rr    = sec_rr_series.reindex(common_idx)
    sec_spot  = sec_spot_series.reindex(common_idx)
    sec_delta = sec_delta_series.reindex(common_idx)

    # ── Lagged quantities (signal decided on t-1, executed on t) ─────────────
    lagged_signal = signals.shift(1)
    lagged_beta   = betas_align.shift(1)

    # ── Stock leg P&L ─────────────────────────────────────────────────────────
    d_stock_rr   = stock_rr.diff()
    stock_spot_prev = stock_spot.shift(1)
    stock_opt_ret   = lagged_signal * d_stock_rr / stock_spot_prev

    d_stock_spot    = stock_spot.diff()
    stock_hedge_ret = -lagged_signal * stock_delta.shift(1) * d_stock_spot / stock_spot_prev

    # ── Sector leg P&L (broadcast single XLF series across tickers) ──────────
    d_sec_rr      = sec_rr.diff()
    sec_spot_prev = sec_spot.shift(1)
    sec_delta_prev= sec_delta.shift(1)
    d_sec_spot    = sec_spot.diff()

    # Sector option return: -signal * β * ΔRR_sector / spot_sector_prev
    sec_opt_ret = lagged_signal.multiply(-lagged_beta, axis="columns").multiply(
        (d_sec_rr / sec_spot_prev).values, axis="index"
    )

    # Sector delta hedge: +signal * β * net_delta_sector * Δspot_sector / spot_sector_prev
    sec_hedge_ret = lagged_signal.multiply(lagged_beta, axis="columns").multiply(
        (sec_delta_prev * d_sec_spot / sec_spot_prev).values, axis="index"
    )

    # ── Total pair return ─────────────────────────────────────────────────────
    pair_ret = stock_opt_ret + stock_hedge_ret + sec_opt_ret + sec_hedge_ret

    # ── Portfolio weighting ───────────────────────────────────────────────────
    n_active = (lagged_signal != 0).sum(axis=1)
    weight   = np.minimum(1.0 / n_active.replace(0, np.nan), max_position_frac)

    gross_returns = pair_ret.multiply(weight, axis=0).sum(axis=1)

    # ── Transaction costs ─────────────────────────────────────────────────────
    # Two components, both expressed as fractions of the respective spot price:
    #
    # 1. Option legs: half the bid-ask spread per leg (we trade at mid, so the
    #    true execution cost is 0.5 × spread on each of the call and put).
    #      cost = 0.5 × (call_spread + put_spread) / spot
    #
    # 2. Delta hedge (stock): transaction_cost_bps of the notional stock value.
    #    Notional = |net_delta| × spot, so as a fraction of spot:
    #      cost = transaction_cost_bps / 10_000 × |net_delta|
    #
    # Transition multipliers (number of complete round-trips):
    #   Entry (0 → ±1):  1
    #   Exit  (±1 → 0):  1
    #   Flip  (±1 → ∓1): 2 (close + open)

    prev_sig = signals.shift(1).fillna(0).astype(int)
    entering = (signals != 0) & (prev_sig == 0)
    exiting  = (signals == 0) & (prev_sig != 0)
    flipping = (signals != 0) & (prev_sig != 0) & (signals != prev_sig)

    rr_pair_trades = entering.astype(float) + exiting.astype(float) + flipping.astype(float) * 2

    # ── Option bid-ask cost ───────────────────────────────────────────────────
    stock_call_spread = stock_rr_legs["call_spread"].unstack("ticker")
    stock_put_spread  = stock_rr_legs["put_spread"].unstack("ticker")
    stock_call_spread.index = pd.to_datetime(stock_call_spread.index)
    stock_put_spread.index  = pd.to_datetime(stock_put_spread.index)
    stock_call_spread = stock_call_spread.reindex(common_idx)[common_tickers]
    stock_put_spread  = stock_put_spread.reindex(common_idx)[common_tickers]
    stock_opt_cost = 0.5 * (stock_call_spread + stock_put_spread) / stock_spot

    sec_call_spread_s = sector_rr_legs["call_spread"].xs(sector_ticker, level="ticker")
    sec_put_spread_s  = sector_rr_legs["put_spread"].xs(sector_ticker, level="ticker")
    sec_call_spread_s.index = pd.to_datetime(sec_call_spread_s.index)
    sec_put_spread_s.index  = pd.to_datetime(sec_put_spread_s.index)
    sec_opt_cost = (
        0.5 * (sec_call_spread_s.reindex(common_idx) + sec_put_spread_s.reindex(common_idx))
    ) / sec_spot

    # ── Delta hedge stock cost ────────────────────────────────────────────────
    stock_hedge_cost = (transaction_cost_bps / 10_000) * stock_delta.shift(1).abs()
    sec_hedge_cost   = (transaction_cost_bps / 10_000) * sec_delta.shift(1).abs()

    # ── Total cost per 1 complete RR pair trade ───────────────────────────────
    cost_per_rr_pair = (
        stock_opt_cost
        + lagged_beta.abs().multiply(sec_opt_cost.values, axis="index")
        + stock_hedge_cost
        + lagged_beta.abs().multiply(sec_hedge_cost.values, axis="index")
    )

    txn_cost = (cost_per_rr_pair * rr_pair_trades).multiply(weight, axis=0).sum(axis=1)

    # Diagnostic: RR pair trades per day
    n_trades = rr_pair_trades.sum(axis=1)

    net_returns = gross_returns - txn_cost

    cumulative_gross = (1 + gross_returns).cumprod()
    cumulative_net   = (1 + net_returns).cumprod()

    # Decompose for diagnostics
    stock_total = (stock_opt_ret + stock_hedge_ret).multiply(weight, axis=0).sum(axis=1)
    sector_total= (sec_opt_ret  + sec_hedge_ret).multiply(weight, axis=0).sum(axis=1)

    return pd.DataFrame({
        "gross_returns":    gross_returns,
        "net_returns":      net_returns,
        "stock_leg_ret":    stock_total,
        "sector_leg_ret":   sector_total,
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

    results["Avg Active Positions"]      = metrics_df["active_positions"].mean()
    results["Avg Daily RR Legs Traded"] = metrics_df["n_trades"].mean()
    results["Total Transaction Cost"]   = metrics_df["transaction_cost"].sum()

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    metrics_df: pd.DataFrame,
    z_scores: pd.DataFrame,
    signals: pd.DataFrame,
    spread_df: pd.DataFrame,
    plot_dir: Path | str = config.plot_dir / "pairs_skew",
) -> None:
    """Produce and save backtest plots to plot_dir."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cumulative returns
    fig, ax = plt.subplots(figsize=(12, 5))
    (metrics_df["cumulative_gross"] - 1).plot(ax=ax, label="Gross", linewidth=1.5)
    (metrics_df["cumulative_net"]   - 1).plot(ax=ax, label="Net",   linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    ax.set_title("Cumulative Returns — Skew Pairs Trade (Stock vs Sector RR)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "cumulative_returns.png", dpi=150)
    plt.close(fig)

    # 2. Leg P&L decomposition
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics_df["stock_leg_ret"].cumsum().plot(ax=ax, label="Stock RR legs (cumulative)", linewidth=1.2)
    metrics_df["sector_leg_ret"].cumsum().plot(ax=ax, label="Sector RR legs (cumulative)", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("P&L Decomposition: Stock Leg vs Sector Leg")
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

    # 4. Skew spread (raw) for each stock
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in spread_df.columns:
        spread_df[ticker].dropna().plot(ax=ax, label=ticker, alpha=0.7, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Skew Spread: Stock Skew − β × Sector Skew")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "skew_spread.png", dpi=150)
    plt.close(fig)

    # 5. Z-scores of the spread
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in z_scores.columns:
        z_scores[ticker].dropna().plot(ax=ax, label=ticker, alpha=0.8, linewidth=1)
    ax.axhline( config.entry_threshold, color="grey", linewidth=0.8, linestyle="--",
                label=f"±{config.entry_threshold}σ entry")
    ax.axhline(-config.entry_threshold, color="grey", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Skew Spread Z-Scores")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "z_scores.png", dpi=150)
    plt.close(fig)

    # 6. Signal heatmap
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
    ax.set_title("Trading Signals (+1 sell stock skew / −1 buy stock skew)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(plot_dir / "signals.png", dpi=150)
    plt.close(fig)

    # 7. Monthly returns heatmap
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

    # 8. Annual returns
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
    betas: pd.DataFrame,
    stock_rr_legs: pd.DataFrame,
    sector_rr_legs: pd.DataFrame,
    z_scores: pd.DataFrame,
    spread_df: pd.DataFrame,
    initial_capital: float = config.initial_capital,
    max_position_frac: float = config.max_position_frac,
    transaction_cost_bps: float = config.transaction_cost_bps,
    risk_free_rate: float = 0.0,
    plot_dir: Path | str = config.plot_dir / "pairs_skew",
) -> dict:
    """Run the full backtest: simulate returns → metrics → plots."""
    print("Running backtest...")

    metrics_df = compute_portfolio_returns(
        signals, betas, stock_rr_legs, sector_rr_legs,
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

    plot_results(metrics_df, z_scores, signals, spread_df, plot_dir=plot_dir)

    metrics_path = Path(plot_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path.resolve()}")

    return {"metrics_df": metrics_df, "metrics": metrics, "plot_dir": str(plot_dir)}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    df = pd.read_parquet(config.cleaned_options_path)

    results = run_strategy(df)

    print("\nSignal counts per ticker:")
    print(
        results["signals"]
        .apply(lambda col: col.value_counts())
        .T.fillna(0).astype(int)
    )

    backtest_results = run_backtest(
        signals=results["signals"],
        betas=results["betas"],
        stock_rr_legs=results["stock_rr_legs"],
        sector_rr_legs=results["sector_rr_legs"],
        z_scores=results["z_scores"],
        spread_df=results["spread_df"],
        plot_dir=config.plot_dir / "pairs_skew",
    )

    return {**results, **backtest_results}


if __name__ == "__main__":
    main()
