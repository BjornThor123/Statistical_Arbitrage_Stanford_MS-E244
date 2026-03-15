"""
Skew pairs trading strategy — stock vs stock risk reversal spread.

Thesis
------
Two stocks' implied vol skews are mispriced relative to each other.
Rather than anchoring each stock to the sector ETF, we trade ALL pairs of
individual stocks directly:

    Long stock_i RR  +  Short β_ij * stock_j RR   (when i skew is high vs j)
    Short stock_i RR +  Long  β_ij * stock_j RR   (when i skew is low  vs j)

P&L comes from convergence of the pairwise skew spread — sector-wide vol
regime shifts cancel to the extent the two stocks share common factor exposure.

Spread definition
-----------------
    spread_{ij,t} = skew_i_{t} - β_ij * skew_j_{t}

where β_ij is estimated via rolling OLS (60-day window), no look-ahead.
We enumerate all N*(N-1)/2 ordered pairs and trade each independently.

P&L model (per pair (i,j), per day)
-------------------------------------
Stock i leg (long RR when signal=+1):
    ret_i_{t}    = signal_{t-1} * Δ(RR_i_{t}) / spot_i_{t-1}
    hedge_i_{t}  = -signal_{t-1} * net_delta_i_{t-1} * Δspot_i_{t} / spot_i_{t-1}

Stock j leg (short β*RR_j when signal=+1):
    ret_j_{t}    = -signal_{t-1} * β_ij * Δ(RR_j_{t}) / spot_j_{t-1}
    hedge_j_{t}  = +signal_{t-1} * β_ij * net_delta_j_{t-1} * Δspot_j_{t} / spot_j_{t-1}

Total pair return:
    pair_ret_{ij,t} = ret_i + hedge_i + ret_j + hedge_j

Notes
-----
- Pairs are keyed as "TICKER_I|TICKER_J" (i < j alphabetically).
- Signal sign convention: +1 means i skew is rich vs j.
- Mid prices assumed tradable (optimistic; ignores bid-ask spread).
- One unit = one option contract on each leg.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import get_config

config = get_config()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pairs(tickers: List[str]) -> List[Tuple[str, str]]:
    """Return all (i, j) combinations with i < j (alphabetical)."""
    return [(a, b) for a, b in itertools.combinations(sorted(tickers), 2)]


def _pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


# ── Option leg selection ──────────────────────────────────────────────────────

def select_risk_reversal_legs(
    df: pd.DataFrame,
    tte_target: int = config.tte_target,
    delta_target: float = config.delta_target,
) -> pd.DataFrame:
    """
    For each (ticker, date), select the OTM put and call legs of a risk
    reversal targeting ~delta_target delta and ~tte_target days to expiry.

    Returns DataFrame indexed by (date, ticker) with columns:
        call_mid, call_delta, call_tte,
        put_mid,  put_delta,  put_tte,
        spot_price, contract_size,
        rr_value   (= call_mid - put_mid),
        net_delta  (= call_delta - put_delta)
    """
    records = []

    for (ticker, date), group in df.groupby(["ticker", "date"]):
        calls = group[(group["cp_flag"] == "C") & (group["log_moneyness"] >= 0)].copy()
        puts  = group[(group["cp_flag"] == "P") & (group["log_moneyness"] <  0)].copy()

        if calls.empty or puts.empty:
            continue

        calls["tte_dist"] = (calls["tte_days"] - tte_target).abs()
        puts["tte_dist"]  = (puts["tte_days"]  - tte_target).abs()

        calls = calls[calls["tte_dist"] == calls["tte_dist"].min()]
        puts  = puts[puts["tte_dist"]   == puts["tte_dist"].min()]

        calls["delta_dist"] = (calls["delta"] - delta_target).abs()
        puts["delta_dist"]  = (puts["delta"].abs() - delta_target).abs()

        best_call = calls.loc[calls["delta_dist"].idxmin()]
        best_put  = puts.loc[puts["delta_dist"].idxmin()]

        if pd.isna(best_call["mid_price"]) or pd.isna(best_put["mid_price"]):
            continue

        contract_size = float(group["contract_size"].iloc[0]) if "contract_size" in group.columns else 100.0
        spot = float(group["spot_price"].iloc[0])
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


# ── Rolling beta estimation ───────────────────────────────────────────────────

def compute_pair_betas(
    skew_pivot: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    estimation_window: int = config.estimation_window,
    min_periods_frac: float = config.min_periods_frac,
) -> pd.DataFrame:
    """
    For each pair (i, j), estimate rolling OLS beta:
        β_ij = cov(skew_i, skew_j) / var(skew_j)
    over a trailing window of `estimation_window` days.

    Returns DataFrame indexed by date, columns = pair keys "i|j".
    """
    betas = pd.DataFrame(np.nan, index=skew_pivot.index,
                         columns=[_pair_key(a, b) for a, b in pairs])
    min_periods = max(2, int(estimation_window * min_periods_frac))

    for a, b in pairs:
        if a not in skew_pivot.columns or b not in skew_pivot.columns:
            continue
        rolling_cov = skew_pivot[a].rolling(estimation_window, min_periods=min_periods).cov(skew_pivot[b])
        rolling_var = skew_pivot[b].rolling(estimation_window, min_periods=min_periods).var()
        betas[_pair_key(a, b)] = rolling_cov / rolling_var

    return betas


# ── Signal construction ───────────────────────────────────────────────────────

def compute_pair_signals(
    skew_pivot: pd.DataFrame,
    betas: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    signal_window: int = config.signal_window,
    min_periods_frac: float = config.min_periods_frac,
    entry_threshold_mode: str = config.entry_threshold_mode,
    entry_threshold: float = config.entry_threshold,
    entry_threshold_pct: float = config.entry_threshold_pct,
    exit_threshold: float = config.exit_threshold,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise skew spreads and generate trading signals.

    Spread:  spread_{ij,t} = skew_i_t - β_ij_t * skew_j_t
    Z-score: z_{ij,t} = (spread - rolling_mean) / rolling_std

    Signal:
        +1  i skew high vs j → long i RR, short β*j RR
        -1  i skew low  vs j → short i RR, long β*j RR
         0  flat

    Returns (signals, z_scores, spread_df), all with pair-key columns.
    """
    pair_keys = [_pair_key(a, b) for a, b in pairs]
    spread_df = pd.DataFrame(np.nan, index=skew_pivot.index, columns=pair_keys)

    for a, b in pairs:
        key = _pair_key(a, b)
        if a not in skew_pivot.columns or b not in skew_pivot.columns:
            continue
        spread_df[key] = skew_pivot[a] - betas[key] * skew_pivot[b]

    min_periods = max(2, int(signal_window * min_periods_frac))
    rolling_mean = spread_df.rolling(signal_window, min_periods=min_periods).mean()
    rolling_std  = spread_df.rolling(signal_window, min_periods=min_periods).std()
    z_scores     = (spread_df - rolling_mean) / rolling_std

    # ── Entry thresholds ──────────────────────────────────────────────────────
    if entry_threshold_mode == "percentile":
        upper_thresh = z_scores.expanding(min_periods=signal_window).quantile(entry_threshold_pct)
        lower_thresh = z_scores.expanding(min_periods=signal_window).quantile(1.0 - entry_threshold_pct)
        up_arr = upper_thresh.to_numpy()
        lo_arr = lower_thresh.to_numpy()
    elif entry_threshold_mode == "absolute":
        up_arr = np.full_like(z_scores.to_numpy(), fill_value=entry_threshold)
        lo_arr = np.full_like(z_scores.to_numpy(), fill_value=-entry_threshold)
    else:
        raise ValueError(f"Unknown entry_threshold_mode: {entry_threshold_mode!r}.")

    # ── Hysteresis signal loop ────────────────────────────────────────────────
    z_arr = z_scores.to_numpy()
    sig   = np.zeros_like(z_arr, dtype=int)

    for t in range(1, len(z_arr)):
        z    = z_arr[t]
        prev = sig[t - 1]
        z_up = up_arr[t]
        z_lo = lo_arr[t]

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

    signals = pd.DataFrame(sig, index=spread_df.index, columns=pair_keys)
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
    skew_path: Path | str = config.skew_path,
) -> dict:
    """
    Full stock-vs-stock pairs-trading strategy pipeline.

    1. Load pre-computed skew from parquet.
    2. Enumerate all N*(N-1)/2 stock pairs.
    3. Estimate rolling OLS betas for each pair.
    4. Compute skew spread and z-score signals.
    5. Select risk-reversal legs for every individual stock.

    Returns
    -------
    dict with keys:
        skew_pivot, pairs, betas, spread_df, z_scores, signals, stock_rr_legs
    """
    print("Step 1/4  Loading skew from parquet...")
    skew_df = pd.read_parquet(skew_path)
    skew_pivot = (
        skew_df.reset_index()
        .pivot(index="date", columns="ticker", values="skew")
    )
    skew_pivot.columns.name = None

    tickers = list(skew_pivot.columns)
    pairs   = _make_pairs(tickers)
    print(f"          {len(tickers)} tickers → {len(pairs)} pairs")

    print("Step 2/4  Computing rolling pair betas...")
    betas = compute_pair_betas(skew_pivot, pairs, estimation_window=estimation_window)

    print("Step 3/4  Computing spread z-scores and signals...")
    signals, z_scores, spread_df = compute_pair_signals(
        skew_pivot, betas, pairs,
        signal_window=estimation_window,
        entry_threshold_mode=entry_threshold_mode,
        entry_threshold=entry_threshold,
        entry_threshold_pct=entry_threshold_pct,
        exit_threshold=exit_threshold,
    )

    print("Step 4/4  Selecting risk-reversal legs for all stocks...")
    stock_rr_legs = select_risk_reversal_legs(
        df, tte_target=tte_days, delta_target=delta_target
    )

    print("Done.")
    return {
        "skew_pivot":    skew_pivot,
        "pairs":         pairs,
        "betas":         betas,
        "spread_df":     spread_df,
        "z_scores":      z_scores,
        "signals":       signals,
        "stock_rr_legs": stock_rr_legs,
    }


# ── Portfolio simulation ──────────────────────────────────────────────────────

def compute_portfolio_returns(
    signals: pd.DataFrame,
    betas: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    stock_rr_legs: pd.DataFrame,
    initial_capital: float = config.initial_capital,
    max_position_frac: float = config.max_position_frac,
    transaction_cost_bps: float = config.transaction_cost_bps,
    option_cost_mode: str = config.option_cost_mode,
) -> pd.DataFrame:
    """
    Simulate daily P&L from the stock-vs-stock skew spread pairs trade.

    For pair (i, j) with lagged signal s_{t-1}:

      i-leg option return:    s * Δ(RR_i) / spot_i_prev
      i-leg delta hedge:     -s * net_delta_i_prev * Δspot_i / spot_i_prev
      j-leg option return:   -s * β_ij * Δ(RR_j) / spot_j_prev
      j-leg delta hedge:     +s * β_ij * net_delta_j_prev * Δspot_j / spot_j_prev

    Portfolio: equal-weight across active pairs, capped at max_position_frac.
    """
    # ── Pivot single-stock series ─────────────────────────────────────────────
    rr    = stock_rr_legs["rr_value"].unstack("ticker")
    spot  = stock_rr_legs["spot_price"].unstack("ticker")
    delta = stock_rr_legs["net_delta"].unstack("ticker")

    for df_ in (rr, spot, delta):
        df_.index = pd.to_datetime(df_.index)

    # ── Common date index across all needed tickers ───────────────────────────
    needed_tickers = sorted({t for pair in pairs for t in pair})
    available = [t for t in needed_tickers if t in rr.columns]

    common_idx = (
        signals.index
        .intersection(rr.index)
        .intersection(spot.index)
    )

    signals_a  = signals.reindex(common_idx)
    betas_a    = betas.reindex(common_idx)
    rr_a       = rr.reindex(common_idx)[available]
    spot_a     = spot.reindex(common_idx)[available]
    delta_a    = delta.reindex(common_idx)[available]

    # Pre-compute daily diffs and lags (per ticker)
    d_rr    = rr_a.diff()
    d_spot  = spot_a.diff()
    spot_prev  = spot_a.shift(1)
    delta_prev = delta_a.shift(1)

    # ── Accumulate P&L across pairs ───────────────────────────────────────────
    pair_ret_df = pd.DataFrame(0.0, index=common_idx, columns=signals_a.columns)

    for a, b in pairs:
        key = _pair_key(a, b)
        if key not in signals_a.columns or a not in available or b not in available:
            continue

        sig  = signals_a[key].shift(1)      # lagged signal
        beta = betas_a[key].shift(1)        # lagged beta

        # i-leg (stock a)
        i_opt   =  sig * d_rr[a] / spot_prev[a]
        i_hedge = -sig * delta_prev[a] * d_spot[a] / spot_prev[a]

        # j-leg (stock b, β-scaled, opposite direction)
        j_opt   = -sig * beta * d_rr[b] / spot_prev[b]
        j_hedge =  sig * beta * delta_prev[b] * d_spot[b] / spot_prev[b]

        pair_ret_df[key] = i_opt + i_hedge + j_opt + j_hedge

    # ── Portfolio weighting ───────────────────────────────────────────────────
    lagged_signal = signals_a.shift(1)
    n_active = (lagged_signal != 0).sum(axis=1)
    weight   = np.minimum(1.0 / n_active.replace(0, np.nan), max_position_frac)

    gross_returns = pair_ret_df.multiply(weight, axis=0).sum(axis=1)

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

    prev_sig  = signals_a.shift(1).fillna(0).astype(int)
    entering  = (signals_a != 0) & (prev_sig == 0)
    exiting   = (signals_a == 0) & (prev_sig != 0)
    flipping  = (signals_a != 0) & (prev_sig != 0) & (signals_a != prev_sig)
    rr_trades = entering.astype(float) + exiting.astype(float) + flipping.astype(float) * 2

    if option_cost_mode == "spread":
        opt_num_a = stock_rr_legs["call_spread"].unstack("ticker")
        opt_num_b = stock_rr_legs["put_spread"].unstack("ticker")
        for df_ in (opt_num_a, opt_num_b):
            df_.index = pd.to_datetime(df_.index)
        opt_num_a = opt_num_a.reindex(common_idx)[available]
        opt_num_b = opt_num_b.reindex(common_idx)[available]
        def _opt_cost(ticker, beta_lag=None):  # noqa: E306
            return 0.5 * (opt_num_a[ticker] + opt_num_b[ticker]) / spot_a[ticker]
    elif option_cost_mode == "bps":
        opt_num_a = stock_rr_legs["call_mid"].unstack("ticker")
        opt_num_b = stock_rr_legs["put_mid"].unstack("ticker")
        for df_ in (opt_num_a, opt_num_b):
            df_.index = pd.to_datetime(df_.index)
        opt_num_a = opt_num_a.reindex(common_idx)[available]
        opt_num_b = opt_num_b.reindex(common_idx)[available]
        def _opt_cost(ticker, beta_lag=None):  # noqa: E306
            return (transaction_cost_bps / 10_000) * (opt_num_a[ticker] + opt_num_b[ticker]) / spot_a[ticker]
    else:
        raise ValueError(f"Unknown option_cost_mode: {option_cost_mode!r}. Use 'spread' or 'bps'.")

    bidask_cost_df = pd.DataFrame(0.0, index=common_idx, columns=signals_a.columns)
    hedge_cost_df  = pd.DataFrame(0.0, index=common_idx, columns=signals_a.columns)
    for a, b in pairs:
        key = _pair_key(a, b)
        if key not in bidask_cost_df.columns or a not in available or b not in available:
            continue
        beta_lag = betas_a[key].shift(1).abs()
        # Delta hedge cost: bps of notional (|net_delta| × spot) / spot = bps × |net_delta|
        hedge_cost_a = (transaction_cost_bps / 10_000) * delta_prev[a].abs()
        hedge_cost_b = (transaction_cost_bps / 10_000) * delta_prev[b].abs()
        bidask_cost_df[key] = _opt_cost(a) + beta_lag * _opt_cost(b)
        hedge_cost_df[key]  = hedge_cost_a + beta_lag * hedge_cost_b

    txn_bidask = (bidask_cost_df * rr_trades).multiply(weight, axis=0).sum(axis=1)
    txn_hedge  = (hedge_cost_df  * rr_trades).multiply(weight, axis=0).sum(axis=1)
    txn_cost   = txn_bidask + txn_hedge
    n_trades   = rr_trades.sum(axis=1)

    net_returns      = gross_returns - txn_cost
    cumulative_gross = (1 + gross_returns).cumprod()
    cumulative_net   = (1 + net_returns).cumprod()

    return pd.DataFrame({
        "gross_returns":    gross_returns,
        "net_returns":      net_returns,
        "portfolio_value":  initial_capital * cumulative_net,
        "transaction_cost": txn_cost,
        "txn_bidask":       txn_bidask,
        "txn_hedge":        txn_hedge,
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

    results["Avg Active Pairs"]        = metrics_df["active_positions"].mean()
    results["Avg Daily RR Legs Traded"] = metrics_df["n_trades"].mean()
    results["Total Transaction Cost"]   = metrics_df["transaction_cost"].sum()

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    metrics_df: pd.DataFrame,
    z_scores: pd.DataFrame,
    signals: pd.DataFrame,
    spread_df: pd.DataFrame,
    plot_dir: Path | str = config.plot_dir / "pairs_skew_stock",
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
    ax.set_title("Cumulative Returns — Skew Pairs Trade (Stock vs Stock RR)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "cumulative_returns.png", dpi=150)
    plt.close(fig)

    # 2. Drawdown
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

    # 3. Skew spreads (sample up to 10 pairs for legibility)
    sample_pairs = list(spread_df.columns[:10])
    fig, ax = plt.subplots(figsize=(12, 5))
    for key in sample_pairs:
        spread_df[key].dropna().plot(ax=ax, label=key, alpha=0.7, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Skew Spread: skew_i − β·skew_j (first 10 pairs)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "skew_spread.png", dpi=150)
    plt.close(fig)

    # 4. Z-scores (sample up to 10 pairs)
    fig, ax = plt.subplots(figsize=(12, 5))
    for key in sample_pairs:
        z_scores[key].dropna().plot(ax=ax, label=key, alpha=0.8, linewidth=1)
    ax.axhline( config.entry_threshold, color="grey", linewidth=0.8, linestyle="--",
                label=f"±{config.entry_threshold}σ entry")
    ax.axhline(-config.entry_threshold, color="grey", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Skew Spread Z-Scores (first 10 pairs)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "z_scores.png", dpi=150)
    plt.close(fig)

    # 5. Signal heatmap
    fig, ax = plt.subplots(figsize=(14, max(4, len(signals.columns) * 0.4)))
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
    ax.set_title("Trading Signals — Stock vs Stock Pairs (+1 long i / −1 short i)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(plot_dir / "signals.png", dpi=150)
    plt.close(fig)

    # 6. Monthly returns heatmap
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

    # 8. Transaction cost breakdown: bid-ask vs delta hedge
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics_df["txn_bidask"].cumsum().plot(ax=ax, label="Bid-ask spread", linewidth=1.5)
    metrics_df["txn_hedge"].cumsum().plot(ax=ax, label="Delta hedge", linewidth=1.5)
    metrics_df["transaction_cost"].cumsum().plot(ax=ax, label="Total", linewidth=1.5, linestyle="--", color="black")
    ax.yaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    ax.set_title("Cumulative Transaction Cost: Bid-Ask Spread vs Delta Hedge")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "txn_cost_breakdown.png", dpi=150)
    plt.close(fig)

    # 9. Annual Sharpe ratio
    annual_sharpe = (
        metrics_df["net_returns"]
        .groupby(metrics_df["net_returns"].index.year)
        .apply(lambda r: (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan)
    )
    colors_sharpe = ["green" if s > 0 else "red" for s in annual_sharpe]
    fig, ax = plt.subplots(figsize=(max(6, len(annual_sharpe) * 1.2), 4))
    annual_sharpe.plot(kind="bar", ax=ax, color=colors_sharpe, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="grey", linewidth=0.8, linestyle="--", label="Sharpe = 1")
    ax.set_title("Annual Sharpe Ratio (Net Returns)")
    ax.set_xlabel("Year")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "annual_sharpe.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved → {plot_dir.resolve()}")


# ── Backtest orchestration ────────────────────────────────────────────────────

def run_backtest(
    signals: pd.DataFrame,
    betas: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    stock_rr_legs: pd.DataFrame,
    z_scores: pd.DataFrame,
    spread_df: pd.DataFrame,
    initial_capital: float = config.initial_capital,
    max_position_frac: float = config.max_position_frac,
    transaction_cost_bps: float = config.transaction_cost_bps,
    option_cost_mode: str = config.option_cost_mode,
    risk_free_rate: float = 0.0,
    plot_dir: Path | str = config.plot_dir / "pairs_skew_stock",
) -> dict:
    """Run the full backtest: simulate returns → metrics → plots."""
    print("Running backtest...")

    metrics_df = compute_portfolio_returns(
        signals, betas, pairs, stock_rr_legs,
        initial_capital=initial_capital,
        max_position_frac=max_position_frac,
        transaction_cost_bps=transaction_cost_bps,
        option_cost_mode=option_cost_mode,
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

    print(f"\n{len(results['pairs'])} pairs traded.")
    print("\nSignal counts per pair (sample of first 10):")
    sample = list(results["signals"].columns[:10])
    print(
        results["signals"][sample]
        .apply(lambda col: col.value_counts())
        .T.fillna(0).astype(int)
    )

    backtest_results = run_backtest(
        signals=results["signals"],
        betas=results["betas"],
        pairs=results["pairs"],
        stock_rr_legs=results["stock_rr_legs"],
        z_scores=results["z_scores"],
        spread_df=results["spread_df"],
        plot_dir=config.plot_dir / "pairs_skew_stock",
    )

    return {**results, **backtest_results}


if __name__ == "__main__":
    main()
