"""
Signal analytics.

plot_signal_analytics(signals)  — three-panel plot: residual, z-score, position.
trade_log(signals)              — DataFrame of entry/exit/flip events per ticker.
summarize_signals(signals)      — per-ticker summary table.
"""
from __future__ import annotations

from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from interfaces import SignalOutput

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None


def _stack_signal_output(signals: SignalOutput, tickers: List[str] | None = None) -> pd.DataFrame:
    keep = set(tickers) if tickers is not None else None
    parts = []
    for t, df in signals.signal_map.items():
        if keep is not None and t not in keep:
            continue
        if df is None or df.empty:
            continue
        x = df.copy()
        x["ticker"] = t
        x["date"] = pd.to_datetime(x["date"])
        parts.append(x)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def _position_label(position: float, instrument: str) -> str:
    """Human-readable description of what is being traded, e.g. 'long straddle'."""
    direction = "long" if position > 0 else "short"
    return f"{direction} {instrument}"


def trade_log(signals: SignalOutput, tickers: List[str] | None = None) -> pd.DataFrame:
    """Return a row-per-trade-event table with entry, exit, and flip events."""
    df = _stack_signal_output(signals, tickers=tickers)
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "event", "instrument", "position", "zscore", "residual"])

    rows = []
    for t, g in df.groupby("ticker", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        pos = g["position"].to_numpy(dtype=np.float64)
        for i in range(len(g)):
            prev = pos[i - 1] if i > 0 else 0.0
            curr = pos[i]
            if abs(prev) < 1e-10 and abs(curr) > 1e-10:
                event = "entry_long" if curr > 0 else "entry_short"
            elif abs(prev) > 1e-10 and abs(curr) < 1e-10:
                event = "exit"
            elif abs(prev) > 1e-10 and abs(curr) > 1e-10 and np.sign(prev) != np.sign(curr):
                event = "flip_to_long" if curr > 0 else "flip_to_short"
            else:
                continue
            row = g.iloc[i]
            instrument = str(row.get("signal_instrument", "unknown")) if "signal_instrument" in g.columns else "unknown"
            rows.append({
                "date": row["date"],
                "ticker": t,
                "event": event,
                "instrument": instrument,
                "position": float(curr),
                "zscore": float(row.get("zscore", np.nan)),
                "residual": float(row.get("residual", np.nan)),
            })
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "event", "instrument", "position", "zscore", "residual"])
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def summarize_signals(signals: SignalOutput, tickers: List[str] | None = None) -> pd.DataFrame:
    """Per-ticker signal summary: trade counts, average holding, hit rate on residual reversion."""
    df = _stack_signal_output(signals, tickers=tickers)
    if df.empty:
        return pd.DataFrame()

    log = trade_log(signals, tickers=tickers)
    rows = []
    for t, g in df.groupby("ticker", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        pos = g["position"].to_numpy(dtype=np.float64)
        n_days_in = int(np.sum(np.abs(pos) > 1e-10))
        n_entries = int(log[log["ticker"] == t]["event"].str.startswith("entry").sum()) if not log.empty else 0
        instrument = g["signal_instrument"].iloc[0] if "signal_instrument" in g.columns else "unknown"
        zscore = g["zscore"].dropna()
        residual = g["residual"].dropna()
        rows.append({
            "ticker": t,
            "instrument": instrument,
            "n_dates": len(g),
            "n_days_in_position": n_days_in,
            "n_entries": n_entries,
            "mean_abs_zscore": float(zscore.abs().mean()) if len(zscore) else np.nan,
            "max_abs_zscore": float(zscore.abs().max()) if len(zscore) else np.nan,
            "mean_residual": float(residual.mean()) if len(residual) else np.nan,
            "std_residual": float(residual.std()) if len(residual) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def plot_signal_analytics(
    signals: SignalOutput,
    tickers: List[str] | None = None,
    engine: Literal["plotly", "matplotlib"] = "plotly",
) -> pd.DataFrame:
    """Three-panel plot: residual, z-score with thresholds, and position over time.

    The position panel shows what instrument is being traded (straddle / risk_reversal)
    and the direction (long = buy, short = sell) in the hover tooltip and entry/exit markers.

    Returns the trade log DataFrame.
    """
    df = _stack_signal_output(signals, tickers=tickers)
    if df.empty:
        raise ValueError("No signal rows for the requested tickers.")

    log = trade_log(signals, tickers=tickers)
    summary = summarize_signals(signals, tickers=tickers)
    print("Signal summary:")
    print(summary.round(4).to_string(index=False))
    if not log.empty:
        print("\nTrade log:")
        print(log.to_string(index=False))

    active_tickers = sorted(df["ticker"].unique().tolist())

    # Determine the instrument name for the position panel subtitle
    instrument_label = df["signal_instrument"].iloc[0] if "signal_instrument" in df.columns else "options"

    color_map = {
        "entry_long": "green", "entry_short": "red",
        "exit": "grey", "flip_to_long": "lime", "flip_to_short": "orange",
    }
    symbol_map = {
        "entry_long": "triangle-up", "entry_short": "triangle-down",
        "exit": "x", "flip_to_long": "triangle-up", "flip_to_short": "triangle-down",
    }

    if engine == "plotly":
        if go is None:
            raise ImportError("plotly not installed. Use engine='matplotlib'.")

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                "Residual (skew vs benchmark)",
                "Z-Score  [entry ±1.25  |  exit ±0.30]",
                f"Position — {instrument_label}  (+ = long, − = short)",
            ),
            vertical_spacing=0.08,
        )
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, t in enumerate(active_tickers):
            g = df[df["ticker"] == t].sort_values("date")
            color = colors[i % len(colors)]
            instrument = g["signal_instrument"].iloc[0] if "signal_instrument" in g.columns else instrument_label

            # Build position hover: "long straddle (w=+0.75)"
            pos_vals = g["position"].to_numpy(dtype=np.float64)
            pos_labels = [
                _position_label(w, instrument) if abs(w) > 1e-10 else "flat"
                for w in pos_vals
            ]

            fig.add_trace(go.Scatter(
                x=g["date"], y=g["residual"],
                mode="lines", name=t, legendgroup=t,
                line=dict(color=color, width=1.5),
                hovertemplate=f"{t}<br>%{{x|%Y-%m-%d}}<br>residual=%{{y:.4f}}<extra></extra>",
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=g["date"], y=g["zscore"],
                mode="lines", name=t, legendgroup=t, showlegend=False,
                line=dict(color=color, width=1.5),
                hovertemplate=f"{t}<br>%{{x|%Y-%m-%d}}<br>z=%{{y:.3f}}<extra></extra>",
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=g["date"], y=g["position"],
                mode="lines", name=t, legendgroup=t, showlegend=False,
                line=dict(color=color, width=2.0),
                fill="tozeroy",
                text=pos_labels,
                hovertemplate=f"{t}<br>%{{x|%Y-%m-%d}}<br>%{{text}}<br>weight=%{{y:.4f}}<extra></extra>",
            ), row=3, col=1)

        # Z-score threshold lines
        for level, dash in [(1.25, "dash"), (-1.25, "dash"), (0.30, "dot"), (-0.30, "dot")]:
            fig.add_hline(y=level, line=dict(color="grey", width=1, dash=dash), row=2, col=1)

        # Entry/exit markers on both z-score and position panels
        if not log.empty:
            for evt, grp in log.groupby("event"):
                hover_text = [
                    f"{row['ticker']} — {_position_label(row['position'], row['instrument'])}"
                    for _, row in grp.iterrows()
                ]
                shared = dict(
                    mode="markers",
                    name=str(evt),
                    legendgroup=f"evt_{evt}",
                    marker=dict(symbol=symbol_map.get(evt, "circle"), size=10, color=color_map.get(evt, "black")),
                    text=hover_text,
                )
                fig.add_trace(go.Scatter(
                    x=grp["date"], y=grp["zscore"],
                    hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>z=%{y:.3f}<extra></extra>",
                    **shared,
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=grp["date"], y=grp["position"],
                    showlegend=False, legendgroup=f"evt_{evt}",
                    hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>weight=%{y:.4f}<extra></extra>",
                    **{k: v for k, v in shared.items() if k != "legendgroup"},
                ), row=3, col=1)

        fig.update_layout(
            height=820,
            title="Signal Analytics",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.05),
        )
        fig.update_yaxes(title_text="residual", row=1, col=1)
        fig.update_yaxes(title_text="z-score", row=2, col=1)
        fig.update_yaxes(title_text="weight", row=3, col=1)
        fig.show()
        return log

    # --- matplotlib fallback ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, t in enumerate(active_tickers):
        g = df[df["ticker"] == t].sort_values("date")
        c = colors[i % len(colors)]
        axes[0].plot(g["date"], g["residual"], label=t, color=c, linewidth=1.5)
        axes[1].plot(g["date"], g["zscore"], color=c, linewidth=1.5)
        axes[2].plot(g["date"], g["position"], color=c, linewidth=2.0, label=t)
        axes[2].fill_between(g["date"], g["position"], 0, alpha=0.15, color=c)

    for level in [1.25, -1.25]:
        axes[1].axhline(level, color="grey", linestyle="--", linewidth=1)
    for level in [0.30, -0.30]:
        axes[1].axhline(level, color="grey", linestyle=":", linewidth=1)

    mpl_marker_map = {"entry_long": ("^", "green"), "entry_short": ("v", "red"), "exit": ("x", "grey")}
    if not log.empty:
        for evt, (marker, color) in mpl_marker_map.items():
            sub = log[log["event"] == evt]
            if not sub.empty:
                axes[1].scatter(sub["date"], sub["zscore"], marker=marker, color=color, s=80, zorder=5, label=evt)
                axes[2].scatter(sub["date"], sub["position"], marker=marker, color=color, s=80, zorder=5)

    axes[0].set_title("Residual")
    axes[1].set_title("Z-Score  [entry ±1.25 | exit ±0.30]")
    axes[2].set_title(f"Position — {instrument_label}  (+ = long, − = short)")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", ncol=3, fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()
    return log
