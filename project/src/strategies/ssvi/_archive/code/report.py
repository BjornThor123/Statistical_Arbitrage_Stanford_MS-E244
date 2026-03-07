from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGY_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation report and plots for tuned SSVI strategy.")
    parser.add_argument(
        "--run-dir",
        default=str(STRATEGY_ROOT / "results" / "current"),
        help="Directory containing portfolio_backtest.csv and backtests/*.csv",
    )
    return parser.parse_args()


def _max_drawdown(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min()) if len(dd) else 0.0


def _rolling_sharpe(x: pd.Series, window: int = 63) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std(ddof=0)
    rs = np.sqrt(252.0) * (mu / sd.replace(0.0, np.nan))
    return rs


def _load_data(run_dir: Path) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], dict]:
    port_path = run_dir / "portfolio_backtest_tuned.csv"
    if not port_path.exists():
        port_path = run_dir / "portfolio_backtest.csv"
    if not port_path.exists():
        raise FileNotFoundError(f"Missing {port_path}")
    portfolio = pd.read_csv(port_path, parse_dates=["date"]).sort_values("date")

    backtests: Dict[str, pd.DataFrame] = {}
    bt_dir = run_dir / "backtests_tuned"
    if not bt_dir.exists():
        bt_dir = run_dir / "backtests"
    if bt_dir.exists():
        for p in sorted(bt_dir.glob("*_backtest*.csv")):
            t = p.stem.replace("_backtest_tuned", "").replace("_backtest", "")
            backtests[t] = pd.read_csv(p, parse_dates=["date"]).sort_values("date")

    signals: Dict[str, pd.DataFrame] = {}
    sig_dir = run_dir / "signals"
    if sig_dir.exists():
        for p in sorted(sig_dir.glob("*_signal.csv")):
            t = p.stem.replace("_signal", "")
            signals[t] = pd.read_csv(p, parse_dates=["date"]).sort_values("date")

    skew_series: Dict[str, pd.DataFrame] = {}
    skew_dir = run_dir / "skew_series"
    if skew_dir.exists():
        for p in sorted(skew_dir.glob("*_skew.csv")):
            t = p.stem.replace("_skew", "")
            skew_series[t] = pd.read_csv(p, parse_dates=["date"]).sort_values("date")

    tune_summary = {}
    ts_path = run_dir / "tuning_summary.json"
    if ts_path.exists():
        tune_summary = json.loads(ts_path.read_text())
    return portfolio, backtests, signals, skew_series, tune_summary


def _plot_timeseries(ax, x, y, title: str, ylabel: str, hline0: bool = False) -> None:
    ax.plot(x, y, lw=1.5)
    if hline0:
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def _make_plots(
    run_dir: Path,
    portfolio: pd.DataFrame,
    backtests: Dict[str, pd.DataFrame],
    signals: Dict[str, pd.DataFrame],
    skew_series: Dict[str, pd.DataFrame],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    figs = run_dir / "report_figures"
    figs.mkdir(exist_ok=True)

    # 1) Cumulative portfolio PnL
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(portfolio["date"], portfolio["portfolio_cum_net_pnl"], lw=2.0)
    ax.set_title("Portfolio Cumulative Net PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Net PnL")
    ax.grid(alpha=0.25)
    p1 = figs / "portfolio_cum_pnl.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    out["portfolio_cum_pnl"] = str(p1)

    # 2) Daily PnL histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(portfolio["portfolio_net_pnl"], bins=40, alpha=0.85, edgecolor="black")
    ax.set_title("Daily Portfolio Net PnL Distribution")
    ax.set_xlabel("Daily Net PnL")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    p2 = figs / "portfolio_daily_pnl_hist.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=180)
    plt.close(fig)
    out["portfolio_daily_pnl_hist"] = str(p2)

    # 3) Rolling Sharpe
    fig, ax = plt.subplots(figsize=(11, 5))
    rs = _rolling_sharpe(portfolio["portfolio_net_pnl"], window=63)
    ax.plot(portfolio["date"], rs, lw=1.7)
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.6)
    ax.set_title("63-Day Rolling Sharpe (Portfolio Net PnL)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Sharpe")
    ax.grid(alpha=0.25)
    p3 = figs / "portfolio_rolling_sharpe.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=180)
    plt.close(fig)
    out["portfolio_rolling_sharpe"] = str(p3)

    # 4) Drawdown curve
    cum = portfolio["portfolio_cum_net_pnl"].to_numpy(dtype=np.float64)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(portfolio["date"], dd, 0.0, alpha=0.75)
    ax.set_title("Portfolio Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    p4 = figs / "portfolio_drawdown.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=180)
    plt.close(fig)
    out["portfolio_drawdown"] = str(p4)

    # 5) Per-name final cumulative net pnl (bar)
    if backtests:
        names = sorted(backtests.keys())
        finals = [float(backtests[n]["cum_net_pnl"].iloc[-1]) if len(backtests[n]) else 0.0 for n in names]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(names, finals)
        ax.set_title("Per-Name Final Cumulative Net PnL")
        ax.set_xlabel("Ticker")
        ax.set_ylabel("Final Cum Net PnL")
        ax.grid(alpha=0.25, axis="y")
        p5 = figs / "per_name_final_cum_pnl.png"
        fig.tight_layout()
        fig.savefig(p5, dpi=180)
        plt.close(fig)
        out["per_name_final_cum_pnl"] = str(p5)

    # 6) Portfolio gross/cost/net decomposition
    if {"portfolio_gross_pnl", "portfolio_cost", "portfolio_net_pnl"}.issubset(portfolio.columns):
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(portfolio["date"], portfolio["portfolio_gross_pnl"], label="Gross", lw=1.2)
        ax.plot(portfolio["date"], -portfolio["portfolio_cost"], label="-Cost", lw=1.2)
        ax.plot(portfolio["date"], portfolio["portfolio_net_pnl"], label="Net", lw=1.6)
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.6)
        ax.set_title("Portfolio PnL Decomposition")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily PnL")
        ax.grid(alpha=0.25)
        ax.legend()
        p6 = figs / "portfolio_pnl_decomposition.png"
        fig.tight_layout()
        fig.savefig(p6, dpi=180)
        plt.close(fig)
        out["portfolio_pnl_decomposition"] = str(p6)

    # 7) Portfolio exposure and breadth
    if {"gross_leverage", "n_names"}.issubset(portfolio.columns):
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        _plot_timeseries(ax[0], portfolio["date"], portfolio["gross_leverage"], "Portfolio Gross Leverage", "Gross Leverage")
        _plot_timeseries(ax[1], portfolio["date"], portfolio["n_names"], "Active Names", "Count")
        p7 = figs / "portfolio_exposure_and_breadth.png"
        fig.tight_layout()
        fig.savefig(p7, dpi=180)
        plt.close(fig)
        out["portfolio_exposure_and_breadth"] = str(p7)

    # 8+) Per-name diagnostics
    for ticker, sig in signals.items():
        if sig.empty:
            continue
        bt = backtests.get(ticker, pd.DataFrame())
        sk = skew_series.get(ticker, pd.DataFrame())
        sec = skew_series.get("XLF", pd.DataFrame())

        # Skew vs sector skew
        if not sk.empty and not sec.empty:
            m = sk[["date", "skew"]].rename(columns={"skew": "stock_skew"}).merge(
                sec[["date", "skew"]].rename(columns={"skew": "sector_skew"}), on="date", how="inner"
            )
            if not m.empty:
                fig, ax = plt.subplots(figsize=(11, 5))
                ax.plot(m["date"], m["stock_skew"], label=f"{ticker} skew", lw=1.2)
                ax.plot(m["date"], m["sector_skew"], label="XLF skew", lw=1.2, alpha=0.9)
                ax.set_title(f"{ticker}: Stock vs Sector Skew")
                ax.set_xlabel("Date")
                ax.set_ylabel("Skew")
                ax.grid(alpha=0.25)
                ax.legend()
                p = figs / f"{ticker}_skew_vs_sector.png"
                fig.tight_layout()
                fig.savefig(p, dpi=180)
                plt.close(fig)
                out[f"{ticker}_skew_vs_sector"] = str(p)

        # Residual and z-score
        if {"residual", "zscore"}.issubset(sig.columns):
            fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            _plot_timeseries(ax[0], sig["date"], sig["residual"], f"{ticker}: Residual", "Residual", hline0=True)
            _plot_timeseries(ax[1], sig["date"], sig["zscore"], f"{ticker}: Z-Score", "Z", hline0=True)
            ax[1].axhline(1.0, color="red", lw=0.8, alpha=0.7)
            ax[1].axhline(-1.0, color="red", lw=0.8, alpha=0.7)
            p = figs / f"{ticker}_residual_zscore.png"
            fig.tight_layout()
            fig.savefig(p, dpi=180)
            plt.close(fig)
            out[f"{ticker}_residual_zscore"] = str(p)

        # Edge vs cost and trade_allowed
        if {"expected_edge", "estimated_unit_cost", "trade_allowed"}.issubset(sig.columns):
            fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            _plot_timeseries(ax[0], sig["date"], sig["expected_edge"], f"{ticker}: Expected Edge", "Edge")
            ax[0].plot(sig["date"], sig["estimated_unit_cost"], lw=1.1, label="Estimated Unit Cost")
            ax[0].legend()
            ax[1].plot(sig["date"], sig["trade_allowed"].astype(float), lw=1.0)
            ax[1].set_ylim(-0.1, 1.1)
            ax[1].set_title(f"{ticker}: Trade Allowed")
            ax[1].set_xlabel("Date")
            ax[1].set_ylabel("0/1")
            ax[1].grid(alpha=0.25)
            p = figs / f"{ticker}_edge_cost_tradegate.png"
            fig.tight_layout()
            fig.savefig(p, dpi=180)
            plt.close(fig)
            out[f"{ticker}_edge_cost_tradegate"] = str(p)

        # Weights/turnover/pnl
        if not bt.empty and {"weight", "turnover", "net_pnl"}.issubset(bt.columns):
            fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
            _plot_timeseries(ax[0], bt["date"], bt["weight"], f"{ticker}: Weight", "Weight", hline0=True)
            _plot_timeseries(ax[1], bt["date"], bt["turnover"], f"{ticker}: Turnover", "Turnover")
            _plot_timeseries(ax[2], bt["date"], bt["cum_net_pnl"], f"{ticker}: Cumulative Net PnL", "Cum Net PnL", hline0=True)
            p = figs / f"{ticker}_weight_turnover_pnl.png"
            fig.tight_layout()
            fig.savefig(p, dpi=180)
            plt.close(fig)
            out[f"{ticker}_weight_turnover_pnl"] = str(p)

    # 9) Signal-direction efficacy (same residual alpha, opposite trade direction)
    if signals:
        rows = []
        for t, sig in signals.items():
            if "zscore" not in sig.columns or "residual" not in sig.columns:
                continue
            s = sig[["date", "zscore", "residual"]].copy().sort_values("date")
            s["ticker"] = t
            s["dresid_next"] = s["residual"].shift(-1) - s["residual"]
            s = s.dropna(subset=["zscore", "dresid_next"])
            if not s.empty:
                s["residual_contrarian_edge"] = s["zscore"] * s["dresid_next"]
                s["residual_trend_edge"] = -s["residual_contrarian_edge"]
                rows.append(s[["date", "ticker", "residual_contrarian_edge", "residual_trend_edge"]])
        if rows:
            e = pd.concat(rows, ignore_index=True)
            g = e.groupby("date", as_index=False).agg(
                residual_contrarian_edge=("residual_contrarian_edge", "mean"),
                residual_trend_edge=("residual_trend_edge", "mean"),
            ).sort_values("date")
            g["cum_residual_contrarian_edge"] = g["residual_contrarian_edge"].cumsum()
            g["cum_residual_trend_edge"] = g["residual_trend_edge"].cumsum()
            fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            ax[0].plot(g["date"], g["residual_contrarian_edge"], label="Residual Contrarian Edge", lw=1.1)
            ax[0].plot(g["date"], g["residual_trend_edge"], label="Residual Trend Edge", lw=1.1)
            ax[0].axhline(0.0, color="black", lw=1.0, alpha=0.6)
            ax[0].set_title("Daily Residual Direction Efficacy")
            ax[0].set_ylabel("Edge Proxy")
            ax[0].grid(alpha=0.25)
            ax[0].legend()
            ax[1].plot(
                g["date"],
                g["cum_residual_contrarian_edge"],
                label="Cum Residual Contrarian Edge",
                lw=1.3,
            )
            ax[1].plot(
                g["date"],
                g["cum_residual_trend_edge"],
                label="Cum Residual Trend Edge",
                lw=1.3,
            )
            ax[1].axhline(0.0, color="black", lw=1.0, alpha=0.6)
            ax[1].set_title("Cumulative Residual Direction Efficacy")
            ax[1].set_xlabel("Date")
            ax[1].set_ylabel("Cumulative Edge")
            ax[1].grid(alpha=0.25)
            ax[1].legend()
            p9 = figs / "signal_direction_efficacy.png"
            fig.tight_layout()
            fig.savefig(p9, dpi=180)
            plt.close(fig)
            out["signal_direction_efficacy"] = str(p9)

    # 10) Surface-evolution diagnostics (if present)
    evo_csv = run_dir / "surface_evolution" / "metrics_over_time.csv"
    if evo_csv.exists():
        try:
            evo = pd.read_csv(evo_csv, parse_dates=["date"]).sort_values("date")
        except Exception:
            evo = pd.DataFrame()
        if not evo.empty:
            if {"rho", "eta", "gamma"}.issubset(evo.columns):
                fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
                _plot_timeseries(ax[0], evo["date"], evo["rho"], "SSVI rho over time", "rho")
                _plot_timeseries(ax[1], evo["date"], evo["eta"], "SSVI eta over time", "eta")
                _plot_timeseries(ax[2], evo["date"], evo["gamma"], "SSVI gamma over time", "gamma")
                p = figs / "surface_ssvi_params_over_time.png"
                fig.tight_layout()
                fig.savefig(p, dpi=180)
                plt.close(fig)
                out["surface_ssvi_params_over_time"] = str(p)

            if {"rmse_implied_volatility", "n_points", "n_maturities"}.issubset(evo.columns):
                fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
                _plot_timeseries(ax[0], evo["date"], evo["rmse_implied_volatility"], "Surface Fit RMSE (IV)", "RMSE")
                _plot_timeseries(ax[1], evo["date"], evo["n_points"], "Option points used", "Count")
                _plot_timeseries(ax[2], evo["date"], evo["n_maturities"], "Maturities used", "Count")
                p = figs / "surface_fit_quality_over_time.png"
                fig.tight_layout()
                fig.savefig(p, dpi=180)
                plt.close(fig)
                out["surface_fit_quality_over_time"] = str(p)

    return out


def _compute_metrics(portfolio: pd.DataFrame, backtests: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    r = portfolio["portfolio_net_pnl"].to_numpy(dtype=np.float64)
    cum = portfolio["portfolio_cum_net_pnl"].to_numpy(dtype=np.float64)
    mu = float(np.mean(r)) if len(r) else 0.0
    sd = float(np.std(r)) if len(r) else 0.0
    sharpe = float(np.sqrt(252.0) * mu / sd) if sd > 1e-12 else 0.0
    mdd = _max_drawdown(cum)
    hit_rate = float(np.mean(r > 0.0)) if len(r) else 0.0

    per_name = {}
    for t, df in backtests.items():
        rr = df["net_pnl"].to_numpy(dtype=np.float64) if len(df) else np.array([])
        cc = df["cum_net_pnl"].to_numpy(dtype=np.float64) if len(df) else np.array([])
        mu_t = float(np.mean(rr)) if len(rr) else 0.0
        sd_t = float(np.std(rr)) if len(rr) else 0.0
        sh_t = float(np.sqrt(252.0) * mu_t / sd_t) if sd_t > 1e-12 else 0.0
        per_name[t] = {
            "n_days": int(len(df)),
            "avg_daily_net_pnl": mu_t,
            "std_daily_net_pnl": sd_t,
            "daily_sharpe_like": sh_t,
            "hit_rate": float(np.mean(rr > 0.0)) if len(rr) else 0.0,
            "max_drawdown": _max_drawdown(cc),
            "final_cum_net_pnl": float(cc[-1]) if len(cc) else 0.0,
            "avg_turnover": float(df["turnover"].mean()) if "turnover" in df.columns and len(df) else 0.0,
        }

    return {
        "portfolio": {
            "n_days": int(len(portfolio)),
            "avg_daily_net_pnl": mu,
            "std_daily_net_pnl": sd,
            "daily_sharpe_like": sharpe,
            "hit_rate": hit_rate,
            "max_drawdown": mdd,
            "final_cum_net_pnl": float(cum[-1]) if len(cum) else 0.0,
            "avg_n_names": float(portfolio["n_names"].mean()) if "n_names" in portfolio.columns and len(portfolio) else 0.0,
        },
        "per_name": per_name,
    }


def _write_markdown(
    run_dir: Path,
    metrics: Dict[str, object],
    figs: Dict[str, str],
    tune_summary: Dict[str, object],
) -> None:
    p = metrics["portfolio"]
    lines = []
    lines.append("# SSVI Strategy Evaluation Report")
    lines.append("")
    lines.append("## Portfolio Metrics")
    lines.append("")
    lines.append(f"- Days: {p['n_days']}")
    lines.append(f"- Avg daily net PnL: {p['avg_daily_net_pnl']:.6f}")
    lines.append(f"- Std daily net PnL: {p['std_daily_net_pnl']:.6f}")
    lines.append(f"- Daily Sharpe-like (annualized): {p['daily_sharpe_like']:.4f}")
    lines.append(f"- Hit rate: {p['hit_rate']:.2%}")
    lines.append(f"- Max drawdown: {p['max_drawdown']:.6f}")
    lines.append(f"- Final cumulative net PnL: {p['final_cum_net_pnl']:.6f}")
    lines.append(f"- Avg active names/day: {p['avg_n_names']:.2f}")
    lines.append("")

    if tune_summary:
        lines.append("## Best Tuned Config")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(
            {
                "best_signal_config": tune_summary.get("best_signal_config"),
                "best_cost_config": tune_summary.get("best_cost_config"),
                "best_portfolio_stats": tune_summary.get("best_portfolio_stats"),
            },
            indent=2,
        ))
        lines.append("```")
        lines.append("")

    lines.append("## Per-Name Metrics")
    lines.append("")
    lines.append("| Ticker | Days | Avg PnL | Std PnL | Sharpe | Hit Rate | Max DD | Final Cum PnL | Avg Turnover |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t, m in sorted(metrics["per_name"].items()):
        lines.append(
            f"| {t} | {m['n_days']} | {m['avg_daily_net_pnl']:.6f} | {m['std_daily_net_pnl']:.6f} | "
            f"{m['daily_sharpe_like']:.4f} | {m['hit_rate']:.2%} | {m['max_drawdown']:.6f} | "
            f"{m['final_cum_net_pnl']:.6f} | {m['avg_turnover']:.4f} |"
        )
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    for name, path in figs.items():
        rel = Path(path).relative_to(run_dir)
        lines.append(f"### {name}")
        lines.append(f"![{name}]({rel.as_posix()})")
        lines.append("")

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    portfolio, backtests, signals, skew_series, tune_summary = _load_data(run_dir)

    figs = _make_plots(run_dir, portfolio, backtests, signals, skew_series)
    metrics = _compute_metrics(portfolio, backtests)
    with (run_dir / "report_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    _write_markdown(run_dir, metrics, figs, tune_summary)

    print("Report generated.")
    print(json.dumps({"run_dir": str(run_dir), "figures": figs, "portfolio": metrics["portfolio"]}, indent=2))


if __name__ == "__main__":
    main()
