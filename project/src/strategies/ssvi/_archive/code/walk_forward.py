from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.skew_signal_backtest import (
    SignalConfig,
    TransactionCostConfig,
    backtest_cross_sectional_residual_mean_reversion,
    build_residual_signal,
)
from src.strategies.ssvi.tune import _param_grid, _score_portfolio

STRATEGY_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward tuning and OOS backtest for SSVI skew strategy.")
    parser.add_argument("--input-dir", default=str(STRATEGY_ROOT / "results" / "current"))
    parser.add_argument("--output-dir", default=str(STRATEGY_ROOT / "results" / "current" / "walk_forward"))
    parser.add_argument("--max-combos", type=int, default=120)
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--step-days", type=int, default=63)
    parser.add_argument("--factor-model", choices=["sector", "pca"], default="pca")
    parser.add_argument("--n-pca-factors", type=int, default=2)
    return parser.parse_args()


def _load_skew_inputs(input_dir: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    skew_dir = input_dir / "skew_series"
    sector = pd.read_csv(skew_dir / "XLF_skew.csv", parse_dates=["date"])
    stocks: Dict[str, pd.DataFrame] = {}
    for p in sorted(skew_dir.glob("*_skew.csv")):
        if p.name == "XLF_skew.csv":
            continue
        stocks[p.stem.replace("_skew", "")] = pd.read_csv(p, parse_dates=["date"])
    return sector, stocks


def _prep_combos(max_combos: int) -> List[Tuple[SignalConfig, TransactionCostConfig]]:
    sig_cfgs, cost_cfgs = _param_grid()
    all_combos = [(s, c) for s in sig_cfgs for c in cost_cfgs]
    if len(all_combos) <= max_combos:
        return all_combos
    idx = np.linspace(0, len(all_combos) - 1, max_combos, dtype=int)
    return [all_combos[i] for i in idx]


def _combo_portfolio(
    signal_cfg: SignalConfig,
    cost_cfg: TransactionCostConfig,
    sector: pd.DataFrame,
    stocks: Dict[str, pd.DataFrame],
    factor_model: str,
    n_pca_factors: int,
) -> pd.DataFrame:
    signal_map: Dict[str, pd.DataFrame] = {}
    for ticker, skew_df in stocks.items():
        sig = build_residual_signal(
            skew_df,
            sector,
            signal_cfg=signal_cfg,
            cost_cfg=cost_cfg,
            factor_model=factor_model,
            universe_skews=stocks,
            n_pca_factors=n_pca_factors,
        )
        if not sig.empty:
            signal_map[ticker] = sig
    portfolio, _ = backtest_cross_sectional_residual_mean_reversion(
        signal_map=signal_map,
        signal_cfg=signal_cfg,
        cost_cfg=cost_cfg,
    )
    return portfolio


def _slice_dates(df: pd.DataFrame, dates: List[pd.Timestamp]) -> pd.DataFrame:
    if df.empty:
        return df
    dset = set(pd.Timestamp(x) for x in dates)
    return df[df["date"].isin(dset)].sort_values("date").reset_index(drop=True)


def _wf_windows(
    all_dates: List[pd.Timestamp],
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
    windows = []
    i = 0
    n = len(all_dates)
    while i + train_days + test_days <= n:
        train = all_dates[i : i + train_days]
        test = all_dates[i + train_days : i + train_days + test_days]
        windows.append((train, test))
        i += step_days
    return windows


def _summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "n_days": 0.0,
            "avg_daily_net_pnl": 0.0,
            "std_daily_net_pnl": 0.0,
            "daily_sharpe_like": 0.0,
            "final_cum_net_pnl": 0.0,
        }
    r = df["portfolio_net_pnl"].to_numpy(dtype=np.float64)
    mu = float(np.mean(r))
    sd = float(np.std(r))
    sharpe = float(np.sqrt(252.0) * mu / sd) if sd > 1e-12 else 0.0
    return {
        "n_days": float(len(df)),
        "avg_daily_net_pnl": mu,
        "std_daily_net_pnl": sd,
        "daily_sharpe_like": sharpe,
        "final_cum_net_pnl": float(df["portfolio_cum_net_pnl"].iloc[-1]),
    }


def _plot_wf(out_dir: Path, wf_portfolio: pd.DataFrame, wf_windows_df: pd.DataFrame) -> None:
    if wf_portfolio.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(wf_portfolio["date"], wf_portfolio["portfolio_cum_net_pnl"], lw=1.8)
    ax.set_title("Walk-Forward OOS Cumulative Net PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Net PnL")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "wf_cum_pnl.png", dpi=180)
    plt.close(fig)

    if not wf_windows_df.empty:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(pd.to_datetime(wf_windows_df["test_start"]), wf_windows_df["train_score"], marker="o", lw=1.2, label="Train score")
        ax.plot(pd.to_datetime(wf_windows_df["test_start"]), wf_windows_df["test_score"], marker="o", lw=1.2, label="Test score")
        ax.set_title("Walk-Forward Window Scores")
        ax.set_xlabel("Window Test Start")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "wf_window_scores.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sector, stocks = _load_skew_inputs(input_dir)
    combos = _prep_combos(args.max_combos)
    print(f"Precomputing {len(combos)} combo portfolios...")

    combo_portfolios: List[pd.DataFrame] = []
    for sig_cfg, cost_cfg in combos:
        p = _combo_portfolio(
            sig_cfg,
            cost_cfg,
            sector,
            stocks,
            factor_model=args.factor_model,
            n_pca_factors=args.n_pca_factors,
        )
        combo_portfolios.append(p)

    all_dates = sorted({pd.Timestamp(d) for p in combo_portfolios for d in p["date"].unique()}) if combo_portfolios else []
    windows = _wf_windows(all_dates, args.train_days, args.test_days, args.step_days)
    if not windows:
        raise RuntimeError("No walk-forward windows available with current date range and train/test settings.")

    wf_rows = []
    oos_parts = []
    for w_i, (train_dates, test_dates) in enumerate(windows, start=1):
        best_idx = None
        best_train_score = -1e18
        best_test_score = -1e18
        for i, ((sig_cfg, cost_cfg), p_all) in enumerate(zip(combos, combo_portfolios)):
            p_train = _slice_dates(p_all, train_dates)
            p_test = _slice_dates(p_all, test_dates)
            train_stats = _score_portfolio(p_train, signal_cfg=sig_cfg)
            test_stats = _score_portfolio(p_test, signal_cfg=sig_cfg)
            if train_stats["score"] > best_train_score:
                best_train_score = train_stats["score"]
                best_test_score = test_stats["score"]
                best_idx = i
        if best_idx is None:
            continue

        best_sig, best_cost = combos[best_idx]
        p_test_best = _slice_dates(combo_portfolios[best_idx], test_dates).copy()
        p_test_best["window_id"] = w_i
        p_test_best["selected_combo_id"] = best_idx + 1
        p_test_best["selected_direction"] = best_sig.signal_direction
        oos_parts.append(p_test_best)

        wf_rows.append(
            {
                "window_id": w_i,
                "train_start": str(train_dates[0].date()),
                "train_end": str(train_dates[-1].date()),
                "test_start": str(test_dates[0].date()),
                "test_end": str(test_dates[-1].date()),
                "selected_combo_id": best_idx + 1,
                "selected_direction": best_sig.signal_direction,
                "train_score": float(best_train_score),
                "test_score": float(best_test_score),
                "selected_signal_config": json.dumps(asdict(best_sig)),
                "selected_cost_config": json.dumps(asdict(best_cost)),
            }
        )

    if not oos_parts:
        raise RuntimeError("No OOS slices produced from walk-forward.")

    wf_portfolio = pd.concat(oos_parts, ignore_index=True).sort_values("date").drop_duplicates(subset=["date"], keep="first")
    wf_portfolio = wf_portfolio.reset_index(drop=True)
    wf_portfolio["portfolio_cum_net_pnl"] = wf_portfolio["portfolio_net_pnl"].cumsum()
    wf_windows_df = pd.DataFrame(wf_rows)

    wf_portfolio.to_csv(out_dir / "wf_portfolio_backtest.csv", index=False)
    wf_windows_df.to_csv(out_dir / "wf_windows.csv", index=False)
    _plot_wf(out_dir, wf_portfolio, wf_windows_df)

    summary = {
        "input_dir": str(input_dir),
        "n_combos": int(len(combos)),
        "train_days": int(args.train_days),
        "test_days": int(args.test_days),
        "step_days": int(args.step_days),
        "n_windows": int(len(wf_windows_df)),
        "factor_model": args.factor_model,
        "n_pca_factors": args.n_pca_factors,
        "oos_summary": _summary_metrics(wf_portfolio),
    }
    with (out_dir / "wf_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
