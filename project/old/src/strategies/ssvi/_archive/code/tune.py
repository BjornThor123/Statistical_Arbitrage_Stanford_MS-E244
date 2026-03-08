from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
import sys
from typing import Dict, List, Tuple

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
    summarize_performance,
)

STRATEGY_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search tuning for SSVI idiosyncratic skew strategy."
    )
    parser.add_argument(
        "--input-dir",
        default=str(STRATEGY_ROOT / "results" / "current"),
        help="Directory containing skew_series/*.csv from run strategy.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(STRATEGY_ROOT / "results" / "current"),
        help="Directory to save tuned results.",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=1800,
        help="Safety cap on number of evaluated parameter combinations.",
    )
    parser.add_argument("--factor-model", choices=["sector", "pca"], default="pca")
    parser.add_argument("--n-pca-factors", type=int, default=2)
    return parser.parse_args()


def _max_drawdown(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min()) if len(dd) else 0.0


def _score_portfolio(portfolio: pd.DataFrame, signal_cfg: SignalConfig) -> Dict[str, float]:
    summary = summarize_performance(portfolio)
    if portfolio.empty:
        return {
            **summary,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
            "score": -1e9,
        }
    pnl = portfolio["portfolio_net_pnl"].to_numpy(dtype=np.float64)
    cum = portfolio["portfolio_cum_net_pnl"].to_numpy(dtype=np.float64)
    mdd = _max_drawdown(cum)
    hit = float(np.mean(pnl > 0.0))

    # Prefer robust and positive outcomes:
    # higher Sharpe and final PnL, lower drawdown and daily volatility.
    score = (
        summary["daily_sharpe_like"]
        + 0.10 * summary["final_cum_net_pnl"]
        + 0.50 * hit
        - 0.50 * abs(mdd)
        - 0.25 * summary["std_daily_net_pnl"]
    )
    # Very sparse signals are discouraged.
    if summary["n_days"] < max(60.0, float(signal_cfg.regression_window + signal_cfg.zscore_window)):
        score -= 3.0

    return {
        **summary,
        "max_drawdown": mdd,
        "hit_rate": hit,
        "score": float(score),
    }


def _load_skew_inputs(input_dir: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    skew_dir = input_dir / "skew_series"
    if not skew_dir.exists():
        raise FileNotFoundError(f"Missing skew series directory: {skew_dir}")
    sector_path = skew_dir / "XLF_skew.csv"
    if not sector_path.exists():
        raise FileNotFoundError(f"Missing sector skew file: {sector_path}")

    sector = pd.read_csv(sector_path, parse_dates=["date"])
    stocks: Dict[str, pd.DataFrame] = {}
    for p in sorted(skew_dir.glob("*_skew.csv")):
        if p.name == "XLF_skew.csv":
            continue
        ticker = p.stem.replace("_skew", "")
        stocks[ticker] = pd.read_csv(p, parse_dates=["date"])
    if not stocks:
        raise ValueError("No stock skew files found.")
    return sector, stocks


def _param_grid() -> Tuple[List[SignalConfig], List[TransactionCostConfig]]:
    signal_cfgs: List[SignalConfig] = []
    for rw, zw, ez, xz, mp, mhd, cs_k, direction in product(
        [40, 60, 90, 120],
        [30, 45, 60, 90],
        [0.75, 1.0, 1.25, 1.5, 2.0],
        [0.10, 0.25, 0.40, 0.60],
        [0.5, 1.0],
        [2, 3, 5],
        [1, 2],
        ["mean_revert", "momentum", "auto"],
    ):
        if xz >= ez:
            continue
        min_obs = max(25, int(0.65 * rw))
        signal_cfgs.append(
            SignalConfig(
                regression_window=rw,
                min_regression_obs=min_obs,
                zscore_window=zw,
                entry_z=ez,
                exit_z=xz,
                max_abs_position=mp,
                min_hold_days=mhd,
                cross_section_top_k=cs_k,
                cross_section_bottom_k=cs_k,
                signal_direction=direction,
                auto_sign_window=63,
            )
        )

    cost_cfgs: List[TransactionCostConfig] = []
    for hs, cm, hd in product(
        [0.0025, 0.005, 0.01, 0.02],
        [0.001, 0.0025, 0.005],
        [0.0, 0.0025, 0.005, 0.01],
    ):
        cost_cfgs.append(
            TransactionCostConfig(
                half_spread_cost=hs,
                impact_cost=0.0,
                commission_cost=cm,
                hedge_drag_gamma=hd,
                hurdle_rate=0.0,
            )
        )
    return signal_cfgs, cost_cfgs


def _run_one_combo(
    signal_cfg: SignalConfig,
    cost_cfg: TransactionCostConfig,
    sector: pd.DataFrame,
    stocks: Dict[str, pd.DataFrame],
    factor_model: str,
    n_pca_factors: int,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, float]]:
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
        if sig.empty:
            continue
        signal_map[ticker] = sig
    portfolio, per_name_bt = backtest_cross_sectional_residual_mean_reversion(
        signal_map=signal_map,
        signal_cfg=signal_cfg,
        cost_cfg=cost_cfg,
    )
    stats = _score_portfolio(portfolio, signal_cfg=signal_cfg)
    return portfolio, per_name_bt, stats


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backtests_tuned").mkdir(exist_ok=True)

    sector, stocks = _load_skew_inputs(input_dir)
    signal_cfgs, cost_cfgs = _param_grid()

    all_combos = [(s, c) for s in signal_cfgs for c in cost_cfgs]
    if len(all_combos) > args.max_combos:
        # deterministic downsample across grid
        idx = np.linspace(0, len(all_combos) - 1, args.max_combos, dtype=int)
        combos = [all_combos[i] for i in idx]
    else:
        combos = all_combos

    print(f"Evaluating {len(combos)} parameter combinations...")
    rows: List[Dict[str, float]] = []
    best_score = -1e18
    best_pack = None

    for i, (sig_cfg, cost_cfg) in enumerate(combos, start=1):
        portfolio, per_name_bt, stats = _run_one_combo(
            sig_cfg,
            cost_cfg,
            sector,
            stocks,
            factor_model=args.factor_model,
            n_pca_factors=args.n_pca_factors,
        )
        row = {
            "combo_id": float(i),
            **{f"signal_{k}": v for k, v in asdict(sig_cfg).items()},
            **{f"cost_{k}": v for k, v in asdict(cost_cfg).items()},
            **stats,
        }
        rows.append(row)
        if stats["score"] > best_score:
            best_score = stats["score"]
            best_pack = (sig_cfg, cost_cfg, portfolio, per_name_bt, stats)

    grid_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    grid_df.to_csv(out_dir / "tuning_grid_search_results.csv", index=False)

    if best_pack is None:
        raise RuntimeError("No valid parameter combination produced a portfolio.")
    best_signal, best_cost, best_portfolio, best_per_name, best_stats = best_pack

    best_portfolio.to_csv(out_dir / "portfolio_backtest_tuned.csv", index=False)
    for ticker, bt in best_per_name.items():
        bt.to_csv(out_dir / "backtests_tuned" / f"{ticker}_backtest_tuned.csv", index=False)

    top10 = grid_df.head(10).to_dict(orient="records")
    summary = {
        "input_dir": str(input_dir),
        "n_combos_evaluated": int(len(combos)),
        "best_signal_config": asdict(best_signal),
        "best_cost_config": asdict(best_cost),
        "factor_model": args.factor_model,
        "n_pca_factors": args.n_pca_factors,
        "best_portfolio_stats": best_stats,
        "top10": top10,
    }
    with (out_dir / "tuning_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Tuning complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
