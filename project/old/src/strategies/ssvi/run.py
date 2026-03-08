from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

STRATEGY_ROOT = Path(__file__).resolve().parent
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
from src.common.option_backtest import (
    OptionBacktestConfig,
    backtest_option_cross_sectional,
)
from src.strategies.ssvi.dashboard import build_dashboard
from src.strategies.ssvi.pipeline import (
    PanelFilters,
    build_skew_series_from_duckdb,
    build_skew_series_from_zip,
    infer_ticker_from_zip_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SSVI-based idiosyncratic skew mean-reversion backtest."
    )
    parser.add_argument(
        "--data-source",
        choices=["zip", "duckdb"],
        default="zip",
        help="Input source for option history.",
    )
    parser.add_argument(
        "--db-path",
        default="data/market_data.duckdb",
        help="DuckDB database path (used when --data-source duckdb).",
    )
    parser.add_argument(
        "--db-table",
        default="options_enriched",
        help="DuckDB table to read option rows from (used when --data-source duckdb).",
    )
    parser.add_argument(
        "--sector-ticker",
        default="XLF",
        help="Sector proxy ticker (used when --data-source duckdb).",
    )
    parser.add_argument(
        "--stock-tickers",
        nargs="+",
        default=None,
        help="Optional explicit stock tickers (used when --data-source duckdb).",
    )
    parser.add_argument("--stock-zips", nargs="+", default=None, help="Optional explicit list of stock option zip files.")
    parser.add_argument(
        "--sector-zip",
        default="data/options/data/options_XLF_etf.zip",
        help="Sector ETF options zip for systematic skew proxy.",
    )
    parser.add_argument(
        "--stock-zip-glob",
        default="data/options/data/options_*.zip",
        help="Glob used to auto-discover stock option zips when --stock-zips is not provided.",
    )
    parser.add_argument("--start-date", default="2006-01-03")
    parser.add_argument("--end-date", default="2008-12-31")
    parser.add_argument("--target-days", type=int, default=30, help="Reference maturity in days for skew slope.")
    parser.add_argument("--k0", type=float, default=0.0, help="Log-moneyness where skew is measured.")
    parser.add_argument("--output-dir", default=str(STRATEGY_ROOT / "results" / "current"))
    parser.add_argument(
        "--snapshots-dir",
        default=str(STRATEGY_ROOT / "results" / "snapshots"),
        help="Optional historical snapshots directory for dashboard context.",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip dashboard generation at the end of the run.",
    )
    parser.add_argument(
        "--calibration-backend",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Backend for SSVI calibration during skew fitting.",
    )

    parser.add_argument("--min-sigma", type=float, default=0.01)
    parser.add_argument("--max-sigma", type=float, default=5.0)
    parser.add_argument("--max-abs-k", type=float, default=1.5)
    parser.add_argument("--min-points-per-day", type=int, default=120)
    parser.add_argument("--min-maturities-per-day", type=int, default=3)

    parser.add_argument("--regression-window", type=int, default=60)
    parser.add_argument("--min-regression-obs", type=int, default=40)
    parser.add_argument("--zscore-window", type=int, default=60)
    parser.add_argument("--entry-z", type=float, default=1.0)
    parser.add_argument("--exit-z", type=float, default=0.25)
    parser.add_argument("--max-abs-position", type=float, default=1.0)
    parser.add_argument("--winsor-quantile", type=float, default=0.02)
    parser.add_argument("--no-mad-zscore", action="store_true")
    parser.add_argument("--min-hold-days", type=int, default=3)
    parser.add_argument("--edge-horizon-days", type=int, default=5)
    parser.add_argument("--edge-cost-buffer", type=float, default=0.25)
    parser.add_argument("--regime-sector-vol-window", type=int, default=21)
    parser.add_argument("--regime-sector-abs-z-max", type=float, default=3.0)
    parser.add_argument("--regime-sector-vol-z-max", type=float, default=2.0)
    parser.add_argument("--min-liquidity-points", type=int, default=150)
    parser.add_argument("--max-fit-rmse-iv", type=float, default=0.60)
    parser.add_argument("--cross-section-top-k", type=int, default=2)
    parser.add_argument("--cross-section-bottom-k", type=int, default=2)
    parser.add_argument("--vol-target-daily", type=float, default=0.02)
    parser.add_argument("--max-gross-leverage", type=float, default=1.5)
    parser.add_argument("--max-name-weight", type=float, default=0.35)
    parser.add_argument("--signal-direction", choices=["mean_revert", "momentum", "auto"], default="auto")
    parser.add_argument("--auto-sign-window", type=int, default=63)
    parser.add_argument("--factor-model", choices=["sector", "pca"], default="pca")
    parser.add_argument("--n-pca-factors", type=int, default=2)

    parser.add_argument("--half-spread-cost", type=float, default=0.02)
    parser.add_argument("--impact-cost", type=float, default=0.0)
    parser.add_argument("--commission-cost", type=float, default=0.005)
    parser.add_argument("--hedge-drag-gamma", type=float, default=0.01)
    parser.add_argument("--hurdle-rate", type=float, default=0.0)
    parser.add_argument(
        "--backtest-mode",
        choices=["idealized", "option"],
        default="option",
        help="PnL engine: idealized residual PnL or realistic option portfolio simulator.",
    )
    parser.add_argument("--rr-k-put", type=float, default=-0.10)
    parser.add_argument("--rr-k-call", type=float, default=0.10)
    parser.add_argument("--roll-threshold-days", type=float, default=5.0)
    parser.add_argument("--option-half-spread-iv", type=float, default=0.005)
    parser.add_argument("--hedge-half-spread-pct", type=float, default=0.0005)
    return parser.parse_args()


def _build_filters(args: argparse.Namespace) -> PanelFilters:
    return PanelFilters(
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        max_abs_k=args.max_abs_k,
        min_points_per_day=args.min_points_per_day,
        min_maturities_per_day=args.min_maturities_per_day,
    )


def _build_signal_config(args: argparse.Namespace) -> SignalConfig:
    return SignalConfig(
        regression_window=args.regression_window,
        min_regression_obs=args.min_regression_obs,
        zscore_window=args.zscore_window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        max_abs_position=args.max_abs_position,
        winsor_quantile=args.winsor_quantile,
        use_mad_zscore=(not args.no_mad_zscore),
        min_hold_days=args.min_hold_days,
        edge_horizon_days=args.edge_horizon_days,
        edge_cost_buffer=args.edge_cost_buffer,
        regime_sector_vol_window=args.regime_sector_vol_window,
        regime_sector_abs_z_max=args.regime_sector_abs_z_max,
        regime_sector_vol_z_max=args.regime_sector_vol_z_max,
        min_liquidity_points=args.min_liquidity_points,
        max_fit_rmse_iv=args.max_fit_rmse_iv,
        cross_section_top_k=args.cross_section_top_k,
        cross_section_bottom_k=args.cross_section_bottom_k,
        vol_target_daily=args.vol_target_daily,
        max_gross_leverage=args.max_gross_leverage,
        max_name_weight=args.max_name_weight,
        signal_direction=args.signal_direction,
        auto_sign_window=args.auto_sign_window,
    )


def _build_cost_config(args: argparse.Namespace) -> TransactionCostConfig:
    return TransactionCostConfig(
        half_spread_cost=args.half_spread_cost,
        impact_cost=args.impact_cost,
        commission_cost=args.commission_cost,
        hedge_drag_gamma=args.hedge_drag_gamma,
        hurdle_rate=args.hurdle_rate,
    )


def _fit_skew_for_zip(zip_path: str, args: argparse.Namespace, filters: PanelFilters) -> pd.DataFrame:
    target_t = args.target_days / 365.0
    return build_skew_series_from_zip(
        zip_path=zip_path,
        start_date=args.start_date,
        end_date=args.end_date,
        target_t=target_t,
        k0=args.k0,
        filters=filters,
        calibration_backend=args.calibration_backend,
    )


def _fit_skew_for_ticker(ticker: str, args: argparse.Namespace, filters: PanelFilters) -> pd.DataFrame:
    target_t = args.target_days / 365.0
    return build_skew_series_from_duckdb(
        db_path=args.db_path,
        ticker=ticker,
        table=args.db_table,
        start_date=args.start_date,
        end_date=args.end_date,
        target_t=target_t,
        k0=args.k0,
        filters=filters,
        calibration_backend=args.calibration_backend,
    )


def _build_option_backtest_config(args: argparse.Namespace) -> OptionBacktestConfig:
    return OptionBacktestConfig(
        k_put=args.rr_k_put,
        k_call=args.rr_k_call,
        target_t=args.target_days / 365.0,
        roll_threshold_t=args.roll_threshold_days / 365.0,
        option_half_spread_iv=args.option_half_spread_iv,
        hedge_half_spread_pct=args.hedge_half_spread_pct,
    )


def _resolve_data_path(p: str) -> str:
    candidate = Path(p)
    if candidate.exists():
        return str(candidate)
    for base in [PROJECT_ROOT, PROJECT_ROOT.parent, Path()]:
        alt = base / p
        if alt.exists():
            return str(alt)
    return str(candidate)


def _discover_stock_zips(glob_pattern: str, sector_zip: str) -> List[str]:
    candidates: List[Path] = []
    for base in [Path(), PROJECT_ROOT, PROJECT_ROOT.parent]:
        candidates.extend(sorted(base.glob(glob_pattern)))
    # dedupe preserving order
    seen = set()
    uniq: List[Path] = []
    for p in candidates:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)

    out = []
    sector_abs = str(Path(sector_zip).resolve())
    for p in uniq:
        s = str(p.resolve())
        if s == sector_abs:
            continue
        name = p.name.lower()
        if "_etf" in name:
            continue
        out.append(str(p))
    return out


def _discover_stock_tickers_from_duckdb(db_path: str, table: str, sector_ticker: str) -> List[str]:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "duckdb is required for --data-source duckdb. Install with `pip install duckdb`."
        ) from exc

    safe_table = table.strip()
    if not safe_table or not safe_table.replace("_", "").isalnum():
        raise ValueError(f"Unsafe DuckDB table name: {table!r}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            f"""
            SELECT DISTINCT UPPER(ticker) AS ticker
            FROM {safe_table}
            WHERE ticker IS NOT NULL
            ORDER BY ticker
            """
        ).fetchall()
    finally:
        con.close()

    sector = sector_ticker.upper()
    out = [str(r[0]).upper() for r in rows if r and r[0] and str(r[0]).upper() != sector]
    return out


def main() -> None:
    args = parse_args()
    if args.data_source == "duckdb":
        args.db_path = _resolve_data_path(args.db_path)
        args.sector_ticker = args.sector_ticker.upper()
        if args.stock_tickers is None:
            args.stock_tickers = _discover_stock_tickers_from_duckdb(
                db_path=args.db_path,
                table=args.db_table,
                sector_ticker=args.sector_ticker,
            )
        else:
            args.stock_tickers = [t.upper() for t in args.stock_tickers]
        if not args.stock_tickers:
            raise RuntimeError("No stock tickers available for DuckDB run.")
        print(
            f"Using DuckDB source {args.db_path} ({args.db_table}) with "
            f"{len(args.stock_tickers)} stock tickers."
        )
    else:
        args.sector_zip = _resolve_data_path(args.sector_zip)
        if args.stock_zips is None:
            auto = _discover_stock_zips(args.stock_zip_glob, args.sector_zip)
            args.stock_zips = auto
        args.stock_zips = [_resolve_data_path(p) for p in args.stock_zips]
        print(f"Using {len(args.stock_zips)} stock zip files.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skew_dir = out_dir / "skew_series"
    signal_dir = out_dir / "signals"
    bt_dir = out_dir / "backtests"
    for p in (skew_dir, signal_dir, bt_dir):
        p.mkdir(parents=True, exist_ok=True)

    filters = _build_filters(args)
    signal_cfg = _build_signal_config(args)
    cost_cfg = _build_cost_config(args)
    option_bt_cfg = _build_option_backtest_config(args)

    print("Fitting sector SSVI skew series...")
    if args.data_source == "duckdb":
        sector_skew = _fit_skew_for_ticker(args.sector_ticker, args=args, filters=filters)
        sector_ticker = args.sector_ticker
    else:
        sector_skew = _fit_skew_for_zip(args.sector_zip, args=args, filters=filters)
        sector_ticker = infer_ticker_from_zip_path(args.sector_zip)
    if sector_skew.empty:
        raise RuntimeError("Sector skew series is empty after SSVI calibration.")
    sector_path = skew_dir / f"{sector_ticker}_skew.csv"
    sector_skew.to_csv(sector_path, index=False)

    stock_skews: Dict[str, pd.DataFrame] = {}
    skipped_stocks: Dict[str, str] = {}
    stock_inputs = args.stock_tickers if args.data_source == "duckdb" else args.stock_zips
    for inp in stock_inputs:
        ticker = inp.upper() if args.data_source == "duckdb" else infer_ticker_from_zip_path(inp)
        print(f"Fitting stock SSVI skew series for {ticker}...")
        try:
            if args.data_source == "duckdb":
                df = _fit_skew_for_ticker(ticker, args=args, filters=filters)
            else:
                df = _fit_skew_for_zip(inp, args=args, filters=filters)
        except Exception as exc:
            skipped_stocks[ticker] = str(exc)
            print(f"Skipping {ticker}: {exc}")
            continue
        if df.empty:
            print(f"Skipping {ticker}: no calibrated dates.")
            skipped_stocks[ticker] = "No calibrated dates after filtering."
            continue
        stock_skews[ticker] = df
        df.to_csv(skew_dir / f"{ticker}_skew.csv", index=False)

    if not stock_skews:
        raise RuntimeError("No stock skew series available; cannot run idiosyncratic signal backtest.")

    signal_map: Dict[str, pd.DataFrame] = {}
    for ticker, skew_df in stock_skews.items():
        sig = build_residual_signal(
            skew_df,
            sector_skew,
            signal_cfg=signal_cfg,
            cost_cfg=cost_cfg,
            factor_model=args.factor_model,
            universe_skews=stock_skews,
            n_pca_factors=args.n_pca_factors,
        )
        if sig.empty:
            continue
        sig.to_csv(signal_dir / f"{ticker}_signal.csv", index=False)
        signal_map[ticker] = sig

    if not signal_map:
        raise RuntimeError("Signals produced no tradable backtests.")

    if args.backtest_mode == "idealized":
        portfolio, per_name_backtests = backtest_cross_sectional_residual_mean_reversion(
            signal_map=signal_map,
            signal_cfg=signal_cfg,
            cost_cfg=cost_cfg,
        )
    else:
        portfolio, per_name_backtests = backtest_option_cross_sectional(
            signal_map=signal_map,
            skew_map=stock_skews,
            signal_cfg=signal_cfg,
            option_cfg=option_bt_cfg,
        )
    per_name_stats: Dict[str, Dict[str, float]] = {}
    for ticker, bt in per_name_backtests.items():
        bt.to_csv(bt_dir / f"{ticker}_backtest.csv", index=False)
        if bt.empty:
            continue
        per_name_stats[ticker] = {
            "n_days": float(len(bt)),
            "avg_daily_net_pnl": float(bt["net_pnl"].mean()),
            "std_daily_net_pnl": float(bt["net_pnl"].std(ddof=0)),
            "final_cum_net_pnl": float(bt["cum_net_pnl"].iloc[-1]),
            "avg_weight": float(bt["weight"].abs().mean()) if "weight" in bt.columns else 0.0,
        }

    portfolio.to_csv(out_dir / "portfolio_backtest.csv", index=False)
    portfolio_summary = summarize_performance(portfolio)

    summary = {
        "date_range": {"start": args.start_date, "end": args.end_date},
        "sector_proxy": sector_ticker,
        "stock_universe": sorted(list(per_name_backtests.keys())),
        "skipped_stocks": skipped_stocks,
        "target_maturity_days": args.target_days,
        "k0": args.k0,
        "factor_model": args.factor_model,
        "n_pca_factors": args.n_pca_factors,
        "backtest_mode": args.backtest_mode,
        "data_source": args.data_source,
        "duckdb": (
            {"db_path": args.db_path, "table": args.db_table}
            if args.data_source == "duckdb"
            else {}
        ),
        "calibration_backend": args.calibration_backend,
        "filters": asdict(filters),
        "signal_config": asdict(signal_cfg),
        "transaction_cost_config": asdict(cost_cfg),
        "option_backtest_config": asdict(option_bt_cfg),
        "portfolio_summary": portfolio_summary,
        "per_name_summary": per_name_stats,
        "artifacts": {
            "sector_skew": str(sector_path),
            "skew_dir": str(skew_dir),
            "signal_dir": str(signal_dir),
            "backtest_dir": str(bt_dir),
            "portfolio_backtest": str(out_dir / "portfolio_backtest.csv"),
            "dashboard": str(out_dir / "dashboard.html"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    dashboard_path = None
    if not args.skip_dashboard:
        dashboard_path = build_dashboard(out_dir, Path(args.snapshots_dir))
        print(f"Wrote {dashboard_path}")

    print("SSVI stat-arb run complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
