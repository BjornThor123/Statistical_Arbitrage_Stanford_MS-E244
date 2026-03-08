from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

STRATEGY_ROOT = Path(__file__).resolve().parent


# ----------------------------- I/O helpers -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tabbed SSVI strategy monitoring and validation dashboard.")
    parser.add_argument(
        "--results-dir",
        default=str(STRATEGY_ROOT / "results" / "current"),
        help="Path to SSVI results folder.",
    )
    parser.add_argument(
        "--snapshots-dir",
        default=str(STRATEGY_ROOT / "results" / "snapshots"),
        help="Optional historical snapshots directory.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _load_ticker_map(folder: Path, suffix: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not folder.exists():
        return out
    for p in sorted(folder.glob(f"*{suffix}.csv")):
        ticker = p.stem.replace(suffix, "")
        out[ticker] = _read_csv(p)
    return out


def _snapshot_table(snapshots_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not snapshots_dir.exists():
        return pd.DataFrame()
    for d in sorted([p for p in snapshots_dir.iterdir() if p.is_dir()]):
        summary = _load_json(d / "summary.json")
        ps = summary.get("portfolio_summary", {}) if summary else {}
        rows.append(
            {
                "snapshot": d.name,
                "start": summary.get("date_range", {}).get("start", ""),
                "end": summary.get("date_range", {}).get("end", ""),
                "n_days": ps.get("n_days", np.nan),
                "daily_sharpe_like": ps.get("daily_sharpe_like", np.nan),
                "final_cum_net_pnl": ps.get("final_cum_net_pnl", np.nan),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------- utility helpers -----------------------------

def _to_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=bool)
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    txt = s.astype(str).str.strip().str.lower()
    return txt.isin(["1", "true", "t", "yes", "y"])


def _no_butterfly_margin(df: pd.DataFrame) -> pd.Series:
    if "no_butterfly_constraint_score" in df.columns:
        return pd.to_numeric(df["no_butterfly_constraint_score"], errors="coerce")
    if {"eta", "rho"}.issubset(df.columns):
        eta = pd.to_numeric(df["eta"], errors="coerce")
        rho = pd.to_numeric(df["rho"], errors="coerce")
        return 2.0 - eta * (1.0 + np.abs(rho))
    return pd.Series(dtype=float)


def _safe(v: Any) -> Any:
    """Convert NaN/Inf to None for JSON safety."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (np.floating, np.integer)):
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    return v


def _series_to_list(s: pd.Series) -> list:
    """Convert pandas Series to JSON-safe list (NaN -> None)."""
    return [_safe(x) for x in s.tolist()]


def _dates_to_list(s: pd.Series) -> list:
    """Convert datetime series to ISO string list."""
    return [str(d.date()) if pd.notna(d) else None for d in pd.to_datetime(s, errors="coerce")]


# ----------------------------- metrics tables (kept, extended) -----------------------------

def _portfolio_metrics(portfolio: pd.DataFrame) -> Dict[str, float]:
    if portfolio.empty or "portfolio_net_pnl" not in portfolio.columns:
        return {k: 0.0 for k in [
            "n_days", "avg_daily_net_pnl", "std_daily_net_pnl",
            "daily_sharpe_like", "ann_sharpe", "ann_return", "ann_vol",
            "sortino_like", "calmar",
            "final_cum_net_pnl", "hit_rate", "max_drawdown",
            "var_95", "cvar_95", "gross_total", "cost_total", "net_total",
        ]}

    r = pd.to_numeric(portfolio["portfolio_net_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    mu = float(np.mean(r))
    sd = float(np.std(r))
    dn = r[r < 0.0]
    dd_std = float(np.std(dn)) if dn.size else 0.0
    ann_sharpe = float(mu / sd * np.sqrt(252.0)) if sd > 1e-12 else 0.0
    sortino = float(mu / dd_std * np.sqrt(252.0)) if dd_std > 1e-12 else 0.0
    ann_return = float(mu * 252.0)
    ann_vol = float(sd * np.sqrt(252.0))

    cum = (
        pd.to_numeric(portfolio["portfolio_cum_net_pnl"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "portfolio_cum_net_pnl" in portfolio.columns
        else np.cumsum(r)
    )
    peak = np.maximum.accumulate(cum)
    draw = cum - peak
    max_dd = float(draw.min()) if draw.size else 0.0
    calmar = float(ann_return / abs(max_dd)) if abs(max_dd) > 1e-12 else 0.0

    gross_total = float(pd.to_numeric(portfolio.get("portfolio_gross_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    cost_total = float(pd.to_numeric(portfolio.get("portfolio_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    net_total = float(pd.to_numeric(portfolio.get("portfolio_net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())

    var_95 = float(np.nanquantile(r, 0.05)) if r.size else 0.0
    cvar_95 = float(np.mean(r[r <= var_95])) if r.size else 0.0

    return {
        "n_days": float(len(portfolio)),
        "avg_daily_net_pnl": mu,
        "std_daily_net_pnl": sd,
        "daily_sharpe_like": float(mu / sd) if sd > 1e-12 else 0.0,
        "ann_sharpe": ann_sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sortino_like": sortino,
        "calmar": calmar,
        "final_cum_net_pnl": float(cum[-1]) if cum.size else 0.0,
        "hit_rate": float(np.mean(r > 0.0)) if r.size else 0.0,
        "max_drawdown": max_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "gross_total": gross_total,
        "cost_total": cost_total,
        "net_total": net_total,
    }


# ----------------------------- kept table builders (return DataFrames) -----------------------------

def _build_universe_table(
    summary: Dict[str, object],
    skew_map: Dict[str, pd.DataFrame],
    signal_map: Dict[str, pd.DataFrame],
    backtest_map: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    tickers = set(summary.get("stock_universe", []))
    tickers.update(skew_map.keys())
    tickers.update(signal_map.keys())
    tickers.update(backtest_map.keys())
    rows: List[Dict[str, object]] = []
    for t in sorted(tickers):
        sk = skew_map.get(t, pd.DataFrame())
        sig = signal_map.get(t, pd.DataFrame())
        bt = backtest_map.get(t, pd.DataFrame())
        s = pd.to_datetime(sk.get("date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna()
        rows.append({
            "ticker": t,
            "skew_rows": int(len(sk)),
            "signal_rows": int(len(sig)),
            "backtest_rows": int(len(bt)),
            "coverage_start": str(s.min().date()) if len(s) else "",
            "coverage_end": str(s.max().date()) if len(s) else "",
            "signal_to_skew_ratio": float(len(sig) / len(sk)) if len(sk) else np.nan,
        })
    return pd.DataFrame(rows)


def _build_surface_table(skew_map: Dict[str, pd.DataFrame], total_days: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for t, df in sorted(skew_map.items()):
        if df.empty:
            continue
        rmse = pd.to_numeric(df.get("rmse_implied_volatility", pd.Series(dtype=float)), errors="coerce")
        npts = pd.to_numeric(df.get("n_points", pd.Series(dtype=float)), errors="coerce")
        rho = pd.to_numeric(df.get("rho", pd.Series(dtype=float)), errors="coerce")
        eta = pd.to_numeric(df.get("eta", pd.Series(dtype=float)), errors="coerce")
        gamma = pd.to_numeric(df.get("gamma", pd.Series(dtype=float)), errors="coerce")
        no_bfly = _no_butterfly_margin(df)
        rows.append({
            "ticker": t,
            "fit_days": int(len(df)),
            "coverage_ratio": float(len(df) / max(1, total_days)),
            "rmse_iv_median": float(rmse.median()) if len(rmse) else np.nan,
            "rmse_iv_p95": float(rmse.quantile(0.95)) if len(rmse) else np.nan,
            "n_points_mean": float(npts.mean()) if len(npts) else np.nan,
            "butterfly_viol_rate": float((no_bfly < 0).mean()) if len(no_bfly.dropna()) else np.nan,
        })
    return pd.DataFrame(rows)


def _build_bottleneck_table(
    skew_map: Dict[str, pd.DataFrame],
    signal_map: Dict[str, pd.DataFrame],
    backtest_map: Dict[str, pd.DataFrame],
    total_days: int,
) -> pd.DataFrame:
    tickers = set(skew_map.keys()) | set(signal_map.keys()) | set(backtest_map.keys())
    rows: List[Dict[str, object]] = []
    for t in sorted(tickers):
        sk = skew_map.get(t, pd.DataFrame())
        sig = signal_map.get(t, pd.DataFrame())
        bt = backtest_map.get(t, pd.DataFrame())
        skew_rows = int(len(sk))
        signal_rows = int(len(sig))
        regime = _to_bool_series(sig, "regime_pass")
        liquid = _to_bool_series(sig, "liquidity_pass")
        allowed = _to_bool_series(sig, "trade_allowed")
        regime_pass = int(regime.sum()) if len(regime) else 0
        liquid_pass = int(liquid.sum()) if len(liquid) else 0
        allowed_rows = int(allowed.sum()) if len(allowed) else 0
        w = pd.to_numeric(bt.get("weight", pd.Series(dtype=float)), errors="coerce")
        exec_active = int((w.abs() > 1e-9).sum()) if len(w.dropna()) else 0
        rows.append({
            "ticker": t,
            "total_days": int(total_days),
            "skew_rows": skew_rows,
            "signal_rows": signal_rows,
            "regime_pass_rows": regime_pass,
            "liquidity_pass_rows": liquid_pass,
            "trade_allowed_rows": allowed_rows,
            "exec_active_rows": exec_active,
        })
    return pd.DataFrame(rows)


def _build_pipeline_stage_table(
    skew_map: Dict[str, pd.DataFrame],
    signal_map: Dict[str, pd.DataFrame],
    backtest_map: Dict[str, pd.DataFrame],
    total_days: int,
) -> pd.DataFrame:
    n_names = max(1, len(set(skew_map.keys()) | set(signal_map.keys()) | set(backtest_map.keys())))
    potential_rows = int(total_days * n_names)
    skew_rows = int(sum(len(df) for df in skew_map.values()))
    signal_rows = int(sum(len(df) for df in signal_map.values()))
    regime_rows = int(sum(_to_bool_series(df, "regime_pass").sum() for df in signal_map.values()))
    liquidity_rows = int(sum(_to_bool_series(df, "liquidity_pass").sum() for df in signal_map.values()))
    allowed_rows = int(sum(_to_bool_series(df, "trade_allowed").sum() for df in signal_map.values()))
    active_rows = 0
    for df in backtest_map.values():
        w = pd.to_numeric(df.get("weight", pd.Series(dtype=float)), errors="coerce")
        if len(w.dropna()):
            active_rows += int((w.abs() > 1e-9).sum())
    stages = [
        ("Potential (days x names)", potential_rows),
        ("Skew fit", skew_rows),
        ("Signal generated", signal_rows),
        ("Regime pass", regime_rows),
        ("Liquidity pass", liquidity_rows),
        ("Trade allowed", allowed_rows),
        ("Executed (|w|>0)", active_rows),
    ]
    out = []
    for stage, count in stages:
        out.append({"stage": stage, "rows": int(count)})
    return pd.DataFrame(out)


def _build_execution_table(backtest_map: Dict[str, pd.DataFrame], signal_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for t, bt in sorted(backtest_map.items()):
        if bt.empty:
            continue
        w = pd.to_numeric(bt.get("weight", pd.Series(dtype=float)), errors="coerce")
        gp = pd.to_numeric(bt.get("gross_pnl", pd.Series(dtype=float)), errors="coerce")
        cp = pd.to_numeric(bt.get("cost", pd.Series(dtype=float)), errors="coerce")
        npnl = pd.to_numeric(bt.get("net_pnl", pd.Series(dtype=float)), errors="coerce")
        gross_total = float(gp.sum()) if len(gp) else np.nan
        cost_total = float(cp.sum()) if len(cp) else np.nan
        rows.append({
            "ticker": t,
            "net_pnl_total": float(npnl.sum()) if len(npnl) else np.nan,
            "gross_pnl_total": gross_total,
            "cost_total": cost_total,
            "hit_rate": float((npnl > 0).mean()) if len(npnl) else np.nan,
            "avg_abs_weight": float(np.nanmean(np.abs(w))) if len(w) else np.nan,
            "max_abs_weight": float(np.nanmax(np.abs(w))) if len(w.dropna()) else np.nan,
        })
    return pd.DataFrame(rows)


def _validation_checks(
    summary: Dict[str, object],
    results_dir: Path,
    portfolio: pd.DataFrame,
    skew_map: Dict[str, pd.DataFrame],
    signal_map: Dict[str, pd.DataFrame],
    execution_df: pd.DataFrame,
) -> List[Dict[str, object]]:
    checks: List[Dict[str, object]] = []

    def add(name: str, status: str, value: object, details: str) -> None:
        checks.append({"check": name, "status": status, "value": _safe(value) if isinstance(value, (float, int, np.floating, np.integer)) else value, "details": details})

    required = [
        results_dir / "summary.json",
        results_dir / "portfolio_backtest.csv",
        results_dir / "skew_series",
        results_dir / "signals",
        results_dir / "backtests",
    ]
    missing = [p.name for p in required if not p.exists()]
    add("Required artifacts present", "pass" if not missing else "fail", "ok" if not missing else ", ".join(missing), "Core files/folders required for run integrity.")

    if portfolio.empty:
        add("Portfolio non-empty", "fail", 0, "No portfolio rows.")
        return checks

    req_cols = ["portfolio_gross_pnl", "portfolio_cost", "portfolio_net_pnl", "portfolio_cum_net_pnl"]
    missing_cols = [c for c in req_cols if c not in portfolio.columns]
    add("Portfolio schema", "pass" if not missing_cols else "fail", "ok" if not missing_cols else ", ".join(missing_cols), "Required accounting columns.")

    if not missing_cols:
        gross = pd.to_numeric(portfolio["portfolio_gross_pnl"], errors="coerce").fillna(0.0)
        cost = pd.to_numeric(portfolio["portfolio_cost"], errors="coerce").fillna(0.0)
        net = pd.to_numeric(portfolio["portfolio_net_pnl"], errors="coerce").fillna(0.0)
        cum = pd.to_numeric(portfolio["portfolio_cum_net_pnl"], errors="coerce").fillna(0.0)

        pnl_err = float(np.abs((gross - cost) - net).max())
        add("PnL identity (gross-cost=net)", "pass" if pnl_err <= 1e-10 else ("warn" if pnl_err <= 1e-6 else "fail"), pnl_err, "Pointwise accounting consistency.")

        cum_err = float(np.abs(net.cumsum() - cum).max())
        add("Cumulative net consistency", "pass" if cum_err <= 1e-8 else ("warn" if cum_err <= 1e-5 else "fail"), cum_err, "Recomputed cumulative net vs stored.")

        cost_drag = float(cost.sum() / max(1e-12, abs(gross.sum())))
        add("Cost drag ratio", "pass" if cost_drag <= 0.5 else ("warn" if cost_drag <= 0.8 else "fail"), cost_drag, "Total costs / |total gross pnl|.")

    max_lev_cfg = float(summary.get("signal_config", {}).get("max_gross_leverage", np.nan))
    if np.isfinite(max_lev_cfg) and "gross_leverage" in portfolio.columns:
        lev_obs = float(pd.to_numeric(portfolio["gross_leverage"], errors="coerce").max())
        add("Gross leverage bound", "pass" if lev_obs <= max_lev_cfg + 1e-8 else "fail", lev_obs, f"Configured max_gross_leverage={max_lev_cfg}")

    max_w_cfg = float(summary.get("signal_config", {}).get("max_name_weight", np.nan))
    if np.isfinite(max_w_cfg) and not execution_df.empty and "max_abs_weight" in execution_df.columns:
        max_w_obs = float(pd.to_numeric(execution_df["max_abs_weight"], errors="coerce").max())
        add("Per-name weight bound", "pass" if max_w_obs <= max_w_cfg + 1e-8 else "fail", max_w_obs, f"Configured max_name_weight={max_w_cfg}")

    if skew_map:
        all_sk = pd.concat([d for d in skew_map.values() if not d.empty], ignore_index=True)
        if {"eta", "rho"}.issubset(all_sk.columns):
            eta = pd.to_numeric(all_sk["eta"], errors="coerce")
            rho = pd.to_numeric(all_sk["rho"], errors="coerce")
            viol = float((eta * (1.0 + np.abs(rho)) > 2.0).mean())
            add("SSVI no-butterfly bound", "pass" if viol == 0.0 else ("warn" if viol <= 0.01 else "fail"), viol, "eta*(1+|rho|)<=2 violation rate.")

        score = _no_butterfly_margin(all_sk)
        if len(score.dropna()):
            score_med = float(score.median())
            add("No-butterfly margin median", "pass" if np.isfinite(score_med) and score_med > 0.05 else ("warn" if np.isfinite(score_med) and score_med > 0.0 else "fail"), score_med, "Margin to boundary (higher is safer).")

    if signal_map:
        all_sig = pd.concat([d for d in signal_map.values() if not d.empty], ignore_index=True)
        if "zscore" in all_sig.columns:
            z = pd.to_numeric(all_sig["zscore"], errors="coerce")
            zmu = float(z.mean())
            zsd = float(z.std(ddof=0))
            add("Z-score center", "pass" if abs(zmu) < 0.25 else ("warn" if abs(zmu) < 0.5 else "fail"), zmu, "Mean z-score should be near zero.")
            add("Z-score scale", "pass" if 0.5 <= zsd <= 2.0 else ("warn" if 0.3 <= zsd <= 3.0 else "fail"), zsd, "Std(zscore) should be reasonable.")

    if "portfolio_net_pnl" in portfolio.columns:
        r = pd.to_numeric(portfolio["portfolio_net_pnl"], errors="coerce").fillna(0.0)
        mu = float(r.mean())
        sd = float(r.std(ddof=0))
        sharpe = float(mu / sd * np.sqrt(252.0)) if sd > 1e-12 else 0.0
        add("Sharpe realism monitor", "pass" if sharpe <= 4.0 else ("warn" if sharpe <= 8.0 else "fail"), sharpe, "Extremely high Sharpe can indicate overfitting/accounting leakage.")

    return checks


def _file_table(results_dir: Path, limit: int = 400) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    files = sorted([p for p in results_dir.rglob("*") if p.is_file()])
    for p in files[:limit]:
        rows.append({"path": p.relative_to(results_dir).as_posix(), "size_kb": round(p.stat().st_size / 1024.0, 2)})
    return rows


# ----------------------------- NEW computation functions -----------------------------

def _cost_sensitivity(portfolio: pd.DataFrame) -> Dict[str, list]:
    """Cumulative PnL at different cost multipliers."""
    if portfolio.empty:
        return {"dates": [], "multipliers": [], "series": {}}
    gross = pd.to_numeric(portfolio.get("portfolio_gross_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    cost = pd.to_numeric(portfolio.get("portfolio_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    dates = _dates_to_list(portfolio["date"]) if "date" in portfolio.columns else list(range(len(portfolio)))
    multipliers = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
    series = {}
    for m in multipliers:
        net = gross - cost * m
        cum = net.cumsum()
        series[str(m)] = _series_to_list(cum)
    return {"dates": dates, "multipliers": [str(m) for m in multipliers], "series": series}


def _crisis_decomposition(portfolio: pd.DataFrame) -> List[Dict[str, object]]:
    """PnL/Sharpe for Pre-Crisis/Crisis/Recovery periods (financial crisis 2008)."""
    if portfolio.empty or "date" not in portfolio.columns:
        return []
    df = portfolio.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    r = pd.to_numeric(df["portfolio_net_pnl"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(df.get("portfolio_gross_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    periods = [
        ("Pre-Crisis", "2006-01-01", "2007-06-30"),
        ("Crisis", "2007-07-01", "2009-03-31"),
        ("Recovery", "2009-04-01", "2010-12-31"),
    ]
    result = []
    for name, start, end in periods:
        mask = (df["date"] >= start) & (df["date"] <= end)
        sub = r[mask]
        sub_gross = gross[mask]
        if len(sub) < 2:
            result.append({"period": name, "n_days": 0, "net_pnl": 0, "gross_pnl": 0, "sharpe": None})
            continue
        mu = float(sub.mean())
        sd = float(sub.std())
        sharpe = float(mu / sd * np.sqrt(252.0)) if sd > 1e-12 else None
        result.append({
            "period": name,
            "n_days": int(mask.sum()),
            "net_pnl": _safe(float(sub.sum())),
            "gross_pnl": _safe(float(sub_gross.sum())),
            "sharpe": _safe(sharpe),
        })
    return result


def _rolling_sharpe(portfolio: pd.DataFrame) -> Dict[str, Any]:
    """63d and 126d rolling annualized Sharpe."""
    if portfolio.empty or "portfolio_net_pnl" not in portfolio.columns:
        return {"dates": [], "sharpe_63": [], "sharpe_126": []}
    r = pd.to_numeric(portfolio["portfolio_net_pnl"], errors="coerce").fillna(0.0)
    dates = _dates_to_list(portfolio["date"]) if "date" in portfolio.columns else list(range(len(portfolio)))

    def rolling_sh(s: pd.Series, w: int) -> list:
        mu = s.rolling(w, min_periods=w).mean()
        sd = s.rolling(w, min_periods=w).std()
        sh = (mu / sd * np.sqrt(252.0)).where(sd > 1e-12, other=np.nan)
        return _series_to_list(sh)

    return {
        "dates": dates,
        "sharpe_63": rolling_sh(r, 63),
        "sharpe_126": rolling_sh(r, 126),
    }


def _rolling_hit_rate(portfolio: pd.DataFrame) -> Dict[str, Any]:
    """63d rolling win rate."""
    if portfolio.empty or "portfolio_net_pnl" not in portfolio.columns:
        return {"dates": [], "hit_rate_63": []}
    r = pd.to_numeric(portfolio["portfolio_net_pnl"], errors="coerce").fillna(0.0)
    dates = _dates_to_list(portfolio["date"]) if "date" in portfolio.columns else list(range(len(portfolio)))
    wins = (r > 0).astype(float)
    hr = wins.rolling(63, min_periods=63).mean()
    return {"dates": dates, "hit_rate_63": _series_to_list(hr)}


def _capacity_analysis(portfolio: pd.DataFrame, summary: Dict[str, object]) -> Dict[str, Any]:
    """Estimated Sharpe at various capital levels with sqrt-impact scaling."""
    if portfolio.empty or "portfolio_net_pnl" not in portfolio.columns:
        return {"capitals": [], "sharpes": []}
    r = pd.to_numeric(portfolio["portfolio_net_pnl"], errors="coerce").fillna(0.0)
    mu = float(r.mean())
    sd = float(r.std())
    if sd < 1e-12:
        return {"capitals": [], "sharpes": []}
    base_sharpe = mu / sd * np.sqrt(252.0)
    tc_cfg = summary.get("transaction_cost_config", {})
    impact = float(tc_cfg.get("impact_cost", 0.0)) + float(tc_cfg.get("half_spread_cost", 0.02))
    base_capital = 1e5  # $100K base
    capitals = [1e5, 5e5, 1e6, 5e6, 1e7, 2.5e7, 5e7]
    sharpes = []
    for cap in capitals:
        scale = np.sqrt(cap / base_capital)
        cost_penalty = impact * (scale - 1) * 252  # rough annual cost increase
        adj_sharpe = max(0, base_sharpe - cost_penalty)
        sharpes.append(_safe(adj_sharpe))
    return {
        "capitals": [f"${c/1e6:.1f}M" if c >= 1e6 else f"${c/1e3:.0f}K" for c in capitals],
        "sharpes": sharpes,
    }


def _residual_autocorrelation(signal_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Lag 1-10 autocorrelation of residuals per ticker."""
    result: Dict[str, list] = {}
    for t, df in sorted(signal_map.items()):
        if df.empty or "residual" not in df.columns:
            continue
        resid = pd.to_numeric(df["residual"], errors="coerce").dropna()
        if len(resid) < 20:
            continue
        acorrs = []
        for lag in range(1, 11):
            ac = float(resid.autocorr(lag=lag)) if len(resid) > lag + 1 else None
            acorrs.append(_safe(ac))
        result[t] = acorrs
    return result


def _signal_predictiveness(signal_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """z-score(t) vs residual_change(t+5) scatter data + R-squared per ticker."""
    result: Dict[str, Any] = {}
    for t, df in sorted(signal_map.items()):
        if df.empty or "zscore" not in df.columns or "residual" not in df.columns:
            continue
        z = pd.to_numeric(df["zscore"], errors="coerce")
        resid = pd.to_numeric(df["residual"], errors="coerce")
        fwd = resid.shift(-5) - resid
        valid = pd.DataFrame({"z": z, "fwd": fwd}).dropna()
        if len(valid) < 20:
            continue
        corr = float(valid["z"].corr(valid["fwd"]))
        r2 = corr ** 2
        # Subsample for scatter (max 500 points)
        if len(valid) > 500:
            valid = valid.sample(500, random_state=42)
        result[t] = {
            "z": _series_to_list(valid["z"]),
            "fwd": _series_to_list(valid["fwd"]),
            "r2": _safe(r2),
            "corr": _safe(corr),
        }
    return result


def _drawdown_periods(portfolio: pd.DataFrame) -> List[Dict[str, object]]:
    """Identify drawdown episodes."""
    if portfolio.empty or "portfolio_cum_net_pnl" not in portfolio.columns:
        return []
    cum = pd.to_numeric(portfolio["portfolio_cum_net_pnl"], errors="coerce").fillna(0.0)
    dates = pd.to_datetime(portfolio["date"], errors="coerce") if "date" in portfolio.columns else pd.RangeIndex(len(portfolio))
    peak = cum.cummax()
    dd = cum - peak
    in_dd = dd < -1e-10
    episodes = []
    start_idx = None
    for i in range(len(dd)):
        if in_dd.iloc[i] and start_idx is None:
            start_idx = i
        elif not in_dd.iloc[i] and start_idx is not None:
            trough_idx = dd.iloc[start_idx:i].idxmin()
            depth = float(dd.iloc[trough_idx])
            episodes.append({
                "start": str(dates.iloc[start_idx].date()) if hasattr(dates.iloc[start_idx], "date") else str(start_idx),
                "trough": str(dates.iloc[trough_idx].date()) if hasattr(dates.iloc[trough_idx], "date") else str(trough_idx),
                "recovery": str(dates.iloc[i].date()) if hasattr(dates.iloc[i], "date") else str(i),
                "depth": _safe(depth),
                "duration_days": i - start_idx,
            })
            start_idx = None
    # Handle ongoing drawdown
    if start_idx is not None:
        trough_idx = dd.iloc[start_idx:].idxmin()
        episodes.append({
            "start": str(dates.iloc[start_idx].date()) if hasattr(dates.iloc[start_idx], "date") else str(start_idx),
            "trough": str(dates.iloc[trough_idx].date()) if hasattr(dates.iloc[trough_idx], "date") else str(trough_idx),
            "recovery": "(ongoing)",
            "depth": _safe(float(dd.iloc[trough_idx])),
            "duration_days": len(dd) - start_idx,
        })
    # Sort by depth, return top 10
    episodes.sort(key=lambda e: e["depth"] if e["depth"] is not None else 0)
    return episodes[:10]


def _monthly_returns_grid(portfolio: pd.DataFrame) -> Dict[str, Any]:
    """Year x Month PnL grid."""
    if portfolio.empty or "portfolio_net_pnl" not in portfolio.columns or "date" not in portfolio.columns:
        return {"years": [], "months": list(range(1, 13)), "grid": []}
    df = portfolio.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    r = pd.to_numeric(df["portfolio_net_pnl"], errors="coerce").fillna(0.0)
    monthly = df.groupby(["year", "month"]).apply(lambda g: float(r.loc[g.index].sum()), include_groups=False)
    years = sorted(df["year"].dropna().unique())
    grid = []
    for y in years:
        row = []
        for m in range(1, 13):
            val = monthly.get((y, m), None)
            row.append(_safe(float(val)) if val is not None else None)
        grid.append(row)
    return {"years": [int(y) for y in years], "months": list(range(1, 13)), "grid": grid}


def _net_exposure_series(backtest_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Sum of signed weights per date."""
    if not backtest_map:
        return {"dates": [], "net_exposure": []}
    frames = []
    for t, df in backtest_map.items():
        if df.empty or "date" not in df.columns or "weight" not in df.columns:
            continue
        sub = df[["date", "weight"]].copy()
        sub["weight"] = pd.to_numeric(sub["weight"], errors="coerce").fillna(0.0)
        sub = sub.rename(columns={"weight": t})
        frames.append(sub.set_index("date")[t])
    if not frames:
        return {"dates": [], "net_exposure": []}
    wide = pd.concat(frames, axis=1).sort_index().fillna(0.0)
    net = wide.sum(axis=1)
    return {
        "dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in net.index],
        "net_exposure": _series_to_list(net),
    }


def _per_name_monthly_pnl(backtest_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Monthly PnL per ticker for heatmap."""
    result: Dict[str, Any] = {"tickers": [], "months": [], "values": {}}
    for t, df in sorted(backtest_map.items()):
        if df.empty or "net_pnl" not in df.columns or "date" not in df.columns:
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["ym"] = d["date"].dt.to_period("M").astype(str)
        r = pd.to_numeric(d["net_pnl"], errors="coerce").fillna(0.0)
        monthly = d.groupby("ym").apply(lambda g: float(r.loc[g.index].sum()), include_groups=False)
        result["tickers"].append(t)
        result["values"][t] = {str(k): _safe(float(v)) for k, v in monthly.items()}
    # Collect all months
    all_months = set()
    for v in result["values"].values():
        all_months.update(v.keys())
    result["months"] = sorted(all_months)
    return result


def _residual_correlation_matrix(signal_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Ticker x ticker residual correlation."""
    wide = []
    for t, df in sorted(signal_map.items()):
        if not df.empty and {"date", "residual"}.issubset(df.columns):
            tmp = df[["date", "residual"]].copy().rename(columns={"residual": t})
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
            wide.append(tmp.set_index("date")[t])
    if not wide:
        return {"tickers": [], "matrix": []}
    merged = pd.concat(wide, axis=1).dropna()
    if len(merged) < 5:
        return {"tickers": [], "matrix": []}
    corr = merged.corr()
    tickers = corr.columns.tolist()
    matrix = []
    for t in tickers:
        matrix.append([_safe(float(corr.loc[t, t2])) for t2 in tickers])
    return {"tickers": tickers, "matrix": matrix}


def _yearly_performance(portfolio: pd.DataFrame) -> List[Dict[str, object]]:
    """Year-by-year performance table."""
    if portfolio.empty or "portfolio_net_pnl" not in portfolio.columns or "date" not in portfolio.columns:
        return []
    df = portfolio.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    r = pd.to_numeric(df["portfolio_net_pnl"], errors="coerce").fillna(0.0)
    cum = pd.to_numeric(df.get("portfolio_cum_net_pnl", r.cumsum()), errors="coerce").fillna(0.0)
    result = []
    for yr, grp in df.groupby("year"):
        sub_r = r.loc[grp.index]
        sub_cum = cum.loc[grp.index]
        peak = sub_cum.cummax()
        dd = sub_cum - peak
        mu = float(sub_r.mean())
        sd = float(sub_r.std())
        sharpe = float(mu / sd * np.sqrt(252)) if sd > 1e-12 else None
        result.append({
            "year": int(yr),
            "net_pnl": _safe(float(sub_r.sum())),
            "sharpe": _safe(sharpe),
            "max_dd": _safe(float(dd.min())),
            "hit_rate": _safe(float((sub_r > 0).mean())),
            "n_days": int(len(grp)),
        })
    return result


# ----------------------------- Hedging & Risk computations --------------------

def _factor_model_diagnostics(signal_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Per-ticker factor model regression diagnostics: beta, alpha, R² over time."""
    result: Dict[str, Any] = {}
    for t, df in sorted(signal_map.items()):
        if df.empty:
            continue
        d = _dates_to_list(df["date"]) if "date" in df.columns else []
        beta = pd.to_numeric(df.get("beta", pd.Series(dtype=float)), errors="coerce")
        alpha = pd.to_numeric(df.get("alpha", pd.Series(dtype=float)), errors="coerce")
        stock_sk = pd.to_numeric(df.get("stock_skew", pd.Series(dtype=float)), errors="coerce")
        sector_sk = pd.to_numeric(df.get("sector_skew", pd.Series(dtype=float)), errors="coerce")
        residual = pd.to_numeric(df.get("residual", pd.Series(dtype=float)), errors="coerce")

        # Compute rolling R² (what fraction of skew variance does the factor model explain?)
        # R² = 1 - var(residual) / var(stock_skew), rolling 60d
        r2_series = pd.Series(np.nan, index=df.index)
        window = 60
        if len(stock_sk.dropna()) > window and len(residual.dropna()) > window:
            var_stock = stock_sk.rolling(window, min_periods=30).var()
            var_resid = residual.rolling(window, min_periods=30).var()
            r2_series = (1.0 - var_resid / var_stock.replace(0, np.nan)).clip(0, 1)

        # Systematic component = alpha + beta * sector_skew
        systematic = alpha + beta * sector_sk

        result[t] = {
            "dates": d,
            "beta": _series_to_list(beta),
            "alpha": _series_to_list(alpha),
            "r2_rolling": _series_to_list(r2_series),
            "stock_skew": _series_to_list(stock_sk),
            "sector_skew": _series_to_list(sector_sk),
            "systematic": _series_to_list(systematic),
            "residual": _series_to_list(residual),
        }
    return result


def _variance_decomposition(signal_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Per-ticker: what % of skew variance is systematic vs idiosyncratic?"""
    rows = []
    for t, df in sorted(signal_map.items()):
        if df.empty:
            continue
        stock_sk = pd.to_numeric(df.get("stock_skew", pd.Series(dtype=float)), errors="coerce").dropna()
        residual = pd.to_numeric(df.get("residual", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(stock_sk) < 20 or len(residual) < 20:
            continue
        var_total = float(stock_sk.var())
        var_resid = float(residual.var())
        var_systematic = max(0, var_total - var_resid)
        r2 = var_systematic / var_total if var_total > 1e-12 else 0
        rows.append({
            "ticker": t,
            "var_total": _safe(var_total),
            "var_systematic": _safe(var_systematic),
            "var_idiosyncratic": _safe(var_resid),
            "r2": _safe(r2),
            "pct_systematic": _safe(r2 * 100),
            "pct_idiosyncratic": _safe((1 - r2) * 100),
        })
    return rows


def _realized_portfolio_vol(portfolio: pd.DataFrame) -> Dict[str, Any]:
    """21d rolling realized portfolio vol vs vol target."""
    if portfolio.empty or "portfolio_net_pnl" not in portfolio.columns:
        return {"dates": [], "realized_vol": [], "vol_target": []}
    r = pd.to_numeric(portfolio["portfolio_net_pnl"], errors="coerce").fillna(0.0)
    dates = _dates_to_list(portfolio["date"]) if "date" in portfolio.columns else []
    vol_21 = r.rolling(21, min_periods=10).std()
    return {
        "dates": dates,
        "realized_vol": _series_to_list(vol_21),
    }


# ----------------------------- Cost & Execution computations ------------------

def _cost_model_breakdown(backtest_map: Dict[str, pd.DataFrame], signal_map: Dict[str, pd.DataFrame],
                          summary: Dict[str, object]) -> Dict[str, Any]:
    """Detailed cost model breakdown per ticker."""
    tc_cfg = summary.get("transaction_cost_config", {})
    half_spread = float(tc_cfg.get("half_spread_cost", 0.02))
    commission = float(tc_cfg.get("commission_cost", 0.005))
    impact = float(tc_cfg.get("impact_cost", 0.0))
    gamma_drag = float(tc_cfg.get("hedge_drag_gamma", 0.01))
    round_trip = 2.0 * (half_spread + impact + commission)

    per_ticker: Dict[str, Any] = {}
    for t, bt in sorted(backtest_map.items()):
        if bt.empty:
            continue
        cost = pd.to_numeric(bt.get("cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        gross = pd.to_numeric(bt.get("gross_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        turnover = pd.to_numeric(bt.get("turnover", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        w = pd.to_numeric(bt.get("weight", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

        # Decompose: spread_cost ≈ turnover * (half_spread+commission)/(half_spread+commission+gamma*|w|) * total_cost
        # gamma_cost ≈ gamma_drag * |w| / (est_unit_cost + gamma_drag*|w|) * total_cost
        total_cost = float(cost.sum())
        total_turnover = float(turnover.sum())
        avg_abs_w = float(w.abs().mean())

        # Approximate decomposition based on the cost formula: cost = turnover * (est_unit_cost + gamma*|w|)
        # est_unit_cost ≈ round_trip/2 * scale (we approximate scale from actual data)
        if total_turnover > 1e-12 and avg_abs_w > 1e-12:
            avg_cost_per_turnover = total_cost / total_turnover if total_turnover > 0 else 0
            approx_gamma_portion = gamma_drag * avg_abs_w
            approx_spread_portion = max(0, avg_cost_per_turnover - approx_gamma_portion)
        else:
            approx_gamma_portion = 0
            approx_spread_portion = 0

        per_ticker[t] = {
            "total_cost": _safe(total_cost),
            "total_turnover": _safe(total_turnover),
            "avg_cost_per_turnover": _safe(total_cost / total_turnover if total_turnover > 1e-12 else 0),
            "approx_spread_commission": _safe(approx_spread_portion * total_turnover),
            "approx_gamma_drag": _safe(approx_gamma_portion * total_turnover),
            "cost_pct_of_gross": _safe(abs(total_cost / gross.sum()) if abs(gross.sum()) > 1e-12 else None),
        }

    return {
        "config": {
            "half_spread_cost": _safe(half_spread),
            "commission_cost": _safe(commission),
            "impact_cost": _safe(impact),
            "hedge_drag_gamma": _safe(gamma_drag),
            "round_trip_cost": _safe(round_trip),
        },
        "per_ticker": per_ticker,
    }


def _edge_vs_cost_analysis(signal_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Edge vs estimated cost: scatter data + pass/fail rates."""
    all_edges = []
    all_costs = []
    all_passed = []
    per_ticker: Dict[str, Any] = {}
    for t, df in sorted(signal_map.items()):
        if df.empty:
            continue
        ee = pd.to_numeric(df.get("expected_edge", pd.Series(dtype=float)), errors="coerce")
        uc = pd.to_numeric(df.get("estimated_unit_cost", pd.Series(dtype=float)), errors="coerce")
        ta = _to_bool_series(df, "trade_allowed")
        valid = pd.DataFrame({"edge": ee, "cost": uc, "allowed": ta}).dropna(subset=["edge", "cost"])
        if len(valid) < 5:
            continue
        # Subsample for scatter
        sample = valid.sample(min(300, len(valid)), random_state=42) if len(valid) > 300 else valid
        per_ticker[t] = {
            "edge": _series_to_list(sample["edge"]),
            "cost": _series_to_list(sample["cost"]),
            "allowed": [bool(x) for x in sample["allowed"].tolist()],
            "pass_rate": _safe(float(ta.mean())) if len(ta) else None,
            "edge_gt_cost_rate": _safe(float((ee > uc).mean())) if len(ee.dropna()) else None,
        }
        all_edges.extend(_series_to_list(valid["edge"]))
        all_costs.extend(_series_to_list(valid["cost"]))
        all_passed.extend([bool(x) for x in valid["allowed"].tolist()])
    return {"per_ticker": per_ticker, "all_edges": all_edges, "all_costs": all_costs, "all_passed": all_passed}


def _turnover_analysis(backtest_map: Dict[str, pd.DataFrame], portfolio: pd.DataFrame) -> Dict[str, Any]:
    """Turnover over time, per ticker and portfolio level."""
    # Per-ticker turnover time series
    per_ticker: Dict[str, Any] = {}
    for t, bt in sorted(backtest_map.items()):
        if bt.empty or "turnover" not in bt.columns:
            continue
        d = _dates_to_list(bt["date"]) if "date" in bt.columns else []
        to = pd.to_numeric(bt["turnover"], errors="coerce").fillna(0.0)
        per_ticker[t] = {
            "dates": d,
            "turnover": _series_to_list(to),
            "total": _safe(float(to.sum())),
            "avg": _safe(float(to.mean())),
        }

    # Portfolio-level daily turnover = sum of per-name turnovers
    if backtest_map:
        frames = []
        for t, bt in backtest_map.items():
            if bt.empty or "turnover" not in bt.columns or "date" not in bt.columns:
                continue
            sub = bt[["date", "turnover"]].copy()
            sub["turnover"] = pd.to_numeric(sub["turnover"], errors="coerce").fillna(0.0)
            sub = sub.set_index("date")["turnover"].rename(t)
            frames.append(sub)
        if frames:
            wide = pd.concat(frames, axis=1).fillna(0.0)
            portfolio_turnover = wide.sum(axis=1)
            p_dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in portfolio_turnover.index]
        else:
            p_dates = []
            portfolio_turnover = pd.Series(dtype=float)
    else:
        p_dates = []
        portfolio_turnover = pd.Series(dtype=float)

    return {
        "per_ticker": per_ticker,
        "portfolio_dates": p_dates,
        "portfolio_turnover": _series_to_list(portfolio_turnover) if len(portfolio_turnover) else [],
    }


def _unit_cost_over_time(signal_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Estimated unit cost over time per ticker."""
    result: Dict[str, Any] = {}
    for t, df in sorted(signal_map.items()):
        if df.empty or "estimated_unit_cost" not in df.columns:
            continue
        d = _dates_to_list(df["date"]) if "date" in df.columns else []
        uc = pd.to_numeric(df["estimated_unit_cost"], errors="coerce")
        result[t] = {"dates": d, "unit_cost": _series_to_list(uc)}
    return result


def _hold_period_distribution(backtest_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Distribution of hold periods (consecutive days with non-zero weight)."""
    all_durations: List[int] = []
    for t, bt in sorted(backtest_map.items()):
        if bt.empty or "weight" not in bt.columns:
            continue
        w = pd.to_numeric(bt["weight"], errors="coerce").fillna(0.0)
        in_pos = (w.abs() > 1e-9).astype(int)
        # Find runs of consecutive 1s
        changes = in_pos.diff().fillna(in_pos.iloc[0] if len(in_pos) else 0)
        starts = changes[changes == 1].index.tolist()
        ends = changes[changes == -1].index.tolist()
        # Handle position at end of series
        if len(starts) > len(ends):
            ends.append(in_pos.index[-1])
        for s, e in zip(starts, ends):
            dur = int(e - s)
            if dur > 0:
                all_durations.append(dur)
    return {"durations": all_durations}


# ----------------------------- JSON data assembly -----------------------------

def _build_dashboard_data(
    results_dir: Path,
    summary: Dict[str, object],
    portfolio: pd.DataFrame,
    skew_map: Dict[str, pd.DataFrame],
    signal_map: Dict[str, pd.DataFrame],
    backtest_map: Dict[str, pd.DataFrame],
    snapshots: pd.DataFrame,
) -> Dict[str, Any]:
    pm = _portfolio_metrics(portfolio)
    total_days = int(len(portfolio))
    if total_days == 0:
        day_candidates: List[pd.Series] = []
        for mp in [skew_map, signal_map, backtest_map]:
            for df in mp.values():
                if "date" in df.columns:
                    day_candidates.append(pd.to_datetime(df["date"], errors="coerce"))
        if day_candidates:
            total_days = int(pd.concat(day_candidates, ignore_index=True).dropna().nunique())

    execution_df = _build_execution_table(backtest_map, signal_map)
    checks = _validation_checks(summary, results_dir, portfolio, skew_map, signal_map, execution_df)

    # --- Portfolio time series ---
    p_dates = _dates_to_list(portfolio["date"]) if "date" in portfolio.columns and not portfolio.empty else []
    cum_net = _series_to_list(pd.to_numeric(portfolio.get("portfolio_cum_net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)) if not portfolio.empty else []
    gross_pnl_series = pd.to_numeric(portfolio.get("portfolio_gross_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    cost_series = pd.to_numeric(portfolio.get("portfolio_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    cum_gross = _series_to_list(gross_pnl_series.cumsum()) if not portfolio.empty else []
    cum_cost = _series_to_list(cost_series.cumsum()) if not portfolio.empty else []
    net_pnl_daily = _series_to_list(pd.to_numeric(portfolio.get("portfolio_net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)) if not portfolio.empty else []

    # Drawdown
    if not portfolio.empty and "portfolio_cum_net_pnl" in portfolio.columns:
        c = pd.to_numeric(portfolio["portfolio_cum_net_pnl"], errors="coerce").fillna(0.0)
        drawdown = _series_to_list(c - c.cummax())
    else:
        drawdown = []

    # Gross leverage
    gross_lev = _series_to_list(pd.to_numeric(portfolio.get("gross_leverage", pd.Series(dtype=float)), errors="coerce").fillna(0.0)) if not portfolio.empty else []
    direction_mult = _series_to_list(pd.to_numeric(portfolio.get("direction_multiplier", pd.Series(dtype=float)), errors="coerce").fillna(0.0)) if not portfolio.empty else []

    # --- Per-name data ---
    per_name_weights: Dict[str, Any] = {}
    per_name_cum_pnl: Dict[str, Any] = {}
    per_name_total: Dict[str, float] = {}
    for t, bt in sorted(backtest_map.items()):
        if bt.empty:
            continue
        d = _dates_to_list(bt["date"]) if "date" in bt.columns else []
        w = _series_to_list(pd.to_numeric(bt.get("weight", pd.Series(dtype=float)), errors="coerce").fillna(0.0))
        npnl = pd.to_numeric(bt.get("net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        per_name_weights[t] = {"dates": d, "weights": w}
        per_name_cum_pnl[t] = {"dates": d, "cum_pnl": _series_to_list(npnl.cumsum())}
        per_name_total[t] = _safe(float(npnl.sum()))

    # --- Skew & signal per-ticker ---
    skew_series_data: Dict[str, Any] = {}
    for t, df in sorted(skew_map.items()):
        if df.empty:
            continue
        d = _dates_to_list(df["date"]) if "date" in df.columns else []
        skew_series_data[t] = {
            "dates": d,
            "skew": _series_to_list(pd.to_numeric(df.get("skew", pd.Series(dtype=float)), errors="coerce")),
            "rmse": _series_to_list(pd.to_numeric(df.get("rmse_implied_volatility", pd.Series(dtype=float)), errors="coerce")),
            "rho": _series_to_list(pd.to_numeric(df.get("rho", pd.Series(dtype=float)), errors="coerce")),
            "eta": _series_to_list(pd.to_numeric(df.get("eta", pd.Series(dtype=float)), errors="coerce")),
            "gamma": _series_to_list(pd.to_numeric(df.get("gamma", pd.Series(dtype=float)), errors="coerce")),
            "no_bfly": _series_to_list(_no_butterfly_margin(df)),
            "n_points": _series_to_list(pd.to_numeric(df.get("n_points", pd.Series(dtype=float)), errors="coerce")),
        }

    zscore_series_data: Dict[str, Any] = {}
    all_zscores: list = []
    gate_rates: Dict[str, Dict[str, float]] = {}
    for t, df in sorted(signal_map.items()):
        if df.empty:
            continue
        d = _dates_to_list(df["date"]) if "date" in df.columns else []
        z = pd.to_numeric(df.get("zscore", pd.Series(dtype=float)), errors="coerce")
        all_zscores.extend(_series_to_list(z.dropna()))
        zscore_series_data[t] = {
            "dates": d,
            "zscore": _series_to_list(z),
        }
        # Gate rates per ticker
        n = len(df)
        if n > 0:
            ta = _to_bool_series(df, "trade_allowed")
            rp = _to_bool_series(df, "regime_pass")
            lp = _to_bool_series(df, "liquidity_pass")
            gate_rates[t] = {
                "trade_allowed": _safe(float(ta.mean())) if len(ta) else 0,
                "regime_pass": _safe(float(rp.mean())) if len(rp) else 0,
                "liquidity_pass": _safe(float(lp.mean())) if len(lp) else 0,
            }

    # --- Surface quality KPIs ---
    surface_kpis: Dict[str, Any] = {}
    if skew_map:
        all_rmse = pd.concat([pd.to_numeric(df.get("rmse_implied_volatility", pd.Series(dtype=float)), errors="coerce") for df in skew_map.values() if not df.empty], ignore_index=True).dropna()
        all_bfly = pd.concat([_no_butterfly_margin(df) for df in skew_map.values() if not df.empty], ignore_index=True).dropna()
        all_npts = pd.concat([pd.to_numeric(df.get("n_points", pd.Series(dtype=float)), errors="coerce") for df in skew_map.values() if not df.empty], ignore_index=True).dropna()
        surface_kpis = {
            "median_rmse": _safe(float(all_rmse.median())) if len(all_rmse) else None,
            "p95_rmse": _safe(float(all_rmse.quantile(0.95))) if len(all_rmse) else None,
            "butterfly_viol_rate": _safe(float((all_bfly < 0).mean())) if len(all_bfly) else None,
            "avg_fit_points": _safe(float(all_npts.mean())) if len(all_npts) else None,
            "rmse_all": _series_to_list(all_rmse),
        }

    # --- Build stage funnel ---
    stage_df = _build_pipeline_stage_table(skew_map, signal_map, backtest_map, total_days)
    funnel = []
    if not stage_df.empty:
        for _, row in stage_df.iterrows():
            funnel.append({"stage": row["stage"], "rows": int(row["rows"])})

    # --- Bottleneck table ---
    bottleneck_df = _build_bottleneck_table(skew_map, signal_map, backtest_map, total_days)
    bottleneck_rows = bottleneck_df.to_dict("records") if not bottleneck_df.empty else []

    # --- Universe ---
    universe_df = _build_universe_table(summary, skew_map, signal_map, backtest_map)
    universe_rows = universe_df.to_dict("records") if not universe_df.empty else []

    # --- Surface table ---
    surface_df = _build_surface_table(skew_map, total_days)
    surface_rows = surface_df.to_dict("records") if not surface_df.empty else []

    # --- Execution table ---
    exec_rows = execution_df.to_dict("records") if not execution_df.empty else []

    # --- Skew dist stats ---
    skew_stats: Dict[str, Any] = {}
    if skew_map:
        all_skew = pd.concat([pd.to_numeric(df.get("skew", pd.Series(dtype=float)), errors="coerce") for df in skew_map.values() if not df.empty], ignore_index=True).dropna()
        if len(all_skew) > 3:
            from scipy import stats as sp_stats
            skew_stats = {
                "skewness": _safe(float(sp_stats.skew(all_skew))),
                "kurtosis": _safe(float(sp_stats.kurtosis(all_skew))),
            }

    # --- Daily return dist stats ---
    ret_stats: Dict[str, Any] = {}
    if not portfolio.empty and "portfolio_net_pnl" in portfolio.columns:
        r = pd.to_numeric(portfolio["portfolio_net_pnl"], errors="coerce").fillna(0.0)
        try:
            from scipy import stats as sp_stats
            ret_stats = {
                "skewness": _safe(float(sp_stats.skew(r))),
                "kurtosis": _safe(float(sp_stats.kurtosis(r))),
            }
        except ImportError:
            ret_stats = {}

    # --- Config blob ---
    cfg = {
        "date_range": summary.get("date_range", {}),
        "data_source": summary.get("data_source", ""),
        "sector_proxy": summary.get("sector_proxy", ""),
        "factor_model": summary.get("factor_model", ""),
        "n_pca_factors": summary.get("n_pca_factors", ""),
        "filters": summary.get("filters", {}),
        "signal_config": summary.get("signal_config", {}),
        "transaction_cost_config": summary.get("transaction_cost_config", {}),
    }

    max_gross_lev = float(summary.get("signal_config", {}).get("max_gross_leverage", 1.5))
    entry_z = float(summary.get("signal_config", {}).get("entry_z", 1.0))
    exit_z = float(summary.get("signal_config", {}).get("exit_z", 0.25))

    # --- Exposure KPIs ---
    exposure_kpis: Dict[str, Any] = {}
    if not portfolio.empty:
        gl = pd.to_numeric(portfolio.get("gross_leverage", pd.Series(dtype=float)), errors="coerce")
        nn = pd.to_numeric(portfolio.get("n_names", pd.Series(dtype=float)), errors="coerce")
        exposure_kpis = {
            "avg_gross_leverage": _safe(float(gl.mean())) if len(gl.dropna()) else None,
            "max_gross_leverage": _safe(float(gl.max())) if len(gl.dropna()) else None,
            "avg_names_active": _safe(float(nn.mean())) if len(nn.dropna()) else None,
        }
        # avg absolute weight across all backtests
        all_w = pd.concat([pd.to_numeric(df.get("weight", pd.Series(dtype=float)), errors="coerce").abs() for df in backtest_map.values() if not df.empty], ignore_index=True).dropna()
        exposure_kpis["avg_abs_weight"] = _safe(float(all_w.mean())) if len(all_w) else None

    # Cost per-ticker breakdown
    cost_breakdown: Dict[str, Any] = {}
    for t, bt in sorted(backtest_map.items()):
        if bt.empty:
            continue
        gp = pd.to_numeric(bt.get("gross_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        cp = pd.to_numeric(bt.get("cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        np_ = pd.to_numeric(bt.get("net_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
        cost_breakdown[t] = {"gross": _safe(float(gp)), "cost": _safe(float(cp)), "net": _safe(float(np_))}

    # Gross-to-net ratio over time (rolling 21d)
    gross_to_net_ratio: Dict[str, Any] = {}
    if not portfolio.empty and {"portfolio_gross_pnl", "portfolio_cost"}.issubset(portfolio.columns):
        gp = pd.to_numeric(portfolio["portfolio_gross_pnl"], errors="coerce").fillna(0.0).abs()
        cp = pd.to_numeric(portfolio["portfolio_cost"], errors="coerce").fillna(0.0)
        roll_cost = cp.rolling(21, min_periods=5).sum()
        roll_gross = gp.rolling(21, min_periods=5).sum()
        ratio = (roll_cost / roll_gross.replace(0, np.nan))
        gross_to_net_ratio = {
            "dates": p_dates,
            "ratio": _series_to_list(ratio),
        }

    # Position heatmap (weekly)
    position_heatmap: Dict[str, Any] = {"dates": [], "tickers": [], "weights": []}
    if backtest_map:
        frames = []
        for t, df in sorted(backtest_map.items()):
            if df.empty or "date" not in df.columns or "weight" not in df.columns:
                continue
            sub = df[["date", "weight"]].copy()
            sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
            sub["weight"] = pd.to_numeric(sub["weight"], errors="coerce").fillna(0.0)
            sub = sub.set_index("date").resample("W").last().fillna(0.0)
            sub.columns = [t]
            frames.append(sub)
        if frames:
            wide = pd.concat(frames, axis=1).fillna(0.0)
            position_heatmap = {
                "dates": [str(d.date()) for d in wide.index],
                "tickers": wide.columns.tolist(),
                "weights": [_series_to_list(wide[c]) for c in wide.columns],
            }

    data = {
        "metrics": {k: _safe(v) for k, v in pm.items()},
        "ret_stats": ret_stats,
        "portfolio": {
            "dates": p_dates,
            "cum_net": cum_net,
            "cum_gross": cum_gross,
            "cum_cost": cum_cost,
            "net_pnl_daily": net_pnl_daily,
            "drawdown": drawdown,
            "gross_leverage": gross_lev,
            "direction_multiplier": direction_mult,
        },
        "exposure_kpis": exposure_kpis,
        "net_exposure": _net_exposure_series(backtest_map),
        "per_name_weights": per_name_weights,
        "position_heatmap": position_heatmap,
        "per_name_cum_pnl": per_name_cum_pnl,
        "per_name_total": per_name_total,
        "per_name_monthly": _per_name_monthly_pnl(backtest_map),
        "cost_breakdown": cost_breakdown,
        "cost_sensitivity": _cost_sensitivity(portfolio),
        "crisis_decomposition": _crisis_decomposition(portfolio),
        "gross_to_net_ratio": gross_to_net_ratio,
        "skew_series": skew_series_data,
        "zscore_series": zscore_series_data,
        "all_zscores": all_zscores,
        "gate_rates": gate_rates,
        "residual_autocorr": _residual_autocorrelation(signal_map),
        "residual_corr_matrix": _residual_correlation_matrix(signal_map),
        "signal_predictiveness": _signal_predictiveness(signal_map),
        "surface_kpis": surface_kpis,
        "surface_table": surface_rows,
        "rolling_sharpe": _rolling_sharpe(portfolio),
        "rolling_hit_rate": _rolling_hit_rate(portfolio),
        "drawdown_periods": _drawdown_periods(portfolio),
        "capacity": _capacity_analysis(portfolio, summary),
        "yearly_performance": _yearly_performance(portfolio),
        "monthly_returns": _monthly_returns_grid(portfolio),
        "validation_checks": checks,
        "funnel": funnel,
        "bottleneck_table": bottleneck_rows,
        "universe": universe_rows,
        "execution_table": exec_rows,
        "config": cfg,
        "files": _file_table(results_dir),
        "max_gross_leverage": _safe(max_gross_lev),
        "entry_z": _safe(entry_z),
        "exit_z": _safe(exit_z),
        "vol_target_daily": _safe(float(summary.get("signal_config", {}).get("vol_target_daily", 0.02))),
        "results_dir": str(results_dir),
        # Hedging & Risk
        "factor_diagnostics": _factor_model_diagnostics(signal_map),
        "variance_decomposition": _variance_decomposition(signal_map),
        "realized_vol": _realized_portfolio_vol(portfolio),
        # Cost & Execution
        "cost_model": _cost_model_breakdown(backtest_map, signal_map, summary),
        "edge_vs_cost": _edge_vs_cost_analysis(signal_map),
        "turnover_analysis": _turnover_analysis(backtest_map, portfolio),
        "unit_cost_ts": _unit_cost_over_time(signal_map),
        "hold_periods": _hold_period_distribution(backtest_map),
    }
    return data


# ----------------------------- HTML template -----------------------------

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>SSVI Strategy Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>
    :root {
      --bg1: #f0f4f8; --bg2: #e2ecf5; --ink: #0f1b2d; --muted: #51617a;
      --card: #ffffff; --line: #cfdced; --accent: #0b6b90;
      --pass: #127b5a; --warn: #9a6a00; --fail: #a12a2a;
    }
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
           margin: 0; padding: 20px; color: var(--ink); background: linear-gradient(135deg, var(--bg1), var(--bg2)); }
    h1 { margin: 0 0 4px 0; font-size: 22px; }
    .subtitle { color: var(--muted); font-size: 12px; margin-bottom: 12px; }
    .tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 14px 0; }
    .tab-btn { border: 1px solid var(--line); background: #f8fbff; color: var(--ink);
               border-radius: 8px; padding: 7px 14px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all .15s; }
    .tab-btn:hover { background: #edf3fa; }
    .tab-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
            padding: 16px; margin: 0 0 14px 0; box-shadow: 0 2px 8px rgba(12,40,74,.04); }
    .card h3 { margin: 0 0 10px 0; font-size: 15px; }
    .grid { display: grid; gap: 10px; grid-template-columns: repeat(4, 1fr); }
    .grid-2 { display: grid; gap: 10px; grid-template-columns: 1fr 1fr; }
    .kpi { border: 1px solid #d8e3f1; background: linear-gradient(180deg,#f8fbff,#edf4ff);
           border-radius: 8px; padding: 10px; text-align: center; }
    .kpi .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
    .kpi .value { font-size: 22px; font-weight: 700; margin-top: 2px; }
    .kpi.red { border-color: #e6b5b5; background: #fff1f1; }
    .kpi.red .value { color: var(--fail); }
    .banner { padding: 12px 16px; border-radius: 8px; margin-bottom: 14px; font-size: 13px; line-height: 1.5; }
    .banner.warn { background: #fff6e0; border: 1px solid #ead39d; color: #7a5500; }
    .banner.info { background: #e8f4fd; border: 1px solid #b8d8f0; color: #1a5276; }
    .tbl { border-collapse: collapse; width: 100%; font-size: 12px; }
    .tbl th, .tbl td { border: 1px solid var(--line); padding: 6px 8px; text-align: right; }
    .tbl th:first-child, .tbl td:first-child { text-align: left; }
    .tbl th { background: #f5f8fb; font-weight: 600; }
    .pill { padding: 2px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; display: inline-block; }
    .pill.pass { background: #e7f7f1; color: var(--pass); }
    .pill.warn { background: #fff6e8; color: var(--warn); }
    .pill.fail { background: #fdecec; color: var(--fail); }
    .pill.badge { background: #fdecec; color: var(--fail); font-size: 10px; vertical-align: middle; margin-left: 6px; }
    .chart-container { width: 100%; min-height: 300px; }
    pre { white-space: pre-wrap; font-size: 11px; background: #f7fbff; border: 1px solid var(--line);
          border-radius: 8px; padding: 10px; max-height: 400px; overflow: auto; }
    details { margin-bottom: 10px; }
    details > summary { cursor: pointer; font-weight: 600; padding: 6px 0; }
    @media (max-width: 900px) { .grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 500px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<h1>SSVI Strategy Dashboard</h1>
<div class="subtitle">Results: <code>__RESULTS_DIR__</code></div>
<div class="tabs">
  <button class="tab-btn active" data-tab="summary">Executive Summary</button>
  <button class="tab-btn" data-tab="exposure">Exposure & Positions</button>
  <button class="tab-btn" data-tab="hedging">Hedging & Risk</button>
  <button class="tab-btn" data-tab="pnl">PnL Attribution</button>
  <button class="tab-btn" data-tab="costexec">Cost & Execution</button>
  <button class="tab-btn" data-tab="signal">Signal & Skew</button>
  <button class="tab-btn" data-tab="surface">Vol Surface</button>
  <button class="tab-btn" data-tab="robustness">Robustness</button>
  <button class="tab-btn" data-tab="validation">Validation & Data</button>
</div>
<section id="tab-summary" class="tab-panel active"></section>
<section id="tab-exposure" class="tab-panel"></section>
<section id="tab-hedging" class="tab-panel"></section>
<section id="tab-pnl" class="tab-panel"></section>
<section id="tab-costexec" class="tab-panel"></section>
<section id="tab-signal" class="tab-panel"></section>
<section id="tab-surface" class="tab-panel"></section>
<section id="tab-robustness" class="tab-panel"></section>
<section id="tab-validation" class="tab-panel"></section>

<script>
const D = __DASHBOARD_DATA__;
const PLOTLY_LAYOUT = {margin:{t:30,r:20,b:40,l:60}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fbfdff',
  font:{family:'-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif', size:11, color:'#0f1b2d'},
  xaxis:{gridcolor:'#e0e8f0'}, yaxis:{gridcolor:'#e0e8f0'}};
const COLORS = ['#0d6f92','#2f9e8f','#7f56d9','#cc7000','#a12a2a','#3b7a57','#6f4e37','#c44569'];
function PL(o){return Object.assign(JSON.parse(JSON.stringify(PLOTLY_LAYOUT)),o||{});}
function fmt(v,d){if(v==null)return'—';return typeof v==='number'?v.toFixed(d===undefined?2:d):v;}
function pct(v){return v==null?'—':(v*100).toFixed(1)+'%';}

// Lazy tab rendering
const rendered = {};
function renderTab(name){
  if(rendered[name]) return;
  rendered[name] = true;
  const el = document.getElementById('tab-'+name);
  switch(name){
    case 'summary': renderSummary(el); break;
    case 'exposure': renderExposure(el); break;
    case 'hedging': renderHedging(el); break;
    case 'pnl': renderPnL(el); break;
    case 'costexec': renderCostExec(el); break;
    case 'signal': renderSignal(el); break;
    case 'surface': renderSurface(el); break;
    case 'robustness': renderRobustness(el); break;
    case 'validation': renderValidation(el); break;
  }
}

function card(title, id){
  return `<div class="card"><h3>${title}</h3><div id="${id}" class="chart-container"></div></div>`;
}
function cardHtml(title, html){
  return `<div class="card"><h3>${title}</h3>${html}</div>`;
}

// ==================== TAB: Executive Summary ====================
function renderSummary(el){
  const m = D.metrics;
  const annSharpe = m.ann_sharpe||0;
  const unrealistic = Math.abs(annSharpe) > 4;
  let html = '';

  // Strategy overview
  const cfg = D.config||{};
  const sc = cfg.signal_config||{};
  const tc = cfg.transaction_cost_config||{};
  const universe = (D.universe||[]).map(u=>u.ticker).filter(t=>t!=='XLF').join(', ');
  html += `<div class="card"><h3>What This Strategy Trades</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:13px;line-height:1.6;">
    <div>
      <strong>Instrument:</strong> Implied volatility skew on single-name equity options<br>
      <strong>Implementation:</strong> OTM put/call spreads (or vega-weighted option portfolios) to isolate skew exposure, delta-hedged<br>
      <strong>Universe:</strong> ${universe || 'N/A'} vs ${cfg.sector_proxy||'XLF'} sector ETF<br>
      <strong>Signal:</strong> Mean-reversion of idiosyncratic skew residual (stock skew minus factor model prediction)<br>
      <strong>Factor model:</strong> ${cfg.factor_model||'N/A'}${cfg.n_pca_factors?' ('+cfg.n_pca_factors+' factors)':''}
    </div>
    <div>
      <strong>Entry:</strong> |z-score| &ge; ${sc.entry_z||1.0} &rarr; trade against the z (short skew if rich, long if cheap)<br>
      <strong>Exit:</strong> |z-score| &lt; ${sc.exit_z||0.25} or mean-reversion complete<br>
      <strong>PnL driver:</strong> <code>weight &times; (&minus;&Delta;residual)</code> &mdash; profit when skew reverts toward fair value<br>
      <strong>Costs:</strong> half-spread ${pct(tc.half_spread_cost||0.02)}, commission ${pct(tc.commission_cost||0.005)}, gamma drag ${pct(tc.hedge_drag_gamma||0.01)}<br>
      <strong>Constraints:</strong> max ${pct(sc.max_name_weight||0.35)}/name, max ${fmt(sc.max_gross_leverage||1.5,1)}&times; gross leverage, ${sc.vol_target_daily||0.02} daily vol target
    </div>
    </div>
  </div>`;

  // Warning banner
  if(unrealistic){
    html += `<div class="banner warn">
      <strong>Sharpe Realism Warning:</strong> Annualized Sharpe of <strong>${fmt(annSharpe,1)}</strong> is unrealistically high.
      Typical ranges: Equity L/S 0.3-0.8, Vol Arb 0.5-1.5, Stat Arb 1.0-2.5.
      A value this high almost certainly indicates in-sample overfitting, look-ahead bias, or a very short/unusual sample period.
      The daily Sharpe of ${fmt(m.daily_sharpe_like,3)} is being annualized by &times;&radic;252.
    </div>`;
  }

  // KPI grid
  html += `<div class="card"><h3>Portfolio KPIs</h3><div class="grid">
    <div class="kpi ${unrealistic?'red':''}"><div class="label">Ann. Sharpe ${unrealistic?'<span class="pill badge">Unrealistic</span>':''}</div><div class="value">${fmt(annSharpe,2)}</div></div>
    <div class="kpi"><div class="label">Ann. Return</div><div class="value">${fmt(m.ann_return,4)}</div></div>
    <div class="kpi"><div class="label">Ann. Vol</div><div class="value">${fmt(m.ann_vol,4)}</div></div>
    <div class="kpi"><div class="label">Max Drawdown</div><div class="value">${fmt(m.max_drawdown,4)}</div></div>
    <div class="kpi"><div class="label">Hit Rate</div><div class="value">${pct(m.hit_rate)}</div></div>
    <div class="kpi"><div class="label">Sortino</div><div class="value">${fmt(m.sortino_like,2)}</div></div>
    <div class="kpi"><div class="label">Calmar</div><div class="value">${fmt(m.calmar,2)}</div></div>
    <div class="kpi"><div class="label">Total Net PnL</div><div class="value">${fmt(m.final_cum_net_pnl,3)}</div></div>
  </div></div>`;

  // Comparison context
  html += cardHtml('Sharpe Context (Typical Ranges)', `<table class="tbl">
    <tr><th>Strategy Type</th><th>Typical Ann. Sharpe</th><th>This Strategy</th></tr>
    <tr><td>Equity Long/Short</td><td>0.3 – 0.8</td><td rowspan="3" style="font-size:18px;font-weight:700;color:${unrealistic?'var(--fail)':'var(--pass)'};">${fmt(annSharpe,1)}</td></tr>
    <tr><td>Volatility Arbitrage</td><td>0.5 – 1.5</td></tr>
    <tr><td>Statistical Arbitrage</td><td>1.0 – 2.5</td></tr>
  </table>`);

  // Cum PnL chart
  html += card('Cumulative PnL (Net / Gross / Cost)', 'chart-cum-pnl');
  html += card('Drawdown', 'chart-drawdown');
  html += card('Daily Return Distribution', 'chart-ret-dist');

  el.innerHTML = html;

  // Plot cum PnL
  const pd_ = D.portfolio;
  if(pd_.dates.length){
    Plotly.newPlot('chart-cum-pnl', [
      {x:pd_.dates, y:pd_.cum_net, name:'Net', line:{color:'#0d6f92',width:2}},
      {x:pd_.dates, y:pd_.cum_gross, name:'Gross', line:{color:'#2f9e8f',width:1,dash:'dot'}},
      {x:pd_.dates, y:pd_.cum_cost, name:'Cumulative Cost', line:{color:'#a12a2a',width:1,dash:'dash'}},
    ], PL({xaxis:{rangeslider:{visible:true},gridcolor:'#e0e8f0'},
      shapes:[{type:'rect',x0:'2007-07-01',x1:'2009-03-31',y0:0,y1:1,yref:'paper',fillcolor:'rgba(161,42,42,0.08)',line:{width:0}}],
      annotations:[{x:'2008-09-15',y:1,yref:'paper',text:'Crisis Period',showarrow:false,font:{size:10,color:'#a12a2a'}}]
    }), {responsive:true});
  }
  // Drawdown
  if(pd_.dates.length){
    Plotly.newPlot('chart-drawdown', [{x:pd_.dates, y:pd_.drawdown, fill:'tozeroy',
      line:{color:'#a12a2a',width:1}, fillcolor:'rgba(161,42,42,0.2)'}],
      PL({yaxis:{gridcolor:'#e0e8f0',title:'Drawdown'}}), {responsive:true});
  }
  // Return distribution
  if(pd_.net_pnl_daily.length){
    const vals = pd_.net_pnl_daily.filter(v=>v!=null);
    const traces = [{x:vals, type:'histogram', nbinsx:50, marker:{color:'rgba(13,111,146,0.7)'}, name:'Daily Returns'}];
    // Normal overlay
    const mu = vals.reduce((a,b)=>a+b,0)/vals.length;
    const sd = Math.sqrt(vals.reduce((a,b)=>a+(b-mu)**2,0)/vals.length);
    if(sd>1e-12){
      const xs = []; const ys = [];
      for(let i=-4;i<=4;i+=0.1){
        const x = mu+i*sd; xs.push(x);
        ys.push(Math.exp(-0.5*((x-mu)/sd)**2)/(sd*Math.sqrt(2*Math.PI)));
      }
      const binW = (Math.max(...vals)-Math.min(...vals))/50;
      traces.push({x:xs, y:ys.map(v=>v*vals.length*binW), mode:'lines', name:'Normal',
        line:{color:'#cc7000',width:2,dash:'dash'}});
    }
    const rs = D.ret_stats||{};
    const ann = [];
    if(rs.skewness!=null) ann.push({x:0.98,y:0.95,xref:'paper',yref:'paper',text:`Skew: ${fmt(rs.skewness,3)}`,showarrow:false,font:{size:11}});
    if(rs.kurtosis!=null) ann.push({x:0.98,y:0.88,xref:'paper',yref:'paper',text:`Kurt: ${fmt(rs.kurtosis,3)}`,showarrow:false,font:{size:11}});
    Plotly.newPlot('chart-ret-dist', traces, PL({barmode:'overlay',annotations:ann}), {responsive:true});
  }
}

// ==================== TAB: Exposure & Positions ====================
function renderExposure(el){
  const ek = D.exposure_kpis||{};
  let html = `<div class="banner info">
    <strong>Reading these charts:</strong> "Weight" = portfolio allocation to each name's <em>skew mean-reversion trade</em>.
    A positive weight means the strategy is <strong>long idiosyncratic skew</strong> (skew is cheap vs factor model &rarr;
    buy OTM puts / sell OTM calls). Negative weight = <strong>short skew</strong> (skew is rich &rarr; sell OTM puts / buy OTM calls).
    PnL accrues as <code>weight &times; (&minus;&Delta;residual)</code> when skew reverts toward the factor model.
  </div>`;
  html += `<div class="card"><h3>Exposure KPIs</h3><div class="grid">
    <div class="kpi"><div class="label">Avg Gross Leverage</div><div class="value">${fmt(ek.avg_gross_leverage)}</div></div>
    <div class="kpi"><div class="label">Max Gross Leverage</div><div class="value">${fmt(ek.max_gross_leverage)}</div></div>
    <div class="kpi"><div class="label">Avg Names Active</div><div class="value">${fmt(ek.avg_names_active,1)}</div></div>
    <div class="kpi"><div class="label">Avg |Weight|</div><div class="value">${fmt(ek.avg_abs_weight,4)}</div></div>
  </div></div>`;
  html += card('Gross Leverage Over Time (sum of |skew weights|)', 'chart-gross-lev');
  html += card('Per-Name Skew Exposure (Stacked Area)', 'chart-weights-stacked');
  html += card('Net Skew Exposure (sum of signed weights — market neutrality)', 'chart-net-exp');
  html += card('Skew Position Heatmap (Weekly)', 'chart-pos-heatmap');
  html += card('Direction Multiplier (mean-revert vs momentum)', 'chart-direction');
  el.innerHTML = html;

  const pd_ = D.portfolio;
  // Gross leverage
  if(pd_.dates.length){
    const traces = [{x:pd_.dates, y:pd_.gross_leverage, name:'Gross Leverage', line:{color:'#0d6f92',width:2}}];
    const maxLev = D.max_gross_leverage;
    const shapes = maxLev ? [{type:'line',x0:pd_.dates[0],x1:pd_.dates[pd_.dates.length-1],y0:maxLev,y1:maxLev,
      line:{color:'#a12a2a',width:1,dash:'dash'}}] : [];
    Plotly.newPlot('chart-gross-lev', traces, PL({shapes,yaxis:{title:'Leverage',gridcolor:'#e0e8f0'}}), {responsive:true});
  }
  // Stacked area
  const tickers = Object.keys(D.per_name_weights);
  if(tickers.length){
    const traces = tickers.map((t,i)=>{
      const d = D.per_name_weights[t];
      return {x:d.dates, y:d.weights, name:t, stackgroup:'one', line:{color:COLORS[i%COLORS.length],width:0}};
    });
    Plotly.newPlot('chart-weights-stacked', traces, PL({yaxis:{title:'Weight'}}), {responsive:true});
  }
  // Net exposure
  const ne = D.net_exposure;
  if(ne.dates.length){
    Plotly.newPlot('chart-net-exp', [{x:ne.dates, y:ne.net_exposure, line:{color:'#7f56d9',width:2}, fill:'tozeroy',
      fillcolor:'rgba(127,86,217,0.1)'}], PL({yaxis:{title:'Net Exposure',zeroline:true}}), {responsive:true});
  }
  // Position heatmap
  const ph = D.position_heatmap;
  if(ph.dates.length && ph.tickers.length){
    Plotly.newPlot('chart-pos-heatmap', [{z:ph.weights, x:ph.dates, y:ph.tickers, type:'heatmap',
      colorscale:'RdBu', zmid:0, colorbar:{title:'Weight'}}],
      PL({yaxis:{autorange:true},height:200+ph.tickers.length*40}), {responsive:true});
  }
  // Direction
  if(pd_.dates.length && pd_.direction_multiplier.some(v=>v!=null&&v!==0)){
    Plotly.newPlot('chart-direction', [{x:pd_.dates, y:pd_.direction_multiplier, line:{color:'#cc7000',width:2},
      fill:'tozeroy', fillcolor:'rgba(204,112,0,0.1)'}], PL({yaxis:{title:'Direction Multiplier'}}), {responsive:true});
  }
}

// ==================== TAB: PnL Attribution ====================
function renderPnL(el){
  let html = `<div class="banner info">
    <strong>PnL mechanics:</strong> Gross PnL = <code>weight &times; (&minus;&Delta;residual_skew)</code>.
    Each day the strategy profits when a name's idiosyncratic skew reverts toward the factor model prediction.
    Costs include option bid-ask spread, commissions, and gamma hedging drag.
  </div>`;
  html += card('Per-Name Cumulative Skew PnL', 'chart-pername-cum');
  html += card('Per-Name Total Skew PnL', 'chart-pername-bar');
  html += card('Per-Name Monthly PnL Heatmap', 'chart-pername-monthly');

  // Crisis decomposition
  const crisis = D.crisis_decomposition;
  if(crisis.length){
    html += cardHtml('Crisis vs Calm Decomposition', `<table class="tbl">
      <tr><th>Period</th><th>Days</th><th>Net PnL</th><th>Gross PnL</th><th>Sharpe</th></tr>
      ${crisis.map(c=>`<tr><td>${c.period}</td><td>${c.n_days}</td><td>${fmt(c.net_pnl,4)}</td><td>${fmt(c.gross_pnl,4)}</td><td>${fmt(c.sharpe,2)}</td></tr>`).join('')}
    </table>`);
  }
  html += card('Cost Breakdown Per Ticker', 'chart-cost-breakdown');
  html += card('Cost Sensitivity (Cumulative PnL at Different Cost Multiples)', 'chart-cost-sens');
  html += card('Gross-to-Net Cost Ratio (21d Rolling)', 'chart-gross-net-ratio');
  el.innerHTML = html;

  // Per-name cum PnL
  const tickers = Object.keys(D.per_name_cum_pnl);
  if(tickers.length){
    const traces = tickers.map((t,i)=>{
      const d = D.per_name_cum_pnl[t];
      return {x:d.dates, y:d.cum_pnl, name:t, line:{color:COLORS[i%COLORS.length],width:2}};
    });
    Plotly.newPlot('chart-pername-cum', traces, PL({yaxis:{title:'Cumulative PnL'}}), {responsive:true});
  }
  // Per-name total bar
  const totals = D.per_name_total;
  const tks = Object.keys(totals);
  if(tks.length){
    Plotly.newPlot('chart-pername-bar', [{y:tks, x:tks.map(t=>totals[t]), type:'bar', orientation:'h',
      marker:{color:tks.map(t=>totals[t]>=0?'#0d6f92':'#a12a2a')}}],
      PL({xaxis:{title:'Total Net PnL'},margin:{l:80}}), {responsive:true});
  }
  // Monthly heatmap
  const pm = D.per_name_monthly;
  if(pm.tickers.length && pm.months.length){
    const z = pm.tickers.map(t=>pm.months.map(m=>pm.values[t][m]||0));
    Plotly.newPlot('chart-pername-monthly', [{z:z, x:pm.months, y:pm.tickers, type:'heatmap',
      colorscale:'RdBu', zmid:0}], PL({xaxis:{title:'Month'},margin:{l:80}}), {responsive:true});
  }
  // Cost breakdown
  const cb = D.cost_breakdown;
  const cbTickers = Object.keys(cb);
  if(cbTickers.length){
    Plotly.newPlot('chart-cost-breakdown', [
      {x:cbTickers, y:cbTickers.map(t=>cb[t].gross), name:'Gross PnL', type:'bar', marker:{color:'#2f9e8f'}},
      {x:cbTickers, y:cbTickers.map(t=>-cb[t].cost), name:'Cost (neg)', type:'bar', marker:{color:'#a12a2a'}},
      {x:cbTickers, y:cbTickers.map(t=>cb[t].net), name:'Net PnL', type:'bar', marker:{color:'#0d6f92'}},
    ], PL({barmode:'group',yaxis:{title:'PnL'}}), {responsive:true});
  }
  // Cost sensitivity
  const cs = D.cost_sensitivity;
  if(cs.dates.length){
    const traces = cs.multipliers.map((m,i)=>({
      x:cs.dates, y:cs.series[m], name:m+'x cost', line:{color:COLORS[i%COLORS.length],width:m==='1'?2.5:1.5, dash:m==='1'?'solid':'dot'}
    }));
    Plotly.newPlot('chart-cost-sens', traces, PL({yaxis:{title:'Cumulative PnL'}}), {responsive:true});
  }
  // Gross-to-net ratio
  const gn = D.gross_to_net_ratio;
  if(gn && gn.dates && gn.dates.length){
    Plotly.newPlot('chart-gross-net-ratio', [{x:gn.dates, y:gn.ratio, line:{color:'#cc7000',width:2}}],
      PL({yaxis:{title:'Cost/|Gross| (21d rolling)'}}), {responsive:true});
  }
}

// ==================== TAB: Signal & Skew ====================
function renderSignal(el){
  let html = `<div class="banner info">
    <strong>Signal construction:</strong>
    (1) Fit SSVI surface &rarr; extract 30-day ATM skew per name.
    (2) Regress stock skew on factor model (sector/PCA) &rarr; residual = idiosyncratic skew.
    (3) Z-score the residual (${D.config?.signal_config?.zscore_window||60}d rolling, MAD-robust).
    (4) Enter when |z| &ge; ${D.entry_z||1.0}, exit when |z| &lt; ${D.exit_z||0.25}.
    Filters: regime (sector vol), liquidity (option data points), edge &gt; cost.
  </div>`;
  html += card('Raw Skew Time Series (SSVI 30d ATM, per name + XLF sector)', 'chart-skew-ts');
  html += card('Residual Z-Score (signal that drives entry/exit)', 'chart-zscore-ts');
  html += card('Z-Score Distribution (should be ~N(0,1) if well-calibrated)', 'chart-zscore-dist');
  html += card('Signal Gate Pass Rates (what % of days pass each filter?)', 'chart-gate-rates');
  html += card('Residual Autocorrelation (lag 1-10 — persistence = tradeable)', 'chart-resid-acf');
  html += card('Residual Cross-Correlation (low = diversification)', 'chart-resid-corr');
  html += card('Signal Predictive Power: z(t) vs Δresidual(t+5) — does the signal predict reversion?', 'chart-sig-pred');
  el.innerHTML = html;

  // Skew time series
  const skew = D.skew_series;
  const skewTickers = Object.keys(skew);
  if(skewTickers.length){
    const traces = skewTickers.map((t,i)=>({
      x:skew[t].dates, y:skew[t].skew, name:t, line:{color:COLORS[i%COLORS.length],width:1.5}
    }));
    Plotly.newPlot('chart-skew-ts', traces, PL({xaxis:{rangeslider:{visible:true}},yaxis:{title:'Skew'}}), {responsive:true});
  }
  // Z-score time series
  const zs = D.zscore_series;
  const zsTickers = Object.keys(zs);
  if(zsTickers.length){
    const traces = zsTickers.map((t,i)=>({
      x:zs[t].dates, y:zs[t].zscore, name:t, line:{color:COLORS[i%COLORS.length],width:1.5}
    }));
    const entryZ = D.entry_z||1;
    const exitZ = D.exit_z||0.25;
    const shapes = [
      {type:'line',x0:0,x1:1,xref:'paper',y0:entryZ,y1:entryZ,line:{color:'#a12a2a',width:1,dash:'dash'}},
      {type:'line',x0:0,x1:1,xref:'paper',y0:-entryZ,y1:-entryZ,line:{color:'#a12a2a',width:1,dash:'dash'}},
      {type:'line',x0:0,x1:1,xref:'paper',y0:exitZ,y1:exitZ,line:{color:'#9a6a00',width:1,dash:'dot'}},
      {type:'line',x0:0,x1:1,xref:'paper',y0:-exitZ,y1:-exitZ,line:{color:'#9a6a00',width:1,dash:'dot'}},
    ];
    Plotly.newPlot('chart-zscore-ts', traces, PL({shapes,yaxis:{title:'Z-Score'}}), {responsive:true});
  }
  // Z-score distribution
  const allZ = D.all_zscores.filter(v=>v!=null);
  if(allZ.length){
    Plotly.newPlot('chart-zscore-dist', [{x:allZ, type:'histogram', nbinsx:50, marker:{color:'rgba(13,111,146,0.7)'}}],
      PL({xaxis:{title:'Z-Score'},yaxis:{title:'Count'}}), {responsive:true});
  }
  // Gate rates
  const gr = D.gate_rates;
  const grTickers = Object.keys(gr);
  if(grTickers.length){
    const gates = ['trade_allowed','regime_pass','liquidity_pass'];
    const traces = gates.map((g,i)=>({
      x:grTickers, y:grTickers.map(t=>(gr[t][g]||0)*100), name:g, type:'bar', marker:{color:COLORS[i]}
    }));
    Plotly.newPlot('chart-gate-rates', traces, PL({barmode:'group',yaxis:{title:'Pass Rate (%)'}}), {responsive:true});
  }
  // Residual autocorrelation
  const acf = D.residual_autocorr;
  const acfTickers = Object.keys(acf);
  if(acfTickers.length){
    const lags = Array.from({length:10},(_,i)=>i+1);
    const traces = acfTickers.map((t,i)=>({
      x:lags, y:acf[t], name:t, type:'bar', marker:{color:COLORS[i%COLORS.length]}
    }));
    Plotly.newPlot('chart-resid-acf', traces, PL({barmode:'group',xaxis:{title:'Lag',dtick:1},yaxis:{title:'Autocorrelation'}}), {responsive:true});
  }
  // Residual correlation matrix
  const rc = D.residual_corr_matrix;
  if(rc.tickers.length){
    Plotly.newPlot('chart-resid-corr', [{z:rc.matrix, x:rc.tickers, y:rc.tickers, type:'heatmap',
      colorscale:'RdBu', zmid:0, zmin:-1, zmax:1, text:rc.matrix.map(r=>r.map(v=>v!=null?v.toFixed(2):'')),
      texttemplate:'%{text}', colorbar:{title:'Corr'}}],
      PL({width:400,height:400,margin:{l:80}}), {responsive:true});
  }
  // Signal predictiveness
  const sp = D.signal_predictiveness;
  const spTickers = Object.keys(sp);
  if(spTickers.length){
    const traces = spTickers.map((t,i)=>({
      x:sp[t].z, y:sp[t].fwd, mode:'markers', name:`${t} (R²=${fmt(sp[t].r2,3)})`,
      marker:{color:COLORS[i%COLORS.length], size:4, opacity:0.5}
    }));
    Plotly.newPlot('chart-sig-pred', traces, PL({xaxis:{title:'z-score(t)'},yaxis:{title:'Δresidual(t+5)'}}), {responsive:true});
  }
}

// ==================== TAB: Vol Surface ====================
function renderSurface(el){
  const sk = D.surface_kpis||{};
  let html = `<div class="card"><h3>Calibration KPIs</h3><div class="grid">
    <div class="kpi"><div class="label">Median RMSE(IV)</div><div class="value">${fmt(sk.median_rmse,4)}</div></div>
    <div class="kpi"><div class="label">P95 RMSE(IV)</div><div class="value">${fmt(sk.p95_rmse,4)}</div></div>
    <div class="kpi"><div class="label">Butterfly Viol Rate</div><div class="value">${pct(sk.butterfly_viol_rate)}</div></div>
    <div class="kpi"><div class="label">Avg Fit Points</div><div class="value">${fmt(sk.avg_fit_points,0)}</div></div>
  </div></div>`;
  html += `<div class="banner info">
    <strong>Why this matters:</strong> The SSVI surface calibration quality directly affects signal quality.
    Poor fits (high RMSE) produce noisy skew estimates &rarr; noisy residuals &rarr; false signals.
    Butterfly arbitrage violations indicate the fitted surface is not arbitrage-free.
  </div>`;
  html += card('RMSE(IV) Over Time — how well does SSVI fit market option prices?', 'chart-rmse-ts');
  html += card('SSVI Parameters (rho=skew, eta=curvature, gamma=ATM term structure)', 'chart-params');
  html += card('No-Butterfly Margin (>0 = arbitrage-free, <0 = violation)', 'chart-nobfly');
  html += card('RMSE Distribution (All Fits)', 'chart-rmse-dist');
  html += card('Fit Points Per Day (option data availability)', 'chart-fitpts');
  el.innerHTML = html;

  const skew = D.skew_series;
  const tickers = Object.keys(skew);
  // RMSE
  if(tickers.length){
    const traces = tickers.map((t,i)=>({
      x:skew[t].dates, y:skew[t].rmse, name:t, line:{color:COLORS[i%COLORS.length],width:1.5}
    }));
    Plotly.newPlot('chart-rmse-ts', traces, PL({yaxis:{title:'RMSE(IV)'}}), {responsive:true});
  }
  // Params: use subplots approach with 3 rows
  if(tickers.length){
    const paramDiv = document.getElementById('chart-params');
    paramDiv.innerHTML = '<div id="chart-rho" style="height:220px"></div><div id="chart-eta" style="height:220px"></div><div id="chart-gamma" style="height:220px"></div>';
    ['rho','eta','gamma'].forEach((p,pi)=>{
      const traces = tickers.map((t,i)=>({
        x:skew[t].dates, y:skew[t][p], name:t, line:{color:COLORS[i%COLORS.length],width:1.5},
        showlegend:pi===0
      }));
      Plotly.newPlot('chart-'+p, traces, PL({yaxis:{title:p},margin:{t:20,b:30}}), {responsive:true});
    });
  }
  // No-butterfly margin
  if(tickers.length){
    const traces = tickers.map((t,i)=>({
      x:skew[t].dates, y:skew[t].no_bfly, name:t, line:{color:COLORS[i%COLORS.length],width:1.5}
    }));
    traces.push({x:[skew[tickers[0]].dates[0],skew[tickers[0]].dates[skew[tickers[0]].dates.length-1]],
      y:[0,0], mode:'lines', name:'Violation Boundary', line:{color:'#a12a2a',width:2,dash:'dash'}, showlegend:true});
    Plotly.newPlot('chart-nobfly', traces, PL({yaxis:{title:'Margin'}}), {responsive:true});
  }
  // RMSE dist
  if(sk.rmse_all && sk.rmse_all.length){
    const vals = sk.rmse_all.filter(v=>v!=null);
    Plotly.newPlot('chart-rmse-dist', [{x:vals, type:'histogram', nbinsx:50, marker:{color:'rgba(47,158,143,0.7)'}}],
      PL({xaxis:{title:'RMSE(IV)'},yaxis:{title:'Count'}}), {responsive:true});
  }
  // Fit points
  if(tickers.length){
    const traces = tickers.map((t,i)=>({
      x:skew[t].dates, y:skew[t].n_points, name:t, line:{color:COLORS[i%COLORS.length],width:1.5}
    }));
    Plotly.newPlot('chart-fitpts', traces, PL({yaxis:{title:'Points/Day'}}), {responsive:true});
  }
}

// ==================== TAB: Robustness ====================
function renderRobustness(el){
  let html = '';
  html += `<div class="banner info">
    <strong>In-Sample Warning:</strong> All parameters were selected on this period. There is no out-of-sample validation.
    Interpret all metrics with extreme caution.
  </div>`;
  html += card('Rolling Sharpe (63d & 126d)', 'chart-roll-sharpe');
  html += card('Rolling Hit Rate (63d)', 'chart-roll-hr');

  // Drawdown periods table
  const ddp = D.drawdown_periods;
  if(ddp.length){
    html += cardHtml('Drawdown Periods (Top 10 by Depth)', `<table class="tbl">
      <tr><th>Start</th><th>Trough</th><th>Recovery</th><th>Depth</th><th>Duration (days)</th></tr>
      ${ddp.map(d=>`<tr><td>${d.start}</td><td>${d.trough}</td><td>${d.recovery}</td><td>${fmt(d.depth,4)}</td><td>${d.duration_days}</td></tr>`).join('')}
    </table>`);
  }
  html += card('Capacity Analysis (Sharpe vs Capital)', 'chart-capacity');

  // Yearly performance
  const yp = D.yearly_performance;
  if(yp.length){
    html += cardHtml('Year-by-Year Performance', `<table class="tbl">
      <tr><th>Year</th><th>Days</th><th>Net PnL</th><th>Sharpe</th><th>Max DD</th><th>Hit Rate</th></tr>
      ${yp.map(y=>`<tr><td>${y.year}</td><td>${y.n_days}</td><td>${fmt(y.net_pnl,4)}</td><td>${fmt(y.sharpe,2)}</td><td>${fmt(y.max_dd,4)}</td><td>${pct(y.hit_rate)}</td></tr>`).join('')}
    </table>`);
  }
  html += card('Monthly Returns Heatmap', 'chart-monthly-ret');
  el.innerHTML = html;

  // Rolling Sharpe
  const rs = D.rolling_sharpe;
  if(rs.dates.length){
    Plotly.newPlot('chart-roll-sharpe', [
      {x:rs.dates, y:rs.sharpe_63, name:'63d', line:{color:'#0d6f92',width:2}},
      {x:rs.dates, y:rs.sharpe_126, name:'126d', line:{color:'#7f56d9',width:2}},
    ], PL({yaxis:{title:'Annualized Sharpe'},shapes:[
      {type:'line',x0:0,x1:1,xref:'paper',y0:0,y1:0,line:{color:'#999',width:1,dash:'dash'}}
    ]}), {responsive:true});
  }
  // Rolling hit rate
  const rh = D.rolling_hit_rate;
  if(rh.dates.length){
    Plotly.newPlot('chart-roll-hr', [{x:rh.dates, y:rh.hit_rate_63, line:{color:'#2f9e8f',width:2}}],
      PL({yaxis:{title:'Hit Rate (63d)',range:[0,1]},shapes:[
        {type:'line',x0:0,x1:1,xref:'paper',y0:0.5,y1:0.5,line:{color:'#999',width:1,dash:'dash'}}
      ]}), {responsive:true});
  }
  // Capacity
  const cap = D.capacity;
  if(cap.capitals.length){
    Plotly.newPlot('chart-capacity', [{x:cap.capitals, y:cap.sharpes, type:'bar',
      marker:{color:cap.sharpes.map(s=>s!=null&&s>2?'#0d6f92':'#cc7000')}}],
      PL({xaxis:{title:'Capital'},yaxis:{title:'Est. Sharpe'}}), {responsive:true});
  }
  // Monthly returns heatmap
  const mr = D.monthly_returns;
  if(mr.years.length){
    const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    Plotly.newPlot('chart-monthly-ret', [{z:mr.grid, x:monthNames, y:mr.years.map(String), type:'heatmap',
      colorscale:'RdBu', zmid:0, text:mr.grid.map(r=>r.map(v=>v!=null?v.toFixed(3):'')),
      texttemplate:'%{text}', colorbar:{title:'PnL'}}],
      PL({yaxis:{autorange:'reversed'},margin:{l:60}}), {responsive:true});
  }
}

// ==================== TAB: Hedging & Risk ====================
function renderHedging(el){
  const cfg = D.config||{};
  const sc = cfg.signal_config||{};
  const fm = cfg.factor_model||'sector';
  const proxy = cfg.sector_proxy||'XLF';
  let html = '';

  // Explanation
  html += `<div class="card"><h3>Hedging Architecture</h3>
    <div style="font-size:13px;line-height:1.7;">
    <p>The strategy isolates <strong>idiosyncratic skew</strong> by regressing each stock's IV skew against a factor model,
    then trades only the <em>residual</em>. This hedges out systematic skew exposures:</p>
    <table class="tbl" style="max-width:700px;">
      <tr><th>Exposure</th><th>How Hedged</th><th>Residual Risk</th></tr>
      <tr><td>Sector skew (${proxy})</td>
          <td>${fm==='pca'?'PCA factor regression ('+cfg.n_pca_factors+' components)':'OLS regression: stock_skew = &alpha; + &beta; &times; sector_skew'}</td>
          <td>Residual not explained by factor model</td></tr>
      <tr><td>Option delta</td>
          <td>Assumed continuous delta-hedging (cost modeled as gamma drag = ${pct(D.config?.transaction_cost_config?.hedge_drag_gamma||0.01)}/day)</td>
          <td>Discrete hedging slippage, gap risk</td></tr>
      <tr><td>Vega (parallel IV shift)</td>
          <td>Skew trade is long+short options at different strikes &rarr; ~vega-neutral to parallel shifts</td>
          <td>Non-parallel vol moves (term structure, smile shift)</td></tr>
      <tr><td>Overall market</td>
          <td>Long/short skew across names &rarr; net exposure near zero (see Exposure tab)</td>
          <td>Correlation between names in stress</td></tr>
    </table>
    <p style="margin-top:10px;"><strong>Key assumption:</strong> The factor model captures enough systematic skew variation that the residual is truly idiosyncratic and mean-reverting.
    The R&sup2; charts below show how much skew variance the model explains &mdash; low R&sup2; means the hedge is weak.</p>
    </div>
  </div>`;

  // Variance decomposition table
  const vd = D.variance_decomposition||[];
  if(vd.length){
    html += cardHtml('Skew Variance Decomposition: Systematic vs Idiosyncratic', `<table class="tbl">
      <tr><th>Ticker</th><th>Total Var</th><th>Systematic Var</th><th>Idiosyncratic Var</th><th>R&sup2; (% hedged)</th></tr>
      ${vd.map(r=>`<tr><td>${r.ticker}</td><td>${fmt(r.var_total,6)}</td><td>${fmt(r.var_systematic,6)}</td>
        <td>${fmt(r.var_idiosyncratic,6)}</td>
        <td><strong style="color:${r.r2>0.3?'var(--pass)':'var(--fail)'}">${pct(r.r2/100)}</strong> ${r.r2<30?'<span class="pill warn">Weak hedge</span>':''}</td></tr>`).join('')}
    </table>`);
  }

  html += card('Factor Model R&sup2; Over Time (rolling 60d) &mdash; higher = more variance hedged', 'chart-r2-ts');
  html += card('Factor Model &beta; Over Time (sensitivity to sector/factor skew)', 'chart-beta-ts');
  html += card('Stock Skew vs Systematic Component (does the model track?)', 'chart-skew-decomp');
  html += card('Realized Portfolio Vol (21d rolling) vs Vol Target', 'chart-realized-vol');

  // Unhedged exposures warning
  html += `<div class="card"><h3>Residual (Unhedged) Exposures</h3>
    <div style="font-size:13px;line-height:1.7;">
    <table class="tbl" style="max-width:700px;">
      <tr><th>Risk</th><th>Status</th><th>Mitigation</th></tr>
      <tr><td>Delta (stock price)</td><td><span class="pill warn">NOT HEDGED IN BACKTEST</span></td>
          <td>Cost modeled as gamma drag (${pct(D.config?.transaction_cost_config?.hedge_drag_gamma||0.01)}/day &times; |position|). Real implementation requires daily delta rebalancing.</td></tr>
      <tr><td>Cross-name correlation</td><td><span class="pill warn">PARTIALLY</span></td>
          <td>Cross-sectional top-K/bottom-K selection + gross leverage cap ${fmt(sc.max_gross_leverage||1.5,1)}&times;</td></tr>
      <tr><td>Regime/crisis</td><td><span class="pill pass">FILTERED</span></td>
          <td>Regime filter: sector |z| &lt; ${sc.regime_sector_abs_z_max||3.0}, sector vol z &lt; ${sc.regime_sector_vol_z_max||2.0}</td></tr>
      <tr><td>Liquidity</td><td><span class="pill pass">FILTERED</span></td>
          <td>Min ${sc.min_liquidity_points||150} option data points, RMSE cap ${sc.max_fit_rmse_iv||0.6}</td></tr>
      <tr><td>Position concentration</td><td><span class="pill pass">CAPPED</span></td>
          <td>Max ${pct(sc.max_name_weight||0.35)} per name, vol target ${sc.vol_target_daily||0.02} daily</td></tr>
    </table>
    </div>
  </div>`;

  el.innerHTML = html;

  // R² time series
  const fd = D.factor_diagnostics||{};
  const fdTickers = Object.keys(fd);
  if(fdTickers.length){
    const traces = fdTickers.map((t,i)=>({
      x:fd[t].dates, y:fd[t].r2_rolling, name:t, line:{color:COLORS[i%COLORS.length],width:2}
    }));
    Plotly.newPlot('chart-r2-ts', traces, PL({yaxis:{title:'R² (factor model)',range:[0,1]},
      shapes:[{type:'line',x0:0,x1:1,xref:'paper',y0:0.3,y1:0.3,line:{color:'#a12a2a',width:1,dash:'dash'}}],
      annotations:[{x:0.02,y:0.3,xref:'paper',text:'Weak hedge threshold',showarrow:false,font:{size:10,color:'#a12a2a'},yshift:10}]
    }), {responsive:true});
  }
  // Beta time series
  if(fdTickers.length){
    const traces = fdTickers.map((t,i)=>({
      x:fd[t].dates, y:fd[t].beta, name:t, line:{color:COLORS[i%COLORS.length],width:2}
    }));
    Plotly.newPlot('chart-beta-ts', traces, PL({yaxis:{title:'&beta; (factor loading)'},
      shapes:[{type:'line',x0:0,x1:1,xref:'paper',y0:1,y1:1,line:{color:'#999',width:1,dash:'dot'}},
              {type:'line',x0:0,x1:1,xref:'paper',y0:0,y1:0,line:{color:'#999',width:1,dash:'dot'}}]
    }), {responsive:true});
  }
  // Skew decomposition: pick first ticker with data
  if(fdTickers.length){
    const t = fdTickers[0];
    const d = fd[t];
    Plotly.newPlot('chart-skew-decomp', [
      {x:d.dates, y:d.stock_skew, name:t+' stock skew', line:{color:'#0d6f92',width:2}},
      {x:d.dates, y:d.systematic, name:t+' systematic (hedged)', line:{color:'#cc7000',width:2,dash:'dash'}},
      {x:d.dates, y:d.residual, name:t+' residual (traded)', line:{color:'#a12a2a',width:2}},
    ], PL({yaxis:{title:'Skew'},
      annotations:[{x:0.5,y:1.05,xref:'paper',yref:'paper',text:'Showing: '+t+' (first ticker)',showarrow:false,font:{size:11,color:'#51617a'}}]
    }), {responsive:true});
    // Add dropdown for other tickers if >1
    if(fdTickers.length>1){
      const div = document.getElementById('chart-skew-decomp');
      const sel = document.createElement('select');
      sel.style.cssText='margin:8px 0;padding:4px 8px;font-size:12px;border:1px solid #cfdced;border-radius:4px;';
      fdTickers.forEach(tk=>{const o=document.createElement('option');o.value=tk;o.text=tk;sel.appendChild(o);});
      sel.onchange=function(){
        const tk=this.value; const dd=fd[tk];
        Plotly.react('chart-skew-decomp', [
          {x:dd.dates, y:dd.stock_skew, name:tk+' stock skew', line:{color:'#0d6f92',width:2}},
          {x:dd.dates, y:dd.systematic, name:tk+' systematic (hedged)', line:{color:'#cc7000',width:2,dash:'dash'}},
          {x:dd.dates, y:dd.residual, name:tk+' residual (traded)', line:{color:'#a12a2a',width:2}},
        ], PL({yaxis:{title:'Skew'},annotations:[{x:0.5,y:1.05,xref:'paper',yref:'paper',text:'Showing: '+tk,showarrow:false,font:{size:11,color:'#51617a'}}]}));
      };
      div.parentNode.insertBefore(sel, div);
    }
  }
  // Realized vol
  const rv = D.realized_vol||{};
  if(rv.dates && rv.dates.length){
    const volTarget = D.vol_target_daily||0.02;
    Plotly.newPlot('chart-realized-vol', [
      {x:rv.dates, y:rv.realized_vol, name:'Realized Vol (21d)', line:{color:'#0d6f92',width:2}},
    ], PL({yaxis:{title:'Daily Vol'},
      shapes:[{type:'line',x0:0,x1:1,xref:'paper',y0:volTarget,y1:volTarget,line:{color:'#2f9e8f',width:2,dash:'dash'}}],
      annotations:[{x:0.98,y:volTarget,xref:'paper',text:'Vol Target: '+volTarget,showarrow:false,font:{size:10,color:'#2f9e8f'},yshift:12}]
    }), {responsive:true});
  }
}

// ==================== TAB: Cost & Execution ====================
function renderCostExec(el){
  const cm = D.cost_model||{};
  const cmCfg = cm.config||{};
  const tc = D.config?.transaction_cost_config||{};
  let html = '';

  // Cost model explanation
  html += `<div class="card"><h3>Transaction Cost Model</h3>
    <div style="font-size:13px;line-height:1.7;">
    <p>Each day, the cost for each name is:</p>
    <pre style="font-size:14px;background:#f0f5fa;padding:12px;border-radius:8px;margin:8px 0;">cost = turnover &times; (estimated_unit_cost + hedge_drag_gamma &times; |position|)</pre>
    <p>Where <strong>turnover</strong> = |new_weight &minus; old_weight| (how much the position changed).</p>
    <div class="grid" style="grid-template-columns:1fr 1fr;">
      <div>
        <h4 style="margin:8px 0 4px">Spread + Commission (per side)</h4>
        <table class="tbl">
          <tr><td>Half-spread (option bid-ask)</td><td><strong>${pct(cmCfg.half_spread_cost)}</strong></td></tr>
          <tr><td>Commission</td><td><strong>${pct(cmCfg.commission_cost)}</strong></td></tr>
          <tr><td>Market impact</td><td><strong>${pct(cmCfg.impact_cost)}</strong></td></tr>
          <tr><td>Round-trip total</td><td><strong>${pct(cmCfg.round_trip_cost)}</strong></td></tr>
        </table>
      </div>
      <div>
        <h4 style="margin:8px 0 4px">Estimated Unit Cost (adaptive)</h4>
        <pre style="font-size:12px;margin:4px 0;">est_unit_cost = (round_trip / 2) &times; scale
scale = 1 + liquidity_pen + fit_pen
liquidity_pen = 0.5 &times; &radic;(150 / n_points)
fit_pen = min(1.5, rmse_iv / 0.25)</pre>
        <p style="font-size:12px;color:var(--muted);">Illiquid options (few data points) and poor SSVI fits (high RMSE) increase estimated costs.</p>
      </div>
    </div>
    <h4 style="margin:12px 0 4px">Gamma Drag (delta-hedging cost)</h4>
    <pre style="font-size:12px;margin:4px 0;">gamma_cost = hedge_drag_gamma &times; |position| = ${cmCfg.hedge_drag_gamma} &times; |w| per day</pre>
    <p style="font-size:12px;color:var(--muted);">Approximates the cost of daily delta-rebalancing. A position of 50% costs ${fmt((cmCfg.hedge_drag_gamma||0.01)*0.5*100,2)}bps/day in gamma drag alone.
    This is a linear approximation; real gamma P&amp;L scales as &frac12;&Gamma;&times;(&Delta;S)&sup2;.</p>

    <h4 style="margin:12px 0 4px">Edge-Cost Buffer</h4>
    <pre style="font-size:12px;margin:4px 0;">trade_allowed only if: expected_edge > est_unit_cost &times; (1 + ${D.config?.signal_config?.edge_cost_buffer||0.25})
expected_edge = |residual| / ${D.config?.signal_config?.edge_horizon_days||5} days</pre>
    <p style="font-size:12px;color:var(--muted);">A ${pct(D.config?.signal_config?.edge_cost_buffer||0.25)} buffer is required above breakeven to account for execution slippage and estimation error.</p>
    </div>
  </div>`;

  // Cost breakdown per ticker
  const cmPt = cm.per_ticker||{};
  const cmTickers = Object.keys(cmPt);
  if(cmTickers.length){
    html += cardHtml('Cost Breakdown Per Ticker', `<table class="tbl">
      <tr><th>Ticker</th><th>Total Cost</th><th>Total Turnover</th><th>Avg Cost/Turnover</th><th>Est. Spread+Comm</th><th>Est. Gamma Drag</th><th>Cost % of |Gross|</th></tr>
      ${cmTickers.map(t=>{const r=cmPt[t];return `<tr><td>${t}</td><td>${fmt(r.total_cost,4)}</td><td>${fmt(r.total_turnover,2)}</td>
        <td>${fmt(r.avg_cost_per_turnover,4)}</td><td>${fmt(r.approx_spread_commission,4)}</td><td>${fmt(r.approx_gamma_drag,4)}</td>
        <td>${pct(r.cost_pct_of_gross)}</td></tr>`;}).join('')}
    </table>`);
  }

  html += card('Estimated Unit Cost Over Time (data-driven, per ticker)', 'chart-unit-cost-ts');
  html += card('Edge vs Cost Scatter (does expected profit exceed cost?)', 'chart-edge-cost');
  html += card('Portfolio Turnover Over Time', 'chart-turnover-ts');
  html += card('Per-Ticker Turnover Over Time', 'chart-turnover-ticker');
  html += card('Hold Period Distribution (consecutive days in position)', 'chart-hold-dist');

  el.innerHTML = html;

  // Unit cost time series
  const ucts = D.unit_cost_ts||{};
  const ucTickers = Object.keys(ucts);
  if(ucTickers.length){
    const traces = ucTickers.map((t,i)=>({
      x:ucts[t].dates, y:ucts[t].unit_cost, name:t, line:{color:COLORS[i%COLORS.length],width:1.5}
    }));
    Plotly.newPlot('chart-unit-cost-ts', traces, PL({yaxis:{title:'Estimated Unit Cost'}}), {responsive:true});
  }
  // Edge vs cost scatter
  const evc = D.edge_vs_cost||{};
  const evcTickers = Object.keys(evc.per_ticker||{});
  if(evcTickers.length){
    const traces = evcTickers.map((t,i)=>{
      const d = evc.per_ticker[t];
      return {x:d.cost, y:d.edge, mode:'markers', name:`${t} (pass ${pct(d.pass_rate)})`,
        marker:{color:d.allowed.map(a=>a?COLORS[i%COLORS.length]:'#ccc'), size:4, opacity:0.5}};
    });
    // Add 45-degree line (edge = cost)
    const allC = evc.all_costs.filter(v=>v!=null);
    const maxC = allC.length?Math.max(...allC)*1.2:0.1;
    traces.push({x:[0,maxC],y:[0,maxC], mode:'lines', name:'Breakeven (edge=cost)',
      line:{color:'#a12a2a',width:2,dash:'dash'}, showlegend:true});
    // Edge = cost * (1+buffer) line
    const buf = D.config?.signal_config?.edge_cost_buffer||0.25;
    traces.push({x:[0,maxC],y:[0,maxC*(1+buf)], mode:'lines', name:`Required (edge=cost×${1+buf})`,
      line:{color:'#9a6a00',width:1.5,dash:'dot'}, showlegend:true});
    Plotly.newPlot('chart-edge-cost', traces, PL({
      xaxis:{title:'Estimated Unit Cost'},yaxis:{title:'Expected Edge (|residual|/5d)'},
      annotations:[{x:0.02,y:0.95,xref:'paper',yref:'paper',
        text:'Points above orange line = trade allowed',showarrow:false,font:{size:10,color:'#51617a'}}]
    }), {responsive:true});
  }
  // Portfolio turnover
  const ta = D.turnover_analysis||{};
  if(ta.portfolio_dates && ta.portfolio_dates.length){
    Plotly.newPlot('chart-turnover-ts', [
      {x:ta.portfolio_dates, y:ta.portfolio_turnover, line:{color:'#0d6f92',width:2}, name:'Daily Portfolio Turnover'},
    ], PL({yaxis:{title:'Turnover (sum of |Δweight|)'}}), {responsive:true});
  }
  // Per-ticker turnover
  const taPt = ta.per_ticker||{};
  const taTickers = Object.keys(taPt);
  if(taTickers.length){
    const traces = taTickers.map((t,i)=>({
      x:taPt[t].dates, y:taPt[t].turnover, name:`${t} (total: ${fmt(taPt[t].total,2)})`,
      line:{color:COLORS[i%COLORS.length],width:1.5}
    }));
    Plotly.newPlot('chart-turnover-ticker', traces, PL({yaxis:{title:'Turnover'}}), {responsive:true});
  }
  // Hold period distribution
  const hp = D.hold_periods||{};
  if(hp.durations && hp.durations.length){
    Plotly.newPlot('chart-hold-dist', [{x:hp.durations, type:'histogram', nbinsx:30,
      marker:{color:'rgba(47,158,143,0.7)'},
    }], PL({xaxis:{title:'Hold Period (days)'},yaxis:{title:'Count'},
      annotations:[{x:0.98,y:0.95,xref:'paper',yref:'paper',
        text:`Median: ${hp.durations.sort((a,b)=>a-b)[Math.floor(hp.durations.length/2)]}d, n=${hp.durations.length} trades`,
        showarrow:false,font:{size:11}}]
    }), {responsive:true});
  }
}

// ==================== TAB: Validation & Data ====================
function renderValidation(el){
  let html = '';
  // Validation checks
  const vc = D.validation_checks;
  if(vc.length){
    html += cardHtml('Automated Validation Checks', `<table class="tbl">
      <tr><th>Check</th><th>Status</th><th>Value</th><th>Details</th></tr>
      ${vc.map(c=>`<tr><td>${c.check}</td><td><span class="pill ${c.status}">${c.status.toUpperCase()}</span></td><td>${fmt(c.value)}</td><td>${c.details}</td></tr>`).join('')}
    </table>`);
  }
  // Pipeline funnel
  html += card('Pipeline Bottleneck Funnel', 'chart-funnel');

  // Bottleneck table
  const bt = D.bottleneck_table;
  if(bt.length){
    html += cardHtml('Per-Ticker Bottleneck Table', `<table class="tbl">
      <tr><th>Ticker</th><th>Total Days</th><th>Skew</th><th>Signal</th><th>Regime</th><th>Liquidity</th><th>Allowed</th><th>Executed</th></tr>
      ${bt.map(r=>`<tr><td>${r.ticker}</td><td>${r.total_days}</td><td>${r.skew_rows}</td><td>${r.signal_rows}</td><td>${r.regime_pass_rows}</td><td>${r.liquidity_pass_rows}</td><td>${r.trade_allowed_rows}</td><td>${r.exec_active_rows}</td></tr>`).join('')}
    </table>`);
  }
  // Universe
  const uni = D.universe;
  if(uni.length){
    html += cardHtml('Universe Coverage', `<table class="tbl">
      <tr><th>Ticker</th><th>Skew Rows</th><th>Signal Rows</th><th>Backtest Rows</th><th>Start</th><th>End</th></tr>
      ${uni.map(u=>`<tr><td>${u.ticker}</td><td>${u.skew_rows}</td><td>${u.signal_rows}</td><td>${u.backtest_rows}</td><td>${u.coverage_start}</td><td>${u.coverage_end}</td></tr>`).join('')}
    </table>`);
  }
  // Config
  html += cardHtml('Run Configuration', `<details><summary>Show/Hide Config</summary><pre>${JSON.stringify(D.config,null,2)}</pre></details>`);
  // Files
  const files = D.files;
  if(files.length){
    html += cardHtml('File Inventory', `<details><summary>${files.length} files</summary><table class="tbl">
      <tr><th>Path</th><th>Size (KB)</th></tr>
      ${files.map(f=>`<tr><td>${f.path}</td><td>${f.size_kb}</td></tr>`).join('')}
    </table></details>`);
  }
  el.innerHTML = html;

  // Funnel chart
  const fn = D.funnel;
  if(fn.length){
    Plotly.newPlot('chart-funnel', [{type:'funnel', y:fn.map(f=>f.stage), x:fn.map(f=>f.rows),
      textinfo:'value+percent initial', marker:{color:fn.map((_,i)=>COLORS[i%COLORS.length])}}],
      PL({margin:{l:180}}), {responsive:true});
  }
}

// ==================== Tab switching ====================
const buttons = [...document.querySelectorAll('.tab-btn')];
const panels = [...document.querySelectorAll('.tab-panel')];
function activate(name){
  buttons.forEach(b=>b.classList.toggle('active', b.dataset.tab===name));
  panels.forEach(p=>p.classList.toggle('active', p.id==='tab-'+name));
  renderTab(name);
}
buttons.forEach(b=>b.addEventListener('click', ()=>activate(b.dataset.tab)));
renderTab('summary');
</script>
</body>
</html>"""


# ----------------------------- HTML builder -----------------------------

def _build_html(
    results_dir: Path,
    summary: Dict[str, object],
    portfolio: pd.DataFrame,
    skew_map: Dict[str, pd.DataFrame],
    signal_map: Dict[str, pd.DataFrame],
    backtest_map: Dict[str, pd.DataFrame],
    snapshots: pd.DataFrame,
) -> str:
    data = _build_dashboard_data(
        results_dir, summary, portfolio, skew_map, signal_map, backtest_map, snapshots
    )
    json_blob = json.dumps(data, default=_safe, allow_nan=False)
    html = _HTML_TEMPLATE.replace("__DASHBOARD_DATA__", json_blob)
    html = html.replace("__RESULTS_DIR__", str(results_dir))
    return html


# ----------------------------- entry points -----------------------------

def main() -> None:
    args = parse_args()
    out = build_dashboard(Path(args.results_dir), Path(args.snapshots_dir))
    print(f"Wrote {out}")


def build_dashboard(results_dir: Path, snapshots_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(results_dir / "summary.json")
    portfolio = _read_csv(results_dir / "portfolio_backtest.csv")
    if portfolio.empty:
        portfolio = _read_csv(results_dir / "portfolio_backtest_tuned.csv")

    skew_map = _load_ticker_map(results_dir / "skew_series", "_skew")
    signal_map = _load_ticker_map(results_dir / "signals", "_signal")
    backtest_map = _load_ticker_map(results_dir / "backtests", "_backtest")

    snapshots = _snapshot_table(snapshots_dir)
    html = _build_html(
        results_dir=results_dir,
        summary=summary,
        portfolio=portfolio,
        skew_map=skew_map,
        signal_map=signal_map,
        backtest_map=backtest_map,
        snapshots=snapshots,
    )

    out = results_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
