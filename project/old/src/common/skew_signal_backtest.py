from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SignalConfig:
    regression_window: int = 60
    min_regression_obs: int = 40
    zscore_window: int = 60
    entry_z: float = 1.0
    exit_z: float = 0.25
    max_abs_position: float = 1.0
    winsor_quantile: float = 0.02
    use_mad_zscore: bool = True
    min_hold_days: int = 3
    edge_horizon_days: int = 5
    edge_cost_buffer: float = 0.25
    regime_sector_vol_window: int = 21
    regime_sector_abs_z_max: float = 3.0
    regime_sector_vol_z_max: float = 2.0
    min_liquidity_points: int = 150
    max_fit_rmse_iv: float = 0.60
    cross_section_top_k: int = 2
    cross_section_bottom_k: int = 2
    vol_target_daily: float = 0.02
    max_gross_leverage: float = 1.5
    max_name_weight: float = 0.35
    signal_direction: str = "mean_revert"  # mean_revert | momentum | auto
    auto_sign_window: int = 63


@dataclass
class TransactionCostConfig:
    half_spread_cost: float = 0.02
    impact_cost: float = 0.0
    commission_cost: float = 0.005
    hedge_drag_gamma: float = 0.01
    hurdle_rate: float = 0.0

    @property
    def round_trip_unit_cost(self) -> float:
        return 2.0 * (self.half_spread_cost + self.impact_cost + self.commission_cost)


def _winsorize(x: np.ndarray, q: float) -> np.ndarray:
    if len(x) == 0:
        return x
    lo = np.quantile(x, q)
    hi = np.quantile(x, 1.0 - q)
    return np.clip(x, lo, hi)


def _rolling_robust_ols_predict(
    y_hist: np.ndarray,
    x_hist: np.ndarray,
    x_now: float,
    winsor_q: float,
) -> tuple[float, float, float]:
    xw = _winsorize(np.asarray(x_hist, dtype=np.float64), winsor_q)
    yw = _winsorize(np.asarray(y_hist, dtype=np.float64), winsor_q)
    X = np.column_stack([np.ones(len(xw), dtype=np.float64), xw])
    beta, *_ = np.linalg.lstsq(X, yw, rcond=None)
    alpha, b = float(beta[0]), float(beta[1])
    y_hat = alpha + b * float(x_now)
    return alpha, b, y_hat


def _robust_center_scale(x: np.ndarray, use_mad: bool) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0, np.nan
    if use_mad:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = 1.4826 * mad
        return med, scale
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return mu, sd


def _estimate_unit_cost(
    n_points: float,
    rmse_iv: float,
    cost_cfg: TransactionCostConfig,
) -> float:
    n = max(float(n_points), 1.0)
    rmse = max(float(rmse_iv), 1e-6)
    liquidity_pen = 0.5 * np.sqrt(150.0 / n)
    fit_pen = min(1.5, rmse / 0.25)
    scale = 1.0 + liquidity_pen + fit_pen
    base = cost_cfg.round_trip_unit_cost / 2.0
    return float(base * scale)


def _sector_regime_flags(sector_skew: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    s = sector_skew[["date", "skew"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)
    w = max(10, int(cfg.regime_sector_vol_window))
    s["sector_roll_mu"] = s["skew"].rolling(w).mean()
    s["sector_roll_sd"] = s["skew"].rolling(w).std(ddof=0)
    s["sector_abs_z"] = (s["skew"] - s["sector_roll_mu"]) / s["sector_roll_sd"].replace(0.0, np.nan)
    s["sector_vol_level"] = s["sector_roll_sd"]
    vol_mu = s["sector_vol_level"].rolling(w).mean()
    vol_sd = s["sector_vol_level"].rolling(w).std(ddof=0)
    s["sector_vol_z"] = (s["sector_vol_level"] - vol_mu) / vol_sd.replace(0.0, np.nan)
    s["regime_pass"] = (
        s["sector_abs_z"].abs().fillna(0.0) <= cfg.regime_sector_abs_z_max
    ) & (
        s["sector_vol_z"].fillna(0.0) <= cfg.regime_sector_vol_z_max
    )
    return s[["date", "regime_pass", "sector_abs_z", "sector_vol_z", "sector_vol_level"]]


def build_residual_signal(
    stock_skew: pd.DataFrame,
    sector_skew: pd.DataFrame,
    signal_cfg: SignalConfig | None = None,
    cost_cfg: TransactionCostConfig | None = None,
    factor_model: str = "sector",
    universe_skews: Dict[str, pd.DataFrame] | None = None,
    n_pca_factors: int = 2,
) -> pd.DataFrame:
    cfg = signal_cfg or SignalConfig()
    cc = cost_cfg or TransactionCostConfig()

    cols = ["date", "ticker", "skew", "n_points", "rmse_implied_volatility"]
    s = stock_skew[cols].rename(
        columns={
            "skew": "stock_skew",
            "n_points": "stock_n_points",
            "rmse_implied_volatility": "stock_rmse_iv",
        }
    ).copy()
    m = sector_skew[["date", "skew"]].rename(columns={"skew": "sector_skew"}).copy()
    regime = _sector_regime_flags(sector_skew, cfg)

    if factor_model not in {"sector", "pca"}:
        raise ValueError("factor_model must be one of: sector, pca")

    if factor_model == "sector":
        df = s.merge(m, on="date", how="inner")
        df = df.merge(regime, on="date", how="left")
        df = df.sort_values("date").reset_index(drop=True)
    else:
        if not universe_skews:
            raise ValueError("universe_skews is required for factor_model='pca'")
        panel_parts = []
        for tk, sdf in universe_skews.items():
            sub = sdf[["date", "skew"]].copy()
            sub["date"] = pd.to_datetime(sub["date"])
            sub["ticker"] = tk
            panel_parts.append(sub)
        panel = pd.concat(panel_parts, ignore_index=True)
        wide = panel.pivot_table(index="date", columns="ticker", values="skew", aggfunc="last").sort_index()

        tkr = str(stock_skew["ticker"].iloc[0])
        if tkr not in wide.columns:
            return pd.DataFrame()
        pca_rows = []
        dates = list(wide.index)
        for i, d in enumerate(dates):
            start = max(0, i - cfg.regression_window)
            hist = wide.iloc[start:i].copy()
            if len(hist) < cfg.min_regression_obs:
                pca_rows.append({"date": d, "ticker": tkr, "pc1": np.nan, "pc2": np.nan})
                continue

            cols_ok = hist.columns[hist.notna().mean() >= 0.6]
            hist = hist[cols_ok]
            if tkr not in hist.columns or hist.shape[1] < 2:
                pca_rows.append({"date": d, "ticker": tkr, "pc1": np.nan, "pc2": np.nan})
                continue
            hist_f = hist.fillna(hist.mean())
            mu = hist_f.mean(axis=0)
            sd = hist_f.std(axis=0).replace(0.0, 1.0)
            z_hist = (hist_f - mu) / sd
            x = z_hist.to_numpy(dtype=np.float64)
            _, _, vt = np.linalg.svd(x, full_matrices=False)
            k = max(1, min(int(n_pca_factors), vt.shape[0], 3))
            load = vt[:k].T  # n_assets x k

            row_now = wide.loc[d, cols_ok].copy()
            row_now = row_now.fillna(mu)
            z_now = ((row_now - mu) / sd).to_numpy(dtype=np.float64).reshape(1, -1)
            f_now = z_now @ load
            pc = [float(f_now[0, j]) for j in range(k)]
            while len(pc) < 2:
                pc.append(np.nan)
            pca_rows.append({"date": d, "ticker": tkr, "pc1": pc[0], "pc2": pc[1]})

        pca_df = pd.DataFrame(pca_rows)
        df = s.merge(pca_df, on=["date", "ticker"], how="inner")
        df = df.merge(regime, on="date", how="left")
        df = df.sort_values("date").reset_index(drop=True)
        df["sector_skew"] = np.nan

    if df.empty:
        return df

    out: List[Dict[str, float | str | pd.Timestamp | bool]] = []
    residual_hist: List[float] = []
    for i in range(len(df)):
        date_i = pd.Timestamp(df.loc[i, "date"])
        if factor_model == "sector":
            x_now = float(df.loc[i, "sector_skew"])
            x_now_vec = None
        else:
            x_now_vec = np.asarray([df.loc[i, "pc1"], df.loc[i, "pc2"]], dtype=np.float64)
            x_now_vec = x_now_vec[np.isfinite(x_now_vec)]
            x_now = float(x_now_vec[0]) if len(x_now_vec) else np.nan
        y_now = float(df.loc[i, "stock_skew"])
        n_pts = float(df.loc[i, "stock_n_points"])
        rmse_iv = float(df.loc[i, "stock_rmse_iv"])

        start = max(0, i - cfg.regression_window)
        hist = df.iloc[start:i]
        if factor_model == "sector":
            hist = hist[np.isfinite(hist["stock_skew"]) & np.isfinite(hist["sector_skew"])]
        else:
            hist = hist[np.isfinite(hist["stock_skew"])]

        alpha = np.nan
        beta = np.nan
        resid = np.nan
        z = np.nan
        resid_scale = np.nan
        expected_edge = np.nan
        est_unit_cost = _estimate_unit_cost(n_pts, rmse_iv, cc)
        liquidity_pass = (n_pts >= cfg.min_liquidity_points) and (rmse_iv <= cfg.max_fit_rmse_iv)
        regime_pass = bool(df.loc[i, "regime_pass"]) if pd.notna(df.loc[i, "regime_pass"]) else True

        if len(hist) >= cfg.min_regression_obs:
            if factor_model == "sector":
                alpha, beta, y_hat = _rolling_robust_ols_predict(
                    y_hist=hist["stock_skew"].to_numpy(dtype=np.float64),
                    x_hist=hist["sector_skew"].to_numpy(dtype=np.float64),
                    x_now=x_now,
                    winsor_q=cfg.winsor_quantile,
                )
            else:
                x_hist = hist[["pc1", "pc2"]].copy()
                valid = x_hist.notna().all(axis=1)
                yv = hist.loc[valid, "stock_skew"].to_numpy(dtype=np.float64)
                xv = x_hist.loc[valid, ["pc1", "pc2"]].to_numpy(dtype=np.float64)
                if len(yv) >= cfg.min_regression_obs and np.isfinite(x_now_vec).all() and len(x_now_vec) == 2:
                    yv = _winsorize(yv, cfg.winsor_quantile)
                    xv[:, 0] = _winsorize(xv[:, 0], cfg.winsor_quantile)
                    xv[:, 1] = _winsorize(xv[:, 1], cfg.winsor_quantile)
                    X = np.column_stack([np.ones(len(yv)), xv])
                    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
                    alpha = float(coef[0])
                    beta = float(coef[1])
                    y_hat = float(alpha + coef[1] * x_now_vec[0] + coef[2] * x_now_vec[1])
                else:
                    y_hat = np.nan
            resid = y_now - y_hat
            if np.isfinite(resid):
                residual_hist.append(float(resid))

            if np.isfinite(resid) and len(residual_hist) >= cfg.zscore_window:
                tail = np.asarray(residual_hist[-cfg.zscore_window:], dtype=np.float64)
                center, scale = _robust_center_scale(tail, use_mad=cfg.use_mad_zscore)
                resid_scale = scale
                if scale > 1e-12:
                    z = (float(resid) - center) / scale
                    expected_edge = abs(float(resid)) / max(cfg.edge_horizon_days, 1)

        trade_allowed = (
            np.isfinite(z)
            and liquidity_pass
            and regime_pass
            and np.isfinite(expected_edge)
            and expected_edge > est_unit_cost * (1.0 + cfg.edge_cost_buffer)
        )

        out.append(
            {
                "date": date_i,
                "ticker": str(df.loc[i, "ticker"]),
                "stock_skew": y_now,
                "sector_skew": x_now,
                "factor_model": factor_model,
                "alpha": alpha,
                "beta": beta,
                "residual": resid,
                "residual_scale": resid_scale,
                "zscore": z,
                "stock_n_points": n_pts,
                "stock_rmse_iv": rmse_iv,
                "regime_pass": regime_pass,
                "liquidity_pass": liquidity_pass,
                "expected_edge": expected_edge,
                "estimated_unit_cost": est_unit_cost,
                "trade_allowed": bool(trade_allowed),
            }
        )
    return pd.DataFrame(out)


def _target_position_from_z(z: float, prev_pos: float, hold_days: int, cfg: SignalConfig) -> float:
    if not np.isfinite(z):
        return 0.0
    if abs(z) >= cfg.entry_z:
        return float(np.clip(-z, -cfg.max_abs_position, cfg.max_abs_position))
    if abs(z) <= cfg.exit_z:
        if hold_days >= cfg.min_hold_days:
            return 0.0
        return prev_pos
    return float(prev_pos)


def backtest_residual_mean_reversion(
    signal_df: pd.DataFrame,
    signal_cfg: SignalConfig | None = None,
    cost_cfg: TransactionCostConfig | None = None,
) -> pd.DataFrame:
    cfg = signal_cfg or SignalConfig()
    cost = cost_cfg or TransactionCostConfig()

    df = signal_df.sort_values("date").reset_index(drop=True).copy()
    if df.empty:
        return df

    positions = np.zeros(len(df), dtype=np.float64)
    turnover = np.zeros(len(df), dtype=np.float64)
    gross = np.zeros(len(df), dtype=np.float64)
    costs = np.zeros(len(df), dtype=np.float64)
    net = np.zeros(len(df), dtype=np.float64)
    holds = np.zeros(len(df), dtype=np.float64)

    prev_pos = 0.0
    hold_days = 0
    for i in range(len(df)):
        z = float(df.loc[i, "zscore"]) if np.isfinite(df.loc[i, "zscore"]) else np.nan
        allowed = bool(df.loc[i, "trade_allowed"]) if "trade_allowed" in df.columns else True
        if not allowed:
            pos = 0.0 if hold_days >= cfg.min_hold_days else prev_pos
        else:
            pos = _target_position_from_z(z, prev_pos=prev_pos, hold_days=hold_days, cfg=cfg)

        positions[i] = pos
        turnover[i] = abs(pos - prev_pos)
        if abs(pos) > 1e-12:
            hold_days = hold_days + 1 if abs(prev_pos) > 1e-12 else 1
        else:
            hold_days = 0
        holds[i] = hold_days

        if i + 1 < len(df) and np.isfinite(df.loc[i, "residual"]) and np.isfinite(df.loc[i + 1, "residual"]):
            d_resid = float(df.loc[i + 1, "residual"] - df.loc[i, "residual"])
            gross[i + 1] = pos * (-d_resid)

        est_unit_cost = (
            float(df.loc[i, "estimated_unit_cost"])
            if "estimated_unit_cost" in df.columns and np.isfinite(df.loc[i, "estimated_unit_cost"])
            else cost.round_trip_unit_cost / 2.0
        )
        tc = turnover[i] * (est_unit_cost + cost.hedge_drag_gamma * abs(pos))
        costs[i] = tc
        net[i] = gross[i] - tc
        prev_pos = pos

    df["position"] = positions
    df["hold_days"] = holds
    df["turnover"] = turnover
    df["gross_pnl"] = gross
    df["cost"] = costs
    df["net_pnl"] = net
    df["cum_net_pnl"] = np.cumsum(net)
    return df


def _build_date_row_maps(signal_map: Dict[str, pd.DataFrame]) -> Dict[str, Dict[pd.Timestamp, pd.Series]]:
    out: Dict[str, Dict[pd.Timestamp, pd.Series]] = {}
    for ticker, df in signal_map.items():
        rows = {pd.Timestamp(r["date"]): r for _, r in df.iterrows()}
        out[ticker] = rows
    return out


def _candidate_weights(
    rows: List[pd.Series],
    cfg: SignalConfig,
    direction_multiplier: float = 1.0,
) -> Dict[str, float]:
    long_cands: List[Tuple[str, float]] = []
    short_cands: List[Tuple[str, float]] = []
    for r in rows:
        z = float(r["zscore"]) if np.isfinite(r["zscore"]) else np.nan
        z = direction_multiplier * z if np.isfinite(z) else z
        if not np.isfinite(z):
            continue
        if not bool(r.get("trade_allowed", True)):
            continue
        if z <= -cfg.entry_z:
            long_cands.append((str(r["ticker"]), abs(z)))
        elif z >= cfg.entry_z:
            short_cands.append((str(r["ticker"]), abs(z)))

    long_cands = sorted(long_cands, key=lambda x: x[1], reverse=True)[: cfg.cross_section_top_k]
    short_cands = sorted(short_cands, key=lambda x: x[1], reverse=True)[: cfg.cross_section_bottom_k]

    w: Dict[str, float] = {}
    if long_cands:
        s = sum(x[1] for x in long_cands)
        for t, zabs in long_cands:
            w[t] = 0.5 * (zabs / max(s, 1e-12))
    if short_cands:
        s = sum(x[1] for x in short_cands)
        for t, zabs in short_cands:
            w[t] = w.get(t, 0.0) - 0.5 * (zabs / max(s, 1e-12))
    return w


def _infer_direction_multiplier_auto(
    current_date_idx: int,
    all_dates: List[pd.Timestamp],
    date_rows: Dict[str, Dict[pd.Timestamp, pd.Series]],
    window: int,
) -> float:
    if current_date_idx < 2:
        return 1.0
    start = max(0, current_date_idx - window)
    dates = all_dates[start:current_date_idx]
    vals = []
    for d in dates:
        for t, rows in date_rows.items():
            r0 = rows.get(d)
            i = all_dates.index(d)
            if i + 1 >= len(all_dates):
                continue
            r1 = rows.get(all_dates[i + 1])
            if r0 is None or r1 is None:
                continue
            z = float(r0.get("zscore", np.nan))
            e0 = float(r0.get("residual", np.nan))
            e1 = float(r1.get("residual", np.nan))
            if np.isfinite(z) and np.isfinite(e0) and np.isfinite(e1):
                vals.append(z * (e1 - e0))
    if len(vals) < 20:
        return 1.0
    # mean_revert works when z * d_resid > 0 on average
    return 1.0 if float(np.mean(vals)) >= 0.0 else -1.0


def _apply_risk_scaling(
    target_weights: Dict[str, float],
    date_rows: Dict[str, pd.Series],
    cfg: SignalConfig,
) -> Dict[str, float]:
    if not target_weights:
        return {}

    # Vol-aware scaling: lower weight for high residual-scale names.
    raw = {}
    for t, w in target_weights.items():
        rs = float(date_rows[t].get("residual_scale", np.nan))
        inv_vol = 1.0 / max(rs, 1e-3) if np.isfinite(rs) and rs > 0 else 1.0
        raw[t] = w * inv_vol

    gross = sum(abs(v) for v in raw.values())
    if gross <= 1e-12:
        return {}
    scaled = {t: v / gross for t, v in raw.items()}

    # Name cap and gross cap.
    clipped = {t: float(np.clip(v, -cfg.max_name_weight, cfg.max_name_weight)) for t, v in scaled.items()}
    gross2 = sum(abs(v) for v in clipped.values())
    if gross2 > cfg.max_gross_leverage:
        f = cfg.max_gross_leverage / gross2
        clipped = {t: v * f for t, v in clipped.items()}

    # Vol target from expected basket vol proxy.
    vol_proxy = 0.0
    for t, w in clipped.items():
        rs = float(date_rows[t].get("residual_scale", np.nan))
        rs = rs if np.isfinite(rs) and rs > 1e-6 else 0.02
        vol_proxy += (abs(w) * rs) ** 2
    vol_proxy = np.sqrt(vol_proxy)
    if vol_proxy > 1e-6:
        f = min(cfg.max_gross_leverage / max(sum(abs(v) for v in clipped.values()), 1e-12), cfg.vol_target_daily / vol_proxy)
        clipped = {t: v * f for t, v in clipped.items()}
    return clipped


def backtest_cross_sectional_residual_mean_reversion(
    signal_map: Dict[str, pd.DataFrame],
    signal_cfg: SignalConfig | None = None,
    cost_cfg: TransactionCostConfig | None = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    cfg = signal_cfg or SignalConfig()
    cost = cost_cfg or TransactionCostConfig()
    if not signal_map:
        return pd.DataFrame(), {}

    clean_map = {k: v.sort_values("date").reset_index(drop=True) for k, v in signal_map.items() if not v.empty}
    if not clean_map:
        return pd.DataFrame(), {}

    date_rows = _build_date_row_maps(clean_map)
    all_dates = sorted({d for rows in date_rows.values() for d in rows.keys()})

    prev_w: Dict[str, float] = {k: 0.0 for k in clean_map}
    hold_days: Dict[str, int] = {k: 0 for k in clean_map}

    portfolio_rows: List[Dict[str, float | pd.Timestamp]] = []
    per_name_rows: Dict[str, List[Dict[str, float | pd.Timestamp]]] = {k: [] for k in clean_map}

    for i, d in enumerate(all_dates):
        today_rows = []
        today_by_ticker: Dict[str, pd.Series] = {}
        for t, rows in date_rows.items():
            r = rows.get(d)
            if r is not None:
                today_rows.append(r)
                today_by_ticker[t] = r

        if cfg.signal_direction == "mean_revert":
            direction_multiplier = 1.0
        elif cfg.signal_direction == "momentum":
            direction_multiplier = -1.0
        else:
            direction_multiplier = _infer_direction_multiplier_auto(
                current_date_idx=i,
                all_dates=all_dates,
                date_rows=date_rows,
                window=max(20, int(cfg.auto_sign_window)),
            )

        target = _candidate_weights(today_rows, cfg, direction_multiplier=direction_multiplier)
        target = _apply_risk_scaling(target, today_by_ticker, cfg)

        # Hysteresis + minimum hold logic
        final_w = prev_w.copy()
        for t in clean_map.keys():
            w_prev = prev_w.get(t, 0.0)
            w_tar = target.get(t, 0.0)
            if abs(w_prev) > 1e-12 and hold_days[t] < cfg.min_hold_days and abs(w_tar) < 1e-12:
                final_w[t] = w_prev
                continue
            if abs(w_prev) > 1e-12 and hold_days[t] < cfg.min_hold_days and np.sign(w_prev) != np.sign(w_tar) and abs(w_tar) > 1e-12:
                final_w[t] = w_prev
                continue
            final_w[t] = w_tar

        turnover = {t: abs(final_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in clean_map.keys()}

        gross_pnl = 0.0
        cost_pnl = 0.0
        n_active = 0
        for t in clean_map.keys():
            w = float(final_w.get(t, 0.0))
            if abs(w) > 1e-12:
                n_active += 1
            row_t = today_by_ticker.get(t)
            est_cost = cost.round_trip_unit_cost / 2.0
            if row_t is not None and np.isfinite(row_t.get("estimated_unit_cost", np.nan)):
                est_cost = float(row_t["estimated_unit_cost"])
            tc = float(turnover[t] * (est_cost + cost.hedge_drag_gamma * abs(w)))
            cost_pnl += tc

            # realized next-day residual mean reversion pnl
            if i + 1 < len(all_dates):
                d_next = all_dates[i + 1]
                row_next = date_rows[t].get(d_next)
                if row_t is not None and row_next is not None:
                    r0 = float(row_t.get("residual", np.nan))
                    r1 = float(row_next.get("residual", np.nan))
                    if np.isfinite(r0) and np.isfinite(r1):
                        gross_t = w * (-(r1 - r0))
                    else:
                        gross_t = 0.0
                else:
                    gross_t = 0.0
            else:
                gross_t = 0.0

            per_name_rows[t].append(
                {
                    "date": d,
                    "weight": w,
                    "turnover": float(turnover[t]),
                    "gross_pnl": float(gross_t),
                    "cost": float(tc),
                    "net_pnl": float(gross_t - tc),
                    "zscore": float(row_t["zscore"]) if row_t is not None and np.isfinite(row_t.get("zscore", np.nan)) else np.nan,
                    "trade_allowed": bool(row_t["trade_allowed"]) if row_t is not None and "trade_allowed" in row_t else False,
                }
            )
            gross_pnl += gross_t

        net_pnl = gross_pnl - cost_pnl
        portfolio_rows.append(
            {
                "date": d,
                "portfolio_gross_pnl": float(gross_pnl),
                "portfolio_cost": float(cost_pnl),
                "portfolio_net_pnl": float(net_pnl),
                "n_names": float(n_active),
                "gross_leverage": float(sum(abs(v) for v in final_w.values())),
                "direction_multiplier": float(direction_multiplier),
                "direction_label": "mean_revert" if direction_multiplier > 0 else "momentum",
            }
        )

        for t in clean_map.keys():
            if abs(final_w[t]) > 1e-12:
                hold_days[t] = hold_days[t] + 1 if abs(prev_w[t]) > 1e-12 else 1
            else:
                hold_days[t] = 0
        prev_w = final_w

    portfolio = pd.DataFrame(portfolio_rows).sort_values("date").reset_index(drop=True)
    if not portfolio.empty:
        portfolio["portfolio_cum_net_pnl"] = portfolio["portfolio_net_pnl"].cumsum()

    per_name_df: Dict[str, pd.DataFrame] = {}
    for t, rows in per_name_rows.items():
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        if not df.empty:
            df["cum_net_pnl"] = df["net_pnl"].cumsum()
        per_name_df[t] = df
    return portfolio, per_name_df


def aggregate_portfolio(backtests: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for ticker, df in backtests.items():
        if df.empty:
            continue
        sub = df[["date", "net_pnl", "gross_pnl", "cost"]].copy()
        sub["ticker"] = ticker
        parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=["date", "portfolio_net_pnl", "portfolio_gross_pnl", "portfolio_cost"])

    all_rows = pd.concat(parts, ignore_index=True)
    g = all_rows.groupby("date", as_index=False).agg(
        portfolio_net_pnl=("net_pnl", "mean"),
        portfolio_gross_pnl=("gross_pnl", "mean"),
        portfolio_cost=("cost", "mean"),
        n_names=("ticker", "nunique"),
    )
    g = g.sort_values("date").reset_index(drop=True)
    g["portfolio_cum_net_pnl"] = g["portfolio_net_pnl"].cumsum()
    return g


def summarize_performance(portfolio_df: pd.DataFrame) -> Dict[str, float]:
    if portfolio_df.empty:
        return {
            "n_days": 0.0,
            "avg_daily_net_pnl": 0.0,
            "std_daily_net_pnl": 0.0,
            "daily_sharpe_like": 0.0,
            "final_cum_net_pnl": 0.0,
        }
    r = portfolio_df["portfolio_net_pnl"].to_numpy(dtype=np.float64)
    mu = float(np.mean(r))
    sd = float(np.std(r))
    sharpe = mu / sd * np.sqrt(252.0) if sd > 1e-12 else 0.0
    return {
        "n_days": float(len(portfolio_df)),
        "avg_daily_net_pnl": mu,
        "std_daily_net_pnl": sd,
        "daily_sharpe_like": sharpe,
        "final_cum_net_pnl": float(portfolio_df["portfolio_cum_net_pnl"].iloc[-1]),
    }


def config_to_dict(signal_cfg: SignalConfig, cost_cfg: TransactionCostConfig) -> Dict[str, Dict[str, float]]:
    return {
        "signal": asdict(signal_cfg),
        "transaction_costs": asdict(cost_cfg),
    }
