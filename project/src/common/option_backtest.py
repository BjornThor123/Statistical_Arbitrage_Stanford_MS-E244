from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.skew_signal_backtest import (
    SignalConfig,
    _apply_risk_scaling,
    _candidate_weights,
    _infer_direction_multiplier_auto,
)
from src.strategies.ssvi.model import ssvi_total_variance


@dataclass
class OptionBacktestConfig:
    k_put: float = -0.10
    k_call: float = 0.10
    target_t: float = 30.0 / 365.0
    roll_threshold_t: float = 5.0 / 365.0
    option_half_spread_iv: float = 0.005
    hedge_half_spread_pct: float = 0.0005


@dataclass
class RiskReversalPosition:
    direction: float
    notional: float
    k_put: float
    k_call: float
    iv_put: float
    iv_call: float
    price_put: float
    price_call: float
    remaining_t: float
    hedge_delta: float
    delta_rr: float
    atm_iv: float
    forward_price: float
    entry_date: pd.Timestamp


def _norm_cdf(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    z = x_arr / np.sqrt(2.0)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z))


def _norm_pdf(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x_arr * x_arr) / np.sqrt(2.0 * np.pi)


def _bs_d1_d2(k: float, t: float, sigma: float) -> Tuple[float, float]:
    t_eff = max(float(t), 1e-12)
    vol = max(float(sigma), 1e-8)
    vt = vol * np.sqrt(t_eff)
    d1 = (-float(k) + 0.5 * vol * vol * t_eff) / max(vt, 1e-12)
    d2 = d1 - vt
    return float(d1), float(d2)


def _bs_price_fwd(k: float, t: float, sigma: float, cp: str) -> float:
    d1, d2 = _bs_d1_d2(k=k, t=t, sigma=sigma)
    k_exp = np.exp(float(k))
    if cp.upper() == "C":
        return float(_norm_cdf(d1) - k_exp * _norm_cdf(d2))
    if cp.upper() == "P":
        return float(k_exp * _norm_cdf(-d2) - _norm_cdf(-d1))
    raise ValueError("cp must be 'C' or 'P'")


def _bs_delta_fwd(k: float, t: float, sigma: float, cp: str) -> float:
    d1, _ = _bs_d1_d2(k=k, t=t, sigma=sigma)
    if cp.upper() == "C":
        return float(_norm_cdf(d1))
    if cp.upper() == "P":
        return float(_norm_cdf(d1) - 1.0)
    raise ValueError("cp must be 'C' or 'P'")


def _bs_gamma_fwd(k: float, t: float, sigma: float) -> float:
    d1, _ = _bs_d1_d2(k=k, t=t, sigma=sigma)
    t_eff = max(float(t), 1e-12)
    vol = max(float(sigma), 1e-8)
    return float(_norm_pdf(d1) / max(vol * np.sqrt(t_eff), 1e-12))


def _bs_vega_fwd(k: float, t: float, sigma: float) -> float:
    d1, _ = _bs_d1_d2(k=k, t=t, sigma=sigma)
    return float(_norm_pdf(d1) * np.sqrt(max(float(t), 1e-12)))


def _bs_theta_fwd(k: float, t: float, sigma: float, cp: str) -> float:
    _ = cp
    d1, _ = _bs_d1_d2(k=k, t=t, sigma=sigma)
    theta_per_year = -0.5 * _norm_pdf(d1) * max(float(sigma), 1e-8) / np.sqrt(max(float(t), 1e-12))
    return float(theta_per_year / 365.0)


def _iv_from_ssvi(k: float, theta_target: float, rho: float, eta: float, gamma: float, target_t: float) -> float:
    w = float(ssvi_total_variance(np.asarray([k]), np.asarray([theta_target]), rho=rho, eta=eta, gamma=gamma)[0])
    return float(np.sqrt(max(w, 1e-12) / max(target_t, 1e-12)))


def _option_trade_cost(notional_traded: float, k_put: float, k_call: float, t: float, iv_put: float, iv_call: float, cfg: OptionBacktestConfig) -> float:
    qty = abs(float(notional_traded))
    if qty <= 1e-12:
        return 0.0
    vega_put = _bs_vega_fwd(k_put, t, iv_put)
    vega_call = _bs_vega_fwd(k_call, t, iv_call)
    return float(qty * (vega_put + vega_call) * cfg.option_half_spread_iv)


def _rr_state_from_row(row: pd.Series, direction: float, notional: float, remaining_t: float, cfg: OptionBacktestConfig) -> RiskReversalPosition:
    theta_target = float(row["theta_target"])
    rho = float(row["rho"])
    eta = float(row["eta"])
    gamma = float(row["gamma"])
    pricing_t = cfg.target_t

    iv_put = _iv_from_ssvi(cfg.k_put, theta_target, rho, eta, gamma, pricing_t)
    iv_call = _iv_from_ssvi(cfg.k_call, theta_target, rho, eta, gamma, pricing_t)
    price_put = _bs_price_fwd(cfg.k_put, pricing_t, iv_put, "P")
    price_call = _bs_price_fwd(cfg.k_call, pricing_t, iv_call, "C")
    delta_put = _bs_delta_fwd(cfg.k_put, pricing_t, iv_put, "P")
    delta_call = _bs_delta_fwd(cfg.k_call, pricing_t, iv_call, "C")
    delta_rr = float(direction * (delta_call - delta_put))
    hedge_delta = float(-notional * delta_rr)

    return RiskReversalPosition(
        direction=float(np.sign(direction)) if abs(direction) > 1e-12 else 0.0,
        notional=float(abs(notional)),
        k_put=float(cfg.k_put),
        k_call=float(cfg.k_call),
        iv_put=float(iv_put),
        iv_call=float(iv_call),
        price_put=float(price_put),
        price_call=float(price_call),
        remaining_t=float(max(remaining_t, 0.0)),
        hedge_delta=float(hedge_delta),
        delta_rr=float(delta_rr),
        atm_iv=float(row["atm_iv"]),
        forward_price=float(row["forward_price"]),
        entry_date=pd.Timestamp(row["date"]),
    )


def _step_risk_reversal(
    position: RiskReversalPosition | None,
    weight: float,
    row: pd.Series | None,
    cfg: OptionBacktestConfig,
) -> Tuple[RiskReversalPosition | None, Dict[str, float | bool]]:
    out = {
        "total_pnl": 0.0,
        "skew_vega_pnl": 0.0,
        "level_vega_pnl": 0.0,
        "gamma_pnl": 0.0,
        "theta_pnl": 0.0,
        "hedge_cost": 0.0,
        "option_trade_cost": 0.0,
        "net_pnl": 0.0,
        "unexplained_pnl": 0.0,
        "is_roll": False,
        "rr_price": np.nan,
        "delta_rr": np.nan,
        "hedge_delta": np.nan,
        "remaining_t": np.nan,
        "iv_put": np.nan,
        "iv_call": np.nan,
        "direction": 0.0,
    }

    target_notional = abs(float(weight))
    target_direction = 0.0 if target_notional <= 1e-12 else float(np.sign(weight))
    day_step = 1.0 / 365.0

    if row is None:
        if position is not None:
            out["rr_price"] = position.direction * (position.price_call - position.price_put)
            out["delta_rr"] = position.delta_rr
            out["hedge_delta"] = position.hedge_delta
            out["remaining_t"] = position.remaining_t
            out["iv_put"] = position.iv_put
            out["iv_call"] = position.iv_call
            out["direction"] = position.direction
        return position, out

    today = pd.Timestamp(row["date"])

    # Realize MTM and hedge PnL for existing inventory before trading decisions.
    if position is not None:
        theta_target = float(row["theta_target"])
        rho = float(row["rho"])
        eta = float(row["eta"])
        gamma = float(row["gamma"])
        atm_iv_now = float(row["atm_iv"])
        fwd_now = float(row["forward_price"])

        iv_put_now = _iv_from_ssvi(position.k_put, theta_target, rho, eta, gamma, cfg.target_t)
        iv_call_now = _iv_from_ssvi(position.k_call, theta_target, rho, eta, gamma, cfg.target_t)
        price_put_now = _bs_price_fwd(position.k_put, cfg.target_t, iv_put_now, "P")
        price_call_now = _bs_price_fwd(position.k_call, cfg.target_t, iv_call_now, "C")

        delta_put_old = _bs_delta_fwd(position.k_put, cfg.target_t, position.iv_put, "P")
        delta_call_old = _bs_delta_fwd(position.k_call, cfg.target_t, position.iv_call, "C")
        gamma_put_old = _bs_gamma_fwd(position.k_put, cfg.target_t, position.iv_put)
        gamma_call_old = _bs_gamma_fwd(position.k_call, cfg.target_t, position.iv_call)
        vega_put_old = _bs_vega_fwd(position.k_put, cfg.target_t, position.iv_put)
        vega_call_old = _bs_vega_fwd(position.k_call, cfg.target_t, position.iv_call)
        theta_put_old = _bs_theta_fwd(position.k_put, cfg.target_t, position.iv_put, "P")
        theta_call_old = _bs_theta_fwd(position.k_call, cfg.target_t, position.iv_call, "C")

        fwd_ret = 0.0
        if position.forward_price > 1e-12 and np.isfinite(position.forward_price) and np.isfinite(fwd_now):
            fwd_ret = float(fwd_now / position.forward_price - 1.0)

        option_mark = position.notional * (
            position.direction * ((price_call_now - position.price_call) - (price_put_now - position.price_put))
        )
        hedge_pnl = float(position.hedge_delta * fwd_ret)
        total_mark_change = float(option_mark + hedge_pnl)

        level_dsigma = atm_iv_now - position.atm_iv
        dvol_put = iv_put_now - position.iv_put
        dvol_call = iv_call_now - position.iv_call

        skew_vega = position.notional * position.direction * (
            vega_call_old * (dvol_call - level_dsigma) - vega_put_old * (dvol_put - level_dsigma)
        )
        level_vega = position.notional * position.direction * (
            vega_call_old * level_dsigma - vega_put_old * level_dsigma
        )
        gamma_attr = position.notional * position.direction * (
            0.5 * gamma_call_old * (fwd_ret ** 2) - 0.5 * gamma_put_old * (fwd_ret ** 2)
        )
        theta_attr = position.notional * position.direction * (theta_call_old - theta_put_old)
        unexplained = total_mark_change - (skew_vega + level_vega + gamma_attr + theta_attr)

        out["total_pnl"] = float(total_mark_change)
        out["skew_vega_pnl"] = float(skew_vega)
        out["level_vega_pnl"] = float(level_vega)
        out["gamma_pnl"] = float(gamma_attr)
        out["theta_pnl"] = float(theta_attr)
        out["unexplained_pnl"] = float(unexplained)

        position.iv_put = float(iv_put_now)
        position.iv_call = float(iv_call_now)
        position.price_put = float(price_put_now)
        position.price_call = float(price_call_now)
        position.atm_iv = float(atm_iv_now)
        position.forward_price = float(fwd_now)
        position.remaining_t = float(max(position.remaining_t - day_step, 0.0))

        delta_put_now = _bs_delta_fwd(position.k_put, cfg.target_t, position.iv_put, "P")
        delta_call_now = _bs_delta_fwd(position.k_call, cfg.target_t, position.iv_call, "C")
        delta_rr_now = float(position.direction * (delta_call_now - delta_put_now))
        target_hedge = float(-position.notional * delta_rr_now)
        hedge_trade = float(target_hedge - position.hedge_delta)
        out["hedge_cost"] = float(out["hedge_cost"] + abs(hedge_trade) * cfg.hedge_half_spread_pct)
        position.hedge_delta = target_hedge
        position.delta_rr = delta_rr_now

    # Roll inventory when remaining maturity bucket is below threshold.
    if position is not None and position.remaining_t < cfg.roll_threshold_t:
        out["is_roll"] = True
        out["option_trade_cost"] = float(
            out["option_trade_cost"]
            + _option_trade_cost(
                notional_traded=position.notional,
                k_put=position.k_put,
                k_call=position.k_call,
                t=cfg.target_t,
                iv_put=position.iv_put,
                iv_call=position.iv_call,
                cfg=cfg,
            )
            + _option_trade_cost(
                notional_traded=position.notional,
                k_put=position.k_put,
                k_call=position.k_call,
                t=cfg.target_t,
                iv_put=position.iv_put,
                iv_call=position.iv_call,
                cfg=cfg,
            )
        )
        position.remaining_t = float(cfg.target_t)

    # Target inventory transition.
    if position is None and target_notional > 1e-12:
        position = _rr_state_from_row(
            row=row,
            direction=target_direction,
            notional=target_notional,
            remaining_t=cfg.target_t,
            cfg=cfg,
        )
        out["option_trade_cost"] = float(
            out["option_trade_cost"]
            + _option_trade_cost(
                notional_traded=target_notional,
                k_put=position.k_put,
                k_call=position.k_call,
                t=cfg.target_t,
                iv_put=position.iv_put,
                iv_call=position.iv_call,
                cfg=cfg,
            )
        )
        out["hedge_cost"] = float(out["hedge_cost"] + abs(position.hedge_delta) * cfg.hedge_half_spread_pct)
    elif position is not None and target_notional <= 1e-12:
        out["option_trade_cost"] = float(
            out["option_trade_cost"]
            + _option_trade_cost(
                notional_traded=position.notional,
                k_put=position.k_put,
                k_call=position.k_call,
                t=cfg.target_t,
                iv_put=position.iv_put,
                iv_call=position.iv_call,
                cfg=cfg,
            )
        )
        out["hedge_cost"] = float(out["hedge_cost"] + abs(position.hedge_delta) * cfg.hedge_half_spread_pct)
        position = None
    elif position is not None and target_notional > 1e-12:
        if target_direction != position.direction:
            out["option_trade_cost"] = float(
                out["option_trade_cost"]
                + _option_trade_cost(
                    notional_traded=position.notional,
                    k_put=position.k_put,
                    k_call=position.k_call,
                    t=cfg.target_t,
                    iv_put=position.iv_put,
                    iv_call=position.iv_call,
                    cfg=cfg,
                )
            )
            out["hedge_cost"] = float(out["hedge_cost"] + abs(position.hedge_delta) * cfg.hedge_half_spread_pct)
            position = _rr_state_from_row(
                row=row,
                direction=target_direction,
                notional=target_notional,
                remaining_t=cfg.target_t,
                cfg=cfg,
            )
            out["option_trade_cost"] = float(
                out["option_trade_cost"]
                + _option_trade_cost(
                    notional_traded=target_notional,
                    k_put=position.k_put,
                    k_call=position.k_call,
                    t=cfg.target_t,
                    iv_put=position.iv_put,
                    iv_call=position.iv_call,
                    cfg=cfg,
                )
            )
            out["hedge_cost"] = float(out["hedge_cost"] + abs(position.hedge_delta) * cfg.hedge_half_spread_pct)
        else:
            d_notional = float(target_notional - position.notional)
            if abs(d_notional) > 1e-12:
                out["option_trade_cost"] = float(
                    out["option_trade_cost"]
                    + _option_trade_cost(
                        notional_traded=abs(d_notional),
                        k_put=position.k_put,
                        k_call=position.k_call,
                        t=cfg.target_t,
                        iv_put=position.iv_put,
                        iv_call=position.iv_call,
                        cfg=cfg,
                    )
                )
                new_hedge = float(-target_notional * position.delta_rr)
                out["hedge_cost"] = float(out["hedge_cost"] + abs(new_hedge - position.hedge_delta) * cfg.hedge_half_spread_pct)
                position.notional = float(target_notional)
                position.hedge_delta = new_hedge

    out["net_pnl"] = float(out["total_pnl"] - out["hedge_cost"] - out["option_trade_cost"])

    if position is not None:
        out["rr_price"] = float(position.direction * (position.price_call - position.price_put))
        out["delta_rr"] = float(position.delta_rr)
        out["hedge_delta"] = float(position.hedge_delta)
        out["remaining_t"] = float(position.remaining_t)
        out["iv_put"] = float(position.iv_put)
        out["iv_call"] = float(position.iv_call)
        out["direction"] = float(position.direction)

    _ = today
    return position, out


def _build_date_row_maps(signal_map: Dict[str, pd.DataFrame]) -> Dict[str, Dict[pd.Timestamp, pd.Series]]:
    out: Dict[str, Dict[pd.Timestamp, pd.Series]] = {}
    for ticker, df in signal_map.items():
        rows = {pd.Timestamp(r["date"]): r for _, r in df.iterrows()}
        out[ticker] = rows
    return out


def backtest_option_cross_sectional(
    signal_map: Dict[str, pd.DataFrame],
    skew_map: Dict[str, pd.DataFrame],
    signal_cfg: SignalConfig | None = None,
    option_cfg: OptionBacktestConfig | None = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    cfg = signal_cfg or SignalConfig()
    ocfg = option_cfg or OptionBacktestConfig()
    if not signal_map or not skew_map:
        return pd.DataFrame(), {}

    clean_map: Dict[str, pd.DataFrame] = {}
    for ticker, sig in signal_map.items():
        sk = skew_map.get(ticker)
        if sk is None or sig.empty or sk.empty:
            continue
        a = sig.copy()
        b = sk.copy()
        a["date"] = pd.to_datetime(a["date"])
        b["date"] = pd.to_datetime(b["date"])
        merged = a.merge(
            b[
                [
                    "date",
                    "ticker",
                    "rho",
                    "eta",
                    "gamma",
                    "theta_target",
                    "forward_price",
                    "atm_iv",
                ]
            ],
            on=["date", "ticker"],
            how="inner",
        )
        if merged.empty:
            continue
        clean_map[ticker] = merged.sort_values("date").reset_index(drop=True)

    if not clean_map:
        return pd.DataFrame(), {}

    date_rows = _build_date_row_maps(clean_map)
    all_dates = sorted({d for rows in date_rows.values() for d in rows.keys()})

    prev_w: Dict[str, float] = {k: 0.0 for k in clean_map}
    hold_days: Dict[str, int] = {k: 0 for k in clean_map}
    positions: Dict[str, RiskReversalPosition | None] = {k: None for k in clean_map}

    portfolio_rows: List[Dict[str, float | pd.Timestamp]] = []
    per_name_rows: Dict[str, List[Dict[str, float | pd.Timestamp | bool]]] = {k: [] for k in clean_map}

    for i, d in enumerate(all_dates):
        today_rows: List[pd.Series] = []
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

        p_total = 0.0
        p_skew = 0.0
        p_level = 0.0
        p_gamma = 0.0
        p_theta = 0.0
        p_hedge_cost = 0.0
        p_opt_cost = 0.0
        p_net = 0.0
        n_active = 0

        for t in clean_map.keys():
            w = float(final_w.get(t, 0.0))
            if abs(w) > 1e-12:
                n_active += 1
            row_t = today_by_ticker.get(t)
            pos_next, pnl = _step_risk_reversal(
                position=positions.get(t),
                weight=w,
                row=row_t,
                cfg=ocfg,
            )
            positions[t] = pos_next

            p_total += float(pnl["total_pnl"])
            p_skew += float(pnl["skew_vega_pnl"])
            p_level += float(pnl["level_vega_pnl"])
            p_gamma += float(pnl["gamma_pnl"])
            p_theta += float(pnl["theta_pnl"])
            p_hedge_cost += float(pnl["hedge_cost"])
            p_opt_cost += float(pnl["option_trade_cost"])
            p_net += float(pnl["net_pnl"])

            per_name_rows[t].append(
                {
                    "date": d,
                    "weight": w,
                    "direction": float(pnl["direction"]),
                    "remaining_t": float(pnl["remaining_t"]) if np.isfinite(pnl["remaining_t"]) else np.nan,
                    "iv_put": float(pnl["iv_put"]) if np.isfinite(pnl["iv_put"]) else np.nan,
                    "iv_call": float(pnl["iv_call"]) if np.isfinite(pnl["iv_call"]) else np.nan,
                    "rr_price": float(pnl["rr_price"]) if np.isfinite(pnl["rr_price"]) else np.nan,
                    "delta_rr": float(pnl["delta_rr"]) if np.isfinite(pnl["delta_rr"]) else np.nan,
                    "hedge_delta": float(pnl["hedge_delta"]) if np.isfinite(pnl["hedge_delta"]) else np.nan,
                    "total_pnl": float(pnl["total_pnl"]),
                    "skew_vega_pnl": float(pnl["skew_vega_pnl"]),
                    "level_vega_pnl": float(pnl["level_vega_pnl"]),
                    "gamma_pnl": float(pnl["gamma_pnl"]),
                    "theta_pnl": float(pnl["theta_pnl"]),
                    "hedge_cost": float(pnl["hedge_cost"]),
                    "option_trade_cost": float(pnl["option_trade_cost"]),
                    "net_pnl": float(pnl["net_pnl"]),
                    "unexplained_pnl": float(pnl["unexplained_pnl"]),
                    "is_roll": bool(pnl["is_roll"]),
                    "zscore": float(row_t["zscore"]) if row_t is not None and np.isfinite(row_t.get("zscore", np.nan)) else np.nan,
                    "trade_allowed": bool(row_t["trade_allowed"]) if row_t is not None and "trade_allowed" in row_t else False,
                }
            )

        portfolio_rows.append(
            {
                "date": d,
                "portfolio_total_pnl": float(p_total),
                "portfolio_skew_vega_pnl": float(p_skew),
                "portfolio_level_vega_pnl": float(p_level),
                "portfolio_gamma_pnl": float(p_gamma),
                "portfolio_theta_pnl": float(p_theta),
                "portfolio_hedge_cost": float(p_hedge_cost),
                "portfolio_option_trade_cost": float(p_opt_cost),
                "portfolio_net_pnl": float(p_net),
                "portfolio_gross_pnl": float(p_total),
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
