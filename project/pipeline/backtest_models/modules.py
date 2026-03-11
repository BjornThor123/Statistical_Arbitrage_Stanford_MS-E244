from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math

import duckdb
import numpy as np
import pandas as pd

from interfaces import BacktestEngineModule, BacktestOutput, RunSpec, SignalOutput, SkewOutput

"""
Backtest engine.

DeltaHedgedOptionBacktestEngine simulates trading risk reversals with:
  - Daily mark-to-market on the option position
  - Delta hedging via stock trades
  - Explicit bid/ask spread costs + per-contract fees
  - Contract rolling when TTE falls below roll_threshold_days
  - Greek decomposition (skew vega, level vega, gamma, theta)
"""


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_d1(s: float, k: float, t: float, r: float, sigma: float) -> float:
    s = max(float(s), 1e-12)
    k = max(float(k), 1e-12)
    t = max(float(t), 1e-8)
    sigma = max(float(sigma), 1e-6)
    return (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))


def _bs_delta(cp_flag: str, s: float, k: float, t: float, r: float, sigma: float) -> float:
    d1 = _bs_d1(s=s, k=k, t=t, r=r, sigma=sigma)
    if cp_flag.upper() == "C":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def _bs_gamma(s: float, k: float, t: float, r: float, sigma: float) -> float:
    d1 = _bs_d1(s=s, k=k, t=t, r=r, sigma=sigma)
    s = max(float(s), 1e-12)
    t = max(float(t), 1e-8)
    sigma = max(float(sigma), 1e-6)
    return _norm_pdf(d1) / (s * sigma * math.sqrt(t))


def _bs_vega(s: float, k: float, t: float, r: float, sigma: float) -> float:
    d1 = _bs_d1(s=s, k=k, t=t, r=r, sigma=sigma)
    s = max(float(s), 1e-12)
    t = max(float(t), 1e-8)
    return s * _norm_pdf(d1) * math.sqrt(t)


def _bs_theta_daily(s: float, k: float, t: float, r: float, sigma: float) -> float:
    # Simple daily carry approximation; sign convention: long option usually negative theta.
    d1 = _bs_d1(s=s, k=k, t=t, r=r, sigma=sigma)
    s = max(float(s), 1e-12)
    t = max(float(t), 1e-8)
    sigma = max(float(sigma), 1e-6)
    theta_per_year = -0.5 * s * _norm_pdf(d1) * sigma / math.sqrt(t)
    return theta_per_year / 365.0


@dataclass
class DeltaHedgedOptionBacktestEngine(BacktestEngineModule):
    db_path: str
    table: str = "options_enriched"
    contract_multiplier: float = 100.0
    target_days: int = 30
    moneyness_weight_days: float = 20.0
    stock_fee_bps: float = 1.0
    option_fee_per_contract: float = 0.65
    sizing_mode: str = "contracts"  # contracts | dollar_vega
    max_contracts_per_signal: float = 10.0
    roll_threshold_days: int = 5
    rr_put_log_moneyness: float = -0.15
    rr_call_log_moneyness: float = 0.15
    # When rr_put_delta > 0, contract selection uses BS delta instead of log-moneyness.
    # rr_put_delta=0.25 selects the 25-delta put and call (standard market convention).
    # Set to 0.0 to fall back to fixed log-moneyness.
    rr_put_delta: float = 0.25
    rr_call_delta: float = 0.25
    target_dollar_vega_per_signal: float = 20000.0

    def _load_pairs_for_ticker(self, ticker: str, spec: RunSpec) -> pd.DataFrame:
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            q = f"""
                WITH base AS (
                    SELECT
                        CAST(date AS DATE) AS date,
                        CAST(exdate AS DATE) AS exdate,
                        UPPER(cp_flag) AS cp_flag,
                        COALESCE(CAST(strike AS DOUBLE), CAST(strike_price AS DOUBLE) / 1000.0) AS strike,
                        CAST(best_bid AS DOUBLE) AS bid,
                        CAST(best_offer AS DOUBLE) AS ask,
                        CAST(mid_price AS DOUBLE) AS mid,
                        CAST(impl_volatility AS DOUBLE) AS iv,
                        CAST(spot_price AS DOUBLE) AS spot,
                        CAST(risk_free_rate AS DOUBLE) AS r
                    FROM {self.table}
                    WHERE UPPER(ticker)=UPPER(?)
                      AND CAST(date AS DATE) >= CAST(? AS DATE)
                      AND CAST(date AS DATE) <= CAST(? AS DATE)
                )
                SELECT
                    c.date,
                    c.exdate,
                    c.strike,
                    c.spot,
                    COALESCE(c.r, 0.0) AS r,
                    GREATEST(DATEDIFF('day', c.date, c.exdate), 1) / 365.0 AS t,
                    GREATEST(DATEDIFF('day', c.date, c.exdate), 1) AS t_days,
                    c.bid AS c_bid,
                    c.ask AS c_ask,
                    c.mid AS c_mid,
                    c.iv  AS c_iv,
                    p.bid AS p_bid,
                    p.ask AS p_ask,
                    p.mid AS p_mid,
                    p.iv  AS p_iv
                FROM base c
                JOIN base p
                  ON c.date = p.date
                 AND c.exdate = p.exdate
                 AND ABS(c.strike - p.strike) < 1e-10
                WHERE c.cp_flag='C' AND p.cp_flag='P'
            """
            df = con.execute(q, [ticker, spec.start_date, spec.end_date]).fetchdf()
        finally:
            con.close()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df["exdate"] = pd.to_datetime(df["exdate"])
        for c in ["spot", "strike", "t", "t_days", "c_bid", "c_ask", "c_mid", "p_bid", "p_ask", "p_mid", "c_iv", "p_iv", "r"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["spot", "strike", "t", "c_bid", "c_ask", "p_bid", "p_ask"])
        df = df[(df["spot"] > 0.0) & (df["strike"] > 0.0) & (df["t"] > 0.0)]
        return df

    def _pick_contract(self, daily_pairs: pd.DataFrame) -> pd.Series | None:
        if daily_pairs.empty:
            return None
        x = daily_pairs.copy()
        x = x[x["t_days"] > self.roll_threshold_days]
        if x.empty:
            return None
        x["lm"] = np.log(x["strike"] / x["spot"])
        x["score"] = np.abs(x["t_days"] - self.target_days) + self.moneyness_weight_days * np.abs(x["lm"])
        x = x.sort_values(["score", "t_days", "strike"]).reset_index(drop=True)
        return x.iloc[0]

    @staticmethod
    def _compose_rr_quote(call_row: pd.Series, put_row: pd.Series) -> pd.Series:
        return pd.Series(
            {
                "date": pd.Timestamp(call_row["date"]),
                "exdate": pd.Timestamp(call_row["exdate"]),
                "spot": float(call_row["spot"]),
                "r": float(call_row["r"]),
                "t": float(call_row["t"]),
                "t_days": float(call_row["t_days"]),
                "c_strike": float(call_row["strike"]),
                "p_strike": float(put_row["strike"]),
                "strike": float(0.5 * (float(call_row["strike"]) + float(put_row["strike"]))),
                "c_bid": float(call_row["c_bid"]),
                "c_ask": float(call_row["c_ask"]),
                "c_mid": float(call_row["c_mid"]),
                "c_iv": float(call_row["c_iv"]),
                "p_bid": float(put_row["p_bid"]),
                "p_ask": float(put_row["p_ask"]),
                "p_mid": float(put_row["p_mid"]),
                "p_iv": float(put_row["p_iv"]),
            }
        )

    def _pick_risk_reversal_contract(self, daily_pairs: pd.DataFrame) -> pd.Series | None:
        if daily_pairs.empty:
            return None
        x = daily_pairs.copy()
        x["lm"] = np.log(x["strike"] / x["spot"])
        # Only consider expirations with enough TTE to avoid immediate rolling.
        x = x[x["t_days"] > self.roll_threshold_days]
        if x.empty:
            return None
        expiry = (
            x.groupby("exdate", as_index=False)
            .agg(t_days=("t_days", "first"))
            .assign(score=lambda df: (df["t_days"] - self.target_days).abs())
            .sort_values(["score", "t_days"])
            .reset_index(drop=True)
        )
        if expiry.empty:
            return None
        best_exdate = pd.Timestamp(expiry.iloc[0]["exdate"])
        sub = x[x["exdate"] == best_exdate].copy()
        puts = sub[sub["lm"] <= 0.0].copy()
        calls = sub[sub["lm"] >= 0.0].copy()
        if puts.empty or calls.empty:
            return None

        if self.rr_put_delta > 0.0:
            # Delta-targeting: compute BS delta for each available option and
            # pick the put/call closest to the target delta magnitude.
            calls["delta_c"] = calls.apply(
                lambda r: _bs_delta("C", float(r["spot"]), float(r["strike"]),
                                    float(r["t"]), float(r.get("r", 0.0)), float(r["c_iv"])),
                axis=1,
            )
            puts["delta_p"] = puts.apply(
                lambda r: _bs_delta("P", float(r["spot"]), float(r["strike"]),
                                    float(r["t"]), float(r.get("r", 0.0)), float(r["p_iv"])),
                axis=1,
            )
            call_row = calls.iloc[(calls["delta_c"] - self.rr_call_delta).abs().argmin()]
            put_row  = puts.iloc[(puts["delta_p"] - (-self.rr_put_delta)).abs().argmin()]
        else:
            # Fixed log-moneyness fallback.
            put_row  = puts.iloc[(puts["lm"] - self.rr_put_log_moneyness).abs().argmin()]
            call_row = calls.iloc[(calls["lm"] - self.rr_call_log_moneyness).abs().argmin()]

        return self._compose_rr_quote(call_row=call_row, put_row=put_row)

    def _find_risk_reversal_quote(
        self,
        day_df: pd.DataFrame | None,
        exdate: pd.Timestamp,
        call_strike: float,
        put_strike: float,
    ) -> pd.Series | None:
        if day_df is None or day_df.empty:
            return None
        sub = day_df[day_df["exdate"] == exdate]
        call_match = sub[np.abs(sub["strike"] - call_strike) < 1e-10]
        put_match = sub[np.abs(sub["strike"] - put_strike) < 1e-10]
        if call_match.empty or put_match.empty:
            return None
        return self._compose_rr_quote(call_row=call_match.iloc[0], put_row=put_match.iloc[0])

    def _quote_package_dollar_vega(self, q: pd.Series, instrument: str) -> float:
        spot = float(q["spot"])
        t = float(q["t"])
        r = float(q["r"])
        c_iv = float(q["c_iv"])
        p_iv = float(q["p_iv"])
        c_strike = float(q["c_strike"]) if "c_strike" in q.index else float(q["strike"])
        p_strike = float(q["p_strike"]) if "p_strike" in q.index else float(q["strike"])
        vega_c = _bs_vega(spot, c_strike, t, r, c_iv)
        vega_p = _bs_vega(spot, p_strike, t, r, p_iv)
        if instrument in {"risk_reversal", "straddle"}:
            return float(self.contract_multiplier * (vega_c + vega_p))
        raise ValueError(f"Unsupported signal_instrument '{instrument}'.")

    @staticmethod
    def _find_quote(day_df: pd.DataFrame | None, exdate: pd.Timestamp, strike: float) -> pd.Series | None:
        """Find today's quote for a vanilla contract by expiry and strike."""
        if day_df is None or day_df.empty:
            return None
        m = (day_df["exdate"] == exdate) & (np.abs(day_df["strike"] - strike) < 1e-10)
        sub = day_df[m]
        return None if sub.empty else sub.iloc[0]

    @staticmethod
    def _option_trade_cost_static(dq_call: float, dq_put: float, q: pd.Series, contract_multiplier: float, option_fee_per_contract: float) -> float:
        """Compute total option trade cost (spread + per-contract fee)."""
        spread_c = max(float(q["c_ask"] - q["c_bid"]), 0.0)
        spread_p = max(float(q["p_ask"] - q["p_bid"]), 0.0)
        spread_cost = (abs(dq_call) * spread_c * 0.5 + abs(dq_put) * spread_p * 0.5) * contract_multiplier
        fee_cost = (abs(dq_call) + abs(dq_put)) * option_fee_per_contract
        return float(spread_cost + fee_cost)

    def _get_quote(
        self,
        day_df: pd.DataFrame | None,
        instrument: str,
        exdate: pd.Timestamp | None = None,
        c_strike: float | None = None,
        p_strike: float | None = None,
        strike: float | None = None,
    ) -> pd.Series | None:
        """Find today's quote for the current contract, dispatching on instrument type."""
        if day_df is None or day_df.empty or exdate is None:
            return None
        if instrument == "risk_reversal":
            return self._find_risk_reversal_quote(day_df, exdate=exdate, call_strike=c_strike, put_strike=p_strike)
        sub = day_df[(day_df["exdate"] == exdate) & (np.abs(day_df["strike"] - strike) < 1e-10)]
        return None if sub.empty else sub.iloc[0]

    def _mark_to_market(self, state: Dict[str, object] | None, today: pd.DataFrame | None) -> Dict[str, float] | None:
        """Mark existing position to market. Updates state price fields in place. Returns None if position should be flattened."""
        if state is None:
            return {
                "gross_opt": 0.0,
                "gross_hedge": 0.0,
                "skew_vega_pnl": 0.0,
                "level_vega_pnl": 0.0,
                "gamma_pnl": 0.0,
                "theta_pnl": 0.0,
                "unexplained_pnl": 0.0,
            }
        instrument = str(state["instrument"])
        q = self._get_quote(
            today,
            instrument=instrument,
            exdate=pd.Timestamp(state["exdate"]),
            c_strike=float(state["c_strike"]),
            p_strike=float(state["p_strike"]),
            strike=float(state["strike"]),
        )
        if q is None:
            return None  # Signal caller to flatten position

        q_call_prev = float(state["q_call"])
        q_put_prev = float(state["q_put"])
        c_mid_prev = float(state["c_mid"])
        p_mid_prev = float(state["p_mid"])
        spot_prev = float(state["spot"])
        c_mid_now = float(q["c_mid"])
        p_mid_now = float(q["p_mid"])
        spot_now = float(q["spot"])
        gross_opt = self.contract_multiplier * (
            q_call_prev * (c_mid_now - c_mid_prev) + q_put_prev * (p_mid_now - p_mid_prev)
        )
        gross_hedge = float(state["hedge_shares"]) * (spot_now - spot_prev)

        # Approximate greek decomposition from previous state.
        r_prev = float(state["r"])
        t_prev = max(float(state["t"]), 1e-8)
        c_iv_prev = max(float(state["c_iv"]), 1e-6)
        p_iv_prev = max(float(state["p_iv"]), 1e-6)
        c_strike_prev = float(state["c_strike"])
        p_strike_prev = float(state["p_strike"])
        level_dsigma = 0.5 * ((float(q["c_iv"]) + float(q["p_iv"])) - (c_iv_prev + p_iv_prev))
        dvol_call = float(q["c_iv"]) - c_iv_prev
        dvol_put = float(q["p_iv"]) - p_iv_prev
        vega_c = _bs_vega(spot_prev, c_strike_prev, t_prev, r_prev, c_iv_prev)
        vega_p = _bs_vega(spot_prev, p_strike_prev, t_prev, r_prev, p_iv_prev)
        gamma_c = _bs_gamma(spot_prev, c_strike_prev, t_prev, r_prev, c_iv_prev)
        gamma_p = _bs_gamma(spot_prev, p_strike_prev, t_prev, r_prev, p_iv_prev)
        theta_c = _bs_theta_daily(spot_prev, c_strike_prev, t_prev, r_prev, c_iv_prev)
        theta_p = _bs_theta_daily(spot_prev, p_strike_prev, t_prev, r_prev, p_iv_prev)
        ret = (spot_now - spot_prev) / max(abs(spot_prev), 1e-12)
        skew_vega_pnl = self.contract_multiplier * (
            q_call_prev * vega_c * (dvol_call - level_dsigma)
            + q_put_prev * vega_p * (dvol_put - level_dsigma)
        )
        level_vega_pnl = self.contract_multiplier * (
            q_call_prev * vega_c * level_dsigma + q_put_prev * vega_p * level_dsigma
        )
        gamma_pnl = self.contract_multiplier * 0.5 * (
            q_call_prev * gamma_c + q_put_prev * gamma_p
        ) * (ret**2) * (spot_prev**2)
        theta_pnl = self.contract_multiplier * (q_call_prev * theta_c + q_put_prev * theta_p)
        gross = gross_opt + gross_hedge
        unexplained_pnl = gross - (skew_vega_pnl + level_vega_pnl + gamma_pnl + theta_pnl)

        state["c_mid"] = c_mid_now
        state["p_mid"] = p_mid_now
        state["spot"] = spot_now
        state["c_iv"] = float(q["c_iv"])
        state["p_iv"] = float(q["p_iv"])
        state["r"] = float(q["r"])
        state["t"] = float(q["t"])
        state["t_days"] = float(q["t_days"])

        return {
            "gross_opt": gross_opt,
            "gross_hedge": gross_hedge,
            "skew_vega_pnl": skew_vega_pnl,
            "level_vega_pnl": level_vega_pnl,
            "gamma_pnl": gamma_pnl,
            "theta_pnl": theta_pnl,
            "unexplained_pnl": unexplained_pnl,
        }

    def _select_contract(self, state: Dict[str, object] | None, today: pd.DataFrame | None, instrument: str) -> tuple[pd.Series | None, bool]:
        """Select the target contract for the day. Returns (target_quote, is_roll)."""
        if today is None or today.empty:
            return None, False
        is_roll = False
        if state is not None and str(state["instrument"]) == instrument and float(state["t_days"]) > float(self.roll_threshold_days):
            q_same = self._get_quote(
                today,
                instrument=instrument,
                exdate=pd.Timestamp(state["exdate"]),
                c_strike=float(state["c_strike"]),
                p_strike=float(state["p_strike"]),
                strike=float(state["strike"]),
            )
            if q_same is not None:
                return q_same, False
            else:
                target_quote = self._pick_risk_reversal_contract(today) if instrument == "risk_reversal" else self._pick_contract(today)
                return target_quote, True
        else:
            target_quote = self._pick_risk_reversal_contract(today) if instrument == "risk_reversal" else self._pick_contract(today)
            if state is not None:
                is_roll = True
            return target_quote, is_roll

    def _execute_trade(
        self,
        state: Dict[str, object] | None,
        target_quote: pd.Series | None,
        target_q_call: float,
        target_q_put: float,
        today: pd.DataFrame | None,
        instrument: str,
        same_contract: bool,
    ) -> tuple[float, bool, Dict[str, object] | None]:
        """Execute close/open/resize on today's quote. Returns (option_trade_cost, traded, updated_state)."""
        option_trade_cost = 0.0
        traded = False

        # Close existing position if switching contracts or flattening.
        if state is not None and (target_quote is None or not same_contract or str(state["instrument"]) != instrument):
            q_close = self._get_quote(
                today,
                instrument=str(state["instrument"]),
                exdate=pd.Timestamp(state["exdate"]),
                c_strike=float(state["c_strike"]),
                p_strike=float(state["p_strike"]),
                strike=float(state["strike"]),
            )
            if q_close is not None:
                option_trade_cost += self._option_trade_cost_static(
                    dq_call=abs(float(state["q_call"])),
                    dq_put=abs(float(state["q_put"])),
                    q=q_close,
                    contract_multiplier=self.contract_multiplier,
                    option_fee_per_contract=self.option_fee_per_contract,
                )
            state = None

        if target_quote is not None:
            if state is None:
                state = {
                    "instrument": instrument,
                    "exdate": pd.Timestamp(target_quote["exdate"]),
                    "strike": float(target_quote["strike"]),
                    "c_strike": float(target_quote["c_strike"]) if "c_strike" in target_quote.index else float(target_quote["strike"]),
                    "p_strike": float(target_quote["p_strike"]) if "p_strike" in target_quote.index else float(target_quote["strike"]),
                    "q_call": 0.0,
                    "q_put": 0.0,
                    "hedge_shares": 0.0,
                    "c_mid": float(target_quote["c_mid"]),
                    "p_mid": float(target_quote["p_mid"]),
                    "spot": float(target_quote["spot"]),
                    "c_iv": float(target_quote["c_iv"]),
                    "p_iv": float(target_quote["p_iv"]),
                    "r": float(target_quote["r"]),
                    "t": float(target_quote["t"]),
                    "t_days": float(target_quote["t_days"]),
                }
            dq_call = target_q_call - float(state["q_call"])
            dq_put = target_q_put - float(state["q_put"])
            if abs(dq_call) > 1e-12 or abs(dq_put) > 1e-12:
                traded = True
                option_trade_cost += self._option_trade_cost_static(
                    dq_call=dq_call,
                    dq_put=dq_put,
                    q=target_quote,
                    contract_multiplier=self.contract_multiplier,
                    option_fee_per_contract=self.option_fee_per_contract,
                )
                state["q_call"] = float(target_q_call)
                state["q_put"] = float(target_q_put)
                state["instrument"] = instrument
                state["exdate"] = pd.Timestamp(target_quote["exdate"])
                state["strike"] = float(target_quote["strike"])
                state["c_strike"] = float(target_quote["c_strike"]) if "c_strike" in target_quote.index else float(target_quote["strike"])
                state["p_strike"] = float(target_quote["p_strike"]) if "p_strike" in target_quote.index else float(target_quote["strike"])
                state["c_mid"] = float(target_quote["c_mid"])
                state["p_mid"] = float(target_quote["p_mid"])
                state["spot"] = float(target_quote["spot"])
                state["c_iv"] = float(target_quote["c_iv"])
                state["p_iv"] = float(target_quote["p_iv"])
                state["r"] = float(target_quote["r"])
                state["t"] = float(target_quote["t"])
                state["t_days"] = float(target_quote["t_days"])

        return option_trade_cost, traded, state

    def _delta_hedge(self, state: Dict[str, object] | None, today: pd.DataFrame | None, instrument: str) -> tuple[float, float, float, float, float, float, float]:
        """Compute and apply delta hedge. Returns (hedge_cost, hedge_shares, n_contracts, spot_t, strike_t, tte_days_t, actual_dollar_vega)."""
        hedge_cost = 0.0
        hedge_shares = 0.0
        n_contracts = 0.0
        spot_t = np.nan
        strike_t = np.nan
        tte_days_t = np.nan
        actual_dollar_vega = 0.0

        if state is None or today is None:
            return hedge_cost, hedge_shares, n_contracts, spot_t, strike_t, tte_days_t, actual_dollar_vega

        q_live = self._get_quote(
            today,
            instrument=instrument,
            exdate=pd.Timestamp(state["exdate"]),
            c_strike=float(state["c_strike"]),
            p_strike=float(state["p_strike"]),
            strike=float(state["strike"]),
        )
        if q_live is not None:
            c_strike_live = float(q_live["c_strike"]) if "c_strike" in q_live.index else float(q_live["strike"])
            p_strike_live = float(q_live["p_strike"]) if "p_strike" in q_live.index else float(q_live["strike"])
            dc = _bs_delta("C", float(q_live["spot"]), c_strike_live, float(q_live["t"]), float(q_live["r"]), float(q_live["c_iv"]))
            dp = _bs_delta("P", float(q_live["spot"]), p_strike_live, float(q_live["t"]), float(q_live["r"]), float(q_live["p_iv"]))
            delta_total = float(state["q_call"]) * dc + float(state["q_put"]) * dp
            hedge_target = -self.contract_multiplier * delta_total
            hedge_trade = hedge_target - float(state["hedge_shares"])
            hedge_cost = abs(hedge_trade) * float(q_live["spot"]) * (self.stock_fee_bps * 1e-4)
            state["hedge_shares"] = float(hedge_target)
            hedge_shares = float(hedge_target)
            n_contracts = float(max(abs(float(state["q_call"])), abs(float(state["q_put"]))))
            spot_t = float(q_live["spot"])
            strike_t = float(0.5 * (c_strike_live + p_strike_live))
            tte_days_t = float(q_live["t_days"])
            vega_c_live = _bs_vega(float(q_live["spot"]), c_strike_live, float(q_live["t"]), float(q_live["r"]), float(q_live["c_iv"]))
            vega_p_live = _bs_vega(float(q_live["spot"]), p_strike_live, float(q_live["t"]), float(q_live["r"]), float(q_live["p_iv"]))
            actual_dollar_vega = float(
                self.contract_multiplier
                * (abs(float(state["q_call"])) * vega_c_live + abs(float(state["q_put"])) * vega_p_live)
            )

        return hedge_cost, hedge_shares, n_contracts, spot_t, strike_t, tte_days_t, actual_dollar_vega

    def _backtest_one(self, ticker: str, signal_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
        df = signal_df.sort_values("date").reset_index(drop=True).copy()
        if df.empty:
            return df
        required = {"target_weight", "signal_instrument"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Signal contract missing required columns {sorted(missing)} for ticker '{ticker}'. "
                "Signal stage must emit explicit execution fields."
            )
        df["date"] = pd.to_datetime(df["date"])
        by_date = {d: sub.copy() for d, sub in pair_df.groupby("date")} if not pair_df.empty else {}

        rows = []
        cum = 0.0
        prev_target_weight = 0.0
        state: Dict[str, object] | None = None

        for i in range(len(df)):
            d = pd.Timestamp(df.loc[i, "date"])
            pos = float(df.loc[i, "target_weight"])
            if "execution_enabled" in df.columns and not bool(df.loc[i, "execution_enabled"]):
                pos = 0.0
            instrument = str(df.loc[i, "signal_instrument"])
            turnover = abs(pos - prev_target_weight)
            target_dollar_vega = 0.0
            is_roll = False

            if instrument not in {"risk_reversal", "straddle"}:
                raise ValueError(f"Unsupported signal_instrument '{instrument}' for ticker={ticker} date={d.date()}.")

            today = by_date.get(d)

            # 1) Mark existing inventory to today's market.
            mark = self._mark_to_market(state, today)
            if mark is None:
                # Missing quote for live contract: flatten position.
                state = None
                mark = {
                    "gross_opt": 0.0,
                    "gross_hedge": 0.0,
                    "skew_vega_pnl": 0.0,
                    "level_vega_pnl": 0.0,
                    "gamma_pnl": 0.0,
                    "theta_pnl": 0.0,
                    "unexplained_pnl": 0.0,
                }
            gross_opt = mark["gross_opt"]
            gross_hedge = mark["gross_hedge"]
            gross = gross_opt + gross_hedge
            skew_vega_pnl = mark["skew_vega_pnl"]
            level_vega_pnl = mark["level_vega_pnl"]
            gamma_pnl = mark["gamma_pnl"]
            theta_pnl = mark["theta_pnl"]
            unexplained_pnl = mark["unexplained_pnl"]

            # 2) Choose target contract before sizing so size is based on actual package vega.
            target_quote = None
            same_contract = False
            if abs(pos) > 1e-12 and today is not None and not today.empty:
                target_quote, is_roll = self._select_contract(state, today, instrument)
                same_contract = (target_quote is not None and not is_roll and state is not None)

            # Sizing logic.
            target_q = 0.0
            if abs(pos) > 1e-12 and target_quote is not None:
                package_dollar_vega = self._quote_package_dollar_vega(target_quote, instrument=instrument)
                if np.isfinite(package_dollar_vega) and package_dollar_vega > 1e-12:
                    if self.sizing_mode == "dollar_vega":
                        desired_dollar_vega = abs(float(pos)) * float(self.target_dollar_vega_per_signal)
                        q_mag = min(
                            float(self.max_contracts_per_signal),
                            float(desired_dollar_vega) / float(package_dollar_vega),
                        )
                        target_dollar_vega = float(q_mag * package_dollar_vega)
                    elif self.sizing_mode == "contracts":
                        q_mag = min(float(self.max_contracts_per_signal), abs(float(pos)) * float(self.max_contracts_per_signal))
                        target_dollar_vega = float(q_mag * package_dollar_vega)
                    else:
                        raise ValueError(
                            f"Unsupported sizing_mode '{self.sizing_mode}'. Use one of: contracts, dollar_vega."
                        )
                    target_q = float(np.sign(pos) * q_mag)
            if instrument == "risk_reversal":
                target_q_call, target_q_put = target_q, -target_q
            else:
                target_q_call, target_q_put = target_q, target_q

            # 3) Execute close/open/resize on today's quote.
            option_trade_cost, traded, state = self._execute_trade(
                state, target_quote, target_q_call, target_q_put, today, instrument, same_contract
            )

            # 4) Delta-hedge rebalance.
            hedge_cost, hedge_shares, n_contracts, spot_t, strike_t, tte_days_t, actual_dollar_vega = self._delta_hedge(state, today, instrument)

            cost = option_trade_cost + hedge_cost

            net = gross - cost
            cum += net
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "position": pos,
                    "target_weight": pos,
                    "signal_instrument": instrument,
                    "residual": float(df.loc[i, "residual"]) if "residual" in df.columns and np.isfinite(df.loc[i, "residual"]) else np.nan,
                    "zscore": float(df.loc[i, "zscore"]) if "zscore" in df.columns and np.isfinite(df.loc[i, "zscore"]) else np.nan,
                    "stock_skew": float(df.loc[i, "stock_skew"]) if "stock_skew" in df.columns and np.isfinite(df.loc[i, "stock_skew"]) else np.nan,
                    "sector_skew": float(df.loc[i, "sector_skew"]) if "sector_skew" in df.columns and np.isfinite(df.loc[i, "sector_skew"]) else np.nan,
                    "trade_allowed": bool(df.loc[i, "trade_allowed"]) if "trade_allowed" in df.columns else False,
                    "turnover": float(turnover),
                    "n_contracts": float(n_contracts),
                    "target_dollar_vega": float(target_dollar_vega),
                    "actual_dollar_vega": float(actual_dollar_vega),
                    "hedge_shares": float(hedge_shares),
                    "spot": float(spot_t) if np.isfinite(spot_t) else np.nan,
                    "strike": float(strike_t) if np.isfinite(strike_t) else np.nan,
                    "call_strike": float(state["c_strike"]) if state is not None and np.isfinite(float(state["c_strike"])) else np.nan,
                    "put_strike": float(state["p_strike"]) if state is not None and np.isfinite(float(state["p_strike"])) else np.nan,
                    "tte_days": float(tte_days_t) if np.isfinite(tte_days_t) else np.nan,
                    "trade_executed": bool(traded),
                    "is_roll": bool(is_roll),
                    "option_trade_cost": float(option_trade_cost),
                    "hedge_cost": float(hedge_cost),
                    "skew_vega_pnl": float(skew_vega_pnl),
                    "level_vega_pnl": float(level_vega_pnl),
                    "gamma_pnl": float(gamma_pnl),
                    "theta_pnl": float(theta_pnl),
                    "unexplained_pnl": float(unexplained_pnl),
                    "gross_option_pnl": float(gross_opt),
                    "gross_hedge_pnl": float(gross_hedge),
                    "gross_pnl": float(gross),
                    "cost": float(cost),
                    "net_pnl": float(net),
                    "cum_net_pnl": float(cum),
                }
            )
            prev_target_weight = pos
        return pd.DataFrame(rows)

    def run(self, signals: SignalOutput, skew: SkewOutput, spec: RunSpec) -> BacktestOutput:
        by_ticker: Dict[str, pd.DataFrame] = {}
        for t, sig_df in signals.signal_map.items():
            pair_df = self._load_pairs_for_ticker(ticker=t, spec=spec)
            by_ticker[t] = self._backtest_one(ticker=t, signal_df=sig_df, pair_df=pair_df)

        parts = [
            df[
                [
                    "date",
                    "net_pnl",
                    "gross_pnl",
                    "gross_option_pnl",
                    "gross_hedge_pnl",
                    "skew_vega_pnl",
                    "level_vega_pnl",
                    "gamma_pnl",
                    "theta_pnl",
                    "unexplained_pnl",
                    "target_dollar_vega",
                    "actual_dollar_vega",
                    "cost",
                    "turnover",
                    "trade_executed",
                    "is_roll",
                ]
            ].assign(ticker=t)
            for t, df in by_ticker.items()
            if not df.empty
        ]
        if parts:
            all_rows = pd.concat(parts, ignore_index=True)
            portfolio = (
                all_rows.groupby("date", as_index=False)
                .agg(
                    portfolio_net_pnl=("net_pnl", "mean"),
                    portfolio_gross_pnl=("gross_pnl", "mean"),
                    portfolio_option_pnl=("gross_option_pnl", "mean"),
                    portfolio_hedge_pnl=("gross_hedge_pnl", "mean"),
                    portfolio_skew_vega_pnl=("skew_vega_pnl", "mean"),
                    portfolio_level_vega_pnl=("level_vega_pnl", "mean"),
                    portfolio_gamma_pnl=("gamma_pnl", "mean"),
                    portfolio_theta_pnl=("theta_pnl", "mean"),
                    portfolio_unexplained_pnl=("unexplained_pnl", "mean"),
                    portfolio_target_dollar_vega=("target_dollar_vega", "mean"),
                    portfolio_actual_dollar_vega=("actual_dollar_vega", "mean"),
                    portfolio_cost=("cost", "mean"),
                    portfolio_turnover=("turnover", "mean"),
                    portfolio_trades=("trade_executed", "sum"),
                    portfolio_rolls=("is_roll", "sum"),
                    n_names=("ticker", "nunique"),
                )
                .sort_values("date")
                .reset_index(drop=True)
            )
            portfolio["portfolio_cum_net_pnl"] = portfolio["portfolio_net_pnl"].cumsum()
        else:
            portfolio = pd.DataFrame(
                columns=[
                    "date",
                    "portfolio_net_pnl",
                    "portfolio_gross_pnl",
                    "portfolio_option_pnl",
                    "portfolio_hedge_pnl",
                    "portfolio_skew_vega_pnl",
                    "portfolio_level_vega_pnl",
                    "portfolio_gamma_pnl",
                    "portfolio_theta_pnl",
                    "portfolio_unexplained_pnl",
                    "portfolio_target_dollar_vega",
                    "portfolio_actual_dollar_vega",
                    "portfolio_cost",
                    "portfolio_turnover",
                    "portfolio_trades",
                    "portfolio_rolls",
                    "n_names",
                    "portfolio_cum_net_pnl",
                ]
            )

        if portfolio.empty:
            summary = {
                "n_days": 0.0,
                "avg_daily_net_pnl": 0.0,
                "std_daily_net_pnl": 0.0,
                "daily_sharpe_like": 0.0,
                "final_cum_net_pnl": 0.0,
            }
        else:
            r = portfolio["portfolio_net_pnl"].to_numpy(dtype=np.float64)
            mu = float(np.mean(r))
            sd = float(np.std(r))
            summary = {
                "n_days": float(len(portfolio)),
                "avg_daily_net_pnl": mu,
                "std_daily_net_pnl": sd,
                "daily_sharpe_like": float(mu / sd * np.sqrt(252.0)) if sd > 1e-12 else 0.0,
                "final_cum_net_pnl": float(portfolio["portfolio_cum_net_pnl"].iloc[-1]),
            }
        return BacktestOutput(portfolio=portfolio, by_ticker=by_ticker, summary=summary)
