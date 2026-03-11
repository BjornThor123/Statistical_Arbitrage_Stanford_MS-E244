from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from interfaces import ModelOutput, RunSpec, SkewCalculatorModule, SkewOutput
from volatility_models.model import ssvi_dw_dk, ssvi_total_variance

"""
Skew calculator.

GenericSkewCalculator reads the fitted volatility model state from ModelOutput
and computes a scalar skew measure for each ticker-date across all configured
tenors (tenor_days). Output rows are keyed by (ticker, date, tenor_days).

  technique="local_derivative"             →  dσ/dk  via finite difference (default)
  technique="rr_approx"                    →  σ(k_call) - σ(k_put) at symmetric log-moneyness
  technique="ssvi_analytic_derivative"     → exact dσ/dk from SSVI formula

Strike selection for rr_approx:
  rr_delta > 0  →  target-delta mode: find the log-moneyness of the rr_delta-delta call and
                    put via a Newton iteration on the SSVI surface, then compute σ(k_call) - σ(k_put).
                    This is the standard "25-delta risk reversal" when rr_delta=0.25.
  rr_delta = 0  →  fixed log-moneyness mode: call at k0+rr_k, put at k0-rr_k.

tenor_days controls which maturities are evaluated. Each entry produces one row
per (ticker, date). Downstream consumers (SignalGenerator, analytics) filter by
the tenor they need.
"""

from math import erf as _erf, sqrt as _sqrt, log as _log, exp as _exp


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + _erf(x / _sqrt(2.0)))


def _nppf(p: float) -> float:
    """Inverse normal CDF via rational approximation (Beasley-Springer-Moro)."""
    p = max(1e-10, min(1.0 - 1e-10, p))
    if p < 0.5:
        return -_nppf(1.0 - p)
    q = p - 0.5
    r = q * q
    a = 2.515517 + r * (0.802853 + r * 0.010328)
    b = 1.0 + r * (1.432788 + r * (0.189269 + r * 0.001308))
    return q + q * r * (-a / b) / (1.0 + r * (1.432788 + r * (0.189269 + r * 0.001308)))


@dataclass
class GenericSkewCalculator(SkewCalculatorModule):
    technique: str = "local_derivative"
    delta_k: float = 0.01
    rr_k: float = 0.15
    # When rr_delta > 0, strike selection for rr_approx uses the log-moneyness
    # that corresponds to this BS delta (e.g. 0.25 = 25-delta RR).
    # When rr_delta = 0, falls back to fixed log-moneyness ±rr_k.
    rr_delta: float = 0.25
    tenor_days: List[int] = field(default_factory=lambda: [30, 60, 90])

    def __post_init__(self) -> None:
        if isinstance(self.tenor_days, int):
            self.tenor_days = [self.tenor_days]

    @staticmethod
    def _parse_state(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            return dict(json.loads(raw))
        return {}

    @staticmethod
    def _theta_interp(t: np.ndarray, t_nodes: np.ndarray, theta_nodes: np.ndarray) -> np.ndarray:
        return np.interp(
            np.maximum(t, t_nodes[0]),
            t_nodes,
            theta_nodes,
            left=theta_nodes[0],
            right=theta_nodes[-1],
        )

    def _evaluate_sigma(
        self,
        model_family: str,
        state: Dict[str, Any],
        k: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        fam = str(model_family).lower()
        if fam == "ssvi_surface":
            rho = float(state["rho"])
            eta = float(state["eta"])
            gamma = float(state["gamma"])
            t_nodes = np.asarray(state["t_nodes"], dtype=np.float64)
            theta_nodes = np.asarray(state["theta_nodes"], dtype=np.float64)
            theta = self._theta_interp(t=t, t_nodes=t_nodes, theta_nodes=theta_nodes)
            w = ssvi_total_variance(k=k, theta=theta, rho=rho, eta=eta, gamma=gamma)
            return np.sqrt(np.maximum(w, 1e-12) / np.maximum(t, 1e-12))
        if fam == "linear_smile":
            slope = float(state["slope"])
            intercept = float(state["intercept"])
            return intercept + slope * k
        raise ValueError(f"Unsupported model_family '{model_family}'.")

    def _find_delta_k(
        self,
        model_family: str,
        state: Dict[str, Any],
        target_t: float,
        call_delta: float,
    ) -> float:
        """Return the log-moneyness k such that BS delta equals call_delta.

        Uses a Newton-style iteration: given σ(k) from the SSVI surface,
          k = σ²T/2 - d1_target * σ√T
        where d1_target = Φ⁻¹(call_delta).  Three iterations are sufficient.
        Works for both calls (call_delta ∈ (0,0.5]) and puts expressed as
        call-equivalent delta (call_delta = 1 - |put_delta|).
        """
        t = float(max(target_t, 1e-6))
        d1_target = _nppf(float(call_delta))
        sqrt_t = _sqrt(t)
        # Seed from ATM vol
        t_arr = np.asarray([t], dtype=np.float64)
        sigma = float(self._evaluate_sigma(model_family, state, np.zeros(1), t_arr)[0])
        sigma = max(sigma, 1e-4)
        k = sigma ** 2 * t / 2.0 - d1_target * sigma * sqrt_t
        for _ in range(4):
            k_arr = np.asarray([k], dtype=np.float64)
            sigma = float(self._evaluate_sigma(model_family, state, k_arr, t_arr)[0])
            sigma = max(sigma, 1e-4)
            k = sigma ** 2 * t / 2.0 - d1_target * sigma * sqrt_t
        return float(k)

    def _compute_skew_from_state(
        self,
        model_family: str,
        state: Dict[str, Any],
        target_t: float,
        k0: float,
    ) -> float:
        if self.technique == "ssvi_analytic_derivative":
            if str(model_family).lower() != "ssvi_surface":
                raise ValueError("Technique 'ssvi_analytic_derivative' is only supported for model_family='ssvi_surface'.")
            rho = float(state["rho"])
            eta = float(state["eta"])
            gamma = float(state["gamma"])
            t_nodes = np.asarray(state["t_nodes"], dtype=np.float64)
            theta_nodes = np.asarray(state["theta_nodes"], dtype=np.float64)
            t = np.asarray([target_t], dtype=np.float64)
            k = np.asarray([k0], dtype=np.float64)
            theta = self._theta_interp(t=t, t_nodes=t_nodes, theta_nodes=theta_nodes)
            w = ssvi_total_variance(k=k, theta=theta, rho=rho, eta=eta, gamma=gamma)
            dwdk = ssvi_dw_dk(k=k, theta=theta, rho=rho, eta=eta, gamma=gamma)
            denom = 2.0 * np.sqrt(np.maximum(w, 1e-12)) * np.sqrt(np.maximum(t, 1e-12))
            return float((dwdk / np.maximum(denom, 1e-12))[0])

        t = np.asarray([target_t], dtype=np.float64)
        if self.technique in {"local_derivative", "finite_diff"}:
            kp = np.asarray([k0 + self.delta_k], dtype=np.float64)
            km = np.asarray([k0 - self.delta_k], dtype=np.float64)
            sp = float(self._evaluate_sigma(model_family=model_family, state=state, k=kp, t=t)[0])
            sm = float(self._evaluate_sigma(model_family=model_family, state=state, k=km, t=t)[0])
            return float((sp - sm) / (2.0 * self.delta_k))
        if self.technique in {"rr_approx", "risk_reversal"}:
            if self.rr_delta > 0.0:
                # Delta-targeting: find the log-moneyness of the rr_delta-delta call and put.
                # call leg: delta = rr_delta (e.g. 0.25) → k > 0 (OTM call)
                # put leg:  delta = -(rr_delta) → expressed as call-equivalent = 1 - rr_delta
                k_call = self._find_delta_k(model_family, state, target_t, self.rr_delta)
                k_put  = self._find_delta_k(model_family, state, target_t, 1.0 - self.rr_delta)
            else:
                k_call = k0 + self.rr_k
                k_put  = k0 - self.rr_k
            sp = float(self._evaluate_sigma(model_family=model_family, state=state,
                                            k=np.asarray([k_call]), t=t)[0])
            sm = float(self._evaluate_sigma(model_family=model_family, state=state,
                                            k=np.asarray([k_put]), t=t)[0])
            return float(sp - sm)
        raise ValueError(
            f"Unsupported skew technique '{self.technique}'. "
            "Use one of: local_derivative, finite_diff, rr_approx, risk_reversal, ssvi_analytic_derivative."
        )

    def _to_skew(self, df: pd.DataFrame, ticker_hint: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[
                "date", "ticker", "tenor_days", "target_t", "skew",
                "n_points", "rmse_implied_volatility", "technique", "k0", "model_family",
            ])
        if "model_state" not in df.columns or "model_family" not in df.columns:
            raise ValueError(
                "Model output must include 'model_family' and 'model_state' for modular skew computation. "
                "Re-run run_model with the updated volatility model module."
            )

        out_rows = []
        for _, row in df.iterrows():
            model_family = str(row["model_family"])
            state = self._parse_state(row["model_state"])
            k0 = float(row.get("k0", 0.0))
            if not np.isfinite(k0):
                k0 = 0.0
            for td in self.tenor_days:
                target_t = td / 365.0
                skew_val = self._compute_skew_from_state(
                    model_family=model_family,
                    state=state,
                    target_t=target_t,
                    k0=k0,
                )
                out_rows.append({
                    "date": pd.Timestamp(row["date"]),
                    "ticker": str(row.get("ticker", ticker_hint)),
                    "tenor_days": int(td),
                    "target_t": float(target_t),
                    "skew": float(skew_val),
                    "n_points": float(row.get("n_points", np.nan)),
                    "rmse_implied_volatility": float(row.get("rmse_implied_volatility", np.nan)),
                    "technique": self.technique,
                    "k0": float(k0),
                    "model_family": model_family,
                })
        return (
            pd.DataFrame(out_rows)
            .sort_values(["ticker", "date", "tenor_days"])
            .reset_index(drop=True)
        )

    def compute(self, modeled: ModelOutput, spec: RunSpec) -> SkewOutput:
        skew_by_ticker = {t: self._to_skew(df, ticker_hint=t) for t, df in modeled.model_by_ticker.items()}
        return SkewOutput(skew_by_ticker=skew_by_ticker)
