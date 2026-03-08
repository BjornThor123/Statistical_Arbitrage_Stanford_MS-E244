from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict

import numpy as np
import pandas as pd

from interfaces import ModelOutput, RunSpec, SkewCalculatorModule, SkewOutput
from volatility_models.model import ssvi_dw_dk, ssvi_total_variance


@dataclass
class GenericSkewCalculator(SkewCalculatorModule):
    technique: str = "local_derivative"
    delta_k: float = 0.01
    rr_k: float = 0.25

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
            kp = np.asarray([k0 + self.rr_k], dtype=np.float64)
            km = np.asarray([k0 - self.rr_k], dtype=np.float64)
            sp = float(self._evaluate_sigma(model_family=model_family, state=state, k=kp, t=t)[0])
            sm = float(self._evaluate_sigma(model_family=model_family, state=state, k=km, t=t)[0])
            return float(sp - sm)
        raise ValueError(
            f"Unsupported skew technique '{self.technique}'. "
            "Use one of: local_derivative, finite_diff, rr_approx, risk_reversal, ssvi_analytic_derivative."
        )

    def _to_skew(self, df: pd.DataFrame, ticker_hint: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "skew",
                    "n_points",
                    "rmse_implied_volatility",
                    "technique",
                    "target_t",
                    "k0",
                    "model_family",
                ]
            )
        if "model_state" not in df.columns or "model_family" not in df.columns:
            raise ValueError(
                "Model output must include 'model_family' and 'model_state' for modular skew computation. "
                "Re-run run_model with the updated volatility model module."
            )

        out_rows = []
        for _, row in df.iterrows():
            model_family = str(row["model_family"])
            state = self._parse_state(row["model_state"])
            target_t = float(row.get("target_t", np.nan))
            k0 = float(row.get("k0", np.nan))
            if not np.isfinite(target_t):
                target_t = float(state.get("target_t", np.nan))
            if not np.isfinite(k0):
                k0 = 0.0
            skew_val = self._compute_skew_from_state(
                model_family=model_family,
                state=state,
                target_t=float(target_t),
                k0=float(k0),
            )
            out_rows.append(
                {
                    "date": pd.Timestamp(row["date"]),
                    "ticker": str(row.get("ticker", ticker_hint)),
                    "skew": float(skew_val),
                    "n_points": float(row.get("n_points", np.nan)),
                    "rmse_implied_volatility": float(row.get("rmse_implied_volatility", np.nan)),
                    "technique": self.technique,
                    "target_t": float(target_t),
                    "k0": float(k0),
                    "model_family": model_family,
                }
            )
        return pd.DataFrame(out_rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    def compute(self, modeled: ModelOutput, spec: RunSpec) -> SkewOutput:
        skew_by_ticker = {t: self._to_skew(df, ticker_hint=t) for t, df in modeled.model_by_ticker.items()}
        return SkewOutput(skew_by_ticker=skew_by_ticker)
