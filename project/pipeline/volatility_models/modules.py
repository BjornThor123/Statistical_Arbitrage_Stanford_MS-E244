from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from interfaces import ModelOutput, ProcessedData, RunSpec, VolatilityModelModule
from volatility_models.model import CalibratedSSVISurface, calibrate_ssvi_surface


@dataclass
class SSVISurfaceModeler(VolatilityModelModule):
    """Fits an SSVI surface to each day's option cross-section for each ticker."""

    n_theta_steps: int = 3
    n_restarts: int = 4
    raise_on_all_calibration_failures: bool = True
    show_progress: bool = True

    def _make_model_row(self, surf: CalibratedSSVISurface, spec: RunSpec, day: pd.DataFrame, date: object) -> Dict[str, object]:
        """Build the output row dict for a single calibrated surface."""
        k = day["k"].to_numpy(dtype=np.float64)
        t = day["t"].to_numpy(dtype=np.float64)
        sigma = day["sigma"].to_numpy(dtype=np.float64)
        skew = surf.sigma_skew(t=spec.target_t, k0=spec.k0)
        theta_target = float(surf.theta_at(np.asarray([spec.target_t]))[0])
        atm_iv = float(np.sqrt(max(theta_target, 1e-12) / max(spec.target_t, 1e-12)))
        day_fwd = day[["t", "F"]].drop_duplicates(subset=["t"])
        closest_idx = (day_fwd["t"] - spec.target_t).abs().idxmin()
        forward_price = float(day_fwd.loc[closest_idx, "F"])
        return {
            "date": pd.Timestamp(date),
            "ticker": str(day["ticker"].iloc[0]),
            "skew": float(skew),
            "target_t": float(spec.target_t),
            "k0": float(spec.k0),
            "rho": float(surf.params.rho),
            "eta": float(surf.params.eta),
            "gamma": float(surf.params.gamma),
            "theta_target": theta_target,
            "atm_iv": atm_iv,
            "forward_price": forward_price,
            "n_points": int(surf.diagnostics.n_points),
            "rmse_implied_volatility": float(surf.diagnostics.rmse_implied_volatility),
            "model_family": "ssvi_surface",
            "model_state": json.dumps(
                {
                    "rho": float(surf.params.rho),
                    "eta": float(surf.params.eta),
                    "gamma": float(surf.params.gamma),
                    "t_nodes": surf.t_nodes.tolist(),
                    "theta_nodes": surf.theta_nodes.tolist(),
                }
            ),
            "obs_k": k.tolist(),
            "obs_t": t.tolist(),
            "obs_sigma": sigma.tolist(),
        }

    def _fit_one(self, panel: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        n_calib_fail = 0
        first_calib_error: Exception | None = None
        prev_params = None
        prev_theta = None
        day_groups = list(panel.groupby("date"))
        day_iter = tqdm(day_groups, desc="SSVI days", leave=False) if self.show_progress else day_groups
        for d, day in day_iter:
            k = day["k"].to_numpy(dtype=np.float64)
            t = day["t"].to_numpy(dtype=np.float64)
            sigma = day["sigma"].to_numpy(dtype=np.float64)

            try:
                surf: CalibratedSSVISurface = calibrate_ssvi_surface(
                    k=k,
                    t=t,
                    sigma=sigma,
                    n_theta_steps=self.n_theta_steps,
                    n_restarts=self.n_restarts,
                    initial_params=prev_params,
                    initial_theta_nodes=prev_theta,
                )
            except Exception as exc:
                n_calib_fail += 1
                if first_calib_error is None:
                    first_calib_error = exc
                ticker_name = str(day["ticker"].iloc[0]) if "ticker" in day.columns and len(day) else "unknown"
                print(
                    "SSVI calibration error | "
                    f"ticker={ticker_name} date={pd.Timestamp(d).date()} "
                    f"error={type(exc).__name__}: {exc}"
                )
                continue

            prev_params = (surf.params.rho, surf.params.eta, surf.params.gamma)
            prev_theta = surf.theta_nodes.copy()
            rows.append(self._make_model_row(surf, spec, day, d))
        if not rows:
            if self.raise_on_all_calibration_failures and n_calib_fail > 0 and first_calib_error is not None:
                raise RuntimeError(
                    "All SSVI calibrations failed for this ticker. "
                    f"first_error={type(first_calib_error).__name__}: {first_calib_error}"
                ) from first_calib_error
            return pd.DataFrame(columns=["date", "ticker", "skew"])
        return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    def fit(self, processed: ProcessedData, spec: RunSpec) -> ModelOutput:
        items = list(processed.panel_by_ticker.items())
        ticker_iter = tqdm(items, desc="SSVI tickers") if self.show_progress else items
        model_by_ticker = {t: self._fit_one(panel, spec) for t, panel in ticker_iter}
        return ModelOutput(model_by_ticker=model_by_ticker, representation="surface")


@dataclass
class LinearSmileModeler(VolatilityModelModule):
    """Fits a linear IV smile (IV = intercept + slope * k) near target maturity."""

    t_band: float = 10.0 / 365.0
    min_points_per_day: int = 60
    show_progress: bool = True

    def _fit_one(self, panel: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        day_groups = list(panel.groupby("date"))
        day_iter = tqdm(day_groups, desc="Smile days", leave=False) if self.show_progress else day_groups
        for d, day in day_iter:
            sub = day[np.abs(day["t"] - spec.target_t) <= self.t_band].copy()
            k = sub["k"].to_numpy(dtype=np.float64)
            sigma = sub["sigma"].to_numpy(dtype=np.float64)
            if np.unique(k).shape[0] < 5:
                continue
            slope, intercept = np.polyfit(k, sigma, deg=1)
            pred = intercept + slope * k
            rows.append(
                {
                    "date": pd.Timestamp(d),
                    "ticker": str(sub["ticker"].iloc[0]),
                    "smile_slope": float(slope),
                    "smile_intercept": float(intercept),
                    "target_t": float(spec.target_t),
                    "n_points": int(len(sub)),
                    "rmse_implied_volatility": float(np.sqrt(np.mean((sigma - pred) ** 2))),
                    "model_family": "linear_smile",
                    "model_state": json.dumps(
                        {
                            "slope": float(slope),
                            "intercept": float(intercept),
                            "target_t": float(spec.target_t),
                        }
                    ),
                    "obs_k": k.tolist(),
                    "obs_sigma": sigma.tolist(),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["date", "ticker", "smile_slope"])
        return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    def fit(self, processed: ProcessedData, spec: RunSpec) -> ModelOutput:
        items = list(processed.panel_by_ticker.items())
        ticker_iter = tqdm(items, desc="Smile tickers") if self.show_progress else items
        model_by_ticker = {t: self._fit_one(panel, spec) for t, panel in ticker_iter}
        return ModelOutput(model_by_ticker=model_by_ticker, representation="smile")
