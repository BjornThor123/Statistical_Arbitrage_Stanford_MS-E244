from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from interfaces import RunSpec, SignalGeneratorModule, SignalOutput, SkewOutput


def _winsorize(x: np.ndarray, q: float) -> np.ndarray:
    if len(x) == 0:
        return x
    lo = float(np.quantile(x, q))
    hi = float(np.quantile(x, 1.0 - q))
    return np.clip(x, lo, hi)


def _robust_center_scale(x: np.ndarray, use_mad: bool) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0, np.nan
    if use_mad:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        return med, float(1.4826 * mad)
    return float(np.mean(x)), float(np.std(x))


def _estimate_unit_cost(n_points: float, rmse_iv: float, half_spread_cost: float, commission_cost: float) -> float:
    n = max(float(n_points), 1.0)
    rmse = max(float(rmse_iv), 1e-6)
    liquidity_pen = 0.5 * np.sqrt(150.0 / n)
    fit_pen = min(1.5, rmse / 0.25)
    base = half_spread_cost + commission_cost
    return float(base * (1.0 + liquidity_pen + fit_pen))


def _sector_regime_flags(sector_skew: pd.DataFrame, vol_window: int, abs_z_max: float, vol_z_max: float) -> pd.DataFrame:
    s = sector_skew[["date", "skew"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)
    w = max(10, int(vol_window))
    s["roll_mu"] = s["skew"].rolling(w).mean()
    s["roll_sd"] = s["skew"].rolling(w).std(ddof=0)
    s["sector_abs_z"] = (s["skew"] - s["roll_mu"]) / s["roll_sd"].replace(0.0, np.nan)
    s["sector_vol_level"] = s["roll_sd"]
    vol_mu = s["sector_vol_level"].rolling(w).mean()
    vol_sd = s["sector_vol_level"].rolling(w).std(ddof=0)
    s["sector_vol_z"] = (s["sector_vol_level"] - vol_mu) / vol_sd.replace(0.0, np.nan)
    s["regime_pass"] = (s["sector_abs_z"].abs().fillna(0.0) <= abs_z_max) & (s["sector_vol_z"].fillna(0.0) <= vol_z_max)
    return s[["date", "regime_pass", "sector_abs_z", "sector_vol_z"]]


@dataclass
class ResidualZScoreSignalGenerator(SignalGeneratorModule):
    benchmark_preference: str = "XLF"
    regression_window: int = 60
    min_regression_obs: int = 40
    zscore_window: int = 60
    entry_z: float = 1.25
    exit_z: float = 0.30
    min_hold_days: int = 3
    max_abs_position: float = 1.0
    min_liquidity_points: float = 120.0
    max_fit_rmse_iv: float = 0.50
    edge_horizon_days: int = 5
    edge_cost_buffer: float = 0.25
    half_spread_cost: float = 0.02
    commission_cost: float = 0.005
    signal_direction: str = "auto"  # mean_revert | momentum | auto
    cross_section_top_k: int = 2
    cross_section_bottom_k: int = 2
    max_name_weight: float = 0.35
    max_gross_leverage: float = 1.5
    vol_target_daily: float = 0.02
    signal_instrument: str = "straddle"
    signal_direction_mode: str = "fixed"  # fixed | auto
    auto_sign_window: int = 63
    winsor_quantile: float = 0.02
    use_mad_zscore: bool = True
    regime_sector_vol_window: int = 21
    regime_sector_abs_z_max: float = 3.0
    regime_sector_vol_z_max: float = 2.0
    factor_model: str = "sector"  # sector | pca
    n_pca_factors: int = 2

    def _build_signal_from_features(
        self,
        stock: pd.DataFrame,
        factor_df: pd.DataFrame,
        feature_cols: List[str],
        regime: pd.DataFrame,
    ) -> pd.DataFrame:
        s = stock[["date", "ticker", "skew", "n_points", "rmse_implied_volatility"]].rename(columns={"skew": "stock_skew"})
        df = s.merge(factor_df, on="date", how="inner").merge(regime, on="date", how="left").sort_values("date").reset_index(drop=True)
        if df.empty:
            return df

        rows: List[Dict[str, object]] = []
        resid_hist: List[float] = []
        prev_pos = 0.0
        hold_days = 0
        for i in range(len(df)):
            row = df.iloc[i]
            start = max(0, i - self.regression_window)
            hist = df.iloc[start:i].copy()
            valid = np.isfinite(hist["stock_skew"])
            for c in feature_cols:
                valid &= np.isfinite(hist[c])
            hist = hist[valid]

            resid = np.nan
            z = np.nan
            sd = np.nan
            alpha = np.nan
            betas = [np.nan for _ in feature_cols]
            if len(hist) >= self.min_regression_obs:
                y_hist = _winsorize(hist["stock_skew"].to_numpy(dtype=np.float64), self.winsor_quantile)
                x_hist = [_winsorize(hist[c].to_numpy(dtype=np.float64), self.winsor_quantile) for c in feature_cols]
                x_now = np.asarray([row.get(c, np.nan) for c in feature_cols], dtype=np.float64)
                if np.all(np.isfinite(x_now)):
                    X = np.column_stack([np.ones(len(y_hist), dtype=np.float64), *x_hist])
                    coef, *_ = np.linalg.lstsq(X, y_hist, rcond=None)
                    alpha = float(coef[0])
                    betas = [float(v) for v in coef[1:]]
                    y_hat = alpha + float(np.dot(np.asarray(betas, dtype=np.float64), x_now))
                    resid = float(row["stock_skew"] - y_hat)
                    resid_hist.append(resid)

                    if len(resid_hist) >= self.zscore_window:
                        tail = np.asarray(resid_hist[-self.zscore_window:], dtype=np.float64)
                        mu, sd = _robust_center_scale(tail, use_mad=self.use_mad_zscore)
                        if sd > 1e-12:
                            z = (resid - mu) / sd

            n_points = float(row.get("n_points", np.nan))
            rmse_iv = float(row.get("rmse_implied_volatility", np.nan))
            liquidity_pass = np.isfinite(n_points) and (n_points >= self.min_liquidity_points)
            fit_pass = np.isfinite(rmse_iv) and (rmse_iv <= self.max_fit_rmse_iv)
            est_unit_cost = _estimate_unit_cost(
                n_points=n_points,
                rmse_iv=rmse_iv,
                half_spread_cost=self.half_spread_cost,
                commission_cost=self.commission_cost,
            )
            expected_edge = abs(float(resid)) / max(self.edge_horizon_days, 1) if np.isfinite(resid) else np.nan
            edge_pass = np.isfinite(expected_edge) and (expected_edge > est_unit_cost * (1.0 + self.edge_cost_buffer))
            regime_pass = bool(row.get("regime_pass", True)) if pd.notna(row.get("regime_pass", np.nan)) else True
            trade_allowed = bool(np.isfinite(z) and liquidity_pass and fit_pass and edge_pass and regime_pass)

            if trade_allowed:
                if abs(z) >= self.entry_z:
                    pos = float(np.clip(-z, -self.max_abs_position, self.max_abs_position))
                    hold_days = 0
                elif abs(z) <= self.exit_z:
                    if hold_days >= self.min_hold_days:
                        pos = 0.0
                        hold_days = 0
                    else:
                        pos = prev_pos
                        hold_days += 1
                else:
                    pos = prev_pos
                    hold_days += 1
            else:
                pos = 0.0
                hold_days = 0

            out_row = {
                "date": pd.Timestamp(row["date"]),
                "ticker": str(row["ticker"]),
                "stock_skew": float(row["stock_skew"]),
                "sector_skew": float(row["sector_skew"]) if "sector_skew" in row and np.isfinite(row["sector_skew"]) else np.nan,
                "residual": float(resid) if np.isfinite(resid) else np.nan,
                "zscore": float(z) if np.isfinite(z) else np.nan,
                "position": float(pos),
                "trade_allowed": trade_allowed,
                "regime_pass": regime_pass,
                "liquidity_pass": liquidity_pass,
                "fit_pass": fit_pass,
                "expected_edge": float(expected_edge) if np.isfinite(expected_edge) else np.nan,
                "estimated_unit_cost": float(est_unit_cost),
                "residual_scale": float(sd) if np.isfinite(sd) else np.nan,
                "n_points": float(row.get("n_points", np.nan)),
                "rmse_implied_volatility": float(row.get("rmse_implied_volatility", np.nan)),
                "factor_model": self.factor_model,
                "alpha": float(alpha) if np.isfinite(alpha) else np.nan,
            }
            for c in feature_cols:
                out_row[c] = float(row[c]) if c in row and np.isfinite(row[c]) else np.nan
            for j, beta in enumerate(betas, start=1):
                out_row[f"beta_{j}"] = float(beta) if np.isfinite(beta) else np.nan
            rows.append(out_row)
            prev_pos = pos

        return pd.DataFrame(rows)

    def _build_pca_factor_frame(self, skew: SkewOutput, benchmark: str, benchmark_skew: pd.DataFrame) -> pd.DataFrame:
        stock_parts = []
        for t, df in skew.skew_by_ticker.items():
            if t == benchmark or df is None or df.empty:
                continue
            sub = df[["date", "skew"]].copy()
            sub["date"] = pd.to_datetime(sub["date"])
            sub["ticker"] = t
            stock_parts.append(sub)
        if not stock_parts:
            raise ValueError("PCA factor model requires at least one non-benchmark ticker with skew data.")

        panel = pd.concat(stock_parts, ignore_index=True)
        wide = panel.pivot_table(index="date", columns="ticker", values="skew", aggfunc="last").sort_index()
        pc_cols = [f"pc{i + 1}" for i in range(max(1, int(self.n_pca_factors)))]
        rows: List[Dict[str, object]] = []
        dates = list(wide.index)
        for i, d in enumerate(dates):
            row = {"date": pd.Timestamp(d)}
            for c in pc_cols:
                row[c] = np.nan
            start = max(0, i - self.regression_window)
            hist = wide.iloc[start:i].copy()
            if len(hist) < self.min_regression_obs:
                rows.append(row)
                continue
            cols_ok = hist.columns[hist.notna().mean() >= 0.6]
            hist = hist[cols_ok]
            if hist.shape[1] < 2:
                rows.append(row)
                continue
            hist_f = hist.fillna(hist.mean())
            mu = hist_f.mean(axis=0)
            sd = hist_f.std(axis=0).replace(0.0, 1.0)
            z_hist = (hist_f - mu) / sd
            x = z_hist.to_numpy(dtype=np.float64)
            try:
                _, _, vt = np.linalg.svd(x, full_matrices=False)
            except np.linalg.LinAlgError:
                rows.append(row)
                continue
            k = max(1, min(int(self.n_pca_factors), vt.shape[0], x.shape[1]))
            load = vt[:k].T
            row_now = wide.loc[d, cols_ok].copy().fillna(mu)
            z_now = ((row_now - mu) / sd).to_numpy(dtype=np.float64).reshape(1, -1)
            f_now = z_now @ load
            for j in range(k):
                row[pc_cols[j]] = float(f_now[0, j])
            rows.append(row)

        pca_df = pd.DataFrame(rows)
        sector_df = benchmark_skew[["date", "skew"]].rename(columns={"skew": "sector_skew"}).copy()
        sector_df["date"] = pd.to_datetime(sector_df["date"])
        return sector_df.merge(pca_df, on="date", how="left").sort_values("date").reset_index(drop=True)

    def _build_signal_map(self, skew: SkewOutput, benchmark: str, benchmark_skew: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        regime = _sector_regime_flags(
            sector_skew=benchmark_skew,
            vol_window=self.regime_sector_vol_window,
            abs_z_max=self.regime_sector_abs_z_max,
            vol_z_max=self.regime_sector_vol_z_max,
        )
        if self.factor_model == "sector":
            factor_df = benchmark_skew[["date", "skew"]].rename(columns={"skew": "sector_skew"}).copy()
            factor_df["date"] = pd.to_datetime(factor_df["date"])
            feature_cols = ["sector_skew"]
        elif self.factor_model == "pca":
            factor_df = self._build_pca_factor_frame(skew=skew, benchmark=benchmark, benchmark_skew=benchmark_skew)
            feature_cols = [f"pc{i + 1}" for i in range(max(1, int(self.n_pca_factors)))]
        else:
            raise ValueError("factor_model must be one of: sector, pca")

        return {
            t: self._build_signal_from_features(stock=df, factor_df=factor_df, feature_cols=feature_cols, regime=regime)
            for t, df in skew.skew_by_ticker.items()
            if t != benchmark
        }

    def _select_benchmark(self, tickers: List[str]) -> str:
        pref_upper = self.benchmark_preference.upper()
        for t in tickers:
            if t.upper() == pref_upper:
                return t
        if not tickers:
            raise ValueError("No tickers available to select benchmark.")
        return tickers[0]

    def _attach_execution_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if "signal_instrument" not in out.columns:
            out["signal_instrument"] = self.signal_instrument
        out["target_weight"] = pd.to_numeric(out["position"], errors="coerce").fillna(0.0).astype(float)
        out["execution_enabled"] = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0).abs() > 0.0
        return out

    def _infer_direction_multiplier_auto(self, current_idx: int, dates: List[pd.Timestamp], all_sig: pd.DataFrame) -> float:
        if current_idx < 2:
            return 1.0
        start = max(0, current_idx - max(20, int(self.auto_sign_window)))
        hist_dates = set(dates[start:current_idx])
        h = all_sig[all_sig["date"].isin(hist_dates)].copy()
        if h.empty:
            return 1.0
        h = h.sort_values(["ticker", "date"]).reset_index(drop=True)
        h["resid_next"] = h.groupby("ticker")["residual"].shift(-1)
        h = h[np.isfinite(h["zscore"]) & np.isfinite(h["residual"]) & np.isfinite(h["resid_next"])]
        if len(h) < 20:
            return 1.0
        score = np.mean(h["zscore"].to_numpy(dtype=np.float64) * (h["resid_next"].to_numpy(dtype=np.float64) - h["residual"].to_numpy(dtype=np.float64)))
        return 1.0 if float(score) >= 0.0 else -1.0

    def _direction_multiplier_for_date(self, current_idx: int, dates: List[pd.Timestamp], all_sig: pd.DataFrame) -> float:
        if self.signal_direction_mode == "auto" or str(self.signal_direction).lower() == "auto":
            return self._infer_direction_multiplier_auto(current_idx, dates, all_sig)
        if self.signal_direction == "mean_revert":
            return 1.0
        if self.signal_direction == "momentum":
            return -1.0
        raise ValueError("signal_direction must be one of: mean_revert, momentum, auto")

    def _candidate_weights(self, sub: pd.DataFrame, direction_multiplier: float) -> Dict[str, float]:
        if sub.empty:
            return {}
        z_eff = direction_multiplier * pd.to_numeric(sub["zscore"], errors="coerce")
        x = sub.assign(_z_eff=z_eff).copy()
        long_cands = x[x["_z_eff"] <= -self.entry_z].copy().sort_values("_z_eff", ascending=True).head(self.cross_section_bottom_k)
        short_cands = x[x["_z_eff"] >= self.entry_z].copy().sort_values("_z_eff", ascending=False).head(self.cross_section_top_k)

        out: Dict[str, float] = {}
        if not long_cands.empty:
            zabs = long_cands["_z_eff"].abs().to_numpy(dtype=np.float64)
            zsum = float(np.sum(zabs))
            for _, row in long_cands.iterrows():
                out[str(row["ticker"])] = float(0.5 * abs(float(row["_z_eff"])) / max(zsum, 1e-12))
        if not short_cands.empty:
            zabs = short_cands["_z_eff"].abs().to_numpy(dtype=np.float64)
            zsum = float(np.sum(zabs))
            for _, row in short_cands.iterrows():
                out[str(row["ticker"])] = float(-0.5 * abs(float(row["_z_eff"])) / max(zsum, 1e-12))
        return out

    def _apply_risk_scaling(self, target_weights: Dict[str, float], date_rows: Dict[str, pd.Series]) -> Dict[str, float]:
        if not target_weights:
            return {}
        raw: Dict[str, float] = {}
        for t, w in target_weights.items():
            row = date_rows.get(t)
            rs = float(row.get("residual_scale", np.nan)) if row is not None else np.nan
            inv_vol = 1.0 / max(rs, 1e-3) if np.isfinite(rs) and rs > 0.0 else 1.0
            raw[t] = float(w * inv_vol)

        gross = float(sum(abs(v) for v in raw.values()))
        if gross <= 1e-12:
            return {}
        scaled = {t: float(v / gross) for t, v in raw.items()}
        clipped = {t: float(np.clip(v, -self.max_name_weight, self.max_name_weight)) for t, v in scaled.items()}
        gross2 = float(sum(abs(v) for v in clipped.values()))
        if gross2 > self.max_gross_leverage and gross2 > 1e-12:
            clipped = {t: float(v * (self.max_gross_leverage / gross2)) for t, v in clipped.items()}

        vol_proxy = 0.0
        for t, w in clipped.items():
            row = date_rows.get(t)
            rs = float(row.get("residual_scale", np.nan)) if row is not None else np.nan
            rs = rs if np.isfinite(rs) and rs > 1e-6 else 0.02
            vol_proxy += (abs(w) * rs) ** 2
        vol_proxy = float(np.sqrt(vol_proxy))
        if vol_proxy > 1e-12:
            gross3 = float(sum(abs(v) for v in clipped.values()))
            f = min(self.max_gross_leverage / max(gross3, 1e-12), self.vol_target_daily / vol_proxy)
            clipped = {t: float(v * f) for t, v in clipped.items()}
        return clipped

    def generate(self, skew: SkewOutput, spec: RunSpec) -> SignalOutput:
        tickers = list(skew.skew_by_ticker.keys())
        benchmark = self._select_benchmark(tickers)
        benchmark_skew = skew.skew_by_ticker[benchmark]
        signal_map = self._build_signal_map(skew=skew, benchmark=benchmark, benchmark_skew=benchmark_skew)
        if not signal_map:
            return SignalOutput(signal_map=signal_map)

        stacked = []
        for t, df in signal_map.items():
            if df is None or df.empty:
                continue
            x = df.copy()
            x["ticker"] = t
            stacked.append(x)
        if stacked:
            all_sig = pd.concat(stacked, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
            all_sig["position_cs"] = 0.0
            all_sig["target_weight_raw"] = 0.0
            all_sig["direction_multiplier"] = 1.0
            all_sig["gross_leverage"] = 0.0
            all_dates = list(all_sig["date"].drop_duplicates().sort_values())
            prev_w = {str(t): 0.0 for t in all_sig["ticker"].drop_duplicates().tolist()}
            hold_days = {str(t): 0 for t in all_sig["ticker"].drop_duplicates().tolist()}
            for i, d in enumerate(all_dates):
                g = all_sig[all_sig["date"] == d]
                direction_multiplier = self._direction_multiplier_for_date(i, all_dates, all_sig)
                all_sig.loc[g.index, "direction_multiplier"] = float(direction_multiplier)

                sub = g[g["trade_allowed"] & np.isfinite(g["zscore"])].copy()
                target = {str(t): 0.0 for t in g["ticker"].tolist()}
                if not sub.empty:
                    day_rows = {str(row["ticker"]): row for _, row in g.iterrows()}
                    target_raw = self._candidate_weights(sub=sub, direction_multiplier=direction_multiplier)
                    target_scaled = self._apply_risk_scaling(target_weights=target_raw, date_rows=day_rows)
                    for t, w_tar in target_scaled.items():
                        target[t] = float(w_tar)

                final = {}
                for t in g["ticker"].tolist():
                    t = str(t)
                    w_prev = float(prev_w.get(t, 0.0))
                    w_tar = float(target.get(t, 0.0))
                    if abs(w_prev) > 1e-12 and hold_days.get(t, 0) < self.min_hold_days:
                        if abs(w_tar) <= 1e-12:
                            w_fin = w_prev
                        elif np.sign(w_prev) != np.sign(w_tar):
                            w_fin = w_prev
                        else:
                            w_fin = w_tar
                    else:
                        w_fin = w_tar
                    final[t] = float(w_fin)

                gross = float(sum(abs(v) for v in final.values()))
                for idx in g.index.tolist():
                    t = str(all_sig.loc[idx, "ticker"])
                    all_sig.loc[idx, "target_weight_raw"] = float(target.get(t, 0.0))
                    all_sig.loc[idx, "position_cs"] = float(final.get(t, 0.0))
                    all_sig.loc[idx, "gross_leverage"] = gross

                for t in g["ticker"].tolist():
                    t = str(t)
                    w_fin = float(final.get(t, 0.0))
                    if abs(w_fin) > 1e-12:
                        hold_days[t] = (hold_days.get(t, 0) + 1) if abs(prev_w.get(t, 0.0)) > 1e-12 else 1
                    else:
                        hold_days[t] = 0
                    prev_w[t] = w_fin
            all_sig["position"] = all_sig["position_cs"]
            all_sig = all_sig.drop(columns=["position_cs"])
            signal_map = {t: g.drop(columns=["ticker"]).reset_index(drop=True) for t, g in all_sig.groupby("ticker", sort=False)}
        signal_map = {t: self._attach_execution_fields(df) for t, df in signal_map.items()}
        return SignalOutput(signal_map=signal_map)


@dataclass
class RiskReversalSignalGenerator(ResidualZScoreSignalGenerator):
    """Signal generator specialized for risk-reversal trading.

    Builds on residual/z-score logic and emits explicit RR intent fields
    while preserving `position` for downstream backtest compatibility.
    """

    signal_instrument: str = "risk_reversal"

    def _attach_execution_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        out = super()._attach_execution_fields(df)
        if out.empty:
            return out
        out["signal_instrument"] = "risk_reversal"
        rr_dir = np.sign(pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
        out["rr_direction"] = rr_dir  # +1 long RR, -1 short RR
        out["rr_weight"] = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0.0).astype(float)
        return out
