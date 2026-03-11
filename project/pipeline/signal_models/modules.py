"""
Signal generation.

ResidualZScoreSignalGenerator:
  For each non-benchmark ticker, fits a rolling OLS of stock_skew on sector_skew (XLF).
  The regression residual is z-scored. Each day, the top-k most positive z (skew rich)
  are shorted and the bottom-k most negative z (skew cheap) are bought — equal weight.
  Positions are held for at least min_hold_days before exiting.

  zscore_method controls how the residual is standardised:
    "gaussian"   — (resid - rolling_mean) / rolling_std  (default)
    "empirical"  — norm.ppf(percentile_rank) within the rolling window; robust to
                   fat tails (ν~3 in empirical skew changes).

  Multi-tenor support:
  When SkewOutput contains a 'tenor_days' column, the generator:
    1. Extracts the signal_tenor_days slice for z-score computation.
    2. Optionally filters event-driven tickers by comparing the short-term skew
       (ts_short_tenor) against the long-term skew (ts_long_tenor). Days where the
       term-structure slope z-score exceeds ts_event_threshold are marked as
       event_day=True and excluded from the trading universe.

RiskReversalSignalGenerator:
  Subclass that tags the signal as "risk_reversal" and adds rr_direction / rr_weight
  fields for the backtest engine.

MomentumSignalGenerator:
  Rides the short-run trend in the OLS residual (empirical H~0.8 in the residual).
  Instead of fading extremes, enter when the N-day change in the residual (momentum)
  is itself extreme. The residual level is preserved in df["residual"]; the momentum
  series and its z-score drive position sizing.
    momentum_window  — look-back in days for the momentum (default 5)
  signal_direction defaults to "momentum" so the position sign matches the trend
  direction (not against it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm as _scipy_norm

from interfaces import RunSpec, SignalGeneratorModule, SignalOutput, SkewOutput


@dataclass
class ResidualZScoreSignalGenerator(SignalGeneratorModule):
    benchmark_preference: str = "XLF"
    regression_window: int = 60
    min_regression_obs: int = 40
    zscore_window: int = 60
    entry_z: float = 1.25
    exit_z: float = 0.30
    min_hold_days: int = 1
    min_liquidity_points: float = 120.0
    max_fit_rmse_iv: float = 0.50
    signal_direction: str = "mean_revert"  # mean_revert | momentum
    cross_section_top_k: int = 2
    cross_section_bottom_k: int = 2
    max_gross_leverage: float = 1.5
    zscore_position_scaling: bool = False
    # "gaussian": (resid - mean) / std;  "empirical": norm.ppf(rank) — fat-tail robust
    zscore_method: str = "gaussian"
    # Used only by MomentumSignalGenerator; stored here so pipeline kwargs pass cleanly.
    momentum_window: int = 5
    signal_instrument: str = "straddle"
    # Which tenor (calendar days) drives the z-score signal and execution.
    # Must be present in the tenor_days list used by GenericSkewCalculator.
    trade_tenor_days: int = 30
    # Event filter: flag days where the term-structure slope (shortest vs longest
    # available tenor) z-score exceeds this threshold.
    ts_event_threshold: float = 2.0
    event_filter_enabled: bool = True

    def _build_signal_inputs(self, skew: SkewOutput) -> Tuple[Dict[str, pd.DataFrame], bool]:
        """Extract trade-tenor DataFrames and attach event flags.

        Returns (signal_dfs, has_multi_tenor). Each value in signal_dfs is a
        DataFrame with one row per date at trade_tenor_days, plus optional
        'event_day', 'ts_slope', 'ts_slope_z' columns.

        The event filter compares the shortest available tenor against the longest
        (min and max of whatever tenors exist in the data), so it works regardless
        of how many tenors are configured.
        """
        has_multi_tenor = any(
            "tenor_days" in df.columns
            for df in skew.skew_by_ticker.values()
            if df is not None and not df.empty
        )
        out: Dict[str, pd.DataFrame] = {}
        for t, df in skew.skew_by_ticker.items():
            if df is None or df.empty:
                continue
            if has_multi_tenor:
                avail = sorted(df["tenor_days"].unique())
                # Select the requested trade tenor, falling back to the closest.
                if self.trade_tenor_days in avail:
                    trade_td = self.trade_tenor_days
                else:
                    trade_td = min(avail, key=lambda x: abs(x - self.trade_tenor_days))
                sig_df = df[df["tenor_days"] == trade_td].copy().reset_index(drop=True)

                if self.event_filter_enabled and len(avail) >= 2:
                    ts_short = avail[0]
                    ts_long = avail[-1]
                    short_df = (
                        df[df["tenor_days"] == ts_short][["date", "skew"]]
                        .rename(columns={"skew": "skew_short"})
                    )
                    long_df = (
                        df[df["tenor_days"] == ts_long][["date", "skew"]]
                        .rename(columns={"skew": "skew_long"})
                    )
                    slope_df = (
                        short_df.merge(long_df, on="date", how="inner")
                        .sort_values("date")
                        .reset_index(drop=True)
                    )
                    slope_df["ts_slope"] = slope_df["skew_short"] - slope_df["skew_long"]
                    roll_mean = slope_df["ts_slope"].rolling(
                        self.regression_window, min_periods=5
                    ).mean()
                    roll_std = (
                        slope_df["ts_slope"]
                        .rolling(self.regression_window, min_periods=5)
                        .std()
                        .replace(0.0, np.nan)
                    )
                    slope_df["ts_slope_z"] = (slope_df["ts_slope"] - roll_mean) / roll_std
                    slope_df["event_day"] = slope_df["ts_slope_z"].gt(self.ts_event_threshold)
                    sig_df = sig_df.merge(
                        slope_df[["date", "ts_slope", "ts_slope_z", "event_day"]],
                        on="date",
                        how="left",
                    )
                    sig_df["event_day"] = sig_df["event_day"].fillna(False)
            else:
                sig_df = df.copy()
            out[t] = sig_df
        return out, has_multi_tenor

    def _compute_residuals(self, stock_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        """Rolling OLS residual and z-score for one ticker vs the benchmark."""
        s = stock_df[
            ["date", "ticker", "skew", "n_points", "rmse_implied_volatility"]
            + ([c for c in ["event_day", "ts_slope", "ts_slope_z"] if c in stock_df.columns])
        ].rename(columns={"skew": "stock_skew"})
        b = benchmark_df[["date", "skew"]].rename(columns={"skew": "sector_skew"})
        df = s.merge(b, on="date", how="inner").sort_values("date").reset_index(drop=True)

        residuals: List[float] = []
        zscores: List[float] = []
        resid_hist: List[float] = []

        for i in range(len(df)):
            row = df.iloc[i]
            hist = df.iloc[max(0, i - self.regression_window):i].dropna(subset=["stock_skew", "sector_skew"])

            resid = np.nan
            z = np.nan
            if len(hist) >= self.min_regression_obs:
                y = hist["stock_skew"].to_numpy(dtype=np.float64)
                x = hist["sector_skew"].to_numpy(dtype=np.float64)
                coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), x]), y, rcond=None)
                y_hat = coef[0] + coef[1] * float(row["sector_skew"])
                if np.isfinite(y_hat) and np.isfinite(row["stock_skew"]):
                    resid = float(row["stock_skew"]) - y_hat
                    resid_hist.append(resid)
                    if len(resid_hist) >= self.zscore_window:
                        tail = np.asarray(resid_hist[-self.zscore_window:], dtype=np.float64)
                        if self.zscore_method == "empirical":
                            rank = float(np.sum(tail[:-1] < resid)) / max(len(tail) - 1, 1)
                            rank = float(np.clip(rank, 0.01, 0.99))
                            z = float(_scipy_norm.ppf(rank))
                        else:
                            sd = float(np.std(tail))
                            if sd > 1e-12:
                                z = (resid - float(np.mean(tail))) / sd

            residuals.append(resid)
            zscores.append(z)

        df["residual"] = residuals
        df["zscore"] = zscores
        event_ok = ~df["event_day"] if "event_day" in df.columns else pd.Series(True, index=df.index)
        df["quality_pass"] = (
            df["n_points"].ge(self.min_liquidity_points)
            & df["rmse_implied_volatility"].le(self.max_fit_rmse_iv)
            & event_ok
        )
        return df

    def _attach_execution_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out["signal_instrument"] = self.signal_instrument
        out["target_weight"] = pd.to_numeric(out["position"], errors="coerce").fillna(0.0).astype(float)
        out["execution_enabled"] = out["target_weight"].abs() > 0.0
        return out

    def generate(self, skew: SkewOutput, spec: RunSpec) -> SignalOutput:
        signal_dfs, _ = self._build_signal_inputs(skew)
        tickers = list(signal_dfs.keys())
        pref = self.benchmark_preference.upper()
        benchmark = next((t for t in tickers if t.upper() == pref), tickers[0])
        benchmark_df = signal_dfs[benchmark]

        residual_map: Dict[str, pd.DataFrame] = {
            t: self._compute_residuals(df, benchmark_df)
            for t, df in signal_dfs.items()
            if t != benchmark and df is not None and not df.empty
        }
        if not residual_map:
            return SignalOutput(signal_map={})

        all_sig = (
            pd.concat([df.assign(ticker=t) for t, df in residual_map.items()], ignore_index=True)
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )
        all_sig["position"] = 0.0

        direction = 1.0 if self.signal_direction == "mean_revert" else -1.0

        prev_w: Dict[str, float] = {t: 0.0 for t in residual_map}
        hold_days: Dict[str, int] = {t: 0 for t in residual_map}

        for d in sorted(all_sig["date"].unique()):
            g = all_sig[all_sig["date"] == d]
            eligible = g[g["quality_pass"] & g["zscore"].notna()].copy()

            target: Dict[str, float] = {}
            if not eligible.empty:
                z_eff = direction * eligible["zscore"].to_numpy(dtype=np.float64)
                tickers_arr = eligible["ticker"].tolist()

                long_idx = np.where(z_eff <= -self.entry_z)[0]
                short_idx = np.where(z_eff >= self.entry_z)[0]

                long_idx = long_idx[np.argsort(z_eff[long_idx])][: self.cross_section_bottom_k]
                short_idx = short_idx[np.argsort(-z_eff[short_idx])][: self.cross_section_top_k]

                n = len(long_idx) + len(short_idx)
                if n > 0:
                    if self.zscore_position_scaling:
                        z_long = np.abs(z_eff[long_idx])
                        z_short = np.abs(z_eff[short_idx])
                        total_z = z_long.sum() + z_short.sum()
                        for i, idx in enumerate(long_idx):
                            target[tickers_arr[idx]] = +z_long[i] / total_z * self.max_gross_leverage
                        for i, idx in enumerate(short_idx):
                            target[tickers_arr[idx]] = -z_short[i] / total_z * self.max_gross_leverage
                    else:
                        w = self.max_gross_leverage / n
                        for idx in long_idx:
                            target[tickers_arr[idx]] = +w
                        for idx in short_idx:
                            target[tickers_arr[idx]] = -w

            # Build a fast z_eff lookup for the hold/exit decision below.
            z_eff_by_ticker: Dict[str, float] = {}
            if not eligible.empty:
                for _t, _row in eligible.set_index("ticker")["zscore"].items():
                    z_eff_by_ticker[str(_t)] = direction * float(_row)

            final: Dict[str, float] = {}
            for t in residual_map:
                w_prev = prev_w[t]
                w_tar = target.get(t, 0.0)

                # Hold an open position while the z-score hasn't reverted past exit_z.
                # Without this, any day where |z| < entry_z would immediately close
                # the position (the ticker would simply be absent from `target`).
                if abs(w_prev) > 1e-10 and abs(w_tar) < 1e-10:
                    z_eff_t = z_eff_by_ticker.get(t, None)
                    if z_eff_t is not None:
                        # Maintain long while z_eff is still sufficiently negative.
                        # Maintain short while z_eff is still sufficiently positive.
                        if (np.sign(w_prev) > 0 and z_eff_t <= -self.exit_z) or \
                           (np.sign(w_prev) < 0 and z_eff_t >= +self.exit_z):
                            w_tar = w_prev  # hold
                        # else: z has reverted past exit_z → exit (w_tar stays 0)
                    # If ticker not in eligible (quality fail today), exit.

                # Min-hold-days override: never exit before min_hold_days regardless of z.
                if abs(w_prev) > 1e-10 and hold_days[t] < self.min_hold_days:
                    if abs(w_tar) < 1e-10 or np.sign(w_prev) != np.sign(w_tar):
                        w_tar = w_prev

                final[t] = w_tar

            for idx in g.index:
                t = str(all_sig.loc[idx, "ticker"])
                all_sig.loc[idx, "position"] = final.get(t, 0.0)

            for t in residual_map:
                w_fin = final[t]
                if abs(w_fin) > 1e-10:
                    hold_days[t] = hold_days[t] + 1 if abs(prev_w[t]) > 1e-10 else 1
                else:
                    hold_days[t] = 0
                prev_w[t] = w_fin

        signal_map = {
            t: self._attach_execution_fields(g.drop(columns=["ticker"]).reset_index(drop=True))
            for t, g in all_sig.groupby("ticker", sort=False)
        }
        return SignalOutput(signal_map=signal_map)


@dataclass
class MomentumSignalGenerator(ResidualZScoreSignalGenerator):
    """Momentum signal on the OLS residual.

    Uses the empirical H~0.8 finding: the OLS residual has short-run positive
    autocorrelation before it mean-reverts.  Instead of fading extremes, we enter
    in the direction of the recent N-day change in the residual.

    df["residual"] is preserved as the OLS level residual for analytics.
    df["momentum"] stores the raw N-day change.
    df["zscore"]   is the z-score of the momentum series (drives all entry/exit logic).

    signal_direction="momentum" flips the entry sign so that:
      z > +entry_z  →  long  (residual trending up  → skew becoming less negative)
      z < -entry_z  →  short (residual trending down → skew becoming more negative)
    """

    momentum_window: int = 5
    signal_direction: str = "momentum"
    min_hold_days: int = 2
    signal_instrument: str = "risk_reversal"

    def _compute_residuals(self, stock_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        # Compute the OLS residual level first (reuse parent logic).
        df = super()._compute_residuals(stock_df, benchmark_df)

        resid = df["residual"].to_numpy(dtype=np.float64)
        mom_vals = np.full(len(resid), np.nan)
        mom_z_vals = np.full(len(resid), np.nan)
        mom_hist: List[float] = []

        for i in range(len(resid)):
            j = i - self.momentum_window
            if j >= 0 and np.isfinite(resid[i]) and np.isfinite(resid[j]):
                m = float(resid[i] - resid[j])
                mom_vals[i] = m
                mom_hist.append(m)
                if len(mom_hist) >= self.zscore_window:
                    tail = np.asarray(mom_hist[-self.zscore_window:], dtype=np.float64)
                    if self.zscore_method == "empirical":
                        rank = float(np.sum(tail[:-1] < m)) / max(len(tail) - 1, 1)
                        rank = float(np.clip(rank, 0.01, 0.99))
                        mom_z_vals[i] = float(_scipy_norm.ppf(rank))
                    else:
                        sd = float(np.std(tail))
                        if sd > 1e-12:
                            mom_z_vals[i] = (m - float(np.mean(tail))) / sd

        df["momentum"] = mom_vals
        # Override zscore with momentum z-score; residual level is preserved for analytics.
        df["zscore"] = mom_z_vals
        return df


@dataclass
class RiskReversalSignalGenerator(ResidualZScoreSignalGenerator):
    signal_instrument: str = "risk_reversal"

    def _attach_execution_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        out = super()._attach_execution_fields(df)
        if out.empty:
            return out
        out["signal_instrument"] = "risk_reversal"
        pos = pd.to_numeric(out["position"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        out["rr_direction"] = np.sign(pos)
        out["rr_weight"] = out["target_weight"]
        return out
