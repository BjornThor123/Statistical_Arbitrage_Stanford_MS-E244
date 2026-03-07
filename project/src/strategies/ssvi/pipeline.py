from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.errors import ParserError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.ssvi.model import CalibratedSSVISurface, calibrate_ssvi_surface


@dataclass
class PanelFilters:
    min_sigma: float = 0.01
    max_sigma: float = 5.0
    min_t: float = 1.0 / 365.0
    max_t: float = 2.5
    max_abs_k: float = 1.5
    min_points_per_day: int = 120
    min_maturities_per_day: int = 3


def load_options_history(
    zip_path: str,
    start_date: str,
    end_date: str,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    cols = [
        "date",
        "exdate",
        "cp_flag",
        "strike_price",
        "impl_volatility",
        "best_bid",
        "best_offer",
        "open_interest",
        "volume",
        "ticker",
    ]
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    parts: List[pd.DataFrame] = []

    def _read_chunks(engine: str | None) -> None:
        kw = {"compression": "zip", "usecols": cols, "chunksize": chunksize}
        if engine is not None:
            kw["engine"] = engine
        for chunk in pd.read_csv(zip_path, **kw):
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            mask = (chunk["date"] >= start) & (chunk["date"] <= end)
            sub = chunk.loc[mask]
            if not sub.empty:
                parts.append(sub.copy())

    try:
        _read_chunks(engine=None)
    except (ParserError, OSError, ValueError):
        parts.clear()
        _read_chunks(engine="python")

    if not parts:
        raise ValueError(f"No option rows found in range [{start_date}, {end_date}] for {zip_path}")
    return pd.concat(parts, ignore_index=True)


def load_options_history_duckdb(
    db_path: str,
    ticker: str,
    start_date: str,
    end_date: str,
    table: str = "options_enriched",
) -> pd.DataFrame:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "duckdb is required for DuckDB-backed loading. Install with `pip install duckdb`."
        ) from exc

    safe_table = table.strip()
    if not safe_table or not safe_table.replace("_", "").isalnum():
        raise ValueError(f"Unsafe DuckDB table name: {table!r}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        query = f"""
            SELECT
                date,
                exdate,
                cp_flag,
                strike_price,
                impl_volatility,
                best_bid,
                best_offer,
                open_interest,
                volume,
                ticker
            FROM {safe_table}
            WHERE UPPER(ticker) = UPPER(?)
              AND CAST(date AS DATE) >= CAST(? AS DATE)
              AND CAST(date AS DATE) <= CAST(? AS DATE)
        """
        df = con.execute(query, [ticker, start_date, end_date]).fetchdf()
    finally:
        con.close()

    if df.empty:
        raise ValueError(
            f"No option rows found in range [{start_date}, {end_date}] "
            f"for ticker={ticker!r} in table={safe_table!r}"
        )
    return df


def _estimate_forward_per_date_exdate(df: pd.DataFrame) -> Dict[tuple[str, str], float]:
    out: Dict[tuple[str, str], float] = {}
    calls = df[df["cp_flag"] == "C"][["date", "exdate", "strike", "mid"]]
    puts = df[df["cp_flag"] == "P"][["date", "exdate", "strike", "mid"]]
    joined = calls.merge(puts, on=["date", "exdate", "strike"], suffixes=("_c", "_p"))
    for (date, exdate), grp in joined.groupby(["date", "exdate"]):
        if len(grp) < 5:
            continue
        x = grp["strike"].to_numpy(dtype=np.float64)
        y = (grp["mid_c"] - grp["mid_p"]).to_numpy(dtype=np.float64)
        if np.unique(x).shape[0] < 4:
            continue
        slope, intercept = np.polyfit(x, y, deg=1)
        if not np.isfinite(slope) or not np.isfinite(intercept) or slope >= -1e-10:
            continue
        fwd = -intercept / slope
        if np.isfinite(fwd) and fwd > 0.0:
            out[(pd.Timestamp(date).strftime("%Y-%m-%d"), pd.Timestamp(exdate).strftime("%Y-%m-%d"))] = float(fwd)
    return out


def preprocess_options_panel(raw: pd.DataFrame, filters: Optional[PanelFilters] = None) -> pd.DataFrame:
    cfg = filters or PanelFilters()
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["exdate"] = pd.to_datetime(df["exdate"], errors="coerce")
    df["sigma"] = pd.to_numeric(df["impl_volatility"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike_price"], errors="coerce") / 1000.0
    bid = pd.to_numeric(df["best_bid"], errors="coerce")
    ask = pd.to_numeric(df["best_offer"], errors="coerce")
    df["mid"] = 0.5 * (bid + ask)
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["date", "exdate", "cp_flag", "strike", "sigma", "mid", "ticker"]
    )
    df = df[(df["sigma"] > cfg.min_sigma) & (df["sigma"] < cfg.max_sigma)]
    df = df[(df["strike"] > 0.0) & (df["mid"] > 0.0)]
    df = df[df["open_interest"].fillna(0.0) + df["volume"].fillna(0.0) > 0.0]

    forward_map = _estimate_forward_per_date_exdate(df)
    if not forward_map:
        raise RuntimeError("Forward estimation failed for all (date, exdate) groups.")

    key = list(zip(df["date"].dt.strftime("%Y-%m-%d"), df["exdate"].dt.strftime("%Y-%m-%d")))
    df["F"] = [forward_map.get(k) for k in key]
    df = df.dropna(subset=["F"]).copy()

    df["t"] = (df["exdate"] - df["date"]).dt.days / 365.0
    df = df[(df["t"] >= cfg.min_t) & (df["t"] <= cfg.max_t)]
    df["k"] = np.log(df["strike"] / df["F"])
    df = df[np.abs(df["k"]) <= cfg.max_abs_k].copy()
    return df.reset_index(drop=True)


def fit_daily_ssvi_skew(
    panel: pd.DataFrame,
    target_t: float = 30.0 / 365.0,
    k0: float = 0.0,
    min_points_per_day: int = 120,
    min_maturities_per_day: int = 3,
    calibration_backend: str = "mps",
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    prev_params = None
    prev_theta = None
    first_exc: Exception | None = None
    n_exc = 0
    for d, day in panel.groupby("date"):
        if len(day) < min_points_per_day:
            continue
        if day["t"].round(8).nunique() < min_maturities_per_day:
            continue

        k = day["k"].to_numpy(dtype=np.float64)
        t = day["t"].to_numpy(dtype=np.float64)
        sigma = day["sigma"].to_numpy(dtype=np.float64)

        # Robust cross-section trim to reduce bad quotes impact.
        med = float(np.median(sigma))
        mad = float(np.median(np.abs(sigma - med)))
        if mad > 1e-8:
            lo = med - 5.0 * 1.4826 * mad
            hi = med + 5.0 * 1.4826 * mad
            keep = (sigma >= max(0.01, lo)) & (sigma <= min(5.0, hi))
            if np.sum(keep) >= min_points_per_day:
                k, t, sigma = k[keep], t[keep], sigma[keep]

        try:
            surface: CalibratedSSVISurface = calibrate_ssvi_surface(
                k=k,
                t=t,
                sigma=sigma,
                random_seed=42,
                n_param_steps=300,
                n_theta_steps=3,
                n_restarts=4,
                initial_params=prev_params,
                initial_theta_nodes=prev_theta,
                theta_smoothness_lambda=2e-3,
                calibration_backend=calibration_backend,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "backend requested but not available" in msg or "pytorch is not installed" in msg:
                raise RuntimeError(
                    f"Calibration backend '{calibration_backend}' unavailable: {exc}. "
                    "Use --calibration-backend auto or cpu on this machine."
                ) from exc
            if first_exc is None:
                first_exc = exc
            n_exc += 1
            continue

        prev_params = (surface.params.rho, surface.params.eta, surface.params.gamma)
        prev_theta = surface.theta_nodes.copy()
        skew = surface.sigma_skew(t=target_t, k0=k0)
        theta_target = float(surface.theta_at(np.asarray([target_t]))[0])
        atm_iv = float(np.sqrt(max(theta_target, 1e-12) / max(target_t, 1e-12)))
        day_fwd = day[["t", "F"]].drop_duplicates(subset=["t"])
        closest_idx = (day_fwd["t"] - target_t).abs().idxmin()
        forward_price = float(day_fwd.loc[closest_idx, "F"])
        ticker = str(day["ticker"].iloc[0])
        rows.append(
            {
                "date": pd.Timestamp(d),
                "ticker": ticker,
                "target_t": float(target_t),
                "k0": float(k0),
                "skew": float(skew),
                "rho": float(surface.params.rho),
                "eta": float(surface.params.eta),
                "gamma": float(surface.params.gamma),
                "theta_target": theta_target,
                "atm_iv": atm_iv,
                "forward_price": forward_price,
                "rmse_total_variance": float(surface.diagnostics.rmse_total_variance),
                "rmse_implied_volatility": float(surface.diagnostics.rmse_implied_volatility),
                "no_butterfly_constraint_score": float(surface.diagnostics.no_butterfly_constraint_score),
                "n_points": int(surface.diagnostics.n_points),
                "n_maturities": int(surface.diagnostics.n_maturities),
                "theta_min": float(surface.theta_nodes.min()),
                "theta_max": float(surface.theta_nodes.max()),
            }
        )

    if not rows:
        if first_exc is not None:
            raise RuntimeError(
                f"No daily SSVI fits succeeded (failed days: {n_exc}). "
                f"First calibration error: {type(first_exc).__name__}: {first_exc}"
            ) from first_exc
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "target_t",
                "k0",
                "skew",
                "rho",
                "eta",
                "gamma",
                "theta_target",
                "atm_iv",
                "forward_price",
                "rmse_total_variance",
                "rmse_implied_volatility",
                "no_butterfly_constraint_score",
                "n_points",
                "n_maturities",
                "theta_min",
                "theta_max",
            ]
        )
    out = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def build_skew_series_from_zip(
    zip_path: str,
    start_date: str,
    end_date: str,
    target_t: float = 30.0 / 365.0,
    k0: float = 0.0,
    filters: Optional[PanelFilters] = None,
    calibration_backend: str = "mps",
) -> pd.DataFrame:
    raw = load_options_history(zip_path=zip_path, start_date=start_date, end_date=end_date)
    panel = preprocess_options_panel(raw, filters=filters)
    return fit_daily_ssvi_skew(
        panel=panel,
        target_t=target_t,
        k0=k0,
        min_points_per_day=(filters.min_points_per_day if filters else 120),
        min_maturities_per_day=(filters.min_maturities_per_day if filters else 3),
        calibration_backend=calibration_backend,
    )


def build_skew_series_from_duckdb(
    db_path: str,
    ticker: str,
    start_date: str,
    end_date: str,
    target_t: float = 30.0 / 365.0,
    k0: float = 0.0,
    filters: Optional[PanelFilters] = None,
    calibration_backend: str = "mps",
    table: str = "options_enriched",
) -> pd.DataFrame:
    raw = load_options_history_duckdb(
        db_path=db_path,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        table=table,
    )
    panel = preprocess_options_panel(raw, filters=filters)
    return fit_daily_ssvi_skew(
        panel=panel,
        target_t=target_t,
        k0=k0,
        min_points_per_day=(filters.min_points_per_day if filters else 120),
        min_maturities_per_day=(filters.min_maturities_per_day if filters else 3),
        calibration_backend=calibration_backend,
    )


def infer_ticker_from_zip_path(zip_path: str) -> str:
    name = Path(zip_path).stem
    if name.startswith("options_"):
        tail = name[len("options_") :]
        if tail.endswith("_etf"):
            tail = tail[: -len("_etf")]
        return tail.upper()
    return name.upper()
