from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import duckdb
import numpy as np
import pandas as pd

from interfaces import DataLoaderModule, DataProcessorModule, LoadedData, ProcessedData, RunSpec

"""
Data loading and processing.

DuckDBOptionsLoader   — queries options_enriched from DuckDB for the requested tickers/dates.
BasicOptionsProcessor — cleans raw quotes and computes log-moneyness k and maturity t:
  - Filters bad IV, zero prices, zero open interest + volume
  - Estimates the forward price F from put-call parity (linear regression of C-P on K)
  - Computes k = ln(K/F),  t = (exdate - date) / 365
"""


@dataclass
class DuckDBOptionsLoader(DataLoaderModule):
    db_path: str
    table: str = "options_enriched"

    def _query_ticker(self, ticker: str, spec: RunSpec) -> pd.DataFrame:
        safe_table = self.table.strip()
        if not safe_table or not safe_table.replace("_", "").isalnum():
            raise ValueError(f"Unsafe table name: {self.table!r}")

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            q = f"""
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
                WHERE UPPER(ticker)=UPPER(?)
                  AND CAST(date AS DATE) >= CAST(? AS DATE)
                  AND CAST(date AS DATE) <= CAST(? AS DATE)
            """
            df = con.execute(q, [ticker, spec.start_date, spec.end_date]).fetchdf()
        finally:
            con.close()
        return df

    def load(self, spec: RunSpec) -> LoadedData:
        raw_by_ticker = {t: self._query_ticker(t, spec) for t in spec.tickers}
        return LoadedData(raw_by_ticker=raw_by_ticker)


@dataclass
class BasicOptionsProcessor(DataProcessorModule):
    min_sigma: float = 0.01
    max_sigma: float = 5.0
    min_t: float = 1.0 / 365.0
    max_t: float = 2.5
    max_abs_k: float = 1.5
    iv_upper_percentile: float | None = None  # e.g. 99.0 drops top 1% of IV per ticker-date

    def _estimate_forward(self, df: pd.DataFrame) -> Dict[tuple[str, str], float]:
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
            if np.isfinite(fwd) and fwd > 0:
                key = (pd.Timestamp(date).strftime("%Y-%m-%d"), pd.Timestamp(exdate).strftime("%Y-%m-%d"))
                out[key] = float(fwd)
        return out

    def _process_one(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["exdate"] = pd.to_datetime(df["exdate"], errors="coerce")
        df["sigma"] = pd.to_numeric(df["impl_volatility"], errors="coerce")
        df["strike"] = pd.to_numeric(df["strike_price"], errors="coerce") / 1000.0  # raw data stores strike in cents
        df["mid"] = 0.5 * (pd.to_numeric(df["best_bid"], errors="coerce") + pd.to_numeric(df["best_offer"], errors="coerce"))
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["date", "exdate", "cp_flag", "ticker", "strike", "sigma", "mid"]
        )
        df = df[(df["sigma"] >= self.min_sigma) & (df["sigma"] <= self.max_sigma)]
        df = df[(df["strike"] > 0.0) & (df["mid"] > 0.0)]
        df = df[df["open_interest"].fillna(0.0) + df["volume"].fillna(0.0) > 0.0]
        if self.iv_upper_percentile is not None:
            cutoff = df.groupby("date")["sigma"].transform(
                lambda s: np.percentile(s, self.iv_upper_percentile)
            )
            df = df[df["sigma"] <= cutoff]

        fwd_map = self._estimate_forward(df)
        if not fwd_map:
            return pd.DataFrame(columns=["date", "ticker", "k", "t", "sigma", "F"])

        keys = list(zip(df["date"].dt.strftime("%Y-%m-%d"), df["exdate"].dt.strftime("%Y-%m-%d")))
        df["F"] = [fwd_map.get(k) for k in keys]
        df = df.dropna(subset=["F"]).copy()

        df["t"] = (df["exdate"] - df["date"]).dt.days / 365.0
        df = df[(df["t"] >= self.min_t) & (df["t"] <= self.max_t)]
        df["k"] = np.log(df["strike"] / df["F"])
        df = df[np.abs(df["k"]) <= self.max_abs_k]
        return df.reset_index(drop=True)

    def process(self, loaded: LoadedData, spec: RunSpec) -> ProcessedData:
        panel_by_ticker = {t: self._process_one(df) for t, df in loaded.raw_by_ticker.items()}
        return ProcessedData(panel_by_ticker=panel_by_ticker)
