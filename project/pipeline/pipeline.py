"""
Main pipeline orchestrator.

Quickstart
----------
from pipeline import Pipeline, MasterRunSpec, DataSpec

cfg = MasterRunSpec(
    data=DataSpec(
        tickers=["XLF", "GS", "JPM", "BAC", "C"],
        start_date="2008-01-01",
        end_date="2008-12-31",
    ),
)
p = Pipeline(cfg)

# Run individual stages (results can be cached to disk with saved='filename.parquet'):
loaded    = p.run_load()
processed = p.run_process(loaded)
modeled   = p.run_model(processed)
skew      = p.run_skew(modeled)
signals   = p.run_signals(skew)
backtest  = p.run_backtest(signals, skew)

print(backtest.summary)

# Alternatively, run everything in one call:
output = p.run()
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Literal

import pandas as pd
from config import DATA_LOCATION
from factory import build_default_smile_pipeline, build_default_surface_pipeline
from interfaces import (
    BacktestOutput,
    LoadedData,
    ModelOutput,
    PipelineOutput,
    ProcessedData,
    RunSpec,
    SignalOutput,
    SkewOutput,
    StrategyPipeline,
    WalkForwardOutput,
)


@dataclass
class DataSpec:
    tickers: List[str]
    start_date: str
    end_date: str
    db_table: str = "options_enriched"
    iv_upper_percentile: float | None = None  # e.g. 99.0 drops top 1% of IV per date


@dataclass
class VolatilityModelSpec:
    model_kind: Literal["surface", "smile"] = "surface"
    ssvi_n_theta_steps: int = 3
    ssvi_n_restarts: int = 4


@dataclass
class SkewSpec:
    technique: str = "local_derivative"
    delta_k: float = 0.01
    # Single source of truth for risk-reversal wing width.
    # Used by GenericSkewCalculator (rr_k) AND DeltaHedgedOptionBacktestEngine
    # (rr_put_log_moneyness = -rr_k, rr_call_log_moneyness = +rr_k).
    rr_k: float = 0.15
    # When rr_delta > 0, skew is evaluated at the log-moneyness corresponding to
    # this BS delta (e.g. 0.25 = 25-delta RR). Set to 0.0 to use fixed rr_k instead.
    rr_delta: float = 0.25
    # Maturities (in calendar days) at which skew is evaluated each day.
    # All downstream consumers (signal, backtest) select the tenor they need.
    tenor_days: List[int] = field(default_factory=lambda: [30, 60, 90])


@dataclass
class SignalSpec:
    model_kind: str = "risk_reversal"  # risk_reversal | residual | momentum
    regression_window: int = 20
    min_regression_obs: int = 20
    zscore_window: int = 20
    entry_z: float = 3
    exit_z: float = 0.30
    min_hold_days: int = 1
    min_liquidity_points: float = 20
    max_fit_rmse_iv: float = 0.50
    direction: str = "mean_revert"  # mean_revert | momentum
    cross_section_top_k: int = 2
    cross_section_bottom_k: int = 2
    max_gross_leverage: float = 1.5
    benchmark_preference: str = "XLF"
    # Which tenor (calendar days) drives the z-score signal and execution.
    # Must be one of the values in model.tenor_days.
    trade_tenor_days: int = 30
    # Event filter: flag days where the term-structure slope (shortest vs longest
    # tenor in model.tenor_days) z-score exceeds this threshold.
    ts_event_threshold: float = 2.0
    event_filter_enabled: bool = True
    # Z-score method: "gaussian" = (x-mean)/std; "empirical" = norm.ppf(rank) — fat-tail robust.
    zscore_method: str = "gaussian"
    # Momentum look-back in calendar days (MomentumSignalGenerator only).
    momentum_window: int = 5


@dataclass
class BacktestSpec:
    stock_fee_bps: float = 1.0
    option_fee_per_contract: float = 0.65
    sizing_mode: str = "contracts"  # contracts | dollar_vega
    max_contracts_per_signal: float = 10.0
    target_dollar_vega_per_signal: float = 20000.0
    roll_threshold_days: int = 5


@dataclass
class MasterRunSpec:
    data: DataSpec
    model: VolatilityModelSpec = field(default_factory=VolatilityModelSpec)
    skew: SkewSpec = field(default_factory=SkewSpec)
    signal: SignalSpec = field(default_factory=SignalSpec)
    backtest: BacktestSpec = field(default_factory=BacktestSpec)


class Pipeline:
    def __init__(self, config: MasterRunSpec):
        self.config = config
        self.db_path = f"{DATA_LOCATION}/market_data.duckdb"
        self._pipeline = self._build_pipeline()
        self._spec = RunSpec(
            tickers=self.config.data.tickers,
            start_date=self.config.data.start_date,
            end_date=self.config.data.end_date,
        )
        self._stage_save_root = Path(__file__).resolve().parent / "stage_saves"
        self._stage_save_root.mkdir(parents=True, exist_ok=True)

    def _cache_id(self) -> str:
        payload = {
            **asdict(self.config),
            "target_days": self._spec.target_days,
            "k0": self._spec.k0,
        }
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    def _default_stage_path(self, stage: str) -> Path:
        folder_map = {
            "load": "data",
            "process": "data_processing",
            "model": "volatility_models",
            "skew": "skew_models",
            "signals": "signal_models",
            "backtest": "backtest_models",
        }
        folder = folder_map.get(stage, stage)
        stage_dir = self._stage_save_root / folder
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir / f"{stage}_{self._cache_id()}.parquet"

    @staticmethod
    def _meta_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".meta.json")

    @staticmethod
    def _save_dict_frames(path: Path, frame_map: Dict[str, pd.DataFrame], extra_meta: Dict[str, object] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        chunks: List[pd.DataFrame] = []
        for key, df in frame_map.items():
            chunk = df.copy()
            chunk["__stage_key"] = key
            chunks.append(chunk)
        combined = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["__stage_key"])
        combined.to_parquet(path, index=False)
        meta = {"keys": list(frame_map.keys())}
        if extra_meta:
            meta.update(extra_meta)
        Pipeline._meta_path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def _load_dict_frames(path: Path) -> tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
        combined = pd.read_parquet(path)
        meta_path = Pipeline._meta_path(path)
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out: Dict[str, pd.DataFrame] = {}
        if "__stage_key" in combined.columns:
            for key, sub in combined.groupby("__stage_key", sort=False):
                out[str(key)] = sub.drop(columns=["__stage_key"]).reset_index(drop=True)
        for key in meta.get("keys", []):
            if key not in out:
                out[str(key)] = pd.DataFrame()
        return out, meta

    def _resolve_save_path(self, stage: str, saved: str | None) -> Path:
        if saved is None:
            return self._default_stage_path(stage)
        p = Path(saved).expanduser()
        if p.is_absolute():
            return p
        # Bare filename: place it in this stage's default cache directory.
        if p.parent == Path("."):
            return self._default_stage_path(stage).parent / p.name
        # Relative path with folders: respect as provided.
        return p

    def _build_pipeline(self) -> StrategyPipeline:
        surface_kwargs = {
            "n_theta_steps": self.config.model.ssvi_n_theta_steps,
            "n_restarts": self.config.model.ssvi_n_restarts,
        }
        # rr_k is the single source of truth: the skew calculator and backtest
        # both use the same wing width so signal and execution are aligned.
        skew_kwargs = {
            "technique": self.config.skew.technique,
            "delta_k": self.config.skew.delta_k,
            "rr_k": self.config.skew.rr_k,
            "rr_delta": self.config.skew.rr_delta,
            "tenor_days": self.config.skew.tenor_days,
        }
        signal_kwargs = {
            "regression_window": self.config.signal.regression_window,
            "min_regression_obs": self.config.signal.min_regression_obs,
            "zscore_window": self.config.signal.zscore_window,
            "entry_z": self.config.signal.entry_z,
            "exit_z": self.config.signal.exit_z,
            "min_hold_days": self.config.signal.min_hold_days,
            "min_liquidity_points": self.config.signal.min_liquidity_points,
            "max_fit_rmse_iv": self.config.signal.max_fit_rmse_iv,
            "signal_direction": self.config.signal.direction,
            "cross_section_top_k": self.config.signal.cross_section_top_k,
            "cross_section_bottom_k": self.config.signal.cross_section_bottom_k,
            "max_gross_leverage": self.config.signal.max_gross_leverage,
            "benchmark_preference": self.config.signal.benchmark_preference,
            "trade_tenor_days": self.config.signal.trade_tenor_days,
            "ts_event_threshold": self.config.signal.ts_event_threshold,
            "event_filter_enabled": self.config.signal.event_filter_enabled,
            "zscore_method": self.config.signal.zscore_method,
            "momentum_window": self.config.signal.momentum_window,
        }
        backtest_kwargs = {
            "stock_fee_bps": self.config.backtest.stock_fee_bps,
            "option_fee_per_contract": self.config.backtest.option_fee_per_contract,
            "sizing_mode": self.config.backtest.sizing_mode,
            "max_contracts_per_signal": self.config.backtest.max_contracts_per_signal,
            "target_dollar_vega_per_signal": self.config.backtest.target_dollar_vega_per_signal,
            "roll_threshold_days": self.config.backtest.roll_threshold_days,
            # trade_tenor_days drives contract selection in the backtest engine.
            "target_days": self.config.signal.trade_tenor_days,
            # Delta targeting takes precedence; rr_k is the log-moneyness fallback.
            "rr_put_delta": self.config.skew.rr_delta,
            "rr_call_delta": self.config.skew.rr_delta,
            "rr_put_log_moneyness": -self.config.skew.rr_k,
            "rr_call_log_moneyness": +self.config.skew.rr_k,
        }
        processor_kwargs = {
            "iv_upper_percentile": self.config.data.iv_upper_percentile,
        }
        if self.config.model.model_kind == "surface":
            return build_default_surface_pipeline(
                db_path=self.db_path,
                table=self.config.data.db_table,
                processor_kwargs=processor_kwargs,
                surface_kwargs=surface_kwargs,
                skew_kwargs=skew_kwargs,
                signal_kwargs=signal_kwargs,
                backtest_kwargs=backtest_kwargs,
                signal_kind=self.config.signal.model_kind,
            )
        if self.config.model.model_kind == "smile":
            return build_default_smile_pipeline(
                db_path=self.db_path,
                table=self.config.data.db_table,
                processor_kwargs=processor_kwargs,
                skew_kwargs=skew_kwargs,
                signal_kwargs=signal_kwargs,
                backtest_kwargs=backtest_kwargs,
                signal_kind=self.config.signal.model_kind,
            )
        raise ValueError("model.model_kind must be one of: surface, smile")

    @staticmethod
    def _apply_overrides(config: MasterRunSpec, overrides: Dict[str, object]) -> MasterRunSpec:
        """Apply dot-notation overrides (e.g. 'signal.entry_z') to a MasterRunSpec."""
        data_updates: Dict[str, object] = {}
        model_updates: Dict[str, object] = {}
        skew_updates: Dict[str, object] = {}
        signal_updates: Dict[str, object] = {}
        backtest_updates: Dict[str, object] = {}
        section_map = {
            "data": data_updates,
            "model": model_updates,
            "skew": skew_updates,
            "signal": signal_updates,
            "backtest": backtest_updates,
        }
        for key, val in overrides.items():
            if "." not in key:
                raise ValueError(
                    f"Walk-forward override key '{key}' must use dot notation "
                    "(e.g. 'signal.entry_z', 'model.rr_k')."
                )
            section, field_name = key.split(".", 1)
            if section not in section_map:
                raise ValueError(
                    f"Unknown config section '{section}' in override key '{key}'. "
                    "Valid sections: data, model, signal, backtest."
                )
            section_map[section][field_name] = val
        return MasterRunSpec(
            data=replace(config.data, **data_updates) if data_updates else config.data,
            model=replace(config.model, **model_updates) if model_updates else config.model,
            skew=replace(config.skew, **skew_updates) if skew_updates else config.skew,
            signal=replace(config.signal, **signal_updates) if signal_updates else config.signal,
            backtest=replace(config.backtest, **backtest_updates) if backtest_updates else config.backtest,
        )

    @staticmethod
    def _slice_df_by_dates(df: pd.DataFrame, dates: List[pd.Timestamp]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=df.columns if df is not None else None)
        dset = {pd.Timestamp(d) for d in dates}
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        return out[out["date"].isin(dset)].sort_values("date").reset_index(drop=True)

    @staticmethod
    def _slice_skew_output(skew: SkewOutput, dates: List[pd.Timestamp]) -> SkewOutput:
        return SkewOutput(
            skew_by_ticker={
                t: Pipeline._slice_df_by_dates(df, dates)
                for t, df in skew.skew_by_ticker.items()
            }
        )

    @staticmethod
    def _walk_forward_windows(
        all_dates: List[pd.Timestamp],
        train_days: int,
        test_days: int,
        step_days: int,
    ) -> List[tuple[List[pd.Timestamp], List[pd.Timestamp]]]:
        windows = []
        i = 0
        n = len(all_dates)
        while i + train_days + test_days <= n:
            train = all_dates[i : i + train_days]
            test = all_dates[i + train_days : i + train_days + test_days]
            windows.append((train, test))
            i += step_days
        return windows

    @staticmethod
    def _max_drawdown(cum: pd.Series) -> float:
        if cum.empty:
            return 0.0
        peak = cum.cummax()
        dd = cum - peak
        return float(dd.min())

    @staticmethod
    def _score_portfolio(
        portfolio: pd.DataFrame,
        regression_window: int,
        zscore_window: int,
    ) -> Dict[str, float]:
        if portfolio is None or portfolio.empty:
            return {
                "n_days": 0.0,
                "avg_daily_net_pnl": 0.0,
                "std_daily_net_pnl": 0.0,
                "daily_sharpe_like": 0.0,
                "final_cum_net_pnl": 0.0,
                "max_drawdown": 0.0,
                "hit_rate": 0.0,
                "score": -1e9,
            }
        p = portfolio.copy()
        p["date"] = pd.to_datetime(p["date"])
        p = p.sort_values("date").reset_index(drop=True)
        if "portfolio_cum_net_pnl" not in p.columns:
            p["portfolio_cum_net_pnl"] = p["portfolio_net_pnl"].cumsum()
        r = p["portfolio_net_pnl"].to_numpy(dtype=float)
        mu = float(r.mean()) if len(r) else 0.0
        sd = float(r.std()) if len(r) else 0.0
        sharpe = float(mu / sd * (252.0**0.5)) if sd > 1e-12 else 0.0
        final = float(p["portfolio_cum_net_pnl"].iloc[-1]) if len(p) else 0.0
        mdd = Pipeline._max_drawdown(p["portfolio_cum_net_pnl"])
        hit = float((r > 0.0).mean()) if len(r) else 0.0
        score = sharpe + 0.10 * final + 0.50 * hit - 0.50 * abs(mdd) - 0.25 * sd
        if len(p) < max(60, int(regression_window + zscore_window)):
            score -= 3.0
        return {
            "n_days": float(len(p)),
            "avg_daily_net_pnl": mu,
            "std_daily_net_pnl": sd,
            "daily_sharpe_like": sharpe,
            "final_cum_net_pnl": final,
            "max_drawdown": mdd,
            "hit_rate": hit,
            "score": float(score),
        }

    def _default_walk_forward_candidates(self) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        for entry_z, exit_z in [(1.00, 0.25), (1.25, 0.30), (1.50, 0.50)]:
            for rr_k in [0.10, 0.15]:
                for leverage in [1.0, 1.5]:
                    candidates.append({
                        "signal.entry_z": entry_z,
                        "signal.exit_z": exit_z,
                        "signal.max_gross_leverage": leverage,
                        "model.rr_k": rr_k,
                    })
        return candidates

    def run(self):
        return self._pipeline.run(self._spec)

    def run_load(self, saved: str | None = None, force_recompute: bool = False) -> LoadedData:
        path = self._resolve_save_path("load", saved) if saved is not None else None
        if path is not None and path.exists() and not force_recompute:
            raw_by_ticker, _ = self._load_dict_frames(path)
            return LoadedData(raw_by_ticker=raw_by_ticker)
        out = self._pipeline.run_load(self._spec)
        if saved is not None:
            self._save_dict_frames(path, out.raw_by_ticker)
        return out

    def run_process(
        self,
        loaded: LoadedData | None = None,
        saved: str | None = None,
        force_recompute: bool = False,
    ) -> ProcessedData:
        path = self._resolve_save_path("process", saved) if saved is not None else None
        if path is not None and path.exists() and not force_recompute:
            panel_by_ticker, _ = self._load_dict_frames(path)
            return ProcessedData(panel_by_ticker=panel_by_ticker)
        if loaded is None:
            raise ValueError("run_process requires `loaded` unless `saved` points to an existing parquet.")
        out = self._pipeline.run_process(loaded, self._spec)
        if saved is not None:
            self._save_dict_frames(path, out.panel_by_ticker)
        return out

    def run_model(
        self,
        processed: ProcessedData | None = None,
        saved: str | None = None,
        force_recompute: bool = False,
    ) -> ModelOutput:
        path = self._resolve_save_path("model", saved) if saved is not None else None
        if path is not None and path.exists() and not force_recompute:
            model_by_ticker, meta = self._load_dict_frames(path)
            if "representation" not in meta:
                raise ValueError(
                    "Saved model artifact is missing required metadata field 'representation'. "
                    "Re-run run_model(..., saved='filename.parquet') with the current code."
                )
            representation = str(meta["representation"])
            return ModelOutput(model_by_ticker=model_by_ticker, representation=representation)
        if processed is None:
            raise ValueError("run_model requires `processed` unless `saved` points to an existing parquet.")
        out = self._pipeline.run_model(processed, self._spec)
        if saved is not None:
            self._save_dict_frames(path, out.model_by_ticker, extra_meta={"representation": out.representation})
        return out

    def run_skew(
        self,
        modeled: ModelOutput | None = None,
        saved: str | None = None,
        force_recompute: bool = False,
    ) -> SkewOutput:
        path = self._resolve_save_path("skew", saved) if saved is not None else None
        if path is not None and path.exists() and not force_recompute:
            skew_by_ticker, _ = self._load_dict_frames(path)
            return SkewOutput(skew_by_ticker=skew_by_ticker)
        if modeled is None:
            raise ValueError("run_skew requires `modeled` unless `saved` points to an existing parquet.")
        out = self._pipeline.run_skew(modeled, self._spec)
        if saved is not None:
            self._save_dict_frames(path, out.skew_by_ticker)
        return out

    def run_signals(
        self,
        skew: SkewOutput | None = None,
        saved: str | None = None,
        force_recompute: bool = False,
    ) -> SignalOutput:
        path = self._resolve_save_path("signals", saved) if saved is not None else None
        if path is not None and path.exists() and not force_recompute:
            signal_map, _ = self._load_dict_frames(path)
            return SignalOutput(signal_map=signal_map)
        if skew is None:
            raise ValueError("run_signals requires `skew` unless `saved` points to an existing parquet.")
        out = self._pipeline.run_signals(skew, self._spec)
        if saved is not None:
            self._save_dict_frames(path, out.signal_map)
        return out

    def run_backtest(self, signals: SignalOutput, skew: SkewOutput) -> BacktestOutput:
        return self._pipeline.run_backtest(signals, skew, self._spec)

    def run_walk_forward(
        self,
        skew: SkewOutput | None = None,
        saved: str | None = None,
        candidate_overrides: List[Dict[str, object]] | None = None,
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 63,
    ) -> WalkForwardOutput:
        if skew is None:
            if saved is None:
                raise ValueError("run_walk_forward requires `skew` or `saved` pointing to an existing skew parquet.")
            skew = self.run_skew(saved=saved)

        benchmark = self.config.signal.benchmark_preference
        benchmark_df = skew.skew_by_ticker.get(benchmark)
        if benchmark_df is not None and not benchmark_df.empty:
            all_dates = sorted(pd.to_datetime(benchmark_df["date"]).drop_duplicates().tolist())
        else:
            all_dates = sorted(
                {
                    pd.Timestamp(d)
                    for df in skew.skew_by_ticker.values()
                    if df is not None and not df.empty
                    for d in pd.to_datetime(df["date"]).tolist()
                }
            )
        windows = self._walk_forward_windows(all_dates, train_days=train_days, test_days=test_days, step_days=step_days)
        if not windows:
            raise ValueError("No walk-forward windows available for the supplied date range and train/test settings.")

        candidates = candidate_overrides or self._default_walk_forward_candidates()
        for i, overrides in enumerate(candidates, start=1):
            for key in overrides:
                if "." not in key:
                    raise ValueError(
                        f"Walk-forward candidate {i}: override key '{key}' must use dot notation "
                        "(e.g. 'signal.entry_z', 'model.rr_k')."
                    )
                section = key.split(".", 1)[0]
                if section not in {"data", "model", "signal", "backtest"}:
                    raise ValueError(
                        f"Walk-forward candidate {i}: unknown section '{section}' in key '{key}'."
                    )

        min_train_required = max(
            int(overrides.get("signal.regression_window", self.config.signal.regression_window))
            + int(overrides.get("signal.zscore_window", self.config.signal.zscore_window))
            for overrides in candidates
        )
        if train_days < min_train_required:
            raise ValueError(
                f"train_days={train_days} is too short for the supplied candidate set. "
                f"Need at least {min_train_required} days to cover regression + z-score lookbacks."
            )

        candidate_rows: List[Dict[str, object]] = []
        window_rows: List[Dict[str, object]] = []
        oos_portfolio_parts: List[pd.DataFrame] = []
        oos_by_ticker_parts: Dict[str, List[pd.DataFrame]] = {}

        for window_id, (train_dates, test_dates) in enumerate(windows, start=1):
            combo_dates = train_dates + test_dates
            skew_window = self._slice_skew_output(skew, combo_dates)
            best_idx: int | None = None
            best_train_score = -1e18
            best_backtest: BacktestOutput | None = None
            best_overrides: Dict[str, object] | None = None
            best_test_stats: Dict[str, float] | None = None

            for candidate_id, overrides in enumerate(candidates, start=1):
                cfg = self._apply_overrides(
                    replace(
                        self.config,
                        data=replace(
                            self.config.data,
                            start_date=str(pd.Timestamp(combo_dates[0]).date()),
                            end_date=str(pd.Timestamp(combo_dates[-1]).date()),
                        ),
                    ),
                    overrides,
                )
                pipe = Pipeline(cfg)
                signals = pipe.run_signals(skew_window)
                backtest = pipe.run_backtest(signals, skew_window)
                train_portfolio = self._slice_df_by_dates(backtest.portfolio, train_dates)
                test_portfolio = self._slice_df_by_dates(backtest.portfolio, test_dates)
                train_stats = self._score_portfolio(
                    train_portfolio,
                    regression_window=cfg.signal.regression_window,
                    zscore_window=cfg.signal.zscore_window,
                )
                test_stats = self._score_portfolio(
                    test_portfolio,
                    regression_window=cfg.signal.regression_window,
                    zscore_window=cfg.signal.zscore_window,
                )
                candidate_rows.append(
                    {
                        "window_id": window_id,
                        "candidate_id": candidate_id,
                        "train_start": str(pd.Timestamp(train_dates[0]).date()),
                        "train_end": str(pd.Timestamp(train_dates[-1]).date()),
                        "test_start": str(pd.Timestamp(test_dates[0]).date()),
                        "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                        "overrides_json": json.dumps(overrides, sort_keys=True),
                        "train_score": float(train_stats["score"]),
                        "train_sharpe": float(train_stats["daily_sharpe_like"]),
                        "train_final_cum_net_pnl": float(train_stats["final_cum_net_pnl"]),
                        "test_score": float(test_stats["score"]),
                        "test_sharpe": float(test_stats["daily_sharpe_like"]),
                        "test_final_cum_net_pnl": float(test_stats["final_cum_net_pnl"]),
                    }
                )
                if train_stats["score"] > best_train_score:
                    best_train_score = float(train_stats["score"])
                    best_idx = candidate_id
                    best_backtest = backtest
                    best_overrides = overrides
                    best_test_stats = test_stats

            if best_idx is None or best_backtest is None or best_overrides is None or best_test_stats is None:
                continue

            test_portfolio_best = self._slice_df_by_dates(best_backtest.portfolio, test_dates)
            if not test_portfolio_best.empty:
                test_portfolio_best = test_portfolio_best.copy()
                test_portfolio_best["window_id"] = window_id
                test_portfolio_best["selected_candidate_id"] = best_idx
                oos_portfolio_parts.append(test_portfolio_best)

            for ticker, df in best_backtest.by_ticker.items():
                test_df = self._slice_df_by_dates(df, test_dates)
                if test_df.empty:
                    continue
                test_df = test_df.copy()
                test_df["window_id"] = window_id
                test_df["selected_candidate_id"] = best_idx
                oos_by_ticker_parts.setdefault(ticker, []).append(test_df)

            window_rows.append(
                {
                    "window_id": window_id,
                    "train_start": str(pd.Timestamp(train_dates[0]).date()),
                    "train_end": str(pd.Timestamp(train_dates[-1]).date()),
                    "test_start": str(pd.Timestamp(test_dates[0]).date()),
                    "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                    "selected_candidate_id": best_idx,
                    "selected_overrides_json": json.dumps(best_overrides, sort_keys=True),
                    "train_score": float(best_train_score),
                    "test_score": float(best_test_stats["score"]),
                    "test_sharpe": float(best_test_stats["daily_sharpe_like"]),
                    "test_final_cum_net_pnl": float(best_test_stats["final_cum_net_pnl"]),
                }
            )

        if not oos_portfolio_parts:
            raise RuntimeError("Walk-forward produced no out-of-sample portfolio slices.")

        oos_portfolio = (
            pd.concat(oos_portfolio_parts, ignore_index=True)
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="first")
            .reset_index(drop=True)
        )
        oos_portfolio["portfolio_cum_net_pnl"] = oos_portfolio["portfolio_net_pnl"].cumsum()

        oos_by_ticker = {
            ticker: (
                pd.concat(parts, ignore_index=True)
                .sort_values("date")
                .drop_duplicates(subset=["date"], keep="first")
                .reset_index(drop=True)
            )
            for ticker, parts in oos_by_ticker_parts.items()
        }
        summary = self._score_portfolio(
            oos_portfolio,
            regression_window=self.config.signal.regression_window,
            zscore_window=self.config.signal.zscore_window,
        )
        return WalkForwardOutput(
            portfolio=oos_portfolio,
            by_ticker=oos_by_ticker,
            windows=pd.DataFrame(window_rows),
            candidates=pd.DataFrame(candidate_rows),
            summary=summary,
        )


if __name__ == "__main__":
    cfg = MasterRunSpec(
        data=DataSpec(
            tickers=["XLF", "GS", "JPM", "BAC", "BRK", "C"],
            start_date="2008-01-01",
            end_date="2008-04-30",
        ),
    )

    p = Pipeline(cfg)
    loaded = p.run_load()
    processed = p.run_process(loaded)
    modeled = p.run_model(processed)
    skew = p.run_skew(modeled)
    signals = p.run_signals(skew)
    backtest = p.run_backtest(signals, skew)

    print(backtest.portfolio.tail())
    print(backtest.summary)
