from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
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
class PipelineConfig:
    tickers: List[str]
    start_date: str
    end_date: str
    model_kind: Literal["surface", "smile"] = "surface"
    db_table: str = "options_enriched"
    ssvi_calibration_backend: str = "auto"  # auto, cpu, mps, cuda
    ssvi_random_seed: int = 42
    ssvi_n_param_steps: int = 300
    ssvi_n_theta_steps: int = 3
    ssvi_n_restarts: int = 4
    ssvi_theta_smoothness_lambda: float = 2e-3
    skew_technique: str = "local_derivative"
    skew_delta_k: float = 0.01
    skew_rr_k: float = 0.25
    signal_regression_window: int = 60
    signal_min_regression_obs: int = 40
    signal_zscore_window: int = 60
    signal_entry_z: float = 1.25
    signal_exit_z: float = 0.30
    signal_min_hold_days: int = 3
    signal_max_abs_position: float = 1.0
    signal_min_liquidity_points: float = 150.0
    signal_max_fit_rmse_iv: float = 0.60
    signal_edge_horizon_days: int = 5
    signal_edge_cost_buffer: float = 0.25
    signal_half_spread_cost: float = 0.02
    signal_commission_cost: float = 0.005
    signal_direction: str = "auto"
    signal_model_kind: str = "risk_reversal"  # risk_reversal | residual
    signal_cross_section_top_k: int = 2
    signal_cross_section_bottom_k: int = 2
    signal_max_name_weight: float = 0.35
    signal_max_gross_leverage: float = 1.5
    signal_vol_target_daily: float = 0.02
    signal_auto_sign_window: int = 63
    signal_winsor_quantile: float = 0.02
    signal_use_mad_zscore: bool = True
    signal_regime_sector_vol_window: int = 21
    signal_regime_sector_abs_z_max: float = 3.0
    signal_regime_sector_vol_z_max: float = 2.0
    signal_factor_model: str = "sector"  # pca | sector
    signal_n_pca_factors: int = 2
    signal_benchmark_preference: str = "XLF"
    backtest_stock_fee_bps: float = 1.0
    backtest_option_fee_per_contract: float = 0.65
    backtest_sizing_mode: str = "contracts"  # contracts | dollar_vega
    backtest_max_contracts_per_signal: float = 10.0
    backtest_target_dollar_vega_per_signal: float = 20000.0
    backtest_roll_threshold_days: int = 5
    backtest_rr_put_log_moneyness: float = -0.15
    backtest_rr_call_log_moneyness: float = 0.15


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.db_path = f"{DATA_LOCATION}/market_data.duckdb"
        self._pipeline = self._build_pipeline()
        self._spec = RunSpec(
            tickers=self.config.tickers,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        self._stage_save_root = Path(__file__).resolve().parent / "stage_saves"
        self._stage_save_root.mkdir(parents=True, exist_ok=True)

    def _cache_id(self) -> str:
        payload = {
            "tickers": self.config.tickers,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "model_kind": self.config.model_kind,
            "db_table": self.config.db_table,
            "ssvi_calibration_backend": self.config.ssvi_calibration_backend,
            "ssvi_random_seed": self.config.ssvi_random_seed,
            "ssvi_n_param_steps": self.config.ssvi_n_param_steps,
            "ssvi_n_theta_steps": self.config.ssvi_n_theta_steps,
            "ssvi_n_restarts": self.config.ssvi_n_restarts,
            "ssvi_theta_smoothness_lambda": self.config.ssvi_theta_smoothness_lambda,
            "target_days": self._spec.target_days,
            "k0": self._spec.k0,
            "signal_regression_window": self.config.signal_regression_window,
            "signal_min_regression_obs": self.config.signal_min_regression_obs,
            "signal_zscore_window": self.config.signal_zscore_window,
            "signal_entry_z": self.config.signal_entry_z,
            "signal_exit_z": self.config.signal_exit_z,
            "signal_min_hold_days": self.config.signal_min_hold_days,
            "signal_max_abs_position": self.config.signal_max_abs_position,
            "signal_min_liquidity_points": self.config.signal_min_liquidity_points,
            "signal_max_fit_rmse_iv": self.config.signal_max_fit_rmse_iv,
            "signal_edge_horizon_days": self.config.signal_edge_horizon_days,
            "signal_edge_cost_buffer": self.config.signal_edge_cost_buffer,
            "signal_half_spread_cost": self.config.signal_half_spread_cost,
            "signal_commission_cost": self.config.signal_commission_cost,
            "signal_direction": self.config.signal_direction,
            "signal_model_kind": self.config.signal_model_kind,
            "signal_cross_section_top_k": self.config.signal_cross_section_top_k,
            "signal_cross_section_bottom_k": self.config.signal_cross_section_bottom_k,
            "signal_max_name_weight": self.config.signal_max_name_weight,
            "signal_max_gross_leverage": self.config.signal_max_gross_leverage,
            "signal_vol_target_daily": self.config.signal_vol_target_daily,
            "signal_auto_sign_window": self.config.signal_auto_sign_window,
            "signal_winsor_quantile": self.config.signal_winsor_quantile,
            "signal_use_mad_zscore": self.config.signal_use_mad_zscore,
            "signal_regime_sector_vol_window": self.config.signal_regime_sector_vol_window,
            "signal_regime_sector_abs_z_max": self.config.signal_regime_sector_abs_z_max,
            "signal_regime_sector_vol_z_max": self.config.signal_regime_sector_vol_z_max,
            "signal_factor_model": self.config.signal_factor_model,
            "signal_n_pca_factors": self.config.signal_n_pca_factors,
            "signal_benchmark_preference": self.config.signal_benchmark_preference,
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
            "calibration_backend": self.config.ssvi_calibration_backend,
            "random_seed": self.config.ssvi_random_seed,
            "n_param_steps": self.config.ssvi_n_param_steps,
            "n_theta_steps": self.config.ssvi_n_theta_steps,
            "n_restarts": self.config.ssvi_n_restarts,
            "theta_smoothness_lambda": self.config.ssvi_theta_smoothness_lambda,
        }
        skew_kwargs = {
            "technique": self.config.skew_technique,
            "delta_k": self.config.skew_delta_k,
            "rr_k": self.config.skew_rr_k,
        }
        signal_kwargs = {
            "regression_window": self.config.signal_regression_window,
            "min_regression_obs": self.config.signal_min_regression_obs,
            "zscore_window": self.config.signal_zscore_window,
            "entry_z": self.config.signal_entry_z,
            "exit_z": self.config.signal_exit_z,
            "min_hold_days": self.config.signal_min_hold_days,
            "max_abs_position": self.config.signal_max_abs_position,
            "min_liquidity_points": self.config.signal_min_liquidity_points,
            "max_fit_rmse_iv": self.config.signal_max_fit_rmse_iv,
            "edge_horizon_days": self.config.signal_edge_horizon_days,
            "edge_cost_buffer": self.config.signal_edge_cost_buffer,
            "half_spread_cost": self.config.signal_half_spread_cost,
            "commission_cost": self.config.signal_commission_cost,
            "signal_direction": self.config.signal_direction,
            "cross_section_top_k": self.config.signal_cross_section_top_k,
            "cross_section_bottom_k": self.config.signal_cross_section_bottom_k,
            "max_name_weight": self.config.signal_max_name_weight,
            "max_gross_leverage": self.config.signal_max_gross_leverage,
            "vol_target_daily": self.config.signal_vol_target_daily,
            "auto_sign_window": self.config.signal_auto_sign_window,
            "winsor_quantile": self.config.signal_winsor_quantile,
            "use_mad_zscore": self.config.signal_use_mad_zscore,
            "regime_sector_vol_window": self.config.signal_regime_sector_vol_window,
            "regime_sector_abs_z_max": self.config.signal_regime_sector_abs_z_max,
            "regime_sector_vol_z_max": self.config.signal_regime_sector_vol_z_max,
            "factor_model": self.config.signal_factor_model,
            "n_pca_factors": self.config.signal_n_pca_factors,
            "benchmark_preference": self.config.signal_benchmark_preference,
        }
        backtest_kwargs = {
            "stock_fee_bps": self.config.backtest_stock_fee_bps,
            "option_fee_per_contract": self.config.backtest_option_fee_per_contract,
            "sizing_mode": self.config.backtest_sizing_mode,
            "max_contracts_per_signal": self.config.backtest_max_contracts_per_signal,
            "target_dollar_vega_per_signal": self.config.backtest_target_dollar_vega_per_signal,
            "roll_threshold_days": self.config.backtest_roll_threshold_days,
            "rr_put_log_moneyness": self.config.backtest_rr_put_log_moneyness,
            "rr_call_log_moneyness": self.config.backtest_rr_call_log_moneyness,
        }
        if self.config.model_kind == "surface":
            return build_default_surface_pipeline(
                db_path=self.db_path,
                table=self.config.db_table,
                surface_kwargs=surface_kwargs,
                skew_kwargs=skew_kwargs,
                signal_kwargs=signal_kwargs,
                backtest_kwargs=backtest_kwargs,
                signal_kind=self.config.signal_model_kind,
            )
        if self.config.model_kind == "smile":
            return build_default_smile_pipeline(
                db_path=self.db_path,
                table=self.config.db_table,
                skew_kwargs=skew_kwargs,
                signal_kwargs=signal_kwargs,
                backtest_kwargs=backtest_kwargs,
                signal_kind=self.config.signal_model_kind,
            )
        raise ValueError("model_kind must be one of: surface, smile")

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
        base = {
            "signal_direction": "auto",
            "signal_min_liquidity_points": 150.0,
            "signal_max_fit_rmse_iv": 0.60,
            "signal_cross_section_top_k": 2,
            "signal_cross_section_bottom_k": 2,
            "signal_max_name_weight": 0.35,
            "signal_max_gross_leverage": 1.5,
            "signal_vol_target_daily": 0.02,
        }
        candidates: List[Dict[str, object]] = []
        for factor_model in ["pca", "sector"]:
            for entry_z, exit_z in [(1.00, 0.25), (1.25, 0.30)]:
                for rr_width in [0.10, 0.15]:
                    cand = dict(base)
                    cand.update(
                        {
                            "signal_factor_model": factor_model,
                            "signal_n_pca_factors": 2,
                            "signal_entry_z": entry_z,
                            "signal_exit_z": exit_z,
                            "backtest_rr_put_log_moneyness": -rr_width,
                            "backtest_rr_call_log_moneyness": rr_width,
                        }
                    )
                    candidates.append(cand)
        for factor_model in ["pca", "sector"]:
            cand = dict(base)
            cand.update(
                {
                    "signal_factor_model": factor_model,
                    "signal_n_pca_factors": 2,
                    "signal_entry_z": 1.25,
                    "signal_exit_z": 0.30,
                    "signal_max_name_weight": 0.30,
                    "signal_max_gross_leverage": 1.25,
                    "signal_vol_target_daily": 0.015,
                    "backtest_rr_put_log_moneyness": -0.15,
                    "backtest_rr_call_log_moneyness": 0.15,
                }
            )
            candidates.append(cand)
        return candidates

    def run(self):
        return self._pipeline.run(self._spec)

    def run_load(self, save: bool = False, saved: str | None = None, force_recompute: bool = False) -> LoadedData:
        path = self._resolve_save_path("load", saved) if saved is not None else None
        if path is not None and path.exists() and not force_recompute:
            raw_by_ticker, _ = self._load_dict_frames(path)
            return LoadedData(raw_by_ticker=raw_by_ticker)
        out = self._pipeline.run_load(self._spec)
        if save:
            if path is None:
                raise ValueError("Explicit cache path required: pass `saved='filename.parquet'` when save=True.")
            self._save_dict_frames(path, out.raw_by_ticker)
        return out

    def run_process(
        self,
        loaded: LoadedData | None = None,
        save: bool = False,
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
        if save:
            if path is None:
                raise ValueError("Explicit cache path required: pass `saved='filename.parquet'` when save=True.")
            self._save_dict_frames(path, out.panel_by_ticker)
        return out

    def run_model(
        self,
        processed: ProcessedData | None = None,
        save: bool = False,
        saved: str | None = None,
        force_recompute: bool = False,
    ) -> ModelOutput:
        path = self._resolve_save_path("model", saved) if saved is not None else None
        if path is not None and path.exists() and not force_recompute:
            model_by_ticker, meta = self._load_dict_frames(path)
            if "representation" not in meta:
                raise ValueError(
                    "Saved model artifact is missing required metadata field 'representation'. "
                    "Re-run run_model(..., save=True) with the current code."
                )
            representation = str(meta["representation"])
            return ModelOutput(model_by_ticker=model_by_ticker, representation=representation)
        if processed is None:
            raise ValueError("run_model requires `processed` unless `saved` points to an existing parquet.")
        out = self._pipeline.run_model(processed, self._spec)
        if save:
            if path is None:
                raise ValueError("Explicit cache path required: pass `saved='filename.parquet'` when save=True.")
            self._save_dict_frames(path, out.model_by_ticker, extra_meta={"representation": out.representation})
        return out

    def run_skew(
        self,
        modeled: ModelOutput | None = None,
        save: bool = False,
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
        if save:
            if path is None:
                raise ValueError("Explicit cache path required: pass `saved='filename.parquet'` when save=True.")
            self._save_dict_frames(path, out.skew_by_ticker)
        return out

    def run_signals(
        self,
        skew: SkewOutput | None = None,
        save: bool = False,
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
        if save:
            if path is None:
                raise ValueError("Explicit cache path required: pass `saved='filename.parquet'` when save=True.")
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

        benchmark = self.config.signal_benchmark_preference
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
        config_fields = set(asdict(self.config).keys())
        for i, overrides in enumerate(candidates, start=1):
            invalid = set(overrides.keys()) - config_fields
            if invalid:
                raise ValueError(f"Invalid walk-forward candidate override keys for candidate {i}: {sorted(invalid)}")
        min_train_required = max(
            int(overrides.get("signal_regression_window", self.config.signal_regression_window))
            + int(overrides.get("signal_zscore_window", self.config.signal_zscore_window))
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
                cfg = replace(
                    self.config,
                    start_date=str(pd.Timestamp(combo_dates[0]).date()),
                    end_date=str(pd.Timestamp(combo_dates[-1]).date()),
                    **overrides,
                )
                pipe = Pipeline(cfg)
                signals = pipe.run_signals(skew_window)
                backtest = pipe.run_backtest(signals, skew_window)
                train_portfolio = self._slice_df_by_dates(backtest.portfolio, train_dates)
                test_portfolio = self._slice_df_by_dates(backtest.portfolio, test_dates)
                train_stats = self._score_portfolio(
                    train_portfolio,
                    regression_window=cfg.signal_regression_window,
                    zscore_window=cfg.signal_zscore_window,
                )
                test_stats = self._score_portfolio(
                    test_portfolio,
                    regression_window=cfg.signal_regression_window,
                    zscore_window=cfg.signal_zscore_window,
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
        if "portfolio_cum_net_pnl" not in oos_portfolio.columns:
            oos_portfolio["portfolio_cum_net_pnl"] = oos_portfolio["portfolio_net_pnl"].cumsum()
        else:
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
            regression_window=self.config.signal_regression_window,
            zscore_window=self.config.signal_zscore_window,
        )
        return WalkForwardOutput(
            portfolio=oos_portfolio,
            by_ticker=oos_by_ticker,
            windows=pd.DataFrame(window_rows),
            candidates=pd.DataFrame(candidate_rows),
            summary=summary,
        )


if __name__ == "__main__":
    cfg = PipelineConfig(
        tickers=["XLF", "GS", "JPM", "BAC", "BRK", "C", "TLQD"],
        start_date="2008-01-01",
        end_date="2008-12-31",
        model_kind="surface",  # swap to "smile" to test 2D module
    )
    
    #output = Pipeline(cfg).run()
    loaded = Pipeline(cfg).run_load()
    processed = Pipeline(cfg).run_process(loaded)
    modeled = Pipeline(cfg).run_model(processed)
    print(modeled.model_by_ticker["XLF"].head())
    #skew = Pipeline(cfg).run_skew(modeled)
    #signals = Pipeline(cfg).run_signals(skew)
    #backtest = Pipeline(cfg).run_backtest(signals, skew)
    
    #print(backtest.raw_by_ticker["XLF"].head())
    
    #print(output.backtest.summary)
