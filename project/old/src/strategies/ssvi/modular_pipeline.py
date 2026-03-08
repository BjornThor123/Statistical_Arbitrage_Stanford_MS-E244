from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.common.option_backtest import OptionBacktestConfig, backtest_option_cross_sectional
from src.common.skew_signal_backtest import (
    SignalConfig,
    TransactionCostConfig,
    backtest_cross_sectional_residual_mean_reversion,
    build_residual_signal,
    summarize_performance,
)
from src.strategies.ssvi.pipeline import (
    PanelFilters,
    fit_daily_ssvi_skew,
    load_options_history_duckdb,
    preprocess_options_panel,
)


@dataclass
class RunSpec:
    tickers: List[str]
    sector_ticker: str
    start_date: str
    end_date: str
    target_days: int = 30
    k0: float = 0.0

    @property
    def target_t(self) -> float:
        return self.target_days / 365.0


@dataclass
class LoadedUniverse:
    stock_raw: Dict[str, pd.DataFrame]
    sector_raw: pd.DataFrame


@dataclass
class PreparedUniverse:
    stock_panel: Dict[str, pd.DataFrame]
    sector_panel: pd.DataFrame


@dataclass
class ModelOutputs:
    stock_models: Dict[str, pd.DataFrame]
    sector_model: pd.DataFrame
    representation: str


@dataclass
class SkewOutputs:
    stock_skews: Dict[str, pd.DataFrame]
    sector_skew: pd.DataFrame


@dataclass
class SignalOutputs:
    signal_map: Dict[str, pd.DataFrame]


@dataclass
class PipelineOutputs:
    loaded: LoadedUniverse
    prepared: PreparedUniverse
    models: ModelOutputs
    skews: SkewOutputs
    signals: SignalOutputs
    portfolio_df: pd.DataFrame
    per_name: Dict[str, pd.DataFrame]
    summary: Dict[str, float]


class DataLoaderModule(ABC):
    @abstractmethod
    def load(self, spec: RunSpec) -> LoadedUniverse:
        raise NotImplementedError


class PreprocessModule(ABC):
    @abstractmethod
    def process(self, loaded: LoadedUniverse, spec: RunSpec) -> PreparedUniverse:
        raise NotImplementedError


class VolatilityModelModule(ABC):
    @abstractmethod
    def fit(self, prepared: PreparedUniverse, spec: RunSpec) -> ModelOutputs:
        raise NotImplementedError


class SkewModule(ABC):
    @abstractmethod
    def extract(self, models: ModelOutputs, spec: RunSpec) -> SkewOutputs:
        raise NotImplementedError


class SignalModule(ABC):
    @abstractmethod
    def generate(self, skews: SkewOutputs, spec: RunSpec) -> SignalOutputs:
        raise NotImplementedError


class BacktestModule(ABC):
    @abstractmethod
    def run(self, signals: SignalOutputs, skews: SkewOutputs, spec: RunSpec) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, float]]:
        raise NotImplementedError


@dataclass
class DuckDBDataLoader(DataLoaderModule):
    db_path: str
    table: str = "options_enriched"

    def _load_ticker(self, ticker: str, spec: RunSpec) -> pd.DataFrame:
        return load_options_history_duckdb(
            db_path=self.db_path,
            ticker=ticker,
            table=self.table,
            start_date=spec.start_date,
            end_date=spec.end_date,
        )

    def load(self, spec: RunSpec) -> LoadedUniverse:
        stock_raw: Dict[str, pd.DataFrame] = {
            t: self._load_ticker(t, spec) for t in spec.tickers
        }
        sector_raw = self._load_ticker(spec.sector_ticker, spec)
        return LoadedUniverse(stock_raw=stock_raw, sector_raw=sector_raw)


@dataclass
class OptionsPanelPreprocessor(PreprocessModule):
    filters: PanelFilters = field(default_factory=PanelFilters)

    def process(self, loaded: LoadedUniverse, spec: RunSpec) -> PreparedUniverse:
        stock_panel = {
            t: preprocess_options_panel(df, filters=self.filters)
            for t, df in loaded.stock_raw.items()
        }
        sector_panel = preprocess_options_panel(loaded.sector_raw, filters=self.filters)
        return PreparedUniverse(stock_panel=stock_panel, sector_panel=sector_panel)


@dataclass
class SSVISurfaceModel(VolatilityModelModule):
    calibration_backend: str = "auto"
    min_points_per_day: int = 120
    min_maturities_per_day: int = 3

    def _fit_one(self, panel: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
        return fit_daily_ssvi_skew(
            panel=panel,
            target_t=spec.target_t,
            k0=spec.k0,
            min_points_per_day=self.min_points_per_day,
            min_maturities_per_day=self.min_maturities_per_day,
            calibration_backend=self.calibration_backend,
        )

    def fit(self, prepared: PreparedUniverse, spec: RunSpec) -> ModelOutputs:
        stock_models = {t: self._fit_one(panel, spec) for t, panel in prepared.stock_panel.items()}
        sector_model = self._fit_one(prepared.sector_panel, spec)
        return ModelOutputs(stock_models=stock_models, sector_model=sector_model, representation="surface")


@dataclass
class LinearSmileModel(VolatilityModelModule):
    t_band: float = 10.0 / 365.0
    #min_points_per_day: int = 60

    def _fit_one(self, panel: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for d, day in panel.groupby("date"):
            sub = day[np.abs(day["t"] - spec.target_t) <= self.t_band].copy()
            # if len(sub) < self.min_points_per_day:
            #     continue
            k = sub["k"].to_numpy(dtype=np.float64)
            sigma = sub["sigma"].to_numpy(dtype=np.float64)
            if np.unique(k).shape[0] < 5:
                continue
            slope, intercept = np.polyfit(k, sigma, deg=1)
            rows.append(
                {
                    "date": pd.Timestamp(d),
                    "ticker": str(sub["ticker"].iloc[0]),
                    "smile_slope": float(slope),
                    "smile_intercept": float(intercept),
                    "target_t": float(spec.target_t),
                    "n_points": int(len(sub)),
                    "rmse_implied_volatility": float(np.sqrt(np.mean((sigma - (intercept + slope * k)) ** 2))),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["date", "ticker", "smile_slope", "target_t", "n_points", "rmse_implied_volatility"])
        return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    def fit(self, prepared: PreparedUniverse, spec: RunSpec) -> ModelOutputs:
        stock_models = {t: self._fit_one(panel, spec) for t, panel in prepared.stock_panel.items()}
        sector_model = self._fit_one(prepared.sector_panel, spec)
        return ModelOutputs(stock_models=stock_models, sector_model=sector_model, representation="smile")


@dataclass
class GenericSkewExtractor(SkewModule):
    skew_columns: Sequence[str] = ("skew", "smile_slope")

    def _normalize(self, df: pd.DataFrame, ticker_hint: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["date", "ticker", "skew", "n_points", "rmse_implied_volatility"])

        source_col = next((c for c in self.skew_columns if c in df.columns), None)
        if source_col is None:
            raise ValueError(f"No skew-like column found. Tried: {self.skew_columns}")

        out = df.copy()
        if "ticker" not in out.columns:
            out["ticker"] = ticker_hint
        out = out.rename(columns={source_col: "skew"})

        if "n_points" not in out.columns:
            out["n_points"] = np.nan
        if "rmse_implied_volatility" not in out.columns:
            out["rmse_implied_volatility"] = np.nan

        keep = [
            "date",
            "ticker",
            "skew",
            "n_points",
            "rmse_implied_volatility",
            "rho",
            "eta",
            "gamma",
            "theta_target",
            "forward_price",
            "atm_iv",
        ]
        cols = [c for c in keep if c in out.columns]
        return out[cols].sort_values(["ticker", "date"]).reset_index(drop=True)

    def extract(self, models: ModelOutputs, spec: RunSpec) -> SkewOutputs:
        stock_skews = {
            t: self._normalize(df, ticker_hint=t)
            for t, df in models.stock_models.items()
        }
        sector_skew = self._normalize(models.sector_model, ticker_hint=spec.sector_ticker)
        return SkewOutputs(stock_skews=stock_skews, sector_skew=sector_skew)


@dataclass
class ResidualSignalGenerator(SignalModule):
    signal_cfg: SignalConfig = field(default_factory=SignalConfig)
    cost_cfg: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    factor_model: str = "pca"
    n_pca_factors: int = 2

    def generate(self, skews: SkewOutputs, spec: RunSpec) -> SignalOutputs:
        universe_skews = {**skews.stock_skews, spec.sector_ticker: skews.sector_skew}
        signal_map: Dict[str, pd.DataFrame] = {}
        for ticker, stock_skew in skews.stock_skews.items():
            signal_map[ticker] = build_residual_signal(
                stock_skew=stock_skew,
                sector_skew=skews.sector_skew,
                signal_cfg=self.signal_cfg,
                cost_cfg=self.cost_cfg,
                factor_model=self.factor_model,
                universe_skews=universe_skews,
                n_pca_factors=self.n_pca_factors,
            )
        return SignalOutputs(signal_map=signal_map)


@dataclass
class IdealizedBacktestEngine(BacktestModule):
    signal_cfg: SignalConfig = field(default_factory=SignalConfig)
    cost_cfg: TransactionCostConfig = field(default_factory=TransactionCostConfig)

    def run(self, signals: SignalOutputs, skews: SkewOutputs, spec: RunSpec) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, float]]:
        portfolio_df, per_name = backtest_cross_sectional_residual_mean_reversion(
            signal_map=signals.signal_map,
            signal_cfg=self.signal_cfg,
            cost_cfg=self.cost_cfg,
        )
        return portfolio_df, per_name, summarize_performance(portfolio_df)


@dataclass
class OptionBacktestEngine(BacktestModule):
    signal_cfg: SignalConfig = field(default_factory=SignalConfig)
    option_cfg: OptionBacktestConfig = field(default_factory=OptionBacktestConfig)

    def run(self, signals: SignalOutputs, skews: SkewOutputs, spec: RunSpec) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, float]]:
        portfolio_df, per_name = backtest_option_cross_sectional(
            signal_map=signals.signal_map,
            skew_map=skews.stock_skews,
            signal_cfg=self.signal_cfg,
            option_cfg=self.option_cfg,
        )
        return portfolio_df, per_name, summarize_performance(portfolio_df)


@dataclass
class ModularSkewStrategyPipeline:
    data_loader: DataLoaderModule
    preprocessor: PreprocessModule
    vol_model: VolatilityModelModule
    skew_extractor: SkewModule
    signal_generator: SignalModule
    backtest_engine: BacktestModule

    def run(self, spec: RunSpec) -> PipelineOutputs:
        loaded = self.data_loader.load(spec)
        prepared = self.preprocessor.process(loaded, spec)
        models = self.vol_model.fit(prepared, spec)
        skews = self.skew_extractor.extract(models, spec)
        signals = self.signal_generator.generate(skews, spec)
        portfolio_df, per_name, summary = self.backtest_engine.run(signals, skews, spec)
        return PipelineOutputs(
            loaded=loaded,
            prepared=prepared,
            models=models,
            skews=skews,
            signals=signals,
            portfolio_df=portfolio_df,
            per_name=per_name,
            summary=summary,
        )


def make_default_ssvi_pipeline(
    db_path: str,
    table: str = "options_enriched",
    calibration_backend: str = "auto",
    signal_cfg: SignalConfig | None = None,
    cost_cfg: TransactionCostConfig | None = None,
    option_cfg: OptionBacktestConfig | None = None,
    backtest_mode: str = "option",
) -> ModularSkewStrategyPipeline:
    signal_cfg = signal_cfg or SignalConfig()
    cost_cfg = cost_cfg or TransactionCostConfig()

    if backtest_mode == "option":
        backtest_engine: BacktestModule = OptionBacktestEngine(
            signal_cfg=signal_cfg,
            option_cfg=(option_cfg or OptionBacktestConfig()),
        )
    elif backtest_mode == "idealized":
        backtest_engine = IdealizedBacktestEngine(
            signal_cfg=signal_cfg,
            cost_cfg=cost_cfg,
        )
    else:
        raise ValueError("backtest_mode must be one of: option, idealized")

    return ModularSkewStrategyPipeline(
        data_loader=DuckDBDataLoader(db_path=db_path, table=table),
        preprocessor=OptionsPanelPreprocessor(),
        vol_model=SSVISurfaceModel(calibration_backend=calibration_backend),
        skew_extractor=GenericSkewExtractor(),
        signal_generator=ResidualSignalGenerator(
            signal_cfg=signal_cfg,
            cost_cfg=cost_cfg,
            factor_model="pca",
            n_pca_factors=2,
        ),
        backtest_engine=backtest_engine,
    )
