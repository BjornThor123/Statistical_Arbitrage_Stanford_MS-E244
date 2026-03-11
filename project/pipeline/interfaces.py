"""
Pipeline interfaces and data containers.

Architecture overview
---------------------
Data flows through six stages, each represented by an abstract module:

  DataLoaderModule   → LoadedData        (raw options quotes per ticker)
  DataProcessorModule → ProcessedData    (cleaned panel: k, t, sigma, F)
  VolatilityModelModule → ModelOutput    (fitted SSVI surface per date)
  SkewCalculatorModule  → SkewOutput     (scalar skew per ticker per date)
  SignalGeneratorModule → SignalOutput   (z-score signals + position weights)
  BacktestEngineModule  → BacktestOutput (PnL, greeks, costs)

RunSpec carries the shared run parameters (tickers, dates, target maturity).
StrategyPipeline wires all modules together and exposes run_* stage methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class RunSpec:
    tickers: List[str]
    start_date: str
    end_date: str
    target_days: int = 30
    k0: float = 0.0

    @property
    def target_t(self) -> float:
        return self.target_days / 365.0


@dataclass
class LoadedData:
    raw_by_ticker: Dict[str, pd.DataFrame]


@dataclass
class ProcessedData:
    panel_by_ticker: Dict[str, pd.DataFrame]


@dataclass
class ModelOutput:
    model_by_ticker: Dict[str, pd.DataFrame]
    representation: str


@dataclass
class SkewOutput:
    skew_by_ticker: Dict[str, pd.DataFrame]


@dataclass
class SignalOutput:
    signal_map: Dict[str, pd.DataFrame]


@dataclass
class BacktestOutput:
    portfolio: pd.DataFrame
    by_ticker: Dict[str, pd.DataFrame]
    summary: Dict[str, float]


@dataclass
class WalkForwardOutput:
    portfolio: pd.DataFrame
    by_ticker: Dict[str, pd.DataFrame]
    windows: pd.DataFrame
    candidates: pd.DataFrame
    summary: Dict[str, float]


@dataclass
class PipelineOutput:
    loaded: LoadedData
    processed: ProcessedData
    modeled: ModelOutput
    skew: SkewOutput
    signals: SignalOutput
    backtest: BacktestOutput


class DataLoaderModule(ABC):
    @abstractmethod
    def load(self, spec: RunSpec) -> LoadedData:
        raise NotImplementedError


class DataProcessorModule(ABC):
    @abstractmethod
    def process(self, loaded: LoadedData, spec: RunSpec) -> ProcessedData:
        raise NotImplementedError


class VolatilityModelModule(ABC):
    @abstractmethod
    def fit(self, processed: ProcessedData, spec: RunSpec) -> ModelOutput:
        raise NotImplementedError


class SkewCalculatorModule(ABC):
    @abstractmethod
    def compute(self, modeled: ModelOutput, spec: RunSpec) -> SkewOutput:
        raise NotImplementedError


class SignalGeneratorModule(ABC):
    @abstractmethod
    def generate(self, skew: SkewOutput, spec: RunSpec) -> SignalOutput:
        raise NotImplementedError


class BacktestEngineModule(ABC):
    @abstractmethod
    def run(self, signals: SignalOutput, skew: SkewOutput, spec: RunSpec) -> BacktestOutput:
        raise NotImplementedError


@dataclass
class StrategyPipeline:
    data_loader: DataLoaderModule
    data_processor: DataProcessorModule
    volatility_modeler: VolatilityModelModule
    skew_calculator: SkewCalculatorModule
    signal_generator: SignalGeneratorModule
    backtest_engine: BacktestEngineModule

    def run_load(self, spec: RunSpec) -> LoadedData:
        return self.data_loader.load(spec)

    def run_process(self, loaded: LoadedData, spec: RunSpec) -> ProcessedData:
        return self.data_processor.process(loaded, spec)

    def run_model(self, processed: ProcessedData, spec: RunSpec) -> ModelOutput:
        return self.volatility_modeler.fit(processed, spec)

    def run_skew(self, modeled: ModelOutput, spec: RunSpec) -> SkewOutput:
        return self.skew_calculator.compute(modeled, spec)

    def run_signals(self, skew: SkewOutput, spec: RunSpec) -> SignalOutput:
        return self.signal_generator.generate(skew, spec)

    def run_backtest(self, signals: SignalOutput, skew: SkewOutput, spec: RunSpec) -> BacktestOutput:
        return self.backtest_engine.run(signals, skew, spec)

    def run(self, spec: RunSpec) -> PipelineOutput:
        loaded = self.run_load(spec)
        processed = self.run_process(loaded, spec)
        modeled = self.run_model(processed, spec)
        skew = self.run_skew(modeled, spec)
        signals = self.run_signals(skew, spec)
        backtest = self.run_backtest(signals, skew, spec)
        return PipelineOutput(
            loaded=loaded,
            processed=processed,
            modeled=modeled,
            skew=skew,
            signals=signals,
            backtest=backtest,
        )
