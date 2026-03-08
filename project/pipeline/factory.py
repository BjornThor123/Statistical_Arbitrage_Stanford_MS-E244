from __future__ import annotations

from typing import Any

from backtest_models.modules import DeltaHedgedOptionBacktestEngine
from data_processing.modules import BasicOptionsProcessor, DuckDBOptionsLoader
from interfaces import StrategyPipeline
from signal_models.modules import ResidualZScoreSignalGenerator, RiskReversalSignalGenerator
from skew_models.modules import GenericSkewCalculator
from volatility_models.modules import LinearSmileModeler, SSVISurfaceModeler


def build_default_surface_pipeline(
    db_path: str,
    table: str = "options_enriched",
    surface_kwargs: dict[str, Any] | None = None,
    skew_kwargs: dict[str, Any] | None = None,
    signal_kwargs: dict[str, Any] | None = None,
    backtest_kwargs: dict[str, Any] | None = None,
    signal_kind: str = "risk_reversal",
) -> StrategyPipeline:
    if signal_kind == "risk_reversal":
        signal_generator = RiskReversalSignalGenerator(**(signal_kwargs or {}))
    elif signal_kind == "residual":
        signal_generator = ResidualZScoreSignalGenerator(**(signal_kwargs or {}))
    else:
        raise ValueError("signal_kind must be one of: risk_reversal, residual")
    return StrategyPipeline(
        data_loader=DuckDBOptionsLoader(db_path=db_path, table=table),
        data_processor=BasicOptionsProcessor(),
        volatility_modeler=SSVISurfaceModeler(**(surface_kwargs or {})),
        skew_calculator=GenericSkewCalculator(**(skew_kwargs or {})),
        signal_generator=signal_generator,
        backtest_engine=DeltaHedgedOptionBacktestEngine(db_path=db_path, table=table, **(backtest_kwargs or {})),
    )


def build_default_smile_pipeline(
    db_path: str,
    table: str = "options_enriched",
    skew_kwargs: dict[str, Any] | None = None,
    signal_kwargs: dict[str, Any] | None = None,
    backtest_kwargs: dict[str, Any] | None = None,
    signal_kind: str = "risk_reversal",
) -> StrategyPipeline:
    if signal_kind == "risk_reversal":
        signal_generator = RiskReversalSignalGenerator(**(signal_kwargs or {}))
    elif signal_kind == "residual":
        signal_generator = ResidualZScoreSignalGenerator(**(signal_kwargs or {}))
    else:
        raise ValueError("signal_kind must be one of: risk_reversal, residual")
    return StrategyPipeline(
        data_loader=DuckDBOptionsLoader(db_path=db_path, table=table),
        data_processor=BasicOptionsProcessor(),
        volatility_modeler=LinearSmileModeler(),
        skew_calculator=GenericSkewCalculator(**(skew_kwargs or {})),
        signal_generator=signal_generator,
        backtest_engine=DeltaHedgedOptionBacktestEngine(db_path=db_path, table=table, **(backtest_kwargs or {})),
    )
