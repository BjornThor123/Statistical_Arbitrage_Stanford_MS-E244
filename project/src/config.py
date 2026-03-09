from dataclasses import dataclass, field
from typing import List, Literal
from datetime import datetime
from pathlib import Path

@dataclass(frozen=True)
class Config:
    data_path: Path = Path("data")
    skew_path: Path = Path("data/skew.parquet")
    cleaned_options_path: Path = Path("data/cleaned_options.parquet")
    plot_dir: Path = Path("plots")
    start_date: datetime = datetime(2015, 1, 1)
    end_date: datetime = datetime(2020, 12, 31)
    tte_target: int = 30
    max_tte: int = 40
    sector_ticker = "XLF"
    entry_threshold: float = 1.0
    exit_threshold: float = 0.0
    delta_target: float = 0.25
    # Skew measure used for signal construction.
    # "direct"     — IV_25Δ_put − IV_25Δ_call at tte_target (matches the traded instrument)
    # "polynomial" — −β from IV = α + β·log(K/F) + γ·log(K/F)² (negated so high = puts expensive)
    skew_method: Literal["direct", "vega_hedged", "polynomial"] = "direct"
    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 0.5
    max_position_frac: float = 0.20

    relevant_option_columns: List[str] = field(default_factory=lambda: [
        "secid",
        "date",
        "symbol",
        "symbol_flag",
        "exdate",
        "last_date",
        "cp_flag",
        "strike_price",
        "best_bid",
        "best_offer",
        "volume",
        "open_interest",
        "impl_volatility",
        "delta",
        "gamma",
        "vega",
        "theta",
        "optionid",
        # "cfadj",
        # "am_settlement",
        "contract_size",
        # "ss_flag",
        # "forward_price",
        # "expiry_indicator",
        # "root",
        # "suffix",
        # "cusip",
        "ticker",
        # "sic",
        # "index_flag",
        # "exchange_d",
        # "class",
        # "issue_type",
        # "industry_group",
        # "issuer",
        # "div_convention",
        # "exercise_style",
        # "am_set_flag",
        "source_label",
        "strike",
        "mid_price",
        "spread",
        "tte_days",
        "tte",
        "tte_months",
        "spot_price",
        "risk_free_rate",
        "forward_price",
        "spread_pct",
        "log_moneyness",
    ])

def get_config() -> Config:
    return Config()