from dataclasses import dataclass

DATA_LOCATION = "/Users/bjorn/Documents/Skóli/Stanford/Skóli/Q2/StatArb/Statistical_Arbitrage_Stanford_MS-E244/project/data"


@dataclass
class PanelFilters:
    min_sigma: float = 0.01
    max_sigma: float = 5.0
    min_t: float = 1.0 / 365.0
    max_t: float = 2.5
    max_abs_k: float = 1.5
    min_points_per_day: int = 120
    min_maturities_per_day: int = 3
