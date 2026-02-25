"""
Volatility Surface Generator Module
====================================
Constructs implied volatility surfaces from equity option data.

Two approaches are implemented:
  1. BlackScholesVSG  – invert BS formula per option, then interpolate
  2. SSVIVSG          – fit the SVI parameterisation (Gatheral & Jacquier 2013)

Both produce a VolSurface object that supports skew calculation and 3D plotting.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate, stats
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)


# ── Result container ────────────────────────────────────────────────────────

@dataclass
class VolSurface:
    """Container for a fitted volatility surface on a single date."""

    date: str
    ticker: str
    log_moneyness_grid: np.ndarray          # 1-D, sorted
    time_to_expiry_grid: np.ndarray         # 1-D, sorted
    iv_matrix: np.ndarray                   # shape (len(tte), len(k))
    raw_data: pd.DataFrame                  # cleaned option rows used
    interpolator: Optional[object] = field(default=None, repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.iv_matrix.shape

    def iv(self, k: float, T: float) -> float:
        """Interpolated implied vol at log-moneyness *k* and maturity *T*."""
        if self.interpolator is None:
            raise RuntimeError("No interpolator fitted.")
        return float(self.interpolator(T, k))


# ── Abstract base class ────────────────────────────────────────────────────

class VolatilitySurfaceGenerator(ABC):
    """
    Abstract base for volatility surface generators.

    Parameters
    ----------
    option_data : pd.DataFrame
        Enriched option data from ``options_enriched`` table (via DataLoader).
        Expected pre-computed columns: date, exdate, cp_flag, strike,
        best_bid, best_offer, impl_volatility, forward_price, spot_price,
        mid_price, spread_pct, tte_days, tte, log_moneyness,
        volume, open_interest, ticker.
    min_volume : int
        Minimum daily option volume for inclusion.
    min_open_interest : int
        Minimum open interest for inclusion.
    max_spread_pct : float
        Maximum bid-ask spread as fraction of mid-price.
    moneyness_bounds : tuple[float, float]
        (lower, upper) bounds on log-moneyness to keep.
    min_tte_days : int
        Drop options with fewer than this many days to expiry.
    """

    REQUIRED_COLS = [
        "date", "exdate", "cp_flag", "strike",
        "best_bid", "best_offer", "impl_volatility",
        "forward_price", "mid_price", "tte_days", "tte",
        "log_moneyness", "volume", "open_interest", "ticker",
    ]

    def __init__(
        self,
        option_data: pd.DataFrame,
        *,
        min_volume: int = 10,
        min_open_interest: int = 100,
        max_spread_pct: float = 0.50,
        moneyness_bounds: tuple[float, float] = (-0.5, 0.5),
        min_tte_days: int = 7,
    ):
        self._raw = option_data.copy()
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest
        self.max_spread_pct = max_spread_pct
        self.moneyness_bounds = moneyness_bounds
        self.min_tte_days = min_tte_days

        self._validate_columns()
        self.option_data = self._prepare_data()

    # ── validation & cleaning ──────────────────────────────────────────

    def _validate_columns(self) -> None:
        missing = set(self.REQUIRED_COLS) - set(self._raw.columns)
        if missing:
            derived = {"strike", "mid_price", "tte_days", "tte", "log_moneyness"}
            if derived.intersection(missing):
                raise ValueError(
                    "Missing required columns for volatility surface generation: "
                    f"{missing}. Pass data from 'options_enriched' (not raw 'options')."
                )
            raise ValueError(f"Missing required columns: {missing}")

    def _prepare_data(self) -> pd.DataFrame:
        """Apply liquidity filters to pre-enriched option data.

        Expects columns already computed by DataLoader.build_options_enriched_table():
        strike, mid_price, spread_pct, tte_days, tte, forward_price, log_moneyness.
        """
        df = self._raw.copy()

        # Backward compatibility with older options_enriched schema versions
        # where computed forward price was stored as forward_price_1.
        if "forward_price_1" in df.columns:
            fp1 = pd.to_numeric(df["forward_price_1"], errors="coerce")
            if "forward_price" not in df.columns:
                df["forward_price"] = fp1
            else:
                df["forward_price"] = pd.to_numeric(
                    df["forward_price"], errors="coerce"
                ).fillna(fp1)

        # Ensure numeric types (DuckDB may return objects for some columns)
        numeric_cols = [
            "strike", "best_bid", "best_offer", "impl_volatility",
            "forward_price", "mid_price", "spread_pct", "tte_days", "tte",
            "log_moneyness", "volume", "open_interest",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"])
        df["exdate"] = pd.to_datetime(df["exdate"])

        # --- liquidity filters -------------------------------------------
        n_before = len(df)
        df = df.dropna(subset=["impl_volatility", "forward_price", "strike", "log_moneyness"])
        df = df[df["mid_price"] > 0]
        df = df[df["impl_volatility"] > 0]
        df = df[df["tte_days"] >= self.min_tte_days]
        df = df[df["volume"] >= self.min_volume]
        df = df[df["open_interest"] >= self.min_open_interest]
        df = df[df["spread_pct"] <= self.max_spread_pct]
        df = df[df["log_moneyness"].between(*self.moneyness_bounds)]

        n_after = len(df)
        logger.info(
            f"Liquidity filter: {n_before} -> {n_after} rows "
            f"({n_before - n_after} dropped)"
        )

        return df.reset_index(drop=True)

    # ── abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def generate_surface(self, date: str) -> VolSurface:
        """Build a volatility surface for a single observation date."""
        ...

    @abstractmethod
    def visualize_surface(self, surface: VolSurface, **kwargs) -> None:
        """Render a 3-D plot of the volatility surface."""
        ...

    @abstractmethod
    def calculate_skew(
        self,
        surface: VolSurface,
        target_tte: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Compute skew = dσ/dk for each maturity slice.

        Returns a DataFrame with columns [tte, skew].
        """
        ...

    # ── shared helpers ─────────────────────────────────────────────────

    def available_dates(self) -> list[str]:
        """Return sorted list of observation dates in the data."""
        return sorted(
            self.option_data["date"].dt.strftime("%Y-%m-%d").unique()
        )

    def generate_surfaces(self) -> dict[str, VolSurface]:
        """Generate a surface for every available date."""
        surfaces: dict[str, VolSurface] = {}
        for dt in self.available_dates():
            try:
                surfaces[dt] = self.generate_surface(dt)
            except ValueError as e:
                logger.warning(f"Skipping {dt}: {e}")
        return surfaces


# ── Black-Scholes implementation ────────────────────────────────────────────

class BlackScholesVSG(VolatilitySurfaceGenerator):
    """
    Volatility surface via Black-Scholes implied volatility.

    Uses the dataset's pre-computed ``impl_volatility`` column (OptionMetrics
    already inverts BS).  The surface is constructed by interpolating the
    discrete (log_moneyness, tte) grid using a smooth bivariate spline.

    If you want to re-derive IV from option prices, set ``recompute_iv=True``
    and provide risk-free rate data via ``risk_free_df`` (the yield panel from
    DuckDB's ``risk_free`` table) or a scalar ``risk_free_rate`` fallback.

    The yield panel has rows indexed by date and columns ``1, 2, ..., 360``
    representing monthly maturities (annualised zero-coupon rates).
    """

    def __init__(
        self,
        option_data: pd.DataFrame,
        *,
        recompute_iv: bool = False,
        risk_free_df: Optional[pd.DataFrame] = None,
        risk_free_rate: Optional[float] = None,
        n_moneyness: int = 50,
        n_tte: int = 30,
        robust_mode: bool = True,
        spline_s: Optional[float] = None,
        **kwargs,
    ):
        self.recompute_iv = recompute_iv
        self.risk_free_rate = risk_free_rate
        self.n_moneyness = n_moneyness
        self.n_tte = n_tte
        self.robust_mode = robust_mode
        self.spline_s = spline_s
        self._rf_interp = self._build_rf_interpolator(risk_free_df)
        super().__init__(option_data, **kwargs)

    @staticmethod
    def _build_rf_interpolator(
        rf_df: Optional[pd.DataFrame],
    ) -> Optional[dict]:
        """
        Pre-process the yield panel into a lookup {date_str -> interp1d}.

        The panel columns are monthly maturity labels (1..360 months).
        We build a per-date linear interpolator: months -> annualised rate.
        """
        if rf_df is None:
            return None

        df = rf_df.copy()
        # First column is the date (unnamed or index), second is MAX_DATA_TTM
        if df.columns[0] in ("", "Unnamed: 0"):
            df = df.rename(columns={df.columns[0]: "date"})
        if "date" not in df.columns:
            df = df.reset_index().rename(columns={"index": "date"})

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Drop non-numeric helper columns
        df = df.drop(columns=["MAX_DATA_TTM"], errors="ignore")

        # Column names are month strings ("1", "2", ..., "360")
        month_cols = [c for c in df.columns if c.isdigit()]
        months = np.array([int(c) for c in month_cols], dtype=float)

        lookup: dict[str, object] = {}
        for dt, row in df[month_cols].iterrows():
            rates = pd.to_numeric(row, errors="coerce").values
            mask = ~np.isnan(rates)
            if mask.sum() >= 2:
                lookup[dt.strftime("%Y-%m-%d")] = interpolate.interp1d(
                    months[mask], rates[mask],
                    kind="linear", fill_value="extrapolate",
                )
        return lookup

    def _get_risk_free_rate(self, date: str, tte_years: float) -> float:
        """Look up the risk-free rate for a given date and time-to-expiry."""
        if self._rf_interp is not None and date in self._rf_interp:
            months = tte_years * 12.0
            return float(self._rf_interp[date](months))
        if self.risk_free_rate is not None:
            return self.risk_free_rate
        return 0.0

    # ── BS formula helpers ─────────────────────────────────────────────

    @staticmethod
    def _bs_price(
        S: float, K: float, T: float, r: float, sigma: float, cp: str,
    ) -> float:
        """Closed-form Black-Scholes European option price."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if cp == "C":
            return float(
                S * stats.norm.cdf(d1)
                - K * np.exp(-r * T) * stats.norm.cdf(d2)
            )
        return float(
            K * np.exp(-r * T) * stats.norm.cdf(-d2)
            - S * stats.norm.cdf(-d1)
        )

    @staticmethod
    def _bs_vega(
        S: float, K: float, T: float, r: float, sigma: float,
    ) -> float:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return float(S * stats.norm.pdf(d1) * np.sqrt(T))

    def _implied_vol_newton(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        cp: str,
        *,
        tol: float = 1e-8,
        max_iter: int = 100,
        init_vol: float = 0.3,
    ) -> float:
        """Newton-Raphson solver for Black-Scholes implied volatility."""
        sigma = init_vol
        for _ in range(max_iter):
            price = self._bs_price(S, K, T, r, sigma, cp)
            vega = self._bs_vega(S, K, T, r, sigma)
            if vega < 1e-12:
                return np.nan
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            sigma -= diff / vega
            if sigma <= 0:
                return np.nan
        return np.nan

    def _recompute_implied_vols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Re-derive IV from mid-prices using Newton-Raphson.

        Uses ``spot_price`` and ``risk_free_rate`` from the enriched table.
        Falls back to the scalar ``self.risk_free_rate`` or 0 if the
        per-row rate is missing.
        """
        ivs = []
        for _, row in df.iterrows():
            r = row.get("risk_free_rate")
            if r is None or pd.isna(r):
                r = self._get_risk_free_rate(
                    row["date"].strftime("%Y-%m-%d"), row["tte"],
                )
            iv = self._implied_vol_newton(
                market_price=row["mid_price"],
                S=row["spot_price"],
                K=row["strike"],
                T=row["tte"],
                r=float(r),
                cp=row["cp_flag"],
            )
            ivs.append(iv)
        df = df.copy()
        df["impl_volatility"] = ivs
        return df.dropna(subset=["impl_volatility"])

    # ── core interface ─────────────────────────────────────────────────

    def generate_surface(self, date: str) -> VolSurface:
        """
        Build a volatility surface for a single observation date.

        Steps:
        1. Filter data to the given date.
        2. Optionally recompute IV via BS inversion.
        3. Aggregate duplicate (moneyness, tte) points by volume-weighted mean.
        4. Fit a smooth bivariate spline over (tte, log_moneyness) -> IV.
        5. Evaluate on a regular grid and return a VolSurface.
        """
        df = self.option_data[
            self.option_data["date"] == pd.Timestamp(date)
        ].copy()

        if len(df) < 10:
            raise ValueError(
                f"Insufficient data on {date}: {len(df)} rows after filters"
            )

        ticker = df["ticker"].iloc[0]

        if self.recompute_iv:
            df = self._recompute_implied_vols(df)

        # Volume-weighted average IV for duplicate (k, T) points
        df["iv_weighted"] = df["impl_volatility"] * df["volume"]
        grouped = (
            df.groupby(["log_moneyness", "tte"])
            .agg(iv_sum=("iv_weighted", "sum"), vol_sum=("volume", "sum"))
            .reset_index()
        )
        grouped["iv"] = grouped["iv_sum"] / grouped["vol_sum"]
        grouped = grouped.dropna(subset=["iv"])
        grouped = grouped[grouped["iv"] > 0]

        if len(grouped) < 6:
            raise ValueError(
                f"Too few unique (k, T) points on {date}: {len(grouped)}"
            )

        k_vals = grouped["log_moneyness"].values
        t_vals = grouped["tte"].values
        iv_vals = grouped["iv"].values

        # Evaluate on a regular grid
        k_grid = np.linspace(k_vals.min(), k_vals.max(), self.n_moneyness)
        t_grid = np.linspace(t_vals.min(), t_vals.max(), self.n_tte)
        T_mesh, K_mesh = np.meshgrid(t_grid, k_grid, indexing="ij")

        if self.robust_mode:
            # Linear interpolation on scattered points is much more stable in sparse regions.
            points = np.column_stack((t_vals, k_vals))
            linear_interp = interpolate.LinearNDInterpolator(points, iv_vals, fill_value=np.nan)
            nearest_interp = interpolate.NearestNDInterpolator(points, iv_vals)
            hull = Delaunay(points)
            query = np.column_stack((T_mesh.ravel(), K_mesh.ravel()))
            inside = hull.find_simplex(query) >= 0

            iv_flat = np.full(query.shape[0], np.nan, dtype=float)
            iv_lin = linear_interp(query[:, 0], query[:, 1])
            iv_flat[inside] = iv_lin[inside]

            # Fill rare interior NaNs (degenerate simplices) using nearest interior point.
            interior_nan = inside & np.isnan(iv_flat)
            if interior_nan.any():
                iv_flat[interior_nan] = nearest_interp(
                    query[interior_nan, 0], query[interior_nan, 1]
                )

            iv_matrix = iv_flat.reshape(T_mesh.shape)
            surface_interpolator = linear_interp
        else:
            # Smooth bivariate spline for scattered observations
            spline = interpolate.SmoothBivariateSpline(
                t_vals,
                k_vals,
                iv_vals,
                kx=min(3, len(np.unique(t_vals)) - 1),
                ky=min(3, len(np.unique(k_vals)) - 1),
                s=self.spline_s,
            )
            iv_matrix = spline(t_grid, k_grid)
            surface_interpolator = spline

        # Clamp negative IVs to a small positive floor
        iv_matrix = np.where(np.isnan(iv_matrix), np.nan, np.maximum(iv_matrix, 1e-4))

        return VolSurface(
            date=date,
            ticker=ticker,
            log_moneyness_grid=k_grid,
            time_to_expiry_grid=t_grid,
            iv_matrix=iv_matrix,
            raw_data=df,
            interpolator=surface_interpolator,
        )

    def visualize_surface(
        self,
        surface: VolSurface,
        *,
        ax=None,
        title: Optional[str] = None,
        cmap: str = "viridis",
        elev: float = 30,
        azim: float = -60,
        show_points: bool = True,
        point_size: float = 10.0,
        point_alpha: float = 0.35,
        point_color: str = "black",
    ) -> None:
        """Render a 3-D surface plot of implied volatility."""
        import matplotlib.pyplot as plt

        K, T = np.meshgrid(
            surface.log_moneyness_grid, surface.time_to_expiry_grid
        )

        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            K, T, surface.iv_matrix,
            cmap=cmap, alpha=0.85, edgecolor="none",
        )
        if show_points and len(surface.raw_data) > 0:
            ax.scatter(
                surface.raw_data["log_moneyness"].values,
                surface.raw_data["tte"].values,
                surface.raw_data["impl_volatility"].values,
                s=point_size,
                alpha=point_alpha,
                c=point_color,
                depthshade=False,
            )
        ax.set_xlabel("Log-Moneyness (k)")
        ax.set_ylabel("Time to Expiry (years)")
        ax.set_zlabel("Implied Volatility")
        ax.set_title(title or f"{surface.ticker} IV Surface – {surface.date}")
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        plt.show()

    def calculate_skew(
        self,
        surface: VolSurface,
        target_tte: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Skew = dσ/dk at ATM (k=0) for each maturity slice.

        Uses the fitted spline's analytical partial derivative.

        Parameters
        ----------
        surface : VolSurface
            A fitted surface object.
        target_tte : float, optional
            If given, compute skew only at this maturity.
            Otherwise compute for every grid maturity.

        Returns
        -------
        pd.DataFrame with columns [tte, skew].
        """
        if surface.interpolator is None:
            raise RuntimeError("Surface has no interpolator.")

        spline = surface.interpolator
        ttes = (
            np.array([target_tte])
            if target_tte is not None
            else surface.time_to_expiry_grid
        )

        k_atm = 0.0
        skews = []
        for t in ttes:
            try:
                # SmoothBivariateSpline supports analytical derivative via dy=1.
                skew_val = float(spline(t, k_atm, dy=1))
            except TypeError:
                # Linear/interpolators without derivatives: central finite difference.
                h = 1e-3
                up = float(spline(t, k_atm + h))
                dn = float(spline(t, k_atm - h))
                skew_val = (up - dn) / (2.0 * h)
            skews.append({"tte": float(t), "skew": skew_val})
        return pd.DataFrame(skews)

    def calculate_skew_term_structure(
        self,
        surface: VolSurface,
        k_points: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Skew across the full moneyness range for each maturity.

        Returns DataFrame with columns [tte, log_moneyness, skew].
        """
        if surface.interpolator is None:
            raise RuntimeError("Surface has no interpolator.")

        spline = surface.interpolator
        if k_points is None:
            k_points = surface.log_moneyness_grid

        rows = [
            {
                "tte": float(t),
                "log_moneyness": float(k),
                "skew": (
                    float(spline(t, k, dy=1))
                    if "SmoothBivariateSpline" in type(spline).__name__
                    else (float(spline(t, k + 1e-3)) - float(spline(t, k - 1e-3))) / (2e-3)
                ),
            }
            for t in surface.time_to_expiry_grid
            for k in k_points
        ]
        return pd.DataFrame(rows)


# ── SVI / SSVI stub ─────────────────────────────────────────────────────────

class SSVIVSG(VolatilitySurfaceGenerator):
    """
    SVI / SSVI volatility surface (Gatheral & Jacquier 2013).

    Parameterises total implied variance as:
        w(k) = a + b * (ρ(k - m) + sqrt((k - m)² + σ²))

    TODO: Implement SVI slice fitting + SSVI cross-maturity consistency.
    """

    def generate_surface(self, date: str) -> VolSurface:
        raise NotImplementedError("SSVI surface fitting not yet implemented.")

    def visualize_surface(self, surface: VolSurface, **kwargs) -> None:
        raise NotImplementedError

    def calculate_skew(
        self,
        surface: VolSurface,
        target_tte: Optional[float] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError



if __name__ == "__main__":
    from data_loader import DataLoader

    DATA = "/Users/bjorn/Documents/Skóli/Stanford/Skóli/Q2/StatArb/Statistical_Arbitrage_Stanford_MS-E244/project/data"
    loader = DataLoader(data_path=DATA)

    # Single date to test quickly — query from enriched table
    option_data = loader.query(
        "SELECT * FROM options_enriched WHERE ticker = 'GS' AND date = '2019-06-03'"
    )
    print(f"Loaded {len(option_data)} rows")

    # Use pre-computed IVs (fast) — set recompute_iv=True to re-derive via BS
    vsg = BlackScholesVSG(option_data, min_volume=5, min_open_interest=50)
    print(f"After filters: {len(vsg.option_data)} rows")
    print(f"Available dates: {vsg.available_dates()}")

    for dt in vsg.available_dates():
        surface = vsg.generate_surface(dt)
        print(f"Surface {dt}: shape {surface.shape}")
        skew = vsg.calculate_skew(surface)
        print(f"ATM skew:\n{skew}")
        vsg.visualize_surface(surface)
    
    
