"""
Volatility model analytics.

plot_surface_fit_analytics(modeled, ticker)
    — animated 3-D surface fit with a date slider (SSVI only).

plot_smile_fit_analytics(modeled, ticker)
    — animated 2-D smile fit with a date slider (linear smile only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider

from interfaces import ModelOutput
from volatility_models.model import ssvi_total_variance

try:
    import plotly.graph_objects as go
except Exception:
    go = None


@dataclass
class _FramePayload:
    date: pd.Timestamp
    observed: pd.DataFrame
    k_grid: np.ndarray
    t_grid: np.ndarray
    sigma_grid: np.ndarray
    metrics: Dict[str, float]


def _fit_theta_nodes_for_day(day: pd.DataFrame, rho: float, eta: float, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    t_nodes = np.sort(day["t"].to_numpy(dtype=np.float64).round(8).astype(np.float64).copy())
    t_nodes = np.unique(t_nodes)
    theta_nodes: List[float] = []

    for t_val in t_nodes:
        sub = day[np.isclose(day["t"].to_numpy(dtype=np.float64), t_val, atol=1e-8)]
        k_obs = sub["k"].to_numpy(dtype=np.float64)
        sigma_obs = sub["sigma"].to_numpy(dtype=np.float64)
        w_obs = np.maximum(sigma_obs, 1e-8) ** 2 * np.maximum(t_val, 1e-8)
        base = float(np.median(w_obs))
        lo = max(base * 0.25, 1e-8)
        hi = max(base * 4.0, lo * 1.2)
        theta_grid = np.geomspace(lo, hi, 121)
        losses = []
        for th in theta_grid:
            w_pred = ssvi_total_variance(k_obs, np.full_like(k_obs, th), rho=rho, eta=eta, gamma=gamma)
            losses.append(float(np.mean((w_pred - w_obs) ** 2)))
        theta_nodes.append(float(theta_grid[int(np.argmin(losses))]))

    theta_arr = np.maximum.accumulate(np.asarray(theta_nodes, dtype=np.float64))
    return t_nodes, np.maximum(theta_arr, 1e-8)


def _theta_interp(t: np.ndarray, t_nodes: np.ndarray, theta_nodes: np.ndarray) -> np.ndarray:
    return np.interp(
        np.maximum(t, t_nodes[0]),
        t_nodes,
        theta_nodes,
        left=theta_nodes[0],
        right=theta_nodes[-1],
    )


def _error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    max_abs = float(np.max(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "bias": bias, "max_abs_error": max_abs, "r2": r2}


def _modeled_ticker_df(modeled: ModelOutput, ticker: str) -> pd.DataFrame:
    if ticker not in modeled.model_by_ticker:
        raise ValueError(f"Ticker '{ticker}' not found in modeled output. Available: {sorted(modeled.model_by_ticker)}")
    out = modeled.model_by_ticker[ticker].copy()
    if out.empty:
        raise ValueError(f"No modeled rows for ticker '{ticker}'.")
    return out


def _plot_surface_fit_analytics(
    panel: pd.DataFrame,
    modeled_surface: pd.DataFrame,
    ticker: str | None = None,
    engine: Literal["plotly", "matplotlib"] = "plotly",
) -> pd.DataFrame:
    """Interactive SSVI surface diagnostics with a date slider. Returns per-date metrics."""
    p = panel.copy()
    m = modeled_surface.copy()
    p["date"] = pd.to_datetime(p["date"])
    m["date"] = pd.to_datetime(m["date"])

    if ticker is not None:
        p = p[p["ticker"] == ticker].copy()
        m = m[m["ticker"] == ticker].copy()

    missing_panel = {"date", "k", "t", "sigma"} - set(p.columns)
    missing_model = {"date", "rho", "eta", "gamma"} - set(m.columns)
    if missing_panel:
        raise ValueError(f"panel missing required columns: {sorted(missing_panel)}")
    if missing_model:
        raise ValueError(f"modeled_surface missing required columns: {sorted(missing_model)}")
    if p.empty or m.empty:
        raise ValueError("No rows available after filtering.")

    common_dates = sorted(set(p["date"]).intersection(set(m["date"])))
    if not common_dates:
        raise ValueError("No overlapping dates between panel and modeled_surface.")

    frames: List[_FramePayload] = []
    metrics_rows: List[Dict[str, object]] = []
    for d in common_dates:
        day = p[p["date"] == d].copy()
        row = m[m["date"] == d].iloc[0]
        rho, eta, gamma = float(row["rho"]), float(row["eta"]), float(row["gamma"])

        t_nodes, theta_nodes = _fit_theta_nodes_for_day(day, rho=rho, eta=eta, gamma=gamma)
        k_obs = day["k"].to_numpy(dtype=np.float64)
        t_obs = day["t"].to_numpy(dtype=np.float64)
        sigma_obs = day["sigma"].to_numpy(dtype=np.float64)
        theta_obs = _theta_interp(t_obs, t_nodes=t_nodes, theta_nodes=theta_nodes)
        w_pred_obs = ssvi_total_variance(k_obs, theta_obs, rho=rho, eta=eta, gamma=gamma)
        sigma_pred_obs = np.sqrt(np.maximum(w_pred_obs, 1e-12) / np.maximum(t_obs, 1e-12))

        metrics = _error_metrics(sigma_obs, sigma_pred_obs)
        metrics_row: Dict[str, object] = {"date": d, "n_points": int(len(day))}
        metrics_row.update(metrics)
        if "rmse_implied_volatility" in row.index:
            metrics_row["calibration_rmse"] = float(row["rmse_implied_volatility"])
        metrics_rows.append(metrics_row)

        k_grid = np.linspace(np.nanquantile(k_obs, 0.01), np.nanquantile(k_obs, 0.99), 40)
        t_grid = np.linspace(np.nanquantile(t_obs, 0.01), np.nanquantile(t_obs, 0.99), 30)
        K, T = np.meshgrid(k_grid, t_grid)
        theta_grid = _theta_interp(T.ravel(), t_nodes=t_nodes, theta_nodes=theta_nodes)
        w_grid = ssvi_total_variance(K.ravel(), theta_grid, rho=rho, eta=eta, gamma=gamma)
        sigma_grid = np.sqrt(np.maximum(w_grid, 1e-12) / np.maximum(T.ravel(), 1e-12)).reshape(T.shape)

        frames.append(_FramePayload(date=d, observed=day, k_grid=K, t_grid=T, sigma_grid=sigma_grid, metrics=metrics))

    metrics_df = pd.DataFrame(metrics_rows).sort_values("date").reset_index(drop=True)
    print("Surface fit summary:")
    print(metrics_df[["rmse", "mae", "bias", "max_abs_error", "r2"]].describe().round(6))

    if engine == "plotly":
        if go is None:
            raise ImportError("plotly is not installed. Install plotly or use engine='matplotlib'.")
        dates = [f.date.strftime("%Y-%m-%d") for f in frames]
        traces0 = [
            go.Surface(
                x=frames[0].k_grid, y=frames[0].t_grid, z=frames[0].sigma_grid,
                colorscale="Viridis", opacity=0.75, name="Model surface", showscale=False,
            ),
            go.Scatter3d(
                x=frames[0].observed["k"].to_numpy(dtype=np.float64),
                y=frames[0].observed["t"].to_numpy(dtype=np.float64),
                z=frames[0].observed["sigma"].to_numpy(dtype=np.float64),
                mode="markers", marker={"size": 3, "color": "crimson"}, name="Observed",
            ),
        ]
        plotly_frames = [
            go.Frame(
                name=str(i),
                data=[
                    go.Surface(x=fr.k_grid, y=fr.t_grid, z=fr.sigma_grid, colorscale="Viridis", opacity=0.75, showscale=False),
                    go.Scatter3d(
                        x=fr.observed["k"].to_numpy(dtype=np.float64),
                        y=fr.observed["t"].to_numpy(dtype=np.float64),
                        z=fr.observed["sigma"].to_numpy(dtype=np.float64),
                        mode="markers", marker={"size": 3, "color": "crimson"},
                    ),
                ],
                layout=go.Layout(title=f"Surface Fit | {dates[i]} | RMSE={fr.metrics['rmse']:.4f}, MAE={fr.metrics['mae']:.4f}, R2={fr.metrics['r2']:.4f}"),
            )
            for i, fr in enumerate(frames)
        ]
        steps = [
            {"method": "animate", "label": dates[i], "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
            for i in range(len(frames))
        ]
        fig = go.Figure(data=traces0, frames=plotly_frames)
        fig.update_layout(
            title=f"Surface Fit | {dates[0]} | RMSE={frames[0].metrics['rmse']:.4f}, MAE={frames[0].metrics['mae']:.4f}, R2={frames[0].metrics['r2']:.4f}",
            scene={"xaxis_title": "Log-moneyness k", "yaxis_title": "Maturity t (years)", "zaxis_title": "Implied vol sigma"},
            sliders=[{"active": 0, "steps": steps, "x": 0.08, "len": 0.86}],
            margin={"l": 0, "r": 0, "t": 60, "b": 0},
        )
        fig.show()
        return metrics_df

    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.18)
    slider_ax = fig.add_axes([0.17, 0.06, 0.66, 0.03])
    slider = Slider(slider_ax, "Date", 0, len(frames) - 1, valinit=0, valstep=1)

    def _draw(idx: int) -> None:
        frame = frames[idx]
        ax.clear()
        ax.plot_surface(frame.k_grid, frame.t_grid, frame.sigma_grid, alpha=0.6, cmap="viridis", linewidth=0, antialiased=True)
        ax.scatter(
            frame.observed["k"].to_numpy(dtype=np.float64),
            frame.observed["t"].to_numpy(dtype=np.float64),
            frame.observed["sigma"].to_numpy(dtype=np.float64),
            c="crimson", s=14, alpha=0.9,
        )
        ax.set_xlabel("Log-moneyness k")
        ax.set_ylabel("Maturity t (years)")
        ax.set_zlabel("Implied vol sigma")
        ax.set_title(f"Surface Fit | {frame.date.date()} | RMSE={frame.metrics['rmse']:.4f}, MAE={frame.metrics['mae']:.4f}, R2={frame.metrics['r2']:.4f}")
        fig.canvas.draw_idle()

    _draw(0)
    slider.on_changed(lambda val: _draw(int(val)))
    plt.show()
    return metrics_df


def _plot_smile_fit_analytics(
    panel: pd.DataFrame,
    modeled_smile: pd.DataFrame,
    ticker: str | None = None,
    engine: Literal["plotly", "matplotlib"] = "plotly",
) -> pd.DataFrame:
    """Interactive linear-smile diagnostics with a date slider. Returns per-date metrics."""
    p = panel.copy()
    m = modeled_smile.copy()
    p["date"] = pd.to_datetime(p["date"])
    m["date"] = pd.to_datetime(m["date"])

    if ticker is not None:
        p = p[p["ticker"] == ticker].copy()
        m = m[m["ticker"] == ticker].copy()

    missing_panel = {"date", "k", "t", "sigma"} - set(p.columns)
    missing_model = {"date", "smile_slope", "smile_intercept", "target_t"} - set(m.columns)
    if missing_panel:
        raise ValueError(f"panel missing required columns: {sorted(missing_panel)}")
    if missing_model:
        raise ValueError(f"modeled_smile missing required columns: {sorted(missing_model)}")
    if p.empty or m.empty:
        raise ValueError("No rows available after filtering.")

    frames: List[Dict[str, object]] = []
    metrics_rows: List[Dict[str, object]] = []
    for _, row in m.sort_values("date").iterrows():
        d = pd.Timestamp(row["date"])
        target_t = float(row["target_t"])
        t_band = 10.0 / 365.0
        day = p[(p["date"] == d) & (np.abs(p["t"] - target_t) <= t_band)].copy()
        if day.empty:
            continue

        slope = float(row["smile_slope"])
        intercept = float(row["smile_intercept"])
        k_obs = day["k"].to_numpy(dtype=np.float64)
        sigma_obs = day["sigma"].to_numpy(dtype=np.float64)
        sigma_pred = intercept + slope * k_obs
        metrics = _error_metrics(sigma_obs, sigma_pred)

        metrics_row: Dict[str, object] = {"date": d, "n_points": int(len(day))}
        metrics_row.update(metrics)
        if "rmse_implied_volatility" in row.index:
            metrics_row["calibration_rmse"] = float(row["rmse_implied_volatility"])
        metrics_rows.append(metrics_row)

        k_line = np.linspace(np.nanquantile(k_obs, 0.01), np.nanquantile(k_obs, 0.99), 100)
        frames.append({"date": d, "obs": day, "k_line": k_line, "sigma_line": intercept + slope * k_line, "metrics": metrics, "target_t": target_t})

    if not frames:
        raise ValueError("No smile frames built. Verify panel maturities around target_t.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("date").reset_index(drop=True)
    print("Smile fit summary:")
    print(metrics_df[["rmse", "mae", "bias", "max_abs_error", "r2"]].describe().round(6))

    if engine == "plotly":
        if go is None:
            raise ImportError("plotly is not installed. Install plotly or use engine='matplotlib'.")
        dates = [pd.Timestamp(f["date"]).strftime("%Y-%m-%d") for f in frames]
        f0 = frames[0]
        fig = go.Figure(
            data=[
                go.Scatter(x=f0["obs"]["k"].to_numpy(dtype=np.float64), y=f0["obs"]["sigma"].to_numpy(dtype=np.float64), mode="markers", marker={"size": 6, "color": "royalblue"}, name="Observed"),
                go.Scatter(x=f0["k_line"], y=f0["sigma_line"], mode="lines", line={"width": 3, "color": "crimson"}, name="Fitted smile"),
            ]
        )
        fig.frames = [
            go.Frame(
                name=str(i),
                data=[
                    go.Scatter(x=fr["obs"]["k"].to_numpy(dtype=np.float64), y=fr["obs"]["sigma"].to_numpy(dtype=np.float64), mode="markers", marker={"size": 6, "color": "royalblue"}),
                    go.Scatter(x=fr["k_line"], y=fr["sigma_line"], mode="lines", line={"width": 3, "color": "crimson"}),
                ],
                layout=go.Layout(title=f"Smile Fit | {dates[i]} | target_t={fr['target_t']:.4f} | RMSE={fr['metrics']['rmse']:.4f}, MAE={fr['metrics']['mae']:.4f}, R2={fr['metrics']['r2']:.4f}"),
            )
            for i, fr in enumerate(frames)
        ]
        fig.update_layout(
            title=f"Smile Fit | {dates[0]} | target_t={f0['target_t']:.4f} | RMSE={f0['metrics']['rmse']:.4f}, MAE={f0['metrics']['mae']:.4f}, R2={f0['metrics']['r2']:.4f}",
            xaxis_title="Log-moneyness k",
            yaxis_title="Implied vol sigma",
            sliders=[{
                "active": 0,
                "steps": [{"method": "animate", "label": dates[i], "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]} for i in range(len(frames))],
                "x": 0.08, "len": 0.86,
            }],
        )
        fig.show()
        return metrics_df

    fig, ax = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(bottom=0.18)
    slider_ax = fig.add_axes([0.17, 0.06, 0.66, 0.03])
    slider = Slider(slider_ax, "Date", 0, len(frames) - 1, valinit=0, valstep=1)

    def _draw(idx: int) -> None:
        frame = frames[idx]
        ax.clear()
        day = frame["obs"]
        ax.scatter(day["k"].to_numpy(dtype=np.float64), day["sigma"].to_numpy(dtype=np.float64), c="royalblue", s=18, alpha=0.85, label="Observed")
        ax.plot(frame["k_line"], frame["sigma_line"], color="crimson", linewidth=2.2, label="Fitted smile")
        ax.set_xlabel("Log-moneyness k")
        ax.set_ylabel("Implied vol sigma")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        ax.set_title(f"Smile Fit | {frame['date'].date()} | target_t={frame['target_t']:.4f} | RMSE={frame['metrics']['rmse']:.4f}, MAE={frame['metrics']['mae']:.4f}, R2={frame['metrics']['r2']:.4f}")
        fig.canvas.draw_idle()

    _draw(0)
    slider.on_changed(lambda val: _draw(int(val)))
    plt.show()
    return metrics_df


def plot_surface_fit_analytics(modeled: ModelOutput, ticker: str) -> pd.DataFrame:
    """SSVI surface fit diagnostics for one ticker from ModelOutput."""
    if modeled.representation != "surface":
        raise ValueError(f"Expected modeled.representation='surface', got '{modeled.representation}'.")
    modeled_df = _modeled_ticker_df(modeled, ticker=ticker)
    missing = {"obs_k", "obs_t", "obs_sigma"} - set(modeled_df.columns)
    if missing:
        raise ValueError(f"Modeled output missing observed-point columns {sorted(missing)}. Re-run after updating volatility model modules.")

    panel_rows: List[Dict[str, object]] = []
    for _, r in modeled_df.iterrows():
        k = np.asarray(r["obs_k"], dtype=np.float64)
        t = np.asarray(r["obs_t"], dtype=np.float64)
        sigma = np.asarray(r["obs_sigma"], dtype=np.float64)
        for ki, ti, si in zip(k, t, sigma):
            panel_rows.append({"date": pd.Timestamp(r["date"]), "ticker": ticker, "k": ki, "t": ti, "sigma": si})
    panel = pd.DataFrame(panel_rows)
    return _plot_surface_fit_analytics(panel=panel, modeled_surface=modeled_df, ticker=ticker)


def plot_smile_fit_analytics(modeled: ModelOutput, ticker: str) -> pd.DataFrame:
    """Linear smile fit diagnostics for one ticker from ModelOutput."""
    if modeled.representation != "smile":
        raise ValueError(f"Expected modeled.representation='smile', got '{modeled.representation}'.")
    modeled_df = _modeled_ticker_df(modeled, ticker=ticker)
    missing = {"obs_k", "obs_sigma"} - set(modeled_df.columns)
    if missing:
        raise ValueError(f"Modeled output missing observed-point columns {sorted(missing)}. Re-run after updating volatility model modules.")

    panel_rows: List[Dict[str, object]] = []
    for _, r in modeled_df.iterrows():
        k = np.asarray(r["obs_k"], dtype=np.float64)
        sigma = np.asarray(r["obs_sigma"], dtype=np.float64)
        target_t = float(r.get("target_t", np.nan))
        for ki, si in zip(k, sigma):
            panel_rows.append({"date": pd.Timestamp(r["date"]), "ticker": ticker, "k": ki, "t": target_t, "sigma": si})
    panel = pd.DataFrame(panel_rows)
    return _plot_smile_fit_analytics(panel=panel, modeled_smile=modeled_df, ticker=ticker)
