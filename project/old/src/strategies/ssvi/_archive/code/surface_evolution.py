from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

STRATEGY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategies.ssvi.model import calibrate_ssvi_surface
from src.strategies.ssvi.pipeline import (
    PanelFilters,
    load_options_history,
    load_options_history_duckdb,
    preprocess_options_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SSVI volatility-surface evolution animation.")
    parser.add_argument("--data-source", choices=["zip", "duckdb"], default="zip")
    parser.add_argument("--zip-path", default="data/options/data/options_XLF_etf.zip")
    parser.add_argument("--db-path", default="data/market_data.duckdb")
    parser.add_argument("--db-table", default="options_enriched")
    parser.add_argument("--ticker", default="XLF")
    parser.add_argument("--start-date", default="2006-01-03")
    parser.add_argument("--end-date", default="2008-12-31")
    parser.add_argument("--max-anchors", type=int, default=36, help="Number of calibrated anchor dates.")
    parser.add_argument("--interpolate-steps", type=int, default=4, help="Interpolated frames between anchors.")
    parser.add_argument("--frame-selection", choices=["all", "sampled"], default="sampled")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--k-min", type=float, default=-0.6)
    parser.add_argument("--k-max", type=float, default=0.6)
    parser.add_argument("--t-min", type=float, default=7.0 / 365.0)
    parser.add_argument("--t-max", type=float, default=1.5)
    parser.add_argument("--calibration-backend", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", default=str(STRATEGY_ROOT / "results" / "current" / "surface_evolution"))
    return parser.parse_args()


def _resolve_data_path(root: Path, p: str) -> str:
    candidate = Path(p)
    if candidate.exists():
        return str(candidate)
    for base in [root, root.parent, Path()]:
        alt = base / p
        if alt.exists():
            return str(alt)
    return str(candidate)


def _select_dates(panel: pd.DataFrame, max_anchors: int, mode: str) -> List[pd.Timestamp]:
    dates = sorted(pd.to_datetime(panel["date"]).dt.normalize().unique())
    if len(dates) <= max_anchors or mode == "all":
        return [pd.Timestamp(d) for d in dates]
    idx = np.linspace(0, len(dates) - 1, max_anchors, dtype=int)
    return [pd.Timestamp(dates[i]) for i in idx]


def _render_surface_png(
    out_path: Path,
    date_label: str,
    k_grid: np.ndarray,
    t_grid: np.ndarray,
    sigma_grid: np.ndarray,
    sample_k: np.ndarray,
    sample_t: np.ndarray,
    sample_sigma: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(k_grid, t_grid, sigma_grid, cmap="viridis", linewidth=0, antialiased=True, alpha=0.88)
    if len(sample_k) > 0:
        step = max(1, len(sample_k) // 1000)
        ax.scatter(sample_k[::step], sample_t[::step], sample_sigma[::step], c="black", s=5, alpha=0.35)
    ax.set_title(f"SSVI Vol Surface Evolution: {date_label}")
    ax.set_xlabel("Log-moneyness k")
    ax.set_ylabel("Time to maturity t (years)")
    ax.set_zlabel("Implied vol sigma")
    fig.colorbar(surf, shrink=0.62, aspect=14, label="Predicted sigma")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    source_label = ""
    if args.data_source == "duckdb":
        db_path = _resolve_data_path(PROJECT_ROOT, args.db_path)
        ticker = args.ticker.upper()
        raw = load_options_history_duckdb(
            db_path=db_path,
            ticker=ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            table=args.db_table,
        )
        source_label = f"duckdb:{db_path}:{args.db_table}:{ticker}"
    else:
        zip_path = _resolve_data_path(PROJECT_ROOT, args.zip_path)
        raw = load_options_history(zip_path=zip_path, start_date=args.start_date, end_date=args.end_date)
        source_label = f"zip:{zip_path}"

    panel = preprocess_options_panel(raw, filters=PanelFilters())
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    use_dates = _select_dates(panel, args.max_anchors, args.frame_selection)
    if not use_dates:
        raise RuntimeError("No dates available for surface evolution.")

    k_lin = np.linspace(args.k_min, args.k_max, 70)
    t_lin = np.linspace(args.t_min, args.t_max, 70)
    kk, tt = np.meshgrid(k_lin, t_lin)

    metrics = []
    anchors = []
    for d in use_dates:
        day = panel[panel["date"] == d]
        if len(day) < 120 or day["t"].round(8).nunique() < 3:
            continue
        k = day["k"].to_numpy(dtype=np.float64)
        t = day["t"].to_numpy(dtype=np.float64)
        sigma = day["sigma"].to_numpy(dtype=np.float64)
        try:
            surf = calibrate_ssvi_surface(
                k=k,
                t=t,
                sigma=sigma,
                random_seed=42,
                n_param_steps=300,
                n_theta_steps=3,
                calibration_backend=args.calibration_backend,
            )
        except Exception:
            continue

        sigma_grid = surf.implied_volatility(kk.reshape(-1), tt.reshape(-1)).reshape(kk.shape)
        anchors.append(
            {
                "date": d,
                "sigma_grid": sigma_grid,
                "sample_k": k,
                "sample_t": t,
                "sample_sigma": sigma,
            }
        )
        metrics.append(
            {
                "date": d.date().isoformat(),
                "n_points": int(len(day)),
                "n_maturities": int(day["t"].round(8).nunique()),
                "rho": float(surf.params.rho),
                "eta": float(surf.params.eta),
                "gamma": float(surf.params.gamma),
                "rmse_total_variance": float(surf.diagnostics.rmse_total_variance),
                "rmse_implied_volatility": float(surf.diagnostics.rmse_implied_volatility),
            }
        )

    if not anchors:
        raise RuntimeError("No frames produced; check filters/date range.")

    frame_paths: List[Path] = []
    frame_i = 0
    interp = max(0, int(args.interpolate_steps))
    for i in range(len(anchors)):
        a = anchors[i]
        # anchor frame
        frame_path = frames_dir / f"frame_{frame_i:04d}_{a['date'].date().isoformat()}.png"
        _render_surface_png(
            out_path=frame_path,
            date_label=a["date"].date().isoformat(),
            k_grid=kk,
            t_grid=tt,
            sigma_grid=a["sigma_grid"],
            sample_k=a["sample_k"],
            sample_t=a["sample_t"],
            sample_sigma=a["sample_sigma"],
        )
        frame_paths.append(frame_path)
        frame_i += 1

        if i + 1 < len(anchors) and interp > 0:
            b = anchors[i + 1]
            for j in range(1, interp + 1):
                w = j / (interp + 1)
                sig = (1.0 - w) * a["sigma_grid"] + w * b["sigma_grid"]
                label = f"{a['date'].date().isoformat()} → {b['date'].date().isoformat()} ({j}/{interp})"
                ip = frames_dir / f"frame_{frame_i:04d}_interp.png"
                _render_surface_png(
                    out_path=ip,
                    date_label=label,
                    k_grid=kk,
                    t_grid=tt,
                    sigma_grid=sig,
                    sample_k=a["sample_k"],
                    sample_t=a["sample_t"],
                    sample_sigma=a["sample_sigma"],
                )
                frame_paths.append(ip)
                frame_i += 1

    gif_path = out_dir / "vol_surface_evolution.gif"
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    duration_ms = int(round(1000.0 / max(1, int(args.gif_fps))))
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics_over_time.csv", index=False)
    summary = {
        "input_source": source_label,
        "data_source": args.data_source,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_anchors": len(anchors),
        "n_frames": len(frame_paths),
        "interpolate_steps": interp,
        "gif_fps": int(args.gif_fps),
        "calibration_backend": args.calibration_backend,
        "output_gif": str(gif_path),
        "metrics_csv": str(out_dir / "metrics_over_time.csv"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
