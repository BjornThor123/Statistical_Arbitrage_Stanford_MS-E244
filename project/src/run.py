import json
from pathlib import Path

import pandas as pd
from rich import print as rprint
from rich.table import Table
from rich.console import Console

from src.config import get_config
from src.pairs_trading_skew import run_strategy, run_backtest
from src.data_cleaning.extract_skew import extract_skew_df

console = Console()

METRICS_TO_STORE = [
    "Gross Sharpe Ratio",
    "Gross Ann. Return",
    "Gross Max Drawdown",
    "Net Sharpe Ratio",
    "Net Ann. Return",
    "Net Max Drawdown",
    "Total Transaction Cost",
]

# Base parameter values (used when not the varying dimension)
BASE_DELTA_TARGET         = 0.25
BASE_TTE_TARGET           = 15
BASE_TXN_COST_BPS         = 20
BASE_ENTRY_THRESHOLD      = 0.95
BASE_ENTRY_THRESHOLD_MODE = "absolute"
BASE_SKEW_METHOD          = "direct"


def _run_one(
    df: pd.DataFrame,
    plot_dir: Path,
    delta_target: float       = BASE_DELTA_TARGET,
    tte_target: int           = BASE_TTE_TARGET,
    txn_cost_bps: float       = BASE_TXN_COST_BPS,
    entry_threshold: float    = BASE_ENTRY_THRESHOLD,
    entry_threshold_mode: str = BASE_ENTRY_THRESHOLD_MODE,
    skew_path: Path | None    = None,
) -> dict:
    """Run strategy + backtest for one parameter combination."""
    kwargs = dict(
        tte_days=tte_target,
        delta_target=delta_target,
        entry_threshold_mode=entry_threshold_mode,
        entry_threshold=entry_threshold,
    )
    if skew_path is not None:
        kwargs["skew_path"] = skew_path
    results = run_strategy(df, **kwargs)
    backtest = run_backtest(
        signals=results["signals"],
        betas=results["betas"],
        pairs=results["pairs"],
        stock_rr_legs=results["stock_rr_legs"],
        z_scores=results["z_scores"],
        spread_df=results["spread_df"],
        transaction_cost_bps=txn_cost_bps,
        plot_dir=plot_dir,
    )
    return {m: backtest["metrics"].get(m) for m in METRICS_TO_STORE}


def _save_sensitivity_table(results: dict, label: str, save_dir: Path) -> None:
    """Save sensitivity results as CSV + JSON and pretty-print to console."""
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = [{"param": k, **v} for k, v in results.items()]
    df_res = pd.DataFrame(rows).set_index("param")

    pct_cols = [c for c in df_res.columns if any(k in c for k in ("Return", "Drawdown"))]

    csv_path = save_dir / f"{label}_sensitivity.csv"
    df_res.to_csv(csv_path)
    rprint(f"Saved → {csv_path}")

    json_path = save_dir / f"{label}_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    table = Table(title=f"Sensitivity: {label}", show_lines=True)
    table.add_column("Param", style="bold")
    for col in df_res.columns:
        table.add_column(col, justify="right")

    for idx, row in df_res.iterrows():
        cells = [str(idx)]
        for col, val in row.items():
            if val is None:
                cells.append("N/A")
            elif col in pct_cols:
                cells.append(f"{val:.2%}")
            elif "Sharpe" in col or "Ratio" in col:
                cells.append(f"{val:.3f}")
            else:
                cells.append(f"{val:.6f}")
        table.add_row(*cells)

    console.print(table)


def main():
    base_config = get_config()
    df = pd.read_parquet(base_config.cleaned_options_path)

    sens_root = base_config.plot_dir / "sensitivity"

    # ── 1. Delta sensitivity ──────────────────────────────────────────────────
    delta_targets = [0.15, 0.25, 0.50]
    console.rule("[bold blue]Sensitivity: delta_target")
    results_delta = {}
    for delta in delta_targets:
        rprint(f"\n[cyan]>>> delta_target = {delta}[/cyan]")
        results_delta[delta] = _run_one(
            df,
            plot_dir=sens_root / f"delta_{delta}",
            delta_target=delta,
        )
    _save_sensitivity_table(results_delta, "delta", sens_root)

    # ── 2. TTE sensitivity ────────────────────────────────────────────────────
    tte_targets = [15, 20, 25]
    console.rule("[bold blue]Sensitivity: tte_target")
    results_tte = {}
    for tte in tte_targets:
        rprint(f"\n[cyan]>>> tte_target = {tte}[/cyan]")
        results_tte[tte] = _run_one(
            df,
            plot_dir=sens_root / f"tte_{tte}",
            tte_target=tte,
        )
    _save_sensitivity_table(results_tte, "tte", sens_root)

    # ── 3. Transaction cost sensitivity ──────────────────────────────────────
    txn_cost_bps_list = [0, 20, 40]
    console.rule("[bold blue]Sensitivity: transaction_cost_bps")
    results_txn = {}
    for txn_bps in txn_cost_bps_list:
        rprint(f"\n[cyan]>>> transaction_cost_bps = {txn_bps}[/cyan]")
        results_txn[txn_bps] = _run_one(
            df,
            plot_dir=sens_root / f"txn_{txn_bps}bps",
            txn_cost_bps=txn_bps,
        )
    _save_sensitivity_table(results_txn, "txn_cost", sens_root)

    # ── 4. Entry threshold sensitivity ───────────────────────────────────────
    entry_thresholds = [0.95, 0.975, 0.99]
    console.rule("[bold blue]Sensitivity: entry_threshold (percentile)")
    results_entry = {}
    for thresh in entry_thresholds:
        rprint(f"\n[cyan]>>> entry_threshold = {thresh}[/cyan]")
        results_entry[thresh] = _run_one(
            df,
            plot_dir=sens_root / f"entry_{thresh}",
            entry_threshold=thresh,
            entry_threshold_mode="percentile",
        )
    _save_sensitivity_table(results_entry, "entry_threshold", sens_root)

    # ── 5. Skew method sensitivity ────────────────────────────────────────────
    skew_methods = ["direct", "polynomial", "naive"]
    console.rule("[bold blue]Sensitivity: skew_method")
    results_skew = {}
    for method in skew_methods:
        rprint(f"\n[cyan]>>> skew_method = {method}[/cyan]")
        skew_cache_path = base_config.data_path / f"skew_{method}.parquet"
        if not skew_cache_path.exists():
            rprint(f"[yellow]Computing skew ({method})...[/yellow]")
            skew_df, _ = extract_skew_df(
                df,
                tte_days=BASE_TTE_TARGET,
                delta_target=BASE_DELTA_TARGET,
                skew_method=method,
                verbose=False,
            )
            skew_df.to_parquet(skew_cache_path)
            rprint(f"Skew cached → {skew_cache_path}")
        results_skew[method] = _run_one(
            df,
            plot_dir=sens_root / f"skew_{method}",
            skew_path=skew_cache_path,
        )
    _save_sensitivity_table(results_skew, "skew_method", sens_root)

    # ── Summary ───────────────────────────────────────────────────────────────
    all_results = {
        "delta":           results_delta,
        "tte":             results_tte,
        "txn_cost_bps":    results_txn,
        "entry_threshold": results_entry,
        "skew_method":     results_skew,
    }
    summary_path = sens_root / "sensitivity_summary.json"
    sens_root.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {dim: {str(k): v for k, v in vals.items()} for dim, vals in all_results.items()},
            f, indent=2,
        )
    rprint(f"\n[green]Full sensitivity summary saved → {summary_path.resolve()}[/green]")


if __name__ == '__main__':
    main()
