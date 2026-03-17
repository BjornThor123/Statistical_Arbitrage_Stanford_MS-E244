"""
Cointegration tests on implied-vol skew series.

Two modes:
  1. run_cointegration_tests        — each stock vs the sector ETF (XLF)
  2. run_all_pairs_cointegration    — every possible ticker pair

Usage:
    python -m src.cointegration_test

Loads data/skew.parquet and prints summary tables for both modes.
"""

import itertools
from typing import Literal
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.config import get_config

config = get_config()


def load_skew_pivot(skew_path=None) -> pd.DataFrame:
    """Load skew parquet and return a wide DataFrame (date x ticker)."""
    path = skew_path or config.skew_path
    skew_df = pd.read_parquet(path)
    pivot = (
        skew_df.reset_index()
        .pivot(index="date", columns="ticker", values="skew")
    )
    pivot.columns.name = None
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    return pivot


def _test_pair(
    y: np.ndarray,
    x: np.ndarray,
    method: Literal["engle_granger", "johansen"],
    significance: float,
) -> dict:
    """
    Run one cointegration test on two aligned 1-D arrays.

    Engle-Granger: regresses y on x, tests residuals for a unit root.
    Johansen:      tests the bivariate system [y, x] using the trace statistic
                   at det_order=0 (constant outside the cointegrating relation).
                   Reports the rank-0 trace statistic and its 5% critical value;
                   cointegrated when trace_stat > crit_5pct (reject r=0).

    Returns a dict with keys: t_stat, p_value, critical_1pct, critical_5pct,
    critical_10pct, cointegrated.  For Johansen, p_value is set to NaN because
    statsmodels does not return exact p-values for the trace test.
    """
    if method == "engle_granger":
        t_stat, p_value, crit_values = coint(y, x)
        return {
            "t_stat": t_stat,
            "p_value": p_value,
            "critical_1pct":  crit_values[0],
            "critical_5pct":  crit_values[1],
            "critical_10pct": crit_values[2],
            "cointegrated": p_value < significance,
        }
    else:  # johansen
        data = np.column_stack([y, x])
        res = coint_johansen(data, det_order=0, k_ar_diff=1)
        # Trace statistic for H0: rank=0 (no cointegration)
        trace_stat = res.lr1[0]
        crit_90, crit_95, crit_99 = res.cvt[0]   # cvt rows: [90%, 95%, 99%]
        cointegrated = trace_stat > crit_95
        return {
            "t_stat": trace_stat,   # trace stat plays the role of the test stat
            "p_value": np.nan,      # not provided by statsmodels for Johansen
            "critical_1pct":  crit_99,
            "critical_5pct":  crit_95,
            "critical_10pct": crit_90,
            "cointegrated": cointegrated,
        }


def run_cointegration_tests(
    skew_pivot: pd.DataFrame,
    sector_ticker: str = config.sector_ticker,
    significance: float = 0.05,
    method: Literal["engle_granger", "johansen"] = "engle_granger",
) -> pd.DataFrame:
    """
    Run cointegration test between each stock skew and the sector ETF skew.

    Parameters
    ----------
    method : "engle_granger" | "johansen"
        Engle-Granger tests y ~ x residuals for a unit root (p-value available).
        Johansen tests the bivariate system via the trace statistic (no p-value;
        cointegrated when trace stat exceeds the 5% critical value).

    Returns a DataFrame sorted by test statistic with columns:
        ticker, t_stat, p_value, critical_1pct, critical_5pct, critical_10pct,
        cointegrated, n_obs
    """
    if sector_ticker not in skew_pivot.columns:
        raise ValueError(f"Sector ticker '{sector_ticker}' not found in skew data.")

    sector = skew_pivot[sector_ticker].dropna()
    stock_tickers = [t for t in skew_pivot.columns if t != sector_ticker]

    records = []
    for ticker in stock_tickers:
        stock = skew_pivot[ticker].dropna()
        common = sector.index.intersection(stock.index)

        if len(common) < 20:
            records.append({
                "ticker": ticker,
                "t_stat": np.nan, "p_value": np.nan,
                "critical_1pct": np.nan, "critical_5pct": np.nan,
                "critical_10pct": np.nan,
                "cointegrated": False, "n_obs": len(common),
            })
            continue

        result = _test_pair(
            stock.loc[common].values,
            sector.loc[common].values,
            method=method,
            significance=significance,
        )
        records.append({"ticker": ticker, "n_obs": len(common), **result})

    sort_col = "p_value" if method == "engle_granger" else "t_stat"
    ascending = method == "engle_granger"
    return (
        pd.DataFrame(records)
        .sort_values(sort_col, ascending=ascending)
        .reset_index(drop=True)
    )


def run_all_pairs_cointegration(
    skew_pivot: pd.DataFrame,
    significance: float = 0.05,
    method: Literal["engle_granger", "johansen"] = "engle_granger",
) -> pd.DataFrame:
    """
    Run cointegration test for every unique ticker pair.

    Parameters
    ----------
    method : "engle_granger" | "johansen"
        See run_cointegration_tests for details.

    Tests are symmetric and run once per pair (i < j).
    Returns a DataFrame sorted by test statistic with columns:
        ticker_1, ticker_2, t_stat, p_value, critical_1pct, critical_5pct,
        critical_10pct, cointegrated, n_obs
    """
    tickers = sorted(skew_pivot.columns.tolist())
    records = []

    for t1, t2 in itertools.combinations(tickers, 2):
        s1 = skew_pivot[t1].dropna()
        s2 = skew_pivot[t2].dropna()
        common = s1.index.intersection(s2.index)

        if len(common) < 20:
            records.append({
                "ticker_1": t1, "ticker_2": t2,
                "t_stat": np.nan, "p_value": np.nan,
                "critical_1pct": np.nan, "critical_5pct": np.nan,
                "critical_10pct": np.nan,
                "cointegrated": False, "n_obs": len(common),
            })
            continue

        result = _test_pair(
            s1.loc[common].values,
            s2.loc[common].values,
            method=method,
            significance=significance,
        )
        records.append({"ticker_1": t1, "ticker_2": t2, "n_obs": len(common), **result})

    sort_col = "p_value" if method == "engle_granger" else "t_stat"
    ascending = method == "engle_granger"
    return (
        pd.DataFrame(records)
        .sort_values(sort_col, ascending=ascending)
        .reset_index(drop=True)
    )


def _stat_col_header(method: str) -> str:
    return "T-stat" if method == "engle_granger" else "Trace stat"


def _pval_str(p: float, method: str) -> str:
    if method == "johansen":
        return "N/A"
    return f"{p:.4f}" if not pd.isna(p) else "NaN"


def print_pairs_results(
    results: pd.DataFrame,
    method: Literal["engle_granger", "johansen"] = "engle_granger",
) -> None:
    label = "Engle-Granger" if method == "engle_granger" else "Johansen (trace)"
    print(f"\nCointegration Test ({label}): all ticker pairs")
    print(r"H0: no cointegration")
    print("=" * 80)
    stat_hdr = _stat_col_header(method)
    print(f"{'Pair':<20} {stat_hdr:>10} {'P-value':>10} {'5% crit':>10} {'Coint?':>8} {'N obs':>7}")
    print("-" * 80)
    for _, row in results.iterrows():
        pair = f"{row['ticker_1']} / {row['ticker_2']}"
        flag = "YES" if row["cointegrated"] else "no"
        if pd.isna(row["t_stat"]):
            print(f"{pair:<20} {'NaN':>10} {'NaN':>10} {'NaN':>10} {'N/A':>8} {int(row['n_obs']):>7}")
        else:
            print(
                f"{pair:<20} {row['t_stat']:>10.3f} {_pval_str(row['p_value'], method):>10}"
                f" {row['critical_5pct']:>10.3f} {flag:>8} {int(row['n_obs']):>7}"
            )
    print("-" * 80)
    n_coint = results["cointegrated"].sum()
    print(f"\nCointegrated pairs (p < 0.05): {n_coint} / {len(results)}")


def print_results(
    results: pd.DataFrame,
    sector_ticker: str = config.sector_ticker,
    method: Literal["engle_granger", "johansen"] = "engle_granger",
) -> None:
    label = "Engle-Granger" if method == "engle_granger" else "Johansen (trace)"
    print(f"\nCointegration Test ({label}): each stock skew vs {sector_ticker} skew")
    print(r"H0: no cointegration")
    print("=" * 72)
    stat_hdr = _stat_col_header(method)
    print(f"{'Ticker':<10} {stat_hdr:>10} {'P-value':>10} {'5% crit':>10} {'Coint?':>8} {'N obs':>7}")
    print("-" * 72)
    for _, row in results.iterrows():
        flag = "YES" if row["cointegrated"] else "no"
        if pd.isna(row["t_stat"]):
            print(f"{row['ticker']:<10} {'NaN':>10} {'NaN':>10} {'NaN':>10} {'N/A':>8} {int(row['n_obs']):>7}")
        else:
            print(
                f"{row['ticker']:<10} {row['t_stat']:>10.3f} {_pval_str(row['p_value'], method):>10}"
                f" {row['critical_5pct']:>10.3f} {flag:>8} {int(row['n_obs']):>7}"
            )
    print("-" * 72)
    n_coint = results["cointegrated"].sum()
    print(f"\nCointegrated pairs (p < 0.05): {n_coint} / {len(results)}")


def _fmt_pval(p: float) -> str:
    """Format p-value with significance stars."""
    if pd.isna(p):
        return "---"
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    return f"{p:.4f}{stars}"


def save_sector_latex(
    results: pd.DataFrame,
    sector_ticker: str = config.sector_ticker,
    method: Literal["engle_granger", "johansen"] = "engle_granger",
    out_dir=None,
) -> None:
    """Save the stock-vs-sector cointegration results as a LaTeX table."""
    from pathlib import Path
    out_dir = Path(out_dir or config.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = "Engle-Granger" if method == "engle_granger" else "Johansen trace"
    stat_col = r"$\tau$-stat" if method == "engle_granger" else "Trace stat"
    pval_note = r"Stars: ***$p<0.01$, **$p<0.05$, *$p<0.10$." if method == "engle_granger" else \
        r"Cointegrated when trace stat $>$ 5\% critical value."

    caption = (
        f"{label} cointegration test: each stock IV skew vs "
        f"{sector_ticker} IV skew. "
        r"$H_0$: no cointegration. " + pval_note
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        r"\label{tab:coint_sector}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        rf"Ticker & {stat_col} & $p$-value & Crit.\ 1\% & Crit.\ 5\% & $N$ \\",
        r"\midrule",
    ]

    for _, row in results.iterrows():
        if pd.isna(row["t_stat"]):
            lines.append(
                rf"{row['ticker']} & --- & --- & --- & --- & {int(row['n_obs'])} \\"
            )
        else:
            pval_str = _fmt_pval(row["p_value"]) if method == "engle_granger" else "---"
            lines.append(
                rf"{row['ticker']} & {row['t_stat']:.3f} & {pval_str} "
                rf"& {row['critical_1pct']:.3f} & {row['critical_5pct']:.3f} "
                rf"& {int(row['n_obs'])} \\"
            )

    n_coint = results["cointegrated"].sum()
    lines += [
        r"\bottomrule",
        r"\multicolumn{6}{l}{\textit{Note: } "
        rf"Cointegrated at 5\%: {n_coint}/{len(results)} pairs.}} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = out_dir / "coint_sector.tex"
    path.write_text("\n".join(lines))
    print(f"LaTeX table saved → {path.resolve()}")


def save_pairs_latex(
    results: pd.DataFrame,
    method: Literal["engle_granger", "johansen"] = "engle_granger",
    out_dir=None,
) -> None:
    """Save the all-pairs cointegration results as a LaTeX table."""
    from pathlib import Path
    out_dir = Path(out_dir or config.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = "Engle-Granger" if method == "engle_granger" else "Johansen trace"
    stat_col = r"$\tau$-stat" if method == "engle_granger" else "Trace stat"
    pval_note = r"Stars: ***$p<0.01$, **$p<0.05$, *$p<0.10$." if method == "engle_granger" else \
        r"Cointegrated when trace stat $>$ 5\% critical value."

    caption = (
        f"{label} cointegration test: all unique ticker pairs of IV skew. "
        r"$H_0$: no cointegration. " + pval_note
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        r"\label{tab:coint_pairs}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        rf"Ticker 1 & Ticker 2 & {stat_col} & $p$-value & Crit.\ 1\% & Crit.\ 5\% & $N$ \\",
        r"\midrule",
    ]

    for _, row in results.iterrows():
        if pd.isna(row["t_stat"]):
            lines.append(
                rf"{row['ticker_1']} & {row['ticker_2']} & --- & --- & --- & --- "
                rf"& {int(row['n_obs'])} \\"
            )
        else:
            pval_str = _fmt_pval(row["p_value"]) if method == "engle_granger" else "---"
            lines.append(
                rf"{row['ticker_1']} & {row['ticker_2']} & {row['t_stat']:.3f} "
                rf"& {pval_str} "
                rf"& {row['critical_1pct']:.3f} & {row['critical_5pct']:.3f} "
                rf"& {int(row['n_obs'])} \\"
            )

    n_coint = results["cointegrated"].sum()
    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\textit{Note: } "
        rf"Cointegrated at 5\%: {n_coint}/{len(results)} pairs.}} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]

    path = out_dir / "coint_pairs.tex"
    path.write_text("\n".join(lines))
    print(f"LaTeX table saved → {path.resolve()}")


def main(method: Literal["engle_granger", "johansen"] = "engle_granger"):
    print("Loading skew data...")
    skew_pivot = load_skew_pivot()
    print(f"Loaded: {skew_pivot.shape[1]} tickers, {len(skew_pivot)} dates "
          f"({skew_pivot.index.min().date()} → {skew_pivot.index.max().date()})")
    print(f"Tickers: {sorted(skew_pivot.columns.tolist())}")
    print(f"Method:  {method}\n")

    sector_results = run_cointegration_tests(skew_pivot, method=method)
    print_results(sector_results, method=method)
    save_sector_latex(sector_results, method=method)

    print()
    pairs_results = run_all_pairs_cointegration(skew_pivot, method=method)
    print_pairs_results(pairs_results, method=method)
    save_pairs_latex(pairs_results, method=method)

    return {"sector_vs_stock": sector_results, "all_pairs": pairs_results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run cointegration tests on skew series.')
    parser.add_argument(
        '--method',
        choices=["engle_granger", "johansen"],
        default="engle_granger",
        help='Cointegration test method: engle_granger or johansen',
    )
    args = parser.parse_args()
    main(method=args.method)
