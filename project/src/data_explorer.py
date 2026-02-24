from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from data_loader import DataLoader, DB_FILENAME


class DataExplorer:
    """Exploratory data analysis on the DuckDB market data tables."""

    def __init__(self, data_path: str, output_dir: Optional[str] = None):
        self.data_path = Path(data_path)
        self.db_path = self.data_path / DB_FILENAME
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. Run DataLoader.build_all() first."
            )
        self.con = duckdb.connect(str(self.db_path), read_only=True)
        self.output_dir = Path(output_dir) if output_dir else self.data_path / "eda"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _q(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).fetchdf()

    @property
    def tables(self) -> List[str]:
        return self._q("SHOW TABLES")["name"].tolist()

    # ── per-table summary ────────────────────────────────────────────

    def summary(self, table: str) -> pd.DataFrame:
        """Return row count, column count, and total missing values for a table."""
        schema = self._q(f"DESCRIBE {table}")
        cols = schema["column_name"].tolist()
        n_rows = self._q(f"SELECT count(*) AS n FROM {table}")["n"][0]

        # Count nulls + empty strings per column (data is all VARCHAR)
        parts = []
        for c in cols:
            parts.append(
                f"SUM(CASE WHEN \"{c}\" IS NULL OR TRIM(\"{c}\") = '' THEN 1 ELSE 0 END) AS \"{c}\""
            )
        null_counts = self._q(f"SELECT {', '.join(parts)} FROM {table}")
        null_series = null_counts.iloc[0]

        df = pd.DataFrame({
            "column": cols,
            "missing_count": null_series.values.astype(int),
            "missing_pct": (null_series.values.astype(float) / n_rows * 100).round(2),
        })
        df.attrs["n_rows"] = int(n_rows)
        df.attrs["n_cols"] = len(cols)
        return df

    def print_summary(self, table: str) -> None:
        df = self.summary(table)
        print(f"\n{'=' * 60}")
        print(f"Table: {table}  |  Rows: {df.attrs['n_rows']:,}  |  Cols: {df.attrs['n_cols']}")
        print(f"{'=' * 60}")
        print(df.to_string(index=False))

    # ── missing-value heatmap ────────────────────────────────────────

    def missing_heatmap(self, table: str, sample_n: int = 5_000, save: bool = True) -> plt.Figure:
        """Heatmap showing missing values across a random sample of rows."""
        df = self._q(f"SELECT * FROM {table} USING SAMPLE {sample_n}")
        # Mark missing: NULL or empty string
        is_missing = df.isin(["", None]) | df.isna()

        fig, ax = plt.subplots(figsize=(max(14, len(df.columns) * 0.4), 8))
        sns.heatmap(
            is_missing.T.astype(int),
            cbar=False,
            yticklabels=True,
            xticklabels=False,
            cmap=["#e8e8e8", "#d62728"],
            ax=ax,
        )
        ax.set_title(f"Missing Values — {table} (sample n={sample_n:,})")
        ax.set_xlabel("Row index (sampled)")
        ax.set_ylabel("")
        fig.tight_layout()

        if save:
            path = self.output_dir / f"missing_heatmap_{table}.png"
            fig.savefig(path, dpi=150)
            print(f"Saved {path}")
        return fig

    # ── missing by column bar chart ──────────────────────────────────

    def missing_bar(self, table: str, save: bool = True) -> plt.Figure:
        """Bar chart of missing-value percentage per column."""
        df = self.summary(table)
        df = df[df["missing_pct"] > 0].sort_values("missing_pct", ascending=True)

        if df.empty:
            print(f"No missing values in {table}.")
            fig, ax = plt.subplots()
            ax.set_title(f"No missing values — {table}")
            return fig

        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
        ax.barh(df["column"], df["missing_pct"], color="#d62728")
        ax.set_xlabel("Missing %")
        ax.set_title(f"Missing Values by Column — {table}")
        for i, (pct, cnt) in enumerate(zip(df["missing_pct"], df["missing_count"])):
            ax.text(pct + 0.5, i, f"{pct:.1f}% ({cnt:,})", va="center", fontsize=8)
        fig.tight_layout()

        if save:
            path = self.output_dir / f"missing_bar_{table}.png"
            fig.savefig(path, dpi=150)
            print(f"Saved {path}")
        return fig

    # ── row count over time ──────────────────────────────────────────

    def row_count_over_time(
        self, table: str, date_col: str = "date", save: bool = True
    ) -> plt.Figure:
        """Line plot of row counts per month."""
        df = self._q(f"""
            SELECT DATE_TRUNC('month', CAST("{date_col}" AS DATE)) AS month,
                   COUNT(*) AS n
            FROM {table}
            WHERE "{date_col}" IS NOT NULL AND TRIM("{date_col}") != ''
            GROUP BY month ORDER BY month
        """)
        df["month"] = pd.to_datetime(df["month"])

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["month"], df["n"], linewidth=0.8)
        ax.set_title(f"Row Count per Month — {table}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rows")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()

        if save:
            path = self.output_dir / f"rows_over_time_{table}.png"
            fig.savefig(path, dpi=150)
            print(f"Saved {path}")
        return fig

    # ── source_label distribution ────────────────────────────────────

    def source_label_distribution(self, table: str, save: bool = True) -> plt.Figure:
        """Bar chart of row counts per source_label."""
        df = self._q(f"""
            SELECT source_label, COUNT(*) AS n
            FROM {table}
            GROUP BY source_label
            ORDER BY n DESC
        """)

        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6), 5))
        ax.bar(df["source_label"], df["n"], color=sns.color_palette("muted", len(df)))
        ax.set_title(f"Rows per Source Label — {table}")
        ax.set_ylabel("Row count")
        ax.set_xlabel("source_label")
        plt.xticks(rotation=45, ha="right")
        for i, (label, n) in enumerate(zip(df["source_label"], df["n"])):
            ax.text(i, n, f"{n:,}", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()

        if save:
            path = self.output_dir / f"source_label_dist_{table}.png"
            fig.savefig(path, dpi=150)
            print(f"Saved {path}")
        return fig

    # ── numeric column distributions ─────────────────────────────────

    def numeric_histograms(
        self,
        table: str,
        columns: List[str],
        sample_n: int = 100_000,
        save: bool = True,
    ) -> plt.Figure:
        """Histogram grid for numeric columns (cast from VARCHAR)."""
        cols_sql = ", ".join(f'TRY_CAST("{c}" AS DOUBLE) AS "{c}"' for c in columns)
        df = self._q(
            f"SELECT {cols_sql} FROM {table} USING SAMPLE {sample_n}"
        )

        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten() if len(columns) > 1 else [axes]

        for ax, col in zip(axes, columns):
            series = df[col].dropna()
            if series.empty:
                ax.set_title(f"{col} (no data)")
                continue
            ax.hist(series, bins=80, edgecolor="none", alpha=0.8)
            ax.set_title(col)
            ax.set_ylabel("Count")

        # Hide empty subplots
        for ax in axes[len(columns):]:
            ax.set_visible(False)

        fig.suptitle(f"Distributions — {table} (sample n={sample_n:,})", y=1.02)
        fig.tight_layout()

        if save:
            path = self.output_dir / f"histograms_{table}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved {path}")
        return fig

    # ── correlation heatmap ──────────────────────────────────────────

    def correlation_heatmap(
        self,
        table: str,
        columns: List[str],
        sample_n: int = 100_000,
        save: bool = True,
    ) -> plt.Figure:
        """Correlation matrix heatmap for numeric columns."""
        cols_sql = ", ".join(f'TRY_CAST("{c}" AS DOUBLE) AS "{c}"' for c in columns)
        df = self._q(
            f"SELECT {cols_sql} FROM {table} USING SAMPLE {sample_n}"
        )
        corr = df.corr()

        fig, ax = plt.subplots(figsize=(max(8, len(columns) * 0.8), max(6, len(columns) * 0.7)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr,
            mask=mask,
            annot=len(columns) <= 15,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
        )
        ax.set_title(f"Correlation — {table}")
        fig.tight_layout()

        if save:
            path = self.output_dir / f"correlation_{table}.png"
            fig.savefig(path, dpi=150)
            print(f"Saved {path}")
        return fig

    # ── duplicate analysis ───────────────────────────────────────────

    def duplicate_check(self, table: str, key_cols: List[str]) -> pd.DataFrame:
        """Check for duplicate rows based on key columns."""
        key_str = ", ".join(f'"{c}"' for c in key_cols)
        df = self._q(f"""
            SELECT {key_str}, COUNT(*) AS dup_count
            FROM {table}
            GROUP BY {key_str}
            HAVING COUNT(*) > 1
            ORDER BY dup_count DESC
            LIMIT 20
        """)
        total_dups = self._q(f"""
            SELECT SUM(c) AS total FROM (
                SELECT COUNT(*) - 1 AS c FROM {table}
                GROUP BY {key_str} HAVING COUNT(*) > 1
            )
        """)["total"][0]
        print(f"Duplicate rows (by {key_cols}): {int(total_dups or 0):,}")
        return df

    # ── run all ──────────────────────────────────────────────────────

    def run_all(self) -> None:
        """Run a standard EDA suite on all tables."""
        plt.switch_backend("Agg")

        OPTIONS_NUMERIC = [
            "strike_price", "best_bid", "best_offer", "volume",
            "open_interest", "impl_volatility", "delta", "gamma", "vega", "theta",
        ]
        EQUITIES_NUMERIC = [
            "PRC", "VOL", "RET", "BID", "ASK", "SHROUT",
            "BIDLO", "ASKHI", "OPENPRC", "VOLUSD",
        ]

        for table in self.tables:
            self.print_summary(table)
            self.missing_bar(table)
            self.missing_heatmap(table)

        # Time series & source labels for options/equities
        for table in ["options", "equities"]:
            if table in self.tables:
                self.row_count_over_time(table)
                self.source_label_distribution(table)

        # Numeric distributions & correlations
        if "options" in self.tables:
            self.numeric_histograms("options", OPTIONS_NUMERIC)
            self.correlation_heatmap("options", OPTIONS_NUMERIC)

        if "equities" in self.tables:
            self.numeric_histograms("equities", EQUITIES_NUMERIC)
            self.correlation_heatmap("equities", EQUITIES_NUMERIC)

        plt.close("all")
        print(f"\nAll plots saved to {self.output_dir}")


if __name__ == "__main__":
    data_path = "/Users/bjorn/Documents/Skóli/Stanford/Skóli/Q2/StatArb/Statistical_Arbitrage_Stanford_MS-E244/project/data"

    with DataExplorer(data_path) as explorer:
        explorer.run_all()
