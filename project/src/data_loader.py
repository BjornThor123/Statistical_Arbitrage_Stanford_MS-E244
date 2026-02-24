import time
import logging
import zipfile
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DB_FILENAME = "market_data.duckdb"


def timer(func):
    """Decorator to time function execution if verbose mode is on."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.verbose:
            start_time = time.time()
            result = func(self, *args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} took {elapsed:.2f}s")
            print(f"{func.__name__} took {elapsed:.2f}s")
        else:
            result = func(self, *args, **kwargs)
        return result
    return wrapper


class DataLoader:
    def __init__(self, data_path: str, verbose: bool = False):
        self.data_path = Path(data_path)
        self.options_path = self.data_path / "options"
        self.options_data_path = self.options_path / "data"

        self.equities_path = self.data_path / "equities"
        self.equities_data_path = self.equities_path / "data"

        self.rf_path = self.data_path / "risk_free"
        self.rf_data_path = self.rf_path / "data"

        self.verbose = verbose
        self.db_path = self.data_path / DB_FILENAME
        self.con = duckdb.connect(str(self.db_path))

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── helpers ──────────────────────────────────────────────────────

    def _unzip(self, *zip_files: Path, extract_to: Path) -> None:
        def extract_single(zip_file: Path):
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            except zipfile.BadZipFile:
                logger.error(f"Corrupt zip file: {zip_file}")
                raise

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(extract_single, zf): zf for zf in zip_files}
            for future in as_completed(futures):
                future.result()

    def _ensure_unzipped(self, data_path: Path) -> None:
        """Unzip all zip files and rename extensionless files to .csv."""
        zip_files = list(data_path.glob("*.zip"))
        if zip_files:
            self._unzip(*zip_files, extract_to=data_path)

        for f in data_path.iterdir():
            if f.is_file() and f.suffix == '' and not f.name.startswith('.'):
                f.rename(f.with_suffix('.csv'))

    @staticmethod
    def _extract_options_label(filename: str) -> str:
        """options_BAC.zip -> BAC, options_XLF_etf.csv -> XLF_etf"""
        name = Path(filename).stem  # strip .zip / .csv
        if name.startswith("options_"):
            return name[len("options_"):]
        return name

    @staticmethod
    def _extract_equities_label(filename: str) -> str:
        """crsp_daily_agricultural.csv.gz -> agricultural"""
        name = filename
        for suffix in ('.gz', '.csv'):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        if name.startswith("crsp_daily_"):
            return name[len("crsp_daily_"):]
        return name

    def _table_exists(self, table_name: str) -> bool:
        result = self.con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return result[0] > 0

    # ── ingestion ────────────────────────────────────────────────────

    def _collect_options_files(self) -> List[tuple[Path, str]]:
        """Unzip options zips and return list of (csv_path, label) pairs.

        The label comes from the zip filename (options_XXX.zip -> XXX),
        since extracted files may have unrelated names.
        """
        pairs = []

        # Process zip files: extract and map label from zip name
        for zip_file in self.options_data_path.glob("*.zip"):
            label = self._extract_options_label(zip_file.name)
            with zipfile.ZipFile(zip_file, 'r') as zf:
                for member in zf.namelist():
                    extracted = self.options_data_path / member
                    if not extracted.exists():
                        zf.extract(member, self.options_data_path)
                    # Rename extensionless files to .csv
                    if extracted.suffix == '':
                        csv_path = extracted.with_suffix('.csv')
                        if not csv_path.exists():
                            extracted.rename(csv_path)
                        extracted = csv_path
                    pairs.append((extracted, label))

        # Also pick up any loose CSV files that aren't from zips
        zip_extracted = {p for p, _ in pairs}
        for f in self.options_data_path.iterdir():
            if (f.is_file()
                and f.name.endswith(('.csv', '.csv.gz'))
                and f.name not in ("all_unzipped.csv",)
                and f not in zip_extracted):
                pairs.append((f, self._extract_options_label(f.name)))

        return pairs

    @timer
    def build_options_table(self) -> None:
        """Ingest all options data files into the 'options' DuckDB table with a source_label column."""
        if self._table_exists("options"):
            if self.verbose:
                print("Table 'options' already exists, skipping.")
            return

        file_label_pairs = self._collect_options_files()
        if not file_label_pairs:
            raise FileNotFoundError(f"No data files found in {self.options_data_path}")

        if self.verbose:
            print(f"Ingesting {len(file_label_pairs)} options files into DuckDB...")

        first = True
        for data_file, label in file_label_pairs:
            escaped_path = str(data_file).replace("'", "''")
            if first:
                self.con.execute(f"""
                    CREATE TABLE options AS
                    SELECT *, '{label}' AS source_label
                    FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                """)
                first = False
            else:
                self.con.execute(f"""
                    INSERT INTO options
                    SELECT *, '{label}' AS source_label
                    FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                """)

        count = self.con.execute("SELECT count(*) FROM options").fetchone()[0]
        if self.verbose:
            print(f"Options table: {count:,} rows")

    @timer
    def build_equities_table(self) -> None:
        """Ingest all equities data files into the 'equities' DuckDB table with a source_label column."""
        if self._table_exists("equities"):
            if self.verbose:
                print("Table 'equities' already exists, skipping.")
            return

        data_files = [
            f for f in self.equities_data_path.iterdir()
            if f.is_file()
            and f.name.endswith(('.csv', '.csv.gz'))
            and f.name not in ("all_unzipped.csv", "crsp_daily_all.csv")
        ]
        if not data_files:
            raise FileNotFoundError(f"No data files found in {self.equities_data_path}")

        if self.verbose:
            print(f"Ingesting {len(data_files)} equities files into DuckDB...")

        first = True
        for data_file in data_files:
            label = self._extract_equities_label(data_file.name)
            escaped_path = str(data_file).replace("'", "''")
            if first:
                self.con.execute(f"""
                    CREATE TABLE equities AS
                    SELECT *, '{label}' AS source_label
                    FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                """)
                first = False
            else:
                self.con.execute(f"""
                    INSERT INTO equities
                    SELECT *, '{label}' AS source_label
                    FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                """)

        count = self.con.execute("SELECT count(*) FROM equities").fetchone()[0]
        if self.verbose:
            print(f"Equities table: {count:,} rows")

    @timer
    def build_rf_table(self) -> None:
        """Ingest risk-free rate data into the 'risk_free' DuckDB table."""
        if self._table_exists("risk_free"):
            if self.verbose:
                print("Table 'risk_free' already exists, skipping.")
            return

        data_files = [
            f for f in self.rf_data_path.iterdir()
            if f.is_file() and f.name.endswith(('.csv', '.csv.gz'))
        ]
        if not data_files:
            raise FileNotFoundError(f"No data files found in {self.rf_data_path}")

        if self.verbose:
            print(f"Ingesting {len(data_files)} risk-free files into DuckDB...")

        first = True
        for data_file in data_files:
            escaped_path = str(data_file).replace("'", "''")
            if first:
                self.con.execute(f"""
                    CREATE TABLE risk_free AS
                    SELECT * FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                """)
                first = False
            else:
                self.con.execute(f"""
                    INSERT INTO risk_free
                    SELECT * FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                """)

        count = self.con.execute("SELECT count(*) FROM risk_free").fetchone()[0]
        if self.verbose:
            print(f"Risk-free table: {count:,} rows")

    def build_all(self) -> None:
        """Build all tables."""
        self.build_options_table()
        self.build_equities_table()
        self.build_rf_table()

    # ── query helpers ────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        """Run arbitrary SQL and return a DataFrame."""
        return self.con.execute(sql).fetchdf()

    def tables(self) -> List[str]:
        """List all tables in the database."""
        return self.con.execute("SHOW TABLES").fetchdf()["name"].tolist()

    def describe(self, table_name: str) -> pd.DataFrame:
        """Describe a table's schema."""
        return self.con.execute(f"DESCRIBE {table_name}").fetchdf()


if __name__ == "__main__":
    data_path = "/Users/bjorn/Documents/Skóli/Stanford/Skóli/Q2/StatArb/Statistical_Arbitrage_Stanford_MS-E244/project/data"

    with DataLoader(data_path, verbose=True) as dl:
        dl.build_all()

        print("\nTables:", dl.tables())
        print("\nOptions schema:")
        print(dl.describe("options"))
        print("\nEquities schema:")
        print(dl.describe("equities"))

        print("\nOptions source labels:")
        print(dl.query("SELECT DISTINCT source_label FROM options"))
        print("\nEquities source labels:")
        print(dl.query("SELECT DISTINCT source_label FROM equities"))
