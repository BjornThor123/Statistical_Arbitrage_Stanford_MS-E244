import time
import logging
import zipfile
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict

import duckdb
import numpy as np
import pandas as pd
from pandas.errors import ParserError

logger = logging.getLogger(__name__)


# Map metadata Type strings to DuckDB types (case-insensitive)
METADATA_TYPE_MAP = {
    "character": "VARCHAR",
    "char": "VARCHAR",
    "varchar": "VARCHAR",
    "string": "VARCHAR",
    "text": "VARCHAR",
    "numeric": "DOUBLE",
    "double": "DOUBLE",
    "float": "DOUBLE",
    "real": "DOUBLE",
    "decimal": "DOUBLE",
    "integer": "INTEGER",
    "int": "INTEGER",
    "number": "INTEGER",
    "bigint": "BIGINT",
    "long": "BIGINT",
    "date": "DATE",
    "datetime": "DATE",
    "timestamp": "DATE",
}

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
        self.options_metadata_path = self.options_path / "metadata"

        self.equities_path = self.data_path / "equities"
        self.equities_data_path = self.equities_path / "data"
        self.equities_metadata_path = self.equities_path / "metadata"

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

    def _quoted(self, col: str) -> str:
        """Quote column name for SQL (handles reserved words like 'class')."""
        return f'"{col}"'

    def _find_metadata_file(self, metadata_dir: Path, data_dir: Path) -> Optional[Path]:
        """Find metadata CSV in metadata dir or recursively in data dir.
        Metadata must have 'Variable Name' (or 'VariableName') and 'Type' columns.
        """
        for search_dir in (metadata_dir, data_dir):
            if not search_dir.exists():
                continue
            for pattern in ("*dictionary*.csv", "*metadata*.csv", "*.csv"):
                for p in search_dir.rglob(pattern):
                    if p.is_file() and "sp500" not in p.name.lower():
                        try:
                            df = pd.read_csv(p, nrows=1)
                            cols = [c.strip() for c in df.columns]
                            name_col = next(
                                (c for c in cols if "variable" in c.lower() and "name" in c.lower()),
                                None,
                            )
                            type_col = next((c for c in cols if c.lower() == "type"), None)
                            if name_col and type_col:
                                return p
                        except Exception:
                            continue
        return None

    def _load_metadata_schema(self, metadata_path: Path) -> Dict[str, str]:
        """Load column name -> DuckDB type from metadata CSV.
        Metadata has columns: Variable Name, Type, Description.
        """
        df = pd.read_csv(metadata_path)
        name_col = next(
            (c for c in df.columns if "variable" in c.lower() and "name" in c.lower()),
            df.columns[0],
        )
        type_col = next((c for c in df.columns if c.strip().lower() == "type"), None)
        if type_col is None:
            return {}

        schema = {}
        for _, row in df.iterrows():
            var_name = str(row[name_col]).strip()
            if pd.isna(var_name) or var_name == "":
                continue
            raw_type = str(row[type_col]).strip().lower() if pd.notna(row[type_col]) else ""
            # Normalize: "number(10,2)" -> "number", "character(10)" -> "character"
            base_type = raw_type.split("(")[0].split()[0] if raw_type else ""
            duck_type = METADATA_TYPE_MAP.get(base_type, "VARCHAR")
            schema[var_name] = duck_type

        return schema

    def _build_select_with_schema(
            self,
            csv_path: str,
            column_types: Dict[str, str],
            column_order: Optional[List[str]] = None,
            *,
            csv_columns: Optional[set[str]] = None,
            source_label: Optional[str] = None,
        ) -> tuple[str, List[str]]:
        """Build SELECT clause with type casts from CSV.

        Returns (select_sql, column_order). Pass column_order to subsequent
        calls so INSERT uses the same column order as the table.
        If source_label is set, adds that column (for options/equities).

        We use all_varchar in read_csv to read raw strings, then TRY_CAST in
        our SELECT. This avoids DuckDB auto-detect inferring VARCHAR for columns
        with mixed/bad data. TRY_CAST converts invalid values to NULL.
        """
        
        if csv_columns is None:
            escaped = str(csv_path).replace("'", "''")
            cols_df = self.con.execute(f"""
                SELECT * FROM read_csv('{escaped}', auto_detect=true, all_varchar=true)
                LIMIT 0
            """).fetchdf()
            csv_columns_list = cols_df.columns.tolist()
            csv_columns = set(csv_columns_list)
        else:
            csv_columns_list = list(csv_columns)
        order = column_order if column_order is not None else csv_columns_list

        # Case-insensitive lookup: metadata may have "PRC", CSV may have "prc"
        type_lookup = {k.lower(): v for k, v in column_types.items()}
        select_parts = []
        for col in order:
            if col == "source_label":
                if source_label is not None:
                    select_parts.append(f"'{source_label}' AS {self._quoted('source_label')}")
                continue
            q = self._quoted(col)
            target_type = column_types.get(col) or type_lookup.get(col.lower(), "VARCHAR")
            if col not in csv_columns:
                select_parts.append(f"CAST(NULL AS {target_type}) AS {q}")
            elif target_type == "VARCHAR":
                select_parts.append(f"raw.{q} AS {q}")
            else:
                select_parts.append(f"TRY_CAST(raw.{q} AS {target_type}) AS {q}")

        if source_label is not None and "source_label" not in order:
            select_parts.append(f"'{source_label}' AS {self._quoted('source_label')}")
        table_order = order if (source_label is None or "source_label" in order) else order + ["source_label"]
        return ",\n        ".join(select_parts), table_order

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
        """Ingest all options data files into the 'options' DuckDB table.

        Column types are read from metadata (options/metadata/*.csv or
        options/data/**/*dictionary*.csv). Metadata must have columns:
        Variable Name, Type, Description.
        """
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        # Some validation 
        if self._table_exists("options"):
            if self.verbose:
                print("Table 'options' already exists, skipping.")
            return

        t_meta = time.perf_counter()
        metadata_path = self._find_metadata_file(
            self.options_metadata_path, self.options_data_path
        )
        column_types = self._load_metadata_schema(metadata_path) if metadata_path else {}
        timings["metadata"] = time.perf_counter() - t_meta
        if not column_types and self.verbose:
            print("No options metadata found; all columns will be VARCHAR.")

        # Collecting data
        t_collect = time.perf_counter()
        file_label_pairs = self._collect_options_files()
        timings["collect_files"] = time.perf_counter() - t_collect
        if not file_label_pairs:
            raise FileNotFoundError(f"No data files found in {self.options_data_path}")

        if self.verbose:
            print(f"Ingesting {len(file_label_pairs)} options files into DuckDB...")

        # First pass: collect union of all columns across all files
        # and cache per-file headers so we do not introspect each file twice.
        t_schema = time.perf_counter()
        all_columns = []
        seen = set()
        file_columns: Dict[Path, set[str]] = {}
        for data_file, _ in file_label_pairs:
            escaped = str(data_file).replace("'", "''")
            cols = self.con.execute(f"""
                SELECT * FROM read_csv('{escaped}', auto_detect=true, all_varchar=true) LIMIT 0
            """).fetchdf().columns.tolist()
            file_columns[data_file] = set(cols)
            for c in cols:
                if c not in seen:
                    all_columns.append(c)
                    seen.add(c)
        timings["schema_union"] = time.perf_counter() - t_schema

        t_ingest = time.perf_counter()
        column_order = all_columns
        for i, (data_file, label) in enumerate(file_label_pairs):
            escaped_path = str(data_file).replace("'", "''")
            select_clause, column_order = self._build_select_with_schema(
                str(data_file),
                column_types,
                column_order,
                csv_columns=file_columns.get(data_file),
                source_label=label,
            )
            if i == 0:
                self.con.execute(f"""
                    CREATE TABLE options AS
                    SELECT
                        {select_clause}
                    FROM (
                        SELECT * FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                    ) AS raw
                """)
            else:
                self.con.execute(f"""
                    INSERT INTO options
                    SELECT
                        {select_clause}
                    FROM (
                        SELECT * FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                    ) AS raw
                """)
        timings["ingest"] = time.perf_counter() - t_ingest

        t_derived = time.perf_counter()
        self.con.execute("ALTER TABLE options ADD COLUMN strike DOUBLE")
        self.con.execute("ALTER TABLE options ADD COLUMN mid_price DOUBLE")
        self.con.execute("ALTER TABLE options ADD COLUMN spread DOUBLE")
        self.con.execute("ALTER TABLE options ADD COLUMN tte_days INTEGER")
        self.con.execute("ALTER TABLE options ADD COLUMN tte DOUBLE")
        self.con.execute("""
            UPDATE options SET
                strike    = TRY_CAST(strike_price AS DOUBLE) / 1000.0,
                mid_price = (TRY_CAST(best_bid AS DOUBLE) + TRY_CAST(best_offer AS DOUBLE)) / 2.0,
                spread    = TRY_CAST(best_offer AS DOUBLE) - TRY_CAST(best_bid AS DOUBLE),
                tte_days  = CAST(exdate AS DATE) - CAST(date AS DATE),
                tte       = (CAST(exdate AS DATE) - CAST(date AS DATE)) / 365.0
        """)
        timings["derived_columns"] = time.perf_counter() - t_derived
        timings["total"] = time.perf_counter() - t0

        count = self.con.execute("SELECT count(*) FROM options").fetchone()[0]
        if self.verbose:
            print(f"Options table: {count:,} rows")
            print(
                "build_options_table timings (s): "
                + ", ".join(f"{k}={v:.2f}" for k, v in timings.items())
            )
            
    
    @timer
    def build_equities_table(self) -> None:
        """Ingest all equities data files into the 'equities' DuckDB table.

        Column types are read from metadata (equities/metadata/*.csv).
        Skips sp500 constituents file. Metadata must have: Variable Name, Type, Description.
        """
        if self._table_exists("equities"):
            if self.verbose:
                print("Table 'equities' already exists, skipping.")
            return

        metadata_path = self._find_metadata_file(
            self.equities_metadata_path, self.equities_data_path
        )

        column_types = self._load_metadata_schema(metadata_path) if metadata_path else {}

        if not column_types and self.verbose:
            print("No equities metadata found; all columns will be VARCHAR.")

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

        column_order = None
        for i, data_file in enumerate(data_files):
            label = self._extract_equities_label(data_file.name)
            escaped_path = str(data_file).replace("'", "''")
            select_clause, column_order = self._build_select_with_schema(
                str(data_file), column_types, column_order, source_label=label
            )
            if i == 0:
                self.con.execute(f"""
                    CREATE TABLE equities AS
                    SELECT
                        {select_clause}
                    FROM (
                        SELECT * FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                    ) AS raw
                """)
            else:
                self.con.execute(f"""
                    INSERT INTO equities
                    SELECT
                        {select_clause}
                    FROM (
                        SELECT * FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                    ) AS raw
                """)

        count = self.con.execute("SELECT count(*) FROM equities").fetchone()[0]
        if self.verbose:
            print(f"Equities table: {count:,} rows")

    def _build_rf_schema(self, csv_path: Path) -> Dict[str, str]:
        """Build risk_free schema from first CSV.
        Col 1: date (DATE), Col 2: MAX_DATA_TTM (INTEGER), Rest: maturity months (DOUBLE).
        """
        cols_df = self.con.execute(f"""
            SELECT * FROM read_csv('{str(csv_path).replace("'", "''")}', auto_detect=true, all_varchar=true)
            LIMIT 0
        """).fetchdf()
        cols = cols_df.columns.tolist()
        schema = {}
        for i, col in enumerate(cols):
            if i == 0:
                schema[col] = "DATE"
            elif i == 1 or "max_data_ttm" in str(col).lower():
                schema[col] = "INTEGER"
            elif col.isdigit():
                schema[col] = "DOUBLE"
            else:
                schema[col] = "VARCHAR"
        return schema

    @timer
    def build_rf_table(self) -> None:
        """Ingest risk-free rate data into the 'risk_free' DuckDB table.

        No metadata file. Schema inferred: first column = DATE, second = MAX_DATA_TTM
        (INTEGER, max 350/360), remaining columns = maturity in months (DOUBLE rates).
        """
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

        column_types = self._build_rf_schema(data_files[0])
        column_order = None
        for i, data_file in enumerate(data_files):
            escaped_path = str(data_file).replace("'", "''")
            select_clause, column_order = self._build_select_with_schema(
                str(data_file), column_types, column_order, source_label=None
            )
            if i == 0:
                self.con.execute(f"""
                    CREATE TABLE risk_free AS
                    SELECT
                        {select_clause}
                    FROM (
                        SELECT * FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                    ) AS raw
                """)
            else:
                self.con.execute(f"""
                    INSERT INTO risk_free
                    SELECT
                        {select_clause}
                    FROM (
                        SELECT * FROM read_csv('{escaped_path}', auto_detect=true, all_varchar=true)
                    ) AS raw
                """)

        count = self.con.execute("SELECT count(*) FROM risk_free").fetchone()[0]
        if self.verbose:
            print(f"Risk-free table: {count:,} rows")

    @timer
    def build_rf_long_table(self) -> None:
        """Unpivot the wide yield panel into a long (date, maturity_months, rate) table."""
        if self._table_exists("rf_long"):
            if self.verbose:
                print("Table 'rf_long' already exists, skipping.")
            return

        if not self._table_exists("risk_free"):
            raise RuntimeError("risk_free table must be built first.")

        # First col = date, second = MAX_DATA_TTM, rest = maturity (1,2,...,360)
        desc = self.con.execute("DESCRIBE risk_free").fetchdf()
        col_key = "column_name" if "column_name" in desc.columns else desc.columns[0]
        cols = desc[col_key].tolist()
        date_col = cols[0]
        maturity_cols = [c for c in cols if c.isdigit()]

        if not maturity_cols:
            raise RuntimeError("No maturity columns found in risk_free table.")

        col_list = ", ".join(f'"{c}"' for c in maturity_cols)
        q_date = self._quoted(date_col)
        self.con.execute(f"""
            CREATE TABLE rf_long AS
            SELECT
                CAST({q_date} AS DATE) AS date,
                CAST(maturity AS INTEGER) AS maturity_months,
                CAST(rate AS DOUBLE) AS rate
            FROM (
                UNPIVOT risk_free
                ON {col_list}
                INTO NAME maturity VALUE rate
            )
            WHERE rate IS NOT NULL
              AND CAST({q_date} AS VARCHAR) NOT IN ('', 'None')
        """)

        count = self.con.execute("SELECT count(*) FROM rf_long").fetchone()[0]
        if self.verbose:
            print(f"rf_long table: {count:,} rows")

    @timer
    def build_options_enriched_table(self) -> None:
        """
        Create an enriched options table by joining with equities (spot price)
        and risk-free rates, then computing derived fields:
          - strike (strike_price / 1000)
          - mid_price, spread, spread_pct
          - tte_days, tte (annualised)
          - spot_price (from equities PRC)
          - risk_free_rate (from rf_long, nearest maturity)
          - forward_price = spot_price * exp(risk_free_rate * tte)
          - log_moneyness = ln(strike / forward_price)
        """
        if self._table_exists("options_enriched"):
            if self.verbose:
                print("Table 'options_enriched' already exists, skipping.")
            return

        for dep in ("options", "equities", "rf_long"):
            if not self._table_exists(dep):
                raise RuntimeError(f"Table '{dep}' must be built first.")

        self.con.execute("""
            CREATE TABLE options_enriched AS
            WITH opt_base AS (
                SELECT
                    o.*,
                    GREATEST(ROUND(o.tte_days / 30.44), 1)            AS tte_months
                FROM options o
            ),
            opt_with_spot AS (
                SELECT
                    ob.*,
                    ABS(CAST(e.PRC AS DOUBLE))                        AS spot_price
                FROM opt_base ob
                LEFT JOIN equities e
                    ON UPPER(ob.ticker) = UPPER(e.TICKER)
                   AND CAST(ob.date AS DATE) = CAST(e.date AS DATE)
            ),
            opt_with_rate AS (
                SELECT
                    os.*,
                    rf.rate                                            AS risk_free_rate
                FROM opt_with_spot os
                LEFT JOIN rf_long rf
                    ON CAST(os.date AS DATE) = rf.date
                   AND CAST(os.tte_months AS INTEGER) = rf.maturity_months
            )
            SELECT
                ow.*,
                -- forward_price = S * exp(r * T)
                CASE
                    WHEN ow.spot_price IS NOT NULL AND ow.tte > 0
                    THEN ow.spot_price * EXP(COALESCE(ow.risk_free_rate, 0) * ow.tte)
                    ELSE ow.spot_price
                END                                                   AS forward_price,
                -- spread as pct of mid
                CASE
                    WHEN ow.mid_price > 0
                    THEN ow.spread / ow.mid_price
                    ELSE NULL
                END                                                   AS spread_pct,
                -- log-moneyness = ln(K / F)
                CASE
                    WHEN ow.spot_price IS NOT NULL AND ow.spot_price > 0 AND ow.tte > 0
                    THEN LN(ow.strike / (
                        ow.spot_price * EXP(COALESCE(ow.risk_free_rate, 0) * ow.tte)
                    ))
                    ELSE NULL
                END                                                   AS log_moneyness
            FROM opt_with_rate ow
        """)

        count = self.con.execute("SELECT count(*) FROM options_enriched").fetchone()[0]
        has_spot = self.con.execute(
            "SELECT count(*) FROM options_enriched WHERE spot_price IS NOT NULL"
        ).fetchone()[0]
        if self.verbose:
            print(f"options_enriched table: {count:,} rows ({has_spot:,} with spot price)")

    def build_all(self) -> None:
        """Build all tables."""
        self.build_options_table()
        self.build_equities_table()
        self.build_rf_table()
        self.build_rf_long_table()
        self.build_options_enriched_table()

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


class Explorer:
    def __init__(self, dl: DataLoader):
        self.dl = dl

    def list_tables(self):
        tables = self.dl.tables()
        for t in tables:
            count = self.dl.query(f"SELECT count(*) AS n FROM {t}").iloc[0, 0]
            print(f"  {t}: {count:,} rows")
        return tables

    def schema(self, table: str) -> pd.DataFrame:
        return self.dl.describe(table)

    def sample(self, table: str, n: int = 5) -> pd.DataFrame:
        return self.dl.query(f"SELECT * FROM {table} LIMIT {n}")

    def summary(self, table: str) -> pd.DataFrame:
        df = self.dl.query(f"SELECT * FROM {table}")
        return df.describe(include="all")

    def _numeric_columns(self, table: str) -> List[str]:
        desc = self.dl.describe(table)
        numeric_types = {"INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL", "HUGEINT"}
        return [
            row["column_name"] for _, row in desc.iterrows()
            if any(t in str(row["column_type"]).upper() for t in numeric_types)
        ]

    def missing_heatmap(self, table: str, figsize=(14, 8)):
        import matplotlib.pyplot as plt
        import seaborn as sns

        cols = self.dl.describe(table)["column_name"].tolist()
        has_date = "date" in cols

        if has_date:
            null_exprs = ", ".join(
                f"AVG(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0.0 END) AS {c}"
                for c in cols if c != "date"
            )
            df = self.dl.query(f"""
                SELECT STRFTIME(CAST(date AS DATE), '%Y%m') AS date, {null_exprs}
                FROM {table}
                GROUP BY STRFTIME(CAST(date AS DATE), '%Y%m')
                ORDER BY MIN(CAST(date AS DATE))
            """)
            df = df.set_index("date")
        else:
            null_exprs = ", ".join(
                f"AVG(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0.0 END) AS {c}"
                for c in cols
            )
            df = self.dl.query(f"SELECT {null_exprs} FROM {table}")

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            df.T.astype(float),
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar=True,
            cbar_kws={"label": "Missing fraction"},
            ax=ax,
        )
        ax.set_title(f"Missing fraction — {table}")
        ax.set_xlabel("date" if has_date else "")
        plt.tight_layout()
        plt.show()

    def histograms(self, table: str, sample: int = 100_000, bins: int = 50, figsize=(16, 12)):
        import matplotlib.pyplot as plt

        cols = self._numeric_columns(table)
        if not cols:
            print(f"No numeric columns in {table}")
            return
        df = self.dl.query(f"SELECT {', '.join(cols)} FROM {table} USING SAMPLE {sample}")
        n_cols = min(4, len(cols))
        n_rows = (len(cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        for i, col in enumerate(cols):
            df[col].dropna().hist(bins=bins, ax=axes[i])
            axes[i].set_title(col, fontsize=9)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"Histograms — {table} (sample {sample:,})", fontsize=12)
        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self, table: str, sample: int = 100_000, figsize=(12, 10)):
        import matplotlib.pyplot as plt
        import seaborn as sns

        cols = self._numeric_columns(table)
        df = self.dl.query(f"SELECT {', '.join(cols)} FROM {table} USING SAMPLE {sample}")
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            print(f"Not enough numeric columns in {table}")
            return
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Correlation — {table}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data_path = "project/data"

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
