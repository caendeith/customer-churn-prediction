"""
Data loading, cleaning and churn-window construction.

Dataset: UCI Machine Learning Repository -- "Online Retail" (ID 352).
Every transaction of a UK-based online giftware retailer between
2010-12-01 and 2011-12-09.

This module is deliberately dependency-light (pandas only) so it can be reused
by `run_all.py` and by the `predict.py` CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

RAW_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

#: Last day covered by the dataset.
DATA_END_DATE = pd.Timestamp("2011-12-09")
#: End of the feature window == start of the label window (end - 90 days).
CUT_DATE = pd.Timestamp("2011-09-10")
#: Snapshot used for the `recency` feature (cut date + 1 day).
SNAPSHOT_DATE = CUT_DATE + pd.Timedelta(days=1)

LABEL_WINDOW_DAYS = (DATA_END_DATE - CUT_DATE).days

#: Candidate locations of the full dataset (relative to the repo root).
FULL_DATA_CANDIDATES = [
    "data/raw/Online Retail.xlsx",
    "data/raw/online_retail.xlsx",
    "data/raw/Online_Retail.xlsx",
    "data/raw/online+retail.xlsx",
]

SAMPLE_DATA_PATH = "data/sample/online_retail_sample.csv"

DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


@dataclass
class CleaningStep:
    """One row-level filter applied during cleaning."""

    step: str
    rows_removed: int
    rows_remaining: int


@dataclass
class Dataset:
    """Everything downstream code needs, plus provenance for the report."""

    raw: pd.DataFrame
    clean: pd.DataFrame
    cancelled: pd.DataFrame
    cleaning_log: list[CleaningStep]
    source_path: Path
    source_kind: str  # "full" | "sample"

    @property
    def n_raw_rows(self) -> int:
        return len(self.raw)

    @property
    def n_clean_rows(self) -> int:
        return len(self.clean)

    @property
    def n_customers(self) -> int:
        return int(self.clean["CustomerID"].nunique())

    @property
    def total_revenue(self) -> float:
        return float(self.clean["revenue"].sum())


# --------------------------------------------------------------------------- #
# Source discovery
# --------------------------------------------------------------------------- #


def resolve_data_source(repo_root: Path | str) -> tuple[Path, str]:
    """Return ``(path, kind)`` for the best available data file.

    Prefers the full UCI workbook when it is present under ``data/raw/`` and
    otherwise falls back to the bundled 437-customer sample.
    """
    repo_root = Path(repo_root)

    for rel in FULL_DATA_CANDIDATES:
        candidate = repo_root / rel
        if candidate.is_file():
            return candidate, "full"

    # Any other xlsx dropped into data/raw/ is treated as the full dataset.
    raw_dir = repo_root / "data" / "raw"
    if raw_dir.is_dir():
        for candidate in sorted(raw_dir.glob("*.xlsx")):
            if not candidate.name.startswith("~$"):
                return candidate, "full"

    sample = repo_root / SAMPLE_DATA_PATH
    if sample.is_file():
        return sample, "sample"

    raise FileNotFoundError(
        "No dataset found. Expected either the full workbook in data/raw/ "
        f"or the bundled sample at {SAMPLE_DATA_PATH}. "
        f"Run `python run_all.py --download` to fetch the full dataset."
    )


def load_raw(path: Path | str) -> pd.DataFrame:
    """Read the raw transactions from ``.xlsx`` or ``.csv``."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")

    return df[RAW_COLUMNS].copy()


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[CleaningStep]]:
    """Apply the six documented cleaning rules, in order.

    Returns ``(clean, cancelled, log)`` where ``cancelled`` holds the rows that
    rule 2 removed. Cancellation behaviour is a genuine customer signal, so we
    keep those rows aside to build ``cancellation_count`` / ``cancellation_ratio``
    features later -- they are still excluded from every monetary aggregate.

    The order matters and is fixed by the project spec:
      1. drop rows with missing CustomerID
      2. drop cancellations (InvoiceNo starts with 'C')
      3. drop Quantity <= 0
      4. drop UnitPrice <= 0
      5. drop exact duplicate rows
      6. revenue = Quantity * UnitPrice
    """
    df = raw.copy()
    log: list[CleaningStep] = []

    def record(step: str, before: int) -> None:
        log.append(CleaningStep(step=step, rows_removed=before - len(df), rows_remaining=len(df)))

    # -- normalise dtypes before filtering ---------------------------------- #
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("string").str.strip()

    # -- 1. missing CustomerID ---------------------------------------------- #
    before = len(df)
    df = df[df["CustomerID"].notna()]
    record("1. drop rows with missing CustomerID", before)

    # Customer IDs are integers in this dataset; keep them as int64 keys.
    df["CustomerID"] = df["CustomerID"].astype("int64")

    # -- 2. cancellations ---------------------------------------------------- #
    is_cancellation = df["InvoiceNo"].str.upper().str.startswith("C").fillna(False)
    cancelled = df[is_cancellation].copy()
    before = len(df)
    df = df[~is_cancellation]
    record("2. drop cancellations (InvoiceNo starts with 'C')", before)

    # -- 3. non-positive quantity ------------------------------------------- #
    before = len(df)
    df = df[df["Quantity"] > 0]
    record("3. drop Quantity <= 0", before)

    # -- 4. non-positive price ---------------------------------------------- #
    before = len(df)
    df = df[df["UnitPrice"] > 0]
    record("4. drop UnitPrice <= 0", before)

    # -- 5. exact duplicate rows -------------------------------------------- #
    before = len(df)
    df = df.drop_duplicates(subset=RAW_COLUMNS, keep="first")
    record("5. drop exact duplicate rows", before)

    # -- 6. revenue ---------------------------------------------------------- #
    df = df.copy()
    df["revenue"] = df["Quantity"] * df["UnitPrice"]
    df = df.sort_values(["InvoiceDate", "InvoiceNo"]).reset_index(drop=True)
    log.append(CleaningStep("6. revenue = Quantity * UnitPrice", 0, len(df)))

    return df, cancelled, log


# --------------------------------------------------------------------------- #
# Churn windows
# --------------------------------------------------------------------------- #


def split_windows(
    clean: pd.DataFrame,
    cut_date: pd.Timestamp = CUT_DATE,
    end_date: pd.Timestamp = DATA_END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cleaned transactions into the feature window and the label window.

    Feature window: ``InvoiceDate <= cut_date``
    Label window:   ``cut_date < InvoiceDate <= end_date``

    The two windows are disjoint and the cut is strictly in the past relative to
    the label window -- that is what makes the setup leak-free.
    """
    dates = clean["InvoiceDate"].dt.normalize()
    feature = clean[dates <= cut_date].copy()
    label = clean[(dates > cut_date) & (dates <= end_date)].copy()
    return feature, label


def build_labels(
    feature_customers: Iterable[int],
    label_window: pd.DataFrame,
) -> pd.DataFrame:
    """One row per in-scope customer: did they buy again in the label window?

    ``churned == 1`` means zero purchases between the cut date and the end of
    the data, i.e. the customer we would have lost.
    """
    customers = pd.Index(sorted(feature_customers), name="CustomerID")

    label_purchases = (
        label_window.groupby("CustomerID")["InvoiceNo"].nunique().reindex(customers).fillna(0)
    )

    labels = pd.DataFrame(index=customers)
    labels["label_invoices"] = label_purchases.astype("int64")
    labels["churned"] = (labels["label_invoices"] == 0).astype("int64")
    return labels


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def load_dataset(repo_root: Path | str, verbose: bool = True) -> Dataset:
    """Load + clean + report provenance for whichever data source is available."""
    repo_root = Path(repo_root)
    path, kind = resolve_data_source(repo_root)
    return load_dataset_from(path, kind, repo_root, verbose=verbose)


def load_dataset_from(
    path: Path | str,
    kind: str,
    repo_root: Path | str,
    verbose: bool = True,
) -> Dataset:
    """Load + clean a specific file, tagging it with ``kind`` ('full' or 'sample')."""
    repo_root = Path(repo_root)
    path = Path(path)

    raw = load_raw(path)
    clean, cancelled, log = clean_transactions(raw)

    dataset = Dataset(
        raw=raw,
        clean=clean,
        cancelled=cancelled,
        cleaning_log=log,
        source_path=path,
        source_kind=kind,
    )

    if verbose:
        label = "FULL DATASET" if kind == "full" else "SAMPLE DATASET"
        print(f"[data] source      : {path}")
        print(f"[data] source kind : {label}")
        if kind == "sample":
            print(
                "[data] WARNING: the full UCI dataset was not found under data/raw/, so this "
                "run uses the bundled sample.\n"
                "       Metrics will differ from the published full-data numbers. "
                "Run `python run_all.py --download` to fetch the full workbook."
            )
        print(f"[data] raw rows    : {len(raw):,}")
        print(f"[data] clean rows  : {len(clean):,}")
        print(f"[data] customers   : {dataset.n_customers:,}")
        print(f"[data] revenue     : {dataset.total_revenue:,.2f}")
        print(f"[data] date range  : {clean['InvoiceDate'].min()} -> {clean['InvoiceDate'].max()}")

    return dataset


def download_full_dataset(repo_root: Path | str) -> Path:
    """Fetch and unzip the full UCI workbook into ``data/raw/``."""
    import urllib.request
    import zipfile

    repo_root = Path(repo_root)
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "online_retail.zip"

    print(f"[data] downloading {DOWNLOAD_URL}")
    urllib.request.urlretrieve(DOWNLOAD_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    print(f"[data] extracted into {raw_dir}")
    return raw_dir
