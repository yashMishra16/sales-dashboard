"""File loading and validation (Requirement 1)."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".json")

# Columns the dashboard cannot work without.
REQUIRED_COLUMNS = ["Order ID", "Order Date", "Sales", "Quantity", "Profit"]
# Columns used for filters/charts; created as "Unknown" if absent.
OPTIONAL_COLUMNS = [
    "Ship Date", "Ship Mode", "Customer ID", "Segment", "Country", "City", "State",
    "Region", "Product ID", "Category", "Sub-Category", "Product Name", "Discount",
]


def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read a CSV / XLSX / JSON upload into a DataFrame.

    Raises ValueError with a user-friendly message on any problem.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Please upload one of: "
            + ", ".join(SUPPORTED_EXTENSIONS)
        )
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    buf = io.BytesIO(file_bytes)
    try:
        if ext == ".csv":
            df = pd.read_csv(buf)
        elif ext == ".xlsx":
            df = pd.read_excel(buf, engine="openpyxl")
        else:
            df = pd.read_json(buf, convert_dates=False, dtype=False)
    except Exception as exc:  # parser errors are varied; surface them clearly
        raise ValueError(f"Could not read '{filename}': {exc}") from exc
    if df.empty:
        raise ValueError("The file was read successfully but contains no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def check_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (missing_required, missing_optional) column names."""
    cols = set(df.columns)
    return (
        [c for c in REQUIRED_COLUMNS if c not in cols],
        [c for c in OPTIONAL_COLUMNS if c not in cols],
    )


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Profile a DataFrame: shape, dtypes, missing values, duplicates."""
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    # Whitespace-only strings are effectively missing too.
    blank = df.apply(
        lambda s: s.astype(str).str.strip().eq("").sum() if s.dtype == object or str(s.dtype) in ("str", "string") else 0
    )
    profile = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Missing": missing.values,
        "Blank Strings": blank.values,
        "Missing %": (missing.values / max(n_rows, 1) * 100).round(2),
        "Unique Values": df.nunique(dropna=True).values,
    })
    dup_mask = df.duplicated(keep="first")
    return {
        "rows": n_rows,
        "columns": n_cols,
        "column_names": list(df.columns),
        "profile": profile,
        "total_missing": int(missing.sum()),
        "duplicate_count": int(dup_mask.sum()),
        "duplicate_rows": df[df.duplicated(keep=False)].sort_values(list(df.columns)[:2]),
    }
