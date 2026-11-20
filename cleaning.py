"""Data preprocessing (Requirement 2): missing values, duplicates,
inconsistent text, invalid numbers and mixed date formats.

Every fix is recorded in a log so the app can show exactly what changed.
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd

from data_loader import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

NUM_COLS = ["Sales", "Quantity", "Discount", "Profit"]
DATE_COLS = ["Order Date", "Ship Date"]
STD_COLS = ["Ship Mode", "Segment", "Country", "Region", "Category", "Sub-Category", "State", "City"]
FILL_UNKNOWN = ["Customer ID", "Product ID", "Product Name", "Ship Mode", "Segment", "Country"]

TEXT_NULLS = {"", "n/a", "na", "nan", "none", "null", "unknown", "not available", "?", "-", "--"}
WORD_NUMS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten".split())}
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
AMBIG_RE = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$")
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y", "%d %b %Y",
                "%b %d, %Y", "%Y/%m/%d", "%m/%d/%Y"]


class _Log:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, step: str, issue: str, count: int, action: str):
        if count and count > 0:
            self.rows.append({"Step": step, "Issue found": issue,
                              "Cells/Rows affected": int(count), "Action taken": action})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["Step", "Issue found", "Cells/Rows affected", "Action taken"])


# ----------------------------------------------------------------- parsers
def parse_number(v) -> float:
    """Convert messy numeric input ('five', '10%', 'ten percent', 'inf', '$1,200') to float/NaN."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
        return float(v) if np.isfinite(v) else np.nan
    s = str(v).strip().lower().replace(",", "").replace("$", "")
    if s in TEXT_NULLS:
        return np.nan
    pct = False
    m = re.match(r"^(.+?)\s*(%|percent|pct)$", s)
    if m:
        s, pct = m.group(1).strip(), True
    try:
        val = float(WORD_NUMS[s]) if s in WORD_NUMS else float(s)
    except ValueError:
        return np.nan
    if not np.isfinite(val):
        return np.nan
    return val / 100 if pct else val


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse a column that mixes several date formats."""
    s = series.astype("object").map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        out = out.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
    return out


def _resolve_ambiguous_ship(ship_raw: pd.Series, ship: pd.Series, order: pd.Series) -> tuple[pd.Series, int]:
    """For dd-mm vs mm-dd ambiguity pick the reading that falls 0-45 days after the order date."""
    fixed = 0
    ship = ship.copy()
    for idx in ship_raw.index[ship_raw.astype("object").map(lambda v: isinstance(v, str) and bool(AMBIG_RE.match(v.strip())))]:
        text, od = ship_raw[idx].strip().replace("/", "-"), order[idx]
        if pd.isna(od):
            continue
        cands = [pd.to_datetime(text, format=f, errors="coerce") for f in ("%d-%m-%Y", "%m-%d-%Y")]
        ok = [c for c in cands if pd.notna(c) and 0 <= (c - od).days <= 45]
        if ok and ok[0] != ship[idx]:
            ship[idx] = ok[0]
            fixed += 1
    return ship, fixed


def _canonicalize(series: pd.Series, cutoff: float = 0.85) -> pd.Series:
    """Unify case/spacing variants ('consumer','Consumer','SameDay') and near-typos ('Office supply')."""
    s = series.astype("object")
    counts = s.dropna().value_counts()
    if counts.empty:
        return s
    keyf = lambda v: re.sub(r"[^a-z0-9]", "", str(v).lower())
    groups: dict[str, list] = {}
    for v, c in counts.items():
        groups.setdefault(keyf(v), []).append((v, c))
    canon = {k: max(g, key=lambda t: (t[1], t[0] != t[0].lower())) [0] for k, g in groups.items()}
    totals = {k: sum(c for _, c in g) for k, g in groups.items()}
    major = [k for k in totals if totals[k] > max(2, 0.01 * len(s))]
    mapping = {}
    for k, g in groups.items():
        target = k
        if k not in major and major:
            best = difflib.get_close_matches(k, [m for m in major if totals[m] > totals[k]], n=1, cutoff=cutoff)
            if best:
                target = best[0]
        for v, _ in g:
            mapping[v] = canon[target]
    return s.map(mapping)


def _majority_map(df: pd.DataFrame, key: str, val: str) -> dict:
    g = df.dropna(subset=[key, val]).groupby(key)[val].agg(lambda x: x.value_counts().idxmax())
    return g.to_dict()


# ----------------------------------------------------------------- main
def clean_data(raw: pd.DataFrame, impute_strategy: str = "median",
               outlier_k: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean a raw sales DataFrame.

    impute_strategy: 'median' (fill numeric gaps with Sub-Category medians) or 'drop'.
    outlier_k: values beyond Q1-k*IQR / Q3+k*IQR are treated as data-entry errors.
    Returns (clean_df, cleaning_log_df, summary_dict).
    """
    log = _Log()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rows_in = len(df)

    for c in OPTIONAL_COLUMNS:           # tolerate absent optional columns
        if c not in df.columns:
            df[c] = np.nan
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    # 1. text hygiene ------------------------------------------------------
    ws = nulls = 0
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        s = df[c].astype("object")
        cleaned = s.map(lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v)
        ws += sum(isinstance(a, str) and a != b for a, b in zip(s, cleaned))
        is_null_tok = cleaned.map(lambda v: isinstance(v, str) and v.lower() in TEXT_NULLS)
        nulls += int(is_null_tok.sum())
        df[c] = cleaned.mask(is_null_tok, np.nan)
    log.add("Text", "Leading/trailing/duplicate whitespace", ws, "Trimmed and collapsed spaces")
    log.add("Text", "Placeholder tokens ('', 'unknown', 'N/A', '?')", nulls, "Converted to missing (NaN)")

    # 2. empty / unusable rows ----------------------------------------------
    n = df.isna().all(axis=1).sum()
    df = df.dropna(how="all")
    log.add("Rows", "Completely empty rows", n, "Dropped")
    n = df["Order ID"].isna().sum()
    df = df.dropna(subset=["Order ID"])
    log.add("Rows", "Rows without an Order ID (cannot be attributed to an order)", n, "Dropped")

    # 3. inconsistent categorical values ----------------------------------
    changed = 0
    for c in STD_COLS:
        before = df[c].copy()
        df[c] = _canonicalize(df[c])
        changed += int((before.notna() & (before != df[c])).sum())
    log.add("Consistency", "Inconsistent case / spelling (e.g. 'consumer', 'SameDay', 'Office supply')",
            changed, "Standardised to the most common spelling")

    # 4. numeric columns -----------------------------------------------------
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        orig = df[c].astype("object")
        parsed = orig.map(parse_number)
        is_str = orig.map(lambda v: isinstance(v, str))
        direct = orig.map(lambda v: isinstance(v, str) and np.isfinite(pd.to_numeric(v, errors="coerce")))
        words = (is_str & parsed.notna() & ~direct).sum()
        is_inf = orig.map(lambda v: isinstance(v, (float, int, np.number)) and not isinstance(v, bool) and np.isinf(v))
        bad = (is_str & parsed.isna()).sum() + is_inf.sum()
        log.add("Numeric", f"{c}: words/percent text (e.g. 'five', 'ten percent')", words, "Converted to numbers")
        log.add("Numeric", f"{c}: non-numeric or infinite values", bad, "Set to missing")
        df[c] = parsed.astype(float)

    # 5. dates ----------------------------------------------------------------
    for c in DATE_COLS:
        raw_col = df[c].copy()
        parsed = parse_dates(raw_col)
        if c == "Ship Date":
            parsed, fixed = _resolve_ambiguous_ship(raw_col, parsed, df["Order Date"] if pd.api.types.is_datetime64_any_dtype(df["Order Date"]) else parse_dates(df["Order Date"]))
            log.add("Dates", "Ship Date: ambiguous dd-mm/mm-dd resolved using Order Date", fixed, "Picked the reading 0-45 days after order")
        non_iso = raw_col.map(lambda v: isinstance(v, str) and not ISO_RE.match(v)).sum()
        unparsable = (raw_col.notna() & parsed.isna()).sum()
        log.add("Dates", f"{c}: non-ISO formats (e.g. 26/12/2024, 21-Mar-2025)", non_iso, "Parsed to YYYY-MM-DD")
        log.add("Dates", f"{c}: unparsable dates", unparsable, "Set to missing")
        df[c] = parsed
    bad = (df["Ship Date"].notna() & df["Order Date"].notna() & (df["Ship Date"] < df["Order Date"]))
    log.add("Dates", "Ship Date earlier than Order Date", bad.sum(), "Ship Date set to missing")
    df.loc[bad, "Ship Date"] = pd.NaT

    # 6. duplicates -----------------------------------------------------------
    n = df.duplicated().sum()
    df = df.drop_duplicates(keep="first")
    log.add("Duplicates", "Fully identical records (Order ID repeats are kept: one order has many lines)", n, "Dropped, first kept")

    # 7. hierarchy / geography repair -----------------------------------------
    cat_map = _majority_map(df, "Sub-Category", "Category")
    new_cat = df["Sub-Category"].map(cat_map)
    n = int((new_cat.notna() & (new_cat != df["Category"])).sum())
    df["Category"] = new_cat.fillna(df["Category"])
    log.add("Consistency", "Category missing or contradicting its Sub-Category", n, "Re-derived from Sub-Category")

    state_map = _majority_map(df, "City", "State")
    new_state = df["City"].map(state_map)
    n = int((new_state.notna() & (new_state != df["State"])).sum())
    df["State"] = new_state.fillna(df["State"])
    log.add("Consistency", "City/State mismatch (e.g. Chicago in Ohio) or missing State", n, "State re-derived from City")

    city_map = _majority_map(df, "State", "City")
    n = int(df["City"].isna().sum())
    df["City"] = df["City"].fillna(df["State"].map(city_map))
    log.add("Missing", "City missing", n - int(df["City"].isna().sum()), "Filled with the State's most common city")

    reg_map = _majority_map(df, "State", "Region")
    new_reg = df["State"].map(reg_map)
    n = int((new_reg.notna() & (new_reg != df["Region"])).sum())
    df["Region"] = new_reg.fillna(df["Region"])
    log.add("Consistency", "Region disagrees with State (e.g. Minnesota='West')", n, "Region re-derived from State")

    for c in FILL_UNKNOWN + ["Region", "Category", "Sub-Category", "State", "City"]:
        n = int(df[c].isna().sum())
        df[c] = df[c].fillna("Unknown")
        log.add("Missing", f"{c} missing", n, "Filled with 'Unknown'")

    # 8. invalid / extreme numeric values --------------------------------------
    def flag(col, mask, why):
        mask = mask.fillna(False)
        log.add("Validity", f"{col}: {why}", mask.sum(), "Set to missing")
        df.loc[mask, col] = np.nan

    flag("Sales", df["Sales"] <= 0, "zero/negative sales")
    flag("Quantity", df["Quantity"] <= 0, "zero/negative quantity")
    flag("Discount", (df["Discount"] < 0) | (df["Discount"] > 1), "discount outside 0-1")
    # Profit is excluded: large positive/negative profits are legitimate, only Sales/Quantity get impossible entries.
    for c in ["Sales", "Quantity"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        flag(c, (df[c] < q1 - outlier_k * iqr) | (df[c] > q3 + outlier_k * iqr),
             f"extreme outlier (beyond {outlier_k:g}×IQR)")

    # 9. missing numeric values -------------------------------------------------
    df["Imputed"] = False
    miss = df[NUM_COLS].isna()
    n_missing = int(miss.sum().sum())
    if impute_strategy == "drop":
        n = int(miss.any(axis=1).sum())
        df = df[~miss.any(axis=1)].copy()
        log.add("Missing", "Rows with a missing/invalid Sales, Quantity, Discount or Profit", n, "Dropped")
    else:
        df["Imputed"] = miss.any(axis=1)
        for c in ["Sales", "Quantity", "Discount"]:
            med = df.groupby("Sub-Category")[c].transform("median").fillna(df[c].median())
            n = int(df[c].isna().sum())
            df[c] = df[c].fillna(med)
            log.add("Missing", f"{c} missing/invalid", n, "Filled with Sub-Category median")
        margin = (df["Profit"] / df["Sales"]).replace([np.inf, -np.inf], np.nan)
        med_margin = margin.groupby(df["Sub-Category"]).transform("median").fillna(margin.median())
        n = int(df["Profit"].isna().sum())
        df["Profit"] = df["Profit"].fillna(df["Sales"] * med_margin)
        log.add("Missing", "Profit missing/invalid", n, "Filled with Sales × Sub-Category median margin")

    df["Quantity"] = df["Quantity"].round().astype(int)
    for c in ["Sales", "Profit"]:
        df[c] = df[c].round(2)
    n = int(df["Order Date"].isna().sum())
    log.add("Missing", "Order Date missing", n, "Kept; excluded from date filter and trend chart when a range is selected")

    # 10. derived columns --------------------------------------------------------
    df["Order Year"] = df["Order Date"].dt.year.astype("Int64")
    df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df = df.sort_values(["Order Date", "Order ID"]).reset_index(drop=True)

    summary = {"rows_in": rows_in, "rows_out": len(df),
               "cells_imputed": n_missing if impute_strategy != "drop" else 0}
    return df, log.frame(), summary
