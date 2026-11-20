import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import compute_kpis, summary_statistics
from cleaning import clean_data, parse_number
from data_loader import load_file, validate_dataframe

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_raw_dataset.csv"


@pytest.fixture(scope="module")
def raw():
    return load_file(SAMPLE.read_bytes(), SAMPLE.name)


@pytest.fixture(scope="module")
def cleaned(raw):
    return clean_data(raw)


def test_parse_number():
    assert parse_number("five") == 5
    assert parse_number("ten percent") == 0.10
    assert np.isnan(parse_number("inf")) and np.isnan(parse_number("N/A"))


def test_validation(raw):
    v = validate_dataframe(raw)
    assert v["rows"] == 186 and v["duplicate_count"] == 4


def test_cleaning_outputs(cleaned):
    df, log, s = cleaned
    assert df.duplicated().sum() == 0
    assert df[["Sales", "Quantity", "Discount", "Profit"]].isna().sum().sum() == 0
    assert (df["Sales"] > 0).all() and (df["Quantity"] > 0).all() and df["Sales"].max() < 20000
    assert df["Discount"].between(0, 1).all()
    assert set(df["Region"]) <= {"East", "West", "South", "Central"}
    assert set(df["Category"]) == {"Technology", "Furniture", "Office Supplies"}
    assert set(df["Segment"]) == {"Consumer", "Corporate", "Home Office"}
    assert (df["Shipping Days"].dropna() >= 0).all()
    assert s["rows_out"] == len(df) == 180


def test_kpis(cleaned):
    df = cleaned[0]
    k = compute_kpis(df)
    assert k["total_orders"] == df["Order ID"].nunique()
    assert k["profit_margin"] == pytest.approx(df["Profit"].sum() / df["Sales"].sum() * 100)
    assert not summary_statistics(df).empty


def test_all_formats_equivalent(raw):
    base = clean_data(raw)[0]
    xl, js = io.BytesIO(), raw.to_json(orient="records")
    raw.to_excel(xl, index=False)
    for data, name in [(xl.getvalue(), "a.xlsx"), (js.encode(), "a.json")]:
        got = clean_data(load_file(data, name))[0]
        assert compute_kpis(got)["total_orders"] == compute_kpis(base)["total_orders"]
        assert len(got) == len(base)


def test_bad_inputs():
    with pytest.raises(ValueError):
        load_file(b"abc", "x.txt")
    with pytest.raises(ValueError):
        load_file(b"", "x.csv")
