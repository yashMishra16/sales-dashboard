"""Reusable statistics and KPI functions (Requirements 2 & 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["Sales", "Quantity", "Discount", "Profit"]


def compute_kpis(df: pd.DataFrame) -> dict:
    """Five headline KPIs. Orders = unique Order IDs; margin = Profit / Sales × 100."""
    sales = float(df["Sales"].sum())
    profit = float(df["Profit"].sum())
    return {
        "total_sales": sales,
        "total_profit": profit,
        "total_orders": int(df["Order ID"].nunique()),
        "total_quantity": int(df["Quantity"].sum()),
        "profit_margin": (profit / sales * 100) if sales else 0.0,
    }


def summary_statistics(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Count, mean, median, mode, std, variance, min/max, quartiles, IQR, skew, kurtosis, CV."""
    cols = [c for c in (cols or NUMERIC) if c in df.columns]
    rows = {}
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        mode = s.mode()
        rows[c] = {
            "count": s.count(), "mean": s.mean(), "median": s.median(),
            "mode": mode.iloc[0] if not mode.empty else np.nan,
            "std": s.std(), "variance": s.var(), "min": s.min(), "Q1": q1,
            "Q3": q3, "max": s.max(), "IQR": q3 - q1, "skewness": s.skew(),
            "kurtosis": s.kurt(),
            "coef. of variation %": (s.std() / s.mean() * 100) if s.mean() else np.nan,
        }
    return pd.DataFrame(rows).T.round(3)


def group_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Sales/profit/orders/quantity/margin per group."""
    g = df.groupby(by).agg(
        Sales=("Sales", "sum"), Profit=("Profit", "sum"),
        Orders=("Order ID", "nunique"), Quantity=("Quantity", "sum"),
        Avg_Discount=("Discount", "mean"),
    )
    g["Profit Margin %"] = (g["Profit"] / g["Sales"] * 100).round(2)
    return g.sort_values("Sales", ascending=False).round(2).reset_index()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[NUMERIC].corr().round(3)


def iqr_outlier_counts(df: pd.DataFrame, k: float = 1.5) -> pd.Series:
    """How many values per numeric column fall outside Q1-k·IQR / Q3+k·IQR."""
    out = {}
    for c in NUMERIC:
        q1, q3 = df[c].quantile([0.25, 0.75])
        i = q3 - q1
        out[c] = int(((df[c] < q1 - k * i) | (df[c] > q3 + k * i)).sum())
    return pd.Series(out, name=f"Outliers ({k}×IQR)")
