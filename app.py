"""Sales Analytics Dashboard - Streamlit app.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import analysis
import charts
from cleaning import clean_data
from data_loader import check_schema, load_file, validate_dataframe

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_raw_dataset.csv"

st.set_page_config(page_title="Sales Analytics Dashboard", page_icon="📊", layout="wide")


# ------------------------------------------------------------------ data layer
@st.cache_data(show_spinner="Reading file…")
def read_input(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return load_file(file_bytes, filename)


@st.cache_data(show_spinner="Cleaning data…")
def run_cleaning(raw: pd.DataFrame, strategy: str, k: float):
    return clean_data(raw, impute_strategy=strategy, outlier_k=k)


def get_raw_data(uploaded) -> tuple[pd.DataFrame, str]:
    if uploaded is not None:
        return read_input(uploaded.getvalue(), uploaded.name), uploaded.name
    return read_input(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name), f"{SAMPLE_PATH.name} (bundled sample)"


# ------------------------------------------------------------------ UI pieces
def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Five sidebar filters; returns the filtered frame."""
    st.sidebar.header("🔎 Filters")
    mask = pd.Series(True, index=df.index)

    dates = df["Order Date"].dropna()
    if not dates.empty:
        lo, hi = dates.min().date(), dates.max().date()
        picked = st.sidebar.date_input("Order date range", (lo, hi), min_value=lo, max_value=hi)
        if isinstance(picked, (tuple, list)) and len(picked) == 2 and (picked[0], picked[1]) != (lo, hi):
            start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
            mask &= df["Order Date"].between(start, end)

    for col, label in [("Region", "Region"), ("Category", "Category"),
                       ("Segment", "Customer segment"), ("Ship Mode", "Ship mode")]:
        options = sorted(df[col].dropna().unique())
        chosen = st.sidebar.multiselect(label, options, default=options)
        mask &= df[col].isin(chosen)
    return df[mask]


def render_kpis(df: pd.DataFrame, full: pd.DataFrame):
    k, k_all = analysis.compute_kpis(df), analysis.compute_kpis(full)
    share = lambda key: f"{k[key] / k_all[key] * 100:.1f}% of all data" if k_all[key] else None
    c = st.columns(5)
    c[0].metric("Total Sales", f"${k['total_sales']:,.0f}", share("total_sales"), delta_color="off")
    c[1].metric("Total Profit", f"${k['total_profit']:,.0f}", share("total_profit"), delta_color="off")
    c[2].metric("Total Orders", f"{k['total_orders']:,}", share("total_orders"), delta_color="off")
    c[3].metric("Quantity Sold", f"{k['total_quantity']:,}", share("total_quantity"), delta_color="off")
    c[4].metric("Profit Margin", f"{k['profit_margin']:.2f}%",
                f"overall {k_all['profit_margin']:.2f}%", delta_color="off")


def render_dashboard(df: pd.DataFrame, full: pd.DataFrame):
    render_kpis(df, full)
    st.divider()
    if df.empty:
        st.warning("No records match the current filters. Widen the filters in the sidebar.")
        return
    try:
        a, b = st.columns(2)
        a.plotly_chart(charts.monthly_trend(df), width="stretch")
        b.plotly_chart(charts.subcategory_bar(df), width="stretch")
        c, d = st.columns(2)
        c.plotly_chart(charts.region_donut(df), width="stretch")
        d.plotly_chart(charts.discount_profit_scatter(df), width="stretch")
        e, f = st.columns(2)
        e.plotly_chart(charts.state_profit_bar(df), width="stretch")
        f.plotly_chart(charts.segment_shipmode_heatmap(df), width="stretch")
    except Exception as exc:
        st.error(f"A chart could not be drawn: {exc}")


def render_validation(raw: pd.DataFrame, label: str):
    st.subheader(f"Raw data validation — {label}")
    v = validate_dataframe(raw)
    c = st.columns(4)
    c[0].metric("Rows", f"{v['rows']:,}")
    c[1].metric("Columns", v["columns"])
    c[2].metric("Missing cells", f"{v['total_missing']:,}")
    c[3].metric("Duplicate rows", v["duplicate_count"])
    st.markdown("**Column names:** " + ", ".join(f"`{c}`" for c in v["column_names"]))
    st.markdown("**Data types & missing values**")
    st.dataframe(v["profile"], width="stretch", hide_index=True)
    if v["duplicate_count"]:
        with st.expander(f"Show duplicate records ({v['duplicate_count']} extra copies)"):
            st.dataframe(v["duplicate_rows"], width="stretch")
    with st.expander("Preview raw data (first 50 rows)"):
        st.dataframe(raw.head(50), width="stretch")


def render_cleaning(raw: pd.DataFrame, clean: pd.DataFrame, log: pd.DataFrame, summary: dict):
    st.subheader("Cleaning report")
    c = st.columns(4)
    c[0].metric("Rows before", summary["rows_in"])
    c[1].metric("Rows after", summary["rows_out"], f"{summary['rows_out'] - summary['rows_in']}")
    c[2].metric("Missing cells before", int(raw.isna().sum().sum()))
    c[3].metric("Missing cells after (core fields)", int(clean[["Sales", "Quantity", "Discount", "Profit", "Order ID"]].isna().sum().sum()))
    st.dataframe(log, width="stretch", hide_index=True)
    st.markdown("**Cleaned data types**")
    st.dataframe(pd.DataFrame({"Column": clean.columns, "Data Type": clean.dtypes.astype(str).values,
                               "Missing": clean.isna().sum().values}), width="stretch", hide_index=True)
    d = st.columns(2)
    d[0].download_button("⬇️ Cleaned data (CSV)", clean.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")
    d[1].download_button("⬇️ Cleaning log (CSV)", log.to_csv(index=False).encode(), "cleaning_log.csv", "text/csv")


def render_statistics(df: pd.DataFrame):
    st.subheader("Statistical summaries (filtered data)")
    if df.empty:
        st.warning("No records match the current filters.")
        return
    st.markdown("**Descriptive statistics**")
    st.dataframe(analysis.summary_statistics(df), width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("**Correlation matrix**")
        st.dataframe(analysis.correlation_matrix(df), width="stretch")
    with right:
        st.markdown("**Outliers by 1.5×IQR rule (kept in data, for information)**")
        st.dataframe(analysis.iqr_outlier_counts(df).to_frame(), width="stretch")
    by = st.selectbox("Group performance by", ["Category", "Sub-Category", "Region", "State", "Segment", "Ship Mode", "Order Year"])
    st.dataframe(analysis.group_summary(df, by), width="stretch", hide_index=True)


# ------------------------------------------------------------------ main
def main():
    st.title("📊 Sales Analytics Dashboard")
    st.caption("Upload a CSV, Excel or JSON sales file — it is validated, cleaned and turned into an interactive dashboard.")

    st.sidebar.header("📂 Data")
    uploaded = st.sidebar.file_uploader("Upload dataset", type=["csv", "xlsx", "json"])
    strategy = st.sidebar.radio("Missing numeric values", ["median", "drop"],
                                format_func=lambda x: "Impute (Sub-Category median)" if x == "median" else "Drop those rows")
    k = st.sidebar.slider("Extreme-outlier threshold (× IQR)", 3.0, 10.0, 5.0, 0.5,
                          help="Sales and Quantity values beyond this are treated as entry errors.")

    try:
        raw, label = get_raw_data(uploaded)
        missing_req, missing_opt = check_schema(raw)
        if missing_req:
            st.error("The file is missing required column(s): " + ", ".join(missing_req))
            st.stop()
        if missing_opt:
            st.warning("Optional columns not found (filled with 'Unknown'): " + ", ".join(missing_opt))
        clean, log, summary = run_cleaning(raw, strategy, k)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # last-resort guard so the app never shows a raw traceback
        st.error(f"Unexpected error while processing the data: {exc}")
        st.stop()

    if uploaded is None:
        st.info("Showing the bundled sample dataset. Upload your own file in the sidebar to replace it.")

    filtered = render_filters(clean)
    st.sidebar.caption(f"{len(filtered):,} of {len(clean):,} records selected")

    t1, t2, t3, t4, t5 = st.tabs(["📈 Dashboard", "✅ Data validation", "🧹 Cleaning report", "📐 Statistics", "🗃️ Data"])
    with t1:
        render_dashboard(filtered, clean)
    with t2:
        render_validation(raw, label)
    with t3:
        render_cleaning(raw, clean, log, summary)
    with t4:
        render_statistics(filtered)
    with t5:
        st.dataframe(filtered, width="stretch", hide_index=True)
        st.download_button("⬇️ Download filtered data (CSV)", filtered.to_csv(index=False).encode(), "filtered_data.csv", "text/csv")


main()
