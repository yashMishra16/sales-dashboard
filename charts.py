"""Plotly chart builders (Requirement 4). Each returns a Figure."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"]
LAYOUT = dict(margin=dict(l=10, r=10, t=50, b=10), height=380,
              font=dict(size=13), legend=dict(orientation="h", y=-0.15))


def _style(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title=title, **LAYOUT)
    return fig


def monthly_trend(df: pd.DataFrame) -> go.Figure:
    m = df.dropna(subset=["Order Month"]).groupby("Order Month")[["Sales", "Profit"]].sum().reset_index()
    fig = px.line(m, x="Order Month", y=["Sales", "Profit"], markers=True,
                  color_discrete_sequence=PALETTE)
    fig.update_yaxes(title="Amount ($)")
    fig.update_xaxes(title=None)
    fig.update_layout(legend_title_text="")
    return _style(fig, "Monthly Sales & Profit Trend")


def subcategory_bar(df: pd.DataFrame) -> go.Figure:
    g = df.groupby(["Category", "Sub-Category"], as_index=False)[["Sales", "Profit"]].sum()
    g = g.sort_values("Sales")
    fig = px.bar(g, x="Sales", y="Sub-Category", color="Category", orientation="h",
                 hover_data={"Profit": ":,.0f", "Sales": ":,.0f"}, color_discrete_sequence=PALETTE)
    fig.update_yaxes(title=None)
    return _style(fig, "Sales by Sub-Category")


def region_donut(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("Region", as_index=False)["Sales"].sum()
    fig = px.pie(g, names="Region", values="Sales", hole=0.55, color_discrete_sequence=PALETTE)
    fig.update_traces(textinfo="percent+label")
    return _style(fig, "Sales Share by Region")


def discount_profit_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="Discount", y="Profit", color="Category", size="Sales",
                     size_max=22, opacity=0.7, hover_data=["Order ID", "Product Name", "Sales"],
                     color_discrete_sequence=PALETTE)
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_xaxes(tickformat=".0%")
    return _style(fig, "Discount vs Profit (bubble = sales)")


def state_profit_bar(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("State", as_index=False)["Profit"].sum().sort_values("Profit")
    g["Result"] = g["Profit"].map(lambda v: "Profit" if v >= 0 else "Loss")
    fig = px.bar(g, x="Profit", y="State", orientation="h", color="Result",
                 color_discrete_map={"Profit": "#10b981", "Loss": "#ef4444"})
    fig.update_yaxes(title=None)
    return _style(fig, "Profit by State")


def segment_shipmode_heatmap(df: pd.DataFrame) -> go.Figure:
    p = df.pivot_table(index="Segment", columns="Ship Mode", values="Sales", aggfunc="sum", fill_value=0)
    fig = px.imshow(p, text_auto=",.0f", aspect="auto", color_continuous_scale="Blues")
    return _style(fig, "Sales: Segment × Ship Mode")
