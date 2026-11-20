# 📊 Sales Analytics Dashboard

Streamlit app that validates, cleans and visualises sales data (CSV / Excel / JSON).

**Live app:**[https://sales-dashboard-9bow4vwpngc9tapkgbaax5.streamlit.app/]
**Source:**[https://github.com/yashMishra16/sales-dashboard]

## Features

| Requirement                                                                                         | Where                                   |
| --------------------------------------------------------------------------------------------------- | --------------------------------------- |
| Upload CSV/XLSX/JSON + validation (shape, columns, dtypes, missing, duplicates)                     | `data_loader.py`, _Data validation_ tab |
| Cleaning (missing values, duplicates, inconsistent text, mixed date formats, invalid numbers) + log | `cleaning.py`, _Cleaning report_ tab    |
| Reusable statistics functions                                                                       | `analysis.py`, _Statistics_ tab         |
| 5 filters (date, region, category, segment, ship mode) and 5 KPIs                                   | `app.py` sidebar + Dashboard tab        |
| 6 interactive Plotly charts                                                                         | `charts.py`                             |
| Error handling (bad type, empty file, missing columns, empty filter result)                         | `app.py`, `data_loader.py`              |
| Tests                                                                                               | `tests/test_pipeline.py`                |

## KPI definitions

Total Sales = Σ Sales · Total Profit = Σ Profit · Total Orders = unique Order IDs · Quantity Sold = Σ Quantity · Profit Margin % = Total Profit ÷ Total Sales × 100

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest tests
```

## Cleaning rules (summary)

Trim whitespace; placeholders (`N/A`, `unknown`, `?`) → missing; drop empty rows and rows with no Order ID; unify case/spelling (`consumer`, `SameDay`, `Office supply`); convert `five` / `ten percent`; parse mixed date formats and resolve dd-mm vs mm-dd using the order date; ship date before order date → missing; drop fully identical rows; repair Category←Sub-Category, State←City, Region←State; invalid values (sales ≤ 0, quantity ≤ 0, discount outside 0–1, extreme Sales/Quantity outliers) → missing; then impute with Sub-Category medians (profit = sales × median margin) or drop rows (user choice).
