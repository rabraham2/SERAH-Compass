# SERAH - Compass (Commerce Optimization, Modeling, Pricing, Analytics, Strategy & Scenarios)

SERAH Compass is an end-to-end retail analytics pipeline on the Olist dataset. It cleans and aggregates data into weekly category features, trains a leak-safe XGBoost model, forecasts 8 weeks into the future, runs price-elasticity scenarios, and outputs profit-aware price recommendations with shareable one-pagers.

---

## Table of Contents
1. [Overview](#overview)
2. [Primary Research Question](#primary-research-question)
3. [System Requirements](#system-requirements)
4. [Installation and Setup](#installation-and-setup)
5. [Instructions to Run](#instructions)
6. [Scripts and Roles](#scripts-and-roles)
7. [Running the Scripts](#running-the-scripts)
8. [Project Structure](#project-structure)
9. [License](#license)

---


## Overview
Predict weekly demand by product category, generate 8-week forecasts, explore price scenarios (±5/10/15%), and recommend profit-aware price moves with one-pager/PDF/PPT outputs.

This project turns the public Olist Brazilian e-commerce data into a production-style pipeline:

1. Clean & normalise raw orders/items/products/reviews
2. Engineer leak-safe weekly features per category
3. Train an XGBoost pipeline to predict weekly units
4. Validate & calibrate predictions to observed scale
5. Forecast 8 weeks into the future (baseline & planned prices)
6. Run elasticity scenarios (price ±5/10/15%)
7. Select recommendations by revenue and profit lift
8. Publish narratives: per-category one-pagers, combined PDF, PPTX

It’s designed to be reproducible, leak-safe, and actionable—the outputs are concrete pricing actions with expected units/revenue/profit impact plus ready-to-share visuals.

---

## Primary Research Question
Given historical order, item price, category and lagged review signals, how should we adjust category-level prices next quarter to maximise revenue or profit while protecting volume?


- **Aims and Objectives**:

  a. Build robust weekly × category demand models (no leakage)
  
  b. Produce 8-week forecasts with clear baselines
  
  c. Quantify price sensitivity via scenario simulation & elasticity
  
  d. Recommend price deltas that optimise revenue and/or profit
  
  e. Communicate results via KPIs, one-pagers, PDF, PPTX


- **Dataset (What, Why, & How)**:

  Olist Brazilian e-commerce dataset (2016–2018, ≈100k orders) covers the full funnel:
  orders → items → products → sellers → payments → logistics → reviews, plus geolocation.
  
    A - Raw grain: order item line (multiple items per order; different sellers possible)
    
    B - Modelling grain: Monday-anchored week × English product category
    
    C - Signals: units, revenue, price proxy, lagged reviews, seasonal encodings, leak-safe lags/rolls
    
    D - Why it fits: rich funnel, enough weekly history for many categories, clear causal anchor (purchase time)
  
  We anchor time at order purchase. Any post-purchase fields (e.g., delivery, reviews) should be entered only as lagged/rolled features to prevent leakage.

  <i>A comprehensive Data Dictionary is included as <b>Data Dictionary.md</b>.</i>


- **Methods Used**:

  a) Feature engineering - Weekly aggregation (Mon-anchored), seasonality (weekofyear, sin/cos), leak-safe lags {1,2,4,8,12,52}, rolling means, review features aggregated per order → per week → then lagged/rolled.

  b) Modeling - scikit-learn Pipeline: OneHotEncoder (category) + XGBRegressor (units). Per-category temporal split: last ≈20% (cap 12 weeks) as validation.

  c) Calibration - Linear per-category: units_cal = a + b * units_base.

  d) Scenario & elasticity - Step-ahead 8-week recursion with planned prices; scenarios ±5/10/15%; elasticity = %ΔQ / %ΔP.

  e) Profit (optional) - unit_cost per category (Dataset/cost_per_category.csv); profit = (price − unit_cost) × units. Guardrails configurable.

  f) Readouts - Validation parity & overlays; KPIs by category & overall; category one-pagers (PNG), PDF, PPTX.
  
---

## System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version:**: Python 3.10–3.12
- **Python Packages**:
  - pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl joblib python-pptx
  - holidays (optional as needed and specified in the code)
- **Memory**: 4–8 GB is plenty for Olist

---

## Installation and Setup
1. **Clone the Repository**:
   - Click the green "**Code**" button in this GitHub repository.
   - Copy the HTTPS or SSH link provided (e.g., `https://github.com/rabraham2/SERAH-Compass.git`).
   - Open your terminal or command prompt and run:
     ```bash
     git clone https://github.com/rabraham2/SERAH-Compass.git
     ```

2. **Install Required Python Packages**:
   Open **PyCharm** or **Other IDE-Enabled Coding Platform** and install the necessary packages:

```python
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl joblib python-pptx holidays
```

3. **Processed and Cleaned Dataet**:
   - The original and unprocessed version of the real dataset can be found in the folder Data in GitHub.
   - If you need to run the processed and data-cleaned version of the dataset after running the Data Preparation Script directly, use the file in the folder Dataset in GitHub.

---

## Instructions

Step 1: Audit & Missingness
  - Produces: audit_exports/missingness_summary.csv, overview tables.

Step 2: Cleaning → clean_exports/
  - Output: fact_orders_clean.csv (order-date × category with units/revenue/avg_price)
  - Output: reviews_by_order.csv (order-level review_count, review_score_avg)

Step 3: Weekly features (no leak)
  - Output: model_outputs/weekly_by_category_full.csv
  - Output: model_outputs/weekly_by_category_no_leak.csv (lags/rolls + split)

Step 4: Train & Validate
  - Output: model_outputs/weekly_units_xgb_no_leak.pkl
  - Output: model_outputs/validation_metrics.csv, valid_predictions.csv
  - Plots: parity scatter, per-category overlays

Step 5: Calibration (optional but recommended)
  - Output: model_outputs/calibration_params.csv (a,b per category)

Step 6: Forecast (baseline plan)
  - Output: model_outputs/forecast_unified.csv (8-week, base+cal)
  - Output: model_outputs/forecast_summary.csv
  - Output: model_outputs/forecast_rollup_by_category.csv, ..._overall.csv
  - Plots: overall/base vs cal, top categories

Step 7: Elasticity scenarios (±5/10/15%)
  - Output: model_outputs/elasticity_scenarios_detailed.csv
  - Output: model_outputs/elasticity_deltas_by_category.csv
  - Plots: ΔUnits/ΔRevenue bars per scenario, top-sensitivity chart

Step 8: Recommendations (revenue & profit)
  - Output: model_outputs/elasticity_recommendations_by_category.csv
  - Output: model_outputs/elasticity_recommendations_profit.csv
  - Output: model_outputs/price_recommendations_summary.xlsx (All, Safe_Top, Risks)
  - Plots: top lifts, biggest risks, safe top

Step 9: Narratives
  - Output: model_outputs/onepagers/onepager_<category>.png (×N)
  - Output: model_outputs/category_onepagers.pdf
  - Output: model_outputs/category_onepagers.pptx

Step 10: Price plan forecasting
  - Generate a plan CSV with order_week, product_category_name_english, planned price and re-forecast using the scenario forecaster.
 
---

## Scripts and Roles

|  #  | Script (role)                         | What it does                          | Key inputs                             |                  Key outputs                     |
|-----|---------------------------------------|---------------------------------------|----------------------------------------|--------------------------------------------------|
|  1  | data_cleaning.py                      | Missingness & dataset overview        | Dataset/*.csv                          | audit_exports/missingness_summary.csv and        |
|     |                                       |                                       |                                        | overview tables                                  | 
|     |                                       |                                       |                                        |                                                  |
|  2  | make_margins_template.py              | Build fact_orders_clean               | weekly_by_category.csv                 | margins.csv                                      |
|     |                                       | (order-date × category),              |                                        |                                                  |
|     |                                       | compute avg_price, units, and         |                                        |                                                  |
|     |                                       | revenue                               |                                        |                                                  |
|     |                                       |                                       |                                        |                                                  | 
|  3  | select_actions_with_budget.py         | Order-level review aggregates         | recommended_actions.csv;               | selected_actions.csv                             |
|     |                                       |                                       | weekly_by_category.csv                 |                                                  |
|     |                                       |                                       |                                        |                                                  |
|  4  | simulate_price_change.py              | weekly_by_category.csv;               | whatif_impact_by_category_10pct.csv;   | weekly_by_category_full.csv                      |
|     |                                       |                                       | whatif_summary_10pct_markdown.csv      |                                                  |
|     |                                       |                                       |                                        |                                                  |
|  5  | build_weekly_features_no_leak.py      | Add leak-safe lags/rolls + split      | weekly_by_category_full.csv            | weekly_by_category_no_leak.csv                   |
|     |                                       |                                       |                                        |                                                  |
|  6  | build_and_train.py                    | Train OHE+XGB on units, save model    | weekly_by_category_no_leak.csv         | weekly_units_xgb_no_leak.pkl,                    |
|     |                                       |                                       |                                        | validation_metrics.csv, valid_predictions.csv    |
|     |                                       |                                       |                                        |                                                  |
|  7  | compute_plan_impact.py                | Compute the best course of action     | forecast_with_plan.csv;                | Forecast with Plan                               |
|     |                                       |                                       | weekly_by_category.csv;                |                                                  |
|     |                                       |                                       | weekly_units_xgb.pkl                   |                                                  |
|     |                                       |                                       |                                        |                                                  | 
|  8  | plot_validation_parity.py             | Overall parity scatter + diagnostics  | valid_predictions.csv                  | val_scatter_overall.png                          |
|     |                                       |                                       |                                        |                                                  |
|  9  | plot_validation_overlays.py           | Per-category time-series overlays     | valid_predictions.csv, features        | val_overlay_*.png                                |
|     |                                       | (actual vs pred)                      |                                        |                                                  |
|     |                                       |                                       |                                        |                                                  |
| 10  | evaluate_validation.py                | Aggregate by category, worst cases,   | valid_predictions.csv                  | calibration_params.csv,                          |
|     |                                       | calibration fit                       |                                        | validation_overall_metrics.csv,                  |
|     |                                       |                                       |                                        | validation_by_category.csv, val_worst_*.png      |
|     |                                       |                                       |                                        |                                                  |
| 11  | forecast_make_plan.py                 | Build flat price plan (next 8 Mondays)| weekly_by_category_full.csv,           | price_plan.csv                                   |
|     |                                       |                                       | selections                             |                                                  |
|     |                                       |                                       |                                        |                                                  |
| 12  | forecast_with_plan.py                 | Roll forward 8w using plan,           | price_plan.csv, model                  | forecast_with_plan.csv                           |
|     |                                       | recompute lags each step              |                                        |                                                  |
|     |                                       |                                       |                                        |                                                  |
| 13  | forecast_unified_rollup.py            | Merge base/cal, rollups & summary     | forecast_with_plan.csv or              | forecast_unified.csv,                            |                                 |     |                                       |                                       | baseline                               | forecast_rollup_by_category.csv,                 |   
|     |                                       |                                       |                                        | forecast_rollup_overall.csv, forecast_summary.csv|                                 |     |                                       |                                       |                                        |                                                  |
| 14  | calibration_scatter.py                | Base vs calibrated scatter per        | calibration_params.csv, preds          | calibration_scatter_*.png                        |
|     |                                       | category                              |                                        |                                                  |
|     |                                       |                                       |                                        |                                                  |
| 15  | elasticity_scenarios.py               | Scenarios ±5/10/15% and detail outputs| history & model                        | elasticity_scenarios_detailed.csv                |
|     |                                       |                                       |                                        |                                                  |
| 16  | elasticity_deltas.py                  | Aggregate deltas vs baseline,         | scenarios detailed                     | elasticity_deltas_by_category.csv                |
|     |                                       | compute elasticity                    |                                        |                                                  |
|     |                                       |                                       |                                        |                                                  |
| 17  | elasticity_plots.py                   | ΔUnits/ΔRevenue bars, top-sensitivity | deltas                                 | elasticity_delta_units_*.png,                    |
|     |                                       |                                       |                                        | elasticity_delta_revenue_*.png,                  |
|     |                                       |                                       |                                        | elasticity_top_sensitive.png                     |
|     |                                       |                                       |                                        |                                                  |
| 18  | elasticity_best_worst.py              | Best-gain & worst-risk tables/plots   | deltas                                 | elasticity_best_by_abs_revenue.csv,              |
|     |                                       |                                       |                                        | elasticity_worst_negative_revenue.csv, plots     |
|     |                                       |                                       |                                        |                                                  |
| 19  | simulate_price_change.py              | Quick “what-if” helper by %           | features, model                        | summary/impact CSVs (ad-hoc)                     |
|     |                                       |                                       |                                        |                                                  |
| 20  | recommendations_revenue.py            | Select actions by revenue/volume      | deltas                                 | elasticity_recommendations_by_category.csv       |
|     |                                       |                                       |                                        |                                                  |
| 21  | recommendations_profit.py             | Profit-aware picks using costs &      | deltas, cost_per_category.csv          | elasticity_recommendations_profit.csv            |
|     |                                       | guardrails                            |                                        |                                                  |
|     |                                       |                                       |                                        |                                                  |
| 22  | export_recommendations_excel.py       | Excel pack (All, Safe_Top, Risks)     | recos                                  | price_recommendations_summary.xlsx               |
|     |                                       |                                       |                                        |                                                  |
| 23  | elasticity_recommendations_profit.py  | Overall/base vs cal & top cat KPIs    | rollups                                | overall_units_base_vs_cal.png,                   |
|     |                                       |                                       |                                        | overall_revenue_base_vs_cal.png,                 |
|     |                                       |                                       |                                        | top10_categories_revenue.png                     |
|     |                                       |                                       |                                        |                                                  |
| 24  | one_pagers.py                         | Category one-pagers (history, base vs | scenarios detailed, history,           | onepagers/onepager_*.png, category_onepagers.pdf |
|     |                                       | prop, KPIs)                           | recos                                  |                                                  |
|     |                                       |                                       |                                        |                                                  |
| 25  | price_move_recommendations.py         | Slide deck with one-pagers            | one-pager PNGs                         | category_onepagers.pptx                          |
|     |                                       |                                       |                                        | elasticity_recommendations_profit.csv            |
|     |                                       |                                       |                                        |                                                  |
| 26  | stakeholder_slide_deck.py             | stakeholder_slide_deck.py             | Export PowerPoint                      |                                                  |
|     |                                       |                                       |                                        |                                                  |
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
---

## Running the Scripts

```Python Code

A]--------> ## data_cleaning.py  ##
# Data Analysis Summary

import pandas as pd
from pathlib import Path

base = Path("Dataset")

# Load
orders = pd.read_csv(base / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"])
order_items = pd.read_csv(base / "olist_order_items_dataset.csv")
products = pd.read_csv(base / "olist_products_dataset.csv")
cats = pd.read_csv(base / "product_category_name_translation.csv")
order_payments = pd.read_csv(base / "olist_order_payments_dataset.csv")
reviews = pd.read_csv(base / "olist_order_reviews_dataset.csv", parse_dates=["review_creation_date","review_answer_timestamp"])
customers = pd.read_csv(base / "olist_customers_dataset.csv")
sellers = pd.read_csv(base / "olist_sellers_dataset.csv")
geo = pd.read_csv(base / "olist_geolocation_dataset.csv")

# Merge English category
products = products.merge(cats, on="product_category_name", how="left")

# Helper for NA report
def na_report(df, cols, name):
    tmp = pd.DataFrame({
        "table": name,
        "column": cols,
        "n_missing": [df[c].isna().sum() for c in cols],
        "pct_missing": [float(df[c].isna().mean()) for c in cols],
        "n_rows": len(df)
    })
    return tmp

rep_orders = na_report(orders, ["order_status","order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date","customer_id"], "orders")
rep_items  = na_report(order_items, ["order_id","product_id","seller_id","price","freight_value"], "order_items")
rep_products = na_report(products, ["product_category_name","product_category_name_english","product_weight_g","product_length_cm","product_height_cm","product_width_cm"], "products")
rep_pay = na_report(order_payments, ["order_id","payment_type","payment_installments","payment_value"], "payments")
rep_rev = na_report(reviews, ["order_id","review_score","review_comment_title","review_comment_message"], "reviews")
rep_cust = na_report(customers, ["customer_id","customer_city","customer_state","customer_zip_code_prefix"], "customers")
rep_sell = na_report(sellers, ["seller_id","seller_city","seller_state","seller_zip_code_prefix"], "sellers")
rep_geo = na_report(geo, ["geolocation_zip_code_prefix","geolocation_lat","geolocation_lng","geolocation_city","geolocation_state"], "geolocation")

na_summary = pd.concat([rep_orders, rep_items, rep_products, rep_pay, rep_rev, rep_cust, rep_sell, rep_geo], ignore_index=True)

# Order status distribution + date range overview
status_counts = orders["order_status"].value_counts().rename_axis("order_status").reset_index(name="count")
orders_total = orders["order_id"].nunique()
orders_with_reviews = reviews["order_id"].nunique()

overview = pd.DataFrame([{
    "orders_date_min": str(orders["order_purchase_timestamp"].min()),
    "orders_date_max": str(orders["order_purchase_timestamp"].max()),
    "orders_total": int(orders_total),
    "order_items_rows": int(len(order_items)),
    "products_rows": int(len(products)),
    "payments_rows": int(len(order_payments)),
    "reviews_rows": int(len(reviews)),
    "customers_rows": int(len(customers)),
    "sellers_rows": int(len(sellers)),
    "geolocation_rows": int(len(geo)),
    "orders_with_reviews_pct": float(orders_with_reviews / max(1, orders_total)),
    "qty_column_present_in_items": "quantity" in order_items.columns.tolist()
}])

# Save files
audit_dir = base / "audit_exports"
audit_dir.mkdir(exist_ok=True)

na_summary_path = audit_dir / "missingness_summary.csv"
status_counts_path = audit_dir / "order_status_counts.csv"
overview_path = audit_dir / "dataset_overview.csv"

na_summary.to_csv(na_summary_path, index=False)
status_counts.to_csv(status_counts_path, index=False)
overview.to_csv(overview_path, index=False)

{
  "exports": {
    "missingness_summary": str(na_summary_path),
    "order_status_counts": str(status_counts_path),
    "dataset_overview": str(overview_path)
  }
}


# Data Cleaning for All Columns
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("Dataset")
OUT  = BASE / "clean_exports"
OUT.mkdir(exist_ok=True)


# 0) Load raw

orders = pd.read_csv(
    BASE/"olist_orders_dataset.csv",
    parse_dates=[
        "order_purchase_timestamp","order_approved_at",
        "order_delivered_carrier_date","order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
)
order_items   = pd.read_csv(BASE/"olist_order_items_dataset.csv")
products_raw  = pd.read_csv(BASE/"olist_products_dataset.csv")
cats          = pd.read_csv(BASE/"product_category_name_translation.csv")
order_payments= pd.read_csv(BASE/"olist_order_payments_dataset.csv")
reviews       = pd.read_csv(BASE/"olist_order_reviews_dataset.csv",
                            parse_dates=["review_creation_date","review_answer_timestamp"])
customers     = pd.read_csv(BASE/"olist_customers_dataset.csv")
sellers       = pd.read_csv(BASE/"olist_sellers_dataset.csv")
geo           = pd.read_csv(BASE/"olist_geolocation_dataset.csv")


# 1) Products: categories + dimensions

products = products_raw.merge(cats, on="product_category_name", how="left")
products["product_category_name_english"] = products["product_category_name_english"].fillna("unknown")

# Impute missing dimensions by category median (fallback to global median)
dim_cols = ["product_weight_g","product_length_cm","product_height_cm","product_width_cm"]
cat_medians = products.groupby("product_category_name_english")[dim_cols].median()

for c in dim_cols:
    # replace zeros or negatives with NaN (bad quality)
    products.loc[products[c].le(0, fill_value=False), c] = np.nan
    # category median
    products[c] = products.apply(
        lambda r: r[c] if pd.notna(r[c])
        else cat_medians.loc[r["product_category_name_english"], c]
             if r["product_category_name_english"] in cat_medians.index and
                pd.notna(cat_medians.loc[r["product_category_name_english"], c])
        else products[c].median(),
        axis=1
    )


# 2) Geolocation → postal prefix reference (mode city/state, mean lat/lng)

# Normalise zip prefixes as strings
for zc in ["geolocation_zip_code_prefix"]:
    geo[zc] = geo[zc].astype(str)

geo_ref = (geo.groupby("geolocation_zip_code_prefix")
             .agg(geo_city=("geolocation_city", lambda s: s.mode(dropna=True)[0] if not s.mode(dropna=True).empty else np.nan),
                  geo_state=("geolocation_state", lambda s: s.mode(dropna=True)[0] if not s.mode(dropna=True).empty else np.nan),
                  geo_lat=("geolocation_lat","mean"),
                  geo_lng=("geolocation_lng","mean"))
             .reset_index())


# 3) Customers & Sellers: fill city/state via geo_ref by zip prefix

customers["customer_zip_code_prefix"] = customers["customer_zip_code_prefix"].astype(str)
sellers["seller_zip_code_prefix"]     = sellers["seller_zip_code_prefix"].astype(str)

cust = customers.merge(
    geo_ref, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left"
)
cust["customer_city"]  = cust["customer_city"].fillna(cust["geo_city"])
cust["customer_state"] = cust["customer_state"].fillna(cust["geo_state"])
cust = cust.drop(columns=["geolocation_zip_code_prefix","geo_city","geo_state","geo_lat","geo_lng"])

sell = sellers.merge(
    geo_ref, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left"
)
sell["seller_city"]  = sell["seller_city"].fillna(sell["geo_city"])
sell["seller_state"] = sell["seller_state"].fillna(sell["geo_state"])
sell = sell.drop(columns=["geolocation_zip_code_prefix","geo_city","geo_state","geo_lat","geo_lng"])

# If still missing state/city, fill with most common values
cust["customer_state"] = cust["customer_state"].fillna(cust["customer_state"].mode(dropna=True).iloc[0])
cust["customer_city"]  = cust["customer_city"].fillna("unknown_city")
sell["seller_state"]   = sell["seller_state"].fillna(sell["seller_state"].mode(dropna=True).iloc[0])
sell["seller_city"]    = sell["seller_city"].fillna("unknown_city")


# 4) Orders: keep status logic & safe durations

# Only compute delivered-based durations when dates exist
orders["purchase_to_delivery_days"] = np.where(
    orders["order_status"].eq("delivered") &
    orders["order_delivered_customer_date"].notna(),
    (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days,
    np.nan
)

# For modelling demand/revenue, typically exclude cancelled/unavailable
orders["is_valid_for_sales"] = orders["order_status"].isin(["delivered","shipped","invoiced"])


# 5) Order items: critical IDs must exist; price/freight clean-up

items = order_items.copy()

# Drop rows missing critical IDs
items = items.dropna(subset=["order_id","product_id","seller_id"])

# Bad price/freight handling
items["price"] = pd.to_numeric(items["price"], errors="coerce")
items["freight_value"] = pd.to_numeric(items["freight_value"], errors="coerce")

# flag & fix nonpositive price → set NaN, will drop later if cannot be imputed
items.loc[items["price"]<=0, "price"] = np.nan
# negative freight is invalid → set NaN
items.loc[items["freight_value"]<0, "freight_value"] = np.nan

# Impute freight by seller_state median freight when available, else global median
tmp = items.merge(sell[["seller_id","seller_state"]], on="seller_id", how="left")
state_med = tmp.groupby("seller_state")["freight_value"].median()
items = items.merge(sell[["seller_id","seller_state"]], on="seller_id", how="left")
items["freight_value"] = items.apply(
    lambda r: r["freight_value"] if pd.notna(r["freight_value"])
    else state_med.get(r["seller_state"], np.nan), axis=1
)
items["freight_value"] = items["freight_value"].fillna(items["freight_value"].median())

# Price imputation is risky; better to drop if price still missing
before = len(items)
items = items.dropna(subset=["price"])
dropped_price_na = before - len(items)

# Each row is one unit (no quantity column). Add units=1 for clarity
items["units"] = 1


# 6) Payments: tidy and aggregate by order

pays = order_payments.copy()
pays["payment_installments"] = pays["payment_installments"].fillna(1).astype(int)
pays["payment_type"] = pays["payment_type"].fillna("unknown")
pays["payment_value"] = pd.to_numeric(pays["payment_value"], errors="coerce")

# If any payment_value missing, impute with 0 for aggregation; we use sums at order level
pays["payment_value"] = pays["payment_value"].fillna(0.0)
pay_agg = (pays.groupby("order_id")
              .agg(payment_total=("payment_value","sum"),
                   n_payments=("payment_value","size"),
                   installments_max=("payment_installments","max"),
                   has_cc=("payment_type", lambda s: int(any(s.str.contains("credit", case=False, na=False)))))
              .reset_index())


# 7) Reviews: keep score rows; text can be empty

rev = reviews.copy()
rev["review_score"] = pd.to_numeric(rev["review_score"], errors="coerce")
rev = rev.dropna(subset=["review_score"])
rev["review_comment_title"] = rev["review_comment_title"].fillna("")
rev["review_comment_message"] = rev["review_comment_message"].fillna("")
rev_agg = (rev.groupby("order_id")
             .agg(review_count=("review_id","count"),
                  review_score_avg=("review_score","mean"))
             .reset_index())


# 8) Build a clean FACT (order line) for modelling

orders_sub = orders[["order_id","customer_id","order_status","order_purchase_timestamp","is_valid_for_sales"]].copy()

fact = (items
        .merge(orders_sub, on="order_id", how="left")
        .merge(products[["product_id","product_category_name_english","product_weight_g","product_length_cm","product_height_cm","product_width_cm"]],
               on="product_id", how="left")
        .merge(sell[["seller_id","seller_city","seller_state"]], on="seller_id", how="left")
        .merge(cust[["customer_id","customer_city","customer_state"]], on="customer_id", how="left")
        .merge(pay_agg, on="order_id", how="left")
        .merge(rev_agg, on="order_id", how="left")
       )

# Keep only rows tied to valid sales for demand/revenue (you can relax this if needed)
fact = fact[fact["is_valid_for_sales"].fillna(False)]

# Derive order_date and revenue fields
fact["order_date"] = pd.to_datetime(fact["order_purchase_timestamp"]).dt.date
fact["order_date"] = pd.to_datetime(fact["order_date"])
fact["revenue_item"] = fact["price"].astype(float)
fact["freight"] = fact["freight_value"].astype(float)


# 9) Save cleaned outputs + a small data-quality log

products.to_csv(OUT/"products_clean.csv", index=False)
cust.to_csv(OUT/"customers_clean.csv", index=False)
sell.to_csv(OUT/"sellers_clean.csv", index=False)
pay_agg.to_csv(OUT/"payments_by_order.csv", index=False)
rev_agg.to_csv(OUT/"reviews_by_order.csv", index=False)
fact.to_csv(OUT/"fact_orders_clean.csv", index=False)

log = pd.DataFrame([{
    "dropped_items_price_na": int(dropped_price_na),
    "remaining_fact_rows": int(len(fact)),
    "products_missing_english_category_now": int(products["product_category_name_english"].eq("unknown").sum()),
    "customers_missing_state_after_fill": int(cust["customer_state"].isna().sum()),
    "sellers_missing_state_after_fill": int(sell["seller_state"].isna().sum()),
}])
log.to_csv(OUT/"cleaning_log.csv", index=False)

print("Saved to:", OUT.resolve())


B]--------> ## build_weekly_features.py ##

# Weekly Features Building
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("Dataset/clean_exports")
OUT  = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Load
# 'fact_orders_clean.csv' may store dates as dd-mm-YYYY; parse robustly
fact = pd.read_csv(BASE / "fact_orders_clean.csv")
fact["order_date"] = pd.to_datetime(fact["order_date"], dayfirst=True, errors="coerce")

# Reviews aggregation (ensure presence & aligned dtypes)
rev_path = BASE / "reviews_by_order.csv"
if rev_path.exists():
    rev = pd.read_csv(rev_path)
else:
    # fallback to raw if needed (optional)
    raw = pd.read_csv(
        Path("Dataset") / "olist_order_reviews_dataset.csv",
        parse_dates=["review_creation_date", "review_answer_timestamp"]
    )
    rev = (raw.groupby("order_id", as_index=False)
              .agg(review_count=("review_id", "count"),
                   review_score_avg=("review_score", "mean")))

# Align merge keys as string
fact["order_id"] = fact["order_id"].astype(str)
rev["order_id"]  = rev["order_id"].astype(str)

# Safe merge; guarantee columns
fact = fact.merge(rev[["order_id", "review_count", "review_score_avg"]],
                  on="order_id", how="left")
if "review_count" not in fact.columns:
    fact["review_count"] = 0
if "review_score_avg" not in fact.columns:
    fact["review_score_avg"] = np.nan

# Fill sensible defaults
fact["review_count"] = fact["review_count"].fillna(0)

# Weekly aggregation
# Monday-start week (stable weekly grain)
fact["order_week"] = fact["order_date"] - pd.to_timedelta(fact["order_date"].dt.weekday, unit="D")

# Defensive fill for category
fact["product_category_name_english"] = fact["product_category_name_english"].fillna("unknown")

wk = (fact.groupby(["order_week", "product_category_name_english"], as_index=False)
          .agg(
              units = ("units", "sum"),
              revenue = ("revenue_item", "sum"),
              avg_price = ("revenue_item", "mean"),
              rev_count = ("review_count", "sum"),
              rev_score_avg = ("review_score_avg", "mean")
          ))

# Fill review score gaps with a rolling mean per category
# Use group_keys=False so index stays aligned with wk
wk = wk.sort_values(["product_category_name_english", "order_week"]).reset_index(drop=True)
wk["rev_score_avg"] = (wk.groupby("product_category_name_english", group_keys=False)["rev_score_avg"]
                         .apply(lambda s: s.fillna(s.rolling(8, min_periods=1).mean())))

# Calendar & seasonal features
wk["weekofyear"] = wk["order_week"].dt.isocalendar().week.astype(int)
wk["year"]       = wk["order_week"].dt.year
wk["sin_woy"]    = np.sin(2*np.pi*wk["weekofyear"]/52.0)
wk["cos_woy"]    = np.cos(2*np.pi*wk["weekofyear"]/52.0)

# Optional: Brazil holiday flag (safe fallback if holidays not installed)
try:
    from holidays import Brazil
    years = range(int(wk["year"].min()), int(wk["year"].max()) + 1)
    br_dates = set(pd.to_datetime(list(Brazil(years=years))).date)
    wk["is_holiday_week"] = wk["order_week"].dt.date.isin(br_dates).astype(int)
except Exception:
    wk["is_holiday_week"] = 0

# Lags & rolling stats
def add_lags(df: pd.DataFrame, col: str, lags=(1, 2, 4, 8, 12, 52)) -> pd.DataFrame:
    for L in lags:
        df[f"{col}_lag{L}"] = (
            df.groupby("product_category_name_english", group_keys=False)[col]
              .shift(L)
        )
    return df

for col in ["units", "revenue", "avg_price", "rev_count", "rev_score_avg"]:
    wk = add_lags(wk, col)

# Helpful short rolling means
wk["units_roll4"]   = (wk.groupby("product_category_name_english", group_keys=False)["units"]
                         .transform(lambda s: s.rolling(4, min_periods=1).mean()))
wk["revenue_roll4"] = (wk.groupby("product_category_name_english", group_keys=False)["revenue"]
                         .transform(lambda s: s.rolling(4, min_periods=1).mean()))

# Save full features
wk_full_path = OUT / "weekly_by_category_full.csv"
wk.to_csv(wk_full_path, index=False)

# Build model-ready view: drop warm-up rows missing any lag
lag_cols = [c for c in wk.columns if "lag" in c]
wk_model = wk.dropna(subset=lag_cols).copy()

# Time-based split flag: last 12 weeks per category -> validation
def flag_split(g: pd.DataFrame, val_weeks=12) -> pd.DataFrame:
    g = g.sort_values("order_week").copy()
    g["split"] = "train"
    if len(g) > val_weeks:
        g.loc[g.index[-val_weeks:], "split"] = "valid"
    return g

# Vectorised time-based split: last 12 rows per category → "valid"
wk_model = wk_model.sort_values(["product_category_name_english","order_week"]).copy()
g = wk_model.groupby("product_category_name_english")
n = g["order_week"].transform("size")
idx_in_group = g.cumcount()  # 0..n-1
wk_model["split"] = np.where(idx_in_group >= (n - 12), "valid", "train")

wk_model_path = OUT / "weekly_by_category.csv"
wk_model.to_csv(wk_model_path, index=False)

print("Saved:", wk_full_path)
print("Saved (model-ready):", wk_model_path)

C]--------> ## train_forecast.py ##

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

BASE_IN  = Path("Dataset/clean_exports")
RAW_IN   = Path("Dataset")
BASE_OUT = Path("Dataset/model_outputs")
BASE_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 0) Load clean fact
# ---------------------------
fact = pd.read_csv(BASE_IN / "fact_orders_clean.csv")
# Robust date parsing (handles dd-mm-YYYY)
fact["order_date"] = pd.to_datetime(fact["order_date"], dayfirst=True, errors="coerce")

required_cols = {"order_id","order_date","units","revenue_item","product_category_name_english"}
missing_req = required_cols - set(fact.columns)
if missing_req:
    raise RuntimeError(f"Missing required columns in fact_orders_clean.csv: {missing_req}")

fact["order_id"] = fact["order_id"].astype(str)
fact["product_category_name_english"] = fact["product_category_name_english"].fillna("unknown")

# ---------------------------
# 1) Merge reviews robustly (or rebuild if needed)
# ---------------------------
def load_or_build_reviews():
    rev_path = BASE_IN / "reviews_by_order.csv"
    if rev_path.exists():
        rev = pd.read_csv(rev_path)
        rev.columns = [c.strip().lower() for c in rev.columns]
        # If someone saved raw reviews by mistake, aggregate
        if "review_count" not in rev.columns and "review_id" in rev.columns:
            raw = pd.read_csv(RAW_IN / "olist_order_reviews_dataset.csv",
                              parse_dates=["review_creation_date","review_answer_timestamp"])
            return (raw.groupby("order_id", as_index=False)
                        .agg(review_count=("review_id","count"),
                             review_score_avg=("review_score","mean")))
        if "review_score_avg" not in rev.columns and "review_score" in rev.columns:
            rev = rev.rename(columns={"review_score":"review_score_avg"})
        # Ensure columns
        for col in ["order_id","review_count","review_score_avg"]:
            if col not in rev.columns:
                raw = pd.read_csv(RAW_IN / "olist_order_reviews_dataset.csv",
                                  parse_dates=["review_creation_date","review_answer_timestamp"])
                return (raw.groupby("order_id", as_index=False)
                            .agg(review_count=("review_id","count"),
                                 review_score_avg=("review_score","mean")))
        return rev

    raw_reviews = RAW_IN / "olist_order_reviews_dataset.csv"
    if not raw_reviews.exists():
        return pd.DataFrame({"order_id": [], "review_count": [], "review_score_avg": []})
    raw = pd.read_csv(raw_reviews,
                      parse_dates=["review_creation_date","review_answer_timestamp"])
    return (raw.groupby("order_id", as_index=False)
              .agg(review_count=("review_id","count"),
                   review_score_avg=("review_score","mean")))

rev = load_or_build_reviews()
if "order_id" not in rev.columns:
    raise RuntimeError("Could not build a review table with 'order_id'.")
rev["order_id"] = rev["order_id"].astype(str)
for col in ["review_count","review_score_avg"]:
    if col not in rev.columns:
        rev[col] = np.nan

fact = fact.merge(rev[["order_id","review_count","review_score_avg"]], on="order_id", how="left")
if "review_count" not in fact.columns:      fact["review_count"] = 0
if "review_score_avg" not in fact.columns:  fact["review_score_avg"] = np.nan
fact["review_count"] = fact["review_count"].fillna(0)

# ---------------------------
# 2) Weekly aggregate (category x week)
# ---------------------------
fact["order_week"] = fact["order_date"] - pd.to_timedelta(fact["order_date"].dt.weekday, unit="D")

wk = (fact.groupby(["order_week","product_category_name_english"], as_index=False)
          .agg(units=("units","sum"),
               revenue=("revenue_item","sum"),
               avg_price=("revenue_item","mean"),
               rev_count=("review_count","sum"),
               rev_score_avg=("review_score_avg","mean")))

if wk.empty:
    raise RuntimeError("Weekly aggregate is empty. Check order_date parsing and that fact_orders_clean.csv has rows.")

wk = wk.sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)

# Fill review score gaps with rolling mean (index-safe)
wk["rev_score_avg"] = (
    wk.groupby("product_category_name_english", group_keys=False)["rev_score_avg"]
      .apply(lambda s: s.fillna(s.rolling(8, min_periods=1).mean()))
)

# Calendar/seasonal features
wk["weekofyear"] = wk["order_week"].dt.isocalendar().week.astype(int)
wk["year"]       = wk["order_week"].dt.year
wk["sin_woy"]    = np.sin(2*np.pi*wk["weekofyear"]/52.0)
wk["cos_woy"]    = np.cos(2*np.pi*wk["weekofyear"]/52.0)

# Optional holiday flag (safe fallback)
try:
    from holidays import Brazil
    years = range(int(wk["year"].min()), int(wk["year"].max()) + 1)
    br_dates = set(pd.to_datetime(list(Brazil(years=years))).date)
    wk["is_holiday_week"] = wk["order_week"].dt.date.isin(br_dates).astype(int)
except Exception:
    wk["is_holiday_week"] = 0

# ---------------------------
# 3) Add lags — vectorised (no groupby.apply, no warnings)
# ---------------------------
ALLOWED_LAGS = [1, 2, 4, 8, 12, 52]
grp = wk.groupby("product_category_name_english", sort=False)

for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
    for L in ALLOWED_LAGS:
        wk[f"{col}_lag{L}"] = grp[col].shift(L)

# Short rolling means (helpful, no leakage)
wk["units_roll4"]   = grp["units"].transform(lambda s: s.rolling(4, min_periods=1).mean())
wk["revenue_roll4"] = grp["revenue"].transform(lambda s: s.rolling(4, min_periods=1).mean())

# Adaptive warm-up trim per category: drop the first max usable lag rows
sizes = grp.size()  # Series: category -> count
def max_lag_for_n(n):
    usable = [L for L in ALLOWED_LAGS if L < n]
    return max(usable) if usable else 0
maxlag_map = {cat: max_lag_for_n(int(n)) for cat, n in sizes.items()}
wk["max_warmup"] = wk["product_category_name_english"].map(maxlag_map).fillna(0).astype(int)

idx_in_group = grp.cumcount()
wk_model = wk[idx_in_group >= wk["max_warmup"]].copy()
wk_model.drop(columns=["max_warmup"], inplace=True)

# If still empty, relax to tiny lags (rare)
if wk_model.empty:
    ALLOWED_LAGS = [1, 2]
    for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
        for L in [1,2]:
            wk[f"{col}_lag{L}"] = grp[col].shift(L)
    sizes = grp.size()
    maxlag_map = {cat: max([L for L in [1,2] if L < int(n)], default=0) for cat, n in sizes.items()}
    wk["max_warmup"] = wk["product_category_name_english"].map(maxlag_map).fillna(0).astype(int)
    idx_in_group = grp.cumcount()
    wk_model = wk[idx_in_group >= wk["max_warmup"]].copy()
    wk_model.drop(columns=["max_warmup"], inplace=True)
    if wk_model.empty:
        raise RuntimeError("Not enough weekly history per category to create lagged features.")

# ---------------------------
# 4) Vectorised robust time split per category (never empty)
# ---------------------------
wk_model = wk_model.sort_values(["product_category_name_english","order_week"]).copy()
g = wk_model.groupby("product_category_name_english", sort=False)
n = g["order_week"].transform("size")
# last ~20% rows per cat (min 1 if n>=2; cap 12)
val_len = (n * 0.20).round().astype(int).clip(lower=1, upper=12)
val_len = np.where(n <= 1, 0, val_len)

last_idx = g.cumcount(ascending=False)  # 0 for last row
wk_model["split"] = np.where(last_idx < val_len, "valid", "train")

# Ensure at least one valid overall
if (wk_model["split"] == "valid").sum() == 0:
    wk_model.loc[g.tail(1).index, "split"] = "valid"

# ---------------------------
# 5) Save features
# ---------------------------
wk.to_csv(BASE_OUT / "weekly_by_category_full.csv", index=False)
wk_model.to_csv(BASE_OUT / "weekly_by_category.csv", index=False)

print("Features saved:",
      BASE_OUT / "weekly_by_category_full.csv",
      BASE_OUT / "weekly_by_category.csv")
print("Rows (full, model):", len(wk), len(wk_model))

# ---------------------------
# 6) Train model
# ---------------------------
TARGET   = "units"
cat_cols = ["product_category_name_english"]
drop_cols = {"order_week","split",TARGET}
num_cols = [c for c in wk_model.columns if c not in set(cat_cols) | drop_cols]

train_df = wk_model[wk_model["split"]=="train"].copy()
valid_df = wk_model[wk_model["split"]=="valid"].copy()

if train_df.empty or valid_df.empty:
    raise RuntimeError(f"After splitting: train={len(train_df)}, valid={len(valid_df)}. Not enough data to train/evaluate.")

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = XGBRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4
)
pipe = Pipeline([("pre", pre), ("xgb", model)])

X_tr, y_tr = train_df[cat_cols+num_cols], train_df[TARGET]
X_va, y_va = valid_df[cat_cols+num_cols], valid_df[TARGET]

pipe.fit(X_tr, y_tr)
pred_va = pipe.predict(X_va)

metrics = {
    "MAE":  float(mean_absolute_error(y_va, pred_va)),
    "MAPE": float(mean_absolute_percentage_error(np.maximum(y_va,1e-6), np.maximum(pred_va,1e-6))),
    "R2":   float(r2_score(y_va, pred_va)),
    "train_rows": int(len(train_df)),
    "valid_rows": int(len(valid_df))
}

pd.Series(metrics).to_csv(BASE_OUT / "validation_metrics.csv")
pd.DataFrame({
    "order_week": valid_df["order_week"],
    "category":   valid_df["product_category_name_english"],
    "actual_units": y_va,
    "pred_units":   pred_va
}).to_csv(BASE_OUT / "valid_predictions.csv", index=False)

joblib.dump(pipe, BASE_OUT / "weekly_units_xgb.pkl")

print("Training complete. Saved:",
      BASE_OUT / "validation_metrics.csv",
      BASE_OUT / "valid_predictions.csv",
      BASE_OUT / "weekly_units_xgb.pkl")
print("Validation metrics:", metrics)


D]--------> ## simulate_price_change.py ##

# Simulate price change
import pandas as pd, numpy as np
from pathlib import Path
import joblib

OUT = Path("Dataset/model_outputs")
wk  = pd.read_csv(OUT/"weekly_by_category.csv", parse_dates=["order_week"]) \
        .sort_values(["product_category_name_english","order_week"]) \
        .reset_index(drop=True)
pipe = joblib.load(OUT/"weekly_units_xgb.pkl")

# Build the exact feature list the model was trained with
TARGET = "units"
cat_cols = ["product_category_name_english"]
drop_cols = {"order_week","split",TARGET}
num_cols = [c for c in wk.columns if c not in set(cat_cols) | drop_cols]
FEAT_COLS = cat_cols + num_cols

def simulate(percent_change=-0.10, categories=None, margin_default=0.30, margin_csv=None):
    base = wk[wk["split"]=="valid"].copy()
    sim  = base.copy()
    mask = sim["product_category_name_english"].isin(categories) if categories else np.ones(len(sim), dtype=bool)

    # Adjust current price & lag-1 price (simple approximation)
    if "avg_price" in sim.columns:
        sim.loc[mask, "avg_price"] *= (1.0 + percent_change)
    if "avg_price_lag1" in sim.columns:
        sim.loc[mask, "avg_price_lag1"] *= (1.0 + percent_change)

    # Predict with EXACT same features as training
    base_pred = pipe.predict(base[FEAT_COLS])
    scn_pred  = pipe.predict(sim[FEAT_COLS])

    # Revenue approximation from price * predicted units
    base_rev = (base["avg_price"] * base_pred).sum()
    scn_rev  = (sim["avg_price"]  * scn_pred).sum()
    delta_units   = float((scn_pred - base_pred).sum())
    delta_revenue = float(scn_rev - base_rev)

    # Optional per-category margins
    margin_map = {}
    if margin_csv:
        mm = pd.read_csv(margin_csv)
        margin_map = dict(zip(mm["category"], mm["margin_rate"]))

    tmp = pd.DataFrame({
        "category": base["product_category_name_english"],
        "base_units": base_pred, "scn_units": scn_pred,
        "base_price": base["avg_price"], "scn_price": sim["avg_price"]
    })
    tmp["base_rev"] = tmp["base_price"] * tmp["base_units"]
    tmp["scn_rev"]  = tmp["scn_price"]  * tmp["scn_units"]
    tmp["mr"] = tmp["category"].map(margin_map).fillna(margin_default)
    delta_margin = float(((tmp["scn_rev"] - tmp["base_rev"]) * tmp["mr"]).sum())

    impact = (tmp.assign(delta_units=lambda d: d["scn_units"]-d["base_units"],
                         delta_revenue=lambda d: d["scn_rev"]-d["base_rev"])
                .groupby("category", as_index=False)
                .agg(delta_units=("delta_units","sum"),
                     delta_revenue=("delta_revenue","sum"))
             )
    summary = {
        "percent_price_change": percent_change,
        "delta_units": delta_units,
        "delta_revenue": delta_revenue,
        "delta_margin": delta_margin
    }
    return summary, impact

# Example: 10% markdown across all categories
summary, impact = simulate(percent_change=-0.10, margin_default=0.30)
pd.DataFrame([summary]).to_csv(OUT/"whatif_summary_10pct_markdown.csv", index=False)
impact.sort_values("delta_revenue", ascending=False).to_csv(
    OUT/"whatif_impact_by_category_10pct.csv", index=False
)
print("Saved:", OUT/"whatif_summary_10pct_markdown.csv", OUT/"whatif_impact_by_category_10pct.csv")
print("Summary:", summary)

E]--------> ## build_and_train.py ##

# build_and_train.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

BASE_IN  = Path("Dataset/clean_exports")
RAW_IN   = Path("Dataset")
BASE_OUT = Path("Dataset/model_outputs")
BASE_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 0) Load clean fact
# ---------------------------
fact = pd.read_csv(BASE_IN / "fact_orders_clean.csv")
# Robust date parsing (handles dd-mm-YYYY)
fact["order_date"] = pd.to_datetime(fact["order_date"], dayfirst=True, errors="coerce")

required_cols = {"order_id","order_date","units","revenue_item","product_category_name_english"}
missing_req = required_cols - set(fact.columns)
if missing_req:
    raise RuntimeError(f"Missing required columns in fact_orders_clean.csv: {missing_req}")

fact["order_id"] = fact["order_id"].astype(str)
fact["product_category_name_english"] = fact["product_category_name_english"].fillna("unknown")

# ---------------------------
# 1) Merge reviews robustly (or rebuild if needed)
# ---------------------------
def load_or_build_reviews():
    rev_path = BASE_IN / "reviews_by_order.csv"
    if rev_path.exists():
        rev = pd.read_csv(rev_path)
        rev.columns = [c.strip().lower() for c in rev.columns]
        # If someone saved raw reviews by mistake, aggregate
        if "review_count" not in rev.columns and "review_id" in rev.columns:
            raw = pd.read_csv(RAW_IN / "olist_order_reviews_dataset.csv",
                              parse_dates=["review_creation_date","review_answer_timestamp"])
            return (raw.groupby("order_id", as_index=False)
                        .agg(review_count=("review_id","count"),
                             review_score_avg=("review_score","mean")))
        if "review_score_avg" not in rev.columns and "review_score" in rev.columns:
            rev = rev.rename(columns={"review_score":"review_score_avg"})
        # Ensure columns
        for col in ["order_id","review_count","review_score_avg"]:
            if col not in rev.columns:
                raw = pd.read_csv(RAW_IN / "olist_order_reviews_dataset.csv",
                                  parse_dates=["review_creation_date","review_answer_timestamp"])
                return (raw.groupby("order_id", as_index=False)
                            .agg(review_count=("review_id","count"),
                                 review_score_avg=("review_score","mean")))
        return rev

    raw_reviews = RAW_IN / "olist_order_reviews_dataset.csv"
    if not raw_reviews.exists():
        return pd.DataFrame({"order_id": [], "review_count": [], "review_score_avg": []})
    raw = pd.read_csv(raw_reviews,
                      parse_dates=["review_creation_date","review_answer_timestamp"])
    return (raw.groupby("order_id", as_index=False)
              .agg(review_count=("review_id","count"),
                   review_score_avg=("review_score","mean")))

rev = load_or_build_reviews()
if "order_id" not in rev.columns:
    raise RuntimeError("Could not build a review table with 'order_id'.")
rev["order_id"] = rev["order_id"].astype(str)
for col in ["review_count","review_score_avg"]:
    if col not in rev.columns:
        rev[col] = np.nan

fact = fact.merge(rev[["order_id","review_count","review_score_avg"]], on="order_id", how="left")
if "review_count" not in fact.columns:      fact["review_count"] = 0
if "review_score_avg" not in fact.columns:  fact["review_score_avg"] = np.nan
fact["review_count"] = fact["review_count"].fillna(0)

# ---------------------------
# 2) Weekly aggregate (category x week)
# ---------------------------
fact["order_week"] = fact["order_date"] - pd.to_timedelta(fact["order_date"].dt.weekday, unit="D")

wk = (fact.groupby(["order_week","product_category_name_english"], as_index=False)
          .agg(units=("units","sum"),
               revenue=("revenue_item","sum"),
               avg_price=("revenue_item","mean"),
               rev_count=("review_count","sum"),
               rev_score_avg=("review_score_avg","mean")))

if wk.empty:
    raise RuntimeError("Weekly aggregate is empty. Check order_date parsing and that fact_orders_clean.csv has rows.")

wk = wk.sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)

# Fill review score gaps with rolling mean (index-safe)
wk["rev_score_avg"] = (
    wk.groupby("product_category_name_english", group_keys=False)["rev_score_avg"]
      .apply(lambda s: s.fillna(s.rolling(8, min_periods=1).mean()))
)

# Calendar/seasonal features
wk["weekofyear"] = wk["order_week"].dt.isocalendar().week.astype(int)
wk["year"]       = wk["order_week"].dt.year
wk["sin_woy"]    = np.sin(2*np.pi*wk["weekofyear"]/52.0)
wk["cos_woy"]    = np.cos(2*np.pi*wk["weekofyear"]/52.0)

# Optional holiday flag (safe fallback)
try:
    from holidays import Brazil
    years = range(int(wk["year"].min()), int(wk["year"].max()) + 1)
    br_dates = set(pd.to_datetime(list(Brazil(years=years))).date)
    wk["is_holiday_week"] = wk["order_week"].dt.date.isin(br_dates).astype(int)
except Exception:
    wk["is_holiday_week"] = 0

# ---------------------------
# 3) Add lags — vectorised (no groupby.apply, no warnings)
# ---------------------------
ALLOWED_LAGS = [1, 2, 4, 8, 12, 52]
grp = wk.groupby("product_category_name_english", sort=False)

for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
    for L in ALLOWED_LAGS:
        wk[f"{col}_lag{L}"] = grp[col].shift(L)

# Short rolling means (helpful, no leakage)
wk["units_roll4"]   = grp["units"].transform(lambda s: s.rolling(4, min_periods=1).mean())
wk["revenue_roll4"] = grp["revenue"].transform(lambda s: s.rolling(4, min_periods=1).mean())

# Adaptive warm-up trim per category: drop the first max usable lag rows
sizes = grp.size()  # Series: category -> count
def max_lag_for_n(n):
    usable = [L for L in ALLOWED_LAGS if L < n]
    return max(usable) if usable else 0
maxlag_map = {cat: max_lag_for_n(int(n)) for cat, n in sizes.items()}
wk["max_warmup"] = wk["product_category_name_english"].map(maxlag_map).fillna(0).astype(int)

idx_in_group = grp.cumcount()
wk_model = wk[idx_in_group >= wk["max_warmup"]].copy()
wk_model.drop(columns=["max_warmup"], inplace=True)

# If still empty, relax to tiny lags (rare)
if wk_model.empty:
    ALLOWED_LAGS = [1, 2]
    for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
        for L in [1,2]:
            wk[f"{col}_lag{L}"] = grp[col].shift(L)
    sizes = grp.size()
    maxlag_map = {cat: max([L for L in [1,2] if L < int(n)], default=0) for cat, n in sizes.items()}
    wk["max_warmup"] = wk["product_category_name_english"].map(maxlag_map).fillna(0).astype(int)
    idx_in_group = grp.cumcount()
    wk_model = wk[idx_in_group >= wk["max_warmup"]].copy()
    wk_model.drop(columns=["max_warmup"], inplace=True)
    if wk_model.empty:
        raise RuntimeError("Not enough weekly history per category to create lagged features.")

# ---------------------------
# 4) Vectorised robust time split per category (never empty)
# ---------------------------
wk_model = wk_model.sort_values(["product_category_name_english","order_week"]).copy()
g = wk_model.groupby("product_category_name_english", sort=False)
n = g["order_week"].transform("size")
# last ~20% rows per cat (min 1 if n>=2; cap 12)
val_len = (n * 0.20).round().astype(int).clip(lower=1, upper=12)
val_len = np.where(n <= 1, 0, val_len)

last_idx = g.cumcount(ascending=False)  # 0 for last row
wk_model["split"] = np.where(last_idx < val_len, "valid", "train")

# Ensure at least one valid overall
if (wk_model["split"] == "valid").sum() == 0:
    wk_model.loc[g.tail(1).index, "split"] = "valid"

# ---------------------------
# 5) Save features
# ---------------------------
wk.to_csv(BASE_OUT / "weekly_by_category_full.csv", index=False)
wk_model.to_csv(BASE_OUT / "weekly_by_category.csv", index=False)

print("Features saved:",
      BASE_OUT / "weekly_by_category_full.csv",
      BASE_OUT / "weekly_by_category.csv")
print("Rows (full, model):", len(wk), len(wk_model))

# ---------------------------
# 6) Train model
# ---------------------------
TARGET   = "units"
cat_cols = ["product_category_name_english"]
drop_cols = {"order_week","split",TARGET}
num_cols = [c for c in wk_model.columns if c not in set(cat_cols) | drop_cols]

train_df = wk_model[wk_model["split"]=="train"].copy()
valid_df = wk_model[wk_model["split"]=="valid"].copy()

if train_df.empty or valid_df.empty:
    raise RuntimeError(f"After splitting: train={len(train_df)}, valid={len(valid_df)}. Not enough data to train/evaluate.")

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = XGBRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4
)
pipe = Pipeline([("pre", pre), ("xgb", model)])

X_tr, y_tr = train_df[cat_cols+num_cols], train_df[TARGET]
X_va, y_va = valid_df[cat_cols+num_cols], valid_df[TARGET]

pipe.fit(X_tr, y_tr)
pred_va = pipe.predict(X_va)

metrics = {
    "MAE":  float(mean_absolute_error(y_va, pred_va)),
    "MAPE": float(mean_absolute_percentage_error(np.maximum(y_va,1e-6), np.maximum(pred_va,1e-6))),
    "R2":   float(r2_score(y_va, pred_va)),
    "train_rows": int(len(train_df)),
    "valid_rows": int(len(valid_df))
}

pd.Series(metrics).to_csv(BASE_OUT / "validation_metrics.csv")
pd.DataFrame({
    "order_week": valid_df["order_week"],
    "category":   valid_df["product_category_name_english"],
    "actual_units": y_va,
    "pred_units":   pred_va
}).to_csv(BASE_OUT / "valid_predictions.csv", index=False)

joblib.dump(pipe, BASE_OUT / "weekly_units_xgb.pkl")

print("Training complete. Saved:",
      BASE_OUT / "validation_metrics.csv",
      BASE_OUT / "valid_predictions.csv",
      BASE_OUT / "weekly_units_xgb.pkl")
print("Validation metrics:", metrics)


F]--------> ## make_margins_template.py ##

# Make margins template
import pandas as pd
from pathlib import Path

OUT = Path("Dataset/model_outputs")
wk  = pd.read_csv(OUT/"weekly_by_category.csv")
cats = sorted(wk["product_category_name_english"].unique())
pd.DataFrame({"category": cats, "margin_rate": 0.30}).to_csv(OUT/"margins.csv", index=False)
print("Edit margin rates here:", OUT/"margins.csv")

G]--------> ## recommend_actions.py ##

# Recommend Actions
import pandas as pd
from pathlib import Path
import joblib

OUT = Path("Dataset/model_outputs")
wk  = pd.read_csv(OUT/"weekly_by_category.csv", parse_dates=["order_week"]) \
        .sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)
pipe = joblib.load(OUT/"weekly_units_xgb.pkl")

# Same features the model saw
TARGET="units"
cat_cols=["product_category_name_english"]
drop_cols={"order_week","split",TARGET}
num_cols=[c for c in wk.columns if c not in set(cat_cols)|drop_cols]
FEAT=cat_cols+num_cols

# margin map
marg = OUT/"margins.csv"
margin_map = {}
if marg.exists():
    m = pd.read_csv(marg)
    margin_map = dict(zip(m["category"], m["margin_rate"]))
margin_default = 0.30

def score(pct):
    base = wk[wk["split"]=="valid"].copy()
    sim  = base.copy()
    if "avg_price" in sim:      sim["avg_price"] *= (1+pct)
    if "avg_price_lag1" in sim: sim["avg_price_lag1"] *= (1+pct)
    base_pred = pipe.predict(base[FEAT])
    scn_pred  = pipe.predict(sim[FEAT])
    tmp = pd.DataFrame({
        "category": base["product_category_name_english"],
        "base_units": base_pred,
        "scn_units": scn_pred,
        "base_price": base["avg_price"],
        "scn_price":  sim["avg_price"]
    })
    tmp["base_rev"]=tmp["base_price"]*tmp["base_units"]
    tmp["scn_rev"] =tmp["scn_price"] *tmp["scn_units"]
    tmp["mr"]=tmp["category"].map(margin_map).fillna(margin_default)
    tmp["delta_margin"]=(tmp["scn_rev"]-tmp["base_rev"])*tmp["mr"]
    return (tmp.groupby("category", as_index=False)
              .agg(delta_margin=("delta_margin","sum")))

grid = [-0.20,-0.15,-0.10,-0.05,0.0,0.05,0.10]
recs=[]
for p in grid:
    s=score(p); s["percent_price_change"]=p; recs.append(s)
recs=pd.concat(recs, ignore_index=True)
best=(recs.sort_values(["category","delta_margin"], ascending=[True,False])
          .groupby("category", as_index=False).head(1)
          .rename(columns={"percent_price_change":"recommended_pct",
                           "delta_margin":"best_delta_margin"}))

best.to_csv(OUT/"recommended_actions.csv", index=False)
print("Saved:", OUT/"recommended_actions.csv")

H]--------> ## # select_actions_with_budget.py ##

# Select actions with budget
import pandas as pd
from pathlib import Path

OUT = Path("Dataset/model_outputs")
recs = pd.read_csv(OUT/"recommended_actions.csv")
wk   = pd.read_csv(OUT/"weekly_by_category.csv", parse_dates=["order_week"])

# base revenue per category (validation window)
base = wk[wk["split"]=="valid"].copy()
base_rev = (base.assign(pred_units=0)  # not used, just to keep shape clear
                 .groupby("product_category_name_english", as_index=False)
                 .agg(base_revenue=("revenue","sum")))

df = recs.merge(base_rev, left_on="category",
                right_on="product_category_name_english", how="left")

df["cost"] = (df["recommended_pct"].abs() * df["base_revenue"]).fillna(0.0)

# ----- set your budget here -----
TOTAL_BUDGET = df["cost"].sum() * 0.3  # e.g., allow 30% of total potential spend
MAX_ACTIONS  = 10                       # or limit by count
# --------------------------------

df = df.sort_values("best_delta_margin", ascending=False).reset_index(drop=True)

picked=[]; spend=0.0
for _, r in df.iterrows():
    if len(picked)>=MAX_ACTIONS: break
    if spend + r["cost"] <= TOTAL_BUDGET:
        picked.append(r)
        spend += r["cost"]

sel = pd.DataFrame(picked)
sel = sel[["category","recommended_pct","best_delta_margin","cost"]]
sel.to_csv(OUT/"selected_actions.csv", index=False)
print("Saved:", OUT/"selected_actions.csv")
print(f"Selected {len(sel)} actions, spend ~ {spend:,.2f} within budget {TOTAL_BUDGET:,.2f}")

I]--------> ## # make_price_plan_and_forecast.py ##

# Make price plan and forecast

import pandas as pd, numpy as np
from pathlib import Path
import joblib

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# --- Load history and model ---
wk_full  = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"]) \
              .sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)
wk_model = pd.read_csv(OUT/"weekly_by_category.csv", parse_dates=["order_week"])
pipe = joblib.load(OUT/"weekly_units_xgb.pkl")

# === 1) BUILD PRICE PLAN (robust) ===========================================
recent = (wk_full.groupby("product_category_name_english")
          .agg(base_price=("avg_price", lambda s: s.tail(4).mean()))
          .reset_index())

sel_path = OUT/"selected_actions.csv"
rec_path = OUT/"recommended_actions.csv"
if sel_path.exists():
    actions = pd.read_csv(sel_path)
elif rec_path.exists():
    actions = pd.read_csv(rec_path)
else:
    actions = pd.DataFrame(columns=["category","recommended_pct"])

if "category" not in actions.columns:
    if "product_category_name_english" in actions.columns:
        actions = actions.rename(columns={"product_category_name_english":"category"})
    else:
        actions["category"] = []

pct_col = None
for c in ["recommended_pct","recommended_price_change","percent_price_change"]:
    if c in actions.columns:
        pct_col = c; break
if pct_col is None:
    actions["recommended_pct"] = 0.0
    pct_col = "recommended_pct"

def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

recent["cat_key"]  = norm(recent["product_category_name_english"])
actions["cat_key"] = norm(actions["category"])

act = actions[["cat_key", pct_col]].copy().rename(columns={pct_col:"applied_pct"})
if (act["applied_pct"].abs() > 1.0).any():
    act["applied_pct"] = act["applied_pct"] / 100.0

# horizon & future grid
H = 8
last_wk = wk_full["order_week"].max()
# FIX: use to_timedelta (or datetime.timedelta). This is the corrected line:
future_weeks = [last_wk + pd.to_timedelta(7*(i+1), unit="D") for i in range(H)]

cats = recent["product_category_name_english"].tolist()
plan = pd.MultiIndex.from_product([future_weeks, cats],
          names=["order_week","product_category_name_english"]).to_frame(index=False)
plan["cat_key"] = norm(plan["product_category_name_english"])

plan = (plan.merge(recent[["product_category_name_english","cat_key","base_price"]],
                   on=["product_category_name_english","cat_key"], how="left")
             .merge(act, on="cat_key", how="left"))

plan["applied_pct"]   = plan["applied_pct"].fillna(0.0)
plan["planned_price"] = plan["base_price"] * (1.0 + plan["applied_pct"])

diag = (plan.groupby("product_category_name_english", as_index=False)
            .agg(base_price=("base_price","first"),
                 applied_pct=("applied_pct","first"),
                 planned_price=("planned_price","first")))
diag["changed"] = np.where(diag["applied_pct"].abs() > 1e-9, "yes", "no")

plan_out = OUT/"price_plan.csv"
diag_out = OUT/"price_plan_debug.csv"
plan.to_csv(plan_out, index=False)
diag.to_csv(diag_out, index=False)
print("Saved price plan:", plan_out)
print("Saved diagnostics:", diag_out)
print(f"Categories with non-zero action: {(diag['changed']=='yes').sum()}/{len(diag)}")

# === 2) FORECAST USING THE PLAN (feature-safe) ===============================
TARGET    = "units"
cat_cols  = ["product_category_name_english"]
drop_cols = {"order_week","split",TARGET}
num_cols  = [c for c in wk_model.columns if c not in set(cat_cols) | drop_cols]
FEAT_COLS = cat_cols + num_cols

history = wk_full.copy()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_category_name_english","order_week"]).copy()
    g  = df.groupby("product_category_name_english", sort=False)

    df["weekofyear"] = df["order_week"].dt.isocalendar().week.astype(int)
    df["year"]       = df["order_week"].dt.year
    df["sin_woy"]    = np.sin(2*np.pi*df["weekofyear"]/52.0)
    df["cos_woy"]    = np.cos(2*np.pi*df["weekofyear"]/52.0)
    if "is_holiday_week" not in df.columns:
        df["is_holiday_week"] = 0

    for col in ["rev_count","rev_score_avg"]:
        if col not in df: df[col] = np.nan
        df[col] = df[col].fillna(g[col].transform(lambda s: s.tail(4).mean()))

    LAGS = [1,2,4,8,12,52]
    for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
        for L in LAGS:
            df[f"{col}_lag{L}"] = g[col].shift(L)

    df["units_roll4"]   = g["units"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["revenue_roll4"] = g["revenue"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    return df

forecasts = []
for wk_i in future_weeks:
    step = plan[plan["order_week"]==wk_i][
        ["order_week","product_category_name_english","planned_price"]
    ].rename(columns={"planned_price":"avg_price"}).copy()

    step["units"] = np.nan
    step["revenue"] = np.nan
    step["rev_count"] = np.nan
    step["rev_score_avg"] = np.nan

    tmp = pd.concat([history, step], ignore_index=True)
    tmp = compute_features(tmp)

    # proxy revenue for current step (model expects a 'revenue' feature)
    mask = tmp["order_week"].eq(wk_i)
    units_proxy = tmp.loc[mask, "units_lag1"].fillna(tmp.loc[mask, "units_roll4"]).fillna(0.0)
    tmp.loc[mask, "revenue"] = tmp.loc[mask, "avg_price"] * units_proxy

    for c in FEAT_COLS:
        if c not in tmp.columns:
            tmp[c] = 0.0

    X = tmp.loc[mask, FEAT_COLS]
    yhat = pipe.predict(X)

    pred = tmp.loc[mask, ["order_week","product_category_name_english","avg_price"]].copy()
    pred["units"]   = yhat
    pred["revenue"] = pred["avg_price"] * pred["units"]

    history = pd.concat([history, pred], ignore_index=True, sort=False)
    forecasts.append(pred)

fcast = pd.concat(forecasts, ignore_index=True)
fcast_out = OUT/"forecast_with_plan.csv"
fcast.to_csv(fcast_out, index=False)
print("Saved forecast:", fcast_out)

J]--------> ## # forecast_next_weeks.py ##

# Forecast next week
# forecast_next_weeks.py
import pandas as pd, numpy as np
from pathlib import Path
import joblib

OUT = Path("Dataset/model_outputs")
wk_full  = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"]) \
              .sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)
wk_model = pd.read_csv(OUT/"weekly_by_category.csv", parse_dates=["order_week"])

pipe = joblib.load(OUT/"weekly_units_xgb.pkl")

# --- exact features used at training ---
TARGET = "units"
cat_cols = ["product_category_name_english"]
drop_cols = {"order_week","split",TARGET}
num_cols = [c for c in wk_model.columns if c not in set(cat_cols) | drop_cols]
FEAT_COLS = cat_cols + num_cols

# Optional plan CSV (order_week, product_category_name_english, planned_price)
PRICE_PLAN = None  # e.g., Path("Dataset/model_outputs/price_plan.csv")

H = 8
cats = wk_full["product_category_name_english"].unique()
def next_monday(d): return d + pd.to_timedelta(7, unit="D")

last_week = wk_full["order_week"].max()
future_weeks = [next_monday(last_week) + pd.to_timedelta(7*i, unit="D") for i in range(H)]
future = pd.MultiIndex.from_product([future_weeks, cats],
                                    names=["order_week","product_category_name_english"]).to_frame(index=False)

# recent averages to carry forward
recent = (wk_full.groupby("product_category_name_english")
          .agg(avg_price_recent=("avg_price",       lambda s: s.tail(4).mean()),
               rev_count_recent=("rev_count",       lambda s: s.tail(4).mean()),
               rev_score_recent=("rev_score_avg",   lambda s: s.tail(4).mean()))
          .reset_index())

future = future.merge(recent, on="product_category_name_english", how="left")

# ensure the avg_price column exists even if no plan is provided
future["avg_price"] = np.nan

# merge plan if provided
if PRICE_PLAN:
    plan = pd.read_csv(PRICE_PLAN, parse_dates=["order_week"])
    future = future.merge(
        plan.rename(columns={"planned_price":"avg_price"}),
        on=["order_week","product_category_name_english"],
        how="left",
        suffixes=("","_plan")
    )
    # if both columns exist due to suffix, prefer plan
    if "avg_price_plan" in future.columns:
        future["avg_price"] = future["avg_price_plan"].combine_first(future["avg_price"])
        future.drop(columns=["avg_price_plan"], inplace=True)

history = wk_full.copy()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_category_name_english","order_week"]).copy()
    g  = df.groupby("product_category_name_english", sort=False)

    # price for future rows: plan if given, else recent avg
    if "avg_price" not in df.columns:
        df["avg_price"] = np.nan
    if "avg_price_recent" not in df.columns:
        # compute fallback from existing history if needed
        tmp_recent = (df.groupby("product_category_name_english")
                        .agg(avg_price_recent=("avg_price", lambda s: s.tail(4).mean()))
                        .reset_index())
        df = df.merge(tmp_recent, on="product_category_name_english", how="left")
    df["avg_price"] = df["avg_price"].fillna(df["avg_price_recent"])

    # reviews carry-forward
    for col, recent_col in [("rev_count","rev_count_recent"),
                            ("rev_score_avg","rev_score_recent")]:
        if col not in df.columns:
            df[col] = np.nan
        if recent_col not in df.columns:
            tmp_recent = (df.groupby("product_category_name_english")
                            .agg(**{recent_col:(col, lambda s: s.tail(4).mean())})
                            .reset_index())
            df = df.merge(tmp_recent, on="product_category_name_english", how="left")
        df[col] = df[col].fillna(df[recent_col])

    # calendar
    df["weekofyear"] = df["order_week"].dt.isocalendar().week.astype(int)
    df["year"]       = df["order_week"].dt.year
    df["sin_woy"]    = np.sin(2*np.pi*df["weekofyear"]/52.0)
    df["cos_woy"]    = np.cos(2*np.pi*df["weekofyear"]/52.0)
    if "is_holiday_week" not in df.columns:
        df["is_holiday_week"] = 0

    # vectorised lags matching training
    LAGS = [1,2,4,8,12,52]
    for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
        for L in LAGS:
            df[f"{col}_lag{L}"] = g[col].shift(L)

    # rolling stats used in training
    df["units_roll4"]   = g["units"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["revenue_roll4"] = g["revenue"].transform(lambda s: s.rolling(4, min_periods=1).mean())

    return df

forecasts = []
for wk_i in future_weeks:
    # build rows for this step (now avg_price column is guaranteed to exist)
    step = future[future["order_week"]==wk_i][[
        "order_week","product_category_name_english",
        "avg_price","avg_price_recent","rev_count_recent","rev_score_recent"
    ]].copy()

    # placeholders to match history columns
    step["units"] = np.nan
    step["revenue"] = np.nan
    step["rev_count"] = np.nan
    step["rev_score_avg"] = np.nan

    # compute features (incl. lags/rolling)
    tmp = pd.concat([history, step], ignore_index=True)
    tmp = compute_features(tmp)

    # supply proxy revenue for CURRENT future week to satisfy pipeline
    mask = tmp["order_week"].eq(wk_i)
    units_proxy = tmp.loc[mask, "units_lag1"].fillna(tmp.loc[mask, "units_roll4"]).fillna(0.0)
    tmp.loc[mask, "revenue"] = tmp.loc[mask, "avg_price"] * units_proxy

    # ensure every expected feature exists
    for c in FEAT_COLS:
        if c not in tmp.columns:
            tmp[c] = 0.0
    if "product_category_name_english" not in tmp.columns:
        tmp["product_category_name_english"] = history["product_category_name_english"]

    X = tmp.loc[mask, FEAT_COLS]
    yhat = pipe.predict(X)

    pred = tmp.loc[mask, ["order_week","product_category_name_english","avg_price"]].copy()
    pred["units"]   = yhat
    pred["revenue"] = pred["avg_price"] * pred["units"]

    # carry forward recents for next iterations
    pred = pred.merge(
        step[["product_category_name_english","avg_price_recent","rev_count_recent","rev_score_recent"]],
        on="product_category_name_english", how="left"
    )

    # append to history
    history = pd.concat([history, pred], ignore_index=True, sort=False)
    forecasts.append(pred[["order_week","product_category_name_english","avg_price","units","revenue"]])

fcast = pd.concat(forecasts, ignore_index=True)
out_path = OUT/"forecasts_next_weeks.csv"
fcast.to_csv(out_path, index=False)
print("Saved:", out_path)

K]--------> ## # # compute_plan_impact.py ##

# compute_plan_impact.py
import pandas as pd, numpy as np
from pathlib import Path
import joblib

OUT = Path("Dataset/model_outputs")
wk   = pd.read_csv(OUT/"weekly_by_category.csv", parse_dates=["order_week"])
fcast= pd.read_csv(OUT/"forecast_with_plan.csv", parse_dates=["order_week"])
pipe = joblib.load(OUT/"weekly_units_xgb.pkl")

# features the model expects
TARGET="units"; cat_cols=["product_category_name_english"]
drop_cols={"order_week","split",TARGET}
feat_cols = cat_cols + [c for c in wk.columns if c not in set(cat_cols)|drop_cols]

# validation slice as baseline behavior
valid = wk[wk["split"]=="valid"].copy().sort_values(["product_category_name_english","order_week"])
base_pred = pipe.predict(valid[feat_cols])
valid = valid.assign(base_units_pred=base_pred, base_rev_pred=valid["avg_price"]*base_pred)

# plan horizon (already predicted in forecast_with_plan.csv)
plan = fcast.copy().rename(columns={"units":"plan_units_pred","revenue":"plan_rev_pred"})

# join recent prices (for safety) and margins
marg = OUT/"margins.csv"
margin_map = {}
if marg.exists():
    mm = pd.read_csv(marg)
    margin_map = dict(zip(mm["category"], mm["margin_rate"]))

# aggregate impacts over the horizon by category
impact = (plan.groupby("product_category_name_english", as_index=False)
              .agg(plan_units=("plan_units_pred","sum"),
                   plan_revenue=("plan_rev_pred","sum")))

# Baseline for the same number of weeks = average valid week × H
H = plan["order_week"].nunique()
base_cat = (valid.groupby("product_category_name_english", as_index=False)
                 .agg(base_units_week=("base_units_pred","mean"),
                      base_rev_week=("base_rev_pred","mean")))
base_cat["base_units"]   = base_cat["base_units_week"]*H
base_cat["base_revenue"] = base_cat["base_rev_week"]*H
impact = impact.merge(base_cat[["product_category_name_english","base_units","base_revenue"]],
                      on="product_category_name_english", how="left")

impact["delta_units"]   = impact["plan_units"]   - impact["base_units"]
impact["delta_revenue"] = impact["plan_revenue"] - impact["base_revenue"]
impact["margin_rate"]   = impact["product_category_name_english"].map(margin_map).fillna(0.30)
impact["delta_margin"]  = impact["delta_revenue"]*impact["margin_rate"]

# totals
totals = (impact[["delta_units","delta_revenue","delta_margin"]].sum()
          .to_frame(name="total").T)
impact_path = OUT/"plan_impact_by_category.csv"
totals_path = OUT/"plan_impact_totals.csv"
impact.sort_values("delta_margin", ascending=False).to_csv(impact_path, index=False)
totals.to_csv(totals_path, index=False)
print("Saved:", impact_path, totals_path)
print(totals)

L]--------> ## # # plot_forecast.py ##

# Plot Forecast

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

OUT = Path("Dataset/model_outputs")

hist  = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"])
fcast = pd.read_csv(OUT/"forecast_with_plan.csv",     parse_dates=["order_week"])

# top 6 categories by revenue
cats = (hist.groupby("product_category_name_english")["revenue"]
            .sum().sort_values(ascending=False).head(6).index.tolist())

def clean_xy(df):
    # ensure datetime64[ns] and numeric y, drop NaT/NaN
    x = pd.to_datetime(df["order_week"], errors="coerce")
    y = pd.to_numeric(df["units"], errors="coerce")
    m = x.notna() & y.notna()
    # IMPORTANT: return numpy datetime64[ns] array (not object) to keep date converter happy
    x = x[m].to_numpy(dtype="datetime64[ns]")
    y = y[m].to_numpy(dtype=float)
    return x, y

for c in cats:
    h = hist[hist["product_category_name_english"]==c].sort_values("order_week")
    f = fcast[fcast["product_category_name_english"]==c].sort_values("order_week")

    xh, yh = clean_xy(h)
    xf, yf = clean_xy(f)

    if xh.size == 0 and xf.size == 0:
        continue  # nothing to plot for this category

    fig, ax = plt.subplots(figsize=(8, 4))

    if xh.size:
        ax.plot(xh, yh, label="history")
    if xf.size:
        ax.plot(xf, yf, label="forecast")

    # Nice date formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    ax.set_title(f"Units — {c}")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"forecast_units_{c.replace('/','_')}.png")
    plt.close(fig)

print("Saved charts to:", OUT)

M]--------> ## # # forecast_trimmed.py ##

import pandas as pd, numpy as np
from pathlib import Path
import joblib

OUT = Path("Dataset/model_outputs")
wk_full  = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"]) \
              .sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)
wk_model = pd.read_csv(OUT/"weekly_by_category.csv", parse_dates=["order_week"])
pipe = joblib.load(OUT/"weekly_units_xgb.pkl")

# exact features used at training
TARGET="units"; cat_cols=["product_category_name_english"]
drop_cols={"order_week","split",TARGET}
num_cols=[c for c in wk_model.columns if c not in set(cat_cols)|drop_cols]
FEAT_COLS = cat_cols + num_cols

# -------- trim suspicious trailing weeks (very low vs recent median) ----------
def trim_tail(df, lookback=8, frac=0.25, max_drop=2):
    # remove up to `max_drop` last rows if they're < frac * median(last lookback)
    g = []
    for cat, grp in df.groupby("product_category_name_english", sort=False):
        grp = grp.sort_values("order_week").copy()
        drops = 0
        while len(grp) >= lookback+1 and drops < max_drop:
            med = grp["units"].tail(lookback+1).head(lookback).median()
            last_val = grp["units"].iloc[-1]
            if pd.notna(last_val) and last_val < frac * med:
                grp = grp.iloc[:-1].copy()
                drops += 1
            else:
                break
        g.append(grp)
    return pd.concat(g, ignore_index=True)

hist = trim_tail(wk_full, lookback=8, frac=0.25, max_drop=2)

# build plan: keep prices flat (= last 4w mean) unless you have price_plan.csv
recent = (hist.groupby("product_category_name_english")
            .agg(planned_price=("avg_price", lambda s: s.tail(4).mean()))
            .reset_index())

H=8
last_wk = hist["order_week"].max()
future_weeks = [last_wk + pd.to_timedelta(7*(i+1), unit="D") for i in range(H)]
plan = pd.MultiIndex.from_product(
    [future_weeks, recent["product_category_name_english"]],
    names=["order_week","product_category_name_english"]
).to_frame(index=False).merge(recent, on="product_category_name_english", how="left")

def compute_features(df):
    df = df.sort_values(["product_category_name_english","order_week"]).copy()
    g  = df.groupby("product_category_name_english", sort=False)
    df["weekofyear"]=df["order_week"].dt.isocalendar().week.astype(int)
    df["year"]=df["order_week"].dt.year
    df["sin_woy"]=np.sin(2*np.pi*df["weekofyear"]/52.0)
    df["cos_woy"]=np.cos(2*np.pi*df["weekofyear"]/52.0)
    if "is_holiday_week" not in df: df["is_holiday_week"]=0
    for col in ["rev_count","rev_score_avg"]:
        if col not in df: df[col]=np.nan
        df[col]=df[col].fillna(g[col].transform(lambda s: s.tail(4).mean()))
    if "avg_price" not in df: df["avg_price"]=np.nan
    LAGS=[1,2,4,8,12,52]
    for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
        for L in LAGS:
            df[f"{col}_lag{L}"]=g[col].shift(L)
    df["units_roll4"]=g["units"].transform(lambda s: s.rolling(4,min_periods=1).mean())
    df["revenue_roll4"]=g["revenue"].transform(lambda s: s.rolling(4,min_periods=1).mean())
    return df

forecasts=[]
history = hist.copy()
for wk_i in future_weeks:
    step = plan[plan["order_week"]==wk_i] \
            .rename(columns={"planned_price":"avg_price"}) \
            [["order_week","product_category_name_english","avg_price"]].copy()
    step["units"]=np.nan; step["revenue"]=np.nan
    step["rev_count"]=np.nan; step["rev_score_avg"]=np.nan
    tmp = pd.concat([history, step], ignore_index=True)
    tmp = compute_features(tmp)
    # proxy revenue since model expects 'revenue'
    mask = tmp["order_week"].eq(wk_i)
    units_proxy = tmp.loc[mask,"units_lag1"].fillna(tmp.loc[mask,"units_roll4"]).fillna(0.0)
    tmp.loc[mask,"revenue"] = tmp.loc[mask,"avg_price"]*units_proxy
    for c in FEAT_COLS:
        if c not in tmp.columns: tmp[c]=0.0
    X = tmp.loc[mask, FEAT_COLS]
    yhat = pipe.predict(X)
    pred = tmp.loc[mask, ["order_week","product_category_name_english","avg_price"]].copy()
    pred["units"]=yhat
    pred["revenue"]=pred["avg_price"]*pred["units"]
    history = pd.concat([history, pred], ignore_index=True, sort=False)
    forecasts.append(pred)

fcast = pd.concat(forecasts, ignore_index=True)
fcast.to_csv(OUT/"forecast_trimmed.csv", index=False)
print("Saved:", OUT/"forecast_trimmed.csv")

N]--------> ## # # build_and_train_no_leak.py ##

import pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

BASE_OUT = Path("Dataset/model_outputs")
wk = pd.read_csv(BASE_OUT/"weekly_by_category_full.csv", parse_dates=["order_week"])
wk = wk.sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)

# add a monotonic week index per category (captures trend)
wk["week_idx"] = wk.groupby("product_category_name_english").cumcount()

# build lags exactly like before (already in your full file)
grp = wk.groupby("product_category_name_english", sort=False)
for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
    for L in [1,2,4,8,12,52]:
        wk[f"{col}_lag{L}"] = grp[col].shift(L)
wk["units_roll4"]   = grp["units"].transform(lambda s: s.rolling(4, min_periods=1).mean())
wk["revenue_roll4"] = grp["revenue"].transform(lambda s: s.rolling(4, min_periods=1).mean())

# warm-up trim
sizes = grp.size()
maxlag_map = {cat: max([L for L in [1,2,4,8,12,52] if L < int(n)], default=0)
              for cat, n in sizes.items()}
wk["max_warmup"] = wk["product_category_name_english"].map(maxlag_map).fillna(0).astype(int)
wk_model = wk[grp.cumcount() >= wk["max_warmup"]].copy()
wk_model.drop(columns=["max_warmup"], inplace=True)

# vectorised time split
wk_model = wk_model.sort_values(["product_category_name_english","order_week"]).copy()
g = wk_model.groupby("product_category_name_english", sort=False)
n = g["order_week"].transform("size")
val_len = (n*0.2).round().astype(int).clip(lower=1, upper=12)
val_len = np.where(n<=1, 0, val_len)
last_idx = g.cumcount(ascending=False)
wk_model["split"] = np.where(last_idx < val_len, "valid", "train")

# features: drop current revenue (leakage) but keep lags/rollings + week_idx
TARGET="units"
cat_cols=["product_category_name_english"]
drop_cols={"order_week","split",TARGET,"revenue"}  # ⬅️ drop current revenue
num_cols=[c for c in wk_model.columns if c not in set(cat_cols)|drop_cols]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=4
)
pipe = Pipeline([("pre", pre), ("xgb", model)])

Xtr = wk_model[wk_model["split"]=="train"][cat_cols+num_cols]
ytr = wk_model[wk_model["split"]=="train"][TARGET]
Xva = wk_model[wk_model["split"]=="valid"][cat_cols+num_cols]
yva = wk_model[wk_model["split"]=="valid"][TARGET]

pipe.fit(Xtr, ytr)
pred = pipe.predict(Xva)

metrics = {
    "MAE": float(mean_absolute_error(yva, pred)),
    "MAPE": float(mean_absolute_percentage_error(np.maximum(yva,1e-6), np.maximum(pred,1e-6))),
    "R2": float(((np.corrcoef(yva, pred)[0,1])**2))  # alt R² estimate
}
pd.Series(metrics).to_csv(BASE_OUT/"validation_metrics_no_leak.csv")
joblib.dump(pipe, BASE_OUT/"weekly_units_xgb_no_leak.pkl")
wk_model.to_csv(BASE_OUT/"weekly_by_category_no_leak.csv", index=False)
print("Saved:", BASE_OUT/"weekly_units_xgb_no_leak.pkl")
print("Metrics:", metrics)

O]--------> ## # # forecast_no_leak.py ##

import pandas as pd, numpy as np
from pathlib import Path
import joblib

OUT = Path("Dataset/model_outputs")
wk_full  = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"]) \
              .sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)
wk_model = pd.read_csv(OUT/"weekly_by_category_no_leak.csv", parse_dates=["order_week"])
pipe = joblib.load(OUT/"weekly_units_xgb_no_leak.pkl")

# feature list used by the new model
TARGET="units"; cat_cols=["product_category_name_english"]
drop_cols={"order_week","split",TARGET,"revenue"}  # current revenue not used
num_cols=[c for c in wk_model.columns if c not in set(cat_cols)|drop_cols]
FEAT = cat_cols + num_cols

# add week_idx into full history (if missing)
wk_full["week_idx"] = wk_full.groupby("product_category_name_english").cumcount()

# trim trailing partial weeks
def trim_tail(df, lookback=8, frac=0.25, max_drop=2):
    g=[];
    for cat, grp in df.groupby("product_category_name_english", sort=False):
        grp=grp.sort_values("order_week").copy(); drops=0
        while len(grp)>=lookback+1 and drops<max_drop:
            med=grp["units"].tail(lookback+1).head(lookback).median()
            last=grp["units"].iloc[-1]
            if pd.notna(last) and last < frac*med:
                grp=grp.iloc[:-1].copy(); drops+=1
            else: break
        g.append(grp)
    return pd.concat(g, ignore_index=True)

history = trim_tail(wk_full)

# horizon and flat price plan = last 4w mean
recent = (history.groupby("product_category_name_english")
          .agg(avg_price=("avg_price", lambda s: s.tail(4).mean()))
          .reset_index())
H=8
last_wk=history["order_week"].max()
future_weeks=[last_wk + pd.to_timedelta(7*(i+1), unit="D") for i in range(H)]
plan = pd.MultiIndex.from_product([future_weeks, recent["product_category_name_english"]],
        names=["order_week","product_category_name_english"]).to_frame(index=False) \
        .merge(recent, on="product_category_name_english", how="left")

# roll forward
forecasts=[]
for i, wk_i in enumerate(future_weeks, start=1):
    step = plan[plan["order_week"]==wk_i][["order_week","product_category_name_english","avg_price"]].copy()
    step["units"]=np.nan; step["revenue"]=np.nan
    # extend week_idx
    step = step.merge(
        history.groupby("product_category_name_english")["week_idx"].max().reset_index()
               .rename(columns={"week_idx":"_last_idx"}),
        on="product_category_name_english", how="left"
    )
    step["week_idx"] = step["_last_idx"] + 1
    step.drop(columns=["_last_idx"], inplace=True)

    tmp = pd.concat([history, step], ignore_index=True)
    # recompute lags/rollings to match training
    g = tmp.groupby("product_category_name_english", sort=False)
    for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
        for L in [1,2,4,8,12,52]:
            tmp[f"{col}_lag{L}"] = g[col].shift(L)
    tmp["units_roll4"]   = g["units"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    tmp["revenue_roll4"] = g["revenue"].transform(lambda s: s.rolling(4, min_periods=1).mean())

    X = tmp[tmp["order_week"]==wk_i][FEAT]
    yhat = pipe.predict(X)

    pred = tmp[tmp["order_week"]==wk_i][["order_week","product_category_name_english","avg_price","week_idx"]].copy()
    pred["units"]=yhat
    pred["revenue"]=pred["avg_price"]*pred["units"]

    history = pd.concat([history, pred], ignore_index=True, sort=False)
    forecasts.append(pred[["order_week","product_category_name_english","avg_price","units","revenue"]])

fcast = pd.concat(forecasts, ignore_index=True)
fcast.to_csv(OUT/"forecast_no_leak.csv", index=False)
print("Saved:", OUT/"forecast_no_leak.csv")

P]--------> ## # # plot_validation_accuracy.py ##

# plot_validation_accuracy.py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# pick whichever predictions file exists
cand = [OUT/"valid_predictions_no_leak.csv", OUT/"valid_predictions.csv"]
pred_path = next((p for p in cand if p.exists()), None)
if pred_path is None:
    raise FileNotFoundError("No validation predictions file found in Dataset/model_outputs/.")

pred = pd.read_csv(pred_path, parse_dates=["order_week"])
pred.rename(columns={
    "category": "product_category_name_english",
    "pred_units": "yhat",
    "actual_units": "y"
}, inplace=True)

# safety: keep clean datetimes & numerics
pred["order_week"] = pd.to_datetime(pred["order_week"], errors="coerce")
pred["y"] = pd.to_numeric(pred["y"], errors="coerce")
pred["yhat"] = pd.to_numeric(pred["yhat"], errors="coerce")
pred = pred.dropna(subset=["order_week", "y", "yhat"])

# choose categories to plot (top by total actual units in validation)
TOP_N = 12
cats = (pred.groupby("product_category_name_english")["y"]
            .sum().sort_values(ascending=False).head(TOP_N).index.tolist())

def clean_xy(df):
    x = pd.to_datetime(df["order_week"], errors="coerce")
    y1 = pd.to_numeric(df["y"], errors="coerce")
    y2 = pd.to_numeric(df["yhat"], errors="coerce")
    m = x.notna() & y1.notna() & y2.notna()
    x = x[m].to_numpy(dtype="datetime64[ns]")  # keeps matplotlib date converter happy
    y1 = y1[m].to_numpy(dtype=float)
    y2 = y2[m].to_numpy(dtype=float)
    return x, y1, y2

# per-category plots with metrics
for c in cats:
    df = pred[pred["product_category_name_english"]==c].sort_values("order_week")
    if df.empty:
        continue
    x, y, yhat = clean_xy(df)

    mae  = np.mean(np.abs(y - yhat))
    mape = np.mean(np.abs((y - yhat) / np.maximum(y, 1e-6)))
    r    = np.corrcoef(y, yhat)[0,1] if len(y) > 1 else np.nan
    r2   = r**2 if not np.isnan(r) else np.nan

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y,    label="Actual",   linewidth=2)
    ax.plot(x, yhat, label="Predicted", linewidth=2)

    ax.set_title(f"Validation — {c}")
    ax.set_xlabel("Week"); ax.set_ylabel("Units")
    ax.legend()

    # neat date ticks
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    # annotate metrics
    txt = f"MAE={mae:.2f}  MAPE={mape:.2%}  R²={r2:.3f}"
    ax.text(0.01, 0.99, txt, transform=ax.transAxes, va="top", ha="left")

    fig.tight_layout()
    fig.savefig(OUT / f"val_ts_{c.replace('/','_')}.png", dpi=150)
    plt.close(fig)

# (optional) overall scatter with y=x
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(pred["y"], pred["yhat"], s=12, alpha=0.6)
lims = [0, max(pred["y"].max(), pred["yhat"].max())*1.05]
ax.plot(lims, lims, linestyle="--")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual units"); ax.set_ylabel("Predicted units")
ax.set_title("Validation — Actual vs Predicted")
fig.tight_layout()
fig.savefig(OUT/"val_scatter_overall.png", dpi=150)
plt.close(fig)

print("Saved per-category PNGs (val_ts_*.png) and val_scatter_overall.png in:", OUT)


Q]--------> ## # # evaluate_validation.py ##


# evaluate_validation.py  (fixed)
import numpy as np
import pandas as pd
from pathlib import Path

# Use a non-interactive backend to avoid Tkinter errors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Pick whichever predictions file exists
pred_path = None
for p in [OUT/"valid_predictions_no_leak.csv", OUT/"valid_predictions.csv"]:
    if p.exists():
        pred_path = p
        break
if pred_path is None:
    raise FileNotFoundError("No valid_predictions*.csv found in Dataset/model_outputs/")

# Read & parse dates (dayfirst) and normalize columns
df = pd.read_csv(pred_path)
df["order_week"] = pd.to_datetime(df["order_week"], dayfirst=True, errors="coerce")
df = df.rename(columns={
    "category": "product_category_name_english",
    "pred_units": "yhat",
    "actual_units": "y",
})
df = df.dropna(subset=["order_week", "y", "yhat"])
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")
df = df.dropna(subset=["y", "yhat"])

# Overall metrics
df["abs_err"] = (df["y"] - df["yhat"]).abs()
mae   = float(df["abs_err"].mean())
wmape = float(df["abs_err"].sum() / max(1e-9, df["y"].abs().sum()))
bias  = float((df["yhat"] - df["y"]).mean())
corr  = float(np.corrcoef(df["y"], df["yhat"])[0, 1]) if len(df) > 1 else np.nan
r2    = float(corr**2) if np.isfinite(corr) else np.nan

pd.Series({"MAE": mae, "wMAPE": wmape, "Bias": bias, "R2": r2}).to_csv(
    OUT / "validation_overall_metrics.csv"
)

# Per-category metrics (no .apply deprecation)
df["bias"] = df["yhat"] - df["y"]
per = (df.groupby("product_category_name_english", as_index=False)
         .agg(n=("y", "size"),
              sum_abs_err=("abs_err", "sum"),
              sum_abs_y=("y", lambda s: s.abs().sum()),
              MAE=("abs_err", "mean"),
              Bias=("bias", "mean"),
              Actual_sum=("y", "sum")))
per["wMAPE"] = per["sum_abs_err"] / per["sum_abs_y"].clip(lower=1e-9)
per = per.sort_values("wMAPE", ascending=False)
per.to_csv(OUT / "validation_by_category.csv", index=False)

# Parity plot + calibration line (ŷ vs y)
lims = [0, max(df["y"].max(), df["yhat"].max()) * 1.05]
# Linear fit y ≈ a + b·ŷ
b, a = np.polyfit(df["yhat"].values, df["y"].values, 1)

plt.figure(figsize=(6, 6))
plt.scatter(df["y"], df["yhat"], s=16, alpha=0.6)
plt.plot(lims, lims, "--", label="y=x")
# Show calibration line rearranged into predicted space: ŷ ≈ (y - a)/b
if abs(b) > 1e-9:
    plt.plot(lims, [(y - a) / b for y in lims], "-", label="calibration", linewidth=2)
plt.xlabel("Actual units"); plt.ylabel("Predicted units")
plt.title("Validation — Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "val_parity_with_calibration.png", dpi=150)
plt.close()

# Worst categories by wMAPE and bias
top_bad = per[per["Actual_sum"] > 0].head(15)
plt.figure(figsize=(9, 5))
plt.barh(top_bad["product_category_name_english"], top_bad["wMAPE"])
plt.gca().invert_yaxis()
plt.xlabel("wMAPE"); plt.title("Worst categories by wMAPE (validation)")
plt.tight_layout(); plt.savefig(OUT / "val_worst_wmape.png", dpi=150); plt.close()

top_bias = per.sort_values("Bias", ascending=False).head(15)
plt.figure(figsize=(9, 5))
plt.barh(top_bias["product_category_name_english"], top_bias["Bias"])
plt.gca().invert_yaxis()
plt.xlabel("Bias (ŷ − y)"); plt.title("Most over-predicted categories")
plt.tight_layout(); plt.savefig(OUT / "val_worst_bias_over.png", dpi=150); plt.close()

bot_bias = per.sort_values("Bias").head(15)
plt.figure(figsize=(9, 5))
plt.barh(bot_bias["product_category_name_english"], bot_bias["Bias"])
plt.gca().invert_yaxis()
plt.xlabel("Bias (ŷ − y)"); plt.title("Most under-predicted categories")
plt.tight_layout(); plt.savefig(OUT / "val_worst_bias_under.png", dpi=150); plt.close()

# Save calibration params
pd.Series({"intercept": a, "slope": b}).to_csv(OUT / "calibration_params.csv")
print("Saved: validation_overall_metrics.csv, validation_by_category.csv, val_parity_with_calibration.png, val_worst_*.png, calibration_params.csv")

R]--------> ## # # plot_validation_overlay ##

# plot_validation_overlay.py
import numpy as np
import pandas as pd
from pathlib import Path

# non-interactive backend (no Tk errors)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ---- load predictions (auto-pick no_leak if present) ----
pred_path = None
for p in [OUT/"valid_predictions_no_leak.csv", OUT/"valid_predictions.csv"]:
    if p.exists():
        pred_path = p; break
if pred_path is None:
    raise FileNotFoundError("No valid_predictions*.csv found in Dataset/model_outputs/")

pred = pd.read_csv(pred_path)
pred["order_week"] = pd.to_datetime(pred["order_week"], dayfirst=True, errors="coerce")
pred = pred.rename(columns={"category":"product_category_name_english",
                            "actual_units":"y", "pred_units":"yhat"})
pred = pred.dropna(subset=["order_week","y","yhat"])

# ---- load full history for context ----
hist = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"])
hist = hist.sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)

# pick top categories by validation volume
TOP_N   = 12
LOOKBACK_WEEKS = 26  # weeks of history to show before validation start

cats = (pred.groupby("product_category_name_english")["y"]
            .sum().sort_values(ascending=False).head(TOP_N).index.tolist())

def per_cat_metrics(df):
    mae   = float(np.mean(np.abs(df.y - df.yhat)))
    wmape = float(np.sum(np.abs(df.y - df.yhat)) / max(1e-9, np.sum(np.abs(df.y))))
    r     = np.corrcoef(df.y, df.yhat)[0,1] if len(df) > 1 else np.nan
    r2    = float(r**2) if np.isfinite(r) else np.nan
    return mae, wmape, r2

for c in cats:
    pv = pred[pred["product_category_name_english"]==c].sort_values("order_week").copy()
    if pv.empty: continue

    # context history (last LOOKBACK_WEEKS before validation)
    vstart = pv["order_week"].min()
    ctx = hist[(hist["product_category_name_english"]==c) &
               (hist["order_week"] < vstart)].tail(LOOKBACK_WEEKS)

    mae, wmape, r2 = per_cat_metrics(pv)

    fig, ax = plt.subplots(figsize=(10,4))

    # light-gray context history
    if not ctx.empty:
        ax.plot(ctx["order_week"], ctx["units"], color="#bbbbbb", label="history (context)")

    # validation actual & predicted
    ax.plot(pv["order_week"], pv["y"],    label="Actual (validation)", linewidth=2)
    ax.plot(pv["order_week"], pv["yhat"], label="Predicted (validation)", linewidth=2)

    ax.set_title(f"Validation — {c}   MAE={mae:.2f}  wMAPE={wmape:.2%}  R²={r2:.3f}")
    ax.set_xlabel("Week"); ax.set_ylabel("Units")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"val_overlay_{c.replace('/','_')}.png", dpi=150)
    plt.close(fig)

print("Saved per-category overlays to:", OUT)

import pandas as pd
from pathlib import Path
OUT = Path("Dataset/model_outputs")
per = pd.read_csv(OUT/"validation_by_category.csv")
# helpful ranking views
per.assign(wMAPE_pct=(per.wMAPE*100))\
   .sort_values(["wMAPE","Actual_sum"], ascending=[True,False])\
   .to_csv(OUT/"validation_leaderboard.csv", index=False)
print("Saved:", OUT/"validation_leaderboard.csv")

S]--------> ## # # build_validation_actions.py ##

# Build validation actions

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("Dataset/model_outputs")

# 1) Load
per = pd.read_csv(OUT / "validation_by_category.csv")

pred_path = OUT / "valid_predictions_no_leak.csv"
if not pred_path.exists():
    pred_path = OUT / "valid_predictions.csv"

pred = pd.read_csv(pred_path)
pred["order_week"] = pd.to_datetime(pred["order_week"], dayfirst=True, errors="coerce")
pred = pred.rename(columns={"category":"product_category_name_english",
                            "actual_units":"y", "pred_units":"yhat"})
pred = pred.dropna(subset=["product_category_name_english","y","yhat"])

hist = pd.read_csv(OUT / "weekly_by_category_full.csv", parse_dates=["order_week"])
vmin, vmax = pred["order_week"].min(), pred["order_week"].max()
price_val = (hist[(hist["order_week"]>=vmin)&(hist["order_week"]<=vmax)]
             .groupby("product_category_name_english", as_index=False)
             .agg(avg_price_val=("avg_price","mean")))

# ---------- 2) Enrich metrics ----------
if "sum_abs_err" not in per.columns or "sum_abs_y" not in per.columns:
    tmp = pred.copy(); tmp["abs_err"] = (tmp["y"]-tmp["yhat"]).abs()
    recon = (tmp.groupby("product_category_name_english", as_index=False)
               .agg(sum_abs_err=("abs_err","sum"),
                    sum_abs_y=("y", lambda s: s.abs().sum())))
    per = per.merge(recon, on="product_category_name_english", how="left")

per = per.merge(price_val, on="product_category_name_english", how="left")
per["wMAPE_pct"]      = per["wMAPE"]*100.0
per["Revenue_at_Risk"]= per["sum_abs_err"] * per["avg_price_val"].fillna(0.0)
per["PriorityScore"]  = per["wMAPE"] * per["Actual_sum"].abs()

# ---------- 3) Per-category calibration ----------
def fit_calib(g: pd.DataFrame) -> pd.Series:
    if len(g)>=2 and g["yhat"].nunique()>=2:
        b, a = np.polyfit(g["yhat"].to_numpy(), g["y"].to_numpy(), 1)
    else:
        b, a = 1.0, 0.0
    return pd.Series({"scale_b": float(b),
                      "offset_a": float(a),
                      "Bias_units": float((g["yhat"]-g["y"]).mean())})

calib = (pred.groupby("product_category_name_english")
           .apply(fit_calib, include_groups=False)  # <- silence future warning
           .reset_index())

per = per.merge(calib, on="product_category_name_english", how="left")

def make_rec(row):
    bias = float(row.get("Bias", 0.0))
    if bias < -5:
        return f"Underpredict by {-bias:.1f} u; try scale x{row['scale_b']:.2f} or add {row['offset_a']:.1f}"
    if bias > 5:
        return f"Overpredict by {bias:.1f} u; try scale x{row['scale_b']:.2f} or add {row['offset_a']:.1f}"
    return f"OK; keep global calibration (x{row['scale_b']:.2f}, +{row['offset_a']:.1f})"

per["Recommendation"] = per.apply(make_rec, axis=1)

# ---------- 4) Actions table ----------
actions = (per.sort_values(["PriorityScore","Revenue_at_Risk"], ascending=False)[
    ["product_category_name_english","n","MAE","wMAPE_pct","Bias",
     "Actual_sum","avg_price_val","sum_abs_err","Revenue_at_Risk",
     "PriorityScore",  # <- keep it so we can sort later
     "scale_b","offset_a","Recommendation"]
])
actions.to_csv(OUT / "validation_actions.csv", index=False)
print("Saved:", OUT / "validation_actions.csv")

# 5) PNGs
topR = actions.head(15)
plt.figure(figsize=(9,5))
plt.barh(topR["product_category_name_english"], topR["Revenue_at_Risk"])
plt.gca().invert_yaxis()
plt.xlabel("Revenue at Risk (validation)")
plt.title("Top categories by $ error (validation)")
plt.tight_layout(); plt.savefig(OUT/"val_top_revenue_at_risk.png", dpi=150); plt.close()

topP = actions.sort_values("PriorityScore", ascending=False).head(15)
plt.figure(figsize=(9,5))
plt.barh(topP["product_category_name_english"], topP["PriorityScore"])
plt.gca().invert_yaxis()
plt.xlabel("PriorityScore = wMAPE × Actual_sum")
plt.title("Top categories to fix first")
plt.tight_layout(); plt.savefig(OUT/"val_top_priority.png", dpi=150); plt.close()

print("Saved PNGs: val_top_revenue_at_risk.png, val_top_priority.png")

T]--------> ## # # apply_calibration_to_forecast.py ##

#Apply calibration to forecast

import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("Dataset/model_outputs")

# 1) load forecast (use the one you generated)
f_path = OUT / "forecast_no_leak.csv"
if not f_path.exists():
    f_path = OUT / "forecast_with_plan.csv"  # fallback

fore = pd.read_csv(f_path, parse_dates=["order_week"])
# expected cols: order_week, product_category_name_english, avg_price, units, revenue

# 2) load per-category calibration
act = pd.read_csv(OUT / "validation_actions.csv")

# keep only what we need
cal = act[["product_category_name_english","scale_b","offset_a"]].copy()
cal["scale_b"]  = cal["scale_b"].fillna(1.0)
cal["offset_a"] = cal["offset_a"].fillna(0.0)

# 3) join + calibrate
df = fore.merge(cal, on="product_category_name_english", how="left")
df["scale_b"]  = df["scale_b"].fillna(1.0)
df["offset_a"] = df["offset_a"].fillna(0.0)

df["units_cal"] = (df["offset_a"] + df["scale_b"] * df["units"]).clip(lower=0)
df["revenue_cal"] = df["avg_price"] * df["units_cal"]

# 4) save
cal_path = OUT / f_path.name.replace(".csv", "_calibrated.csv")
df.to_csv(cal_path, index=False)
print("Saved calibrated forecast:", cal_path)

# 5) quick visual on the TOP priority categories
top = act.sort_values("PriorityScore", ascending=False)["product_category_name_english"].head(6).tolist()
plot_df = df[df["product_category_name_english"].isin(top)].copy()

for cat in top:
    g = plot_df[plot_df["product_category_name_english"]==cat].sort_values("order_week")
    plt.figure(figsize=(10,4))
    plt.plot(g["order_week"], g["units"],      label="base units")
    plt.plot(g["order_week"], g["units_cal"],  label="calibrated units")
    plt.title(f"Forecast units — {cat}")
    plt.xlabel("Week"); plt.ylabel("Units"); plt.legend()
    fn = OUT / f"forecast_calibrated_vs_base_{cat}.png"
    plt.tight_layout(); plt.savefig(fn, dpi=140); plt.close()
    print("Saved:", fn)

# 6) aggregate impact
impact = (df.groupby("product_category_name_english", as_index=False)
            .agg(base_units=("units","sum"),
                 cal_units=("units_cal","sum"),
                 base_rev=("revenue","sum"),
                 cal_rev=("revenue_cal","sum")))
impact["Δunits"] = impact["cal_units"] - impact["base_units"]
impact["Δrevenue"] = impact["cal_rev"] - impact["base_rev"]
impact.sort_values("Δrevenue", ascending=False)\
      .to_csv(OUT / "calibration_impact_summary.csv", index=False)
print("Saved:", OUT / "calibration_impact_summary.csv")


U]--------> ## # # unify_forecast_and_rollup.py ##


# Create unified forecast & rollup

import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Helpers
def read_first_that_exists(paths, parse_dates=None):
    for p in paths:
        p = OUT / p
        if p.exists():
            return pd.read_csv(p, parse_dates=parse_dates)
    raise FileNotFoundError(f"None of these files exist under {OUT}: {paths}")

def dedupe_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Keep first occurrence of duplicate column names."""
    return df.loc[:, ~df.columns.duplicated(keep="first")].copy()

# -------- 1) Load base & calibrated forecasts --------
# Base forecast (pick your available file)
base_fcst = read_first_that_exists(
    ["forecast_no_leak.csv", "forecast_with_plan.csv", "forecast.csv"],
    parse_dates=["order_week"]
).copy()

# Standardise base columns
# expected columns: order_week, product_category_name_english, avg_price, units, revenue (revenue may or may not exist)
base_fcst.rename(columns={"units": "units_base", "revenue": "revenue_base"}, inplace=True)
if "avg_price" not in base_fcst.columns:
    base_fcst["avg_price"] = np.nan

# Calibrated forecast (if you produced it)
try:
    cal_fcst = read_first_that_exists(
        ["forecast_no_leak_calibrated.csv", "forecast_calibrated.csv"],
        parse_dates=["order_week"]
    ).copy()
    # keep only what we need and standardize
    keep = ["order_week", "product_category_name_english", "units", "revenue"]
    have = [c for c in keep if c in cal_fcst.columns]
    cal_fcst = cal_fcst[have].copy()
    if "units" in cal_fcst.columns:
        cal_fcst.rename(columns={"units": "units_cal"}, inplace=True)
    if "revenue" in cal_fcst.columns:
        cal_fcst.rename(columns={"revenue": "revenue_cal"}, inplace=True)
except FileNotFoundError:
    # No calibrated file; create empty frame so merge still works
    cal_fcst = pd.DataFrame(columns=["order_week","product_category_name_english","units_cal","revenue_cal"])

# Optional price plan
plan = None
plan_path = OUT / "price_plan.csv"
if plan_path.exists():
    plan = pd.read_csv(plan_path, parse_dates=["order_week"])
    # expected: order_week, product_category_name_english, planned_price (or base_price + planned_price)
    # normalise column name if needed
    if "planned_price" not in plan.columns and "plan_price" in plan.columns:
        plan = plan.rename(columns={"plan_price": "planned_price"})

# -------- 2) Merge into one table safely --------
# merge on week + category (do NOT join on price so we keep base avg_price as a single column)
fcst = (base_fcst.merge(
            cal_fcst,
            on=["order_week", "product_category_name_english"],
            how="outer",
            suffixes=("", "_calfile")
        )
        .sort_values(["product_category_name_english", "order_week"])
        .reset_index(drop=True))

# remove any duplicate-named columns introduced by repeated runs
fcst = dedupe_cols(fcst)

# attach planned price, if present
if plan is not None and "planned_price" in plan.columns:
    fcst = fcst.merge(
        plan[["order_week", "product_category_name_english", "planned_price"]],
        on=["order_week", "product_category_name_english"],
        how="left"
    )
    fcst = dedupe_cols(fcst)
    fcst["price_for_revenue"] = fcst["planned_price"].fillna(fcst.get("avg_price"))
else:
    fcst["price_for_revenue"] = fcst.get("avg_price")

# -------- 3) Ensure numeric Series and compute revenues --------
for col in ["price_for_revenue", "units_base", "units_cal", "avg_price"]:
    if col in fcst.columns:
        fcst[col] = pd.to_numeric(fcst[col], errors="coerce")

# if calibrated units missing, leave NaN (we’ll still compute base)
if "units_base" not in fcst: fcst["units_base"] = np.nan
if "units_cal"  not in fcst: fcst["units_cal"]  = np.nan

# revenues at the chosen price_for_revenue (independent of any precomputed revenue columns)
fcst["revenue_base_price"] = fcst["price_for_revenue"] * fcst["units_base"]
fcst["revenue_cal_price"]  = fcst["price_for_revenue"] * fcst["units_cal"]

# 4) Tidy columns & save unified forecast
keep_cols = [
    "order_week", "product_category_name_english",
    "avg_price", "planned_price", "price_for_revenue",
    "units_base", "units_cal",
    "revenue_base_price", "revenue_cal_price"
]
keep_cols = [c for c in keep_cols if c in fcst.columns]
fcst_tidy = fcst[keep_cols].copy()

unified_path = OUT / "forecast_unified.csv"
fcst_tidy.to_csv(unified_path, index=False)
print("Saved unified forecast:", unified_path)

# 5) Roll-ups
# by week & category
by_cat = (fcst_tidy.groupby(["order_week", "product_category_name_english"], as_index=False)
          .agg(units_base=("units_base","sum"),
               units_cal=("units_cal","sum"),
               revenue_base=("revenue_base_price","sum"),
               revenue_cal=("revenue_cal_price","sum")))

# overall by week
overall = (fcst_tidy.groupby("order_week", as_index=False)
           .agg(units_base=("units_base","sum"),
                units_cal=("units_cal","sum"),
                revenue_base=("revenue_base_price","sum"),
                revenue_cal=("revenue_cal_price","sum")))

# horizon totals
def _safe_sum(s):
    return float(np.nansum(s.values)) if len(s) else 0.0

summary = pd.DataFrame({
    "metric": ["units_base", "units_cal", "revenue_base", "revenue_cal"],
    "value": [
        _safe_sum(fcst_tidy["units_base"]),
        _safe_sum(fcst_tidy["units_cal"]),
        _safe_sum(fcst_tidy["revenue_base_price"]),
        _safe_sum(fcst_tidy["revenue_cal_price"]),
    ]
})
summary["delta_vs_base"] = np.nan
if "units_cal" in fcst_tidy:
    summary.loc[summary["metric"]=="units_cal", "delta_vs_base"] = (
        summary.loc[summary["metric"]=="units_cal","value"].values[0] -
        summary.loc[summary["metric"]=="units_base","value"].values[0]
    )
if "revenue_cal_price" in fcst_tidy:
    summary.loc[summary["metric"]=="revenue_cal", "delta_vs_base"] = (
        summary.loc[summary["metric"]=="revenue_cal","value"].values[0] -
        summary.loc[summary["metric"]=="revenue_base","value"].values[0]
    )

# Save roll-ups
by_cat_path   = OUT / "forecast_rollup_by_category.csv"
overall_path  = OUT / "forecast_rollup_overall.csv"
summary_path  = OUT / "forecast_summary.csv"

by_cat.to_csv(by_cat_path, index=False)
overall.to_csv(overall_path, index=False)
summary.to_csv(summary_path, index=False)

print("Saved roll-ups:")
print("  by category ->", by_cat_path)
print("  overall     ->", overall_path)
print("  summary     ->", summary_path)

# Quick sanity prints
print("\nRows (unified/base):", len(fcst_tidy), "/", len(base_fcst))
print("Horizon weeks:", fcst_tidy["order_week"].nunique())
print("Categories   :", fcst_tidy["product_category_name_english"].nunique())


V]--------> ## # # qa_and_readout.py ##

# QA and readout

import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path("Dataset/model_outputs")

# --- Load required files ---
unified = pd.read_csv(OUT/"forecast_unified.csv", parse_dates=["order_week"])
roll_cat = pd.read_csv(OUT/"forecast_rollup_by_category.csv", parse_dates=["order_week"])
roll_all = pd.read_csv(OUT/"forecast_rollup_overall.csv", parse_dates=["order_week"])

# Optional summary (not required)
summary_path = OUT/"forecast_summary.csv"
summary = None
if summary_path.exists():
    try:
        summary = pd.read_csv(summary_path)
    except Exception as e:
        print(f"(Note) Could not read forecast_summary.csv cleanly: {e}")

# ---------- 1) QA ----------

# a) no duplicate (week, category) rows
dupes = unified.duplicated(subset=["order_week","product_category_name_english"]).sum()
if dupes:
    raise AssertionError(f"Found {dupes} duplicated (week,category) rows in forecast_unified.")

# b) required columns present?
req_base = ["order_week","product_category_name_english"]
missing_req = [c for c in req_base if c not in unified.columns]
if missing_req:
    raise AssertionError(f"Missing required columns in forecast_unified: {missing_req}")

# choose price column (fallbacks)
price_col = None
for c in ["price_for_revenue","planned_price","avg_price"]:
    if c in unified.columns:
        price_col = c
        break

# numeric fields to check if present
maybe_cols = ["units_base","units_cal","revenue_base_price","revenue_cal_price"]
check_cols = [c for c in ([price_col] if price_col else []) + maybe_cols if c in unified.columns]

# c) NaN checks only on columns that exist
for col in check_cols:
    n = unified[col].isna().sum()
    if n:
        raise AssertionError(f"NaNs in {col}: {n}")

# d) Recompute rollups from unified and compare with saved rollups
# by category-week
chk_cat = (unified.groupby(["order_week","product_category_name_english"], as_index=False)
                  .agg(units_cal=("units_cal","sum") if "units_cal" in unified else ("order_week","size"),
                       revenue_cal=("revenue_cal_price","sum") if "revenue_cal_price" in unified else ("order_week","size")))

m = roll_cat.merge(chk_cat, on=["order_week","product_category_name_english"],
                   how="outer", suffixes=("_roll","_uni"))
for want_roll, want_uni in [("units_cal_roll","units_cal_uni"), ("revenue_cal_roll","revenue_cal_uni")]:
    if want_roll in m.columns and want_uni in m.columns:
        diff = (m[want_roll].fillna(0) - m[want_uni].fillna(0)).abs().sum()
        if diff >= 1e-6:
            raise AssertionError(f"Mismatched totals vs rollup_by_category for {want_roll.split('_')[0]} (sum abs diff={diff}).")

# overall
chk_all = (unified.groupby("order_week", as_index=False)
                  .agg(units_cal=("units_cal","sum") if "units_cal" in unified else ("order_week","size"),
                       revenue_cal=("revenue_cal_price","sum") if "revenue_cal_price" in unified else ("order_week","size")))

m2 = roll_all.merge(chk_all, on="order_week", how="outer", suffixes=("_roll","_uni"))
for want_roll, want_uni in [("units_cal_roll","units_cal_uni"), ("revenue_cal_roll","revenue_cal_uni")]:
    if want_roll in m2.columns and want_uni in m2.columns:
        diff = (m2[want_roll].fillna(0) - m2[want_uni].fillna(0)).abs().sum()
        if diff >= 1e-6:
            raise AssertionError(f"Mismatched totals vs rollup_overall for {want_roll.split('_')[0]} (sum abs diff={diff}).")

print("QA passed ✔")

# ---------- 2) KPIs ----------
tot_units = roll_all["units_cal"].sum() if "units_cal" in roll_all.columns else np.nan
tot_revenue = roll_all["revenue_cal"].sum() if "revenue_cal" in roll_all.columns else np.nan
print("8-week totals:", {"total_units_8w": tot_units, "total_revenue_8w": tot_revenue})

top_rev = (unified.groupby("product_category_name_english", as_index=False)
           .agg(rev_8w=("revenue_cal_price","sum") if "revenue_cal_price" in unified else ("order_week","size"),
                units_8w=("units_cal","sum") if "units_cal" in unified else ("order_week","size"))
           .sort_values("rev_8w", ascending=False))
top_rev.to_csv(OUT/"kpi_top_categories.csv", index=False)

# ---------- 3) Charts ----------
plt.figure(figsize=(10,4))
if {"units_base","units_cal"}.issubset(roll_all.columns):
    plt.plot(roll_all["order_week"], roll_all["units_base"], label="base units")
    plt.plot(roll_all["order_week"], roll_all["units_cal"],  label="calibrated units")
else:
    plt.plot(roll_all["order_week"], roll_all.iloc[:,1], label=roll_all.columns[1])
    if roll_all.shape[1] > 2:
        plt.plot(roll_all["order_week"], roll_all.iloc[:,2], label=roll_all.columns[2])
plt.title("Overall forecast — base vs calibrated (units)")
plt.xlabel("Week"); plt.ylabel("Units"); plt.legend()
plt.tight_layout(); plt.savefig(OUT/"overall_units_base_vs_cal.png", dpi=160)

plt.figure(figsize=(10,4))
if {"revenue_base","revenue_cal"}.issubset(roll_all.columns):
    plt.plot(roll_all["order_week"], roll_all["revenue_base"], label="base revenue")
    plt.plot(roll_all["order_week"], roll_all["revenue_cal"],  label="calibrated revenue")
else:
    plt.plot(roll_all["order_week"], roll_all.iloc[:,1], label=roll_all.columns[1])
    if roll_all.shape[1] > 2:
        plt.plot(roll_all["order_week"], roll_all.iloc[:,2], label=roll_all.columns[2])
plt.title("Overall forecast — base vs calibrated (revenue)")
plt.xlabel("Week"); plt.ylabel("Revenue"); plt.legend()
plt.tight_layout(); plt.savefig(OUT/"overall_revenue_base_vs_cal.png", dpi=160)

top10 = top_rev.head(10).sort_values("rev_8w")
plt.figure(figsize=(10,5))
plt.barh(top10["product_category_name_english"], top10["rev_8w"])
plt.title("Top categories by 8-week forecast revenue (calibrated)")
plt.xlabel("Revenue"); plt.tight_layout()
plt.savefig(OUT/"top10_categories_revenue.png", dpi=160)

print("Saved:",
      OUT/"overall_units_base_vs_cal.png",
      OUT/"overall_revenue_base_vs_cal.png",
      OUT/"top10_categories_revenue.png",
      OUT/"kpi_top_categories.csv")


W]--------> ## # # calibration_scatter.py ##

# Calibration_scatter

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Helpers
def load_csv(path_candidates, parse_dates=None):
    for p in path_candidates:
        if p.exists():
            return pd.read_csv(p, parse_dates=parse_dates), p
    return None, None

def normalize_cols(df):
    # standardise likely names
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"category","cat","product_category"}:
            rename[c] = "product_category_name_english"
        elif cl in {"week","orderweek","week_start","order_week"}:
            rename[c] = "order_week"
        elif cl in {"units_base","base_units","units_pred","units_model","units0"}:
            rename[c] = "units_base"
        elif cl in {"units_cal","calibrated_units","units_calibrated"}:
            rename[c] = "units_cal"
        elif cl == "units":
            # treat 'units' as base when there is also a 'units_cal' elsewhere
            # (we won't rename here unless needed later)
            pass
    return df.rename(columns=rename)

# Try best source first: unified with both columns
unified, unified_path = load_csv([
    OUT/"forecast_unified.csv",                 # produced in your last steps
    OUT/"forecast_with_calibration.csv"
], parse_dates=["order_week"])

if unified is not None:
    unified = normalize_cols(unified)

# Otherwise: merge baseline + calibrated
if unified is None or not ({"units_base","units_cal"} <= set(unified.columns)):
    base, base_path = load_csv([OUT/"forecast_no_leak.csv",
                                OUT/"forecast.csv"], parse_dates=["order_week"])
    cal,  cal_path  = load_csv([OUT/"forecast_no_leak_calibrated.csv",
                                OUT/"forecast_calibrated.csv"], parse_dates=["order_week"])

    if base is not None: base = normalize_cols(base)
    if cal  is not None: cal  = normalize_cols(cal)

    if base is not None and cal is not None:
        # pick a safe key set for merge
        key_cols = [c for c in ["order_week","product_category_name_english"] if c in cal.columns and c in base.columns]
        if not key_cols:
            raise RuntimeError("Cannot find common keys to merge baseline and calibrated forecasts.")

        # if baseline has 'units' but no 'units_base', rename it
        if "units_base" not in base.columns and "units" in base.columns:
            base = base.rename(columns={"units": "units_base"})

        unified = cal.merge(base[key_cols+["units_base"]], on=key_cols, how="left")
        unified_path = cal_path

# Last resort: reconstruct units_base from params
if unified is None:
    raise FileNotFoundError("Could not find any forecast CSVs to plot.")

unified = normalize_cols(unified)

# If we still miss units_base, try inversion using calibration params: units_cal = a + b*units_base
if "units_base" not in unified.columns:
    if "units_cal" not in unified.columns:
        raise RuntimeError(f"{unified_path.name} has neither 'units_base' nor 'units_cal'.")

    params_path = OUT/"calibration_params.csv"
    if not params_path.exists():
        raise RuntimeError("Missing 'units_base' in data and 'calibration_params.csv' not found to reconstruct it.")

    cp = pd.read_csv(params_path)
    cp.columns = [c.lower() for c in cp.columns]
    cp = cp.rename(columns={"category":"product_category_name_english"})
    if not {"a","b","product_category_name_english"} <= set(cp.columns):
        raise RuntimeError("calibration_params.csv must contain columns: category, a, b")

    unified = unified.merge(cp[["product_category_name_english","a","b"]], on="product_category_name_english", how="left")
    if unified["b"].isna().any():
        raise RuntimeError("Some categories in forecast are missing calibration params (b is NaN).")
    unified["units_base"] = (unified["units_cal"] - unified["a"]) / unified["b"]
    # keep for plotting
    HAVE_PARAMS = True
else:
    HAVE_PARAMS = False

# If we still miss units_cal but have base & params, forward-apply
if "units_cal" not in unified.columns:
    params_path = OUT/"calibration_params.csv"
    if not params_path.exists():
        raise RuntimeError("Missing 'units_cal' in data and no calibration_params.csv to compute it.")
    cp = pd.read_csv(params_path)
    cp.columns = [c.lower() for c in cp.columns]
    cp = cp.rename(columns={"category":"product_category_name_english"})
    unified = unified.merge(cp[["product_category_name_english","a","b"]], on="product_category_name_english", how="left")
    unified["units_cal"] = unified["a"] + unified["b"] * unified["units_base"]
    HAVE_PARAMS = True

# Plotting
cats = unified["product_category_name_english"].dropna().unique()
save_dir = OUT / "calibration_scatter"
save_dir.mkdir(exist_ok=True)

for cat in cats:
    df = unified.loc[unified["product_category_name_english"]==cat, ["units_base","units_cal"]].dropna()
    if len(df) < 3:
        continue

    # If params provided/merged, use them; otherwise fit
    if {"a","b"}.issubset(unified.columns):
        row = unified.loc[unified["product_category_name_english"]==cat, ["a","b"]].dropna().tail(1)
        if not row.empty:
            a = float(row["a"].iloc[0]); b = float(row["b"].iloc[0])
        else:
            b, a = np.polyfit(df["units_base"], df["units_cal"], 1)
    else:
        b, a = np.polyfit(df["units_base"], df["units_cal"], 1)

    x = np.linspace(df["units_base"].min(), df["units_base"].max(), 100)
    y = a + b*x

    plt.figure(figsize=(6,6))
    plt.scatter(df["units_base"], df["units_cal"], alpha=0.5)
    plt.plot(x, y, linestyle="--")
    lim = [0, max(df["units_base"].max(), df["units_cal"].max())*1.05]
    plt.plot(lim, lim, alpha=0.4)  # 45° line
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("Base units (model)")
    plt.ylabel("Calibrated units")
    plt.title(f"Calibration — {cat}\n y = {a:.2f} + {b:.3f} x")
    plt.tight_layout()
    plt.savefig(save_dir / f"calibration_scatter_{cat}.png", dpi=150)
    plt.close()

print(f"Saved per-category calibration scatter plots to: {save_dir}")


X]--------> ## # # elasticity_explorer.py ##

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# paths & inputs

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# core artefacts produced earlier
wk_full  = (pd.read_csv(OUT/"weekly_by_category_full.csv",
                        parse_dates=["order_week"])
              .sort_values(["product_category_name_english","order_week"]))

wk_model = pd.read_csv(OUT/"weekly_by_category_no_leak.csv",
                       parse_dates=["order_week"])

pipe     = joblib.load(OUT/"weekly_units_xgb_no_leak.pkl")

# optional calibration (units_cal = a + b * units_base)
calib_path = OUT/"calibration_params.csv"
calib = None
if calib_path.exists():
    calib = pd.read_csv(calib_path)
    calib.columns = [c.lower() for c in calib.columns]
    calib = calib.rename(columns={"category":"product_category_name_english"})


# features exactly like training

TARGET="units"; cat_cols=["product_category_name_english"]
drop_cols={"order_week","split",TARGET,"revenue"}  # 'revenue' excluded by design
num_cols=[c for c in wk_model.columns if c not in set(cat_cols)|drop_cols]
FEAT = cat_cols + num_cols  # columns fed into the pipeline


# helper: trim trailing partial weeks from history (same as before)

def trim_tail(df, lookback=8, frac=0.25, max_drop=2):
    out=[]
    for cat, g in df.groupby("product_category_name_english", sort=False):
        g=g.sort_values("order_week").copy(); drops=0
        while len(g)>=lookback+1 and drops<max_drop:
            med=g["units"].tail(lookback+1).head(lookback).median()
            last=g["units"].iloc[-1]
            if pd.notna(last) and last < frac*med:
                g=g.iloc[:-1].copy(); drops+=1
            else:
                break
        out.append(g)
    return pd.concat(out, ignore_index=True)

history = trim_tail(wk_full).copy()

# ensure week_idx exists (some features may depend on it)
if "week_idx" not in history.columns:
    history["week_idx"] = history.groupby("product_category_name_english").cumcount()


# base price plan = last-4-week mean per category

recent_price = (history.groupby("product_category_name_english")
                .agg(base_price=("avg_price", lambda s: s.tail(4).mean()))
                .reset_index())

# horizon
H=8
last_wk=history["order_week"].max()
future_weeks=[last_wk + pd.to_timedelta(7*(i+1), unit="D") for i in range(H)]


# forecaster under a price multiplier

LAGS=[1,2,4,8,12,52]

def run_forecast(multiplier: float) -> pd.DataFrame:
    """Roll forward H weeks with price = base_price*(1+multiplier)."""
    hist = history.copy()
    out  = []

    # scenario plan (flat price by cat across horizon)
    plan = pd.MultiIndex.from_product([future_weeks, recent_price["product_category_name_english"]],
            names=["order_week","product_category_name_english"]).to_frame(index=False) \
            .merge(recent_price, on="product_category_name_english", how="left")
    plan["avg_price"] = plan["base_price"] * (1.0 + multiplier)

    for wk_i in future_weeks:
        step = plan.loc[plan["order_week"]==wk_i,
                        ["order_week","product_category_name_english","avg_price"]].copy()
        step["units"]=np.nan; step["revenue"]=np.nan

        # extend running week_idx per category
        max_idx = hist.groupby("product_category_name_english")["week_idx"].max().reset_index() \
                      .rename(columns={"week_idx":"_last_idx"})
        step = step.merge(max_idx, on="product_category_name_english", how="left")
        step["week_idx"] = step["_last_idx"] + 1
        step.drop(columns=["_last_idx"], inplace=True)

        tmp = pd.concat([hist, step], ignore_index=True, sort=False)

        # recompute lags/rollings matching training
        g = tmp.groupby("product_category_name_english", sort=False)
        for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
            if col not in tmp.columns: tmp[col]=np.nan
            for L in LAGS:
                tmp[f"{col}_lag{L}"] = g[col].shift(L)
        tmp["units_roll4"]   = g["units"].transform(lambda s: s.rolling(4, min_periods=1).mean())
        tmp["revenue_roll4"] = g["revenue"].transform(lambda s: s.rolling(4, min_periods=1).mean())

        # predict
        X = tmp.loc[tmp["order_week"]==wk_i, FEAT]
        yhat = pipe.predict(X)  # base units

        pred = tmp.loc[tmp["order_week"]==wk_i,
                       ["order_week","product_category_name_english","avg_price","week_idx"]].copy()
        pred["units_base"] = yhat
        pred["revenue_base"] = pred["avg_price"] * pred["units_base"]

        # optional calibration
        if calib is not None and {"a","b"}.issubset(calib.columns):
            pred = pred.merge(calib[["product_category_name_english","a","b"]],
                              on="product_category_name_english", how="left")
            pred["units_cal"] = pred["a"] + pred["b"] * pred["units_base"]
            pred.drop(columns=["a","b"], inplace=True)
        else:
            pred["units_cal"] = pred["units_base"]

        pred["revenue_cal"] = pred["avg_price"] * pred["units_cal"]
        pred["scenario"] = multiplier

        # append back to history for next-step lags (use BASE units for recursion)
        to_hist = pred.rename(columns={"units_base":"units",
                                       "revenue_base":"revenue"})
        hist = pd.concat([hist, to_hist[["order_week","product_category_name_english",
                                         "avg_price","units","revenue","week_idx"]]],
                         ignore_index=True, sort=False)
        out.append(pred.drop(columns=["week_idx"]))

    return pd.concat(out, ignore_index=True)


# scenarios and execution

SCENARIOS = [-0.15,-0.10,-0.05,0.0,0.05,0.10,0.15]
all_runs = []
for s in SCENARIOS:
    print(f"Running scenario {s:+.0%} ...")
    all_runs.append(run_forecast(s))
scen_df = pd.concat(all_runs, ignore_index=True)

# save the detailed per-week outputs
scen_df.to_csv(OUT/"elasticity_scenarios_detailed.csv", index=False)


# aggregate by category for Δ vs baseline (0.0)

base = scen_df[scen_df["scenario"]==0.0] \
        .groupby("product_category_name_english", as_index=False) \
        .agg(units0=("units_cal","sum"),
             revenue0=("revenue_cal","sum"),
             price0=("avg_price","mean"))

rows=[]
for s in [x for x in SCENARIOS if x!=0.0]:
    agg = scen_df[scen_df["scenario"]==s] \
            .groupby("product_category_name_english", as_index=False) \
            .agg(units_s=("units_cal","sum"),
                 revenue_s=("revenue_cal","sum"),
                 price_s=("avg_price","mean"))
    m = agg.merge(base, on="product_category_name_english", how="left")
    m["scenario"] = s
    m["delta_units"]   = m["units_s"]   - m["units0"]
    m["delta_revenue"] = m["revenue_s"] - m["revenue0"]
    # simple elasticity: %ΔQ / %ΔP using baseline means
    m["pct_dQ"] = np.where(m["units0"]>0, m["delta_units"]/m["units0"], np.nan)
    m["pct_dP"] = np.where(m["price0"]>0, (m["price_s"]-m["price0"])/m["price0"], np.nan)
    m["elasticity"] = m["pct_dQ"] / m["pct_dP"]
    rows.append(m)

delta_cat = pd.concat(rows, ignore_index=True)
delta_cat.to_csv(OUT/"elasticity_deltas_by_category.csv", index=False)
print("Saved:",
      OUT/"elasticity_scenarios_detailed.csv",
      OUT/"elasticity_deltas_by_category.csv")


# plotting helpers

def barplot_delta(df, value_col, title_prefix, fname_prefix, top_n=15):
    """Make a bar chart per scenario, ordered by absolute impact, top N categories."""
    for s, g in df.groupby("scenario"):
        g = g.sort_values(value_col, key=lambda x: x.abs(), ascending=False).head(top_n)
        plt.figure(figsize=(12,6))
        plt.barh(g["product_category_name_english"], g[value_col])
        plt.gca().invert_yaxis()
        plt.xlabel("Δ " + ("Units" if "units" in value_col else "Revenue"))
        plt.title(f"{title_prefix} {s:+.0%}")
        plt.tight_layout()
        png = OUT/f"{fname_prefix}_{int(round(s*100)):+d}.png"
        plt.savefig(png, dpi=150)
        plt.close()
        print("Saved plot:", png)

# ΔUnits and ΔRevenue (per scenario)
barplot_delta(delta_cat, "delta_units",
              "ΔUnits by category — price change",
              "elasticity_delta_units")

barplot_delta(delta_cat, "delta_revenue",
              "ΔRevenue by category — price change",
              "elasticity_delta_revenue")

# Optional: a compact “elasticity leaderboard” (most sensitive categories)
sens = (delta_cat.groupby("product_category_name_english", as_index=False)
        .agg(mean_abs_elasticity=("elasticity", lambda s: np.nanmean(np.abs(s)))))
sens = sens.sort_values("mean_abs_elasticity", ascending=False).head(20)
plt.figure(figsize=(10,6))
plt.barh(sens["product_category_name_english"], sens["mean_abs_elasticity"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |Elasticity| across scenarios")
plt.title("Most price-sensitive categories")
plt.tight_layout()
plt.savefig(OUT/"elasticity_top_sensitive.png", dpi=150)
plt.close()
print("Saved plot:", OUT/"elasticity_top_sensitive.png")


Y]--------> ## # # elasticity_recommendations_profit.py ##

import pandas as pd, numpy as np
from pathlib import Path

OUT = Path("Dataset/model_outputs")

# ---- inputs ----
rec = pd.read_csv(OUT/"elasticity_recommendations_by_category.csv")
cols = {c.lower(): c for c in rec.columns}         # case-insensitive map
rec.columns = [c.lower() for c in rec.columns]

# Costs (real or proxy created earlier)
if (OUT/"cost_per_category_proxy.csv").exists():
    cost_df = pd.read_csv(OUT/"cost_per_category_proxy.csv")
    cost_df.columns = [c.lower() for c in cost_df.columns]
else:
    cost_df = pd.read_csv("Dataset/cost_per_category.csv")
    cost_df.columns = [c.lower() for c in cost_df.columns]

# Deltas by scenario (gives us baseline units/price and scenario units/price if needed)
delta = pd.read_csv(OUT/"elasticity_deltas_by_category.csv")
delta.columns = [c.lower() for c in delta.columns]

# ---- normalise column names in rec ----
cat_col = "product_category_name_english"
if cat_col not in rec.columns:
    raise RuntimeError("Recommendations file is missing 'product_category_name_english'.")

# baseline fields (any of these are ok)
price0_col = next((c for c in ["price0","base_price","baseline_price"] if c in rec.columns), None)
units0_col = next((c for c in ["units0","base_units","units_base"] if c in rec.columns), None)

# chosen scenario (percentage change)
scenario_col = next((c for c in ["scenario_best","best_scenario","rec_scenario","scenario","rec_pct","pct_price_change","pct_change"]
                     if c in rec.columns), None)

# recommended price/units (if they already exist)
price_best_col = next((c for c in ["price_best","price_rec","price_recommended","price_s"] if c in rec.columns), None)
units_best_col = next((c for c in ["units_best","units_rec","units_recommended","units_s"] if c in rec.columns), None)

# ---- if baseline fields missing in 'rec', pull them from delta (scenario == 0) ----
base_from_delta = (delta[delta["scenario"]==0.0]
                   [[cat_col,"units0","price0"]].drop_duplicates(cat_col))
if price0_col is None:
    rec = rec.merge(base_from_delta[[cat_col,"price0"]], on=cat_col, how="left")
    price0_col = "price0"
if units0_col is None:
    rec = rec.merge(base_from_delta[[cat_col,"units0"]], on=cat_col, how="left")
    units0_col = "units0"

# ---- if recommended price/units missing, pull from delta using the chosen scenario ----
if (price_best_col is None or units_best_col is None):
    if scenario_col is None:
        raise RuntimeError("No recommended scenario or recommended price/units found in the file.")
    # normalise scenario type to float
    rec["__scenario__"] = rec[scenario_col].astype(float)
    d_pick = delta[[cat_col,"scenario","price_s","units_s"]].copy()
    d_pick.rename(columns={"scenario":"__scenario__"}, inplace=True)
    rec = rec.merge(d_pick, on=[cat_col,"__scenario__"], how="left")
    price_best_col = "price_s"
    units_best_col = "units_s"

# ---- merge unit cost ----
unit_cost_col = "unit_cost"
if unit_cost_col not in cost_df.columns:
    raise RuntimeError("Cost file must contain a 'unit_cost' column.")
rec = rec.merge(cost_df[[cat_col, unit_cost_col]], on=cat_col, how="left")

# sanity checks
for need in [price0_col, units0_col, price_best_col, units_best_col]:
    if need not in rec.columns:
        raise RuntimeError(f"Still missing a required column after normalization: {need}")

miss = rec[unit_cost_col].isna().sum()
if miss:
    print(f"Warning: {miss} categories missing unit_cost; dropping them for profit calc.")
    rec = rec.dropna(subset=[unit_cost_col])

# ---- compute profit lift ----
rec["profit0"]     = (rec[price0_col]     - rec[unit_cost_col]) * rec[units0_col]
rec["profit_best"] = (rec[price_best_col] - rec[unit_cost_col]) * rec[units_best_col]
rec["profit_lift"] = rec["profit_best"] - rec["profit0"]

out_path = OUT/"elasticity_recommendations_profit.csv"
rec.sort_values("profit_lift", ascending=False).to_csv(out_path, index=False)
print("Saved profit recommendations to:", out_path)


Z]--------> ## # # price_move_recommendations.py ##


# --- Price move recommendations: dashboards & deliverable --------------------
# Produces:
#   Dataset/model_outputs/profit_reco_top_lifts.png
#   Dataset/model_outputs/profit_reco_biggest_risks.png
#   Dataset/model_outputs/profit_reco_safe_top.png
#   Dataset/model_outputs/price_recommendations_summary.xlsx

import matplotlib
matplotlib.use("Agg")  # avoid any Tk/Tkinter GUI warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)


# 1) Load recommendations (already includes profit_best / profit_lift)

rec = pd.read_csv(OUT / "elasticity_recommendations_profit.csv")
rec.columns = [c.lower() for c in rec.columns]
CAT = "product_category_name_english"

# Helper: pick first column that exists
def pick(df, *names, required=True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"None of the columns {names} found.")
    return None

# Standardize “new scenario” column names (handle slight naming differences)
col_price_new = pick(rec, "price_best", "price_s", "price_new", required=False)
col_units_new = pick(rec, "units_best", "units_s", required=False)

# If your file didn’t keep price_best/units_best, try reconstructing from deltas if present
if col_price_new is None and {"price0", "price_change_pct"}.issubset(rec.columns):
    rec["price_best"] = rec["price0"] * (1 + rec["price_change_pct"])
    col_price_new = "price_best"
if col_units_new is None and {"units0", "units_change_pct"}.issubset(rec.columns):
    rec["units_best"] = rec["units0"] * (1 + rec["units_change_pct"])
    col_units_new = "units_best"

# Derive extra, easy-to-read metrics
if "price0" in rec.columns and col_price_new:
    rec["price_change_pct"] = (rec[col_price_new] - rec["price0"]) / rec["price0"]

if "units0" in rec.columns and col_units_new:
    rec["units_change_pct"] = np.where(
        rec["units0"] > 0,
        (rec[col_units_new] - rec["units0"]) / rec["units0"],
        np.nan,
    )

# Profit percent change + margins
if {"profit0", "profit_best"}.issubset(rec.columns):
    rec["profit_change_pct"] = np.where(
        rec["profit0"] != 0, rec["profit_lift"] / rec["profit0"], np.nan
    )
    rec["margin0"] = np.where(rec["units0"] > 0, rec["profit0"] / rec["units0"], np.nan)
    rec["margin_best"] = np.where(
        rec[col_units_new] > 0, rec["profit_best"] / rec[col_units_new], np.nan
    )


# 2) Plot top profit lifts & biggest risks

def barh_top(df, value_col, title, fname, top_n=15):
    g = df.sort_values(value_col, ascending=False).head(top_n)
    plt.figure(figsize=(14, 6))
    plt.barh(g[CAT], g[value_col])
    plt.gca().invert_yaxis()
    plt.xlabel(value_col.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    png = OUT / f"{fname}.png"
    plt.savefig(png, dpi=150)
    plt.close()
    print("Saved:", png)

# Top profit lifts (positive)
barh_top(
    rec,
    "profit_lift",
    "Recommended price moves — top PROFIT lifts (8 weeks)",
    "profit_reco_top_lifts",
)

# Biggest profit risks (most negative; plot absolute drop for readability)
tmp_risk = rec.copy()
tmp_risk["profit_drop_abs"] = rec["profit_lift"].clip(upper=0).abs()
barh_top(
    tmp_risk,
    "profit_drop_abs",
    "Recommended price moves — biggest PROFIT risks (8 weeks; absolute drop)",
    "profit_reco_biggest_risks",
)


# 3) Guardrails: show only “safe” wins (units not down >10% & margin per unit not worse)

safe = rec.copy()
if "units_change_pct" in safe.columns and {"margin_best", "margin0"}.issubset(safe.columns):
    safe = safe[
        (safe["profit_lift"] > 0)
        & (safe["units_change_pct"] >= -0.10)        # don’t lose >10% units
        & (safe["margin_best"] >= safe["margin0"])   # per-unit margin not worse
    ]
else:
    # Fallback: no guardrail metrics available → only positive profit lifts
    safe = safe[safe["profit_lift"] > 0]

barh_top(
    safe,
    "profit_lift",
    "TOP safe profit lifts (units & margin guardrails applied)",
    "profit_reco_safe_top",
)



# 4) One tidy Excel deliverable with three tabs  (keyword-only safe)

xlsx_path = OUT / "price_recommendations_summary.xlsx"

# Use a safe engine selection
try:
    writer = pd.ExcelWriter(xlsx_path, engine="xlsxwriter")
except Exception:
    writer = pd.ExcelWriter(xlsx_path, engine="openpyxl")

rec.sort_values("profit_lift", ascending=False).to_excel(
    writer, sheet_name="All_Recommendations", index=False
)
safe.sort_values("profit_lift", ascending=False).to_excel(
    writer, sheet_name="Safe_Top", index=False
)
rec.nsmallest(20, "profit_lift").to_excel(
    writer, sheet_name="Biggest_Risks", index=False
)

writer.close()
print("Saved:", xlsx_path)


AA]--------> ## # # one_pagers.py ##

# Category one-pagers: baseline vs proposed price (warning-free)
# - Uses Safe_Top sheet from price_recommendations_summary.xlsx
# - Plots last ~16w history, 8w baseline forecast (scenario=0.0), 8w proposed scenario
# - Outputs per-category PNGs + a combined PDF

import warnings
warnings.filterwarnings(
    "ignore",
    message="The behaviour of DatetimeProperties.to_pydatetime is deprecated"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import re

# --------------------------
# Paths & inputs
# --------------------------
BASE = Path("Dataset/model_outputs")
BASE.mkdir(parents=True, exist_ok=True)

# Recommendations (must have product_category_name_english and a scenario column)
safe = pd.read_excel(BASE / "price_recommendations_summary.xlsx", sheet_name="Safe_Top")

# Scenario runs (must have order_week, product_category_name_english, scenario, avg_price, units_cal, revenue_cal)
scen = pd.read_csv(BASE / "elasticity_scenarios_detailed.csv", parse_dates=["order_week"])

# Historical weekly (must have order_week, product_category_name_english, units)
history = (
    pd.read_csv(BASE / "weekly_by_category_full.csv", parse_dates=["order_week"])
      .sort_values(["product_category_name_english", "order_week"])
)

# --------------------------
# Helpers
# --------------------------
def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(s).strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "category"

def find_first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def plot_with_dates(ax, x_series, y_series, **kwargs):
    """
    Convert x to datetime, drop rows with NaT or NaN y, and plot.
    Uses numpy array of Python datetimes to avoid FutureWarning.
    """
    x = pd.to_datetime(x_series, errors="coerce")
    y = pd.to_numeric(y_series, errors="coerce")
    mask = x.notna() & pd.notna(y)
    if not mask.any():
        return 0
    x_py = np.array(x[mask].dt.to_pydatetime())  # future-proofed by converting to np.array
    ax.plot(x_py, y[mask].to_numpy(), **kwargs)
    return int(mask.sum())

# Detect scenario column name used in Safe_Top
SCEN_COL = find_first_col(safe, ["scenario", "scenario_best", "s_best"])
if SCEN_COL is None:
    raise ValueError("No scenario column in Safe_Top. Expected one of: scenario, scenario_best, s_best")
safe[SCEN_COL] = pd.to_numeric(safe[SCEN_COL], errors="coerce")

TOP_N = min(20, len(safe))
safe_sel = safe.head(TOP_N).copy()

# Outputs
onepager_dir = BASE / "onepagers"
onepager_dir.mkdir(exist_ok=True)
pdf_path = BASE / "category_onepagers.pdf"
pdf = PdfPages(pdf_path)
png_paths = []

# Date formatting (robust)
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

# --------------------------
# Build one-pagers
# --------------------------
for _, row in safe_sel.iterrows():
    cat = row["product_category_name_english"]
    s_best = float(row[SCEN_COL]) if pd.notna(row[SCEN_COL]) else 0.0

    # History (last 16 weeks)
    h_last = (
        history[history["product_category_name_english"] == cat]
        .sort_values("order_week")
        .tail(16)
        .copy()
    )

    # Baseline scenario
    base = (
        scen[(scen["product_category_name_english"] == cat) & (scen["scenario"] == 0.0)]
        .copy()
        .sort_values("order_week")
    )

    # Proposed scenario: exact, else nearest available for the category
    prop = (
        scen[(scen["product_category_name_english"] == cat) & (np.isclose(scen["scenario"], s_best))]
        .copy()
        .sort_values("order_week")
    )
    s_used = s_best
    if prop.empty:
        avail = (
            scen[scen["product_category_name_english"] == cat]
            .groupby("scenario", as_index=False)
            .size()[["scenario"]]
        )
        if not avail.empty:
            s_used = float(avail.iloc[(avail["scenario"] - s_best).abs().argsort()].iloc[0]["scenario"])
            prop = (
                scen[(scen["product_category_name_english"] == cat) & (np.isclose(scen["scenario"], s_used))]
                .copy()
                .sort_values("order_week")
            )

    if h_last.empty or base.empty or prop.empty:
        continue

    # Metrics
    price0 = pd.to_numeric(base["avg_price"], errors="coerce").mean()
    price1 = pd.to_numeric(prop["avg_price"], errors="coerce").mean()
    dprice = (price1 - price0) / price0 if price0 else np.nan

    units0 = pd.to_numeric(base["units_cal"], errors="coerce").sum()
    units1 = pd.to_numeric(prop["units_cal"], errors="coerce").sum()
    dunits = units1 - units0

    rev0 = pd.to_numeric(base["revenue_cal"], errors="coerce").sum()
    rev1 = pd.to_numeric(prop["revenue_cal"], errors="coerce").sum()
    drev = rev1 - rev0

    # Profit (optional; may be missing)
    prof0 = pd.to_numeric(row.get("profit0", np.nan), errors="coerce")
    prof1 = pd.to_numeric(row.get("profit_best", np.nan), errors="coerce")
    dprof = pd.to_numeric(row.get("profit_lift", np.nan), errors="coerce")

    # ---- Plot ----
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    n1 = plot_with_dates(ax, h_last["order_week"], h_last["units"],
                         label="History (units)", linewidth=1)
    n2 = plot_with_dates(ax, base["order_week"], base["units_cal"],
                         linestyle="--", label="Baseline forecast (units)")
    n3 = plot_with_dates(ax, prop["order_week"], prop["units_cal"],
                         linestyle="-", label=f"Proposed {s_used:+.0%} price (units)")

    if (n1 + n2 + n3) == 0:
        plt.close(fig)
        continue

    ax.set_title(f"{cat} — baseline vs proposed")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    ax.legend(loc="upper right")

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    lines = [
        f"Proposed price change: {s_used:+.0%}",
        "",
        f"Avg price — base: {price0:,.2f}  |  proposed: {price1:,.2f}  ({dprice:+.1%})",
        f"8w units — base: {units0:,.0f}  |  proposed: {units1:,.0f}  (Δ {dunits:+,.0f})",
        f"8w revenue — base: {rev0:,.0f}  |  proposed: {rev1:,.0f}  (Δ {drev:+,.0f})",
    ]
    if pd.notna(prof0) and pd.notna(prof1):
        lines.append(f"8w profit — base: {prof0:,.0f}  |  proposed: {prof1:,.0f}  (Δ {dprof:+,.0f})")

    ax.text(
        1.02, 0.95,
        "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.05)
    )

    fig.tight_layout()

    # Save PNG & add to PDF
    fpath = onepager_dir / f"onepager_{slugify(cat)}.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    png_paths.append(str(fpath))

pdf.close()

print(f"Saved {len(png_paths)} one-pagers")
print("Sample:", png_paths[:5])
print("Combined PDF:", pdf_path)


AB --------> ## # # stakeholder_slide_deck.py ##

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from datetime import date

BASE = Path("Dataset/model_outputs")
png_dir = BASE / "onepagers"
pngs = sorted(png_dir.glob("onepager_*.png"))
out_ppt = BASE / "category_onepagers.pptx"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)  # 16:9

# Cover
slide = prs.slides.add_slide(prs.slide_layouts[6])
tx = slide.shapes.add_textbox(Inches(0.8), Inches(1.2), Inches(11.7), Inches(1.6)).text_frame
tx.text = "Category One-Pagers"
tx.paragraphs[0].font.size = Pt(48)
sub = slide.shapes.add_textbox(Inches(0.8), Inches(2.4), Inches(11.7), Inches(1.6)).text_frame
sub.text = f"Baseline vs proposed price — {date.today():%d %b %Y}"

# Pages
for img in pngs:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_picture(str(img), Inches(0.25), Inches(0.25),
                             width=Inches(12.83), height=Inches(7.0))
print("Saved:", out_ppt)



```

## Project Structure

SERAH-Compass/
├─ Dataset/
│  ├─ audit_exports
│  ├─ ├─ dataset_overview.csv
│  ├─ ├─ missingness_summary.csv
│  ├─ ├─ order_status_counts.csv
│  ├─ clean_exports
│  ├─ ├─ cleaning_log.csv
│  ├─ ├─ customers_clean.csv
│  ├─ ├─ fact_orders_clean.csv
│  ├─ ├─ payments_by_order.csv
│  ├─ ├─ products_clean.csv
│  ├─ ├─ reviews_by_order.csv
│  ├─ ├─ sellers_clean.csv
│  ├─ model_outputs
│  ├─ cost_per_category.csv
│  ├─ olist_customers_dataset.csv
│  ├─ olist_geolocation_dataset.csv
│  ├─ olist_order_items_dataset.csv
│  ├─ olist_order_payments_dataset.csv
│  ├─ olist_order_reviews_dataset.csv
│  ├─ olist_orders_dataset.csv
│  ├─ olist_products_dataset.csv
│  ├─ olist_sellers_dataset.csv
│  ├─ product_category_name_translation.csv
├─ Scripts/
│  ├─ data_cleaning.py
│  ├─ build_weekly_features.py
│  ├─ train_forecast.py
│  ├─ simulate_price_change.py
│  ├─ build_and_train.py
│  ├─ make_margins_template.py
│  ├─ recommend_actions.py
│  ├─ select_actions_with_budget.py
│  ├─ make_price_plan_and_forecast.py
│  ├─ forecast_next_weeks.py
│  ├─ compute_plan_impact.py
│  ├─ plot_forecast.py
│  ├─ forecast_trimmed.py
│  ├─ build_and_train_no_leak.py
│  ├─ forecast_no_leak.py
│  ├─ plot_validation_accuracy.py
│  ├─ evaluate_validation.py
│  ├─ plot_validation_overlay.py
│  ├─ build_validation_actions.py
│  ├─ apply_calibration_to_forecast.py
│  ├─ unify_forecast_and_rollup.py
│  ├─ qa_and_readout.py
│  ├─ calibration_scatter.py
│  ├─ elasticity_explorer.py
│  ├─ elasticity_recommendations_profit.py
│  ├─ price_move_recommendations.py
│  ├─ one_pagers.py
│  ├─ stakeholder_slide_deck.py
├─ README.md
├─ LICENSE



## License

MIT License

Copyright (c) 2025 rabraham2

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


