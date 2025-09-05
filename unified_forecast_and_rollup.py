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

# Standardize base columns
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
    # normalize column name if needed
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
