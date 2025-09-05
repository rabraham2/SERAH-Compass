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
