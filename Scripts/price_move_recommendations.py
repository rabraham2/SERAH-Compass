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
