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

# ---- normalize column names in rec ----
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
    # normalize scenario type to float
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
