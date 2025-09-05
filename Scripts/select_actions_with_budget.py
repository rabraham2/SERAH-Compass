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
