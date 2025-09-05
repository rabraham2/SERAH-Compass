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
