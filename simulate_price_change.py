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
