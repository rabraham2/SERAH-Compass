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
