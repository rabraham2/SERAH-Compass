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

# vectorized time split
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
