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
# 3) Add lags — vectorized (no groupby.apply, no warnings)
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
# 4) Vectorized robust time split per category (never empty)
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
