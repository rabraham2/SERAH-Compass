# Weekly Features Building
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("Dataset/clean_exports")
OUT  = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Load
# 'fact_orders_clean.csv' may store dates as dd-mm-YYYY; parse robustly
fact = pd.read_csv(BASE / "fact_orders_clean.csv")
fact["order_date"] = pd.to_datetime(fact["order_date"], dayfirst=True, errors="coerce")

# Reviews aggregation (ensure presence & aligned dtypes)
rev_path = BASE / "reviews_by_order.csv"
if rev_path.exists():
    rev = pd.read_csv(rev_path)
else:
    # fallback to raw if needed (optional)
    raw = pd.read_csv(
        Path("Dataset") / "olist_order_reviews_dataset.csv",
        parse_dates=["review_creation_date", "review_answer_timestamp"]
    )
    rev = (raw.groupby("order_id", as_index=False)
              .agg(review_count=("review_id", "count"),
                   review_score_avg=("review_score", "mean")))

# Align merge keys as string
fact["order_id"] = fact["order_id"].astype(str)
rev["order_id"]  = rev["order_id"].astype(str)

# Safe merge; guarantee columns
fact = fact.merge(rev[["order_id", "review_count", "review_score_avg"]],
                  on="order_id", how="left")
if "review_count" not in fact.columns:
    fact["review_count"] = 0
if "review_score_avg" not in fact.columns:
    fact["review_score_avg"] = np.nan

# Fill sensible defaults
fact["review_count"] = fact["review_count"].fillna(0)

# Weekly aggregation
# Monday-start week (stable weekly grain)
fact["order_week"] = fact["order_date"] - pd.to_timedelta(fact["order_date"].dt.weekday, unit="D")

# Defensive fill for category
fact["product_category_name_english"] = fact["product_category_name_english"].fillna("unknown")

wk = (fact.groupby(["order_week", "product_category_name_english"], as_index=False)
          .agg(
              units = ("units", "sum"),
              revenue = ("revenue_item", "sum"),
              avg_price = ("revenue_item", "mean"),
              rev_count = ("review_count", "sum"),
              rev_score_avg = ("review_score_avg", "mean")
          ))

# Fill review score gaps with a rolling mean per category
# Use group_keys=False so index stays aligned with wk
wk = wk.sort_values(["product_category_name_english", "order_week"]).reset_index(drop=True)
wk["rev_score_avg"] = (wk.groupby("product_category_name_english", group_keys=False)["rev_score_avg"]
                         .apply(lambda s: s.fillna(s.rolling(8, min_periods=1).mean())))

# Calendar & seasonal features
wk["weekofyear"] = wk["order_week"].dt.isocalendar().week.astype(int)
wk["year"]       = wk["order_week"].dt.year
wk["sin_woy"]    = np.sin(2*np.pi*wk["weekofyear"]/52.0)
wk["cos_woy"]    = np.cos(2*np.pi*wk["weekofyear"]/52.0)

# Optional: Brazil holiday flag (safe fallback if holidays not installed)
try:
    from holidays import Brazil
    years = range(int(wk["year"].min()), int(wk["year"].max()) + 1)
    br_dates = set(pd.to_datetime(list(Brazil(years=years))).date)
    wk["is_holiday_week"] = wk["order_week"].dt.date.isin(br_dates).astype(int)
except Exception:
    wk["is_holiday_week"] = 0

# Lags & rolling stats
def add_lags(df: pd.DataFrame, col: str, lags=(1, 2, 4, 8, 12, 52)) -> pd.DataFrame:
    for L in lags:
        df[f"{col}_lag{L}"] = (
            df.groupby("product_category_name_english", group_keys=False)[col]
              .shift(L)
        )
    return df

for col in ["units", "revenue", "avg_price", "rev_count", "rev_score_avg"]:
    wk = add_lags(wk, col)

# Helpful short rolling means
wk["units_roll4"]   = (wk.groupby("product_category_name_english", group_keys=False)["units"]
                         .transform(lambda s: s.rolling(4, min_periods=1).mean()))
wk["revenue_roll4"] = (wk.groupby("product_category_name_english", group_keys=False)["revenue"]
                         .transform(lambda s: s.rolling(4, min_periods=1).mean()))

# Save full features
wk_full_path = OUT / "weekly_by_category_full.csv"
wk.to_csv(wk_full_path, index=False)

# Build model-ready view: drop warm-up rows missing any lag
lag_cols = [c for c in wk.columns if "lag" in c]
wk_model = wk.dropna(subset=lag_cols).copy()

# Time-based split flag: last 12 weeks per category -> validation
def flag_split(g: pd.DataFrame, val_weeks=12) -> pd.DataFrame:
    g = g.sort_values("order_week").copy()
    g["split"] = "train"
    if len(g) > val_weeks:
        g.loc[g.index[-val_weeks:], "split"] = "valid"
    return g

# Vectorized time-based split: last 12 rows per category → "valid"
wk_model = wk_model.sort_values(["product_category_name_english","order_week"]).copy()
g = wk_model.groupby("product_category_name_english")
n = g["order_week"].transform("size")
idx_in_group = g.cumcount()  # 0..n-1
wk_model["split"] = np.where(idx_in_group >= (n - 12), "valid", "train")

wk_model_path = OUT / "weekly_by_category.csv"
wk_model.to_csv(wk_model_path, index=False)

print("Saved:", wk_full_path)
print("Saved (model-ready):", wk_model_path)
