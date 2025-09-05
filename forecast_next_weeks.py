# Forecast next weeks
# forecast_next_weeks.py
import pandas as pd, numpy as np
from pathlib import Path
import joblib

OUT = Path("Dataset/model_outputs")
wk_full  = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"]) \
              .sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)
wk_model = pd.read_csv(OUT/"weekly_by_category.csv", parse_dates=["order_week"])

pipe = joblib.load(OUT/"weekly_units_xgb.pkl")

# --- exact features used at training ---
TARGET = "units"
cat_cols = ["product_category_name_english"]
drop_cols = {"order_week","split",TARGET}
num_cols = [c for c in wk_model.columns if c not in set(cat_cols) | drop_cols]
FEAT_COLS = cat_cols + num_cols

# Optional plan CSV (order_week, product_category_name_english, planned_price)
PRICE_PLAN = None  # e.g., Path("Dataset/model_outputs/price_plan.csv")

H = 8
cats = wk_full["product_category_name_english"].unique()
def next_monday(d): return d + pd.to_timedelta(7, unit="D")

last_week = wk_full["order_week"].max()
future_weeks = [next_monday(last_week) + pd.to_timedelta(7*i, unit="D") for i in range(H)]
future = pd.MultiIndex.from_product([future_weeks, cats],
                                    names=["order_week","product_category_name_english"]).to_frame(index=False)

# recent averages to carry forward
recent = (wk_full.groupby("product_category_name_english")
          .agg(avg_price_recent=("avg_price",       lambda s: s.tail(4).mean()),
               rev_count_recent=("rev_count",       lambda s: s.tail(4).mean()),
               rev_score_recent=("rev_score_avg",   lambda s: s.tail(4).mean()))
          .reset_index())

future = future.merge(recent, on="product_category_name_english", how="left")

# ensure avg_price column exists even if no plan is provided
future["avg_price"] = np.nan

# merge plan if provided
if PRICE_PLAN:
    plan = pd.read_csv(PRICE_PLAN, parse_dates=["order_week"])
    future = future.merge(
        plan.rename(columns={"planned_price":"avg_price"}),
        on=["order_week","product_category_name_english"],
        how="left",
        suffixes=("","_plan")
    )
    # if both columns exist due to suffix, prefer plan
    if "avg_price_plan" in future.columns:
        future["avg_price"] = future["avg_price_plan"].combine_first(future["avg_price"])
        future.drop(columns=["avg_price_plan"], inplace=True)

history = wk_full.copy()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_category_name_english","order_week"]).copy()
    g  = df.groupby("product_category_name_english", sort=False)

    # price for future rows: plan if given, else recent avg
    if "avg_price" not in df.columns:
        df["avg_price"] = np.nan
    if "avg_price_recent" not in df.columns:
        # compute fallback from existing history if needed
        tmp_recent = (df.groupby("product_category_name_english")
                        .agg(avg_price_recent=("avg_price", lambda s: s.tail(4).mean()))
                        .reset_index())
        df = df.merge(tmp_recent, on="product_category_name_english", how="left")
    df["avg_price"] = df["avg_price"].fillna(df["avg_price_recent"])

    # reviews carry-forward
    for col, recent_col in [("rev_count","rev_count_recent"),
                            ("rev_score_avg","rev_score_recent")]:
        if col not in df.columns:
            df[col] = np.nan
        if recent_col not in df.columns:
            tmp_recent = (df.groupby("product_category_name_english")
                            .agg(**{recent_col:(col, lambda s: s.tail(4).mean())})
                            .reset_index())
            df = df.merge(tmp_recent, on="product_category_name_english", how="left")
        df[col] = df[col].fillna(df[recent_col])

    # calendar
    df["weekofyear"] = df["order_week"].dt.isocalendar().week.astype(int)
    df["year"]       = df["order_week"].dt.year
    df["sin_woy"]    = np.sin(2*np.pi*df["weekofyear"]/52.0)
    df["cos_woy"]    = np.cos(2*np.pi*df["weekofyear"]/52.0)
    if "is_holiday_week" not in df.columns:
        df["is_holiday_week"] = 0

    # vectorized lags matching training
    LAGS = [1,2,4,8,12,52]
    for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
        for L in LAGS:
            df[f"{col}_lag{L}"] = g[col].shift(L)

    # rolling stats used in training
    df["units_roll4"]   = g["units"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["revenue_roll4"] = g["revenue"].transform(lambda s: s.rolling(4, min_periods=1).mean())

    return df

forecasts = []
for wk_i in future_weeks:
    # build rows for this step (now avg_price column is guaranteed to exist)
    step = future[future["order_week"]==wk_i][[
        "order_week","product_category_name_english",
        "avg_price","avg_price_recent","rev_count_recent","rev_score_recent"
    ]].copy()

    # placeholders to match history columns
    step["units"] = np.nan
    step["revenue"] = np.nan
    step["rev_count"] = np.nan
    step["rev_score_avg"] = np.nan

    # compute features (incl. lags/rolling)
    tmp = pd.concat([history, step], ignore_index=True)
    tmp = compute_features(tmp)

    # supply proxy revenue for CURRENT future week to satisfy pipeline
    mask = tmp["order_week"].eq(wk_i)
    units_proxy = tmp.loc[mask, "units_lag1"].fillna(tmp.loc[mask, "units_roll4"]).fillna(0.0)
    tmp.loc[mask, "revenue"] = tmp.loc[mask, "avg_price"] * units_proxy

    # ensure every expected feature exists
    for c in FEAT_COLS:
        if c not in tmp.columns:
            tmp[c] = 0.0
    if "product_category_name_english" not in tmp.columns:
        tmp["product_category_name_english"] = history["product_category_name_english"]

    X = tmp.loc[mask, FEAT_COLS]
    yhat = pipe.predict(X)

    pred = tmp.loc[mask, ["order_week","product_category_name_english","avg_price"]].copy()
    pred["units"]   = yhat
    pred["revenue"] = pred["avg_price"] * pred["units"]

    # carry forward recents for next iterations
    pred = pred.merge(
        step[["product_category_name_english","avg_price_recent","rev_count_recent","rev_score_recent"]],
        on="product_category_name_english", how="left"
    )

    # append to history
    history = pd.concat([history, pred], ignore_index=True, sort=False)
    forecasts.append(pred[["order_week","product_category_name_english","avg_price","units","revenue"]])

fcast = pd.concat(forecasts, ignore_index=True)
out_path = OUT/"forecasts_next_weeks.csv"
fcast.to_csv(out_path, index=False)
print("Saved:", out_path)
