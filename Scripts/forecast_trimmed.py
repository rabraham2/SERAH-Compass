import pandas as pd, numpy as np
from pathlib import Path
import joblib

OUT = Path("Dataset/model_outputs")
wk_full  = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"]) \
              .sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)
wk_model = pd.read_csv(OUT/"weekly_by_category.csv", parse_dates=["order_week"])
pipe = joblib.load(OUT/"weekly_units_xgb.pkl")

# exact features used at training
TARGET="units"; cat_cols=["product_category_name_english"]
drop_cols={"order_week","split",TARGET}
num_cols=[c for c in wk_model.columns if c not in set(cat_cols)|drop_cols]
FEAT_COLS = cat_cols + num_cols

# -------- trim suspicious trailing weeks (very low vs recent median) ----------
def trim_tail(df, lookback=8, frac=0.25, max_drop=2):
    # remove up to `max_drop` last rows if they're < frac * median(last lookback)
    g = []
    for cat, grp in df.groupby("product_category_name_english", sort=False):
        grp = grp.sort_values("order_week").copy()
        drops = 0
        while len(grp) >= lookback+1 and drops < max_drop:
            med = grp["units"].tail(lookback+1).head(lookback).median()
            last_val = grp["units"].iloc[-1]
            if pd.notna(last_val) and last_val < frac * med:
                grp = grp.iloc[:-1].copy()
                drops += 1
            else:
                break
        g.append(grp)
    return pd.concat(g, ignore_index=True)

hist = trim_tail(wk_full, lookback=8, frac=0.25, max_drop=2)

# build plan: keep prices flat (= last 4w mean) unless you have price_plan.csv
recent = (hist.groupby("product_category_name_english")
            .agg(planned_price=("avg_price", lambda s: s.tail(4).mean()))
            .reset_index())

H=8
last_wk = hist["order_week"].max()
future_weeks = [last_wk + pd.to_timedelta(7*(i+1), unit="D") for i in range(H)]
plan = pd.MultiIndex.from_product(
    [future_weeks, recent["product_category_name_english"]],
    names=["order_week","product_category_name_english"]
).to_frame(index=False).merge(recent, on="product_category_name_english", how="left")

def compute_features(df):
    df = df.sort_values(["product_category_name_english","order_week"]).copy()
    g  = df.groupby("product_category_name_english", sort=False)
    df["weekofyear"]=df["order_week"].dt.isocalendar().week.astype(int)
    df["year"]=df["order_week"].dt.year
    df["sin_woy"]=np.sin(2*np.pi*df["weekofyear"]/52.0)
    df["cos_woy"]=np.cos(2*np.pi*df["weekofyear"]/52.0)
    if "is_holiday_week" not in df: df["is_holiday_week"]=0
    for col in ["rev_count","rev_score_avg"]:
        if col not in df: df[col]=np.nan
        df[col]=df[col].fillna(g[col].transform(lambda s: s.tail(4).mean()))
    if "avg_price" not in df: df["avg_price"]=np.nan
    LAGS=[1,2,4,8,12,52]
    for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
        for L in LAGS:
            df[f"{col}_lag{L}"]=g[col].shift(L)
    df["units_roll4"]=g["units"].transform(lambda s: s.rolling(4,min_periods=1).mean())
    df["revenue_roll4"]=g["revenue"].transform(lambda s: s.rolling(4,min_periods=1).mean())
    return df

forecasts=[]
history = hist.copy()
for wk_i in future_weeks:
    step = plan[plan["order_week"]==wk_i] \
            .rename(columns={"planned_price":"avg_price"}) \
            [["order_week","product_category_name_english","avg_price"]].copy()
    step["units"]=np.nan; step["revenue"]=np.nan
    step["rev_count"]=np.nan; step["rev_score_avg"]=np.nan
    tmp = pd.concat([history, step], ignore_index=True)
    tmp = compute_features(tmp)
    # proxy revenue since model expects 'revenue'
    mask = tmp["order_week"].eq(wk_i)
    units_proxy = tmp.loc[mask,"units_lag1"].fillna(tmp.loc[mask,"units_roll4"]).fillna(0.0)
    tmp.loc[mask,"revenue"] = tmp.loc[mask,"avg_price"]*units_proxy
    for c in FEAT_COLS:
        if c not in tmp.columns: tmp[c]=0.0
    X = tmp.loc[mask, FEAT_COLS]
    yhat = pipe.predict(X)
    pred = tmp.loc[mask, ["order_week","product_category_name_english","avg_price"]].copy()
    pred["units"]=yhat
    pred["revenue"]=pred["avg_price"]*pred["units"]
    history = pd.concat([history, pred], ignore_index=True, sort=False)
    forecasts.append(pred)

fcast = pd.concat(forecasts, ignore_index=True)
fcast.to_csv(OUT/"forecast_trimmed.csv", index=False)
print("Saved:", OUT/"forecast_trimmed.csv")
