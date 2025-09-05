import pandas as pd, numpy as np
from pathlib import Path
import joblib

OUT = Path("Dataset/model_outputs")
wk_full  = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"]) \
              .sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)
wk_model = pd.read_csv(OUT/"weekly_by_category_no_leak.csv", parse_dates=["order_week"])
pipe = joblib.load(OUT/"weekly_units_xgb_no_leak.pkl")

# feature list used by the new model
TARGET="units"; cat_cols=["product_category_name_english"]
drop_cols={"order_week","split",TARGET,"revenue"}  # current revenue not used
num_cols=[c for c in wk_model.columns if c not in set(cat_cols)|drop_cols]
FEAT = cat_cols + num_cols

# add week_idx into full history (if missing)
wk_full["week_idx"] = wk_full.groupby("product_category_name_english").cumcount()

# trim trailing partial weeks
def trim_tail(df, lookback=8, frac=0.25, max_drop=2):
    g=[];
    for cat, grp in df.groupby("product_category_name_english", sort=False):
        grp=grp.sort_values("order_week").copy(); drops=0
        while len(grp)>=lookback+1 and drops<max_drop:
            med=grp["units"].tail(lookback+1).head(lookback).median()
            last=grp["units"].iloc[-1]
            if pd.notna(last) and last < frac*med:
                grp=grp.iloc[:-1].copy(); drops+=1
            else: break
        g.append(grp)
    return pd.concat(g, ignore_index=True)

history = trim_tail(wk_full)

# horizon and flat price plan = last 4w mean
recent = (history.groupby("product_category_name_english")
          .agg(avg_price=("avg_price", lambda s: s.tail(4).mean()))
          .reset_index())
H=8
last_wk=history["order_week"].max()
future_weeks=[last_wk + pd.to_timedelta(7*(i+1), unit="D") for i in range(H)]
plan = pd.MultiIndex.from_product([future_weeks, recent["product_category_name_english"]],
        names=["order_week","product_category_name_english"]).to_frame(index=False) \
        .merge(recent, on="product_category_name_english", how="left")

# roll forward
forecasts=[]
for i, wk_i in enumerate(future_weeks, start=1):
    step = plan[plan["order_week"]==wk_i][["order_week","product_category_name_english","avg_price"]].copy()
    step["units"]=np.nan; step["revenue"]=np.nan
    # extend week_idx
    step = step.merge(
        history.groupby("product_category_name_english")["week_idx"].max().reset_index()
               .rename(columns={"week_idx":"_last_idx"}),
        on="product_category_name_english", how="left"
    )
    step["week_idx"] = step["_last_idx"] + 1
    step.drop(columns=["_last_idx"], inplace=True)

    tmp = pd.concat([history, step], ignore_index=True)
    # recompute lags/rollings to match training
    g = tmp.groupby("product_category_name_english", sort=False)
    for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
        for L in [1,2,4,8,12,52]:
            tmp[f"{col}_lag{L}"] = g[col].shift(L)
    tmp["units_roll4"]   = g["units"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    tmp["revenue_roll4"] = g["revenue"].transform(lambda s: s.rolling(4, min_periods=1).mean())

    X = tmp[tmp["order_week"]==wk_i][FEAT]
    yhat = pipe.predict(X)

    pred = tmp[tmp["order_week"]==wk_i][["order_week","product_category_name_english","avg_price","week_idx"]].copy()
    pred["units"]=yhat
    pred["revenue"]=pred["avg_price"]*pred["units"]

    history = pd.concat([history, pred], ignore_index=True, sort=False)
    forecasts.append(pred[["order_week","product_category_name_english","avg_price","units","revenue"]])

fcast = pd.concat(forecasts, ignore_index=True)
fcast.to_csv(OUT/"forecast_no_leak.csv", index=False)
print("Saved:", OUT/"forecast_no_leak.csv")
