# Make price plan and forecast

import pandas as pd, numpy as np
from pathlib import Path
import joblib

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# --- Load history and model ---
wk_full  = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"]) \
              .sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)
wk_model = pd.read_csv(OUT/"weekly_by_category.csv", parse_dates=["order_week"])
pipe = joblib.load(OUT/"weekly_units_xgb.pkl")

# === 1) BUILD PRICE PLAN (robust) ===========================================
recent = (wk_full.groupby("product_category_name_english")
          .agg(base_price=("avg_price", lambda s: s.tail(4).mean()))
          .reset_index())

sel_path = OUT/"selected_actions.csv"
rec_path = OUT/"recommended_actions.csv"
if sel_path.exists():
    actions = pd.read_csv(sel_path)
elif rec_path.exists():
    actions = pd.read_csv(rec_path)
else:
    actions = pd.DataFrame(columns=["category","recommended_pct"])

if "category" not in actions.columns:
    if "product_category_name_english" in actions.columns:
        actions = actions.rename(columns={"product_category_name_english":"category"})
    else:
        actions["category"] = []

pct_col = None
for c in ["recommended_pct","recommended_price_change","percent_price_change"]:
    if c in actions.columns:
        pct_col = c; break
if pct_col is None:
    actions["recommended_pct"] = 0.0
    pct_col = "recommended_pct"

def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

recent["cat_key"]  = norm(recent["product_category_name_english"])
actions["cat_key"] = norm(actions["category"])

act = actions[["cat_key", pct_col]].copy().rename(columns={pct_col:"applied_pct"})
if (act["applied_pct"].abs() > 1.0).any():
    act["applied_pct"] = act["applied_pct"] / 100.0

# horizon & future grid
H = 8
last_wk = wk_full["order_week"].max()
# ✅ FIX: use to_timedelta (or datetime.timedelta). This is the corrected line:
future_weeks = [last_wk + pd.to_timedelta(7*(i+1), unit="D") for i in range(H)]

cats = recent["product_category_name_english"].tolist()
plan = pd.MultiIndex.from_product([future_weeks, cats],
          names=["order_week","product_category_name_english"]).to_frame(index=False)
plan["cat_key"] = norm(plan["product_category_name_english"])

plan = (plan.merge(recent[["product_category_name_english","cat_key","base_price"]],
                   on=["product_category_name_english","cat_key"], how="left")
             .merge(act, on="cat_key", how="left"))

plan["applied_pct"]   = plan["applied_pct"].fillna(0.0)
plan["planned_price"] = plan["base_price"] * (1.0 + plan["applied_pct"])

diag = (plan.groupby("product_category_name_english", as_index=False)
            .agg(base_price=("base_price","first"),
                 applied_pct=("applied_pct","first"),
                 planned_price=("planned_price","first")))
diag["changed"] = np.where(diag["applied_pct"].abs() > 1e-9, "yes", "no")

plan_out = OUT/"price_plan.csv"
diag_out = OUT/"price_plan_debug.csv"
plan.to_csv(plan_out, index=False)
diag.to_csv(diag_out, index=False)
print("Saved price plan:", plan_out)
print("Saved diagnostics:", diag_out)
print(f"Categories with non-zero action: {(diag['changed']=='yes').sum()}/{len(diag)}")

# === 2) FORECAST USING THE PLAN (feature-safe) ===============================
TARGET    = "units"
cat_cols  = ["product_category_name_english"]
drop_cols = {"order_week","split",TARGET}
num_cols  = [c for c in wk_model.columns if c not in set(cat_cols) | drop_cols]
FEAT_COLS = cat_cols + num_cols

history = wk_full.copy()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_category_name_english","order_week"]).copy()
    g  = df.groupby("product_category_name_english", sort=False)

    df["weekofyear"] = df["order_week"].dt.isocalendar().week.astype(int)
    df["year"]       = df["order_week"].dt.year
    df["sin_woy"]    = np.sin(2*np.pi*df["weekofyear"]/52.0)
    df["cos_woy"]    = np.cos(2*np.pi*df["weekofyear"]/52.0)
    if "is_holiday_week" not in df.columns:
        df["is_holiday_week"] = 0

    for col in ["rev_count","rev_score_avg"]:
        if col not in df: df[col] = np.nan
        df[col] = df[col].fillna(g[col].transform(lambda s: s.tail(4).mean()))

    LAGS = [1,2,4,8,12,52]
    for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
        for L in LAGS:
            df[f"{col}_lag{L}"] = g[col].shift(L)

    df["units_roll4"]   = g["units"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["revenue_roll4"] = g["revenue"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    return df

forecasts = []
for wk_i in future_weeks:
    step = plan[plan["order_week"]==wk_i][
        ["order_week","product_category_name_english","planned_price"]
    ].rename(columns={"planned_price":"avg_price"}).copy()

    step["units"] = np.nan
    step["revenue"] = np.nan
    step["rev_count"] = np.nan
    step["rev_score_avg"] = np.nan

    tmp = pd.concat([history, step], ignore_index=True)
    tmp = compute_features(tmp)

    # proxy revenue for current step (model expects a 'revenue' feature)
    mask = tmp["order_week"].eq(wk_i)
    units_proxy = tmp.loc[mask, "units_lag1"].fillna(tmp.loc[mask, "units_roll4"]).fillna(0.0)
    tmp.loc[mask, "revenue"] = tmp.loc[mask, "avg_price"] * units_proxy

    for c in FEAT_COLS:
        if c not in tmp.columns:
            tmp[c] = 0.0

    X = tmp.loc[mask, FEAT_COLS]
    yhat = pipe.predict(X)

    pred = tmp.loc[mask, ["order_week","product_category_name_english","avg_price"]].copy()
    pred["units"]   = yhat
    pred["revenue"] = pred["avg_price"] * pred["units"]

    history = pd.concat([history, pred], ignore_index=True, sort=False)
    forecasts.append(pred)

fcast = pd.concat(forecasts, ignore_index=True)
fcast_out = OUT/"forecast_with_plan.csv"
fcast.to_csv(fcast_out, index=False)
print("Saved forecast:", fcast_out)
