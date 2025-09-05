import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# paths & inputs

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# core artifacts produced earlier
wk_full  = (pd.read_csv(OUT/"weekly_by_category_full.csv",
                        parse_dates=["order_week"])
              .sort_values(["product_category_name_english","order_week"]))

wk_model = pd.read_csv(OUT/"weekly_by_category_no_leak.csv",
                       parse_dates=["order_week"])

pipe     = joblib.load(OUT/"weekly_units_xgb_no_leak.pkl")

# optional calibration (units_cal = a + b * units_base)
calib_path = OUT/"calibration_params.csv"
calib = None
if calib_path.exists():
    calib = pd.read_csv(calib_path)
    calib.columns = [c.lower() for c in calib.columns]
    calib = calib.rename(columns={"category":"product_category_name_english"})


# features exactly like training

TARGET="units"; cat_cols=["product_category_name_english"]
drop_cols={"order_week","split",TARGET,"revenue"}  # 'revenue' excluded by design
num_cols=[c for c in wk_model.columns if c not in set(cat_cols)|drop_cols]
FEAT = cat_cols + num_cols  # columns fed into the pipeline


# helper: trim trailing partial weeks from history (same as before)

def trim_tail(df, lookback=8, frac=0.25, max_drop=2):
    out=[]
    for cat, g in df.groupby("product_category_name_english", sort=False):
        g=g.sort_values("order_week").copy(); drops=0
        while len(g)>=lookback+1 and drops<max_drop:
            med=g["units"].tail(lookback+1).head(lookback).median()
            last=g["units"].iloc[-1]
            if pd.notna(last) and last < frac*med:
                g=g.iloc[:-1].copy(); drops+=1
            else:
                break
        out.append(g)
    return pd.concat(out, ignore_index=True)

history = trim_tail(wk_full).copy()

# ensure week_idx exists (some features may depend on it)
if "week_idx" not in history.columns:
    history["week_idx"] = history.groupby("product_category_name_english").cumcount()


# base price plan = last-4-week mean per category

recent_price = (history.groupby("product_category_name_english")
                .agg(base_price=("avg_price", lambda s: s.tail(4).mean()))
                .reset_index())

# horizon
H=8
last_wk=history["order_week"].max()
future_weeks=[last_wk + pd.to_timedelta(7*(i+1), unit="D") for i in range(H)]


# forecaster under a price multiplier

LAGS=[1,2,4,8,12,52]

def run_forecast(multiplier: float) -> pd.DataFrame:
    """Roll forward H weeks with price = base_price*(1+multiplier)."""
    hist = history.copy()
    out  = []

    # scenario plan (flat price by cat across horizon)
    plan = pd.MultiIndex.from_product([future_weeks, recent_price["product_category_name_english"]],
            names=["order_week","product_category_name_english"]).to_frame(index=False) \
            .merge(recent_price, on="product_category_name_english", how="left")
    plan["avg_price"] = plan["base_price"] * (1.0 + multiplier)

    for wk_i in future_weeks:
        step = plan.loc[plan["order_week"]==wk_i,
                        ["order_week","product_category_name_english","avg_price"]].copy()
        step["units"]=np.nan; step["revenue"]=np.nan

        # extend running week_idx per category
        max_idx = hist.groupby("product_category_name_english")["week_idx"].max().reset_index() \
                      .rename(columns={"week_idx":"_last_idx"})
        step = step.merge(max_idx, on="product_category_name_english", how="left")
        step["week_idx"] = step["_last_idx"] + 1
        step.drop(columns=["_last_idx"], inplace=True)

        tmp = pd.concat([hist, step], ignore_index=True, sort=False)

        # recompute lags/rollings matching training
        g = tmp.groupby("product_category_name_english", sort=False)
        for col in ["units","revenue","avg_price","rev_count","rev_score_avg"]:
            if col not in tmp.columns: tmp[col]=np.nan
            for L in LAGS:
                tmp[f"{col}_lag{L}"] = g[col].shift(L)
        tmp["units_roll4"]   = g["units"].transform(lambda s: s.rolling(4, min_periods=1).mean())
        tmp["revenue_roll4"] = g["revenue"].transform(lambda s: s.rolling(4, min_periods=1).mean())

        # predict
        X = tmp.loc[tmp["order_week"]==wk_i, FEAT]
        yhat = pipe.predict(X)  # base units

        pred = tmp.loc[tmp["order_week"]==wk_i,
                       ["order_week","product_category_name_english","avg_price","week_idx"]].copy()
        pred["units_base"] = yhat
        pred["revenue_base"] = pred["avg_price"] * pred["units_base"]

        # optional calibration
        if calib is not None and {"a","b"}.issubset(calib.columns):
            pred = pred.merge(calib[["product_category_name_english","a","b"]],
                              on="product_category_name_english", how="left")
            pred["units_cal"] = pred["a"] + pred["b"] * pred["units_base"]
            pred.drop(columns=["a","b"], inplace=True)
        else:
            pred["units_cal"] = pred["units_base"]

        pred["revenue_cal"] = pred["avg_price"] * pred["units_cal"]
        pred["scenario"] = multiplier

        # append back to history for next-step lags (use BASE units for recursion)
        to_hist = pred.rename(columns={"units_base":"units",
                                       "revenue_base":"revenue"})
        hist = pd.concat([hist, to_hist[["order_week","product_category_name_english",
                                         "avg_price","units","revenue","week_idx"]]],
                         ignore_index=True, sort=False)
        out.append(pred.drop(columns=["week_idx"]))

    return pd.concat(out, ignore_index=True)


# scenarios and execution

SCENARIOS = [-0.15,-0.10,-0.05,0.0,0.05,0.10,0.15]
all_runs = []
for s in SCENARIOS:
    print(f"Running scenario {s:+.0%} ...")
    all_runs.append(run_forecast(s))
scen_df = pd.concat(all_runs, ignore_index=True)

# save the detailed per-week outputs
scen_df.to_csv(OUT/"elasticity_scenarios_detailed.csv", index=False)


# aggregate by category for Δ vs baseline (0.0)

base = scen_df[scen_df["scenario"]==0.0] \
        .groupby("product_category_name_english", as_index=False) \
        .agg(units0=("units_cal","sum"),
             revenue0=("revenue_cal","sum"),
             price0=("avg_price","mean"))

rows=[]
for s in [x for x in SCENARIOS if x!=0.0]:
    agg = scen_df[scen_df["scenario"]==s] \
            .groupby("product_category_name_english", as_index=False) \
            .agg(units_s=("units_cal","sum"),
                 revenue_s=("revenue_cal","sum"),
                 price_s=("avg_price","mean"))
    m = agg.merge(base, on="product_category_name_english", how="left")
    m["scenario"] = s
    m["delta_units"]   = m["units_s"]   - m["units0"]
    m["delta_revenue"] = m["revenue_s"] - m["revenue0"]
    # simple elasticity: %ΔQ / %ΔP using baseline means
    m["pct_dQ"] = np.where(m["units0"]>0, m["delta_units"]/m["units0"], np.nan)
    m["pct_dP"] = np.where(m["price0"]>0, (m["price_s"]-m["price0"])/m["price0"], np.nan)
    m["elasticity"] = m["pct_dQ"] / m["pct_dP"]
    rows.append(m)

delta_cat = pd.concat(rows, ignore_index=True)
delta_cat.to_csv(OUT/"elasticity_deltas_by_category.csv", index=False)
print("Saved:",
      OUT/"elasticity_scenarios_detailed.csv",
      OUT/"elasticity_deltas_by_category.csv")


# plotting helpers

def barplot_delta(df, value_col, title_prefix, fname_prefix, top_n=15):
    """Make a bar chart per scenario, ordered by absolute impact, top N categories."""
    for s, g in df.groupby("scenario"):
        g = g.sort_values(value_col, key=lambda x: x.abs(), ascending=False).head(top_n)
        plt.figure(figsize=(12,6))
        plt.barh(g["product_category_name_english"], g[value_col])
        plt.gca().invert_yaxis()
        plt.xlabel("Δ " + ("Units" if "units" in value_col else "Revenue"))
        plt.title(f"{title_prefix} {s:+.0%}")
        plt.tight_layout()
        png = OUT/f"{fname_prefix}_{int(round(s*100)):+d}.png"
        plt.savefig(png, dpi=150)
        plt.close()
        print("Saved plot:", png)

# ΔUnits and ΔRevenue (per scenario)
barplot_delta(delta_cat, "delta_units",
              "ΔUnits by category — price change",
              "elasticity_delta_units")

barplot_delta(delta_cat, "delta_revenue",
              "ΔRevenue by category — price change",
              "elasticity_delta_revenue")

# Optional: a compact “elasticity leaderboard” (most sensitive categories)
sens = (delta_cat.groupby("product_category_name_english", as_index=False)
        .agg(mean_abs_elasticity=("elasticity", lambda s: np.nanmean(np.abs(s)))))
sens = sens.sort_values("mean_abs_elasticity", ascending=False).head(20)
plt.figure(figsize=(10,6))
plt.barh(sens["product_category_name_english"], sens["mean_abs_elasticity"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |Elasticity| across scenarios")
plt.title("Most price-sensitive categories")
plt.tight_layout()
plt.savefig(OUT/"elasticity_top_sensitive.png", dpi=150)
plt.close()
print("Saved plot:", OUT/"elasticity_top_sensitive.png")
