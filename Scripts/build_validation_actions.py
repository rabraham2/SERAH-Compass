# Build validation actions

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("Dataset/model_outputs")

# 1) Load
per = pd.read_csv(OUT / "validation_by_category.csv")

pred_path = OUT / "valid_predictions_no_leak.csv"
if not pred_path.exists():
    pred_path = OUT / "valid_predictions.csv"

pred = pd.read_csv(pred_path)
pred["order_week"] = pd.to_datetime(pred["order_week"], dayfirst=True, errors="coerce")
pred = pred.rename(columns={"category":"product_category_name_english",
                            "actual_units":"y", "pred_units":"yhat"})
pred = pred.dropna(subset=["product_category_name_english","y","yhat"])

hist = pd.read_csv(OUT / "weekly_by_category_full.csv", parse_dates=["order_week"])
vmin, vmax = pred["order_week"].min(), pred["order_week"].max()
price_val = (hist[(hist["order_week"]>=vmin)&(hist["order_week"]<=vmax)]
             .groupby("product_category_name_english", as_index=False)
             .agg(avg_price_val=("avg_price","mean")))

# ---------- 2) Enrich metrics ----------
if "sum_abs_err" not in per.columns or "sum_abs_y" not in per.columns:
    tmp = pred.copy(); tmp["abs_err"] = (tmp["y"]-tmp["yhat"]).abs()
    recon = (tmp.groupby("product_category_name_english", as_index=False)
               .agg(sum_abs_err=("abs_err","sum"),
                    sum_abs_y=("y", lambda s: s.abs().sum())))
    per = per.merge(recon, on="product_category_name_english", how="left")

per = per.merge(price_val, on="product_category_name_english", how="left")
per["wMAPE_pct"]      = per["wMAPE"]*100.0
per["Revenue_at_Risk"]= per["sum_abs_err"] * per["avg_price_val"].fillna(0.0)
per["PriorityScore"]  = per["wMAPE"] * per["Actual_sum"].abs()

# ---------- 3) Per-category calibration ----------
def fit_calib(g: pd.DataFrame) -> pd.Series:
    if len(g)>=2 and g["yhat"].nunique()>=2:
        b, a = np.polyfit(g["yhat"].to_numpy(), g["y"].to_numpy(), 1)
    else:
        b, a = 1.0, 0.0
    return pd.Series({"scale_b": float(b),
                      "offset_a": float(a),
                      "Bias_units": float((g["yhat"]-g["y"]).mean())})

calib = (pred.groupby("product_category_name_english")
           .apply(fit_calib, include_groups=False)  # <- silence future warning
           .reset_index())

per = per.merge(calib, on="product_category_name_english", how="left")

def make_rec(row):
    bias = float(row.get("Bias", 0.0))
    if bias < -5:
        return f"Underpredict by {-bias:.1f} u; try scale x{row['scale_b']:.2f} or add {row['offset_a']:.1f}"
    if bias > 5:
        return f"Overpredict by {bias:.1f} u; try scale x{row['scale_b']:.2f} or add {row['offset_a']:.1f}"
    return f"OK; keep global calibration (x{row['scale_b']:.2f}, +{row['offset_a']:.1f})"

per["Recommendation"] = per.apply(make_rec, axis=1)

# ---------- 4) Actions table ----------
actions = (per.sort_values(["PriorityScore","Revenue_at_Risk"], ascending=False)[
    ["product_category_name_english","n","MAE","wMAPE_pct","Bias",
     "Actual_sum","avg_price_val","sum_abs_err","Revenue_at_Risk",
     "PriorityScore",  # <- keep it so we can sort later
     "scale_b","offset_a","Recommendation"]
])
actions.to_csv(OUT / "validation_actions.csv", index=False)
print("Saved:", OUT / "validation_actions.csv")

# 5) PNGs
topR = actions.head(15)
plt.figure(figsize=(9,5))
plt.barh(topR["product_category_name_english"], topR["Revenue_at_Risk"])
plt.gca().invert_yaxis()
plt.xlabel("Revenue at Risk (validation)")
plt.title("Top categories by $ error (validation)")
plt.tight_layout(); plt.savefig(OUT/"val_top_revenue_at_risk.png", dpi=150); plt.close()

topP = actions.sort_values("PriorityScore", ascending=False).head(15)
plt.figure(figsize=(9,5))
plt.barh(topP["product_category_name_english"], topP["PriorityScore"])
plt.gca().invert_yaxis()
plt.xlabel("PriorityScore = wMAPE × Actual_sum")
plt.title("Top categories to fix first")
plt.tight_layout(); plt.savefig(OUT/"val_top_priority.png", dpi=150); plt.close()

print("Saved PNGs: val_top_revenue_at_risk.png, val_top_priority.png")
