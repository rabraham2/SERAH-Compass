#Apply calibration to forecast

import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("Dataset/model_outputs")

# 1) load forecast (use the one you generated)
f_path = OUT / "forecast_no_leak.csv"
if not f_path.exists():
    f_path = OUT / "forecast_with_plan.csv"  # fallback

fore = pd.read_csv(f_path, parse_dates=["order_week"])
# expected cols: order_week, product_category_name_english, avg_price, units, revenue

# 2) load per-category calibration
act = pd.read_csv(OUT / "validation_actions.csv")

# keep only what we need
cal = act[["product_category_name_english","scale_b","offset_a"]].copy()
cal["scale_b"]  = cal["scale_b"].fillna(1.0)
cal["offset_a"] = cal["offset_a"].fillna(0.0)

# 3) join + calibrate
df = fore.merge(cal, on="product_category_name_english", how="left")
df["scale_b"]  = df["scale_b"].fillna(1.0)
df["offset_a"] = df["offset_a"].fillna(0.0)

df["units_cal"] = (df["offset_a"] + df["scale_b"] * df["units"]).clip(lower=0)
df["revenue_cal"] = df["avg_price"] * df["units_cal"]

# 4) save
cal_path = OUT / f_path.name.replace(".csv", "_calibrated.csv")
df.to_csv(cal_path, index=False)
print("Saved calibrated forecast:", cal_path)

# 5) quick visual on the TOP priority categories
top = act.sort_values("PriorityScore", ascending=False)["product_category_name_english"].head(6).tolist()
plot_df = df[df["product_category_name_english"].isin(top)].copy()

for cat in top:
    g = plot_df[plot_df["product_category_name_english"]==cat].sort_values("order_week")
    plt.figure(figsize=(10,4))
    plt.plot(g["order_week"], g["units"],      label="base units")
    plt.plot(g["order_week"], g["units_cal"],  label="calibrated units")
    plt.title(f"Forecast units — {cat}")
    plt.xlabel("Week"); plt.ylabel("Units"); plt.legend()
    fn = OUT / f"forecast_calibrated_vs_base_{cat}.png"
    plt.tight_layout(); plt.savefig(fn, dpi=140); plt.close()
    print("Saved:", fn)

# 6) aggregate impact
impact = (df.groupby("product_category_name_english", as_index=False)
            .agg(base_units=("units","sum"),
                 cal_units=("units_cal","sum"),
                 base_rev=("revenue","sum"),
                 cal_rev=("revenue_cal","sum")))
impact["Δunits"] = impact["cal_units"] - impact["base_units"]
impact["Δrevenue"] = impact["cal_rev"] - impact["base_rev"]
impact.sort_values("Δrevenue", ascending=False)\
      .to_csv(OUT / "calibration_impact_summary.csv", index=False)
print("Saved:", OUT / "calibration_impact_summary.csv")
