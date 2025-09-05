#Actionable Price Moves

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# === 1) Load the scenario deltas you created earlier ===
delta_path = OUT/"elasticity_deltas_by_category.csv"
df = pd.read_csv(delta_path)

# Expected columns. Recompute safely if any are missing.
need = {"product_category_name_english","scenario",
        "units_s","revenue_s","units0","revenue0"}
missing = need - set(df.columns)

if missing:
    # Fall back: rebuild from detailed scenarios if needed.
    detail_path = OUT/"elasticity_scenarios_detailed.csv"
    det = pd.read_csv(detail_path)

    base = (det.query("scenario == 0.0")
            .groupby("product_category_name_english", as_index=False)
            .agg(units0=("units_cal","sum"),
                 revenue0=("revenue_cal","sum")))
    agg  = (det.query("scenario != 0.0")
            .groupby(["product_category_name_english","scenario"], as_index=False)
            .agg(units_s=("units_cal","sum"),
                 revenue_s=("revenue_cal","sum")))
    df = agg.merge(base, on="product_category_name_english", how="left")

# (Re)compute lifts & helpers (idempotent if already present)
if "delta_units" not in df.columns:
    df["delta_units"] = df["units_s"] - df["units0"]
if "delta_revenue" not in df.columns:
    df["delta_revenue"] = df["revenue_s"] - df["revenue0"]

df["rev_lift"]      = df["delta_revenue"]
df["rev_lift_abs"]  = df["rev_lift"].abs()
df["units_lift"]    = df["delta_units"]
df["units_lift_abs"]= df["units_lift"].abs()

# === 2) Select best scenarios per category ===
# Best by absolute revenue impact
idx_abs = (df.groupby("product_category_name_english")["rev_lift_abs"]
             .idxmax().dropna().astype(int))
best_abs = df.loc[idx_abs].reset_index(drop=True)
best_abs = best_abs.sort_values("rev_lift_abs", ascending=False).reset_index(drop=True)

# Best positive (gain) and worst (loss) by revenue
idx_gain = (df[df["rev_lift"]>0]
            .groupby("product_category_name_english")["rev_lift"]
            .idxmax().dropna().astype(int))
best_gain = df.loc[idx_gain].sort_values("rev_lift", ascending=False).reset_index(drop=True)

idx_loss = (df[df["rev_lift"]<0]
            .groupby("product_category_name_english")["rev_lift"]
            .idxmin().dropna().astype(int))
worst_loss = df.loc[idx_loss].sort_values("rev_lift", ascending=True).reset_index(drop=True)

# === 3) Save CSVs ===
best_abs.to_csv(OUT/"elasticity_best_by_abs_revenue.csv", index=False)
best_gain.to_csv(OUT/"elasticity_best_positive_revenue.csv", index=False)
worst_loss.to_csv(OUT/"elasticity_worst_negative_revenue.csv", index=False)
print("Saved:",
      OUT/"elasticity_best_by_abs_revenue.csv",
      OUT/"elasticity_best_positive_revenue.csv",
      OUT/"elasticity_worst_negative_revenue.csv")

# === 4) Plots ===
def barh_top(df_plot, value, title, fname, top_n=15, xlabel=None):
    g = df_plot.head(top_n).copy()
    plt.figure(figsize=(12,6))
    plt.barh(g["product_category_name_english"], g[value])
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel or value)
    plt.title(title)
    plt.tight_layout()
    out = OUT/fname
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved plot:", out)

# Top revenue gains (show scenario that gives the biggest positive lift)
barh_top(best_gain, "rev_lift",
         "Top categories — best revenue gain (choose scenario)",
         "elasticity_best_gain.png",
         xlabel="Revenue lift vs baseline")

# Largest revenue risks (biggest negative lift)
barh_top(worst_loss.assign(rev_lift_neg=worst_loss["rev_lift"].abs()),
         "rev_lift_neg",
         "Top categories — biggest revenue risk (choose scenario)",
         "elasticity_biggest_risk.png",
         xlabel="Absolute revenue drop vs baseline")
