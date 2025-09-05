# Make per-category price recommendations from your scenario runs
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")        # must be before importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("Dataset/model_outputs")

# Detailed runs you already produced
scen = pd.read_csv(OUT/"elasticity_scenarios_detailed.csv", parse_dates=["order_week"])

# Aggregate revenue & units over the 8w horizon per category & scenario
agg = (scen.groupby(["product_category_name_english","scenario"], as_index=False)
            .agg(revenue=("revenue_cal","sum"),
                 units=("units_cal","sum"),
                 price=("avg_price","mean")))

# Baseline (0% change) for Δ calculations
base = (agg[agg["scenario"]==0.0]
        .rename(columns={"revenue":"revenue0","units":"units0","price":"price0"})
        .drop(columns=["scenario"]))

m = agg.merge(base, on="product_category_name_english", how="left")
m["rev_lift"]   = m["revenue"] - m["revenue0"]
m["units_lift"] = m["units"]   - m["units0"]
m["pct_dP"]     = (m["price"] - m["price0"]) / m["price0"]
m["pct_dQ"]     = np.where(m["units0"]>0, m["units_lift"]/m["units0"], np.nan)
m["elasticity"] = m["pct_dQ"] / m["pct_dP"]

# Pick best scenario per category by revenue lift
best = (m.loc[m["scenario"]!=0.0]
          .sort_values(["product_category_name_english","rev_lift"], ascending=[True,False])
          .groupby("product_category_name_english", as_index=False)
          .first())

# Optional guardrails: stay within ±10% and require positive lift
best["within_10pct"] = best["scenario"].abs() <= 0.10
best["positive_lift"] = best["rev_lift"] > 0
best["recommended"] = best["within_10pct"] & best["positive_lift"]

# If a cat fails guardrails, fall back to 0% (no change)
fallback = base.copy()
fallback["scenario"] = 0.0
fallback["rev_lift"] = 0.0
fallback["units_lift"] = 0.0
fallback["elasticity"] = np.nan
fallback["within_10pct"] = True
fallback["positive_lift"] = True
fallback["recommended"] = True

final = pd.concat([
    best[best["recommended"]],
    # any category not recommended gets a 0% move
    fallback[~fallback["product_category_name_english"].isin(best[best["recommended"]]["product_category_name_english"])]
], ignore_index=True, sort=False)

cols = ["product_category_name_english","scenario","rev_lift","units_lift",
        "revenue0","revenue","units0","units","price0","price","elasticity",
        "within_10pct","positive_lift","recommended"]
final = final[cols].sort_values("rev_lift", ascending=False)

out_csv = OUT/"elasticity_recommendations_by_category.csv"
final.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# Quick plot: top 15 revenue lifts from the recommended moves
topN = final.sort_values("rev_lift", ascending=False).head(15)
plt.figure(figsize=(14,6))
plt.barh(topN["product_category_name_english"], topN["rev_lift"])
plt.gca().invert_yaxis()
plt.xlabel("Revenue lift vs baseline (8 weeks)")
plt.title("Recommended price moves — top revenue lifts (guardrails applied)")
plt.tight_layout()
plt.savefig(OUT/"elasticity_recommended_top_lifts.png", dpi=150)
plt.close()
print("Saved plot:", OUT/"elasticity_recommended_top_lifts.png")

