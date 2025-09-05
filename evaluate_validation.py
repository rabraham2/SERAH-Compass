# evaluate_validation.py  (fixed)
import numpy as np
import pandas as pd
from pathlib import Path

# Use a non-interactive backend to avoid Tkinter errors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Pick whichever predictions file exists
pred_path = None
for p in [OUT/"valid_predictions_no_leak.csv", OUT/"valid_predictions.csv"]:
    if p.exists():
        pred_path = p
        break
if pred_path is None:
    raise FileNotFoundError("No valid_predictions*.csv found in Dataset/model_outputs/")

# Read & parse dates (dayfirst) and normalize columns
df = pd.read_csv(pred_path)
df["order_week"] = pd.to_datetime(df["order_week"], dayfirst=True, errors="coerce")
df = df.rename(columns={
    "category": "product_category_name_english",
    "pred_units": "yhat",
    "actual_units": "y",
})
df = df.dropna(subset=["order_week", "y", "yhat"])
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df["yhat"] = pd.to_numeric(df["yhat"], errors="coerce")
df = df.dropna(subset=["y", "yhat"])

# Overall metrics
df["abs_err"] = (df["y"] - df["yhat"]).abs()
mae   = float(df["abs_err"].mean())
wmape = float(df["abs_err"].sum() / max(1e-9, df["y"].abs().sum()))
bias  = float((df["yhat"] - df["y"]).mean())
corr  = float(np.corrcoef(df["y"], df["yhat"])[0, 1]) if len(df) > 1 else np.nan
r2    = float(corr**2) if np.isfinite(corr) else np.nan

pd.Series({"MAE": mae, "wMAPE": wmape, "Bias": bias, "R2": r2}).to_csv(
    OUT / "validation_overall_metrics.csv"
)

# Per-category metrics (no .apply deprecation)
df["bias"] = df["yhat"] - df["y"]
per = (df.groupby("product_category_name_english", as_index=False)
         .agg(n=("y", "size"),
              sum_abs_err=("abs_err", "sum"),
              sum_abs_y=("y", lambda s: s.abs().sum()),
              MAE=("abs_err", "mean"),
              Bias=("bias", "mean"),
              Actual_sum=("y", "sum")))
per["wMAPE"] = per["sum_abs_err"] / per["sum_abs_y"].clip(lower=1e-9)
per = per.sort_values("wMAPE", ascending=False)
per.to_csv(OUT / "validation_by_category.csv", index=False)

# Parity plot + calibration line (ŷ vs y)
lims = [0, max(df["y"].max(), df["yhat"].max()) * 1.05]
# Linear fit y ≈ a + b·ŷ
b, a = np.polyfit(df["yhat"].values, df["y"].values, 1)

plt.figure(figsize=(6, 6))
plt.scatter(df["y"], df["yhat"], s=16, alpha=0.6)
plt.plot(lims, lims, "--", label="y=x")
# Show calibration line rearranged into predicted space: ŷ ≈ (y - a)/b
if abs(b) > 1e-9:
    plt.plot(lims, [(y - a) / b for y in lims], "-", label="calibration", linewidth=2)
plt.xlabel("Actual units"); plt.ylabel("Predicted units")
plt.title("Validation — Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "val_parity_with_calibration.png", dpi=150)
plt.close()

# Worst categories by wMAPE and bias
top_bad = per[per["Actual_sum"] > 0].head(15)
plt.figure(figsize=(9, 5))
plt.barh(top_bad["product_category_name_english"], top_bad["wMAPE"])
plt.gca().invert_yaxis()
plt.xlabel("wMAPE"); plt.title("Worst categories by wMAPE (validation)")
plt.tight_layout(); plt.savefig(OUT / "val_worst_wmape.png", dpi=150); plt.close()

top_bias = per.sort_values("Bias", ascending=False).head(15)
plt.figure(figsize=(9, 5))
plt.barh(top_bias["product_category_name_english"], top_bias["Bias"])
plt.gca().invert_yaxis()
plt.xlabel("Bias (ŷ − y)"); plt.title("Most over-predicted categories")
plt.tight_layout(); plt.savefig(OUT / "val_worst_bias_over.png", dpi=150); plt.close()

bot_bias = per.sort_values("Bias").head(15)
plt.figure(figsize=(9, 5))
plt.barh(bot_bias["product_category_name_english"], bot_bias["Bias"])
plt.gca().invert_yaxis()
plt.xlabel("Bias (ŷ − y)"); plt.title("Most under-predicted categories")
plt.tight_layout(); plt.savefig(OUT / "val_worst_bias_under.png", dpi=150); plt.close()

# Save calibration params
pd.Series({"intercept": a, "slope": b}).to_csv(OUT / "calibration_params.csv")
print("Saved: validation_overall_metrics.csv, validation_by_category.csv, val_parity_with_calibration.png, val_worst_*.png, calibration_params.csv")
