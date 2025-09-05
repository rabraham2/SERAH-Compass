# plot_validation_overlay.py
import numpy as np
import pandas as pd
from pathlib import Path

# non-interactive backend (no Tk errors)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ---- load predictions (auto-pick no_leak if present) ----
pred_path = None
for p in [OUT/"valid_predictions_no_leak.csv", OUT/"valid_predictions.csv"]:
    if p.exists():
        pred_path = p; break
if pred_path is None:
    raise FileNotFoundError("No valid_predictions*.csv found in Dataset/model_outputs/")

pred = pd.read_csv(pred_path)
pred["order_week"] = pd.to_datetime(pred["order_week"], dayfirst=True, errors="coerce")
pred = pred.rename(columns={"category":"product_category_name_english",
                            "actual_units":"y", "pred_units":"yhat"})
pred = pred.dropna(subset=["order_week","y","yhat"])

# ---- load full history for context ----
hist = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"])
hist = hist.sort_values(["product_category_name_english","order_week"]).reset_index(drop=True)

# pick top categories by validation volume
TOP_N   = 12
LOOKBACK_WEEKS = 26  # weeks of history to show before validation start

cats = (pred.groupby("product_category_name_english")["y"]
            .sum().sort_values(ascending=False).head(TOP_N).index.tolist())

def per_cat_metrics(df):
    mae   = float(np.mean(np.abs(df.y - df.yhat)))
    wmape = float(np.sum(np.abs(df.y - df.yhat)) / max(1e-9, np.sum(np.abs(df.y))))
    r     = np.corrcoef(df.y, df.yhat)[0,1] if len(df) > 1 else np.nan
    r2    = float(r**2) if np.isfinite(r) else np.nan
    return mae, wmape, r2

for c in cats:
    pv = pred[pred["product_category_name_english"]==c].sort_values("order_week").copy()
    if pv.empty: continue

    # context history (last LOOKBACK_WEEKS before validation)
    vstart = pv["order_week"].min()
    ctx = hist[(hist["product_category_name_english"]==c) &
               (hist["order_week"] < vstart)].tail(LOOKBACK_WEEKS)

    mae, wmape, r2 = per_cat_metrics(pv)

    fig, ax = plt.subplots(figsize=(10,4))

    # light-gray context history
    if not ctx.empty:
        ax.plot(ctx["order_week"], ctx["units"], color="#bbbbbb", label="history (context)")

    # validation actual & predicted
    ax.plot(pv["order_week"], pv["y"],    label="Actual (validation)", linewidth=2)
    ax.plot(pv["order_week"], pv["yhat"], label="Predicted (validation)", linewidth=2)

    ax.set_title(f"Validation — {c}   MAE={mae:.2f}  wMAPE={wmape:.2%}  R²={r2:.3f}")
    ax.set_xlabel("Week"); ax.set_ylabel("Units")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"val_overlay_{c.replace('/','_')}.png", dpi=150)
    plt.close(fig)

print("Saved per-category overlays to:", OUT)

import pandas as pd
from pathlib import Path
OUT = Path("Dataset/model_outputs")
per = pd.read_csv(OUT/"validation_by_category.csv")
# helpful ranking views
per.assign(wMAPE_pct=(per.wMAPE*100))\
   .sort_values(["wMAPE","Actual_sum"], ascending=[True,False])\
   .to_csv(OUT/"validation_leaderboard.csv", index=False)
print("Saved:", OUT/"validation_leaderboard.csv")

