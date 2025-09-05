# plot_validation_accuracy.py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# pick whichever predictions file exists
cand = [OUT/"valid_predictions_no_leak.csv", OUT/"valid_predictions.csv"]
pred_path = next((p for p in cand if p.exists()), None)
if pred_path is None:
    raise FileNotFoundError("No validation predictions file found in Dataset/model_outputs/.")

pred = pd.read_csv(pred_path, parse_dates=["order_week"])
pred.rename(columns={
    "category": "product_category_name_english",
    "pred_units": "yhat",
    "actual_units": "y"
}, inplace=True)

# safety: keep clean datetimes & numerics
pred["order_week"] = pd.to_datetime(pred["order_week"], errors="coerce")
pred["y"] = pd.to_numeric(pred["y"], errors="coerce")
pred["yhat"] = pd.to_numeric(pred["yhat"], errors="coerce")
pred = pred.dropna(subset=["order_week", "y", "yhat"])

# choose categories to plot (top by total actual units in validation)
TOP_N = 12
cats = (pred.groupby("product_category_name_english")["y"]
            .sum().sort_values(ascending=False).head(TOP_N).index.tolist())

def clean_xy(df):
    x = pd.to_datetime(df["order_week"], errors="coerce")
    y1 = pd.to_numeric(df["y"], errors="coerce")
    y2 = pd.to_numeric(df["yhat"], errors="coerce")
    m = x.notna() & y1.notna() & y2.notna()
    x = x[m].to_numpy(dtype="datetime64[ns]")  # keeps matplotlib date converter happy
    y1 = y1[m].to_numpy(dtype=float)
    y2 = y2[m].to_numpy(dtype=float)
    return x, y1, y2

# per-category plots with metrics
for c in cats:
    df = pred[pred["product_category_name_english"]==c].sort_values("order_week")
    if df.empty:
        continue
    x, y, yhat = clean_xy(df)

    mae  = np.mean(np.abs(y - yhat))
    mape = np.mean(np.abs((y - yhat) / np.maximum(y, 1e-6)))
    r    = np.corrcoef(y, yhat)[0,1] if len(y) > 1 else np.nan
    r2   = r**2 if not np.isnan(r) else np.nan

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, y,    label="Actual",   linewidth=2)
    ax.plot(x, yhat, label="Predicted", linewidth=2)

    ax.set_title(f"Validation — {c}")
    ax.set_xlabel("Week"); ax.set_ylabel("Units")
    ax.legend()

    # neat date ticks
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    # annotate metrics
    txt = f"MAE={mae:.2f}  MAPE={mape:.2%}  R²={r2:.3f}"
    ax.text(0.01, 0.99, txt, transform=ax.transAxes, va="top", ha="left")

    fig.tight_layout()
    fig.savefig(OUT / f"val_ts_{c.replace('/','_')}.png", dpi=150)
    plt.close(fig)

# (optional) overall scatter with y=x
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(pred["y"], pred["yhat"], s=12, alpha=0.6)
lims = [0, max(pred["y"].max(), pred["yhat"].max())*1.05]
ax.plot(lims, lims, linestyle="--")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual units"); ax.set_ylabel("Predicted units")
ax.set_title("Validation — Actual vs Predicted")
fig.tight_layout()
fig.savefig(OUT/"val_scatter_overall.png", dpi=150)
plt.close(fig)

print("Saved per-category PNGs (val_ts_*.png) and val_scatter_overall.png in:", OUT)
