# Plot Forecast

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

OUT = Path("Dataset/model_outputs")

hist  = pd.read_csv(OUT/"weekly_by_category_full.csv", parse_dates=["order_week"])
fcast = pd.read_csv(OUT/"forecast_with_plan.csv",     parse_dates=["order_week"])

# top 6 categories by revenue
cats = (hist.groupby("product_category_name_english")["revenue"]
            .sum().sort_values(ascending=False).head(6).index.tolist())

def clean_xy(df):
    # ensure datetime64[ns] and numeric y, drop NaT/NaN
    x = pd.to_datetime(df["order_week"], errors="coerce")
    y = pd.to_numeric(df["units"], errors="coerce")
    m = x.notna() & y.notna()
    # IMPORTANT: return numpy datetime64[ns] array (not object) to keep date converter happy
    x = x[m].to_numpy(dtype="datetime64[ns]")
    y = y[m].to_numpy(dtype=float)
    return x, y

for c in cats:
    h = hist[hist["product_category_name_english"]==c].sort_values("order_week")
    f = fcast[fcast["product_category_name_english"]==c].sort_values("order_week")

    xh, yh = clean_xy(h)
    xf, yf = clean_xy(f)

    if xh.size == 0 and xf.size == 0:
        continue  # nothing to plot for this category

    fig, ax = plt.subplots(figsize=(8, 4))

    if xh.size:
        ax.plot(xh, yh, label="history")
    if xf.size:
        ax.plot(xf, yf, label="forecast")

    # Nice date formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    ax.set_title(f"Units — {c}")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"forecast_units_{c.replace('/','_')}.png")
    plt.close(fig)

print("Saved charts to:", OUT)

