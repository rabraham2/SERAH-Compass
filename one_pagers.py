# Category one-pagers: baseline vs proposed price (warning-free)
# - Uses Safe_Top sheet from price_recommendations_summary.xlsx
# - Plots last ~16w history, 8w baseline forecast (scenario=0.0), 8w proposed scenario
# - Outputs per-category PNGs + a combined PDF

import warnings
warnings.filterwarnings(
    "ignore",
    message="The behavior of DatetimeProperties.to_pydatetime is deprecated"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import re

# --------------------------
# Paths & inputs
# --------------------------
BASE = Path("Dataset/model_outputs")
BASE.mkdir(parents=True, exist_ok=True)

# Recommendations (must have product_category_name_english and a scenario column)
safe = pd.read_excel(BASE / "price_recommendations_summary.xlsx", sheet_name="Safe_Top")

# Scenario runs (must have order_week, product_category_name_english, scenario, avg_price, units_cal, revenue_cal)
scen = pd.read_csv(BASE / "elasticity_scenarios_detailed.csv", parse_dates=["order_week"])

# Historical weekly (must have order_week, product_category_name_english, units)
history = (
    pd.read_csv(BASE / "weekly_by_category_full.csv", parse_dates=["order_week"])
      .sort_values(["product_category_name_english", "order_week"])
)

# --------------------------
# Helpers
# --------------------------
def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(s).strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "category"

def find_first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def plot_with_dates(ax, x_series, y_series, **kwargs):
    """
    Convert x to datetime, drop rows with NaT or NaN y, and plot.
    Uses numpy array of Python datetimes to avoid FutureWarning.
    """
    x = pd.to_datetime(x_series, errors="coerce")
    y = pd.to_numeric(y_series, errors="coerce")
    mask = x.notna() & pd.notna(y)
    if not mask.any():
        return 0
    x_py = np.array(x[mask].dt.to_pydatetime())  # future-proofed by converting to np.array
    ax.plot(x_py, y[mask].to_numpy(), **kwargs)
    return int(mask.sum())

# Detect scenario column name used in Safe_Top
SCEN_COL = find_first_col(safe, ["scenario", "scenario_best", "s_best"])
if SCEN_COL is None:
    raise ValueError("No scenario column in Safe_Top. Expected one of: scenario, scenario_best, s_best")
safe[SCEN_COL] = pd.to_numeric(safe[SCEN_COL], errors="coerce")

TOP_N = min(20, len(safe))
safe_sel = safe.head(TOP_N).copy()

# Outputs
onepager_dir = BASE / "onepagers"
onepager_dir.mkdir(exist_ok=True)
pdf_path = BASE / "category_onepagers.pdf"
pdf = PdfPages(pdf_path)
png_paths = []

# Date formatting (robust)
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

# --------------------------
# Build one-pagers
# --------------------------
for _, row in safe_sel.iterrows():
    cat = row["product_category_name_english"]
    s_best = float(row[SCEN_COL]) if pd.notna(row[SCEN_COL]) else 0.0

    # History (last 16 weeks)
    h_last = (
        history[history["product_category_name_english"] == cat]
        .sort_values("order_week")
        .tail(16)
        .copy()
    )

    # Baseline scenario
    base = (
        scen[(scen["product_category_name_english"] == cat) & (scen["scenario"] == 0.0)]
        .copy()
        .sort_values("order_week")
    )

    # Proposed scenario: exact, else nearest available for the category
    prop = (
        scen[(scen["product_category_name_english"] == cat) & (np.isclose(scen["scenario"], s_best))]
        .copy()
        .sort_values("order_week")
    )
    s_used = s_best
    if prop.empty:
        avail = (
            scen[scen["product_category_name_english"] == cat]
            .groupby("scenario", as_index=False)
            .size()[["scenario"]]
        )
        if not avail.empty:
            s_used = float(avail.iloc[(avail["scenario"] - s_best).abs().argsort()].iloc[0]["scenario"])
            prop = (
                scen[(scen["product_category_name_english"] == cat) & (np.isclose(scen["scenario"], s_used))]
                .copy()
                .sort_values("order_week")
            )

    if h_last.empty or base.empty or prop.empty:
        continue

    # Metrics
    price0 = pd.to_numeric(base["avg_price"], errors="coerce").mean()
    price1 = pd.to_numeric(prop["avg_price"], errors="coerce").mean()
    dprice = (price1 - price0) / price0 if price0 else np.nan

    units0 = pd.to_numeric(base["units_cal"], errors="coerce").sum()
    units1 = pd.to_numeric(prop["units_cal"], errors="coerce").sum()
    dunits = units1 - units0

    rev0 = pd.to_numeric(base["revenue_cal"], errors="coerce").sum()
    rev1 = pd.to_numeric(prop["revenue_cal"], errors="coerce").sum()
    drev = rev1 - rev0

    # Profit (optional; may be missing)
    prof0 = pd.to_numeric(row.get("profit0", np.nan), errors="coerce")
    prof1 = pd.to_numeric(row.get("profit_best", np.nan), errors="coerce")
    dprof = pd.to_numeric(row.get("profit_lift", np.nan), errors="coerce")

    # ---- Plot ----
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    n1 = plot_with_dates(ax, h_last["order_week"], h_last["units"],
                         label="History (units)", linewidth=1)
    n2 = plot_with_dates(ax, base["order_week"], base["units_cal"],
                         linestyle="--", label="Baseline forecast (units)")
    n3 = plot_with_dates(ax, prop["order_week"], prop["units_cal"],
                         linestyle="-", label=f"Proposed {s_used:+.0%} price (units)")

    if (n1 + n2 + n3) == 0:
        plt.close(fig)
        continue

    ax.set_title(f"{cat} — baseline vs proposed")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    ax.legend(loc="upper right")

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    lines = [
        f"Proposed price change: {s_used:+.0%}",
        "",
        f"Avg price — base: {price0:,.2f}  |  proposed: {price1:,.2f}  ({dprice:+.1%})",
        f"8w units — base: {units0:,.0f}  |  proposed: {units1:,.0f}  (Δ {dunits:+,.0f})",
        f"8w revenue — base: {rev0:,.0f}  |  proposed: {rev1:,.0f}  (Δ {drev:+,.0f})",
    ]
    if pd.notna(prof0) and pd.notna(prof1):
        lines.append(f"8w profit — base: {prof0:,.0f}  |  proposed: {prof1:,.0f}  (Δ {dprof:+,.0f})")

    ax.text(
        1.02, 0.95,
        "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.05)
    )

    fig.tight_layout()

    # Save PNG & add to PDF
    fpath = onepager_dir / f"onepager_{slugify(cat)}.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    png_paths.append(str(fpath))

pdf.close()

print(f"Saved {len(png_paths)} one-pagers")
print("Sample:", png_paths[:5])
print("Combined PDF:", pdf_path)
