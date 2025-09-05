# Calibration_scatter

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("Dataset/model_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Helpers
def load_csv(path_candidates, parse_dates=None):
    for p in path_candidates:
        if p.exists():
            return pd.read_csv(p, parse_dates=parse_dates), p
    return None, None

def normalize_cols(df):
    # standardize likely names
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"category","cat","product_category"}:
            rename[c] = "product_category_name_english"
        elif cl in {"week","orderweek","week_start","order_week"}:
            rename[c] = "order_week"
        elif cl in {"units_base","base_units","units_pred","units_model","units0"}:
            rename[c] = "units_base"
        elif cl in {"units_cal","calibrated_units","units_calibrated"}:
            rename[c] = "units_cal"
        elif cl == "units":
            # treat 'units' as base when there is also a 'units_cal' elsewhere
            # (we won't rename here unless needed later)
            pass
    return df.rename(columns=rename)

# Try best source first: unified with both columns
unified, unified_path = load_csv([
    OUT/"forecast_unified.csv",                 # produced in your last steps
    OUT/"forecast_with_calibration.csv"
], parse_dates=["order_week"])

if unified is not None:
    unified = normalize_cols(unified)

# Otherwise: merge baseline + calibrated
if unified is None or not ({"units_base","units_cal"} <= set(unified.columns)):
    base, base_path = load_csv([OUT/"forecast_no_leak.csv",
                                OUT/"forecast.csv"], parse_dates=["order_week"])
    cal,  cal_path  = load_csv([OUT/"forecast_no_leak_calibrated.csv",
                                OUT/"forecast_calibrated.csv"], parse_dates=["order_week"])

    if base is not None: base = normalize_cols(base)
    if cal  is not None: cal  = normalize_cols(cal)

    if base is not None and cal is not None:
        # pick a safe key set for merge
        key_cols = [c for c in ["order_week","product_category_name_english"] if c in cal.columns and c in base.columns]
        if not key_cols:
            raise RuntimeError("Cannot find common keys to merge baseline and calibrated forecasts.")

        # if baseline has 'units' but no 'units_base', rename it
        if "units_base" not in base.columns and "units" in base.columns:
            base = base.rename(columns={"units": "units_base"})

        unified = cal.merge(base[key_cols+["units_base"]], on=key_cols, how="left")
        unified_path = cal_path

# Last resort: reconstruct units_base from params
if unified is None:
    raise FileNotFoundError("Could not find any forecast CSVs to plot.")

unified = normalize_cols(unified)

# If we still miss units_base, try inversion using calibration params: units_cal = a + b*units_base
if "units_base" not in unified.columns:
    if "units_cal" not in unified.columns:
        raise RuntimeError(f"{unified_path.name} has neither 'units_base' nor 'units_cal'.")

    params_path = OUT/"calibration_params.csv"
    if not params_path.exists():
        raise RuntimeError("Missing 'units_base' in data and 'calibration_params.csv' not found to reconstruct it.")

    cp = pd.read_csv(params_path)
    cp.columns = [c.lower() for c in cp.columns]
    cp = cp.rename(columns={"category":"product_category_name_english"})
    if not {"a","b","product_category_name_english"} <= set(cp.columns):
        raise RuntimeError("calibration_params.csv must contain columns: category, a, b")

    unified = unified.merge(cp[["product_category_name_english","a","b"]], on="product_category_name_english", how="left")
    if unified["b"].isna().any():
        raise RuntimeError("Some categories in forecast are missing calibration params (b is NaN).")
    unified["units_base"] = (unified["units_cal"] - unified["a"]) / unified["b"]
    # keep for plotting
    HAVE_PARAMS = True
else:
    HAVE_PARAMS = False

# If we still miss units_cal but have base & params, forward-apply
if "units_cal" not in unified.columns:
    params_path = OUT/"calibration_params.csv"
    if not params_path.exists():
        raise RuntimeError("Missing 'units_cal' in data and no calibration_params.csv to compute it.")
    cp = pd.read_csv(params_path)
    cp.columns = [c.lower() for c in cp.columns]
    cp = cp.rename(columns={"category":"product_category_name_english"})
    unified = unified.merge(cp[["product_category_name_english","a","b"]], on="product_category_name_english", how="left")
    unified["units_cal"] = unified["a"] + unified["b"] * unified["units_base"]
    HAVE_PARAMS = True

# Plotting
cats = unified["product_category_name_english"].dropna().unique()
save_dir = OUT / "calibration_scatter"
save_dir.mkdir(exist_ok=True)

for cat in cats:
    df = unified.loc[unified["product_category_name_english"]==cat, ["units_base","units_cal"]].dropna()
    if len(df) < 3:
        continue

    # If params provided/merged, use them; otherwise fit
    if {"a","b"}.issubset(unified.columns):
        row = unified.loc[unified["product_category_name_english"]==cat, ["a","b"]].dropna().tail(1)
        if not row.empty:
            a = float(row["a"].iloc[0]); b = float(row["b"].iloc[0])
        else:
            b, a = np.polyfit(df["units_base"], df["units_cal"], 1)
    else:
        b, a = np.polyfit(df["units_base"], df["units_cal"], 1)

    x = np.linspace(df["units_base"].min(), df["units_base"].max(), 100)
    y = a + b*x

    plt.figure(figsize=(6,6))
    plt.scatter(df["units_base"], df["units_cal"], alpha=0.5)
    plt.plot(x, y, linestyle="--")
    lim = [0, max(df["units_base"].max(), df["units_cal"].max())*1.05]
    plt.plot(lim, lim, alpha=0.4)  # 45° line
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("Base units (model)")
    plt.ylabel("Calibrated units")
    plt.title(f"Calibration — {cat}\n y = {a:.2f} + {b:.3f} x")
    plt.tight_layout()
    plt.savefig(save_dir / f"calibration_scatter_{cat}.png", dpi=150)
    plt.close()

print(f"Saved per-category calibration scatter plots to: {save_dir}")
