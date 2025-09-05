# Data Analysis Summary

import pandas as pd
from pathlib import Path

base = Path("Dataset")

# Load
orders = pd.read_csv(base / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"])
order_items = pd.read_csv(base / "olist_order_items_dataset.csv")
products = pd.read_csv(base / "olist_products_dataset.csv")
cats = pd.read_csv(base / "product_category_name_translation.csv")
order_payments = pd.read_csv(base / "olist_order_payments_dataset.csv")
reviews = pd.read_csv(base / "olist_order_reviews_dataset.csv", parse_dates=["review_creation_date","review_answer_timestamp"])
customers = pd.read_csv(base / "olist_customers_dataset.csv")
sellers = pd.read_csv(base / "olist_sellers_dataset.csv")
geo = pd.read_csv(base / "olist_geolocation_dataset.csv")

# Merge English category
products = products.merge(cats, on="product_category_name", how="left")

# Helper for NA report
def na_report(df, cols, name):
    tmp = pd.DataFrame({
        "table": name,
        "column": cols,
        "n_missing": [df[c].isna().sum() for c in cols],
        "pct_missing": [float(df[c].isna().mean()) for c in cols],
        "n_rows": len(df)
    })
    return tmp

rep_orders = na_report(orders, ["order_status","order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date","customer_id"], "orders")
rep_items  = na_report(order_items, ["order_id","product_id","seller_id","price","freight_value"], "order_items")
rep_products = na_report(products, ["product_category_name","product_category_name_english","product_weight_g","product_length_cm","product_height_cm","product_width_cm"], "products")
rep_pay = na_report(order_payments, ["order_id","payment_type","payment_installments","payment_value"], "payments")
rep_rev = na_report(reviews, ["order_id","review_score","review_comment_title","review_comment_message"], "reviews")
rep_cust = na_report(customers, ["customer_id","customer_city","customer_state","customer_zip_code_prefix"], "customers")
rep_sell = na_report(sellers, ["seller_id","seller_city","seller_state","seller_zip_code_prefix"], "sellers")
rep_geo = na_report(geo, ["geolocation_zip_code_prefix","geolocation_lat","geolocation_lng","geolocation_city","geolocation_state"], "geolocation")

na_summary = pd.concat([rep_orders, rep_items, rep_products, rep_pay, rep_rev, rep_cust, rep_sell, rep_geo], ignore_index=True)

# Order status distribution + date range overview
status_counts = orders["order_status"].value_counts().rename_axis("order_status").reset_index(name="count")
orders_total = orders["order_id"].nunique()
orders_with_reviews = reviews["order_id"].nunique()

overview = pd.DataFrame([{
    "orders_date_min": str(orders["order_purchase_timestamp"].min()),
    "orders_date_max": str(orders["order_purchase_timestamp"].max()),
    "orders_total": int(orders_total),
    "order_items_rows": int(len(order_items)),
    "products_rows": int(len(products)),
    "payments_rows": int(len(order_payments)),
    "reviews_rows": int(len(reviews)),
    "customers_rows": int(len(customers)),
    "sellers_rows": int(len(sellers)),
    "geolocation_rows": int(len(geo)),
    "orders_with_reviews_pct": float(orders_with_reviews / max(1, orders_total)),
    "qty_column_present_in_items": "quantity" in order_items.columns.tolist()
}])

# Save files
audit_dir = base / "audit_exports"
audit_dir.mkdir(exist_ok=True)

na_summary_path = audit_dir / "missingness_summary.csv"
status_counts_path = audit_dir / "order_status_counts.csv"
overview_path = audit_dir / "dataset_overview.csv"

na_summary.to_csv(na_summary_path, index=False)
status_counts.to_csv(status_counts_path, index=False)
overview.to_csv(overview_path, index=False)

{
  "exports": {
    "missingness_summary": str(na_summary_path),
    "order_status_counts": str(status_counts_path),
    "dataset_overview": str(overview_path)
  }
}



# Data Cleaning for All Columns
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("Dataset")
OUT  = BASE / "clean_exports"
OUT.mkdir(exist_ok=True)


# 0) Load raw

orders = pd.read_csv(
    BASE/"olist_orders_dataset.csv",
    parse_dates=[
        "order_purchase_timestamp","order_approved_at",
        "order_delivered_carrier_date","order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
)
order_items   = pd.read_csv(BASE/"olist_order_items_dataset.csv")
products_raw  = pd.read_csv(BASE/"olist_products_dataset.csv")
cats          = pd.read_csv(BASE/"product_category_name_translation.csv")
order_payments= pd.read_csv(BASE/"olist_order_payments_dataset.csv")
reviews       = pd.read_csv(BASE/"olist_order_reviews_dataset.csv",
                            parse_dates=["review_creation_date","review_answer_timestamp"])
customers     = pd.read_csv(BASE/"olist_customers_dataset.csv")
sellers       = pd.read_csv(BASE/"olist_sellers_dataset.csv")
geo           = pd.read_csv(BASE/"olist_geolocation_dataset.csv")


# 1) Products: categories + dimensions

products = products_raw.merge(cats, on="product_category_name", how="left")
products["product_category_name_english"] = products["product_category_name_english"].fillna("unknown")

# Impute missing dimensions by category median (fallback to global median)
dim_cols = ["product_weight_g","product_length_cm","product_height_cm","product_width_cm"]
cat_medians = products.groupby("product_category_name_english")[dim_cols].median()

for c in dim_cols:
    # replace zeros or negatives with NaN (bad quality)
    products.loc[products[c].le(0, fill_value=False), c] = np.nan
    # category median
    products[c] = products.apply(
        lambda r: r[c] if pd.notna(r[c])
        else cat_medians.loc[r["product_category_name_english"], c]
             if r["product_category_name_english"] in cat_medians.index and
                pd.notna(cat_medians.loc[r["product_category_name_english"], c])
        else products[c].median(),
        axis=1
    )


# 2) Geolocation → postal prefix reference (mode city/state, mean lat/lng)

# Normalise zip prefixes as strings
for zc in ["geolocation_zip_code_prefix"]:
    geo[zc] = geo[zc].astype(str)

geo_ref = (geo.groupby("geolocation_zip_code_prefix")
             .agg(geo_city=("geolocation_city", lambda s: s.mode(dropna=True)[0] if not s.mode(dropna=True).empty else np.nan),
                  geo_state=("geolocation_state", lambda s: s.mode(dropna=True)[0] if not s.mode(dropna=True).empty else np.nan),
                  geo_lat=("geolocation_lat","mean"),
                  geo_lng=("geolocation_lng","mean"))
             .reset_index())


# 3) Customers & Sellers: fill city/state via geo_ref by zip prefix

customers["customer_zip_code_prefix"] = customers["customer_zip_code_prefix"].astype(str)
sellers["seller_zip_code_prefix"]     = sellers["seller_zip_code_prefix"].astype(str)

cust = customers.merge(
    geo_ref, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left"
)
cust["customer_city"]  = cust["customer_city"].fillna(cust["geo_city"])
cust["customer_state"] = cust["customer_state"].fillna(cust["geo_state"])
cust = cust.drop(columns=["geolocation_zip_code_prefix","geo_city","geo_state","geo_lat","geo_lng"])

sell = sellers.merge(
    geo_ref, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left"
)
sell["seller_city"]  = sell["seller_city"].fillna(sell["geo_city"])
sell["seller_state"] = sell["seller_state"].fillna(sell["geo_state"])
sell = sell.drop(columns=["geolocation_zip_code_prefix","geo_city","geo_state","geo_lat","geo_lng"])

# If still missing state/city, fill with most common values
cust["customer_state"] = cust["customer_state"].fillna(cust["customer_state"].mode(dropna=True).iloc[0])
cust["customer_city"]  = cust["customer_city"].fillna("unknown_city")
sell["seller_state"]   = sell["seller_state"].fillna(sell["seller_state"].mode(dropna=True).iloc[0])
sell["seller_city"]    = sell["seller_city"].fillna("unknown_city")


# 4) Orders: keep status logic & safe durations

# Only compute delivered-based durations when dates exist
orders["purchase_to_delivery_days"] = np.where(
    orders["order_status"].eq("delivered") &
    orders["order_delivered_customer_date"].notna(),
    (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days,
    np.nan
)

# For modelling demand/revenue, typically exclude canceled/unavailable
orders["is_valid_for_sales"] = orders["order_status"].isin(["delivered","shipped","invoiced"])


# 5) Order items: critical IDs must exist; price/freight clean-up

items = order_items.copy()

# Drop rows missing critical IDs
items = items.dropna(subset=["order_id","product_id","seller_id"])

# Bad price/freight handling
items["price"] = pd.to_numeric(items["price"], errors="coerce")
items["freight_value"] = pd.to_numeric(items["freight_value"], errors="coerce")

# flag & fix nonpositive price → set NaN, will drop later if cannot be imputed
items.loc[items["price"]<=0, "price"] = np.nan
# negative freight is invalid → set NaN
items.loc[items["freight_value"]<0, "freight_value"] = np.nan

# Impute freight by seller_state median freight when available, else global median
tmp = items.merge(sell[["seller_id","seller_state"]], on="seller_id", how="left")
state_med = tmp.groupby("seller_state")["freight_value"].median()
items = items.merge(sell[["seller_id","seller_state"]], on="seller_id", how="left")
items["freight_value"] = items.apply(
    lambda r: r["freight_value"] if pd.notna(r["freight_value"])
    else state_med.get(r["seller_state"], np.nan), axis=1
)
items["freight_value"] = items["freight_value"].fillna(items["freight_value"].median())

# Price imputation is risky; better to drop if price still missing
before = len(items)
items = items.dropna(subset=["price"])
dropped_price_na = before - len(items)

# Each row is one unit (no quantity column). Add units=1 for clarity
items["units"] = 1


# 6) Payments: tidy and aggregate by order

pays = order_payments.copy()
pays["payment_installments"] = pays["payment_installments"].fillna(1).astype(int)
pays["payment_type"] = pays["payment_type"].fillna("unknown")
pays["payment_value"] = pd.to_numeric(pays["payment_value"], errors="coerce")

# If any payment_value missing, impute with 0 for aggregation; we use sums at order level
pays["payment_value"] = pays["payment_value"].fillna(0.0)
pay_agg = (pays.groupby("order_id")
              .agg(payment_total=("payment_value","sum"),
                   n_payments=("payment_value","size"),
                   installments_max=("payment_installments","max"),
                   has_cc=("payment_type", lambda s: int(any(s.str.contains("credit", case=False, na=False)))))
              .reset_index())


# 7) Reviews: keep score rows; text can be empty

rev = reviews.copy()
rev["review_score"] = pd.to_numeric(rev["review_score"], errors="coerce")
rev = rev.dropna(subset=["review_score"])
rev["review_comment_title"] = rev["review_comment_title"].fillna("")
rev["review_comment_message"] = rev["review_comment_message"].fillna("")
rev_agg = (rev.groupby("order_id")
             .agg(review_count=("review_id","count"),
                  review_score_avg=("review_score","mean"))
             .reset_index())


# 8) Build a clean FACT (order line) for modelling

orders_sub = orders[["order_id","customer_id","order_status","order_purchase_timestamp","is_valid_for_sales"]].copy()

fact = (items
        .merge(orders_sub, on="order_id", how="left")
        .merge(products[["product_id","product_category_name_english","product_weight_g","product_length_cm","product_height_cm","product_width_cm"]],
               on="product_id", how="left")
        .merge(sell[["seller_id","seller_city","seller_state"]], on="seller_id", how="left")
        .merge(cust[["customer_id","customer_city","customer_state"]], on="customer_id", how="left")
        .merge(pay_agg, on="order_id", how="left")
        .merge(rev_agg, on="order_id", how="left")
       )

# Keep only rows tied to valid sales for demand/revenue (you can relax this if needed)
fact = fact[fact["is_valid_for_sales"].fillna(False)]

# Derive order_date and revenue fields
fact["order_date"] = pd.to_datetime(fact["order_purchase_timestamp"]).dt.date
fact["order_date"] = pd.to_datetime(fact["order_date"])
fact["revenue_item"] = fact["price"].astype(float)
fact["freight"] = fact["freight_value"].astype(float)


# 9) Save cleaned outputs + a small data-quality log

products.to_csv(OUT/"products_clean.csv", index=False)
cust.to_csv(OUT/"customers_clean.csv", index=False)
sell.to_csv(OUT/"sellers_clean.csv", index=False)
pay_agg.to_csv(OUT/"payments_by_order.csv", index=False)
rev_agg.to_csv(OUT/"reviews_by_order.csv", index=False)
fact.to_csv(OUT/"fact_orders_clean.csv", index=False)

log = pd.DataFrame([{
    "dropped_items_price_na": int(dropped_price_na),
    "remaining_fact_rows": int(len(fact)),
    "products_missing_english_category_now": int(products["product_category_name_english"].eq("unknown").sum()),
    "customers_missing_state_after_fill": int(cust["customer_state"].isna().sum()),
    "sellers_missing_state_after_fill": int(sell["seller_state"].isna().sum()),
}])
log.to_csv(OUT/"cleaning_log.csv", index=False)

print("Saved to:", OUT.resolve())

