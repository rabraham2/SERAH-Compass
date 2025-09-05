# Make margins template
import pandas as pd
from pathlib import Path

OUT = Path("Dataset/model_outputs")
wk  = pd.read_csv(OUT/"weekly_by_category.csv")
cats = sorted(wk["product_category_name_english"].unique())
pd.DataFrame({"category": cats, "margin_rate": 0.30}).to_csv(OUT/"margins.csv", index=False)
print("Edit margin rates here:", OUT/"margins.csv")
