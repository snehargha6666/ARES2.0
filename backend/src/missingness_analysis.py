import pandas as pd
import numpy as np

# ==========================================
# ARES 2.0 - MODULE 1
# Missingness-Aware Impact Analysis
# ==========================================

FILE_PATH = "data/emdat.xlsx"

df = pd.read_excel(
    FILE_PATH,
    sheet_name="EM-DAT Data"
)

features = [
    "Total Deaths",
    "No. Affected",
    "No. Injured",
    "Total Damage ('000 US$)"
]

data = df[features].copy()

for column in features:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

print("\n========== MISSINGNESS ANALYSIS ==========")

for column in features:
    missing = data[column].isna().sum()
    available = data[column].notna().sum()

    print(f"\n{column}")
    print(f"Available : {available}")
    print(f"Missing   : {missing}")
    print(f"Missing % : {(missing / len(data)) * 100:.2f}%")

# Number of available impact indicators
data["Available_Indicators"] = data.notna().sum(axis=1)

print("\n========== INDICATORS AVAILABLE PER DISASTER ==========")

print(
    data["Available_Indicators"]
    .value_counts()
    .sort_index()
)

print("\n========== PERCENTAGE ==========")

percentage = (
    data["Available_Indicators"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print(percentage.round(2))

# Distribution of usable records
print("\n========== USABLE RECORDS ==========")

for minimum in range(1, 5):
    count = (
        data["Available_Indicators"] >= minimum
    ).sum()

    percentage_value = (
        count / len(data)
    ) * 100

    print(
        f"At least {minimum} indicator(s): "
        f"{count} records "
        f"({percentage_value:.2f}%)"
    )

print("\n========== ANALYSIS COMPLETE ==========")