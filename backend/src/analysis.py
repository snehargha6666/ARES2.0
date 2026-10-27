import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ARES 2.0 - MODULE 1
# Disaster Impact Analysis
# ==========================================

FILE_PATH = "data/emdat.xlsx"

# Load dataset
df = pd.read_excel(FILE_PATH, sheet_name="EM-DAT Data")

print("\n========== DATASET ==========")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ------------------------------------------
# FEATURES WE ARE INVESTIGATING
# ------------------------------------------

features = [
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Affected",
    "Magnitude",
    "Total Damage ('000 US$)"
]

# ------------------------------------------
# BASIC STATISTICS
# ------------------------------------------

print("\n========== IMPACT STATISTICS ==========")

for feature in features:
    print(f"\n--- {feature} ---")

    series = pd.to_numeric(df[feature], errors="coerce")

    print("Available :", series.notna().sum())
    print("Missing   :", series.isna().sum())

    if series.notna().sum() > 0:
        print("Minimum   :", series.min())
        print("Median    :", series.median())
        print("Mean      :", series.mean())
        print("Maximum   :", series.max())

# ------------------------------------------
# PERCENTILES
# ------------------------------------------

print("\n========== PERCENTILE ANALYSIS ==========")

percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

for feature in features:

    series = pd.to_numeric(df[feature], errors="coerce").dropna()

    if len(series) == 0:
        continue

    print(f"\n--- {feature} ---")

    values = series.quantile(percentiles)

    for percentile, value in values.items():
        print(f"{percentile * 100:.0f}th percentile : {value}")

# ------------------------------------------
# CORRELATION
# ------------------------------------------

print("\n========== CORRELATION MATRIX ==========")

numeric_features = [
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Affected",
    "Magnitude",
    "Total Damage ('000 US$)"
]

correlation = df[numeric_features].corr()

print(correlation.round(3))

# Save correlation matrix
correlation.to_csv("data/correlation_matrix.csv")

# ------------------------------------------
# LOG TRANSFORMATION CHECK
# ------------------------------------------

print("\n========== LOG TRANSFORMATION ==========")

for feature in [
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Affected",
    "Total Damage ('000 US$)"
]:

    series = pd.to_numeric(df[feature], errors="coerce")

    positive_values = series[series > 0]

    if len(positive_values) > 0:

        log_values = np.log1p(positive_values)

        print(f"\n--- {feature} ---")
        print("Original median :", positive_values.median())
        print("Log median      :", log_values.median())
        print("Original max    :", positive_values.max())
        print("Log max         :", log_values.max())

# ------------------------------------------
# HISTOGRAMS
# ------------------------------------------

plot_features = [
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "Total Affected"
]

for feature in plot_features:

    series = pd.to_numeric(df[feature], errors="coerce").dropna()

    if len(series) == 0:
        continue

    plt.figure(figsize=(8, 5))

    plt.hist(series, bins=50)

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of disasters")

    plt.tight_layout()
    plt.show()

print("\n========== ANALYSIS COMPLETE ==========")