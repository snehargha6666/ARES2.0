import pandas as pd
import numpy as np

# ==========================================
# ARES 2.0 - MODULE 1
# Disaster Impact Severity Index (DISI)
# ==========================================

FILE_PATH = "data/emdat.xlsx"
OUTPUT_PATH = "data/disaster_severity.csv"

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------

df = pd.read_excel(
    FILE_PATH,
    sheet_name="EM-DAT Data"
)

print("\n========== DATASET ==========")
print("Total disasters:", len(df))


# ------------------------------------------
# 2. SELECT IMPACT INDICATORS
# ------------------------------------------

features = [
    "Total Deaths",
    "No. Affected",
    "No. Injured",
    "Total Damage ('000 US$)"
]

data = df[features].copy()

# Convert everything to numeric
for column in features:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ------------------------------------------
# 3. LOG TRANSFORMATION
# ------------------------------------------

# Disaster impact variables are highly
# right-skewed.
#
# log1p(x) = log(1 + x)
#
# Missing values remain missing.

for column in features:

    data[column] = np.log1p(
        data[column]
    )


# ------------------------------------------
# 4. CONVERT EACH FEATURE TO PERCENTILE
# ------------------------------------------

def percentile_score(series):
    """
    Convert available values into a
    0-1 percentile score.

    Missing values remain NaN.
    """

    return series.rank(
        pct=True,
        method="average"
    )


for column in features:

    data[column + "_Score"] = (
        percentile_score(
            data[column]
        )
    )


# ------------------------------------------
# 5. COUNT AVAILABLE INDICATORS
# ------------------------------------------

score_columns = [
    column + "_Score"
    for column in features
]

data["Available_Indicators"] = (
    data[score_columns]
    .notna()
    .sum(axis=1)
)


# ------------------------------------------
# 6. BASELINE WEIGHTS
# ------------------------------------------

# Equal weighting is our BASELINE.
#
# We are NOT claiming these are the
# scientifically optimal final weights.
#
# Later we will compare this baseline
# against alternative weighting methods.

WEIGHTS = {
    "Total Deaths": 0.30,
    "No. Affected": 0.30,
    "No. Injured": 0.25,
    "Total Damage ('000 US$)": 0.25
}


# ------------------------------------------
# 7. MISSINGNESS-AWARE DISI
# ------------------------------------------

def calculate_disi(row):

    weighted_sum = 0.0
    available_weight = 0.0

    for feature in features:

        score = row[feature + "_Score"]

        if pd.notna(score):

            weight = WEIGHTS[feature]

            weighted_sum += (
                weight * score
            )

            available_weight += weight

    # Require at least 2 indicators
    if row["Available_Indicators"] < 2:
        return np.nan

    # Re-normalize weights of available
    # indicators so missing indicators
    # do not automatically reduce severity.

    if available_weight == 0:
        return np.nan

    return (
        weighted_sum /
        available_weight
    )


data["DISI"] = data.apply(
    calculate_disi,
    axis=1
)


# ------------------------------------------
# 8. CLASSIFY SEVERITY
# ------------------------------------------

# Calculate thresholds only from valid
# DISI values.

valid_disi = data["DISI"].dropna()

q25 = valid_disi.quantile(0.25)
q50 = valid_disi.quantile(0.50)
q75 = valid_disi.quantile(0.75)

print("\n========== DISI THRESHOLDS ==========")

print("25th percentile:", q25)
print("50th percentile:", q50)
print("75th percentile:", q75)


def classify_severity(score):

    if pd.isna(score):
        return np.nan

    if score <= q25:
        return "Low"

    elif score <= q50:
        return "Moderate"

    elif score <= q75:
        return "High"

    else:
        return "Critical"


data["Severity"] = data["DISI"].apply(
    classify_severity
)


# ------------------------------------------
# 9. CREATE FINAL DATASET
# ------------------------------------------

result = df.copy()

result["DISI"] = data["DISI"]
result["Severity"] = data["Severity"]
result["Available_Indicators"] = (
    data["Available_Indicators"]
)


# ------------------------------------------
# 10. DISPLAY RESULTS
# ------------------------------------------

print("\n========== USABLE RECORDS ==========")

usable = result["Severity"].notna().sum()

print("Usable:", usable)
print("Excluded:", len(result) - usable)


print("\n========== SEVERITY DISTRIBUTION ==========")

print(
    result["Severity"]
    .value_counts()
    .sort_index()
)


print("\n========== SEVERITY PERCENTAGE ==========")

print(
    (
        result["Severity"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


print("\n========== SAMPLE RESULTS ==========")

print(
    result[
        [
            "DISI",
            "Severity",
            "Available_Indicators"
        ]
    ].head(10)
)


# ------------------------------------------
# 11. SAVE
# ------------------------------------------

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n========== COMPLETE ==========")

print(
    f"Saved to: {OUTPUT_PATH}"
)