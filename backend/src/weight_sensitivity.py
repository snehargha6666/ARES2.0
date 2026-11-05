import pandas as pd
import numpy as np

# ==========================================
# ARES 2.0 - MODULE 1
# Weight Sensitivity Analysis
# ==========================================

FILE_PATH = "data/emdat.xlsx"

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------

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

# ------------------------------------------
# 2. LOG TRANSFORMATION
# ------------------------------------------

for column in features:
    data[column] = np.log1p(data[column])

# ------------------------------------------
# 3. PERCENTILE NORMALIZATION
# ------------------------------------------

for column in features:
    data[column + "_Score"] = (
        data[column].rank(
            pct=True,
            method="average"
        )
    )

score_columns = [
    column + "_Score"
    for column in features
]

# ------------------------------------------
# 4. REQUIRE AT LEAST 2 INDICATORS
# ------------------------------------------

data["Available_Indicators"] = (
    data[score_columns]
    .notna()
    .sum(axis=1)
)

data = data[
    data["Available_Indicators"] >= 2
].copy()

print("\n========== SENSITIVITY DATA ==========")
print("Usable disasters:", len(data))


# ------------------------------------------
# 5. CANDIDATE WEIGHTING SCHEMES
# ------------------------------------------

weight_schemes = {

    "Equal":
    {
        "Total Deaths": 0.25,
        "No. Affected": 0.25,
        "No. Injured": 0.25,
        "Total Damage ('000 US$)": 0.25
    },

    "Human_Impact":
    {
        "Total Deaths": 0.40,
        "No. Affected": 0.30,
        "No. Injured": 0.20,
        "Total Damage ('000 US$)": 0.10
    },

    "Balanced":
    {
        "Total Deaths": 0.30,
        "No. Affected": 0.30,
        "No. Injured": 0.20,
        "Total Damage ('000 US$)": 0.20
    },

    "Death_Focused":
    {
        "Total Deaths": 0.50,
        "No. Affected": 0.25,
        "No. Injured": 0.15,
        "Total Damage ('000 US$)": 0.10
    }
}


# ------------------------------------------
# 6. CALCULATE DISI
# ------------------------------------------

def calculate_disi(row, weights):

    weighted_sum = 0.0
    available_weight = 0.0

    for feature in features:

        score = row[feature + "_Score"]

        if pd.notna(score):

            weight = weights[feature]

            weighted_sum += (
                weight * score
            )

            available_weight += weight

    if available_weight == 0:
        return np.nan

    return (
        weighted_sum /
        available_weight
    )


# ------------------------------------------
# 7. CALCULATE SCORES
# ------------------------------------------

for scheme_name, weights in weight_schemes.items():

    data[scheme_name] = data.apply(
        lambda row:
        calculate_disi(row, weights),
        axis=1
    )


# ------------------------------------------
# 8. CREATE QUARTILE CLASSES
# ------------------------------------------

for scheme_name in weight_schemes:

    q25 = data[scheme_name].quantile(0.25)
    q50 = data[scheme_name].quantile(0.50)
    q75 = data[scheme_name].quantile(0.75)

    def classify(score):

        if score <= q25:
            return "Low"

        elif score <= q50:
            return "Moderate"

        elif score <= q75:
            return "High"

        else:
            return "Critical"

    data[scheme_name + "_Class"] = (
        data[scheme_name].apply(classify)
    )


# ------------------------------------------
# 9. COMPARE AGAINST EQUAL BASELINE
# ------------------------------------------

baseline = data["Equal_Class"]

print("\n========== CLASSIFICATION STABILITY ==========")

for scheme_name in weight_schemes:

    current = data[
        scheme_name + "_Class"
    ]

    changed = (
        current != baseline
    ).sum()

    percentage = (
        changed /
        len(data)
        * 100
    )

    print(
        f"\n{scheme_name}"
    )

    print(
        "Classifications changed:",
        changed
    )

    print(
        "Percentage changed:",
        f"{percentage:.2f}%"
    )


# ------------------------------------------
# 10. CORRELATION WITH BASELINE
# ------------------------------------------

print(
    "\n========== SCORE CORRELATION =========="
)

for scheme_name in weight_schemes:

    correlation = data[
        "Equal"
    ].corr(
        data[scheme_name]
    )

    print(
        f"{scheme_name}: "
        f"{correlation:.4f}"
    )


# ------------------------------------------
# 11. DISPLAY CLASS DISTRIBUTION
# ------------------------------------------

print(
    "\n========== CLASS DISTRIBUTION =========="
)

for scheme_name in weight_schemes:

    print(
        f"\n--- {scheme_name} ---"
    )

    print(
        data[
            scheme_name + "_Class"
        ].value_counts()
    )


# ------------------------------------------
# 12. SAVE RESULTS
# ------------------------------------------

data.to_csv(
    "data/weight_sensitivity_results.csv",
    index=False
)

print(
    "\nResults saved to:"
    "\ndata/weight_sensitivity_results.csv"
)

print(
    "\n========== ANALYSIS COMPLETE =========="
)