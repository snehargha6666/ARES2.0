import pandas as pd
import numpy as np

# ==========================================
# ARES 2.0 - MODULE 1
# Weight Robustness / Perturbation Analysis
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
    data[column] = np.log1p(
        data[column]
    )


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

print("\n========== ROBUSTNESS DATA ==========")
print(
    "Usable disasters:",
    len(data)
)


# ------------------------------------------
# 5. BASELINE WEIGHTS
# ------------------------------------------

BASELINE_WEIGHTS = {
    "Total Deaths": 0.30,
    "No. Affected": 0.30,
    "No. Injured": 0.20,
    "Total Damage ('000 US$)": 0.20
}


# ------------------------------------------
# 6. PERTURBED WEIGHT SCHEMES
# ------------------------------------------

weight_schemes = {

    # Baseline
    "Baseline_30_30_20_20": {
        "Total Deaths": 0.30,
        "No. Affected": 0.30,
        "No. Injured": 0.20,
        "Total Damage ('000 US$)": 0.20
    },

    # Deaths +1%, Affected -1%
    "Deaths_Plus_1": {
        "Total Deaths": 0.31,
        "No. Affected": 0.29,
        "No. Injured": 0.20,
        "Total Damage ('000 US$)": 0.20
    },

    # Deaths -1%, Affected +1%
    "Deaths_Minus_1": {
        "Total Deaths": 0.29,
        "No. Affected": 0.31,
        "No. Injured": 0.20,
        "Total Damage ('000 US$)": 0.20
    },

    # Injured +1%, Damage -1%
    "Injured_Plus_1": {
        "Total Deaths": 0.30,
        "No. Affected": 0.30,
        "No. Injured": 0.21,
        "Total Damage ('000 US$)": 0.19
    },

    # Injured -1%, Damage +1%
    "Injured_Minus_1": {
        "Total Deaths": 0.30,
        "No. Affected": 0.30,
        "No. Injured": 0.19,
        "Total Damage ('000 US$)": 0.21
    },

    # Deaths +2%, Affected -2%
    "Deaths_Plus_2": {
        "Total Deaths": 0.32,
        "No. Affected": 0.28,
        "No. Injured": 0.20,
        "Total Damage ('000 US$)": 0.20
    },

    # Deaths -2%, Affected +2%
    "Deaths_Minus_2": {
        "Total Deaths": 0.28,
        "No. Affected": 0.32,
        "No. Injured": 0.20,
        "Total Damage ('000 US$)": 0.20
    },

    # Human impact +2%, Damage -2%
    "Human_Plus_2": {
        "Total Deaths": 0.31,
        "No. Affected": 0.31,
        "No. Injured": 0.20,
        "Total Damage ('000 US$)": 0.18
    },

    # Human impact -2%, Damage +2%
    "Human_Minus_2": {
        "Total Deaths": 0.29,
        "No. Affected": 0.29,
        "No. Injured": 0.20,
        "Total Damage ('000 US$)": 0.22
    }
}


# ------------------------------------------
# 7. DISI CALCULATION FUNCTION
# ------------------------------------------

def calculate_disi(row, weights):

    weighted_sum = 0.0
    available_weight = 0.0

    for feature in features:

        score = row[
            feature + "_Score"
        ]

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
# 8. CALCULATE BASELINE
# ------------------------------------------

data["Baseline_DISI"] = data.apply(
    lambda row:
    calculate_disi(
        row,
        BASELINE_WEIGHTS
    ),
    axis=1
)


# ------------------------------------------
# 9. BASELINE QUARTILES
# ------------------------------------------

baseline_q25 = (
    data["Baseline_DISI"]
    .quantile(0.25)
)

baseline_q50 = (
    data["Baseline_DISI"]
    .quantile(0.50)
)

baseline_q75 = (
    data["Baseline_DISI"]
    .quantile(0.75)
)


def classify_baseline(score):

    if score <= baseline_q25:
        return "Low"

    elif score <= baseline_q50:
        return "Moderate"

    elif score <= baseline_q75:
        return "High"

    else:
        return "Critical"


data["Baseline_Class"] = (
    data["Baseline_DISI"]
    .apply(classify_baseline)
)


# ------------------------------------------
# 10. TEST PERTURBATIONS
# ------------------------------------------

results = []

print(
    "\n========== WEIGHT ROBUSTNESS =========="
)

for scheme_name, weights in weight_schemes.items():

    data["Test_DISI"] = data.apply(
        lambda row:
        calculate_disi(
            row,
            weights
        ),
        axis=1
    )

    # Use the SAME baseline thresholds
    # so we test whether the weighting
    # itself changes classifications.

    def classify_test(score):

        if score <= baseline_q25:
            return "Low"

        elif score <= baseline_q50:
            return "Moderate"

        elif score <= baseline_q75:
            return "High"

        else:
            return "Critical"

    data["Test_Class"] = (
        data["Test_DISI"]
        .apply(classify_test)
    )

    # Score difference
    score_difference = (
        data["Test_DISI"]
        -
        data["Baseline_DISI"]
    )

    mean_absolute_change = (
        score_difference
        .abs()
        .mean()
    )

    maximum_change = (
        score_difference
        .abs()
        .max()
    )

    # Classification changes
    changed = (
        data["Test_Class"]
        !=
        data["Baseline_Class"]
    ).sum()

    changed_percentage = (
        changed /
        len(data)
        * 100
    )

    # Correlation
    correlation = (
        data["Baseline_DISI"]
        .corr(
            data["Test_DISI"]
        )
    )

    results.append({
        "Scheme": scheme_name,
        "Changed_Classes": changed,
        "Changed_Percentage":
            changed_percentage,
        "Mean_Absolute_DISI_Change":
            mean_absolute_change,
        "Maximum_DISI_Change":
            maximum_change,
        "Correlation_With_Baseline":
            correlation
    })

    print(
        f"\n{scheme_name}"
    )

    print(
        "Changed classifications:",
        changed
    )

    print(
        "Changed percentage:",
        f"{changed_percentage:.2f}%"
    )

    print(
        "Mean absolute DISI change:",
        f"{mean_absolute_change:.6f}"
    )

    print(
        "Maximum DISI change:",
        f"{maximum_change:.6f}"
    )

    print(
        "Correlation:",
        f"{correlation:.6f}"
    )


# ------------------------------------------
# 11. RESULTS TABLE
# ------------------------------------------

results_df = pd.DataFrame(results)

print(
    "\n========== ROBUSTNESS SUMMARY =========="
)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}"
    )
)


# ------------------------------------------
# 12. SAVE RESULTS
# ------------------------------------------

results_df.to_csv(
    "data/weight_robustness_results.csv",
    index=False
)

print(
    "\nResults saved to:"
)

print(
    "data/weight_robustness_results.csv"
)

print(
    "\n========== ANALYSIS COMPLETE =========="
)