import pandas as pd
import numpy as np

# ==========================================
# ARES 2.0 - MODULE 1
# Feature Analysis for MLP
# ==========================================

FILE_PATH = "data/emdat.xlsx"

# ------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------

df = pd.read_excel(
    FILE_PATH,
    sheet_name="EM-DAT Data"
)

print("\n========== DATASET OVERVIEW ==========")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ------------------------------------------
# 2. DISPLAY ALL COLUMNS
# ------------------------------------------

print("\n========== ALL COLUMNS ==========")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")


# ------------------------------------------
# 3. DATA TYPE + MISSINGNESS
# ------------------------------------------

analysis = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [
        str(df[column].dtype)
        for column in df.columns
    ],
    "Missing_Count": [
        df[column].isna().sum()
        for column in df.columns
    ],
    "Missing_Percentage": [
        df[column].isna().mean() * 100
        for column in df.columns
    ],
    "Unique_Values": [
        df[column].nunique(dropna=True)
        for column in df.columns
    ]
})


print("\n========== FEATURE INFORMATION ==========")

print(
    analysis.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ------------------------------------------
# 4. POTENTIAL TARGET-CONSTRUCTION FEATURES
# ------------------------------------------

target_related = [
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Affected",
    "Total Damage ('000 US$)"
]

print(
    "\n========== TARGET-RELATED FEATURES =========="
)

for column in target_related:

    if column in df.columns:

        missing = df[column].isna().sum()
        missing_pct = (
            df[column].isna().mean() * 100
        )

        print(
            f"{column}: "
            f"missing={missing} "
            f"({missing_pct:.2f}%)"
        )


# ------------------------------------------
# 5. POSSIBLE PREDICTION FEATURES
# ------------------------------------------

print(
    "\n========== POSSIBLE PREDICTION FEATURES =========="
)

exclude_keywords = [
    "death",
    "injur",
    "affected",
    "homeless",
    "damage"
]

for column in df.columns:

    column_lower = column.lower()

    if not any(
        keyword in column_lower
        for keyword in exclude_keywords
    ):

        print(column)


# ------------------------------------------
# 6. CATEGORICAL FEATURES
# ------------------------------------------

print(
    "\n========== CATEGORICAL FEATURES =========="
)

categorical = df.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical:

    print(
        f"{column}: "
        f"{df[column].nunique(dropna=True)} unique values"
    )


# ------------------------------------------
# 7. NUMERICAL FEATURES
# ------------------------------------------

print(
    "\n========== NUMERICAL FEATURES =========="
)

numerical = df.select_dtypes(
    include=np.number
).columns

for column in numerical:

    print(
        f"{column}: "
        f"min={df[column].min()} | "
        f"max={df[column].max()} | "
        f"missing={df[column].isna().sum()}"
    )


# ------------------------------------------
# 8. HIGH-MISSINGNESS FEATURES
# ------------------------------------------

print(
    "\n========== HIGH MISSINGNESS (>50%) =========="
)

high_missing = analysis[
    analysis["Missing_Percentage"] > 50
]

if len(high_missing) == 0:

    print("No features exceed 50% missingness.")

else:

    print(
        high_missing[
            [
                "Column",
                "Missing_Percentage"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )


# ------------------------------------------
# 9. LOW-CARDINALITY CATEGORICAL FEATURES
# ------------------------------------------

print(
    "\n========== LOW-CARDINALITY CATEGORICAL =========="
)

for column in categorical:

    unique_count = (
        df[column]
        .nunique(dropna=True)
    )

    if unique_count <= 30:

        print(
            f"\n{column}"
        )

        print(
            df[column]
            .dropna()
            .unique()
        )


# ------------------------------------------
# 10. SAVE ANALYSIS
# ------------------------------------------

analysis.to_csv(
    "data/feature_analysis.csv",
    index=False
)

print(
    "\nAnalysis saved to:"
)

print(
    "data/feature_analysis.csv"
)

print(
    "\n========== FEATURE ANALYSIS COMPLETE =========="
)