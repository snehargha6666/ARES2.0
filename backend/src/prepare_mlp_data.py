import os
import pandas as pd
import numpy as np
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# ARES 2.0 - MODULE 1
# MLP DATA PREPARATION
# ==========================================

EMDAT_PATH = "data/emdat.xlsx"
SEVERITY_PATH = "data/disaster_severity.csv"

# ------------------------------------------
# 1. FEATURES
# ------------------------------------------

CATEGORICAL_FEATURES = [
    "Disaster Subgroup",
    "Disaster Type",
    "Disaster Subtype",
    "ISO",
    "Subregion",
    "Magnitude Scale"
]

NUMERICAL_FEATURES = [
    "Start Year",
    "Start Month",
    "Start Day",
    "End Year",
    "End Month",
    "End Day",
    "CPI",
"Total Deaths",
    "No. Affected",
    "No. Injured",
    "Total Damage ('000 US$)"
]

FEATURES = (
    CATEGORICAL_FEATURES +
    NUMERICAL_FEATURES
)

TARGET = "Severity"


# ------------------------------------------
# 2. LOAD DATA
# ------------------------------------------

print("\n========== LOADING DATA ==========")

df = pd.read_excel(
    EMDAT_PATH,
    sheet_name="EM-DAT Data"
)

severity = pd.read_csv(
    SEVERITY_PATH
)

print("EM-DAT records:", len(df))
print("Severity records:", len(severity))


# ------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ------------------------------------------

required_columns = FEATURES

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns in EM-DAT: "
        + str(missing_columns)
    )


if TARGET not in severity.columns:

    raise ValueError(
        "Severity column not found."
    )


# ------------------------------------------
# 4. MATCH RECORDS
# ------------------------------------------

# disaster_severity.csv contains the severity
# results in the same original row order.

if len(df) != len(severity):

    raise ValueError(
        "EM-DAT and severity datasets "
        "have different numbers of rows."
    )

df[TARGET] = severity[TARGET]


# ------------------------------------------
# 5. KEEP ONLY USABLE TARGET RECORDS
# ------------------------------------------

df = df[
    df[TARGET].notna()
].copy()

print(
    "\nUsable records:",
    len(df)
)


# ------------------------------------------
# 6. CREATE X AND y
# ------------------------------------------

X = df[FEATURES].copy()

y = df[TARGET].copy()


print(
    "\n========== TARGET DISTRIBUTION =========="
)

print(
    y.value_counts()
)


# ------------------------------------------
# 7. TRAIN / VALIDATION / TEST SPLIT
# ------------------------------------------

# First:
# 70% training
# 30% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Split temporary 50/50:
# 15% validation
# 15% test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print(
    "\n========== DATA SPLIT =========="
)

print(
    "Training   :",
    len(X_train)
)

print(
    "Validation :",
    len(X_val)
)

print(
    "Testing    :",
    len(X_test)
)


# ------------------------------------------
# 8. NUMERICAL PIPELINE
# ------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ------------------------------------------
# 9. CATEGORICAL PIPELINE
# ------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ------------------------------------------
# 10. COMBINED PREPROCESSOR
# ------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES
        ),

        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)


# ------------------------------------------
# 11. FIT ONLY ON TRAINING DATA
# ------------------------------------------

print(
    "\n========== PREPROCESSING =========="
)

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_val_processed = preprocessor.transform(
    X_val
)

X_test_processed = preprocessor.transform(
    X_test
)

# Save the fitted preprocessor for API predictions
joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print(
    "Saved preprocessor: models/preprocessor.pkl"
)


print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed validation shape:",
    X_val_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ------------------------------------------
# 12. ENCODE TARGET
# ------------------------------------------

target_mapping = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Critical": 3
}

y_train_encoded = y_train.map(
    target_mapping
).to_numpy()

y_val_encoded = y_val.map(
    target_mapping
).to_numpy()

y_test_encoded = y_test.map(
    target_mapping
).to_numpy()


# ------------------------------------------
# 13. SAVE PROCESSED DATA
# ------------------------------------------

np.save(
    "data/X_train.npy",
    X_train_processed
)

np.save(
    "data/X_val.npy",
    X_val_processed
)

np.save(
    "data/X_test.npy",
    X_test_processed
)

np.save(
    "data/y_train.npy",
    y_train_encoded
)

np.save(
    "data/y_val.npy",
    y_val_encoded
)

np.save(
    "data/y_test.npy",
    y_test_encoded
)


# ------------------------------------------
# 14. SAVE FEATURE NAMES
# ------------------------------------------

feature_names = (
    preprocessor
    .get_feature_names_out()
)

pd.DataFrame({
    "Feature": feature_names
}).to_csv(
    "data/mlp_features.csv",
    index=False
)


# ------------------------------------------
# 15. SAVE TARGET MAPPING
# ------------------------------------------

pd.DataFrame({
    "Severity": [
        "Low",
        "Moderate",
        "High",
        "Critical"
    ],
    "Encoded_Value": [
        0,
        1,
        2,
        3
    ]
}).to_csv(
    "data/target_mapping.csv",
    index=False
)


# ------------------------------------------
# 16. FINAL REPORT
# ------------------------------------------

print(
    "\n========== TARGET ENCODING =========="
)

print(
    "Low       -> 0"
)

print(
    "Moderate  -> 1"
)

print(
    "High      -> 2"
)

print(
    "Critical  -> 3"
)


print(
    "\n========== FILES CREATED =========="
)

print("data/X_train.npy")
print("data/X_val.npy")
print("data/X_test.npy")
print("data/y_train.npy")
print("data/y_val.npy")
print("data/y_test.npy")
print("data/mlp_features.csv")
print("data/target_mapping.csv")

print(
    "\n========== MLP DATA PREPARATION COMPLETE =========="
)