import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ============================================================
# ARES 2.0 - RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

MODEL_PATH = "models/ares_random_forest.pkl"
FEATURE_PATH = "data/mlp_features.csv"

OUTPUT_CSV = "data/random_forest_feature_importance.csv"
OUTPUT_PLOT = "data/random_forest_feature_importance.png"

TOP_N = 25


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n" + "=" * 60)
print("ARES 2.0 - RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Random Forest model not found: {MODEL_PATH}"
    )

if not os.path.exists(FEATURE_PATH):
    raise FileNotFoundError(
        f"Feature file not found: {FEATURE_PATH}"
    )

print("\n========== FILE VALIDATION ==========")

print(
    f"Random Forest model exists"
    f"                         : PASS"
)

print(
    f"Feature names file exists"
    f"                          : PASS"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\n========== LOADING MODEL ==========")

model = joblib.load(MODEL_PATH)

print("Random Forest loaded successfully.")

print(
    "Number of estimators :",
    model.n_estimators
)

print(
    "Number of features   :",
    model.n_features_in_
)


# ============================================================
# 3. LOAD FEATURE NAMES
# ============================================================

print("\n========== LOADING FEATURES ==========")

feature_df = pd.read_csv(
    FEATURE_PATH
)

feature_names = (
    feature_df["Feature"]
    .tolist()
)

print(
    "Feature names loaded :",
    len(feature_names)
)


# ============================================================
# 4. VALIDATE FEATURE COUNT
# ============================================================

if len(feature_names) != model.n_features_in_:

    raise ValueError(
        "Feature count mismatch!\n"
        f"Model expects {model.n_features_in_} features, "
        f"but feature file contains {len(feature_names)}."
    )

print(
    "Feature count matches model"
    "                         : PASS"
)


# ============================================================
# 5. EXTRACT FEATURE IMPORTANCE
# ============================================================

print("\n========== CALCULATING IMPORTANCE ==========")

importance = model.feature_importances_

if len(importance) != len(feature_names):

    raise ValueError(
        "Feature importance length does not "
        "match feature names."
    )


importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})


# ============================================================
# 6. SORT FEATURES
# ============================================================

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

importance_df["Rank"] = (
    importance_df.index + 1
)


# Reorder columns

importance_df = importance_df[
    [
        "Rank",
        "Feature",
        "Importance"
    ]
]


# ============================================================
# 7. DISPLAY TOP FEATURES
# ============================================================

print("\n========== TOP FEATURE IMPORTANCE ==========")

print(
    importance_df
    .head(TOP_N)
    .to_string(index=False)
)


# ============================================================
# 8. SAVE COMPLETE RESULTS
# ============================================================

importance_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n========== FILE SAVED ==========")

print(
    "Feature importance saved to:"
)

print(
    OUTPUT_CSV
)


# ============================================================
# 9. TOP FEATURES PLOT
# ============================================================

print("\n========== GENERATING PLOT ==========")

top_features = (
    importance_df
    .head(TOP_N)
    .sort_values(
        by="Importance",
        ascending=True
    )
)

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "ARES 2.0 - Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=300
)

plt.close()


print(
    "Feature importance plot saved to:"
)

print(
    OUTPUT_PLOT
)


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE COMPLETE")
print("=" * 60)

print(
    "\nTotal features analyzed :",
    len(feature_names)
)

print(
    "Top features displayed  :",
    TOP_N
)

print(
    "\nOutput files:"
)

print(
    OUTPUT_CSV
)

print(
    OUTPUT_PLOT
)

print("\n" + "=" * 60)