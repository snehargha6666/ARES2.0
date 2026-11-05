import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# ARES 2.0 - MODULE 1
# PCA-Based Impact Analysis
# ==========================================

FILE_PATH = "data/emdat.xlsx"

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------

df = pd.read_excel(
    FILE_PATH,
    sheet_name="EM-DAT Data"
)

print("\n========== DATASET ==========")
print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ------------------------------------------
# 2. SELECT IMPACT VARIABLES
# ------------------------------------------

features = [
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Damage ('000 US$)"
]

data = df[features].copy()


# ------------------------------------------
# 3. CONVERT TO NUMERIC
# ------------------------------------------

for column in features:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ------------------------------------------
# 4. LOG TRANSFORMATION
# ------------------------------------------

# Disaster-impact variables are heavily
# right-skewed, so reduce the effect of
# extreme values using log1p.

for column in features:
    data[column] = np.log1p(data[column])


# ------------------------------------------
# 5. REMOVE ROWS WITH MISSING VALUES
# ------------------------------------------

pca_data = data.dropna().copy()

print("\n========== PCA DATA ==========")
print("Original records :", len(data))
print("Records for PCA  :", len(pca_data))
print("Removed records  :", len(data) - len(pca_data))


# ------------------------------------------
# 6. STANDARDIZATION
# ------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(pca_data)

print("\n========== STANDARDIZATION ==========")
print("Mean approximately :", X_scaled.mean(axis=0))
print("Std approximately  :", X_scaled.std(axis=0))


# ------------------------------------------
# 7. PCA
# ------------------------------------------

pca = PCA()

X_pca = pca.fit_transform(X_scaled)


# ------------------------------------------
# 8. EXPLAINED VARIANCE
# ------------------------------------------

print("\n========== EXPLAINED VARIANCE ==========")

for i, variance in enumerate(
    pca.explained_variance_ratio_
):
    print(
        f"PC{i + 1}: "
        f"{variance:.4f} "
        f"({variance * 100:.2f}%)"
    )


print(
    "\nCumulative variance:"
)

cumulative = np.cumsum(
    pca.explained_variance_ratio_
)

for i, value in enumerate(cumulative):
    print(
        f"PC{i + 1}: "
        f"{value:.4f} "
        f"({value * 100:.2f}%)"
    )


# ------------------------------------------
# 9. PCA LOADINGS
# ------------------------------------------

loadings = pd.DataFrame(
    pca.components_,
    columns=features,
    index=[
        f"PC{i + 1}"
        for i in range(len(features))
    ]
)

print("\n========== PCA LOADINGS ==========")
print(loadings.round(4))


# ------------------------------------------
# 10. FIRST PRINCIPAL COMPONENT
# ------------------------------------------

pc1 = loadings.iloc[0]

print("\n========== PC1 LOADINGS ==========")

for feature, loading in pc1.items():
    print(
        f"{feature}: {loading:.4f}"
    )


# ------------------------------------------
# 11. ABSOLUTE PC1 CONTRIBUTION
# ------------------------------------------

absolute_loadings = pc1.abs()

pc1_weights = (
    absolute_loadings /
    absolute_loadings.sum()
)

print("\n========== CANDIDATE PCA WEIGHTS ==========")

for feature, weight in pc1_weights.items():
    print(
        f"{feature}: "
        f"{weight:.4f} "
        f"({weight * 100:.2f}%)"
    )


# ------------------------------------------
# 12. SAVE RESULTS
# ------------------------------------------

results = pd.DataFrame({
    "Feature": features,
    "PC1_Loading": pc1.values,
    "Absolute_Loading": absolute_loadings.values,
    "Candidate_Weight": pc1_weights.values
})

results.to_csv(
    "data/pca_weights.csv",
    index=False
)

print(
    "\nPCA results saved to:"
    "\ndata/pca_weights.csv"
)

print("\n========== PCA ANALYSIS COMPLETE ==========")