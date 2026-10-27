import pandas as pd

# ==============================
# ARES 2.0 - MODULE 1
# Exploratory Data Analysis
# ==============================

FILE_PATH = "data/emdat.xlsx"

# Load EM-DAT
df = pd.read_excel(FILE_PATH, sheet_name="EM-DAT Data")

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
for i, column in enumerate(df.columns):
    print(f"{i}: {column}")

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()
missing_percentage = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing": missing,
    "Percentage": missing_percentage
})

print(missing_table.sort_values("Percentage", ascending=False))

print("\n========== DISASTER TYPES ==========")
print(df["Disaster Type"].value_counts())

print("\n========== BASIC STATISTICS ==========")
print(df.describe(include="all").T)