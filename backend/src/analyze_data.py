import os
import numpy as np
import pandas as pd


# ============================================================
# ARES 2.0 - DATA & FEATURE DIAGNOSTICS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
X_VAL_PATH = os.path.join(DATA_DIR, "X_val.npy")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")

Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.npy")
Y_VAL_PATH = os.path.join(DATA_DIR, "y_val.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.npy")


CLASS_NAMES = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Critical"
}


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("ARES 2.0 - DATA & FEATURE DIAGNOSTICS")
print("=" * 60)

X_train = np.load(X_TRAIN_PATH)
X_val = np.load(X_VAL_PATH)
X_test = np.load(X_TEST_PATH)

y_train = np.load(Y_TRAIN_PATH)
y_val = np.load(Y_VAL_PATH)
y_test = np.load(Y_TEST_PATH)


print("\n========== DATASET SHAPES ==========")

print(f"X_train : {X_train.shape}")
print(f"X_val   : {X_val.shape}")
print(f"X_test  : {X_test.shape}")

print(f"y_train : {y_train.shape}")
print(f"y_val   : {y_val.shape}")
print(f"y_test  : {y_test.shape}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n========== CLASS DISTRIBUTION ==========")

for name, labels in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Testing", y_test)
]:

    print(f"\n{name}")

    values, counts = np.unique(
        labels,
        return_counts=True
    )

    for value, count in zip(values, counts):

        percentage = (
            count / len(labels)
        ) * 100

        print(
            f"{CLASS_NAMES.get(int(value), value):<10}"
            f": {count:4d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# DATA TYPES
# ============================================================

print("\n========== DATA TYPES ==========")

print("X_train dtype :", X_train.dtype)
print("y_train dtype :", y_train.dtype)


# ============================================================
# MISSING / NAN / INF VALUES
# ============================================================

print("\n========== MISSING / INVALID VALUES ==========")

print(
    "NaN values :",
    np.isnan(X_train).sum()
)

print(
    "Inf values :",
    np.isinf(X_train).sum()
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n========== FEATURE STATISTICS ==========")

feature_mean = np.mean(X_train, axis=0)
feature_std = np.std(X_train, axis=0)
feature_min = np.min(X_train, axis=0)
feature_max = np.max(X_train, axis=0)


print(
    f"Global minimum : {np.min(X_train):.6f}"
)

print(
    f"Global maximum : {np.max(X_train):.6f}"
)

print(
    f"Global mean    : {np.mean(X_train):.6f}"
)

print(
    f"Global std     : {np.std(X_train):.6f}"
)


# ============================================================
# CONSTANT FEATURES
# ============================================================

constant_features = np.where(
    feature_std == 0
)[0]

print("\n========== CONSTANT FEATURES ==========")

print(
    "Number of constant features :",
    len(constant_features)
)

if len(constant_features) > 0:

    print(
        "Feature indices:",
        constant_features.tolist()
    )


# ============================================================
# NEAR-CONSTANT FEATURES
# ============================================================

print("\n========== NEAR-CONSTANT FEATURES ==========")

near_constant = []

for i in range(X_train.shape[1]):

    unique_values = np.unique(
        X_train[:, i]
    )

    if len(unique_values) <= 2:

        near_constant.append(i)


print(
    "Binary / near-binary features :",
    len(near_constant)
)

if near_constant:

    print(
        "Feature indices:",
        near_constant
    )


# ============================================================
# FEATURE SCALE CHECK
# ============================================================

print("\n========== FEATURE SCALE CHECK ==========")

large_scale = []
small_scale = []

for i, std in enumerate(feature_std):

    if std > 10:
        large_scale.append(i)

    if 0 < std < 0.01:
        small_scale.append(i)


print(
    "Features with std > 10 :",
    len(large_scale)
)

print(
    "Features with std < 0.01:",
    len(small_scale)
)


# ============================================================
# DUPLICATE SAMPLES
# ============================================================

print("\n========== DUPLICATE SAMPLE CHECK ==========")

unique_train = np.unique(
    X_train,
    axis=0
).shape[0]

duplicate_train = (
    len(X_train) - unique_train
)

print(
    "Training samples          :",
    len(X_train)
)

print(
    "Unique training samples    :",
    unique_train
)

print(
    "Duplicate training samples:",
    duplicate_train
)


# ============================================================
# TRAIN / VALIDATION OVERLAP
# ============================================================

print("\n========== TRAIN / VALIDATION OVERLAP ==========")

train_set = {
    tuple(row)
    for row in X_train
}

val_set = {
    tuple(row)
    for row in X_val
}

test_set = {
    tuple(row)
    for row in X_test
}

train_val_overlap = len(
    train_set.intersection(val_set)
)

train_test_overlap = len(
    train_set.intersection(test_set)
)

val_test_overlap = len(
    val_set.intersection(test_set)
)

print(
    "Train ↔ Validation overlap:",
    train_val_overlap
)

print(
    "Train ↔ Test overlap      :",
    train_test_overlap
)

print(
    "Validation ↔ Test overlap :",
    val_test_overlap
)


# ============================================================
# FEATURE VARIANCE
# ============================================================

print("\n========== LOW-VARIANCE FEATURES ==========")

variance = np.var(
    X_train,
    axis=0
)

lowest_indices = np.argsort(
    variance
)[:20]

print(
    "20 lowest-variance feature indices:"
)

for index in lowest_indices:

    print(
        f"Feature {index:3d} "
        f"variance = {variance[index]:.8f}"
    )


# ============================================================
# SAVE FEATURE REPORT
# ============================================================

feature_report = pd.DataFrame({

    "feature_index":
        np.arange(X_train.shape[1]),

    "mean":
        feature_mean,

    "std":
        feature_std,

    "min":
        feature_min,

    "max":
        feature_max,

    "variance":
        variance

})


REPORT_PATH = os.path.join(
    DATA_DIR,
    "feature_diagnostics.csv"
)

feature_report.to_csv(
    REPORT_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== DIAGNOSTIC SUMMARY ==========")

print(
    f"Total features       : {X_train.shape[1]}"
)

print(
    f"Training samples     : {X_train.shape[0]}"
)

print(
    f"Validation samples   : {X_val.shape[0]}"
)

print(
    f"Testing samples      : {X_test.shape[0]}"
)

print(
    f"Constant features    : {len(constant_features)}"
)

print(
    f"Duplicate train rows : {duplicate_train}"
)

print(
    f"Train-Val overlap    : {train_val_overlap}"
)

print(
    f"Train-Test overlap   : {train_test_overlap}"
)

print(
    f"Val-Test overlap     : {val_test_overlap}"
)

print("\nFeature report saved to:")
print(REPORT_PATH)

print("\n" + "=" * 60)
print("DATA DIAGNOSTICS COMPLETE")
print("=" * 60)