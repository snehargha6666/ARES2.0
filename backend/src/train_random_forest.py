import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# ARES 2.0
# RANDOM FOREST SEVERITY CLASSIFIER
# ==========================================

print("\n" + "=" * 60)
print("ARES 2.0 - RANDOM FOREST TRAINING")
print("=" * 60)


# ------------------------------------------
# 1. LOAD PREPARED DATA
# ------------------------------------------

print("\n========== LOADING DATA ==========")

X_train = np.load("data/X_train.npy")
X_val = np.load("data/X_val.npy")
X_test = np.load("data/X_test.npy")

y_train = np.load("data/y_train.npy")
y_val = np.load("data/y_val.npy")
y_test = np.load("data/y_test.npy")

print("Training shape   :", X_train.shape)
print("Validation shape :", X_val.shape)
print("Testing shape    :", X_test.shape)


# ------------------------------------------
# 2. MODEL CONFIGURATION
# ------------------------------------------

print("\n========== MODEL CONFIGURATION ==========")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

print("Estimators        : 300")
print("Max features      : sqrt")
print("Random state      : 42")


# ------------------------------------------
# 3. TRAIN
# ------------------------------------------

print("\n========== TRAINING ==========")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ------------------------------------------
# 4. VALIDATION
# ------------------------------------------

print("\n========== VALIDATION ==========")

val_predictions = model.predict(X_val)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision = precision_score(
    y_val,
    val_predictions,
    average="weighted",
    zero_division=0
)

val_recall = recall_score(
    y_val,
    val_predictions,
    average="weighted",
    zero_division=0
)

val_f1 = f1_score(
    y_val,
    val_predictions,
    average="weighted",
    zero_division=0
)

print("Validation Accuracy  :", round(val_accuracy, 4))
print("Validation Precision :", round(val_precision, 4))
print("Validation Recall    :", round(val_recall, 4))
print("Validation F1        :", round(val_f1, 4))


# ------------------------------------------
# 5. TEST
# ------------------------------------------

print("\n========== TESTING ==========")

test_predictions = model.predict(X_test)

test_probabilities = model.predict_proba(
    X_test
)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)


print("\n========== FINAL TEST METRICS ==========")

print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1 Score  :", round(f1, 4))


# ------------------------------------------
# 6. CLASSIFICATION REPORT
# ------------------------------------------

class_names = [
    "Low",
    "Moderate",
    "High",
    "Critical"
]

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# ------------------------------------------
# 7. CONFUSION MATRIX
# ------------------------------------------

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\n========== CONFUSION MATRIX ==========")

print(
    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )
)


# ------------------------------------------
# 8. PREDICTION DISTRIBUTION
# ------------------------------------------

print("\n========== PREDICTION DISTRIBUTION ==========")

unique, counts = np.unique(
    test_predictions,
    return_counts=True
)

for class_id, count in zip(
    unique,
    counts
):

    percentage = (
        count / len(test_predictions)
    ) * 100

    print(
        f"{class_names[class_id]:<10}: "
        f"{count:4d} "
        f"({percentage:6.2f}%)"
    )


# ------------------------------------------
# 9. SAVE MODEL
# ------------------------------------------

import joblib

MODEL_PATH = (
    "models/"
    "ares_random_forest.pkl"
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved to:")
print(MODEL_PATH)


# ------------------------------------------
# 10. SAVE PREDICTIONS
# ------------------------------------------

prediction_df = pd.DataFrame({

    "Actual_Severity": [
        class_names[x]
        for x in y_test
    ],

    "Predicted_Severity": [
        class_names[x]
        for x in test_predictions
    ],

    "Confidence": (
        np.max(
            test_probabilities,
            axis=1
        ) * 100
    )
})

prediction_df.to_csv(
    "data/random_forest_predictions.csv",
    index=False
)

print("\nPredictions saved to:")
print(
    "data/random_forest_predictions.csv"
)


# ------------------------------------------
# 11. FINAL
# ------------------------------------------

print(
    "\n========== RANDOM FOREST COMPLETE =========="
)