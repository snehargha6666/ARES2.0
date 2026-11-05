import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# ARES 2.0 - MODULE 2
# FINAL MODEL EVALUATION
# ============================================================

print("\n==========================================")
print("ARES 2.0 - FINAL MODEL EVALUATION")
print("==========================================\n")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ares_mlp_final.keras"
)


# ============================================================
# DATA FILES
# ============================================================

X_TEST_PATH = os.path.join(
    DATA_DIR,
    "X_test.npy"
)

Y_TEST_PATH = os.path.join(
    DATA_DIR,
    "y_test.npy"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Low",
    "Moderate",
    "High",
    "Critical"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("========== LOADING FINAL MODEL ==========")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n========== LOADING TEST DATA ==========")

if not os.path.exists(X_TEST_PATH):
    raise FileNotFoundError(
        f"Test features not found: {X_TEST_PATH}"
    )

if not os.path.exists(Y_TEST_PATH):
    raise FileNotFoundError(
        f"Test labels not found: {Y_TEST_PATH}"
    )


X_test = np.load(
    X_TEST_PATH
).astype(np.float32)

y_test = np.load(
    Y_TEST_PATH
).astype(np.int32)


print(
    "Test features:",
    X_test.shape
)

print(
    "Test labels:",
    y_test.shape
)


# ============================================================
# PREDICTION
# ============================================================

print("\n========== GENERATING PREDICTIONS ==========")

probabilities = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# PRECISION
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# RECALL
# ============================================================

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# F1 SCORE
# ============================================================

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# DISPLAY OVERALL METRICS
# ============================================================

print("\n==========================================")
print("OVERALL PERFORMANCE")
print("==========================================")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3]
)


print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(
    "             Predicted"
)

print(
    "             Low  Mod  High  Crit"
)

for i, row in enumerate(cm):

    print(
        f"Actual "
        f"{CLASS_NAMES[i]:8s}",
        row
    )


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n==========================================")
print("PER-CLASS ACCURACY")
print("==========================================")


for i, class_name in enumerate(CLASS_NAMES):

    total = np.sum(
        cm[i]
    )

    correct = cm[i, i]

    if total > 0:

        class_accuracy = (
            correct / total
        ) * 100

    else:

        class_accuracy = 0.0

    print(
        f"{class_name:10s}: "
        f"{class_accuracy:.2f}%"
    )


# ============================================================
# SAVE METRICS
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "evaluation_results.txt"
)


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ARES 2.0 - FINAL MODEL EVALUATION\n"
    )

    file.write(
        "=================================\n\n"
    )

    file.write(
        f"Model: ares_mlp_final.keras\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall: {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1-Score: {f1 * 100:.2f}%\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\nClassification Report:\n"
    )

    file.write(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("EVALUATION COMPLETE")
print("==========================================")

print(
    "Results saved to:"
)

print(
    RESULTS_PATH
)

print(
    "\n=========================================="
)