import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# ARES 2.0 - MODULE 2
# COMPLETE MODEL VISUALIZATION
# ============================================================

print("\n==========================================")
print("ARES 2.0 - COMPLETE MODEL VISUALIZATION")
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ares_mlp_final.keras"
)

X_TEST_PATH = os.path.join(
    DATA_DIR,
    "X_test.npy"
)

Y_TEST_PATH = os.path.join(
    DATA_DIR,
    "y_test.npy"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.json"
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
# LOAD FINAL MODEL
# ============================================================

print("========== LOADING FINAL MODEL ==========")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Final model not found:\n{MODEL_PATH}"
    )


model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Final model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n========== LOADING TEST DATA ==========")

if not os.path.exists(X_TEST_PATH):

    raise FileNotFoundError(
        f"Test features not found:\n{X_TEST_PATH}"
    )


if not os.path.exists(Y_TEST_PATH):

    raise FileNotFoundError(
        f"Test labels not found:\n{Y_TEST_PATH}"
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
# GENERATE PREDICTIONS
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
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3]
)


print("\nConfusion Matrix:")
print(cm)


# ============================================================
# VISUAL 1
# CONFUSION MATRIX
# ============================================================

print(
    "\n========== 1. CONFUSION MATRIX =========="
)


plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "ARES 2.0 - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    np.arange(4),
    CLASS_NAMES
)

plt.yticks(
    np.arange(4),
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)


for i in range(4):

    for j in range(4):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()


confusion_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    confusion_path
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3],
    average=None,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3],
    average=None,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3],
    average=None,
    zero_division=0
)


# ============================================================
# VISUAL 2
# PER-CLASS PERFORMANCE
# ============================================================

print(
    "\n========== 2. PER-CLASS PERFORMANCE =========="
)


x = np.arange(
    len(CLASS_NAMES)
)

width = 0.25


plt.figure(
    figsize=(10, 6)
)


plt.bar(
    x - width,
    precision,
    width,
    label="Precision"
)

plt.bar(
    x,
    recall,
    width,
    label="Recall"
)

plt.bar(
    x + width,
    f1,
    width,
    label="F1-Score"
)


plt.xticks(
    x,
    CLASS_NAMES
)

plt.ylim(
    0,
    1
)

plt.xlabel(
    "Severity Class"
)

plt.ylabel(
    "Score"
)

plt.title(
    "ARES 2.0 - Per-Class Performance"
)

plt.legend()

plt.tight_layout()


metrics_path = os.path.join(
    RESULTS_DIR,
    "per_class_metrics.png"
)

plt.savefig(
    metrics_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    metrics_path
)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = np.mean(
    y_test == y_pred
)

weighted_precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


overall_values = [
    accuracy,
    weighted_precision,
    weighted_recall,
    weighted_f1
]

overall_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]


# ============================================================
# VISUAL 3
# OVERALL PERFORMANCE
# ============================================================

print(
    "\n========== 3. OVERALL PERFORMANCE =========="
)


plt.figure(
    figsize=(9, 6)
)

bars = plt.bar(
    overall_names,
    overall_values
)

plt.ylim(
    0,
    1
)

plt.xlabel(
    "Metric"
)

plt.ylabel(
    "Score"
)

plt.title(
    "ARES 2.0 - Overall Model Performance"
)


for bar, value in zip(
    bars,
    overall_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 0.02,

        f"{value * 100:.2f}%",

        ha="center"
    )


plt.tight_layout()


overall_path = os.path.join(
    RESULTS_DIR,
    "overall_performance.png"
)

plt.savefig(
    overall_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    overall_path
)


# ============================================================
# TRAINING HISTORY
# ============================================================

print(
    "\n========== LOADING TRAINING HISTORY =========="
)


if not os.path.exists(HISTORY_PATH):

    print(
        "\nWARNING:"
    )

    print(
        "training_history.json was not found."
    )

    print(
        "Training graphs will not be generated."
    )

    print(
        "Run train_final.py first."
    )

else:

    with open(
        HISTORY_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        history = json.load(
            file
        )


    print(
        "Training history loaded successfully."
    )


    # ========================================================
    # VISUAL 4
    # TRAINING VS VALIDATION LOSS
    # ========================================================

    print(
        "\n========== 4. TRAINING VS VALIDATION LOSS =========="
    )


    train_loss = history.get(
        "loss",
        []
    )

    val_loss = history.get(
        "val_loss",
        []
    )


    if train_loss and val_loss:

        epochs = range(
            1,
            len(train_loss) + 1
        )


        plt.figure(
            figsize=(10, 6)
        )


        plt.plot(
            epochs,
            train_loss,
            label="Training Loss"
        )

        plt.plot(
            epochs,
            val_loss,
            label="Validation Loss"
        )


        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Loss"
        )

        plt.title(
            "ARES 2.0 - Training vs Validation Loss"
        )

        plt.legend()

        plt.grid(
            True
        )

        plt.tight_layout()


        loss_path = os.path.join(
            RESULTS_DIR,
            "training_validation_loss.png"
        )


        plt.savefig(
            loss_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        print(
            "Saved:",
            loss_path
        )


    else:

        print(
            "Loss history is unavailable."
        )


    # ========================================================
    # VISUAL 5
    # TRAINING VS VALIDATION ACCURACY
    # ========================================================

    print(
        "\n========== 5. TRAINING VS VALIDATION ACCURACY =========="
    )


    train_accuracy = history.get(
        "accuracy",
        []
    )

    val_accuracy = history.get(
        "val_accuracy",
        []
    )


    if train_accuracy and val_accuracy:

        epochs = range(
            1,
            len(train_accuracy) + 1
        )


        plt.figure(
            figsize=(10, 6)
        )


        plt.plot(
            epochs,
            train_accuracy,
            label="Training Accuracy"
        )

        plt.plot(
            epochs,
            val_accuracy,
            label="Validation Accuracy"
        )


        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Accuracy"
        )

        plt.title(
            "ARES 2.0 - Training vs Validation Accuracy"
        )

        plt.legend()

        plt.grid(
            True
        )

        plt.tight_layout()


        accuracy_path = os.path.join(
            RESULTS_DIR,
            "training_validation_accuracy.png"
        )


        plt.savefig(
            accuracy_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        print(
            "Saved:",
            accuracy_path
        )


    else:

        print(
            "Accuracy history is unavailable."
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("ALL VISUALIZATIONS COMPLETE")
print("==========================================")

print("\nYour Module 2 visual files are in:")

print(
    RESULTS_DIR
)

print("\nExpected files:")

print(
    "1. confusion_matrix.png"
)

print(
    "2. per_class_metrics.png"
)

print(
    "3. overall_performance.png"
)

print(
    "4. training_validation_loss.png"
)

print(
    "5. training_validation_accuracy.png"
)

print("\n==========================================")