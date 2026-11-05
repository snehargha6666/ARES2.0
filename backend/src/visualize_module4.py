
import os
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# ============================================================
# ARES 2.0 - MODULE 4
# EVALUATION AND GRAPH GENERATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

MODEL_NAMES = [
    "RNN",
    "LSTM",
    "GRU",
    "BiLSTM"
]

print("\n============================================")
print("ARES 2.0 - MODULE 4")
print("GRAPH GENERATION AND EVALUATION")
print("============================================")


# ============================================================
# 1. LOAD METRICS AND METADATA
# ============================================================

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "module4_metrics.csv"
)

METADATA_PATH = os.path.join(
    RESULTS_DIR,
    "module4_metadata.json"
)

if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError(
        "module4_metrics.csv not found. "
        "Run train_module4.py first."
    )

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(
        "module4_metadata.json not found. "
        "Run train_module4.py first."
    )

metrics = pd.read_csv(METRICS_PATH)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)

CLASS_NAMES = metadata["class_names"]

NUM_CLASSES = len(CLASS_NAMES)

print("\nModels available:", metrics["Model"].tolist())
print("Classes:", CLASS_NAMES)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def save_plot(filename):

    output_path = os.path.join(
        RESULTS_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Graph generated:", filename)


def safe_name(name):

    return re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        name
    )


def load_predictions(model_name):

    path = os.path.join(
        RESULTS_DIR,
        f"module4_predictions_{model_name.lower()}.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction file missing: {path}"
        )

    return pd.read_csv(path)


# ============================================================
# 3. MODEL ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    metrics["Model"],
    metrics["Accuracy"],
    label="Accuracy"
)

plt.ylim(0, 1)

plt.xlabel("Sequence Models")
plt.ylabel("Accuracy")

plt.title(
    "ARES 2.0 - Module 4 Accuracy Comparison"
)

plt.legend()

for bar in bars:

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.3f}",
        ha="center"
    )

save_plot(
    "module4_accuracy_comparison.png"
)


# ============================================================
# 4. PRECISION / RECALL / F1 COMPARISON
# ============================================================

comparison_metrics = [
    "Precision_Weighted",
    "Recall_Weighted",
    "F1_Weighted"
]

x = np.arange(
    len(metrics["Model"])
)

width = 0.25

plt.figure(figsize=(11, 6))

for i, metric in enumerate(comparison_metrics):

    plt.bar(
        x + (i - 1) * width,
        metrics[metric],
        width,
        label=metric.replace("_Weighted", "")
    )

plt.xticks(
    x,
    metrics["Model"]
)

plt.ylim(0, 1)

plt.xlabel("Models")
plt.ylabel("Performance Score")

plt.title(
    "Module 4 - Precision, Recall and F1 Comparison"
)

plt.legend()

save_plot(
    "module4_precision_recall_f1.png"
)


# ============================================================
# 5. MACRO VS WEIGHTED F1
# ============================================================

x = np.arange(
    len(metrics["Model"])
)

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    metrics["F1_Macro"],
    width,
    label="Macro F1"
)

plt.bar(
    x + width / 2,
    metrics["F1_Weighted"],
    width,
    label="Weighted F1"
)

plt.xticks(
    x,
    metrics["Model"]
)

plt.ylim(0, 1)

plt.xlabel("Models")
plt.ylabel("F1 Score")

plt.title(
    "Module 4 - Macro vs Weighted F1"
)

plt.legend()

save_plot(
    "module4_macro_weighted_f1.png"
)


# ============================================================
# 6. COMPLETE MODEL METRIC COMPARISON
# ============================================================

all_metrics = [
    "Accuracy",
    "Precision_Weighted",
    "Recall_Weighted",
    "F1_Weighted",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro"
]

plt.figure(figsize=(13, 7))

for metric in all_metrics:

    plt.plot(
        metrics["Model"],
        metrics[metric],
        marker="o",
        linewidth=2,
        label=metric
    )

plt.ylim(0, 1)

plt.xlabel("Sequence Models")
plt.ylabel("Score")

plt.title(
    "ARES 2.0 - Complete Module 4 Model Evaluation"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

save_plot(
    "module4_all_metrics.png"
)


# ============================================================
# 7. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 11)
)

axes = axes.flatten()

for i, model_name in enumerate(MODEL_NAMES):

    ax = axes[i]

    prediction_df = load_predictions(
        model_name
    )

    actual = pd.Categorical(
        prediction_df["Actual"],
        categories=CLASS_NAMES
    ).codes

    predicted = pd.Categorical(
        prediction_df["Predicted"],
        categories=CLASS_NAMES
    ).codes

    cm = confusion_matrix(
        actual,
        predicted,
        labels=np.arange(NUM_CLASSES)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest"
    )

    ax.set_title(
        f"{model_name} Confusion Matrix"
    )

    ax.set_xlabel("Predicted Severity")
    ax.set_ylabel("Actual Severity")

    ax.set_xticks(
        np.arange(NUM_CLASSES)
    )

    ax.set_yticks(
        np.arange(NUM_CLASSES)
    )

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=35,
        ha="right"
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    for row in range(NUM_CLASSES):

        for col in range(NUM_CLASSES):

            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046
    )

plt.suptitle(
    "ARES 2.0 - Module 4 Confusion Matrices",
    fontsize=15
)

save_plot(
    "module4_confusion_matrices.png"
)


# ============================================================
# 8. IDENTIFY BEST MODEL
# ============================================================

best_index = metrics[
    "F1_Weighted"
].idxmax()

best_row = metrics.loc[
    best_index
]

BEST_MODEL = best_row["Model"]

print("\nBest Model:", BEST_MODEL)

print(
    "Best Weighted F1:",
    round(best_row["F1_Weighted"], 4)
)


# ============================================================
# 9. BEST MODEL PER-CLASS METRICS
# ============================================================

best_df = load_predictions(
    BEST_MODEL
)

y_true = pd.Categorical(
    best_df["Actual"],
    categories=CLASS_NAMES
).codes

y_pred = pd.Categorical(
    best_df["Predicted"],
    categories=CLASS_NAMES
).codes

report_dict = classification_report(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

precision_values = []
recall_values = []
f1_values = []

for class_name in CLASS_NAMES:

    precision_values.append(
        report_dict[class_name]["precision"]
    )

    recall_values.append(
        report_dict[class_name]["recall"]
    )

    f1_values.append(
        report_dict[class_name]["f1-score"]
    )

x = np.arange(NUM_CLASSES)

width = 0.25

plt.figure(figsize=(11, 6))

plt.bar(
    x - width,
    precision_values,
    width,
    label="Precision"
)

plt.bar(
    x,
    recall_values,
    width,
    label="Recall"
)

plt.bar(
    x + width,
    f1_values,
    width,
    label="F1 Score"
)

plt.xticks(
    x,
    CLASS_NAMES
)

plt.ylim(0, 1)

plt.xlabel("Disaster Severity")

plt.ylabel("Score")

plt.title(
    f"{BEST_MODEL} - Per Class Performance"
)

plt.legend()

save_plot(
    "module4_per_class_metrics.png"
)


# ============================================================
# 10. ROC CURVE - BEST MODEL
# ============================================================

probability_columns = [
    f"Probability_{safe_name(name)}"
    for name in CLASS_NAMES
]

probabilities = best_df[
    probability_columns
].to_numpy()

y_binary = label_binarize(
    y_true,
    classes=np.arange(NUM_CLASSES)
)

plt.figure(figsize=(9, 7))

for i, class_name in enumerate(CLASS_NAMES):

    if len(np.unique(y_binary[:, i])) < 2:

        print(
            "ROC skipped for:",
            class_name,
            "- insufficient class variation."
        )

        continue

    fpr, tpr, thresholds = roc_curve(
        y_binary[:, i],
        probabilities[:, i]
    )

    class_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{class_name} (AUC={class_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    f"{BEST_MODEL} - Multiclass ROC Curve"
)

plt.legend()

save_plot(
    "module4_roc_curve.png"
)


# ============================================================
# 11. TRAINING AND VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(12, 7))

for model_name in MODEL_NAMES:

    history_path = os.path.join(
        RESULTS_DIR,
        f"module4_history_{model_name.lower()}.csv"
    )

    if not os.path.exists(history_path):
        continue

    history = pd.read_csv(
        history_path
    )

    plt.plot(
        history["Epoch"],
        history["accuracy"],
        label=f"{model_name} Training"
    )

    plt.plot(
        history["Epoch"],
        history["val_accuracy"],
        linestyle="--",
        label=f"{model_name} Validation"
    )

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Module 4 - Training vs Validation Accuracy"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

save_plot(
    "module4_training_accuracy.png"
)


# ============================================================
# 12. TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(12, 7))

for model_name in MODEL_NAMES:

    history_path = os.path.join(
        RESULTS_DIR,
        f"module4_history_{model_name.lower()}.csv"
    )

    if not os.path.exists(history_path):
        continue

    history = pd.read_csv(
        history_path
    )

    plt.plot(
        history["Epoch"],
        history["loss"],
        label=f"{model_name} Training"
    )

    plt.plot(
        history["Epoch"],
        history["val_loss"],
        linestyle="--",
        label=f"{model_name} Validation"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Module 4 - Training vs Validation Loss"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

save_plot(
    "module4_training_loss.png"
)


# ============================================================
# 13. TRAINING TIME COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    metrics["Model"],
    metrics["Training_Time_Seconds"]
)

plt.xlabel("Models")
plt.ylabel("Training Time (Seconds)")

plt.title(
    "Module 4 - Training Time Comparison"
)

for bar in bars:

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}s",
        ha="center",
        va="bottom"
    )

save_plot(
    "module4_training_time.png"
)


# ============================================================
# 14. GRADIENT FLOW ANALYSIS
# ============================================================

gradient_path = os.path.join(
    RESULTS_DIR,
    "module4_gradient_profile.csv"
)

if os.path.exists(gradient_path):

    gradient_df = pd.read_csv(
        gradient_path
    )

    plt.figure(figsize=(10, 6))

    for model_name in MODEL_NAMES:

        subset = gradient_df[
            gradient_df["Model"] == model_name
        ]

        if len(subset) == 0:
            continue

        plt.plot(
            subset["Timestep"],
            subset["Mean_Absolute_Input_Gradient"],
            marker="o",
            label=model_name
        )

    plt.xlabel("Sequence Timestep")

    plt.ylabel("Mean Absolute Input Gradient")

    plt.title(
        "Module 4 - Gradient Magnitude Across Timesteps"
    )

    plt.legend()

    save_plot(
        "module4_gradient_flow.png"
    )


# ============================================================
# 15. SAVE FINAL EVALUATION REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "module4_best_model_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ARES 2.0 - MODULE 4 EVALUATION REPORT\n"
    )

    file.write("=" * 55 + "\n\n")

    file.write(
        f"Best Model: {BEST_MODEL}\n\n"
    )

    for column in [
        "Accuracy",
        "Precision_Weighted",
        "Recall_Weighted",
        "F1_Weighted",
        "Precision_Macro",
        "Recall_Macro",
        "F1_Macro",
        "ROC_AUC",
        "Training_Time_Seconds"
    ]:

        file.write(
            f"{column}: {best_row[column]:.4f}\n"
        )

    file.write(
        "\n\nBEST MODEL CLASSIFICATION REPORT\n\n"
    )

    file.write(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    file.write(
        "\n\nALL MODEL COMPARISON RESULTS\n\n"
    )

    file.write(
        metrics.to_string(index=False)
    )


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n============================================")
print("MODULE 4 GRAPH GENERATION COMPLETED")
print("============================================")

print("\nBest Model:", BEST_MODEL)

print("\nModel Comparison:")

print(
    metrics.to_string(index=False)
)

print("\nAll graphs saved to:")

print(RESULTS_DIR)

print("\nAll reports saved successfully.")

