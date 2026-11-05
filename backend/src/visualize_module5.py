
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

DISCRIMINATOR_PATH = os.path.join(
    MODEL_DIR, "ares_gan_discriminator.keras"
)

GENERATOR_PATH = os.path.join(
    MODEL_DIR, "ares_gan_generator.keras"
)

REAL_TEST_PATH = os.path.join(
    DATA_DIR, "gan_real_test.npy"
)

SYNTHETIC_PATH = os.path.join(
    DATA_DIR, "synthetic_gan_samples.npy"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR, "gan_training_history.csv"
)

LATENT_DIM = 32

# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

required_files = [
    DISCRIMINATOR_PATH,
    GENERATOR_PATH,
    REAL_TEST_PATH,
    SYNTHETIC_PATH
]

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

print("\n========== ARES 2.0 MODULE 5 EVALUATION ==========\n")

# --------------------------------------------------
# LOAD MODELS AND DATA
# --------------------------------------------------

discriminator = tf.keras.models.load_model(
    DISCRIMINATOR_PATH,
    compile=False
)

generator = tf.keras.models.load_model(
    GENERATOR_PATH,
    compile=False
)

real_data = np.load(REAL_TEST_PATH).astype(np.float32)

synthetic_data = np.load(SYNTHETIC_PATH).astype(np.float32)

if real_data.ndim != 2 or synthetic_data.ndim != 2:
    raise ValueError("Real and synthetic data must be 2D arrays.")

if real_data.shape[1] != synthetic_data.shape[1]:
    raise ValueError("Real and synthetic feature dimensions do not match.")

if real_data.shape[1] != discriminator.input_shape[-1]:
    raise ValueError("Feature count does not match discriminator input.")

print("Real test samples:", real_data.shape)
print("Synthetic samples:", synthetic_data.shape)

# --------------------------------------------------
# DISCRIMINATOR EVALUATION
# --------------------------------------------------

# Equal number of real and synthetic test samples
n = min(len(real_data), len(synthetic_data))

real_eval = real_data[:n]
fake_eval = synthetic_data[:n]

X_eval = np.concatenate(
    [real_eval, fake_eval],
    axis=0
)

# 1 = real, 0 = synthetic
y_true = np.concatenate([
    np.ones(n, dtype=int),
    np.zeros(n, dtype=int)
])

scores = discriminator.predict(
    X_eval,
    verbose=0
).reshape(-1)

y_pred = (scores >= 0.5).astype(int)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

fpr, tpr, thresholds = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)

print("\n========== DISCRIMINATOR PERFORMANCE ==========")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"ROC AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[1, 0],
    target_names=["Real Data", "Synthetic Data"],
    zero_division=0
))

# --------------------------------------------------
# SAVE TEXT REPORT
# --------------------------------------------------

report_path = os.path.join(
    RESULTS_DIR, "module5_gan_evaluation.txt"
)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("ARES 2.0 - MODULE 5 GAN EVALUATION\n")
    file.write("=" * 45 + "\n\n")

    file.write("Discriminator: Real vs Synthetic Data\n\n")

    file.write(f"Accuracy  : {accuracy * 100:.2f}%\n")
    file.write(f"Precision : {precision * 100:.2f}%\n")
    file.write(f"Recall    : {recall * 100:.2f}%\n")
    file.write(f"F1 Score  : {f1 * 100:.2f}%\n")
    file.write(f"ROC AUC   : {roc_auc:.4f}\n\n")

    file.write("Confusion Matrix:\n")
    file.write(str(cm))
    file.write("\n\n")

    file.write(classification_report(
        y_true,
        y_pred,
        labels=[1, 0],
        target_names=["Real Data", "Synthetic Data"],
        zero_division=0
    ))

print("Saved:", report_path)

# --------------------------------------------------
# GRAPH 1: CONFUSION MATRIX
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("ARES 2.0 GAN - Discriminator Confusion Matrix")
plt.colorbar()

labels = ["Real Data", "Synthetic Data"]

plt.xticks([0, 1], labels)
plt.yticks([0, 1], labels)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "gan_confusion_matrix.png"),
    dpi=300
)

plt.close()

# --------------------------------------------------
# GRAPH 2: OVERALL METRICS
# --------------------------------------------------

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

metric_values = [
    accuracy,
    precision,
    recall,
    f1
]

plt.figure(figsize=(9, 6))

bars = plt.bar(metric_names, metric_values)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("ARES 2.0 GAN - Discriminator Performance")

for bar, value in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value * 100:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "gan_overall_performance.png"),
    dpi=300
)

plt.close()

# --------------------------------------------------
# GRAPH 3: ROC CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"GAN Discriminator (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ARES 2.0 GAN - ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "gan_roc_curve.png"),
    dpi=300
)

plt.close()

# --------------------------------------------------
# GRAPH 4: GENERATOR AND DISCRIMINATOR LOSS
# --------------------------------------------------

if os.path.exists(HISTORY_PATH):

    history = pd.read_csv(HISTORY_PATH)

    plt.figure(figsize=(10, 6))

    plt.plot(
        history["Epoch"],
        history["Generator_Loss"],
        label="Generator Loss"
    )

    plt.plot(
        history["Epoch"],
        history["Discriminator_Loss"],
        label="Discriminator Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ARES 2.0 GAN - Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, "gan_training_loss.png"),
        dpi=300
    )

    plt.close()

    # Discriminator training accuracy
    plt.figure(figsize=(9, 5))

    plt.plot(
        history["Epoch"],
        history["Discriminator_Accuracy"],
        label="Discriminator Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ARES 2.0 GAN - Discriminator Training Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, "gan_training_accuracy.png"),
        dpi=300
    )

    plt.close()

else:
    print("WARNING: GAN training history was not found.")

# --------------------------------------------------
# GRAPH 5: FEATURE DISTRIBUTION COMPARISON
# --------------------------------------------------

feature_count = min(6, real_data.shape[1])

fig, axes = plt.subplots(
    feature_count,
    1,
    figsize=(10, 4 * feature_count)
)

if feature_count == 1:
    axes = [axes]

for i in range(feature_count):

    axes[i].hist(
        real_data[:, i],
        bins=30,
        alpha=0.6,
        density=True,
        label="Real Data"
    )

    axes[i].hist(
        synthetic_data[:, i],
        bins=30,
        alpha=0.6,
        density=True,
        label="Synthetic Data"
    )

    axes[i].set_title(f"Feature {i + 1} Distribution")
    axes[i].set_xlabel("Feature Value")
    axes[i].set_ylabel("Density")
    axes[i].legend()

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "gan_feature_distributions.png"),
    dpi=300
)

plt.close()

# --------------------------------------------------
# GRAPH 6: PCA REAL VS SYNTHETIC
# --------------------------------------------------

combined = np.concatenate(
    [real_data, synthetic_data],
    axis=0
)

scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)

pca = PCA(n_components=2, random_state=42)

reduced = pca.fit_transform(combined_scaled)

real_pca = reduced[:len(real_data)]
synthetic_pca = reduced[len(real_data):]

plt.figure(figsize=(9, 7))

plt.scatter(
    real_pca[:, 0],
    real_pca[:, 1],
    alpha=0.5,
    label="Real Data"
)

plt.scatter(
    synthetic_pca[:, 0],
    synthetic_pca[:, 1],
    alpha=0.5,
    label="Synthetic Data"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("ARES 2.0 GAN - Real vs Synthetic Data (PCA)")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "gan_pca_comparison.png"),
    dpi=300
)

plt.close()

# --------------------------------------------------
# COMPLETION
# --------------------------------------------------

print("\n========== MODULE 5 VISUALIZATION COMPLETED ==========")
print("All graphs and evaluation reports saved in:")
print(RESULTS_DIR)
