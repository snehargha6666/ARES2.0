import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# ARES 2.0 - MODULE 1
# ARTIFICIAL NEURAL NETWORK (ANN)
# ============================================================


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. PATHS
# ============================================================

X_TRAIN_PATH = "data/X_train.npy"
X_VAL_PATH = "data/X_val.npy"
X_TEST_PATH = "data/X_test.npy"

Y_TRAIN_PATH = "data/y_train.npy"
Y_VAL_PATH = "data/y_val.npy"
Y_TEST_PATH = "data/y_test.npy"

MODEL_PATH = "models/ares_ann.keras"

os.makedirs("models", exist_ok=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n========== LOADING ANN DATA ==========")

X_train = np.load(X_TRAIN_PATH)
X_val = np.load(X_VAL_PATH)
X_test = np.load(X_TEST_PATH)

y_train = np.load(Y_TRAIN_PATH)
y_val = np.load(Y_VAL_PATH)
y_test = np.load(Y_TEST_PATH)

print("Training shape   :", X_train.shape)
print("Validation shape :", X_val.shape)
print("Testing shape    :", X_test.shape)


# ============================================================
# 4. MODEL CONFIGURATION
# ============================================================

input_features = X_train.shape[1]
num_classes = 4

print("\n========== ANN CONFIGURATION ==========")

print("Input features :", input_features)
print("Output classes :", num_classes)

print("\nArchitecture:")
print("Input          :", input_features)
print("Hidden Layer 1 : 128 neurons + ReLU")
print("Dropout        : 30%")
print("Hidden Layer 2 : 64 neurons + ReLU")
print("Dropout        : 20%")
print("Hidden Layer 3 : 32 neurons + ReLU")
print("Output Layer   : 4 neurons + Softmax")


# ============================================================
# 5. BUILD ANN
# ============================================================

model = Sequential([

    # Input layer
    Input(
        shape=(input_features,)
    ),

    # Hidden Layer 1
    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.30
    ),

    # Hidden Layer 2
    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.20
    ),

    # Hidden Layer 3
    Dense(
        32,
        activation="relu"
    ),

    # Output Layer
    Dense(
        num_classes,
        activation="softmax"
    )
])


# ============================================================
# 6. COMPILE ANN
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# 7. MODEL SUMMARY
# ============================================================

print("\n========== ANN MODEL SUMMARY ==========")

model.summary()


# ============================================================
# 8. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=15,

    restore_best_weights=True,

    verbose=1
)


# ============================================================
# 9. TRAIN ANN
# ============================================================

print("\n========== ANN TRAINING ==========")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=100,

    batch_size=32,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# 10. TRAINING SUMMARY
# ============================================================

print(
    "\n========== TRAINING COMPLETE =========="
)

epochs_used = len(
    history.history["loss"]
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_loss = min(
    history.history["val_loss"]
)

print(
    "Epochs actually used:",
    epochs_used
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)


# ============================================================
# 11. TEST PREDICTION
# ============================================================

print(
    "\n========== ANN TESTING =========="
)

probabilities = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# 12. CLASS NAMES
# ============================================================

class_names = [
    "Low",
    "Moderate",
    "High",
    "Critical"
]


# ============================================================
# 13. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision_weighted = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

precision_macro = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 14. FINAL METRICS
# ============================================================

print(
    "\n========== FINAL ANN TEST METRICS =========="
)

print(
    f"Accuracy           : {accuracy:.4f}"
)

print(
    f"Weighted Precision : {precision_weighted:.4f}"
)

print(
    f"Weighted Recall    : {recall_weighted:.4f}"
)

print(
    f"Weighted F1        : {f1_weighted:.4f}"
)

print(
    f"Macro Precision    : {precision_macro:.4f}"
)

print(
    f"Macro Recall       : {recall_macro:.4f}"
)

print(
    f"Macro F1           : {f1_macro:.4f}"
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print(
    "\n========== CLASSIFICATION REPORT =========="
)

report = classification_report(

    y_test,

    y_pred,

    target_names=class_names,

    zero_division=0
)

print(
    report
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print(
    "\n========== CONFUSION MATRIX =========="
)

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(

    cm,

    index=class_names,

    columns=class_names
)

print(
    cm_df
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print(
    "\nANN model saved to:"
)

print(
    MODEL_PATH
)


# ============================================================
# 18. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(

    "data/ann_training_history.csv",

    index=False
)

print(
    "Training history saved to:"
)

print(
    "data/ann_training_history.csv"
)


# ============================================================
# 19. SAVE CONFUSION MATRIX
# ============================================================

cm_df.to_csv(

    "data/ann_confusion_matrix.csv"
)

print(
    "Confusion matrix saved to:"
)

print(
    "data/ann_confusion_matrix.csv"
)


# ============================================================
# 20. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Weighted Precision",

        "Weighted Recall",

        "Weighted F1",

        "Macro Precision",

        "Macro Recall",

        "Macro F1",

        "Best Validation Accuracy",

        "Best Validation Loss",

        "Epochs Used"
    ],

    "Value": [

        accuracy,

        precision_weighted,

        recall_weighted,

        f1_weighted,

        precision_macro,

        recall_macro,

        f1_macro,

        best_val_accuracy,

        best_val_loss,

        epochs_used
    ]
})

metrics_df.to_csv(

    "data/ann_metrics.csv",

    index=False
)

print(
    "Metrics saved to:"
)

print(
    "data/ann_metrics.csv"
)


# ============================================================
# 21. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual": y_test,

    "Predicted": y_pred
})


prediction_df[
    "Actual_Severity"
] = prediction_df[
    "Actual"
].map({

    0: "Low",

    1: "Moderate",

    2: "High",

    3: "Critical"
})


prediction_df[
    "Predicted_Severity"
] = prediction_df[
    "Predicted"
].map({

    0: "Low",

    1: "Moderate",

    2: "High",

    3: "Critical"
})


prediction_df.to_csv(

    "data/ann_predictions.csv",

    index=False
)

print(
    "Predictions saved to:"
)

print(
    "data/ann_predictions.csv"
)


# ============================================================
# 22. COMPLETE
# ============================================================

print(
    "\n========== ANN TRAINING COMPLETE =========="
)