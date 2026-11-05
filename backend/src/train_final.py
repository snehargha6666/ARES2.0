import os
import json
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# ARES 2.0 - MODULE 2
# FINAL MODEL TRAINING
# ============================================================

print("\n==========================================")
print("ARES 2.0 - FINAL MODEL TRAINING")
print("==========================================\n")


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


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
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

X_TRAIN_PATH = os.path.join(
    DATA_DIR,
    "X_train.npy"
)

Y_TRAIN_PATH = os.path.join(
    DATA_DIR,
    "y_train.npy"
)

X_VAL_PATH = os.path.join(
    DATA_DIR,
    "X_val.npy"
)

Y_VAL_PATH = os.path.join(
    DATA_DIR,
    "y_val.npy"
)

X_TEST_PATH = os.path.join(
    DATA_DIR,
    "X_test.npy"
)

Y_TEST_PATH = os.path.join(
    DATA_DIR,
    "y_test.npy"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ares_mlp_final.keras"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print("========== LOADING DATA ==========")

required_files = [
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_VAL_PATH,
    Y_VAL_PATH,
    X_TEST_PATH,
    Y_TEST_PATH
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required data file not found:\n{path}"
        )


X_train = np.load(
    X_TRAIN_PATH
).astype(np.float32)

y_train = np.load(
    Y_TRAIN_PATH
).astype(np.int32)

X_val = np.load(
    X_VAL_PATH
).astype(np.float32)

y_val = np.load(
    Y_VAL_PATH
).astype(np.int32)

X_test = np.load(
    X_TEST_PATH
).astype(np.float32)

y_test = np.load(
    Y_TEST_PATH
).astype(np.int32)


print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Testing:",
    X_test.shape
)


# ============================================================
# MODEL PARAMETERS
# ============================================================

INPUT_FEATURES = X_train.shape[1]

OUTPUT_CLASSES = 4

LEARNING_RATE = 0.001

L2_VALUE = 0.001

DROPOUT_RATE = 0.3

BATCH_SIZE = 32

MAX_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 12


# ============================================================
# BUILD FINAL MLP
# ============================================================

print("\n========== BUILDING FINAL MODEL ==========")


model = Sequential(
    [

        Dense(
            128,
            activation="relu",
            kernel_regularizer=l2(L2_VALUE),
            input_shape=(INPUT_FEATURES,)
        ),

        Dropout(
            DROPOUT_RATE
        ),

        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(L2_VALUE)
        ),

        Dropout(
            DROPOUT_RATE
        ),

        Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(L2_VALUE)
        ),

        Dense(
            OUTPUT_CLASSES,
            activation="softmax"
        )

    ]
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = RMSprop(
    learning_rate=LEARNING_RATE
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=optimizer,

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=EARLY_STOPPING_PATIENCE,

    restore_best_weights=True,

    verbose=1
)


# ============================================================
# TRAINING
# ============================================================

print("\n========== TRAINING ==========")

print(
    "Optimizer      : RMSprop"
)

print(
    "Learning Rate  :",
    LEARNING_RATE
)

print(
    "L2             :",
    L2_VALUE
)

print(
    "Dropout        :",
    DROPOUT_RATE
)

print(
    "Batch Size     :",
    BATCH_SIZE
)

print(
    "Maximum Epochs :",
    MAX_EPOCHS
)


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=MAX_EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=2
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

print("\n========== SAVING TRAINING HISTORY ==========")


with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history.history,
        file,
        indent=4
    )


print(
    "Training history saved:"
)

print(
    HISTORY_PATH
)


# ============================================================
# FIND BEST EPOCH
# ============================================================

val_losses = history.history[
    "val_loss"
]

val_accuracies = history.history[
    "val_accuracy"
]


best_epoch = int(
    np.argmin(
        val_losses
    )
) + 1

best_val_loss = float(
    val_losses[
        best_epoch - 1
    ]
)

best_val_accuracy = float(
    val_accuracies[
        best_epoch - 1
    ]
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n========== FINAL TEST EVALUATION ==========")


test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0
)


print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")


probabilities = model.predict(
    X_test,
    verbose=0
)


y_pred = np.argmax(
    probabilities,
    axis=1
)


confusion_matrix = np.zeros(
    (
        OUTPUT_CLASSES,
        OUTPUT_CLASSES
    ),
    dtype=int
)


for actual, predicted in zip(
    y_test,
    y_pred
):

    confusion_matrix[
        actual,
        predicted
    ] += 1


print(
    confusion_matrix
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

CONFUSION_PATH = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.npy"
)

np.save(
    CONFUSION_PATH,
    confusion_matrix
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


# ============================================================
# SAVE FINAL TRAINING SUMMARY
# ============================================================

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "final_model_summary.json"
)


summary = {

    "model": "MLP",

    "input_features": int(
        INPUT_FEATURES
    ),

    "output_classes": int(
        OUTPUT_CLASSES
    ),

    "optimizer": "RMSprop",

    "learning_rate": LEARNING_RATE,

    "l2_regularization": L2_VALUE,

    "dropout": DROPOUT_RATE,

    "batch_size": BATCH_SIZE,

    "maximum_epochs": MAX_EPOCHS,

    "epochs_completed": len(
        history.history["loss"]
    ),

    "best_epoch": best_epoch,

    "best_validation_loss":
        best_val_loss,

    "best_validation_accuracy":
        best_val_accuracy,

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(test_accuracy)

}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("FINAL MODEL COMPLETE")
print("==========================================")

print(
    "Optimizer          : RMSprop"
)

print(
    "Best Epoch         :",
    best_epoch
)

print(
    f"Best Val Loss      : {best_val_loss:.4f}"
)

print(
    f"Best Val Accuracy  : "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy      : "
    f"{test_accuracy * 100:.2f}%"
)

print("\nFinal model saved:")

print(
    MODEL_PATH
)

print("\nTraining history saved:")

print(
    HISTORY_PATH
)

print("\nSummary saved:")

print(
    SUMMARY_PATH
)

print("\n==========================================")