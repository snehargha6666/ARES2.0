import os
import numpy as np
import tensorflow as tf


# ============================================================
# ARES 2.0 - MODULE 2
# OPTIMIZED MLP TRAINING
# ============================================================

print("\n==========================================")
print("ARES 2.0 - MODULE 2")
print("MLP MODEL OPTIMIZATION")
print("==========================================\n")


# ============================================================
# 1. PATHS
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "models"

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
X_VAL_PATH = os.path.join(DATA_DIR, "X_val.npy")

Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.npy")
Y_VAL_PATH = os.path.join(DATA_DIR, "y_val.npy")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ares_mlp_optimized.keras"
)


# ============================================================
# 2. HYPERPARAMETERS
# ============================================================

INPUT_SIZE = 119
NUM_CLASSES = 4

EPOCHS = 100
BATCH_SIZE = 32

LEARNING_RATE = 0.001

L2_VALUE = 0.001
DROPOUT_RATE = 0.30

PATIENCE = 12


# ============================================================
# 3. RANDOM SEED
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 4. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 5. LOAD PREPROCESSED DATA
# ============================================================

print("========== LOADING DATA ==========")

X_train = np.load(
    X_TRAIN_PATH
).astype(np.float32)

X_val = np.load(
    X_VAL_PATH
).astype(np.float32)

y_train = np.load(
    Y_TRAIN_PATH
).astype(np.int32)

y_val = np.load(
    Y_VAL_PATH
).astype(np.int32)


print(
    "Training features:",
    X_train.shape
)

print(
    "Validation features:",
    X_val.shape
)

print(
    "Training labels:",
    y_train.shape
)

print(
    "Validation labels:",
    y_val.shape
)


# ============================================================
# 6. VERIFY INPUT SIZE
# ============================================================

if X_train.shape[1] != INPUT_SIZE:

    raise ValueError(
        f"Expected {INPUT_SIZE} input features, "
        f"but received {X_train.shape[1]}"
    )


# ============================================================
# 7. BUILD OPTIMIZED MLP
# ============================================================

print(
    "\n========== BUILDING OPTIMIZED MLP =========="
)


l2_regularizer = tf.keras.regularizers.l2(
    L2_VALUE
)


model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(INPUT_SIZE,)
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=l2_regularizer
    ),

    tf.keras.layers.Dropout(
        DROPOUT_RATE
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=l2_regularizer
    ),

    tf.keras.layers.Dropout(
        DROPOUT_RATE
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=l2_regularizer
    ),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


model.summary()


# ============================================================
# 8. OPTIMIZER
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


# ============================================================
# 9. LOSS FUNCTION
# ============================================================

loss_function = (
    tf.keras.losses.SparseCategoricalCrossentropy()
)


# ============================================================
# 10. DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_train,
        y_train
    )
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(X_train),
    seed=42
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        X_val,
        y_val
    )
)

val_dataset = val_dataset.batch(
    BATCH_SIZE
)


# ============================================================
# 11. TRAINING FUNCTION
# ============================================================

@tf.function
def train_step(
    x_batch,
    y_batch
):

    # --------------------------------------------------------
    # FORWARD PROPAGATION
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        predictions = model(
            x_batch,
            training=True
        )

        classification_loss = loss_function(
            y_batch,
            predictions
        )

        # ----------------------------------------------------
        # L2 REGULARIZATION LOSS
        # ----------------------------------------------------

        regularization_loss = tf.add_n(
            model.losses
        )

        total_loss = (
            classification_loss
            +
            regularization_loss
        )


    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------

    gradients = tape.gradient(
        total_loss,
        model.trainable_variables
    )


    # --------------------------------------------------------
    # UPDATE WEIGHTS
    # --------------------------------------------------------

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables
        )
    )


    return total_loss


# ============================================================
# 12. VALIDATION FUNCTION
# ============================================================

def validate():

    total_loss = 0.0
    total_correct = 0
    total_samples = 0


    for x_batch, y_batch in val_dataset:

        predictions = model(
            x_batch,
            training=False
        )

        classification_loss = loss_function(
            y_batch,
            predictions
        )

        regularization_loss = tf.add_n(
            model.losses
        )

        total_batch_loss = (
            classification_loss
            +
            regularization_loss
        )


        batch_size = tf.shape(
            y_batch
        )[0]


        total_loss += (
            float(total_batch_loss)
            *
            int(batch_size)
        )


        predicted_classes = tf.argmax(
            predictions,
            axis=1,
            output_type=tf.int32
        )


        correct = tf.reduce_sum(
            tf.cast(
                tf.equal(
                    predicted_classes,
                    y_batch
                ),
                tf.int32
            )
        )


        total_correct += int(
            correct
        )

        total_samples += int(
            batch_size
        )


    average_loss = (
        total_loss /
        total_samples
    )

    accuracy = (
        total_correct /
        total_samples
    )


    return (
        average_loss,
        accuracy
    )


# ============================================================
# 13. TRAINING LOOP
# ============================================================

print(
    "\n========== TRAINING =========="
)

print(
    "Optimizer       : Adam"
)

print(
    "Learning Rate   :",
    LEARNING_RATE
)

print(
    "L2 Regularizer  :",
    L2_VALUE
)

print(
    "Dropout Rate    :",
    DROPOUT_RATE
)

print(
    "Batch Size      :",
    BATCH_SIZE
)

print(
    "Maximum Epochs  :",
    EPOCHS
)


best_val_loss = float(
    "inf"
)

best_val_accuracy = 0.0

epochs_without_improvement = 0

best_weights = None


for epoch in range(
    1,
    EPOCHS + 1
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    training_loss = 0.0
    training_samples = 0


    for x_batch, y_batch in train_dataset:

        batch_loss = train_step(
            x_batch,
            y_batch
        )


        batch_size = int(
            tf.shape(y_batch)[0]
        )


        training_loss += (
            float(batch_loss)
            *
            batch_size
        )

        training_samples += (
            batch_size
        )


    training_loss /= training_samples


    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validation_loss, validation_accuracy = (
        validate()
    )


    # --------------------------------------------------------
    # PRINT PROGRESS
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {training_loss:.4f} | "
        f"Val Loss: {validation_loss:.4f} | "
        f"Val Accuracy: {validation_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # CHECK IMPROVEMENT
    # --------------------------------------------------------

    if validation_loss < best_val_loss:

        best_val_loss = validation_loss

        best_val_accuracy = (
            validation_accuracy
        )

        epochs_without_improvement = 0


        # Save best weights in memory
        best_weights = [
            variable.numpy()
            for variable in model.trainable_variables
        ]


        print(
            "  -> Validation loss improved."
        )


    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        print(
            "No validation improvement for",
            PATIENCE,
            "epochs."
        )

        break


# ============================================================
# 14. RESTORE BEST WEIGHTS
# ============================================================

if best_weights is not None:

    for variable, best_value in zip(
        model.trainable_variables,
        best_weights
    ):

        variable.assign(
            best_value
        )


# ============================================================
# 15. FINAL VALIDATION
# ============================================================

final_val_loss, final_val_accuracy = (
    validate()
)


# ============================================================
# 16. SAVE OPTIMIZED MODEL
# ============================================================

model.save(
    MODEL_PATH
)


# ============================================================
# 17. FINAL REPORT
# ============================================================

print(
    "\n=========================================="
)

print(
    "MODULE 2 TRAINING COMPLETE"
)

print(
    "=========================================="
)

print(
    "Model              : MLP"
)

print(
    "Input Features     :",
    INPUT_SIZE
)

print(
    "Output Classes     :",
    NUM_CLASSES
)

print(
    "Optimizer          : Adam"
)

print(
    "Learning Rate      :",
    LEARNING_RATE
)

print(
    "L2 Regularization  :",
    L2_VALUE
)

print(
    "Dropout            :",
    DROPOUT_RATE
)

print(
    "Best Val Loss      :",
    f"{best_val_loss:.4f}"
)

print(
    "Best Val Accuracy  :",
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "Final Val Loss     :",
    f"{final_val_loss:.4f}"
)

print(
    "Final Val Accuracy :",
    f"{final_val_accuracy * 100:.2f}%"
)

print(
    "\nOptimized model saved:"
)

print(
    MODEL_PATH
)

print(
    "\n=========================================="
)