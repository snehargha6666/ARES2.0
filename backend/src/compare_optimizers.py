import os
import csv
import numpy as np
import tensorflow as tf


# ============================================================
# ARES 2.0 - MODULE 2
# OPTIMIZER COMPARISON
# ============================================================

print("\n==========================================")
print("ARES 2.0 - MODULE 2")
print("OPTIMIZER COMPARISON")
print("==========================================\n")


# ============================================================
# 1. PATHS
# ============================================================

DATA_DIR = "data"

X_TRAIN_PATH = os.path.join(
    DATA_DIR,
    "X_train.npy"
)

X_VAL_PATH = os.path.join(
    DATA_DIR,
    "X_val.npy"
)

Y_TRAIN_PATH = os.path.join(
    DATA_DIR,
    "y_train.npy"
)

Y_VAL_PATH = os.path.join(
    DATA_DIR,
    "y_val.npy"
)

RESULTS_PATH = os.path.join(
    DATA_DIR,
    "optimization_results.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

INPUT_SIZE = 119
NUM_CLASSES = 4

EPOCHS = 50
BATCH_SIZE = 32

LEARNING_RATE = 0.001

L2_VALUE = 0.001
DROPOUT_RATE = 0.30

PATIENCE = 8


# ============================================================
# 3. RANDOM SEED
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 4. LOAD DATA
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
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)


# ============================================================
# 5. VERIFY DATA
# ============================================================

if X_train.shape[1] != INPUT_SIZE:

    raise ValueError(
        f"Expected {INPUT_SIZE} features, "
        f"but found {X_train.shape[1]}"
    )


# ============================================================
# 6. CREATE DATASETS
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
# 7. BUILD MODEL
# ============================================================

def build_model():

    regularizer = tf.keras.regularizers.l2(
        L2_VALUE
    )


    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(INPUT_SIZE,)
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizer
        ),

        tf.keras.layers.Dropout(
            DROPOUT_RATE
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizer
        ),

        tf.keras.layers.Dropout(
            DROPOUT_RATE
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=regularizer
        ),

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])


    return model


# ============================================================
# 8. CREATE OPTIMIZER
# ============================================================

def create_optimizer(
    optimizer_name
):

    if optimizer_name == "Adam":

        return tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )


    elif optimizer_name == "RMSprop":

        return tf.keras.optimizers.RMSprop(
            learning_rate=LEARNING_RATE
        )


    elif optimizer_name == "SGD":

        return tf.keras.optimizers.SGD(
            learning_rate=LEARNING_RATE,
            momentum=0.9
        )


    else:

        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )


# ============================================================
# 9. LOSS FUNCTION
# ============================================================

loss_function = (
    tf.keras.losses.SparseCategoricalCrossentropy()
)


# ============================================================
# 10. TRAIN ONE MODEL
# ============================================================

def train_optimizer(
    optimizer_name
):

    print(
        "\n------------------------------------------"
    )

    print(
        f"Testing optimizer: {optimizer_name}"
    )

    print(
        "------------------------------------------"
    )


    # --------------------------------------------------------
    # Reset random seed
    # --------------------------------------------------------

    np.random.seed(42)
    tf.random.set_seed(42)


    # --------------------------------------------------------
    # Build new model
    # --------------------------------------------------------

    model = build_model()


    # --------------------------------------------------------
    # Create optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(
        optimizer_name
    )


    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    best_val_accuracy = 0.0

    best_epoch = 0

    epochs_without_improvement = 0

    best_weights = None


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        training_loss = 0.0

        training_samples = 0


        for x_batch, y_batch in train_dataset:

            with tf.GradientTape() as tape:

                predictions = model(
                    x_batch,
                    training=True
                )


                classification_loss = (
                    loss_function(
                        y_batch,
                        predictions
                    )
                )


                regularization_loss = (
                    tf.add_n(
                        model.losses
                    )
                )


                total_loss = (
                    classification_loss
                    +
                    regularization_loss
                )


            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            gradients = tape.gradient(
                total_loss,
                model.trainable_variables
            )


            # ------------------------------------------------
            # WEIGHT UPDATE
            # ------------------------------------------------

            optimizer.apply_gradients(
                zip(
                    gradients,
                    model.trainable_variables
                )
            )


            batch_size = int(
                tf.shape(y_batch)[0]
            )


            training_loss += (
                float(total_loss)
                *
                batch_size
            )


            training_samples += (
                batch_size
            )


        training_loss /= training_samples


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_loss = 0.0

        validation_correct = 0

        validation_samples = 0


        for x_batch, y_batch in val_dataset:

            predictions = model(
                x_batch,
                training=False
            )


            classification_loss = (
                loss_function(
                    y_batch,
                    predictions
                )
            )


            regularization_loss = (
                tf.add_n(
                    model.losses
                )
            )


            total_validation_loss = (
                classification_loss
                +
                regularization_loss
            )


            batch_size = int(
                tf.shape(y_batch)[0]
            )


            validation_loss += (
                float(total_validation_loss)
                *
                batch_size
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


            validation_correct += int(
                correct
            )

            validation_samples += (
                batch_size
            )


        validation_loss /= (
            validation_samples
        )


        validation_accuracy = (
            validation_correct
            /
            validation_samples
        )


        # ----------------------------------------------------
        # PRINT PROGRESS
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {training_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )


        # ----------------------------------------------------
        # CHECK BEST MODEL
        # ----------------------------------------------------

        if validation_loss < best_val_loss:

            best_val_loss = (
                validation_loss
            )

            best_val_accuracy = (
                validation_accuracy
            )

            best_epoch = epoch

            epochs_without_improvement = 0


            best_weights = [
                variable.numpy()
                for variable
                in model.trainable_variables
            ]


        else:

            epochs_without_improvement += 1


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"Early stopping at epoch {epoch}."
            )

            break


    # ========================================================
    # RESTORE BEST WEIGHTS
    # ========================================================

    if best_weights is not None:

        for variable, best_value in zip(
            model.trainable_variables,
            best_weights
        ):

            variable.assign(
                best_value
            )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        f"\n{optimizer_name} result:"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best epoch: {best_epoch}"
    )


    return {
        "Optimizer": optimizer_name,
        "Learning_Rate": LEARNING_RATE,
        "L2": L2_VALUE,
        "Dropout": DROPOUT_RATE,
        "Best_Epoch": best_epoch,
        "Best_Validation_Loss": best_val_loss,
        "Best_Validation_Accuracy": (
            best_val_accuracy
        )
    }


# ============================================================
# 11. RUN COMPARISON
# ============================================================

print(
    "\n========== OPTIMIZER COMPARISON =========="
)

optimizers = [
    "Adam",
    "RMSprop",
    "SGD"
]


results = []


for optimizer_name in optimizers:

    result = train_optimizer(
        optimizer_name
    )

    results.append(
        result
    )


# ============================================================
# 12. FIND BEST OPTIMIZER
# ============================================================

best_result = max(
    results,
    key=lambda x:
        x["Best_Validation_Accuracy"]
)


# ============================================================
# 13. SAVE CSV
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Optimizer",
            "Learning_Rate",
            "L2",
            "Dropout",
            "Best_Epoch",
            "Best_Validation_Loss",
            "Best_Validation_Accuracy"
        ]
    )


    writer.writeheader()


    for result in results:

        writer.writerow(
            result
        )


# ============================================================
# 14. FINAL REPORT
# ============================================================

print(
    "\n=========================================="
)

print(
    "OPTIMIZER COMPARISON COMPLETE"
)

print(
    "=========================================="
)


for result in results:

    print(
        f"{result['Optimizer']:8s} | "
        f"Val Loss: "
        f"{result['Best_Validation_Loss']:.4f} | "
        f"Val Accuracy: "
        f"{result['Best_Validation_Accuracy'] * 100:.2f}%"
    )


print(
    "\nBest Optimizer:"
)

print(
    best_result["Optimizer"]
)

print(
    "Best Validation Accuracy:"
)

print(
    f"{best_result['Best_Validation_Accuracy'] * 100:.2f}%"
)


print(
    "\nResults saved to:"
)

print(
    RESULTS_PATH
)


print(
    "\n=========================================="
)