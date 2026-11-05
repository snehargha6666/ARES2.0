
import os
import re
import json
import time
import random
import glob

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    SimpleRNN,
    LSTM,
    GRU,
    Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# ============================================================
# ARES 2.0 - MODULE 4
# SEQUENCE BASED DEEP LEARNING
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

WINDOW_SIZE = 5
EPOCHS = 80
BATCH_SIZE = 32
PATIENCE = 12
LEARNING_RATE = 0.001

MODEL_NAMES = ["RNN", "LSTM", "GRU", "BiLSTM"]

# ============================================================
# 1. DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n============================================")
print("ARES 2.0 - MODULE 4")
print("SEQUENCE BASED MODELS")
print("============================================")


# ============================================================
# 2. FIND DATASET
# ============================================================

possible_files = [
    os.path.join(DATA_DIR, "disaster_severity.csv"),
    os.path.join(DATA_DIR, "processed_data.csv"),
    os.path.join(DATA_DIR, "disaster_data.csv"),
    os.path.join(DATA_DIR, "cleaned_data.csv"),
    os.path.join(DATA_DIR, "final_dataset.csv"),
    os.path.join(BASE_DIR, "disaster_severity.csv"),
]

DATA_PATH = next(
    (path for path in possible_files if os.path.isfile(path)),
    None
)

if DATA_PATH is None:
    candidates = []

    for folder in [
        DATA_DIR,
        os.path.join(BASE_DIR, "datasets"),
        os.path.join(BASE_DIR, "data", "processed"),
        os.path.join(BASE_DIR, "datasets", "processed")
    ]:
        if os.path.isdir(folder):
            candidates.extend(
                glob.glob(os.path.join(folder, "*.csv"))
            )

    candidates = [
        p for p in candidates
        if not os.path.basename(p).startswith("module")
        and "history" not in os.path.basename(p).lower()
        and "prediction" not in os.path.basename(p).lower()
    ]

    if len(candidates) == 1:
        DATA_PATH = candidates[0]

    else:
        raise FileNotFoundError(
            "\nCould not automatically identify the Module 1 dataset.\n"
            "Place your processed dataset at:\n"
            + os.path.join(DATA_DIR, "disaster_severity.csv")
            + "\n\nAvailable CSV files:\n"
            + "\n".join(candidates)
        )

print("\nDataset:", DATA_PATH)


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.astype(str).str.strip()

print("Original dataset shape:", df.shape)

if len(df) == 0:
    raise ValueError("Dataset is empty.")


# ============================================================
# 4. DETECT TARGET COLUMN
# ============================================================

target_candidates = [
    "Severity",
    "Disaster Severity",
    "Severity Level",
    "Disaster Severity Class",
    "Target",
    "Label",
    "Class"
]

lookup = {
    col.lower(): col
    for col in df.columns
}

TARGET_COLUMN = None

for candidate in target_candidates:
    if candidate.lower() in lookup:
        TARGET_COLUMN = lookup[candidate.lower()]
        break

if TARGET_COLUMN is None:
    raise ValueError(
        "\nSeverity target column not found.\n"
        "Available columns:\n"
        + str(df.columns.tolist())
        + "\nUpdate the target column in your dataset "
        "or add its name to target_candidates."
    )

print("Target column:", TARGET_COLUMN)


# ============================================================
# 5. DETECT DATE COLUMN
# ============================================================

date_candidates = [
    "Start Date",
    "Date",
    "Event Date",
    "Disaster Date"
]

DATE_COLUMN = None

for candidate in date_candidates:
    if candidate.lower() in lookup:
        DATE_COLUMN = lookup[candidate.lower()]
        break

if DATE_COLUMN is not None:

    df["_date"] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

elif "Start Year" in df.columns:

    year = pd.to_numeric(
        df["Start Year"],
        errors="coerce"
    )

    month = (
        pd.to_numeric(df["Start Month"], errors="coerce")
        if "Start Month" in df.columns
        else pd.Series(1, index=df.index)
    )

    day = (
        pd.to_numeric(df["Start Day"], errors="coerce")
        if "Start Day" in df.columns
        else pd.Series(1, index=df.index)
    )

    df["_date"] = pd.to_datetime(
        {
            "year": year,
            "month": month.fillna(1).clip(1, 12),
            "day": day.fillna(1).clip(1, 28)
        },
        errors="coerce"
    )

else:
    raise ValueError(
        "A chronological date is required. "
        "Expected Start Date, Date, Event Date or Start Year."
    )

df = df.dropna(
    subset=["_date", TARGET_COLUMN]
).copy()

df = df.sort_values(
    "_date"
).reset_index(drop=True)

if len(df) <= WINDOW_SIZE:
    raise ValueError("Not enough chronological records.")


# ============================================================
# 6. ENCODE TARGET
# ============================================================

target = df[TARGET_COLUMN]

numeric_target = pd.to_numeric(
    target,
    errors="coerce"
)

if numeric_target.notna().all():

    unique_values = sorted(
        numeric_target.unique().tolist()
    )

    mapping = {
        value: i
        for i, value in enumerate(unique_values)
    }

    y = numeric_target.map(mapping).to_numpy(
        dtype=np.int32
    )

    if unique_values == [0, 1, 2, 3]:
        CLASS_NAMES = [
            "Low",
            "Moderate",
            "High",
            "Critical"
        ]
    else:
        CLASS_NAMES = [
            str(value)
            for value in unique_values
        ]

else:

    normalized = target.astype(str).str.strip().str.lower()

    severity_mapping = {
        "low": "Low",
        "moderate": "Moderate",
        "medium": "Moderate",
        "high": "High",
        "critical": "Critical"
    }

    if normalized.isin(severity_mapping).all():

        mapped = normalized.map(severity_mapping)

        CLASS_NAMES = [
            name for name in [
                "Low",
                "Moderate",
                "High",
                "Critical"
            ]
            if name in mapped.unique()
        ]

        mapping = {
            name: i
            for i, name in enumerate(CLASS_NAMES)
        }

        y = mapped.map(mapping).to_numpy(
            dtype=np.int32
        )

    else:

        encoder = LabelEncoder()

        y = encoder.fit_transform(
            target.astype(str)
        ).astype(np.int32)

        CLASS_NAMES = encoder.classes_.tolist()

NUM_CLASSES = len(CLASS_NAMES)

if NUM_CLASSES < 2:
    raise ValueError("At least two severity classes are required.")

print("\nSeverity classes:", CLASS_NAMES)


# ============================================================
# 7. COUNTRY GROUPING
# ============================================================

GROUP_COLUMN = None

for candidate in [
    "ISO",
    "Country",
    "Country Name"
]:
    if candidate in df.columns:
        GROUP_COLUMN = candidate
        break

if GROUP_COLUMN:
    groups = df[GROUP_COLUMN].fillna("UNKNOWN").astype(str)
    print("Grouping records by:", GROUP_COLUMN)
else:
    groups = pd.Series("ALL", index=df.index)
    print("No country column found. Using global chronology.")


# ============================================================
# 8. PREPARE FEATURES
# ============================================================

excluded = [
    TARGET_COLUMN,
    "_date"
]

if DATE_COLUMN is not None:
    excluded.append(DATE_COLUMN)

if GROUP_COLUMN is not None:
    excluded.append(GROUP_COLUMN)

# Remove record identifiers
for col in df.columns:

    lower = col.lower()

    if any(term in lower for term in [
        "disno",
        "event id",
        "record id",
        "serial number"
    ]):
        excluded.append(col)

features = df.drop(
    columns=list(set(excluded)),
    errors="ignore"
)

# Convert categorical columns to numerical representation
features = pd.get_dummies(
    features,
    dummy_na=True,
    dtype=float
)

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove completely empty features
features = features.dropna(
    axis=1,
    how="all"
)

if features.shape[1] == 0:
    raise ValueError("No usable input features found.")

X_rows = features.to_numpy(
    dtype=np.float32
)

print("Feature count:", X_rows.shape[1])


# ============================================================
# 9. CREATE NEXT-EVENT SEQUENCES
# ============================================================

sequences = []
sequence_labels = []
sequence_dates = []

for group_name in groups.unique():

    indexes = np.where(
        groups.to_numpy() == group_name
    )[0]

    if len(indexes) <= WINDOW_SIZE:
        continue

    group_features = X_rows[indexes]
    group_labels = y[indexes]
    group_dates = df["_date"].iloc[indexes].to_numpy()

    for i in range(WINDOW_SIZE, len(indexes)):

        # Previous five events
        sequence = group_features[
            i - WINDOW_SIZE:i
        ]

        # Next event's severity
        label = group_labels[i]

        sequences.append(sequence)
        sequence_labels.append(label)
        sequence_dates.append(group_dates[i])

if len(sequences) < 30:
    raise ValueError(
        f"Only {len(sequences)} sequences generated. "
        "At least 30 are required. Check dates and country grouping."
    )

X = np.asarray(sequences, dtype=np.float32)
y_seq = np.asarray(sequence_labels, dtype=np.int32)

sequence_dates = pd.to_datetime(sequence_dates)

order = np.argsort(sequence_dates.to_numpy())

X = X[order]
y_seq = y_seq[order]
sequence_dates = sequence_dates[order]

print("\nSequence shape:", X.shape)
print("Sequence labels:", y_seq.shape)


# ============================================================
# 10. CHRONOLOGICAL DATA SPLIT
# ============================================================

total = len(X)

train_end = int(total * 0.70)
val_end = int(total * 0.85)

if train_end < 1 or val_end <= train_end or val_end >= total:
    raise ValueError("Insufficient data for temporal splitting.")

X_train_raw = X[:train_end]
X_val_raw = X[train_end:val_end]
X_test_raw = X[val_end:]

y_train = y_seq[:train_end]
y_val = y_seq[train_end:val_end]
y_test = y_seq[val_end:]

print("\n========== DATA SPLIT ==========")

print("Training:", X_train_raw.shape)
print("Validation:", X_val_raw.shape)
print("Testing:", X_test_raw.shape)


# ============================================================
# 11. IMPUTATION AND NORMALIZATION
# ============================================================

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

feature_count = X.shape[-1]

train_flat = X_train_raw.reshape(-1, feature_count)
val_flat = X_val_raw.reshape(-1, feature_count)
test_flat = X_test_raw.reshape(-1, feature_count)

# Fit only on training data
train_flat = imputer.fit_transform(train_flat)
val_flat = imputer.transform(val_flat)
test_flat = imputer.transform(test_flat)

train_flat = scaler.fit_transform(train_flat)
val_flat = scaler.transform(val_flat)
test_flat = scaler.transform(test_flat)

X_train = train_flat.reshape(
    -1, WINDOW_SIZE, feature_count
).astype(np.float32)

X_val = val_flat.reshape(
    -1, WINDOW_SIZE, feature_count
).astype(np.float32)

X_test = test_flat.reshape(
    -1, WINDOW_SIZE, feature_count
).astype(np.float32)


# ============================================================
# 12. CLASS BALANCING
# ============================================================

present_classes = np.unique(y_train)

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=present_classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        present_classes,
        class_weight_values
    )
}

print("\nClass weights:", class_weights)


# ============================================================
# 13. MODEL ARCHITECTURES
# ============================================================

INPUT_SHAPE = (
    WINDOW_SIZE,
    feature_count
)


def build_model(model_name):

    model = Sequential(name=f"ARES_{model_name}")

    model.add(Input(shape=INPUT_SHAPE))

    if model_name == "RNN":

        model.add(
            SimpleRNN(
                64,
                activation="tanh"
            )
        )

    elif model_name == "LSTM":

        model.add(
            LSTM(64)
        )

    elif model_name == "GRU":

        model.add(
            GRU(64)
        )

    elif model_name == "BiLSTM":

        model.add(
            Bidirectional(
                LSTM(64)
            )
        )

    model.add(Dropout(0.30))

    model.add(
        Dense(
            32,
            activation="relu"
        )
    )

    model.add(
        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 14. TRAINING AND EVALUATION
# ============================================================

all_metrics = []
all_reports = []
gradient_records = []

for model_name in MODEL_NAMES:

    print("\n============================================")
    print("TRAINING MODEL:", model_name)
    print("============================================")

    model = build_model(model_name)

    model.summary()

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    training_time = time.time() - start_time

    # Save model
    model_path = os.path.join(
        MODEL_DIR,
        f"ares_module4_{model_name.lower()}.keras"
    )

    model.save(model_path)

    # Predict
    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision_weighted = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    precision_macro = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(NUM_CLASSES)
    )

    try:
        auc_score = roc_auc_score(
            y_test,
            probabilities,
            labels=np.arange(NUM_CLASSES),
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        auc_score = np.nan

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    all_reports.append(
        f"\nMODEL: {model_name}\n\n{report}\n"
    )

    # Save predictions
    prediction_df = pd.DataFrame({
        "Actual": [CLASS_NAMES[i] for i in y_test],
        "Predicted": [CLASS_NAMES[i] for i in predictions]
    })

    for i, class_name in enumerate(CLASS_NAMES):

        safe_class = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            class_name
        )

        prediction_df[
            f"Probability_{safe_class}"
        ] = probabilities[:, i]

    prediction_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"module4_predictions_{model_name.lower()}.csv"
        ),
        index=False
    )

    # Save confusion matrix
    np.save(
        os.path.join(
            RESULTS_DIR,
            f"module4_confusion_matrix_{model_name.lower()}.npy"
        ),
        cm
    )

    # Save history
    history_df = pd.DataFrame(history.history)

    history_df.insert(
        0,
        "Epoch",
        np.arange(1, len(history_df) + 1)
    )

    history_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"module4_history_{model_name.lower()}.csv"
        ),
        index=False
    )

    # Input gradient analysis
    sample_count = min(64, len(X_test))

    input_tensor = tf.convert_to_tensor(
        X_test[:sample_count],
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        tape.watch(input_tensor)

        output = model(
            input_tensor,
            training=False
        )

        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_test[:sample_count],
            output
        )

        loss = tf.reduce_mean(loss)

    gradients = tape.gradient(
        loss,
        input_tensor
    )

    gradient_norms = tf.reduce_mean(
        tf.abs(gradients),
        axis=[0, 2]
    ).numpy()

    for timestep, value in enumerate(gradient_norms):

        gradient_records.append({
            "Model": model_name,
            "Timestep": timestep + 1,
            "Mean_Absolute_Input_Gradient": float(value)
        })

    # Store results
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision_Weighted": precision_weighted,
        "Recall_Weighted": recall_weighted,
        "F1_Weighted": f1_weighted,
        "Precision_Macro": precision_macro,
        "Recall_Macro": recall_macro,
        "F1_Macro": f1_macro,
        "ROC_AUC": auc_score,
        "Training_Time_Seconds": training_time,
        "Epochs_Trained": len(history.history["loss"])
    }

    all_metrics.append(metrics)

    print("\nMODEL RESULTS")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("Saved model:", model_path)


# ============================================================
# 15. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(all_metrics)

metrics_df.to_csv(
    os.path.join(RESULTS_DIR, "module4_metrics.csv"),
    index=False
)

with open(
    os.path.join(
        RESULTS_DIR,
        "module4_classification_reports.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "ARES 2.0 - MODULE 4 CLASSIFICATION REPORTS\n"
    )

    for report in all_reports:
        file.write(report)

pd.DataFrame(gradient_records).to_csv(
    os.path.join(
        RESULTS_DIR,
        "module4_gradient_profile.csv"
    ),
    index=False
)


# ============================================================
# 16. SAVE METADATA AND PREPROCESSORS
# ============================================================

import joblib

joblib.dump(
    imputer,
    os.path.join(RESULTS_DIR, "module4_imputer.pkl")
)

joblib.dump(
    scaler,
    os.path.join(RESULTS_DIR, "module4_scaler.pkl")
)

metadata = {
    "module": 4,
    "window_size": WINDOW_SIZE,
    "number_of_features": int(feature_count),
    "number_of_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "models": MODEL_NAMES,
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "testing_samples": int(len(X_test)),
    "sequence_method": "Previous five chronological events predict next event",
    "group_column": GROUP_COLUMN,
    "date_column": DATE_COLUMN,
    "target_column": TARGET_COLUMN,
    "feature_columns": features.columns.tolist()
}

with open(
    os.path.join(
        RESULTS_DIR,
        "module4_metadata.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n============================================")
print("MODULE 4 TRAINING COMPLETED")
print("============================================")

print(metrics_df.to_string(index=False))

print("\nModels saved in:", MODEL_DIR)
print("Metrics saved in:", RESULTS_DIR)

print("\nAll four sequence models trained successfully.")
