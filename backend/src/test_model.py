import os
import sys
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
# ARES 2.0 - MODEL TESTING
# Disaster Severity Prediction
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ares_mlp.keras"
)

X_TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "X_test.npy"
)

Y_TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "y_test.npy"
)

REPORT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "model_test_report.txt"
)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Critical"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_result(test_name, passed):
    """Print a standardized PASS/FAIL result."""

    status = "PASS" if passed else "FAIL"

    print(
        f"{test_name:<45} : {status}"
    )

    return passed


def probabilities_are_valid(probabilities):
    """
    Check that:
    - probabilities are between 0 and 1
    - every row sums approximately to 1
    """

    valid_range = np.all(
        (probabilities >= 0) &
        (probabilities <= 1)
    )

    row_sums = np.sum(
        probabilities,
        axis=1
    )

    valid_sum = np.allclose(
        row_sums,
        1.0,
        atol=1e-5
    )

    return valid_range and valid_sum


# ============================================================
# MAIN TESTING
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("ARES 2.0 - MODEL TESTING")
    print("Disaster Severity Prediction")
    print("=" * 60)


    # ========================================================
    # TEST 1 - FILE EXISTENCE
    # ========================================================

    print("\n========== TEST 1: FILE VALIDATION ==========")

    model_exists = os.path.exists(MODEL_PATH)
    x_test_exists = os.path.exists(X_TEST_PATH)
    y_test_exists = os.path.exists(Y_TEST_PATH)

    print_result(
        "Saved model exists",
        model_exists
    )

    print_result(
        "X_test.npy exists",
        x_test_exists
    )

    print_result(
        "y_test.npy exists",
        y_test_exists
    )

    if not model_exists:
        print("\nERROR: Model file not found.")
        sys.exit(1)

    if not x_test_exists:
        print("\nERROR: X_test.npy not found.")
        sys.exit(1)

    if not y_test_exists:
        print("\nERROR: y_test.npy not found.")
        sys.exit(1)


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\n========== LOADING MODEL ==========")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    input_features = model.input_shape[-1]
    output_classes = model.output_shape[-1]

    print(
        f"Input features : {input_features}"
    )

    print(
        f"Output classes : {output_classes}"
    )


    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print("\n========== LOADING TEST DATA ==========")

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print(
        f"X_test shape : {X_test.shape}"
    )

    print(
        f"y_test shape : {y_test.shape}"
    )


    # ========================================================
    # TEST 2 - INPUT SHAPE
    # ========================================================

    print("\n========== TEST 2: INPUT VALIDATION ==========")

    input_shape_valid = (
        X_test.ndim == 2 and
        X_test.shape[1] == input_features
    )

    print_result(
        "Test data has correct dimensions",
        input_shape_valid
    )

    if not input_shape_valid:
        print(
            f"\nERROR: Model expects {input_features} "
            f"features but received "
            f"{X_test.shape[1]}."
        )
        sys.exit(1)


    # ========================================================
    # TEST 3 - TARGET SHAPE
    # ========================================================

    print("\n========== TEST 3: TARGET VALIDATION ==========")

    target_shape_valid = (
        len(y_test) == len(X_test)
    )

    print_result(
        "X_test and y_test have same number of samples",
        target_shape_valid
    )

    if not target_shape_valid:
        print("\nERROR: X_test and y_test size mismatch.")
        sys.exit(1)


    # ========================================================
    # TEST 4 - FULL MODEL PREDICTION
    # ========================================================

    print("\n========== TEST 4: FULL DATASET PREDICTION ==========")

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    print(
        f"Predictions generated: {len(predictions)}"
    )

    prediction_count_valid = (
        len(predictions) == len(X_test)
    )

    print_result(
        "Prediction generated for every test sample",
        prediction_count_valid
    )


    # ========================================================
    # TEST 5 - PROBABILITY VALIDATION
    # ========================================================

    print("\n========== TEST 5: PROBABILITY VALIDATION ==========")

    probability_valid = probabilities_are_valid(
        probabilities
    )

    print_result(
        "Probabilities are valid and sum to 1",
        probability_valid
    )


    # ========================================================
    # TEST 6 - CLASS VALIDATION
    # ========================================================

    print("\n========== TEST 6: CLASS VALIDATION ==========")

    valid_classes = np.all(
        np.isin(
            predictions,
            list(CLASS_NAMES.keys())
        )
    )

    print_result(
        "All predictions belong to valid classes",
        valid_classes
    )


    # ========================================================
    # TEST 7 - MODEL PERFORMANCE
    # ========================================================

    print("\n========== TEST 7: MODEL PERFORMANCE ==========")

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n========== CLASSIFICATION REPORT ==========")

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            CLASS_NAMES[i]
            for i in range(len(CLASS_NAMES))
        ],
        zero_division=0
    )

    print(report)


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\n========== CONFUSION MATRIX ==========")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    header = (
        f"{'':12}"
        f"{'Low':>10}"
        f"{'Moderate':>10}"
        f"{'High':>10}"
        f"{'Critical':>10}"
    )

    print(header)

    for i, row in enumerate(cm):

        row_text = (
            f"{CLASS_NAMES[i]:12}"
        )

        for value in row:
            row_text += f"{value:10d}"

        print(row_text)


    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print("\n========== PREDICTION DISTRIBUTION ==========")

    for class_id, class_name in CLASS_NAMES.items():

        count = np.sum(
            predictions == class_id
        )

        percentage = (
            count / len(predictions)
        ) * 100

        print(
            f"{class_name:<10}: "
            f"{count:>4} "
            f"({percentage:>6.2f}%)"
        )


    # ========================================================
    # SAMPLE PREDICTIONS
    # ========================================================

    print("\n========== SAMPLE PREDICTIONS ==========")

    number_of_samples = min(
        10,
        len(X_test)
    )

    print(
        f"{'Sample':<10}"
        f"{'Actual':<12}"
        f"{'Predicted':<12}"
        f"{'Confidence':<12}"
    )

    for i in range(number_of_samples):

        actual = CLASS_NAMES[
            int(y_test[i])
        ]

        predicted = CLASS_NAMES[
            int(predictions[i])
        ]

        confidence = (
            probabilities[i][predictions[i]]
            * 100
        )

        print(
            f"{i + 1:<10}"
            f"{actual:<12}"
            f"{predicted:<12}"
            f"{confidence:.2f}%"
        )


    # ========================================================
    # SAVE TEST REPORT
    # ========================================================

    print("\n========== SAVING TEST REPORT ==========")

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ARES 2.0 - MODEL TEST REPORT\n"
        )

        file.write(
            "Disaster Severity Prediction\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Model: {MODEL_PATH}\n"
        )

        file.write(
            f"Test samples: {len(X_test)}\n"
        )

        file.write(
            f"Input features: {input_features}\n"
        )

        file.write(
            f"Output classes: {output_classes}\n\n"
        )

        file.write(
            "METRICS\n"
        )

        file.write(
            f"Accuracy: {accuracy:.4f}\n"
        )

        file.write(
            f"Precision: {precision:.4f}\n"
        )

        file.write(
            f"Recall: {recall:.4f}\n"
        )

        file.write(
            f"F1 Score: {f1:.4f}\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            report
        )

        file.write(
            "\nCONFUSION MATRIX\n"
        )

        file.write(
            str(cm)
        )

        file.write(
            "\n\nTESTS\n"
        )

        file.write(
            f"Model exists: {model_exists}\n"
        )

        file.write(
            f"Input shape valid: {input_shape_valid}\n"
        )

        file.write(
            f"Target shape valid: {target_shape_valid}\n"
        )

        file.write(
            f"Prediction count valid: "
            f"{prediction_count_valid}\n"
        )

        file.write(
            f"Probability validation: "
            f"{probability_valid}\n"
        )

        file.write(
            f"Class validation: "
            f"{valid_classes}\n"
        )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    all_tests_passed = all([
        model_exists,
        x_test_exists,
        y_test_exists,
        input_shape_valid,
        target_shape_valid,
        prediction_count_valid,
        probability_valid,
        valid_classes
    ])

    print(
        f"\nTest report saved to:\n"
        f"{REPORT_PATH}"
    )

    print("\n" + "=" * 60)

    if all_tests_passed:
        print(
            "ALL PIPELINE TESTS PASSED"
        )
    else:
        print(
            "SOME PIPELINE TESTS FAILED"
        )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()