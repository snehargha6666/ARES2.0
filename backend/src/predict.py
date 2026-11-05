import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


# ============================================================
# ARES 2.0 - FINAL DISASTER SEVERITY PREDICTION
# RAW INPUT -> PREPROCESSOR -> FINAL MLP -> SEVERITY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# FINAL MODEL + PREPROCESSOR
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ares_mlp_final.keras"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessor.pkl"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

CATEGORICAL_FEATURES = [
    "Disaster Subgroup",
    "Disaster Type",
    "Disaster Subtype",
    "ISO",
    "Subregion",
    "Magnitude Scale"
]


NUMERICAL_FEATURES = [
    "Start Year",
    "Start Month",
    "Start Day",
    "End Year",
    "End Month",
    "End Day",
    "CPI",
    "Total Deaths",
    "No. Affected",
    "No. Injured",
    "Total Damage ('000 US$)"
]


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
# PREDICTOR CLASS
# ============================================================

class ARESSeverityPredictor:

    def __init__(self):

        print(
            "Loading ARES 2.0 final model...",
            file=sys.stderr
        )


        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"Final model not found: {MODEL_PATH}"
            )


        # ----------------------------------------------------
        # CHECK PREPROCESSOR
        # ----------------------------------------------------

        if not os.path.exists(
            PREPROCESSOR_PATH
        ):

            raise FileNotFoundError(
                f"Preprocessor not found: "
                f"{PREPROCESSOR_PATH}"
            )


        # ----------------------------------------------------
        # LOAD FINAL MODEL
        # ----------------------------------------------------

        self.model = tf.keras.models.load_model(
            MODEL_PATH
        )


        # ----------------------------------------------------
        # LOAD PREPROCESSOR
        # ----------------------------------------------------

        self.preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )


        print(
            "Final model and preprocessor loaded.",
            file=sys.stderr
        )


    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    def validate_input(
        self,
        input_data
    ):

        if not isinstance(
            input_data,
            dict
        ):

            raise ValueError(
                "Input must be a JSON object."
            )


        required_features = (
            CATEGORICAL_FEATURES
            +
            NUMERICAL_FEATURES
        )


        missing_features = []


        for feature in required_features:

            if feature not in input_data:

                missing_features.append(
                    feature
                )


        if missing_features:

            raise ValueError(
                "Missing required fields: "
                +
                ", ".join(
                    missing_features
                )
            )


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        input_data
    ):

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        self.validate_input(
            input_data
        )


        # ----------------------------------------------------
        # Build data dictionary
        # ----------------------------------------------------

        data = {}


        for feature in CATEGORICAL_FEATURES:

            data[feature] = input_data.get(
                feature
            )


        for feature in NUMERICAL_FEATURES:

            value = input_data.get(
                feature
            )


            try:

                data[feature] = float(
                    value
                )

            except (
                TypeError,
                ValueError
            ):

                raise ValueError(
                    f"Invalid numeric value "
                    f"for '{feature}': {value}"
                )


        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        df = pd.DataFrame(
            [data]
        )


        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        processed = (
            self.preprocessor.transform(
                df
            )
        )


        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        if hasattr(
            processed,
            "toarray"
        ):

            processed = (
                processed.toarray()
            )


        processed = np.asarray(
            processed,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Verify feature count
        # ----------------------------------------------------

        if processed.shape[1] != 119:

            raise ValueError(
                "Preprocessed input has "
                f"{processed.shape[1]} features. "
                "Expected 119."
            )


        # ----------------------------------------------------
        # FINAL MODEL PREDICTION
        # ----------------------------------------------------

        probabilities = (
            self.model.predict(
                processed,
                verbose=0
            )[0]
        )


        # ----------------------------------------------------
        # Normalize probabilities
        # ----------------------------------------------------

        probabilities = (
            probabilities
            /
            np.sum(probabilities)
        )


        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        predicted_class = int(
            np.argmax(
                probabilities
            )
        )


        # ----------------------------------------------------
        # Severity
        # ----------------------------------------------------

        severity = CLASS_NAMES[
            predicted_class
        ]


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            probabilities[
                predicted_class
            ]
        )


        # ----------------------------------------------------
        # Probability dictionary
        # ----------------------------------------------------

        probability_dict = {

            CLASS_NAMES[i]:
                float(
                    probabilities[i]
                )

            for i in range(
                len(probabilities)
            )
        }


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {

            "severity":
                severity,

            "class_id":
                predicted_class,

            "confidence":
                confidence,

            "probabilities":
                probability_dict
        }


# ============================================================
# API MODE
# ============================================================

def run_api_prediction():

    try:

        # ----------------------------------------------------
        # Read JSON from Node.js
        # ----------------------------------------------------

        input_data = sys.stdin.read()


        if not input_data.strip():

            raise ValueError(
                "No input data received."
            )


        request = json.loads(
            input_data
        )


        # ----------------------------------------------------
        # Create predictor
        # ----------------------------------------------------

        predictor = (
            ARESSeverityPredictor()
        )


        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        result = predictor.predict(
            request
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # Only JSON is written to stdout.
        # ----------------------------------------------------

        print(
            json.dumps(
                result
            )
        )


    except Exception as e:

        print(
            json.dumps({
                "error": str(e)
            })
        )

        sys.exit(1)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_api_prediction()