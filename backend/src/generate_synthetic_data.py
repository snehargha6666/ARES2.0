
import os
import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

GENERATOR_PATH = os.path.join(
    MODEL_DIR, "ares_gan_generator.keras"
)

OUTPUT_CSV = os.path.join(
    DATA_DIR, "synthetic_disaster_data.csv"
)

OUTPUT_NPY = os.path.join(
    DATA_DIR, "synthetic_gan_samples.npy"
)

LATENT_DIM = 32
NUM_SAMPLES = 1000

print("\n========== GENERATING SYNTHETIC DATA ==========\n")

if not os.path.exists(GENERATOR_PATH):
    raise FileNotFoundError(
        "GAN generator model not found. Run train_gan.py first."
    )

generator = tf.keras.models.load_model(
    GENERATOR_PATH,
    compile=False
)

noise = np.random.normal(
    0, 1, (NUM_SAMPLES, LATENT_DIM)
).astype(np.float32)

synthetic_data = generator.predict(
    noise,
    verbose=0
)

feature_names = [
    f"Feature_{i + 1}"
    for i in range(synthetic_data.shape[1])
]

df = pd.DataFrame(
    synthetic_data,
    columns=feature_names
)

df.to_csv(OUTPUT_CSV, index=False)
np.save(OUTPUT_NPY, synthetic_data)

print("Synthetic records generated:", len(df))
print("Feature count:", len(feature_names))
print("CSV saved:", OUTPUT_CSV)
print("NumPy data saved:", OUTPUT_NPY)
