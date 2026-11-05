
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

X_PATH = os.path.join(DATA_DIR, "X_train.npy")

GENERATOR_PATH = os.path.join(MODEL_DIR, "ares_gan_generator.keras")
DISCRIMINATOR_PATH = os.path.join(MODEL_DIR, "ares_gan_discriminator.keras")
HISTORY_PATH = os.path.join(RESULTS_DIR, "gan_training_history.csv")
REAL_TEST_PATH = os.path.join(DATA_DIR, "gan_real_test.npy")

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

LATENT_DIM = 32
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.0002

np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\n========== ARES 2.0 GAN TRAINING ==========\n")

if not os.path.exists(X_PATH):
    raise FileNotFoundError(f"Training data not found: {X_PATH}")

X = np.load(X_PATH).astype(np.float32)

if X.ndim != 2:
    raise ValueError("X_train.npy must contain a 2D feature matrix.")

if not np.isfinite(X).all():
    raise ValueError("Training data contains NaN or infinite values.")

X_train, X_test = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)

np.save(REAL_TEST_PATH, X_test)

FEATURE_DIM = X_train.shape[1]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Feature count:", FEATURE_DIM)

# --------------------------------------------------
# GENERATOR
# --------------------------------------------------

def build_generator():

    model = Sequential([
        Input(shape=(LATENT_DIM,)),
        Dense(128),
        LeakyReLU(negative_slope=0.2),
        Dense(256),
        LeakyReLU(negative_slope=0.2),
        Dense(512),
        LeakyReLU(negative_slope=0.2),
        Dense(FEATURE_DIM, activation="linear")
    ], name="ARES_Generator")

    return model


# --------------------------------------------------
# DISCRIMINATOR
# --------------------------------------------------

def build_discriminator():

    model = Sequential([
        Input(shape=(FEATURE_DIM,)),
        Dense(256),
        LeakyReLU(negative_slope=0.2),
        Dense(128),
        LeakyReLU(negative_slope=0.2),
        Dense(64),
        LeakyReLU(negative_slope=0.2),
        Dense(1, activation="sigmoid")
    ], name="ARES_Discriminator")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


generator = build_generator()
discriminator = build_discriminator()

# --------------------------------------------------
# COMBINED GAN
# --------------------------------------------------

discriminator.trainable = False

gan_input = Input(shape=(LATENT_DIM,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)

gan = tf.keras.Model(
    gan_input,
    gan_output,
    name="ARES_GAN"
)

gan.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy"
)

discriminator.trainable = True

# --------------------------------------------------
# TRAINING
# --------------------------------------------------

history = []

steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)

print("\nStarting GAN training...\n")

for epoch in range(EPOCHS):

    d_losses = []
    d_accuracies = []
    g_losses = []

    for step in range(steps_per_epoch):

        # Real samples
        indexes = np.random.randint(0, len(X_train), BATCH_SIZE)
        real_samples = X_train[indexes]

        # Fake samples
        noise = np.random.normal(
            0, 1, (BATCH_SIZE, LATENT_DIM)
        ).astype(np.float32)

        fake_samples = generator.predict(noise, verbose=0)

        # Train discriminator
        real_labels = np.ones((BATCH_SIZE, 1), dtype=np.float32) * 0.9
        fake_labels = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

        d_real = discriminator.train_on_batch(
            real_samples, real_labels
        )

        d_fake = discriminator.train_on_batch(
            fake_samples, fake_labels
        )

        d_loss = (float(d_real[0]) + float(d_fake[0])) / 2
        d_accuracy = (float(d_real[1]) + float(d_fake[1])) / 2

        # Train generator through combined GAN
        noise = np.random.normal(
            0, 1, (BATCH_SIZE, LATENT_DIM)
        ).astype(np.float32)

        misleading_labels = np.ones(
            (BATCH_SIZE, 1), dtype=np.float32
        )

        g_loss = gan.train_on_batch(
            noise, misleading_labels
        )

        d_losses.append(d_loss)
        d_accuracies.append(d_accuracy)
        g_losses.append(float(np.asarray(g_loss).mean()))

    epoch_d_loss = float(np.mean(d_losses))
    epoch_d_accuracy = float(np.mean(d_accuracies))
    epoch_g_loss = float(np.mean(g_losses))

    history.append({
        "Epoch": epoch + 1,
        "Discriminator_Loss": epoch_d_loss,
        "Discriminator_Accuracy": epoch_d_accuracy,
        "Generator_Loss": epoch_g_loss
    })

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"D Loss: {epoch_d_loss:.4f} | "
            f"D Accuracy: {epoch_d_accuracy:.4f} | "
            f"G Loss: {epoch_g_loss:.4f}"
        )

# --------------------------------------------------
# SAVE MODELS AND HISTORY
# --------------------------------------------------

generator.save(GENERATOR_PATH)

discriminator.trainable = True
discriminator.save(DISCRIMINATOR_PATH)

pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

print("\n========== TRAINING COMPLETED ==========")
print("Generator saved:", GENERATOR_PATH)
print("Discriminator saved:", DISCRIMINATOR_PATH)
print("Training history saved:", HISTORY_PATH)
print("Real test data saved:", REAL_TEST_PATH)
