#!/usr/bin/env python3
"""
Convolutional auto-encoder for 15000x484 mel-spectrogram patches.
Assumes you already have the five .npz files in ./features/
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
FEATURE_DIR = "features"          # adjust if your files live elsewhere
npz_files = sorted([f for f in os.listdir(FEATURE_DIR) if f.endswith(".npz")])
print(f"Found {len(npz_files)} feature files: {npz_files}")

# Load and stack
mats = []
freqs = np.array([])  # placeholder
for f in npz_files:
    data = np.load(os.path.join(FEATURE_DIR, f))
    mats.append(data["matrix"].astype(np.float32))  # ensure float32
    if freqs.size == 0:
        freqs = data["freqs"]  # same for all clips
X = np.stack(mats, axis=0)      # shape (N, 15000, 484)
print(f"Data shape: {X.shape}")

# Add a channel dimension for Conv2D (H, W, C)
X = X[..., np.newaxis]          # (N, 15000, 484, 1)

# ----------------------------------------------------------------------
# 2. Build a convolutional auto-encoder that pools only along time axis
# ----------------------------------------------------------------------
def build_autoencoder(input_shape):
    """Returns (autoencoder, encoder, decoder) models."""
    inp = keras.Input(shape=input_shape, name="spectrogram_input")

    # ----- Encoder -----
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D((2, 1), padding="same")(x)          # -> (7500, 484, 16)
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 1), padding="same")(x)          # -> (3750, 484, 8)
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2, 1), padding="same")(x)    # -> (1875, 484, 8)

    # ----- Decoder -----
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 1))(x)                          # -> (3750, 484, 8)
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 1))(x)                          # -> (7500, 484, 8)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 1))(x)                          # -> (15000, 484, 16)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)  # back to 1 channel

    autoencoder = keras.Model(inp, x, name="autoencoder")
    encoder = keras.Model(inp, encoded, name="encoder")

    # Decoder model (optional, for standalone decoding)
    encoded_input = keras.Input(shape=encoded.shape[1:])
    # Retrieve the decoder layers from the autoencoder (last 6 layers after encoding)
    decoder_layers = autoencoder.layers[-6:]  # adjust if you change encoder depth
    x = encoded_input
    for layer in decoder_layers:
        x = layer(x)
    decoder = keras.Model(encoded_input, x, name="decoder")

    return autoencoder, encoder, decoder

autoencoder, encoder, decoder = build_autoencoder(input_shape=X.shape[1:])
autoencoder.summary()

autoencoder.compile(optimizer="adam", loss="mse")

# ----------------------------------------------------------------------
# 3. Train
# ----------------------------------------------------------------------
EPOCHS = 30          # feel free to increase
BATCH_SIZE = 2

history = autoencoder.fit(
    X, X,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2,
    verbose=2,
)

# ----------------------------------------------------------------------
# 4. Reconstruct & save
# ----------------------------------------------------------------------
recon = autoencoder.predict(X, batch_size=BATCH_SIZE)
recon = np.squeeze(recon, axis=-1)   # back to (N, 15000, 484)

OUT_DIR = "reconstructed"
os.makedirs(OUT_DIR, exist_ok=True)
for i, fname in enumerate(npz_files):
    out_path = os.path.join(OUT_DIR, f"recon_{fname}")
    np.savez_compressed(
        out_path,
        matrix=recon[i].astype(np.float32),
        freqs=freqs,
    )
    print(f"Saved reconstruction to {out_path}")

# ----------------------------------------------------------------------
# 5. Quick visual check (optional)
# ----------------------------------------------------------------------
def plot_spectrogram(spec, title="", ax=None, vmin=0, vmax=1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(
        spec.T, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax
    )
    ax.set_title(title)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax

# Plot first sample
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
plot_spectrogram(X[0, :, :, 0], title="Original", ax=axs[0])
plot_spectrogram(recon[0], title="Reconstructed", ax=axs[1])
plt.tight_layout()
plt.savefig("reconstruction_example.png", dpi=150)
print("Saved example comparison to reconstruction_example.png")