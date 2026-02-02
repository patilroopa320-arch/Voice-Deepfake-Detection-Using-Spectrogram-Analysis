import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4

# 🔥 TRAIN FROM SPECTROGRAMS (NOT AUDIO)
DATA_DIR = "spectrograms"   # MUST contain ai/ and human/
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# SAFETY CHECKS
# ==============================
assert os.path.exists(os.path.join(DATA_DIR, "ai")), "❌ spectrograms/ai not found"
assert os.path.exists(os.path.join(DATA_DIR, "human")), "❌ spectrograms/human not found"

print("✅ Spectrogram dataset verified")

# ==============================
# DATA GENERATORS
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,

    # Simulate real-world distortions
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    brightness_range=[0.7, 1.3]
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("\n📌 Class Mapping:", train_data.class_indices)
# Expected → {'ai': 0, 'human': 1}

# ==============================
# CLASS WEIGHT (SAFE)
# ==============================
class_counts = np.bincount(train_data.classes)

if len(class_counts) < 2:
    raise ValueError("❌ Both classes must contain images")

total = np.sum(class_counts)

class_weight = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1])
}

print("⚖️ Class Weights:", class_weight)

# ==============================
# CNN MODEL
# ==============================
model = Sequential([

    Conv2D(32, (3, 3), activation="relu",
           input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(1, activation="sigmoid")
])

model.summary()

# ==============================
# COMPILE
# ==============================
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# CALLBACKS
# ==============================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# ==============================
# TRAIN
# ==============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)

print("\n✅ Retraining completed successfully")
print(f"✅ Best model saved at: {MODEL_PATH}")
