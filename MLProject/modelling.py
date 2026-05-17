import os
import dagshub
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.utils import (
    image_dataset_from_directory
)

# =========================================================
# DAGSHUB INITIALIZATION
# =========================================================

dagshub.init(
    repo_owner="arif76440",
    repo_name="MLFlow-Image-Classification",
    mlflow=True
)

# =========================================================
# MLFLOW CONFIGURATION
# =========================================================

mlflow.set_experiment(
    "Intel_Image_Classification"
)

mlflow.tensorflow.autolog()

# =========================================================
# CONFIGURATION
# =========================================================

DATASET_DIR = "intel_image_preprocessing"

IMG_SIZE = (128, 128)

BATCH_SIZE = 32

EPOCHS = 1

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

# =========================================================
# CREATE ARTIFACT DIRECTORY
# =========================================================

os.makedirs(
    "artifacts",
    exist_ok=True
)

# =========================================================
# LOAD DATASET
# =========================================================

print("\n[INFO] Loading dataset...")

train_dataset = image_dataset_from_directory(
    f"{DATASET_DIR}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataset = image_dataset_from_directory(
    f"{DATASET_DIR}/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = image_dataset_from_directory(
    f"{DATASET_DIR}/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================================================
# NORMALIZATION
# =========================================================

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

val_dataset = val_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

test_dataset = test_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# =========================================================
# DATASET OPTIMIZATION
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

val_dataset = val_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

# =========================================================
# BUILD CNN MODEL
# =========================================================

print("\n[INFO] Building CNN model...")

model = Sequential([

    tf.keras.Input(
        shape=(128, 128, 3)
    ),

    Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(2, 2),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(2, 2),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        6,
        activation="softmax"
    )
])

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================================================
# SAVE MODEL SUMMARY
# =========================================================

with open(
    "artifacts/model_summary.txt",
    "w"
) as f:

    model.summary(
        print_fn=lambda x: f.write(x + "\n")
    )

# =========================================================
# TRAIN MODEL
# =========================================================

print("\n[INFO] Training model...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# =========================================================
# EVALUATION
# =========================================================

print("\n[INFO] Evaluating model...")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

# =========================================================
# LOG METRICS
# =========================================================

mlflow.log_metric(
    "test_accuracy",
    test_accuracy
)

mlflow.log_metric(
    "test_loss",
    test_loss
)

# =========================================================
# PREDICTION
# =========================================================

y_true = np.concatenate([
    y.numpy()
    for x, y in test_dataset
])

y_pred_probs = model.predict(
    test_dataset
)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(8, 6))

plt.imshow(
    cm,
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.savefig(
    "artifacts/confusion_matrix.png"
)

plt.close()

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
)

with open(
    "artifacts/classification_report.txt",
    "w"
) as f:

    f.write(report)

# =========================================================
# TRAINING HISTORY
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training History")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig(
    "artifacts/training_history.png"
)

plt.close()

# =========================================================
# SAVE MODEL
# =========================================================

model.save(
    "artifacts/cnn_model.keras"
)

# =========================================================
# LOG ARTIFACTS
# =========================================================

mlflow.log_artifact(
    "artifacts/model_summary.txt"
)

mlflow.log_artifact(
    "artifacts/classification_report.txt"
)

mlflow.log_artifact(
    "artifacts/confusion_matrix.png"
)

mlflow.log_artifact(
    "artifacts/training_history.png"
)

mlflow.log_artifact(
    "artifacts/cnn_model.keras"
)

print("\n[INFO] Training completed successfully!")

print(f"\nTest Accuracy: {test_accuracy:.4f}")

print(f"Test Loss: {test_loss:.4f}")