import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
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
# Disable GPU
# =========================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# =========================================================
# MLflow Configuration
# =========================================================

print("[INFO] Using local MLflow tracking")

# experiment managed by mlflow run

# Automatic logging
mlflow.tensorflow.autolog()

# =========================================================
# Dataset Configuration
# =========================================================

DATASET_DIR = "intel_image_preprocessing"

IMG_SIZE = (128, 128)

BATCH_SIZE = 32

EPOCHS = 5

# =========================================================
# Create Artifact Directory
# =========================================================

os.makedirs(
    "artifacts",
    exist_ok=True
)

# =========================================================
# Load Dataset
# =========================================================

print("\n======================================")
print("LOADING DATASET")
print("======================================")

train_dataset = image_dataset_from_directory(
    f"{DATASET_DIR}/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = image_dataset_from_directory(
    f"{DATASET_DIR}/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset = image_dataset_from_directory(
    f"{DATASET_DIR}/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names

print("\nClass Names:")
print(class_names)

# =========================================================
# Normalize Dataset
# =========================================================

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)

train_dataset = train_dataset.map(
    lambda x, y: (
        normalization_layer(x),
        y
    )
)

val_dataset = val_dataset.map(
    lambda x, y: (
        normalization_layer(x),
        y
    )
)

test_dataset = test_dataset.map(
    lambda x, y: (
        normalization_layer(x),
        y
    )
)

# =========================================================
# Optimize Dataset
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# =========================================================
# Build CNN Model
# =========================================================

print("\n======================================")
print("BUILDING CNN MODEL")
print("======================================")

model = Sequential([

    Input(shape=(128, 128, 3)),

    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),

    Flatten(),

    Dense(
        units=128,
        activation="relu"
    ),

    Dropout(
        rate=0.3
    ),

    Dense(
        units=6,
        activation="softmax"
    )
])

# =========================================================
# Save Model Summary
# =========================================================

model.summary()

with open(
    "artifacts/model_summary.txt",
    "w"
) as f:

    model.summary(
        print_fn=lambda x:
        f.write(x + "\n")
    )

# =========================================================
# Compile Model
# =========================================================

print("\n======================================")
print("COMPILING MODEL")
print("======================================")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================================================
# Start MLflow Run
# =========================================================

# MLflow run is managed by mlflow run command

# =====================================================
# Training Model
# =====================================================

print("\n======================================")
print("TRAINING MODEL")
print("======================================")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# =====================================================
# Evaluate Model
# =====================================================

print("\n======================================")
print("EVALUATING MODEL")
print("======================================")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print(f"\nTest Accuracy : {test_accuracy:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

# =====================================================
# Additional Metrics
# =====================================================

mlflow.log_metric(
    "final_test_accuracy",
    test_accuracy
)

mlflow.log_metric(
    "final_test_loss",
    test_loss
)

# =====================================================
# Save Model
# =====================================================

print("\n======================================")
print("SAVING MODEL")
print("======================================")

model.save(
    "artifacts/cnn_model.keras"
)

mlflow.log_artifact(
    "artifacts/cnn_model.keras"
)

# =====================================================
# Training Visualization
# =====================================================

print("\n======================================")
print("CREATING TRAINING VISUALIZATION")
print("======================================")

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"]
)

plt.plot(
    history.history["val_accuracy"]
)

plt.title(
    "Training and Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend([
    "Training Accuracy",
    "Validation Accuracy"
])

plt.savefig(
    "artifacts/training_history.png"
)

plt.close()

mlflow.log_artifact(
    "artifacts/training_history.png"
)

# =====================================================
# Prediction
# =====================================================

print("\n======================================")
print("GENERATING PREDICTIONS")
print("======================================")

y_true = np.concatenate(
    [y for x, y in test_dataset],
    axis=0
)

y_pred_probs = model.predict(
    test_dataset
)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

# =====================================================
# Confusion Matrix
# =====================================================

print("\n======================================")
print("CREATING CONFUSION MATRIX")
print("======================================")

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

disp.plot(ax=ax)

plt.savefig(
    "artifacts/confusion_matrix.png"
)

plt.close()

mlflow.log_artifact(
    "artifacts/confusion_matrix.png"
)

# =====================================================
# Classification Report
# =====================================================

print("\n======================================")
print("CREATING CLASSIFICATION REPORT")
print("======================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

with open(
    "artifacts/classification_report.txt",
    "w"
) as f:

    f.write(report)

mlflow.log_artifact(
    "artifacts/classification_report.txt"
)

# =====================================================
# Save Model Summary Artifact
# =====================================================

mlflow.log_artifact(
    "artifacts/model_summary.txt"
)

# =====================================================
# Final Output
# =====================================================

print("\n======================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("======================================")

print("\nGenerated Artifacts:")

print("- cnn_model.keras")
print("- training_history.png")
print("- confusion_matrix.png")
print("- classification_report.txt")
print("- model_summary.txt")

print("\nArtifacts Location:")
print("artifacts/")