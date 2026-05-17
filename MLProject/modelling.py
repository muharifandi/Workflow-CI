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
# MLflow Configuration
# =========================================================

print("[INFO] Using local MLflow tracking")

mlflow.set_experiment(
    "Intel_Image_Classification"
)

mlflow.tensorflow.autolog()

# =========================================================
# Dataset Configuration
# =========================================================

DATASET_DIR = "intel_image_preprocessing"

IMG_SIZE = (128, 128)

BATCH_SIZE = 32

EPOCHS = 5

# =========================================================
# Load Dataset
# =========================================================

print("\n[INFO] Loading dataset...")

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

# =========================================================
# Normalize Dataset
# =========================================================

normalization_layer = tf.keras.layers.Rescaling(
    1./255
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

print("\n[INFO] Building CNN model...")

model = Sequential([

    Input(shape=(128, 128, 3)),

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
# Compile Model
# =========================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================================================
# Create Artifact Directory
# =========================================================

os.makedirs(
    "artifacts",
    exist_ok=True
)

# =========================================================
# Start MLflow Run
# =========================================================

with mlflow.start_run():

    # =====================================================
    # Training
    # =====================================================

    print("\n[INFO] Training model...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )

    # =====================================================
    # Evaluation
    # =====================================================

    print("\n[INFO] Evaluating model...")

    test_loss, test_accuracy = model.evaluate(
        test_dataset
    )

    # =====================================================
    # Log Parameters
    # =====================================================

    mlflow.log_param(
        "img_size",
        IMG_SIZE
    )

    mlflow.log_param(
        "batch_size",
        BATCH_SIZE
    )

    mlflow.log_param(
        "epochs",
        EPOCHS
    )

    # =====================================================
    # Log Metrics
    # =====================================================

    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )

    mlflow.log_metric(
        "test_loss",
        test_loss
    )

    # =====================================================
    # Save Model
    # =====================================================

    model.save(
        "artifacts/cnn_model.keras"
    )

    mlflow.log_artifact(
        "artifacts/cnn_model.keras"
    )

    # =====================================================
    # Save Training History Plot
    # =====================================================

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
        "Train Accuracy",
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

    print("\n[INFO] Generating predictions...")

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

    print("\n[INFO] Creating confusion matrix...")

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

    print("\n[INFO] Saving classification report...")

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
    # Save Model Summary
    # =====================================================

    print("\n[INFO] Saving model summary...")

    with open(
        "artifacts/model_summary.txt",
        "w"
    ) as f:

        model.summary(
            print_fn=lambda x:
            f.write(x + "\n")
        )

    mlflow.log_artifact(
        "artifacts/model_summary.txt"
    )

    # =====================================================
    # Final Output
    # =====================================================

    print("\n======================================")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("======================================")

    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")

    print("\nArtifacts saved in:")
    print("artifacts/")

    print("\nGenerated files:")
    print("- cnn_model.keras")
    print("- training_history.png")
    print("- confusion_matrix.png")
    print("- classification_report.txt")
    print("- model_summary.txt")