# Intel Image Classification - Workflow CI with MLflow

## Project Overview

This project implements an automated Machine Learning workflow using:

- MLflow Project
- GitHub Actions CI/CD
- TensorFlow CNN Model
- Docker
- Docker Hub

The workflow automatically retrains the model whenever a push or pull request is triggered on GitHub.

---

# Project Structure

```text
Workflow-CI/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── MLProject/
│   ├── artifacts/
│   │   ├── cnn_model.keras
│   │   ├── confusion_matrix.png
│   │   ├── training_history.png
│   │   ├── classification_report.txt
│   │   └── model_summary.txt
│   │
│   ├── intel_image_preprocessing/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   ├── modelling.py
│   ├── MLProject
│   ├── conda.yaml
│   ├── requirements.txt
│   └── Dockerfile
│
└── README.md
```

---

# Workflow Architecture

```text
GitHub Push
      │
      ▼
GitHub Actions Triggered
      │
      ▼
Install Dependencies
      │
      ▼
Run modelling.py
      │
      ▼
Train CNN Model
      │
      ▼
Evaluate Model
      │
      ▼
Generate Artifacts
      │
      ▼
Upload Artifacts
      │
      ▼
Build Docker Image
      │
      ▼
Push Docker Image to Docker Hub
```

---

# CNN Architecture

```text
Input Layer
      │
      ▼
Conv2D (32 Filters)
      │
      ▼
MaxPooling2D
      │
      ▼
Conv2D (64 Filters)
      │
      ▼
MaxPooling2D
      │
      ▼
Flatten
      │
      ▼
Dense (128)
      │
      ▼
Dropout (0.3)
      │
      ▼
Output Layer (6 Classes)
```

---

# Dataset

Dataset used:
Intel Image Classification Dataset

Dataset split:

- Train
- Validation
- Test

Classes:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

---

# MLflow Tracking

This project uses MLflow for:

- Experiment tracking
- Parameter logging
- Metric logging
- Artifact logging

Logged metrics:

- test_accuracy
- test_loss

Generated artifacts:

- cnn_model.keras
- training_history.png
- confusion_matrix.png
- classification_report.txt
- model_summary.txt

---

# Docker Integration

Docker is used to containerize the ML application.

Docker image automatically built in GitHub Actions pipeline.

---

# Docker Hub

Docker image is pushed automatically to Docker Hub.

Image name:

```text
username/intel-image-classification:latest
```

---

# GitHub Actions CI/CD

GitHub Actions automatically:

1. Install dependencies
2. Run MLflow project
3. Train model
4. Save artifacts
5. Upload artifacts
6. Build Docker image
7. Push Docker image

---

# Running Project Locally

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run training

```bash
python modelling.py
```

---

# Run MLflow Project

```bash
mlflow run .
```

---

# Build Docker Image

```bash
docker build -t intel-image-classification .
```

---

# Run Docker Container

```bash
docker run intel-image-classification
```

---

# GitHub Secrets

Required GitHub Secrets:

| Secret Name     | Description                  |
| --------------- | ---------------------------- |
| DOCKER_USERNAME | Docker Hub username          |
| DOCKER_PASSWORD | Docker Hub password or token |

---

# Output Artifacts

Artifacts generated after training:

| Artifact                  | Description              |
| ------------------------- | ------------------------ |
| cnn_model.keras           | Trained CNN model        |
| training_history.png      | Accuracy visualization   |
| confusion_matrix.png      | Confusion matrix         |
| classification_report.txt | Evaluation report        |
| model_summary.txt         | CNN architecture summary |

---

# Technologies Used

- Python
- TensorFlow
- MLflow
- GitHub Actions
- Docker
- Docker Hub
- Scikit-learn
- Matplotlib

---

# Conclusion

This project successfully implements:

- MLflow Project workflow
- Automated CI/CD pipeline
- CNN image classification
- Automatic artifact storage
- Docker image automation
- Docker Hub integration

This implementation fulfills all requirements for:

Kriteria 3 - Workflow CI
including:

- Basic
- Skilled
- Advanced
