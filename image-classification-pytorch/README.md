# Animal Image Classification with PyTorch

## Project Overview

This project implements a clean, end-to-end **image classification pipeline** using **PyTorch**, designed with a **production-oriented structure**. The codebase emphasizes clear separation between **training**, **evaluation**, **error analysis**, and **inference**, following best practices commonly used in real-world machine learning projects.

The overall workflow:

**EDA → Training → Evaluation → Error Analysis → Inference**

---

## Exploratory Data Analysis (EDA)

EDA is conducted **before model training** to understand the raw dataset characteristics:

* Analyze class distribution using **Pandas**
* Visualize random samples from multiple classes
* Detect potential class imbalance

EDA is intentionally performed **prior to data augmentation** to reflect the true data distribution.

---

## Data Augmentation

Applied **only during training** to improve generalization and reduce overfitting:

* `RandomHorizontalFlip (p=0.5)`
* `RandomRotation (±10°)`
* `ColorJitter (brightness, contrast, saturation)`

Validation and test sets remain unaugmented to ensure fair evaluation.

---

## Model Training

* **Architecture**: ResNet18 (pretrained backbone with custom classification head)
* **Loss Function**: CrossEntropyLoss
* **Optimizer**: Adam
* **Epochs**: 5

The training pipeline is modular and configurable, enabling easy experimentation with hyperparameters.

---

## Training Performance

| Metric              | Value  |
| ------------------- | ------ |
| Train Accuracy      | 94.19% |
| Validation Accuracy | 94.00% |

Training and validation curves show **stable convergence** with no significant overfitting.

---

## Evaluation

Model evaluation includes:

* Validation accuracy monitoring
* Confusion matrix visualization
* Class-wise performance analysis

The confusion matrix demonstrates strong diagonal dominance, indicating effective class separation.

---

## Error Analysis

Misclassified samples from the test set are visualized to:

* Analyze visually ambiguous cases
* Identify class confusion patterns
* Improve model interpretability

This step highlights practical **model debugging and diagnostic skills**.

---

## Inference

External images placed in the `inference_images/` directory can be predicted using the `predict_image` function.

Inference output includes:

* Top-k predicted classes
* Corresponding confidence scores

This component simulates a real-world inference workflow.

---

## Reproducibility

* Config-driven training setup
* Fixed random seeds
* Deterministic evaluation pipeline

These practices ensure consistent and reproducible results.

---

## Tech Stack

* Python
* PyTorch & torchvision
* Pandas
* Matplotlib
* Jupyter Notebook

---

## Dataset

The dataset used for this project is available at:

[https://drive.google.com/drive/folders/1ZA6ujY6zGNLAlPBDGd-JA6pOHSBHOk6a](https://drive.google.com/drive/folders/1ZA6ujY6zGNLAlPBDGd-JA6pOHSBHOk6a)

---

## Key Highlights

* Clean and modular codebase following production standards
* Explicit EDA and error analysis steps
* Clear separation between training, evaluation, and inference
* Suitable as a portfolio project for ML / AI Engineer roles
