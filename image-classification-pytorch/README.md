🐾 Animal Image Classification with PyTorch
📌 Project Overview

This project implements a clean, end-to-end image classification pipeline using PyTorch, following a production-oriented structure with clear separation between training, evaluation, and inference.

The workflow includes:
EDA → Training → Evaluation → Error Analysis → Inference

🗂 Project Structure
animal-classification-pytorch/
├── dataset/                # Train / Val / Test images
├── inference_images/       # External images for prediction
├── notebooks/
│   └── main.ipynb          # Main pipeline notebook
├── src/
│   ├── config.py           # Hyperparameters & paths
│   ├── train.py            # Training logic
│   ├── evaluate.py         # Evaluation & metrics
│   └── predict.py          # predict_image function
├── saved_models/
│   └── best_model.pt
├── classes.txt
├── requirements.txt
└── README.md

📊 Exploratory Data Analysis (EDA)

EDA is performed before model training to:

Analyze class distribution using Pandas

Visualize random samples from multiple classes

Detect potential data imbalance

EDA is conducted prior to data augmentation to reflect the real dataset characteristics.

🔧 Data Augmentation

Applied during training only:

RandomHorizontalFlip (p=0.5)

RandomRotation (±15°)

ColorJitter (brightness, contrast, saturation)

These augmentations help reduce overfitting and improve model generalization.

🧠 Model Training

Architecture: ResNet18 (custom classifier head)

Loss Function: CrossEntropyLoss

Optimizer: Adam

Epochs: 5

📈 Training Performance
Metric	Value
Train Accuracy	93.88%
Validation Accuracy	93.78%

Training and validation curves indicate stable convergence without significant overfitting.

📉 Evaluation

Evaluation includes:

Validation accuracy monitoring

Confusion Matrix

Class-wise performance analysis

The confusion matrix shows strong diagonal dominance, indicating effective class separation.

🔍 Error Analysis

Misclassified samples from the test set are visualized to:

Understand visually ambiguous cases

Identify class confusion patterns

Improve model interpretability

This step demonstrates practical model debugging skills.

🖼 Inference

External images placed in inference_images/ can be predicted using the predict_image function, returning:

Top-k predicted classes

Corresponding confidence scores

🚀 Key Highlights

Clean modular codebase using src/

Explicit EDA and error analysis

Production-oriented project structure

Reproducible training via config-driven setup

🛠 Tech Stack

Python · PyTorch · torchvision · Pandas · Matplotlib · Jupyter Notebook
