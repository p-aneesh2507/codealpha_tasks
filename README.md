# CodeAlpha Machine Learning Internship Tasks

## 📌 Overview

This repository contains the Machine Learning projects completed as part of the **CodeAlpha Machine Learning Internship**.

The tasks cover different areas of Machine Learning and Artificial Intelligence, including **classification, credit risk prediction, speech emotion recognition, and disease prediction**.

The projects demonstrate the complete ML workflow — from data preprocessing and model training to evaluation, visualization, and deployment of selected models through interactive applications.

---

## 📂 Repository Structure

```text
codealpha_tasks/
│
├── Task_1_Credit_Scoring_Model/
│   ├── data/
│   ├── models/
│   ├── results/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
│
├── task2_Emotion-Recognition-from-Speech/
│   ├── dataset/
│   ├── model/
│   │   └── emotion_model.keras
│   ├── src/
│   │   ├── train.py
│   │   └── predict.py
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── Task_4_Disease_Prediction/
│   ├── results/
│   │   ├── confusion matrices
│   │   ├── ROC curves
│   │   ├── feature importance
│   │   └── model comparison
│   ├── src/
│   ├── requirements.txt
│   └── README.md
│
└── README.md
```

---

# 🚀 Tasks Completed

## 🟦 Task 1 – Credit Scoring Model

### Objective

Develop a Machine Learning model to predict the likelihood of a borrower experiencing serious financial delinquency.

### Dataset

The project uses the **Give Me Some Credit** dataset.

The target variable is:

```text
SeriousDlqin2yrs
```

### Models Implemented

* Logistic Regression
* Decision Tree
* Random Forest

### ML Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Preparation
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Performance Comparison
```

### Evaluation Metrics

* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

The project also includes trained model files, evaluation plots, and feature-importance analysis.

---

# 🟩 Task 2 / Emotion Recognition from Speech

## Objective

Build a Machine Learning/Deep Learning system capable of recognizing human emotions from speech audio.

The system processes audio features and predicts the corresponding emotion.

### Workflow

```text
Audio Input
    ↓
Audio Preprocessing
    ↓
Feature Extraction
    ↓
Model Prediction
    ↓
Emotion Classification
```

### Technologies

* Python
* TensorFlow / Keras
* Librosa
* NumPy
* Pandas
* Streamlit

### Model

The trained emotion recognition model is stored as:

```text
model/emotion_model.keras
```

### Interactive Application

A Streamlit application is included for testing the trained model with audio input.

Run the application using:

```bash
python -m streamlit run app.py
```

The application provides an interactive interface for uploading/processing speech and obtaining the predicted emotion.

---

# 🟥 Task 4 – Disease Prediction

## Objective

Develop Machine Learning classification models for predicting disease outcomes from medical dataset features.

### Models Implemented

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost

### Workflow

```text
Medical Dataset
      ↓
Data Preprocessing
      ↓
Feature Preparation
      ↓
Train/Test Split
      ↓
Multiple ML Models
      ↓
Model Evaluation
      ↓
Performance Comparison
```

### Evaluation

The project evaluates the models using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

### Results

The `results/` directory contains:

```text
logistic_regression_confusion_matrix.png
logistic_regression_roc_curve.png

random_forest_confusion_matrix.png
random_forest_feature_importance.png
random_forest_feature_importance.csv
random_forest_roc_curve.png

svm_confusion_matrix.png
svm_roc_curve.png

xgboost_confusion_matrix.png
xgboost_feature_importance.png
xgboost_feature_importance.csv
xgboost_roc_curve.png

roc_curves.png
model_comparison.csv
```

These files provide visual and numerical comparisons of the implemented models.

---

# 🛠️ Technologies Used

The projects in this repository use a combination of:

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **TensorFlow / Keras**
* **Librosa**
* **XGBoost**
* **Matplotlib**
* **Seaborn**
* **Streamlit**
* **Jupyter Notebook**
* **Git & GitHub**

---

# 📊 Machine Learning Concepts Covered

Through these tasks, the following concepts were implemented:

* Data preprocessing
* Data cleaning
* Exploratory Data Analysis
* Feature engineering
* Feature extraction
* Train/test splitting
* Supervised learning
* Binary classification
* Multiclass classification
* Deep Learning
* Model training
* Model comparison
* Hyperparameter-based model experimentation
* Feature importance
* Confusion matrices
* ROC curves
* ROC-AUC evaluation
* Model deployment with Streamlit

---

# ⚙️ General Setup

Clone the repository:

```bash
git clone https://github.com/p-aneesh2507/codealpha_tasks.git
```

Navigate into the repository:

```bash
cd codealpha_tasks
```

Each task contains its own `requirements.txt` file where applicable.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the required dependencies for the specific task:

```bash
pip install -r requirements.txt
```

---

# 📁 Individual Task Documentation

Each completed task contains its own README file with project-specific information, including:

* Project objective
* Dataset information
* Methodology
* Model architecture
* Implementation details
* Evaluation metrics
* Results
* Installation instructions
* Usage instructions

---

# 🔬 Project Workflow

The overall approach followed across the internship tasks is:

```text
                DATA
                  │
                  ▼
          Data Preprocessing
                  │
                  ▼
        Feature Engineering
                  │
```
