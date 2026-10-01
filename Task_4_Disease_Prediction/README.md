# Task 4 – Disease Prediction

## 📌 Overview

This project is part of the **CodeAlpha Machine Learning Internship – Task 4**.

The objective of this task is to build and evaluate machine learning models for **disease prediction** using a dataset containing relevant patient/health-related features.

Multiple classification algorithms are trained and evaluated to compare their predictive performance.

---

## 🎯 Objectives

* Load and preprocess the disease prediction dataset.
* Perform exploratory data analysis and data preparation.
* Split the dataset into training and testing sets.
* Train multiple machine learning classification models.
* Evaluate model performance using appropriate classification metrics.
* Generate confusion matrices and ROC curves.
* Analyze feature importance for tree-based models.
* Compare the performance of different models.

---

## 🤖 Machine Learning Models

The following classification algorithms are used:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest**
4. **XGBoost**

These models provide different approaches to binary/multiclass classification and allow their performance to be compared on the same dataset.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
Feature Preparation
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Confusion Matrix & ROC Curves
   ↓
Feature Importance Analysis
   ↓
Model Comparison
```

---

## 📊 Model Evaluation

The models are evaluated using metrics and visualizations such as:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

The results are stored in the `results` directory.

---

## 📁 Project Structure

```text
Task_4_Disease_Prediction/
│
├── results/
│   ├── logistic_regression_confusion_matrix.png
│   ├── logistic_regression_roc_curve.png
│   ├── model_comparison.csv
│   ├── random_forest_confusion_matrix.png
│   ├── random_forest_feature_importance.csv
│   ├── random_forest_feature_importance.png
│   ├── random_forest_roc_curve.png
│   ├── roc_curves.png
│   ├── svm_confusion_matrix.png
│   ├── svm_roc_curve.png
│   ├── xgboost_confusion_matrix.png
│   ├── xgboost_feature_importance.csv
│   ├── xgboost_feature_importance.png
│   └── xgboost_roc_curve.png
│
├── src/
│   └── ...
│
├── requirements.txt
└── README.md
```

---

## 📂 Results

The `results` directory contains the outputs generated during model training and evaluation.

### Confusion Matrices

Confusion matrices are generated for:

* Logistic Regression
* Random Forest
* SVM
* XGBoost

They provide a visual representation of the model's classification results.

### ROC Curves

ROC curves are generated for the individual models, along with a combined ROC curve for model comparison.

### Feature Importance

Feature importance analysis is provided for:

* Random Forest
* XGBoost

The corresponding `.csv` files contain the numerical feature-importance values, while the `.png` files provide visual representations.

### Model Comparison

`model_comparison.csv` contains the evaluation results used to compare the trained models.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Seaborn**

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/p-aneesh2507/codealpha_tasks.git
```

Navigate to the Task 4 directory:

```bash
cd codealpha_tasks/Task_4_Disease_Prediction
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Navigate to the source directory if required:

```bash
cd src
```

Run the appropriate Python script:

```bash
python <script_name>.py
```

The trained models will generate evaluation metrics and visualization files in the `results` directory.

---

## 📈 Output

The project produces:

* Classification performance metrics
* Confusion matrices
* Individual ROC curves
* Combined ROC curve comparison
* Random Forest feature importance
* XGBoost feature importance
* Model comparison results

---

## ⚠️ Disclaimer

This project is developed for **educational and internship purposes** as part of the CodeAlpha Machine Learning Internship.

The predictions produced by this machine learning project should **not be considered medical diagnoses or a substitute for professional medical advice**.

---

## 👨‍💻 Author

**Aneesh**

Machine Learning Internship Project
**CodeAlpha – Task 4: Disease Prediction**

---

## 🔗 Repository

[CodeAlpha Tasks Repository](https://github.com/p-aneesh2507/codealpha_tasks)

[Task 4 – Disease Prediction](https://github.com/p-aneesh2507/codealpha_tasks/tree/main/Task_4_Disease_Prediction)
