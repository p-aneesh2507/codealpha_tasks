# Credit Scoring Model

## CodeAlpha Machine Learning Internship — Task 1

A machine learning project for predicting the creditworthiness of borrowers using historical financial and credit-related data.

## Project Objective

The objective of this project is to build machine learning models that predict whether a person is likely to experience serious delinquency within the next two years.

The project uses financial and credit-history features such as:

- Revolving credit utilization
- Age
- Past-due payment history
- Debt ratio
- Monthly income
- Number of open credit lines and loans
- Number of 90+ days late payments
- Number of real estate loans
- Number of 60–89 days late payments
- Number of dependents

## Dataset

The project uses the **Give Me Some Credit** dataset.

The dataset contains 150,000 borrower records.

Target variable:

`SeriousDlqin2yrs`

- `0` — No serious delinquency within two years
- `1` — Serious delinquency within two years

The dataset is highly imbalanced:

- Class 0: 93.32%
- Class 1: 6.68%

Because of this class imbalance, accuracy alone is not sufficient for evaluating the models.

## Data Preprocessing

The following preprocessing steps were applied:

1. Removed the unnecessary `Unnamed: 0` column.
2. Treated `age = 0` as a missing value.
3. Treated the values `96` and `98` in delinquency-related columns as missing/sentinel values.
4. Converted the affected columns to numeric format.
5. Missing values were handled using median imputation.
6. The dataset was divided into training and testing sets.
7. Stratified splitting was used to preserve the target-class distribution.

The final train/test split was:

- Training set: 120,000 records
- Test set: 30,000 records

## Machine Learning Algorithms

Three classification algorithms were implemented:

### 1. Logistic Regression

Used as a baseline classification model.

Preprocessing:

- Median imputation
- Standard scaling
- Balanced class weights

### 2. Decision Tree

A tree-based classification model that learns decision rules from the input features.

Balanced class weights were used because of the class imbalance.

### 3. Random Forest

An ensemble of multiple decision trees.

The model was configured with:

- 100 trees
- Balanced class weights
- Parallel processing

## Model Evaluation

The models were evaluated using:

- Precision
- Recall
- F1-Score
- ROC-AUC

These metrics provide a better understanding of model performance on the imbalanced dataset.

### Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.2584 | 0.6249 | 0.3656 | 0.8205 |
| Decision Tree | 0.2473 | 0.2409 | 0.2441 | 0.5943 |
| Random Forest | 0.4160 | 0.3596 | 0.3858 | 0.8423 |

The results show different trade-offs between precision and recall.

Logistic Regression identified a larger proportion of positive cases, resulting in higher recall.

Random Forest produced higher precision, F1-score, and ROC-AUC in this baseline experiment.

## Confusion Matrices

Confusion matrices were generated for all three models.

Files:

```text
results/logistic_regression_confusion_matrix.png
results/decision_tree_confusion_matrix.png
results/random_forest_confusion_matrix.png