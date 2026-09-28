import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


# Fetch the Breast Cancer Wisconsin (Diagnostic) dataset
breast_cancer = fetch_ucirepo(id=17)

X = breast_cancer.data.features.copy()
y = breast_cancer.data.targets.copy()


# Convert target labels:
# B = 0 (Benign)
# M = 1 (Malignant)
y = y["Diagnosis"].map({"B": 0, "M": 1})


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nTarget mapping:")
print("0 = Benign")
print("1 = Malignant")