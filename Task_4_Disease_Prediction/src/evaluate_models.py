import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc
)


# Create results directory
os.makedirs("results", exist_ok=True)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

breast_cancer = fetch_ucirepo(id=17)

X = breast_cancer.data.features.copy()
y = breast_cancer.data.targets["Diagnosis"].map({
    "B": 0,
    "M": 1
})


# --------------------------------------------------
# 2. Same train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Load trained models
# --------------------------------------------------

model_files = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "SVM": "models/svm.joblib",
    "Random Forest": "models/random_forest.joblib",
    "XGBoost": "models/xgboost.joblib"
}


# --------------------------------------------------
# 4. Evaluate each model
# --------------------------------------------------

plt.figure(figsize=(10, 7))

for name, path in model_files.items():

    model = joblib.load(path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification report
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malignant"]
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )

    display.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(f"results/{filename}")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - ROC Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"results/{name.lower().replace(' ', '_')}_roc_curve.png"
    )

    plt.close()


# --------------------------------------------------
# 5. Combined ROC curve
# --------------------------------------------------

plt.figure(figsize=(10, 7))

for name, path in model_files.items():

    model = joblib.load(path)

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.tight_layout()

plt.savefig("results/roc_curves.png")
plt.close()


print("\nEvaluation completed successfully.")
print("All evaluation plots have been saved in the results/ folder.")