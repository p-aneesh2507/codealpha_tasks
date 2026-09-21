import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/cs-training.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. BASIC CLEANING
# ============================================================

df = df.drop(columns=["Unnamed: 0"])

# Treat age = 0 as missing
df.loc[df["age"] == 0, "age"] = pd.NA

delinquency_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse"
]

# Replace suspicious sentinel values
for column in delinquency_columns:
    df[column] = df[column].replace([96, 98], pd.NA)
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 5. CREATE MODELS
# ============================================================

models = {

    "logistic_regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "decision_tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])
}


# ============================================================
# 6. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# 7. TRAIN, EVALUATE AND SAVE MODELS
# ============================================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # Print results
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Save model
    model_path = f"models/{model_name}.joblib"

    joblib.dump(
        model,
        model_path
    )

    print(f"Model saved to: {model_path}")

    # Store metrics
    results.append({
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results_path = "results/model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print(f"\nResults saved to: {results_path}")

print("\nTraining and model saving complete.")