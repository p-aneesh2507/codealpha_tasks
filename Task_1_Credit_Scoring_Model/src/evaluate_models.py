import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/cs-training.csv"

df = pd.read_csv(DATA_PATH)

# Remove ID column
df = df.drop(columns=["Unnamed: 0"])

# Treat age = 0 as missing
df.loc[df["age"] == 0, "age"] = pd.NA

# Delinquency columns
delinquency_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse"
]

# Replace suspicious values
for column in delinquency_columns:
    df[column] = df[column].replace([96, 98], pd.NA)
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": Pipeline([
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

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Random Forest": Pipeline([
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
# 5. TRAIN MODELS AND CREATE CONFUSION MATRICES
# ============================================================

predictions = {}
probabilities = {}
auc_scores = {}

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    predictions[name] = model.predict(X_test)
    probabilities[name] = model.predict_proba(X_test)[:, 1]

    auc_scores[name] = roc_auc_score(
        y_test,
        probabilities[name]
    )


# ============================================================
# 6. CONFUSION MATRICES
# ============================================================

for name in models.keys():

    cm = confusion_matrix(
        y_test,
        predictions[name]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Serious Delinquency", "Serious Delinquency"],
        yticklabels=["No Serious Delinquency", "Serious Delinquency"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        f"results/{filename}",
        dpi=300
    )

    plt.show()

    print(f"\n{name} Confusion Matrix:")
    print(cm)


# ============================================================
# 7. ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for name in models.keys():

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities[name]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc_scores[name]:.4f})"
    )


# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Credit Scoring Models")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/roc_curves.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. PRINT AUC RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ROC-AUC RESULTS")
print("=" * 60)

for name, score in auc_scores.items():
    print(f"{name}: {score:.4f}")

print("\nEvaluation complete.")
print("Graphs saved inside the results folder.")