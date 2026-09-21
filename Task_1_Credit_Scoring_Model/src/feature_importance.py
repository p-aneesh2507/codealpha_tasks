import pandas as pd
import matplotlib.pyplot as plt
import joblib


# ============================================================
# 1. LOAD TRAINED RANDOM FOREST
# ============================================================

MODEL_PATH = "models/random_forest.joblib"

model = joblib.load(MODEL_PATH)


# ============================================================
# 2. FEATURE NAMES
# ============================================================

feature_names = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]


# ============================================================
# 3. EXTRACT FEATURE IMPORTANCE
# ============================================================

random_forest = model.named_steps["model"]

importance = random_forest.feature_importances_


# ============================================================
# 4. CREATE DATAFRAME
# ============================================================

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(importance_df.to_string(index=False))


# ============================================================
# 6. SAVE CSV
# ============================================================

importance_df.to_csv(
    "results/feature_importance.csv",
    index=False
)


# ============================================================
# 7. CREATE VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "results/random_forest_feature_importance.png",
    dpi=300
)

plt.show()


print("\nFeature importance saved to:")
print("results/feature_importance.csv")

print("\nFeature importance chart saved to:")
print("results/random_forest_feature_importance.png")