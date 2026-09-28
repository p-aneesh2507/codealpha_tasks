import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo


# Create results directory
os.makedirs("results", exist_ok=True)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

breast_cancer = fetch_ucirepo(id=17)

X = breast_cancer.data.features.copy()


# --------------------------------------------------
# 2. Feature names
# --------------------------------------------------

feature_names = X.columns.tolist()


# --------------------------------------------------
# 3. Random Forest feature importance
# --------------------------------------------------

random_forest = joblib.load(
    "models/random_forest.joblib"
)

rf_importance = random_forest.feature_importances_

rf_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_importance
})

rf_df = rf_df.sort_values(
    by="Importance",
    ascending=False
)


print("\nRandom Forest Feature Importance:")
print(rf_df.to_string(index=False))


rf_df.to_csv(
    "results/random_forest_feature_importance.csv",
    index=False
)


# Plot Random Forest importance
plt.figure(figsize=(10, 8))

plt.barh(
    rf_df["Feature"].head(15)[::-1],
    rf_df["Importance"].head(15)[::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Random Forest Feature Importances")

plt.tight_layout()

plt.savefig(
    "results/random_forest_feature_importance.png"
)

plt.close()


# --------------------------------------------------
# 4. XGBoost feature importance
# --------------------------------------------------

xgboost_model = joblib.load(
    "models/xgboost.joblib"
)

xgb_importance = xgboost_model.feature_importances_

xgb_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": xgb_importance
})

xgb_df = xgb_df.sort_values(
    by="Importance",
    ascending=False
)


print("\nXGBoost Feature Importance:")
print(xgb_df.to_string(index=False))


xgb_df.to_csv(
    "results/xgboost_feature_importance.csv",
    index=False
)


# Plot XGBoost importance
plt.figure(figsize=(10, 8))

plt.barh(
    xgb_df["Feature"].head(15)[::-1],
    xgb_df["Importance"].head(15)[::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 XGBoost Feature Importances")

plt.tight_layout()

plt.savefig(
    "results/xgboost_feature_importance.png"
)

plt.close()


print("\nFeature importance analysis completed.")
print("Results saved in the results/ folder.")