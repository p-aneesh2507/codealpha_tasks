import pandas as pd

# Load dataset
df = pd.read_csv("data/cs-training.csv")

print("=== AGE 0 ===")
print((df["age"] == 0).sum())

print("\n=== DELINQUENCY VALUE COUNTS ===")

delinquency_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse"
]

for column in delinquency_columns:
    print(f"\n{column}")
    print(df[column].value_counts().sort_index().tail(15))

print("\n=== EXTREME RATIOS ===")
print(
    "Utilization > 1:",
    (df["RevolvingUtilizationOfUnsecuredLines"] > 1).sum()
)

print(
    "DebtRatio > 10:",
    (df["DebtRatio"] > 10).sum()
)