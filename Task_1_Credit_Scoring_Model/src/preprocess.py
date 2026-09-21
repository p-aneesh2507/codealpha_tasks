import pandas as pd

DATA_PATH = "data/cs-training.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# Remove unnecessary ID column
df = df.drop(columns=["Unnamed: 0"])

# Treat age = 0 as missing
df.loc[df["age"] == 0, "age"] = pd.NA

# Delinquency columns
delinquency_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse"
]

# Replace suspicious sentinel values with missing values
for column in delinquency_columns:
    df[column] = df[column].replace([96, 98], pd.NA)

# Convert delinquency columns back to numeric
for column in delinquency_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Fill missing values
df["age"] = df["age"].fillna(df["age"].median())

df["MonthlyIncome"] = df["MonthlyIncome"].fillna(
    df["MonthlyIncome"].median()
)

df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
    df["NumberOfDependents"].median()
)

for column in delinquency_columns:
    df[column] = df[column].fillna(df[column].median())

# Display results
print("Cleaned dataset shape:", df.shape)

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)