from ucimlrepo import fetch_ucirepo

# Fetch Breast Cancer Wisconsin (Diagnostic) dataset
breast_cancer = fetch_ucirepo(id=17)

# Separate features and target
X = breast_cancer.data.features
y = breast_cancer.data.targets

print("Dataset shape:", X.shape)
print("\nFeature names:")
print(X.columns.tolist())

print("\nTarget column:")
print(y.columns.tolist())

print("\nTarget distribution:")
print(y.iloc[:, 0].value_counts())

print("\nMissing values:")
print(X.isnull().sum().sum())

print("\nFirst 5 rows:")
print(X.head())