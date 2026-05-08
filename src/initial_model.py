import pandas as pd

# Load datasets
train_df = pd.read_csv("data/train.csv", low_memory=False)
test_df = pd.read_csv("data/test.csv", low_memory=False)

# Clean column names
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Strip whitespace from all string cells
train_df = train_df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
test_df = test_df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

# Columns that should be numeric
numeric_cols = [
    "CustomerID",
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
    "Churn"
]

# Convert numeric columns
for col in numeric_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

# Remove rows with missing values
train_df = train_df.dropna()
test_df = test_df.dropna()

print("\nTraining Shape After Cleaning:", train_df.shape)
print("Testing Shape After Cleaning:", test_df.shape)

# Check for missing values created during conversion
print("\nMissing Values:")
print(train_df[numeric_cols].isnull().sum())

# Dataset info
print("\nTraining Data Info:")
print(train_df.info())

# Churn distribution
print("\nChurn Distribution:")
print(train_df["Churn"].value_counts())

# Preview cleaned data
print("\nFirst 5 Rows:")
print(train_df.head())