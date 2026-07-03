# ==========================================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Checking for Null (Missing) Values
# ==========================================================

# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("data/OptiCrop_Dataset.csv")   # Update with your dataset path

# ----------------------------------------------------------
# Check Dataset Information
# ----------------------------------------------------------
print("Dataset Shape:", df.shape)
print("\nDataset Information:")
df.info()

# ==========================================================
# 1. Check Total Null Values
# ==========================================================

print("\nTotal Missing Values in Dataset:")
print(df.isnull().sum())

# ==========================================================
# 2. Total Number of Missing Values
# ==========================================================

total_missing = df.isnull().sum().sum()

print("\nTotal Number of Missing Values:", total_missing)

# ==========================================================
# 3. Percentage of Missing Values
# ==========================================================

missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Percentage (%)": missing_percentage.round(2)
})

print("\nMissing Value Summary:")
print(missing_df)

# ==========================================================
# 4. Display Only Columns Having Missing Values
# ==========================================================

missing_columns = missing_df[missing_df["Missing Values"] > 0]

print("\nColumns Containing Missing Values:")
print(missing_columns)

# ==========================================================
# 5. Visualize Missing Values using Heatmap
# ==========================================================

plt.figure(figsize=(12,6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis",
    yticklabels=False
)

plt.title("Missing Values Heatmap")

plt.show()

# ==========================================================
# 6. Missing Values Bar Chart
# ==========================================================

plt.figure(figsize=(10,6))

missing_df["Missing Values"].plot(
    kind="bar",
    color="steelblue"
)

plt.title("Missing Values per Feature")
plt.xlabel("Features")
plt.ylabel("Number of Missing Values")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# ==========================================================
# 7. Check Whether Dataset Contains Missing Values
# ==========================================================

if df.isnull().values.any():
    print("\nDataset contains missing values.")
else:
    print("\nDataset does not contain any missing values.")

# ==========================================================
# 8. Missing Values by Data Type
# ==========================================================

print("\nMissing Values in Numerical Columns:")

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

print(df[numerical_cols].isnull().sum())

print("\nMissing Values in Categorical Columns:")

categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print(df[categorical_cols].isnull().sum())

# ==========================================================
# 9. Missing Value Percentage Sorted
# ==========================================================

sorted_missing = missing_df.sort_values(
    by="Percentage (%)",
    ascending=False
)

print("\nMissing Values (Sorted):")
print(sorted_missing)

# ==========================================================
# 10. Summary
# ==========================================================

print("\n========== Missing Value Analysis Completed ==========")

print(f"Total Rows       : {df.shape[0]}")
print(f"Total Columns    : {df.shape[1]}")
print(f"Missing Entries  : {total_missing}")

if total_missing == 0:
    print("Status           : Dataset is clean.")
else:
    print("Status           : Missing values detected. Data preprocessing is required.")