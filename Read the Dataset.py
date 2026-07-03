# ============================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Read the Dataset
# ============================================

# Import pandas
import pandas as pd

# Specify the dataset path
dataset_path = "data/OptiCrop_Dataset.csv"   # Change this to your dataset location

# Read the dataset
df = pd.read_csv(dataset_path)

# Display the first five records
print("First 5 Rows of the Dataset:")
print(df.head())

# Display the last five records
print("\nLast 5 Rows of the Dataset:")
print(df.tail())

# Display dataset dimensions
print("\nDataset Shape (Rows, Columns):")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display dataset information
print("\nDataset Information:")
df.info()

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())