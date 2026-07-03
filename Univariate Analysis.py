# ============================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Univariate Analysis
# ============================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# --------------------------------------------
# Load Dataset
# --------------------------------------------
df = pd.read_csv("data/OptiCrop_Dataset.csv")   # Update the file path if needed

# --------------------------------------------
# Basic Information
# --------------------------------------------
print("Dataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# --------------------------------------------
# Identify Numerical and Categorical Columns
# --------------------------------------------
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print("\nNumerical Columns:")
print(numerical_cols.tolist())

print("\nCategorical Columns:")
print(categorical_cols.tolist())

# ============================================
# Univariate Analysis for Numerical Features
# ============================================

for column in numerical_cols:

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram
    sns.histplot(df[column], kde=True, bins=30, color='skyblue', ax=axes[0])
    axes[0].set_title(f'Histogram of {column}')

    # Box Plot
    sns.boxplot(x=df[column], color='lightgreen', ax=axes[1])
    axes[1].set_title(f'Box Plot of {column}')

    # Density Plot
    sns.kdeplot(df[column], fill=True, color='orange', ax=axes[2])
    axes[2].set_title(f'Density Plot of {column}')

    plt.tight_layout()
    plt.show()

# ============================================
# Univariate Analysis for Categorical Features
# ============================================

for column in categorical_cols:

    plt.figure(figsize=(8,5))

    sns.countplot(
        data=df,
        x=column,
        palette='viridis',
        order=df[column].value_counts().index
    )

    plt.title(f'Count Plot of {column}')
    plt.xticks(rotation=45)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# ============================================
# Frequency Distribution
# ============================================

for column in categorical_cols:

    print("\n" + "="*50)
    print(f"Frequency Distribution of {column}")
    print("="*50)

    print(df[column].value_counts())

# ============================================
# Skewness and Kurtosis
# ============================================

print("\nSkewness of Numerical Features")
print(df[numerical_cols].skew())

print("\nKurtosis of Numerical Features")
print(df[numerical_cols].kurt())

# ============================================
# Missing Values
# ============================================

print("\nMissing Values")
print(df.isnull().sum())

# ============================================
# Outlier Detection using IQR
# ============================================

print("\nOutlier Summary")

for column in numerical_cols:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(f"{column}: {len(outliers)} outliers")