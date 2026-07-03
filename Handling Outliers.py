# ==========================================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Handling Outliers using IQR Method
# ==========================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("data/OptiCrop_Dataset.csv")   # Update the dataset path if required

# ----------------------------------------------------------
# Identify Numerical Columns
# ----------------------------------------------------------
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("Numerical Columns:")
print(numerical_cols.tolist())

# ==========================================================
# 1. Visualize Outliers Before Treatment
# ==========================================================

for column in numerical_cols:

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[column], color="skyblue")

    plt.title(f"Box Plot Before Outlier Treatment - {column}")
    plt.xlabel(column)

    plt.tight_layout()
    plt.show()

# ==========================================================
# 2. Count Outliers using IQR
# ==========================================================

print("\n========== Outlier Count ==========\n")

outlier_summary = []

for column in numerical_cols:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    outlier_summary.append([column, len(outliers)])

outlier_df = pd.DataFrame(
    outlier_summary,
    columns=["Feature", "Outlier Count"]
)

print(outlier_df)

# ==========================================================
# 3. Handle Outliers using IQR Capping (Winsorization)
# ==========================================================

df_capped = df.copy()

for column in numerical_cols:

    Q1 = df_capped[column].quantile(0.25)
    Q3 = df_capped[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df_capped[column] = np.where(
        df_capped[column] < lower_limit,
        lower_limit,
        df_capped[column]
    )

    df_capped[column] = np.where(
        df_capped[column] > upper_limit,
        upper_limit,
        df_capped[column]
    )

print("\nOutliers have been capped successfully.")

# ==========================================================
# 4. Visualize After Outlier Treatment
# ==========================================================

for column in numerical_cols:

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df_capped[column], color="lightgreen")

    plt.title(f"Box Plot After Outlier Treatment - {column}")
    plt.xlabel(column)

    plt.tight_layout()
    plt.show()

# ==========================================================
# 5. Compare Statistics Before and After
# ==========================================================

print("\n========== Before Treatment ==========\n")
print(df[numerical_cols].describe())

print("\n========== After Treatment ==========\n")
print(df_capped[numerical_cols].describe())

# ==========================================================
# 6. Save Clean Dataset
# ==========================================================

output_file = "data/OptiCrop_Dataset_Cleaned.csv"

df_capped.to_csv(output_file, index=False)

print(f"\nCleaned dataset saved as: {output_file}")

# ==========================================================
# 7. Verify Remaining Outliers
# ==========================================================

print("\n========== Remaining Outlier Count ==========\n")

remaining_outliers = []

for column in numerical_cols:

    Q1 = df_capped[column].quantile(0.25)
    Q3 = df_capped[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    count = df_capped[
        (df_capped[column] < lower_limit) |
        (df_capped[column] > upper_limit)
    ].shape[0]

    remaining_outliers.append([column, count])

remaining_df = pd.DataFrame(
    remaining_outliers,
    columns=["Feature", "Remaining Outliers"]
)

print(remaining_df)

# ==========================================================
# 8. Summary
# ==========================================================

print("\n========== Outlier Handling Completed ==========")
print(f"Original Dataset Shape : {df.shape}")
print(f"Cleaned Dataset Shape  : {df_capped.shape}")
print("Method Used            : IQR-Based Outlier Capping (Winsorization)")