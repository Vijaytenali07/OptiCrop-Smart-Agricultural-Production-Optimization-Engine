# ==========================================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Multivariate Analysis
# ==========================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("data/OptiCrop_Dataset.csv")   # Update with your dataset path

# ----------------------------------------------------------
# Identify Numerical and Categorical Columns
# ----------------------------------------------------------
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print("Numerical Features:")
print(numerical_cols.tolist())

print("\nCategorical Features:")
print(categorical_cols.tolist())

# ==========================================================
# 1. Correlation Heatmap
# ==========================================================

plt.figure(figsize=(12, 8))

corr_matrix = df[numerical_cols].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# 2. Pair Plot
# ==========================================================

selected_features = numerical_cols[:6]   # Select first six numerical columns

sns.pairplot(
    df[selected_features],
    diag_kind="kde"
)

plt.show()

# ==========================================================
# 3. Clustered Heatmap
# ==========================================================

sns.clustermap(
    corr_matrix,
    annot=True,
    cmap="viridis",
    figsize=(10,10)
)

plt.show()

# ==========================================================
# 4. PCA (Principal Component Analysis)
# ==========================================================

# Standardize numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numerical_cols])

# Apply PCA
pca = PCA(n_components=2)

principal_components = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(
    principal_components,
    columns=["PC1", "PC2"]
)

# Add categorical label if available
if len(categorical_cols) > 0:
    pca_df["Category"] = df[categorical_cols[0]]

# Plot PCA
plt.figure(figsize=(9,6))

if len(categorical_cols) > 0:

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Category",
        palette="Set2"
    )

else:

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2"
    )

plt.title("PCA - Principal Component Analysis")
plt.show()

# ==========================================================
# 5. Explained Variance
# ==========================================================

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(7,5))

plt.bar(
    ["PC1", "PC2"],
    pca.explained_variance_ratio_
)

plt.title("Explained Variance by Principal Components")
plt.ylabel("Variance Ratio")

plt.show()

# ==========================================================
# 6. 3D Scatter Plot
# ==========================================================

if len(numerical_cols) >= 3:

    fig = plt.figure(figsize=(9,7))

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        df[numerical_cols[0]],
        df[numerical_cols[1]],
        df[numerical_cols[2]]
    )

    ax.set_xlabel(numerical_cols[0])
    ax.set_ylabel(numerical_cols[1])
    ax.set_zlabel(numerical_cols[2])

    plt.title("3D Scatter Plot")

    plt.show()

# ==========================================================
# 7. Feature Correlation with Target
# ==========================================================

target = "Yield"     # Change if your target column has a different name

if target in df.columns:

    corr_target = df[numerical_cols].corr()[target].sort_values(ascending=False)

    plt.figure(figsize=(8,5))

    sns.barplot(
        x=corr_target.values,
        y=corr_target.index,
        palette="Blues_r"
    )

    plt.title(f"Correlation of Features with {target}")

    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Features")

    plt.show()

# ==========================================================
# 8. Group-wise Statistics
# ==========================================================

if len(categorical_cols) > 0:

    category = categorical_cols[0]

    print(f"\nAverage Values Grouped by {category}\n")

    print(df.groupby(category)[numerical_cols].mean())

# ==========================================================
# 9. Covariance Matrix
# ==========================================================

print("\nCovariance Matrix:\n")

print(df[numerical_cols].cov())

# ==========================================================
# 10. Summary
# ==========================================================

print("\nMultivariate Analysis Completed Successfully!")