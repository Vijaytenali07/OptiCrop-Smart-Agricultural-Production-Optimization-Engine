# ============================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Bivariate Analysis
# ============================================

# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# Load Dataset
# --------------------------------------------
df = pd.read_csv("data/OptiCrop_Dataset.csv")   # Update with your dataset path

# Set Plot Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)

# --------------------------------------------
# Identify Numerical and Categorical Columns
# --------------------------------------------
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# ============================================
# 1. Correlation Matrix
# ============================================

plt.figure(figsize=(12,8))

correlation = df[numerical_cols].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

# ============================================
# 2. Scatter Plots
# ============================================

# Replace 'Yield' with your target variable if different
target = "Yield"

if target in numerical_cols:

    feature_columns = [col for col in numerical_cols if col != target]

    for feature in feature_columns:

        plt.figure(figsize=(7,5))

        sns.scatterplot(
            data=df,
            x=feature,
            y=target,
            color="blue"
        )

        plt.title(f"{feature} vs {target}")
        plt.xlabel(feature)
        plt.ylabel(target)

        plt.tight_layout()
        plt.show()

# ============================================
# 3. Boxplots
# Numerical vs Categorical
# ============================================

if len(categorical_cols) > 0:

    for cat in categorical_cols:

        if target in numerical_cols:

            plt.figure(figsize=(10,5))

            sns.boxplot(
                data=df,
                x=cat,
                y=target,
                palette="Set2"
            )

            plt.title(f"{cat} vs {target}")
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()

# ============================================
# 4. Bar Charts
# Average Target by Category
# ============================================

if len(categorical_cols) > 0:

    for cat in categorical_cols:

        if target in numerical_cols:

            avg = df.groupby(cat)[target].mean().sort_values()

            plt.figure(figsize=(10,5))

            sns.barplot(
                x=avg.index,
                y=avg.values,
                palette="viridis"
            )

            plt.title(f"Average {target} by {cat}")
            plt.xlabel(cat)
            plt.ylabel(f"Average {target}")

            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()

# ============================================
# 5. Pair Plot
# ============================================

selected_columns = numerical_cols[:6]   # Limit to first 6 columns for readability

sns.pairplot(
    df[selected_columns],
    diag_kind="kde"
)

plt.show()

# ============================================
# 6. Regression Plots
# ============================================

if target in numerical_cols:

    feature_columns = [col for col in numerical_cols if col != target]

    for feature in feature_columns:

        plt.figure(figsize=(7,5))

        sns.regplot(
            data=df,
            x=feature,
            y=target,
            scatter_kws={"alpha":0.6},
            line_kws={"color":"red"}
        )

        plt.title(f"Regression Plot: {feature} vs {target}")

        plt.tight_layout()
        plt.show()

# ============================================
# 7. Covariance Matrix
# ============================================

print("\nCovariance Matrix:\n")
print(df[numerical_cols].cov())

# ============================================
# 8. Correlation with Target
# ============================================

if target in numerical_cols:

    print(f"\nCorrelation with {target}:\n")

    corr_target = df[numerical_cols].corr()[target].sort_values(ascending=False)

    print(corr_target)