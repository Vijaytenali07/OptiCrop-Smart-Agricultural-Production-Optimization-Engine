# ==========================================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Splitting Data into Train and Test Sets
# ==========================================================

# Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("OptiCrop_Dataset.csv")   # Update the dataset path if required

# ----------------------------------------------------------
# Display Dataset Information
# ----------------------------------------------------------
print("=" * 60)
print("Dataset Information")
print("=" * 60)

print("Dataset Shape :", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ----------------------------------------------------------
# Define Target Variable
# ----------------------------------------------------------
# For Crop Recommendation
target_column = "Crop"

# Uncomment the following line if predicting crop yield
# target_column = "Yield"

# ----------------------------------------------------------
# Check Target Column
# ----------------------------------------------------------
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

# ----------------------------------------------------------
# Separate Features and Target
# ----------------------------------------------------------
X = df.drop(columns=[target_column])
y = df[target_column]

# ----------------------------------------------------------
# Encode Target Variable (Classification)
# ----------------------------------------------------------
label_encoder = None

if y.dtype == "object":
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    print("\nTarget Classes:")
    print(label_encoder.classes_)

# ----------------------------------------------------------
# Encode Categorical Features
# ----------------------------------------------------------
X = pd.get_dummies(X, drop_first=True)

print("\nFeature Matrix Shape :", X.shape)
print("Target Vector Shape  :", y.shape)

# ----------------------------------------------------------
# Split Dataset (80% Train - 20% Test)
# ----------------------------------------------------------
if len(pd.Series(y).unique()) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

# ----------------------------------------------------------
# Feature Scaling
# ----------------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# Display Dataset Sizes
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("Train-Test Split Summary")
print("=" * 60)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

print(f"\nTraining Features Shape : {X_train_scaled.shape}")
print(f"Testing Features Shape  : {X_test_scaled.shape}")

# ----------------------------------------------------------
# Class Distribution
# ----------------------------------------------------------
print("\nTraining Class Distribution")
print(pd.Series(y_train).value_counts())

print("\nTesting Class Distribution")
print(pd.Series(y_test).value_counts())

# ----------------------------------------------------------
# Save Processed Data (Optional)
# ----------------------------------------------------------
pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("X_test.csv", index=False)

pd.DataFrame(y_train, columns=[target_column]).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test, columns=[target_column]).to_csv("y_test.csv", index=False)

print("\nTrain and Test datasets saved successfully.")

# ----------------------------------------------------------
# Final Summary
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("Data Splitting Completed Successfully")
print("=" * 60)

print("Training Ratio : 80%")
print("Testing Ratio  : 20%")
print(f"Total Features : {X.shape[1]}")
print(f"Target Column  : {target_column}")