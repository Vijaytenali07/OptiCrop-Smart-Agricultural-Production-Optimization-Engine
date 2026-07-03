# ==========================================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Logistic Regression Classification
# ==========================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("OptiCrop_Dataset.csv")

# ----------------------------------------------------------
# Display Dataset Information
# ----------------------------------------------------------
print("="*60)
print("Dataset Shape:", df.shape)
print("="*60)

# ----------------------------------------------------------
# Define Target Variable
# ----------------------------------------------------------
target = "Crop"

# Check target column
if target not in df.columns:
    raise ValueError(f"{target} column not found in dataset.")

# ----------------------------------------------------------
# Separate Features and Target
# ----------------------------------------------------------
X = df.drop(columns=[target])
y = df[target]

# ----------------------------------------------------------
# Encode Categorical Features
# ----------------------------------------------------------
X = pd.get_dummies(X, drop_first=True)

# ----------------------------------------------------------
# Encode Target Variable
# ----------------------------------------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nCrop Classes:")
print(label_encoder.classes_)

# ----------------------------------------------------------
# Train-Test Split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------
# Feature Scaling
# ----------------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------------
# Train Logistic Regression Model
# ----------------------------------------------------------
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    multi_class='auto'
)

model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")

# ----------------------------------------------------------
# Make Predictions
# ----------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------
# Model Evaluation
# ----------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n========== Model Performance ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------
print("\nClassification Report\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Crop")
plt.ylabel("Actual Crop")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Prediction Probabilities
# ----------------------------------------------------------
probabilities = model.predict_proba(X_test)

print("\nPrediction Probabilities (First 5 Samples):")
print(np.round(probabilities[:5], 3))

# ----------------------------------------------------------
# Predict New Sample
# ----------------------------------------------------------
sample = X.iloc[[0]]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

predicted_crop = label_encoder.inverse_transform(prediction)

print("\nPrediction for Sample Record:")
print("Recommended Crop:", predicted_crop[0])

# ----------------------------------------------------------
# Model Coefficients
# ----------------------------------------------------------
coef_df = pd.DataFrame(
    model.coef_,
    columns=X.columns,
    index=label_encoder.classes_
)

print("\nModel Coefficients:")
print(coef_df)

# ----------------------------------------------------------
# Save Trained Model
# ----------------------------------------------------------
import joblib

joblib.dump(model, "LogisticRegression_OptiCrop_Model.pkl")
joblib.dump(scaler, "Scaler.pkl")
joblib.dump(label_encoder, "LabelEncoder.pkl")

print("\nModel, Scaler, and Label Encoder saved successfully.")

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
print("\n======================================")
print("Logistic Regression Completed Successfully")
print("======================================")
print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")
print(f"Number of Features : {X.shape[1]}")
print(f"Number of Crop Classes : {len(label_encoder.classes_)}")