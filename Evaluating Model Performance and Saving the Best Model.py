# ==========================================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Model Evaluation and Best Model Selection
# ==========================================================

# Import Required Libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("OptiCrop_Dataset.csv")

# ----------------------------------------------------------
# Target Variable
# ----------------------------------------------------------
target = "Crop"

# Separate Features and Target
X = df.drop(columns=[target])
y = df[target]

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

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

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# Define Machine Learning Models
# ----------------------------------------------------------
models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000, random_state=42),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "K-Nearest Neighbors":
        KNeighborsClassifier(n_neighbors=5),

    "Support Vector Machine":
        SVC(kernel='rbf', probability=True, random_state=42),

    "Naive Bayes":
        GaussianNB()

}

# ----------------------------------------------------------
# Train and Evaluate Models
# ----------------------------------------------------------
results = []

best_model = None
best_model_name = ""
best_accuracy = 0

print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

for name, model in models.items():

    # Train Model
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Save Best Model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# ----------------------------------------------------------
# Model Comparison Table
# ----------------------------------------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.sort_values(
    by="Accuracy",
    ascending=False
))

# ----------------------------------------------------------
# Evaluate Best Model
# ----------------------------------------------------------
print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Best Model : {best_model_name}")
print(f"Accuracy   : {best_accuracy:.4f}")

y_best = best_model.predict(X_test_scaled)

print("\nClassification Report\n")

print(classification_report(
    y_test,
    y_best,
    target_names=label_encoder.classes_
))

print("Confusion Matrix\n")

print(confusion_matrix(y_test, y_best))

# ----------------------------------------------------------
# Save Best Model
# ----------------------------------------------------------
joblib.dump(best_model, "Best_OptiCrop_Model.pkl")
joblib.dump(scaler, "OptiCrop_Scaler.pkl")
joblib.dump(label_encoder, "OptiCrop_LabelEncoder.pkl")

print("\nBest model saved successfully!")

# ----------------------------------------------------------
# Save Model Comparison Results
# ----------------------------------------------------------
results_df.to_csv(
    "Model_Performance_Results.csv",
    index=False
)

print("Performance report saved successfully!")

# ----------------------------------------------------------
# Final Summary
# ----------------------------------------------------------
print("\n")
print("=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)

print(f"Best Algorithm : {best_model_name}")
print(f"Best Accuracy  : {best_accuracy:.2%}")

print("\nSaved Files")
print("-------------------------")
print("Best_OptiCrop_Model.pkl")
print("OptiCrop_Scaler.pkl")
print("OptiCrop_LabelEncoder.pkl")
print("Model_Performance_Results.csv")