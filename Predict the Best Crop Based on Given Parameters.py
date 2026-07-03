# ==========================================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Predict the Best Crop Based on Given Parameters
# ==========================================================

# Import Required Libraries
import pandas as pd
import joblib

# ----------------------------------------------------------
# Load Saved Model and Preprocessing Objects
# ----------------------------------------------------------
model = joblib.load("Best_OptiCrop_Model.pkl")
scaler = joblib.load("OptiCrop_Scaler.pkl")
label_encoder = joblib.load("OptiCrop_LabelEncoder.pkl")

# ----------------------------------------------------------
# Enter Input Parameters
# (Modify these values as needed)
# ----------------------------------------------------------

sample_data = {
    "Nitrogen": 90,
    "Phosphorus": 42,
    "Potassium": 43,
    "Temperature": 25.8,
    "Humidity": 80.3,
    "pH": 6.5,
    "Rainfall": 202.9
}

# ----------------------------------------------------------
# Convert Input into DataFrame
# ----------------------------------------------------------
input_df = pd.DataFrame([sample_data])

print("=" * 60)
print("Input Parameters")
print("=" * 60)
print(input_df)

# ----------------------------------------------------------
# Feature Scaling
# ----------------------------------------------------------
input_scaled = scaler.transform(input_df)

# ----------------------------------------------------------
# Predict Crop
# ----------------------------------------------------------
prediction = model.predict(input_scaled)

predicted_crop = label_encoder.inverse_transform(prediction)

print("\n==========================================")
print("Recommended Crop")
print("==========================================")
print("Best Crop :", predicted_crop[0])

# ----------------------------------------------------------
# Prediction Probability (If Supported)
# ----------------------------------------------------------
if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(input_scaled)[0]

    probability_df = pd.DataFrame({
        "Crop": label_encoder.classes_,
        "Probability": probabilities
    })

    probability_df = probability_df.sort_values(
        by="Probability",
        ascending=False
    )

    print("\n==========================================")
    print("Prediction Probabilities")
    print("==========================================")

    print(probability_df)

# ----------------------------------------------------------
# Top 3 Recommended Crops
# ----------------------------------------------------------
if hasattr(model, "predict_proba"):

    top3 = probability_df.head(3)

    print("\n==========================================")
    print("Top 3 Recommended Crops")
    print("==========================================")

    print(top3)

print("\nPrediction Completed Successfully!")