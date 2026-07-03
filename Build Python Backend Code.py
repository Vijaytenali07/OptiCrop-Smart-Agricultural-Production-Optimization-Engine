# ==========================================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Flask Backend (app.py)
# ==========================================================

from flask import Flask, render_template, request
import pandas as pd
import joblib

# ----------------------------------------------------------
# Initialize Flask Application
# ----------------------------------------------------------
app = Flask(__name__)

# ----------------------------------------------------------
# Load Saved Model and Preprocessing Objects
# ----------------------------------------------------------
MODEL_PATH = "Best_OptiCrop_Model.pkl"
SCALER_PATH = "OptiCrop_Scaler.pkl"
ENCODER_PATH = "OptiCrop_LabelEncoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# ----------------------------------------------------------
# Home Page
# ----------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------------------------------------
# About Page
# ----------------------------------------------------------
@app.route("/about")
def about():
    return render_template("about.html")


# ----------------------------------------------------------
# Crop Prediction
# ----------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Read user inputs
        nitrogen = float(request.form["Nitrogen"])
        phosphorus = float(request.form["Phosphorus"])
        potassium = float(request.form["Potassium"])
        temperature = float(request.form["Temperature"])
        humidity = float(request.form["Humidity"])
        soil_ph = float(request.form["pH"])
        rainfall = float(request.form["Rainfall"])

        # Create input DataFrame
        input_data = pd.DataFrame([{
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
            "Temperature": temperature,
            "Humidity": humidity,
            "pH": soil_ph,
            "Rainfall": rainfall
        }])

        # Scale input features
        input_scaled = scaler.transform(input_data)

        # Predict crop
        prediction = model.predict(input_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        # Predict probabilities (if supported)
        top_predictions = None
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

            probability_df["Probability"] = (
                probability_df["Probability"] * 100
            ).round(2)

            top_predictions = probability_df.head(3).to_dict("records")

        return render_template(
            "result.html",
            prediction=predicted_crop,
            top_predictions=top_predictions
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction=f"Error: {str(e)}",
            top_predictions=None
        )


# ----------------------------------------------------------
# Run Application
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )