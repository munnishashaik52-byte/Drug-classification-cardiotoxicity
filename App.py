"""
CardioScan AI — Drug Classification System
app.py — Flask Backend API

Loads:
  - best_model.pkl  : Trained Random Forest classifier
  - scaler.pkl      : Fitted StandardScaler (or similar)

Exposes:
  POST /predict     : Accepts JSON patient features, returns prediction + probability
"""

import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ──────────────────────────────────────────────
#  App Initialization
# ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Enable CORS so the frontend (file:// or different port) can reach this API


# ──────────────────────────────────────────────
#  Load Model & Scaler
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    print("[✓] Model loaded successfully.")
except FileNotFoundError:
    model = None
    print(f"[✗] Model file not found at: {MODEL_PATH}")

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = joblib.load(f)
    print("[✓] Scaler loaded successfully.")
except FileNotFoundError:
    scaler = None
    print(f"[✗] Scaler file not found at: {SCALER_PATH}")


# ──────────────────────────────────────────────
#  Feature Order  (must match training order)
# ──────────────────────────────────────────────
FEATURE_COLUMNS = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",       # Systolic blood pressure
    "ap_lo",       # Diastolic blood pressure
    "cholesterol",
    "gluc",        # Glucose
    "smoke",       # Smoking (0 / 1)
    "alco",        # Alcohol intake (0 / 1)
    "active",      # Physical activity (0 / 1)
]


# ──────────────────────────────────────────────
#  Serve Static Files & Index
# ──────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:path>", methods=["GET"])
def serve_static(path):
    return send_from_directory(BASE_DIR, path)


# ──────────────────────────────────────────────
#  Health Check Endpoint
# ──────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    })


# ──────────────────────────────────────────────
#  Prediction Endpoint
# ──────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body with patient features.
    Returns:
        {
            "prediction": 0 or 1,
            "probability": float (0.0 – 1.0)
        }
    """
    # --- Guard: model/scaler must be loaded ---
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded. Check server logs."}), 500

    # --- Parse request body ---
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # --- Validate required fields ---
    missing = [col for col in FEATURE_COLUMNS if col not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # --- Build ordered DataFrame row ---
    try:
        row = {col: [float(data[col])] for col in FEATURE_COLUMNS}
        df  = pd.DataFrame(row, columns=FEATURE_COLUMNS)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid field value: {str(e)}"}), 400

    # --- Scale features ---
    try:
        df_scaled = scaler.transform(df)
    except Exception as e:
        return jsonify({"error": f"Scaling failed: {str(e)}"}), 500

    # --- Predict ---
    try:
        prediction  = int(model.predict(df_scaled)[0])
        # predict_proba returns [[prob_class0, prob_class1]]
        prob_array  = model.predict_proba(df_scaled)[0]
        probability = float(prob_array[prediction])  # confidence for the predicted class
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "prediction":  prediction,
        "probability": round(probability, 4),
    })


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

