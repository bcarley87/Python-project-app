"""
Claim Denial Risk Predictor - Flask backend.

Loads the pipeline exported by export_model.py (model.pkl + feature_schema.json)
and serves:
  GET  /                just renders the page
  GET  /api/features     -> the input schema (used by the frontend to build the form)
  POST /api/predict      -> {feature: value, ...}  =>  prediction result

Run with:
    python app.py
Then open http://127.0.0.1:5000
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")

app = Flask(__name__)

# --- Load model + schema once at startup ----------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCHEMA_PATH):
    raise FileNotFoundError(
        "model.pkl and/or feature_schema.json not found.\n"
        "Run export_model.py at the end of your training notebook first, "
        "then copy both files into this folder next to app.py."
    )

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH) as f:
    schema_data = json.load(f)

feature_schema = schema_data["features"]
risk_cutoffs = schema_data["risk_cutoffs"]
model_name = schema_data.get("model_name", "model")

numeric_features = list(feature_schema["numeric"].keys())
categorical_features = list(feature_schema["categorical"].keys())
all_features = numeric_features + categorical_features


def risk_category(prob_denied: float) -> str:
    if prob_denied < risk_cutoffs["low_cutoff"]:
        return "Low Risk"
    elif prob_denied < risk_cutoffs["high_cutoff"]:
        return "Medium Risk"
    return "High Risk"


@app.route("/")
def index():
    return render_template("index.html", model_name=model_name)


@app.route("/api/features")
def get_features():
    """Tells the frontend what fields to render (numeric ranges, dropdown options)."""
    return jsonify(
        {
            "model_name": model_name,
            "numeric": feature_schema["numeric"],
            "categorical": feature_schema["categorical"],
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True) or {}

    missing = [f for f in all_features if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    row = {}

    for f in numeric_features:
        raw = payload.get(f)
        if raw in (None, ""):
            row[f] = None
            continue
        try:
            row[f] = float(raw)
        except (TypeError, ValueError):
            return jsonify({"error": f"'{f}' must be a number, got: {raw!r}"}), 400

    for f in categorical_features:
        raw = payload.get(f)
        row[f] = None if raw in (None, "") else str(raw)

    input_df = pd.DataFrame([row], columns=all_features)

    try:
        prob_denied = float(model.predict_proba(input_df)[:, 1][0])
    except Exception as exc:  # surface pipeline errors clearly instead of a 500 stack trace
        return jsonify({"error": f"Model could not score this input: {exc}"}), 400

    return jsonify(
        {
            "probability_denied": round(prob_denied, 4),
            "probability_paid": round(1 - prob_denied, 4),
            "risk_category": risk_category(prob_denied),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
