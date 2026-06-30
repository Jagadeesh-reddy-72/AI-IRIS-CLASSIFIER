"""
app.py
------
Flask web interface for the AI Iris Flower Classification System.

Provides:
    - A form-based HTML page (GET /) to enter 4 flower measurements
    - A prediction endpoint (POST /predict) that returns the result
      and re-renders the page with the prediction shown
    - A JSON API endpoint (POST /api/predict) for programmatic use

Run with:
    python app.py
Then open: http://127.0.0.1:5000
"""

import os
import sys

# Make src/ importable regardless of where this script is run from
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(WEB_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from flask import Flask, render_template, request, jsonify  # noqa: E402
from predict import predict_species, load_artifacts          # noqa: E402
from utils import BEST_MODEL_PATH, SCALER_PATH                # noqa: E402

app = Flask(__name__)


def model_is_ready():
    return os.path.exists(BEST_MODEL_PATH) and os.path.exists(SCALER_PATH)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, error=None, model_ready=model_is_ready())


@app.route("/predict", methods=["POST"])
def predict_form():
    """Handle prediction requests submitted from the HTML form."""
    if not model_is_ready():
        error = "No trained model found. Please run 'python main.py' first to train and save a model."
        return render_template("index.html", result=None, error=error, model_ready=False)

    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        result = predict_species([sepal_length, sepal_width, petal_length, petal_width])
        return render_template("index.html", result=result, error=None, model_ready=True,
                                form_values=request.form)
    except (ValueError, KeyError):
        error = "Please enter valid numeric values for all 4 measurements."
        return render_template("index.html", result=None, error=error, model_ready=True,
                                form_values=request.form)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    """JSON API: POST {"sepal_length":..,"sepal_width":..,"petal_length":..,"petal_width":..}"""
    if not model_is_ready():
        return jsonify({"error": "Model not trained yet. Run 'python main.py' first."}), 400

    data = request.get_json(silent=True) or {}
    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    if not all(k in data for k in required):
        return jsonify({"error": f"Missing fields. Required: {required}"}), 400

    try:
        values = [float(data[k]) for k in required]
    except (ValueError, TypeError):
        return jsonify({"error": "All fields must be numeric."}), 400

    result = predict_species(values)
    return jsonify(result)


if __name__ == "__main__":
    if not model_is_ready():
        print("\n[Warning] No trained model found yet.")
        print("Run 'python main.py' first, then restart this app.\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
