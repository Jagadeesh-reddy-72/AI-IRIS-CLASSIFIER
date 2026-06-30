"""
predict.py
----------
Loads the saved best model + scaler and makes predictions on new Iris flower
measurements.

Usage (as a script):
    python predict.py 5.1 3.5 1.4 0.2
    (sepal_length sepal_width petal_length petal_width)

Usage (as a module):
    from predict import predict_species
    predict_species([5.1, 3.5, 1.4, 0.2])
"""

import sys
import os
import numpy as np
from sklearn.datasets import load_iris

from utils import load_object, BEST_MODEL_PATH, SCALER_PATH, log


CLASS_NAMES = load_iris().target_names.tolist()


def load_artifacts():
    """Load the trained model and scaler from disk."""
    if not (os.path.exists(BEST_MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise FileNotFoundError(
            "Model or scaler not found. Please run 'python train.py' "
            "(or main.py) first to train and save the best model."
        )
    model = load_object(BEST_MODEL_PATH)
    scaler = load_object(SCALER_PATH)
    return model, scaler


def predict_species(measurements):
    """
    Predict the Iris species for a single sample.

    Args:
        measurements: list/array of 4 floats
            [sepal_length, sepal_width, petal_length, petal_width]

    Returns:
        dict with predicted species name, class index, and class probabilities
    """
    if len(measurements) != 4:
        raise ValueError("Exactly 4 measurements are required: "
                          "sepal_length, sepal_width, petal_length, petal_width")

    model, scaler = load_artifacts()

    X = np.array(measurements, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred_idx = int(model.predict(X_scaled)[0])
    pred_species = CLASS_NAMES[pred_idx]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        probabilities = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)}

    return {
        "input": measurements,
        "predicted_class_index": pred_idx,
        "predicted_species": pred_species,
        "probabilities": probabilities,
    }


def predict_batch(samples):
    """Predict species for multiple samples at once. samples: list of 4-value lists."""
    model, scaler = load_artifacts()
    X = np.array(samples, dtype=float)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    return [CLASS_NAMES[int(p)] for p in preds]


def _print_result(result):
    print("\nPrediction Result")
    print("-" * 30)
    print(f"Input measurements : {result['input']}")
    print(f"Predicted species  : {result['predicted_species'].upper()}")
    if result["probabilities"]:
        print("Class probabilities:")
        for species, prob in result["probabilities"].items():
            print(f"  - {species:<12}: {prob:.2%}")
    print("-" * 30)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        try:
            values = [float(v) for v in sys.argv[1:5]]
        except ValueError:
            print("Error: please provide 4 numeric values.")
            sys.exit(1)
        result = predict_species(values)
        _print_result(result)
    else:
        log("No CLI arguments provided. Running a demo prediction with sample data...")
        demo_sample = [5.1, 3.5, 1.4, 0.2]  # classic setosa-like sample
        result = predict_species(demo_sample)
        _print_result(result)
        print("\nTip: Run with your own values, e.g.:")
        print("    python predict.py 6.7 3.0 5.2 2.3")
