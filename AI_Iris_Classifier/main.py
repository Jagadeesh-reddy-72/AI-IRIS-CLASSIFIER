"""
main.py
-------
Single entry point for the AI Iris Flower Classification System.

Running this file will:
    1. Train and compare all 5 algorithms
    2. Select & save the best model
    3. Generate all visualizations
    4. Run a sample prediction to confirm everything works end-to-end

Usage:
    python main.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from train import run_training          # noqa: E402
from predict import predict_species, _print_result  # noqa: E402


def main():
    print("\n##########################################################")
    print("#       AI IRIS FLOWER CLASSIFICATION SYSTEM            #")
    print("##########################################################\n")

    # Run full training + evaluation + visualization pipeline
    best_name, best_model, results_df = run_training()

    # Demonstrate the prediction script using the freshly saved model
    print("\nRunning a sample prediction using the saved best model...")
    sample = [6.7, 3.0, 5.2, 2.3]  # typical virginica-like sample
    result = predict_species(sample)
    _print_result(result)

    print("\nAll outputs have been saved to:")
    print("  - models/   -> best_model.pkl, scaler.pkl")
    print("  - outputs/  -> model_comparison.csv, best_model_info.txt")
    print("  - images/   -> accuracy_comparison.png, confusion_matrix.png,")
    print("                 correlation_heatmap.png, feature_importance.png")
    print("\nProject run completed successfully.\n")


if __name__ == "__main__":
    main()
