"""
train.py
--------
End-to-end training pipeline for the AI Iris Flower Classification System.

Steps performed:
    1. Load dataset (and persist as CSV)
    2. Preprocess + feature scale + train/test split
    3. Train & compare 5 algorithms
    4. Select the best model automatically (by F1 Score, tie-break Accuracy)
    5. Generate all required visualizations
    6. Save best model + scaler with joblib
    7. Write a human-readable summary report
"""

from sklearn.metrics import classification_report
from sklearn.datasets import load_iris

from utils import (
    load_dataset,
    preprocess_and_split,
    save_object,
    ensure_dirs,
    log,
    BEST_MODEL_PATH,
    SCALER_PATH,
    BEST_MODEL_INFO,
)
from compare_models import train_and_compare, select_best_model
from visualization import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
)


def run_training():
    ensure_dirs()
    log("=" * 60)
    log("AI IRIS FLOWER CLASSIFICATION - TRAINING PIPELINE STARTED")
    log("=" * 60)

    # 1. Load dataset
    df = load_dataset(save_csv=True)
    class_names = load_iris().target_names.tolist()

    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_and_split(df)
    log(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Train & compare all models
    results_df, trained_models, conf_matrices = train_and_compare(X_train, X_test, y_train, y_test)
    print("\nModel Comparison Table:\n")
    print(results_df.to_string(index=False))

    # 4. Select best model
    best_name, best_model, best_row = select_best_model(results_df, trained_models)
    log(f"\nBest Model Selected: {best_name} "
        f"(Accuracy={best_row['Accuracy']}, F1={best_row['F1 Score']})")

    # Detailed classification report for the best model
    y_pred_best = best_model.predict(X_test)
    report = classification_report(y_test, y_pred_best, target_names=class_names)
    print(f"\nDetailed Classification Report - {best_name}\n")
    print(report)

    # 5. Visualizations
    plot_accuracy_comparison(results_df)
    plot_confusion_matrix(conf_matrices[best_name], class_names, model_name=best_name)
    plot_correlation_heatmap(df, feature_names)
    has_importance = plot_feature_importance(best_model, feature_names, model_name=best_name)

    # 6. Save best model + scaler
    save_object(best_model, BEST_MODEL_PATH)
    save_object(scaler, SCALER_PATH)

    # 7. Save summary report
    with open(BEST_MODEL_INFO, "w") as f:
        f.write("AI IRIS FLOWER CLASSIFICATION - BEST MODEL REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Best Model       : {best_name}\n")
        f.write(f"Accuracy         : {best_row['Accuracy']}\n")
        f.write(f"Precision (macro): {best_row['Precision']}\n")
        f.write(f"Recall (macro)   : {best_row['Recall']}\n")
        f.write(f"F1 Score (macro) : {best_row['F1 Score']}\n\n")
        f.write("Full Comparison Table:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nDetailed Classification Report:\n")
        f.write(report)
        f.write(f"\nFeature importance chart generated: {has_importance}\n")

    log(f"Summary report saved to: {BEST_MODEL_INFO}")
    log("=" * 60)
    log("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    log("=" * 60)

    return best_name, best_model, results_df


if __name__ == "__main__":
    run_training()
