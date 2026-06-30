"""
visualization.py
-----------------
Generates all visual outputs for the AI Iris Classification project:
    - Accuracy comparison bar chart
    - Confusion matrix heatmap (for the best model)
    - Feature correlation heatmap
    - Feature importance chart (for tree-based / coefficient-based models)

All images are saved into the images/ folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

from utils import IMAGES_DIR, ensure_dirs, log


def plot_accuracy_comparison(results_df: pd.DataFrame):
    """Bar chart comparing accuracy of all trained models."""
    ensure_dirs()
    plt.figure(figsize=(8, 5))
    order = results_df.sort_values("Accuracy", ascending=False)
    bars = plt.bar(order["Model"], order["Accuracy"], color="#4C72B0")
    plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")

    for bar, val in zip(bars, order["Accuracy"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2%}",
                  ha="center", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "accuracy_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f"Saved: {out_path}")


def plot_confusion_matrix(cm, class_names, model_name="Best Model"):
    """Heatmap of confusion matrix for a given model."""
    ensure_dirs()
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f"Saved: {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame, feature_names):
    """Correlation heatmap among numeric features of the dataset."""
    ensure_dirs()
    plt.figure(figsize=(7, 6))
    corr = df[feature_names].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "correlation_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f"Saved: {out_path}")


def plot_feature_importance(model, feature_names, model_name="Best Model"):
    """
    Plots feature importance if the model supports it
    (tree-based models -> feature_importances_,
     linear models -> abs(coef_) averaged across classes).
    Skips gracefully if not applicable (e.g. KNN, SVM with rbf kernel).
    """
    ensure_dirs()
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(model.coef_), axis=0)

    if importances is None:
        log(f"Feature importance not applicable for {model_name}; skipping chart.")
        return False

    order = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    plt.figure(figsize=(7, 5))
    plt.barh(sorted_features[::-1], sorted_importances[::-1], color="#55A868")
    plt.title(f"Feature Importance - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "feature_importance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log(f"Saved: {out_path}")
    return True
