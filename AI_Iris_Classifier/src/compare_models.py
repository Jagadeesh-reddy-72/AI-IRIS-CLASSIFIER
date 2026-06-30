"""
compare_models.py
------------------
Trains multiple classification algorithms on the Iris dataset and compares
them using Accuracy, Precision, Recall, and F1 Score (macro-averaged).

Algorithms compared:
    - K-Nearest Neighbors (KNN)
    - Decision Tree
    - Random Forest
    - Logistic Regression
    - Support Vector Machine (SVM)

Outputs:
    - outputs/model_comparison.csv : metrics table for all models
    - Returns trained model objects + metrics dict for downstream use
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from utils import METRICS_CSV, ensure_dirs, log


def get_models(random_state: int = 42):
    """Return a dictionary of model name -> untrained model instance."""
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=random_state),
    }


def train_and_compare(X_train, X_test, y_train, y_test):
    """
    Train every model, evaluate it, and return:
        results_df   - DataFrame of comparison metrics
        trained      - dict of model_name -> fitted model object
        conf_matrices- dict of model_name -> confusion matrix (numpy array)
    """
    ensure_dirs()
    models = get_models()

    rows = []
    trained = {}
    conf_matrices = {}

    for name, model in models.items():
        log(f"Training {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
        })

        trained[name] = model
        conf_matrices[name] = cm

        log(f"{name} -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    results_df = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    results_df.to_csv(METRICS_CSV, index=False)
    log(f"Comparison table saved to: {METRICS_CSV}")

    return results_df, trained, conf_matrices


def select_best_model(results_df: pd.DataFrame, trained: dict):
    """
    Pick the best model based on highest F1 Score (tie-broken by Accuracy).
    Returns: (best_model_name, best_model_object, best_row)
    """
    ranked = results_df.sort_values(by=["F1 Score", "Accuracy"], ascending=False).reset_index(drop=True)
    best_row = ranked.iloc[0]
    best_name = best_row["Model"]
    best_model = trained[best_name]
    return best_name, best_model, best_row
