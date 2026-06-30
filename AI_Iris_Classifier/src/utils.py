"""
utils.py
--------
Shared utility functions used across the AI Iris Classification project:
- Path management
- Data loading
- Preprocessing / scaling
- Train/test splitting
- Logging helper
"""

import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Project paths (resolved relative to this file so the project works
# no matter where it's executed from)
# ----------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
IMAGES_DIR = os.path.join(ROOT_DIR, "images")

DATASET_CSV = os.path.join(DATASET_DIR, "iris.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_CSV = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
BEST_MODEL_INFO = os.path.join(OUTPUTS_DIR, "best_model_info.txt")


def ensure_dirs():
    """Create all required project directories if they don't already exist."""
    for d in [DATASET_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR]:
        os.makedirs(d, exist_ok=True)


def log(msg: str):
    """Simple consistent console logger."""
    print(f"[AI-Iris] {msg}")


def load_dataset(save_csv: bool = True) -> pd.DataFrame:
    """
    Load the Iris dataset from sklearn, convert it into a pandas DataFrame,
    and optionally persist it as a CSV file inside dataset/.
    """
    ensure_dirs()
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].apply(lambda i: iris.target_names[i])

    if save_csv:
        df.to_csv(DATASET_CSV, index=False)
        log(f"Dataset saved to: {DATASET_CSV}")

    return df


def preprocess_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split features/target, perform train/test split, and apply feature scaling.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    feature_names = [c for c in df.columns if c not in ("target", "species")]
    X = df[feature_names].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def save_object(obj, path: str):
    """Persist any python object (model/scaler) using joblib."""
    ensure_dirs()
    joblib.dump(obj, path)
    log(f"Saved: {path}")


def load_object(path: str):
    """Load any joblib-persisted object."""
    return joblib.load(path)
