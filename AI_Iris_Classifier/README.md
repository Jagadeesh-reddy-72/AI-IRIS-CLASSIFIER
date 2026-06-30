# AI Iris Flower Classification System

A complete, professional machine learning project that trains, compares, and
deploys multiple classification algorithms on the classic **Iris flower
dataset**, automatically selects the best-performing model, and provides a
ready-to-use prediction script.

---

## Project Overview

This project demonstrates a full ML engineering workflow:

1. **Data Loading & Preprocessing** — loads the Iris dataset from
   `sklearn.datasets`, converts it to a DataFrame, and saves it as CSV.
2. **Feature Scaling** — standardizes features using `StandardScaler`.
3. **Train/Test Split** — stratified 80/20 split.
4. **Model Training & Comparison** — trains 5 algorithms:
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
   - Logistic Regression
   - Support Vector Machine (SVM)
5. **Evaluation** — Accuracy, Precision, Recall, F1 Score (macro-averaged),
   and Confusion Matrix for every model.
6. **Automatic Best Model Selection** — ranked by F1 Score (tie-break:
   Accuracy).
7. **Model Persistence** — best model + scaler saved with `joblib`.
8. **Visualization** — accuracy comparison chart, confusion matrix heatmap,
   feature correlation heatmap, and feature importance chart.
9. **Prediction Script** — CLI/module to classify new flower measurements.

---

## Folder Structure

```
AI_Iris_Classifier/
├── dataset/                 # Saved CSV copy of the Iris dataset
│   └── iris.csv
├── models/                  # Persisted model + scaler (joblib)
│   ├── best_model.pkl
│   └── scaler.pkl
├── outputs/                 # Metrics & reports
│   ├── model_comparison.csv
│   └── best_model_info.txt
├── images/                  # Generated charts
│   ├── accuracy_comparison.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   └── feature_importance.png
├── src/
│   ├── train.py             # Full training pipeline
│   ├── predict.py           # Prediction script (CLI + importable functions)
│   ├── compare_models.py    # Trains & evaluates all 5 algorithms
│   ├── visualization.py     # All chart-generation functions
│   └── utils.py             # Shared helpers (paths, loading, scaling, I/O)
├── web/                      # HTML/Flask web interface
│   ├── app.py                # Flask server (form UI + JSON API)
│   ├── templates/
│   │   └── index.html        # Prediction form + result page
│   └── static/
│       └── style.css         # Styling
├── requirements.txt
├── README.md
└── main.py                  # Single entry point — runs the whole pipeline
```

---

## Installation

```bash
cd AI_Iris_Classifier
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### 1. Run the full pipeline (training + evaluation + visualization)

```bash
python main.py
```

This will:
- Train all 5 models
- Print a comparison table (Accuracy, Precision, Recall, F1)
- Automatically select & save the best model
- Generate all 4 charts in `images/`
- Run a demo prediction to verify everything works

You can also run the training step directly:

```bash
cd src
python train.py
```

### 2. Make predictions with the saved model

```bash
cd src
python predict.py 5.1 3.5 1.4 0.2
```

Arguments are: `sepal_length sepal_width petal_length petal_width` (cm).

Example output:
```
Prediction Result
------------------------------
Input measurements : [5.1, 3.5, 1.4, 0.2]
Predicted species  : SETOSA
Class probabilities:
  - setosa      : 97.50%
  - versicolor  : 2.30%
  - virginica   : 0.20%
------------------------------
```

You can also import it as a module:

```python
from predict import predict_species
result = predict_species([6.7, 3.0, 5.2, 2.3])
print(result["predicted_species"])
```

### 3. Use the HTML web interface

A Flask-based web UI is included so you can get predictions from a browser
instead of the command line.

```bash
cd web
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Enter the four flower
measurements and click **Predict Species** to see the predicted class and
per-class probabilities.

A JSON API is also available for programmatic use:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":6.7,"sepal_width":3.0,"petal_length":5.2,"petal_width":2.3}'
```

> Note: the web app loads `models/best_model.pkl` and `models/scaler.pkl`,
> so make sure you've run `python main.py` (or `train.py`) at least once
> before starting the web server.

---

## Model Comparison Metrics

After running `main.py` or `train.py`, see:
- `outputs/model_comparison.csv` — table of Accuracy/Precision/Recall/F1 for
  all 5 models
- `outputs/best_model_info.txt` — full report including the detailed
  per-class classification report for the winning model

---

## Generated Visualizations

| File | Description |
|---|---|
| `images/accuracy_comparison.png` | Bar chart comparing accuracy across all 5 algorithms |
| `images/confusion_matrix.png` | Confusion matrix heatmap for the best model |
| `images/correlation_heatmap.png` | Correlation heatmap of the 4 Iris features |
| `images/feature_importance.png` | Feature importance (tree-based models) or coefficient magnitude (linear models); skipped automatically for models without this concept (e.g. KNN, RBF-kernel SVM) |

---

## Tech Stack

- Python 3.9+
- scikit-learn
- pandas / numpy
- matplotlib / seaborn
- joblib
- Flask (web interface)

---

## Notes

- The model-selection logic ranks by **F1 Score** (macro-averaged) first,
  breaking ties with **Accuracy**, since F1 better reflects balanced
  performance across all three Iris species.
- All file paths are resolved relative to the project root, so scripts can be
  run either from the project root (`python main.py`) or from inside `src/`
  (`python train.py`).
