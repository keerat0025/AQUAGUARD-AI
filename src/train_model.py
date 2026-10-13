from pathlib import Path
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.preprocessing import FEATURES, TARGET, DATA_PATH, load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "water_quality_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

def train_and_save():
    """Train a Random Forest pipeline and save the model and actual test metrics."""
    df = load_dataset().dropna(subset=[TARGET])
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1_score": float(f1_score(y_test, predicted, zero_division=0)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "feature_count": int(len(FEATURES)), "model_name": "Random Forest Classifier"
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
