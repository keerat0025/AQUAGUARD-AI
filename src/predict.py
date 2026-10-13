import joblib
import pandas as pd
from src.preprocessing import FEATURES
from src.train_model import MODEL_PATH

def load_model():
    """Load the trained model or raise a helpful error."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Upload the dataset and train the model first.")
    return joblib.load(MODEL_PATH)

def predict_sample(model, values):
    """Predict the dataset label for one sample using the expected feature order."""
    sample = pd.DataFrame([{feature: float(values[feature]) for feature in FEATURES}], columns=FEATURES)
    return int(model.predict(sample)[0])
