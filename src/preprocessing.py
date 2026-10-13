from pathlib import Path
import pandas as pd

FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
TARGET = "Potability"
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "water_potability.csv"

def load_dataset(path=None):
    """Load and validate the dataset CSV."""
    csv_path = Path(path) if path else DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Upload water_potability.csv in the app.")
    df = pd.read_csv(csv_path)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    return df[required].copy()
