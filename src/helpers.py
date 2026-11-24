from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def get_root():
    """Return the absolute Path to the project root directory."""
    return ROOT

def load_classifier():
    """Load the trained best classifier pipeline."""
    return joblib.load(ROOT / "models" / "best_classifier.joblib")

def load_regressor():
    """Load the trained best regression model pipeline."""
    reg_path = ROOT / "models" / "best_regressor.joblib"
    if reg_path.exists():
        return joblib.load(reg_path)
    return None

def load_metadata():
    """Load model metadata and feature/option configurations."""
    with open(ROOT / "models" / "metadata.json", encoding="utf-8") as f:
        return json.load(f)

def read_table(name):
    """Read a CSV file from outputs/tables/ by filename."""
    table_path = ROOT / "outputs" / "tables" / name
    if table_path.exists():
        return pd.read_csv(table_path)
    return pd.DataFrame()

def read_cleaned_accidents():
    """Read the cleaned analysis dataset."""
    data_path = ROOT / "data" / "processed" / "cleaned_accidents.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return pd.DataFrame()
