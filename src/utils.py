# src/utils.py
import os
import joblib
from typing import Any

def save_model(model: Any, path: str):
    """Save model to path, create dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model: {path}")

def load_model(path: str):
    """Load joblib model."""
    return joblib.load(path)
