# src/train_model.py
from typing import Dict, Tuple
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np

def get_models() -> Dict[str, Pipeline]:
    """Return dictionary of baseline pipelines."""
    pipe_lr = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
    pipe_rf = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
    pipe_xgb = Pipeline([('scaler', StandardScaler()), ('clf', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
    return {"LogisticRegression": pipe_lr, "RandomForest": pipe_rf, "XGBoost": pipe_xgb}

def cross_validate_models(models: Dict[str, Pipeline], X, y, n_splits: int = 5) -> Dict[str, Tuple[float,float]]:
    """Return mean and std of F1 for each model using StratifiedKFold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='f1', n_jobs=-1)
        results[name] = (float(np.mean(scores)), float(np.std(scores)))
    return results

def fit_and_save(model_pipeline, X_train, y_train, out_path: str = None):
    """Fit provided pipeline on training data. Optionally save to out_path."""
    model_pipeline.fit(X_train, y_train)
    if out_path:
        joblib.dump(model_pipeline, out_path)
    return model_pipeline
