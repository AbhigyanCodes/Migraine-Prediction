# tests/test_evaluate_model.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate_model import evaluate_model

def test_evaluate_model_basic():
    # small synthetic dataset
    X = pd.DataFrame({
        "Age_num": [20,30,40,50,21,31,41,51],
        "Gender_bin": [0,1,0,1,0,1,0,1],
        "Aura_bin": [1,0,1,0,1,0,1,0],
        "Medication_bin":[0,0,1,1,0,1,0,1],
        "OtherInfo_present":[0,1,0,1,0,1,0,1],
        "OtherInfo_len":[0,5,0,3,1,0,2,6]
    })
    y = pd.Series([1,0,1,0,1,0,1,0])

    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=200))])
    pipe.fit(X, y)
    metrics = evaluate_model(pipe, X, y, plot=False)
    assert 'accuracy' in metrics and 'f1' in metrics
    assert 0.0 <= metrics['accuracy'] <= 1.0
