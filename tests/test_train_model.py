# tests/test_train_model.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np

from src.train_model import get_models, cross_validate_models

def test_get_models_and_cv():
    models = get_models()
    assert isinstance(models, dict)
    # create simple synthetic dataset (balanced)
    rng = np.random.RandomState(0)
    X = pd.DataFrame({
        "Age_num": rng.randint(20, 60, size=20),
        "Gender_bin": rng.randint(0,2,size=20),
        "Aura_bin": rng.randint(0,2,size=20),
        "Medication_bin": rng.randint(0,2,size=20),
        "OtherInfo_present": rng.randint(0,2,size=20),
        "OtherInfo_len": rng.randint(0,10,size=20),
    })
    y = pd.Series([0,1]*10)
    # use 2 splits to be safe in small tests
    results = cross_validate_models(models, X, y, n_splits=2)
    assert set(results.keys()) == set(models.keys())
    for name, (mean, std) in results.items():
        assert 0.0 <= mean <= 1.0
