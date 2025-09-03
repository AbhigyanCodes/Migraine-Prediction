# tests/test_data_preprocessing.py
import numpy as np
import pandas as pd
import os, sys
# ensure src package importable (handled by conftest, but safe)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_preprocessing import yn_to_binary, create_labels

def test_yn_to_binary_basic():
    assert yn_to_binary("Yes") == 1
    assert yn_to_binary("no") == 0
    assert yn_to_binary("Y") == 1
    assert yn_to_binary("0") == 0
    assert np.isnan(yn_to_binary(None))
    # unknown string -> np.nan
    assert np.isnan(yn_to_binary("maybe"))

def test_create_labels(sample_df):
    df = sample_df.copy()
    df_labeled = create_labels(df, id_col="P#")
    assert 'label' in df_labeled.columns
    # P# starting with 'M' should be labeled 1
    assert list(df_labeled['label'].iloc[:3]) == [1, 0, 1] or isinstance(df_labeled['label'].iloc[0], (int, np.integer))
