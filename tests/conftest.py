# tests/conftest.py
import sys, os
import pandas as pd
import numpy as np

# Add project root to sys.path so tests can import src package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

@pytest.fixture
def sample_df():
    """Return a small dataframe mirroring the original Excel columns."""
    data = {
        "P#": ["M1", "C1", "M2", "C2", "M3", "C3"],
        "Gender": ["Male", "Female", "F", "M", "female", "male"],
        "Age": [25, 30, 40, 22, 53, 27],
        "Aura?": ["Yes", "No", "Yes", "No", "No", "Yes"],
        "Medication before the recording session?": ["No", "Yes", "No", "No", "Yes", "No"],
        "Other info": ["nausea", "", "photophobia", None, "trigger: sleep", ""]
    }
    df = pd.DataFrame(data)
    return df
