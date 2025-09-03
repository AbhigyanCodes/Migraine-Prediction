# src/data_preprocessing.py
import pandas as pd
import numpy as np
from typing import Union

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from Excel or CSV."""
    path_str = str(path).lower()
    if path_str.endswith(('.xls', '.xlsx')):
        return pd.read_excel(path)
    return pd.read_csv(path)

def create_labels(df: pd.DataFrame, id_col: str = "P#") -> pd.DataFrame:
    """Create binary migraine label from id_col (prefix 'M' -> migraine)."""
    df = df.copy()
    if id_col not in df.columns:
        raise KeyError(f"{id_col} not found in dataframe columns.")
    df['label'] = df[id_col].astype(str).str.strip().str.upper().str.startswith('M').astype(int)
    return df

def yn_to_binary(s: Union[str, float, int, None]) -> Union[int, float]:
    """Convert yes/no-like values to {1,0} or np.nan for unknowns."""
    if pd.isna(s):
        return np.nan
    s2 = str(s).strip().lower()
    if s2 in ('yes', 'y', '1', 'true', 't'):
        return 1
    if s2 in ('no', 'n', '0', 'false', 'f'):
        return 0
    try:
        return int(float(s2))
    except Exception:
        return np.nan
