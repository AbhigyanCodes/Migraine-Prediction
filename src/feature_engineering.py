# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List

from .data_preprocessing import yn_to_binary

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert raw demographic dataframe into features and label.
    Returns (df_features_with_label, feature_list).
    """
    df_feat = df.copy()

    # Make columns robust to whitespace/casing
    df_feat.columns = [c.strip() for c in df_feat.columns]

    # Column names expected from the Excel file (robust checks)
    aura_col = "Aura?"
    med_col = "Medication before the recording session?"
    gender_col = "Gender"
    age_col = "Age"
    other_col = "Other info"

    # Binary conversions
    if aura_col in df_feat.columns:
        df_feat["Aura_bin"] = df_feat[aura_col].apply(yn_to_binary)
    else:
        df_feat["Aura_bin"] = np.nan

    if med_col in df_feat.columns:
        df_feat["Medication_bin"] = df_feat[med_col].apply(yn_to_binary)
    else:
        df_feat["Medication_bin"] = np.nan

    # Gender
    if gender_col in df_feat.columns:
        df_feat["Gender_clean"] = df_feat[gender_col].astype(str).str.strip().str.lower()
        df_feat["Gender_bin"] = df_feat["Gender_clean"].map({'female':1,'f':1,'male':0,'m':0})
        if df_feat["Gender_bin"].isna().any():
            le = LabelEncoder()
            df_feat["Gender_bin"] = le.fit_transform(df_feat["Gender_clean"].astype(str))
    else:
        df_feat["Gender_bin"] = 0

    # Age numeric
    if age_col in df_feat.columns:
        df_feat["Age_num"] = pd.to_numeric(df_feat[age_col], errors='coerce')
    else:
        df_feat["Age_num"] = np.nan

    # Other info presence / length
    if other_col in df_feat.columns:
        df_feat["OtherInfo_present"] = df_feat[other_col].notna().astype(int)
        df_feat["OtherInfo_len"] = df_feat[other_col].astype(str).fillna("").str.len()
    else:
        df_feat["OtherInfo_present"] = 0
        df_feat["OtherInfo_len"] = 0.0

    features = ["Age_num", "Gender_bin", "Aura_bin", "Medication_bin", "OtherInfo_present", "OtherInfo_len"]

    # Keep label if present
    if 'label' not in df_feat.columns:
        raise KeyError("'label' column is required. Run create_labels() first.")

    return df_feat[features + ["label"]], features
