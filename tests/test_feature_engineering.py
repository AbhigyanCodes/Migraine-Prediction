# tests/test_feature_engineering.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_preprocessing import create_labels
from src.feature_engineering import engineer_features

def test_engineer_features_shapes(sample_df):
    df = create_labels(sample_df, id_col="P#")
    df_feat, features = engineer_features(df)
    # check returned columns
    for f in features:
        assert f in df_feat.columns
    assert 'label' in df_feat.columns
    # ensure the shape matches number of rows
    assert df_feat.shape[0] == df.shape[0]
