# main.py
import os
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_dataset, create_labels
from src.feature_engineering import engineer_features
from src.train_model import get_models, cross_validate_models, fit_and_save
from src.evaluate_model import evaluate_model
from src.utils import save_model

def main():
    data_path = "data/Migraine_Control_Demographics.xlsx"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Put your dataset at: {data_path}")

    # Load + label
    df = load_dataset(data_path)
    df = create_labels(df)

    # Features
    df_feat, features = engineer_features(df)
    X = df_feat[features]
    y = df_feat['label']

    # Small holdout split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print("Train samples:", len(X_train), "Test samples:", len(X_test))

    # Baseline models and CV
    models = get_models()
    cv_results = cross_validate_models(models, X_train, y_train)
    print("Cross-validation (F1 mean, std):")
    for k, v in cv_results.items():
        print(f"  {k}: mean={v[0]:.4f}, std={v[1]:.4f}")

    # Choose logistic regression as final baseline (simple & explainable)
    final_model = models["LogisticRegression"]
    fitted = fit_and_save(final_model, X_train, y_train)  # not saving to disk here

    # Evaluate
    metrics = evaluate_model(fitted, X_test, y_test, plot=True)
    print("Evaluation metrics:", metrics)

    # Save final artifact
    model_path = "models/migraine_demographics_model.joblib"
    save_model(fitted, model_path)

if __name__ == "__main__":
    main()
