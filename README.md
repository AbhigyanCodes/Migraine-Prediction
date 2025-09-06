# 🧠 Migraine Prediction (Demographics)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

A modular, reproducible machine-learning pipeline to predict **migraine vs control** using simple demographic features.  
Designed for clarity, easy extension (EEG features, hyperparameter tuning, deployment), and clean GitHub aesthetics.

**Repository:** `https://github.com/AbhigyanCodes/Migraine-Prediction`  
Clone with:
```bash
git clone https://github.com/AbhigyanCodes/Migraine-Prediction.git
cd Migraine-Prediction
```


# 📂 Project structure
```bash
Migraine-Prediction/
├── data/               # input datasets (Excel/CSV) — gitignored; contains .gitkeep
├── models/             # trained model artifacts (gitignored); contains .gitkeep
├── notebooks/          # optional Jupyter/Colab experiments
├── src/                # modular package (preprocessing, features, training, evaluation)
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── hyperparameter_tuning.py
│   ├── utils.py
│   └── __init__.py
├── tests/              # pytest unit tests (self-contained fixtures)
├── main.py             # entry point: runs the end-to-end pipeline
├── run.sh              # one-line runner: ./run.sh
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE             # MIT License
```

# ⚙️ Quickstart
### 1. Set up environment
```bash
git clone https://github.com/AbhigyanCodes/Migraine-Prediction.git
cd Migraine-Prediction
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### 2. Add your dataset

Place your dataset (e.g. Migraine_Control_Demographics.xlsx) into the data/ folder.

### 3. Run the pipeline
```bash
python main.py
# or
./run.sh
```
What this does:

- Loads the dataset from data/

- Runs preprocessing & feature engineering

- Trains baseline models (LogisticRegression, RandomForest, XGBoost)

- Prints cross-validation metrics

- Evaluates the chosen model on a holdout set, plots confusion matrix & ROC

- Saves the trained pipeline to models/migraine_demographics_model.joblib

# 🧩 Features used

From the provided demographics spreadsheet we extract:

- Age → Age_num (numeric)

- Gender → Gender_bin (binary)

- Aura? → Aura_bin (binary)

- Medication before the recording session? → Medication_bin (binary)

- Other info → presence flag (OtherInfo_present) and text length (OtherInfo_len)

# 📈 Notes on results

- The sample/demo dataset is very small (≈36 samples). Expect high variance and limited generalizability.

- In initial experiments Logistic Regression performed best in simple CV (F1 ≈ 0.75–0.80), with Aura and Age being the strongest predictors.

Important: This is a demo/academic project — not suitable for clinical use.

# 🧪 Testing

Unit tests use synthetic data and are self-contained; they do not require your dataset.

Install pytest if needed and run:
```bash
pip install pytest
pytest -q
```

# 🔧 Extend & improve

Ideas to make this project stronger:

- Collect a larger labeled dataset (most important).

- Add EEG preprocessing & feature extraction (band power, PSD, spectrograms).

- Try deep learning (CNN/LSTM) on raw EEG or spectrogram inputs.

- Add hyperparameter optimization (Grid / Random / Bayesian).

- Add model explainability (SHAP/LIME).

- Wrap trained model in a FastAPI/Flask endpoint for inference.

# ✅ Repo hygiene

- data/ and models/ are included in .gitignore to prevent accidental commits of sensitive or large files.

- Keep .gitkeep files in data/ and models/ so folders are visible on GitHub.



# 👤 Author & Contact

Abhigyan Patnaik — GitHub: @AbhigyanCodes

B.Tech (ECE), IIIT Naya Raipur (2022–2026)

# 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
```bash
::contentReference[oaicite:0]{index=0}
```
