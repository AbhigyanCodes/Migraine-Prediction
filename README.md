# Migraine-Prediction
# Migraine Prediction (Demographics)

A demo ML pipeline to predict migraine vs control using demographic features.

## Project structure
See `src/` for modular code: preprocessing, feature engineering, training, evaluation.

## Quickstart
1. Create virtualenv and install:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows use venv\Scripts\activate
   pip install -r requirements.txt
2. Put Migraine_Control_Demographics.xlsx into data/.
3. Run the pipeline:
   ```bash
   python main.py
   # or
   ./run.sh

5. Run tests:
   ```bash
   pip install -r requirements.txt
   pytest -q
