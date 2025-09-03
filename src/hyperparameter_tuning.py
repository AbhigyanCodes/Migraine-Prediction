# src/hyperparameter_tuning.py
from sklearn.model_selection import GridSearchCV

def tune_model(pipeline, param_grid, X, y, cv):
    """
    Run GridSearchCV on provided pipeline and param_grid.
    Returns best_estimator_.
    """
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    search.fit(X, y)
    print("Best params:", search.best_params_)
    print("Best CV F1:", search.best_score_)
    return search.best_estimator_
