from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def objective_rf(trial, preprocess, X_train, y_train):
    params = {
        'n_estimators':trial.suggest_int('n_estimators', 200, 800),
        'max_depth':trial.suggest_int('max_depth', 4, 8),
        'min_samples_leaf':trial.suggest_int("min_samples_leaf", 20, 80),
        'min_samples_split':trial.suggest_int("min_samples_split", 30, 120),
        'max_features':trial.suggest_categorical("max_features", ['sqrt', 'log2', None]),
        'random_state':4,
        'verbose':1,
        'n_jobs':-1
    }

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(**params))
    ])

    return cross_val_score(
        pipe,
        X_train, 
        y_train,
        cv=5,
        scoring='f1'
    ).mean()