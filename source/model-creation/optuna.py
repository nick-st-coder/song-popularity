from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import mlflow

def objective_rf(trial, preprocess, X_train, y_train):
    params = {
        'n_estimators':trial.suggest_int('n_estimators', 200, 800),
        'max_depth':trial.suggest_int('max_depth', 4, 8),
        'min_samples_leaf':trial.suggest_int("min_samples_leaf", 20, 80),
        'min_samples_split':trial.suggest_int("min_samples_split", 30, 120),
        'max_features':trial.suggest_categorical(
            "max_features", ['sqrt', 'log2', None]
        ),

        'random_state':4,
        'verbose':1,
        'n_jobs':-1
    }

    with mlflow.start_run(nested=True, run_name=f"optuna_rf_trial_{trial.number}"):
        mlflow.set_tag("optuna_trial", trial.number)

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(**params))
        ])

        score =  cross_val_score(
            pipe,
            X_train, 
            y_train,
            cv=5,
            scoring='neg_root_mean_squared_error'
        ).mean()

        mlflow.log_params(params)
        mlflow.log_metric("cv_score", score)
        mlflow.sklearn.log_model(pipe, f"model_trial_{trial.number}")

        return score