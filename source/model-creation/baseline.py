from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import mlflow

def baseline_rf(trial, preprocess, X_train, y_train):
    params = {
        'random_state':4,
        'verbose':1,
        'n_jobs':-1
    }

    with mlflow.start_run(nested=True, run_name="baseline_rf"):
        mlflow.set_tag("baseline_rf", trial)

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