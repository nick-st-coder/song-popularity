from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold 
from sklearn.compose import ColumnTransformer

from lightgbm import LGBMRegressor
import mlflow
import pandas as pd
from optuna import Trial

from .params import get_lgbm_params, get_rf_params

# Random Forest
"""
Optuna objective function for Random Forest regression.

Performs:
- getting params
- preprocessing
- 5-fold cross validation
- RMSE evaluation

Returns
-------
float
    Mean and std RMSE across folds.
"""
def objective_rf(trial:Trial, 
                preprocess:ColumnTransformer, 
                X_train: pd.DataFrame, 
                y_train: pd.DataFrame):
    
    params = get_rf_params(trial)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    with mlflow.start_run(nested=True, run_name=f"optuna_rf_trial_{trial.number}"):
        mlflow.set_tag("optuna_trial", trial.number)

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(**params, n_jobs=-1))
        ])

        scores = -cross_val_score(
            pipe, 
            X_train,
            y_train,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

        mean_rmse = scores.mean()
        std_rmse = scores.std()

        mlflow.log_params(params)
        mlflow.log_metric("mean_rmse", mean_rmse)
        mlflow.log_metric("std_rmse", std_rmse)
        mlflow.sklearn.log_model(pipe, f"model_trial_{trial.number}")

    return mean_rmse
    
# LightGBM
"""
Optuna objective function for LightGBM regression.

Performs:
- getting params
- preprocessing
- 5-fold cross validation
- RMSE evaluation

Returns
-------
float
    Mean and std RMSE across folds.
"""
def objective_lgbm(trial:Trial, 
                   preprocess:ColumnTransformer, 
                   X_train: pd.DataFrame, 
                   y_train: pd.DataFrame):

    params = get_lgbm_params(trial)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("model", LGBMRegressor(**params, n_jobs=-1))
    ])

    scores = -cross_val_score(
        model, 
        X_train,
        y_train,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    mean_rmse = scores.mean()
    std_rmse = scores.std()
        
    return mean_rmse