from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from lightgbm import LGBMRegressor
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from optuna import Trial

# Random Forest

def objective_rf(trial:Trial, 
                preprocess:ColumnTransformer, 
                X_train: pd.Dataframe, 
                y_train: pd.DataFrame):
    
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

    return mean_rmse, std_rmse
    
# LightGBM

def objective_lgbm(trial:Trial, 
                   preprocess:ColumnTransformer, 
                   X_train: pd.Dataframe, 
                   y_train: pd.DataFrame):
    params = {
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample':trial.suggest_float('subsample', 0.5, 1.0),
        'learning_rate':trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth':trial.suggest_int('max_depth', 5, 12),
        'n_estimators':trial.suggest_int('n_estimators', 100, 1200),
        'num_leaves':trial.suggest_int('num_leaves', 6, 255),
        'reg_alpha':trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples':trial.suggest_int('min_child_samples', 5, 100),

        'objective':'regression',
        'random_state':42,
        'verbosity':-1
    }   

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    model = Pipeline([
        ("preprocess", preprocess),
        ("model", LGBMRegressor(**params, n_jobs=-1))
    ])

    with mlflow.start_run(run_name=f'trial_number{trial.number}'):
        mlflow.set_tag("optuna_params_tuning", trial.number)

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

    mlflow.log_params(params)
    mlflow.log_metric("mean_rmse", mean_rmse)
    mlflow.log_metric("std_rmse", std_rmse)
    mlflow.sklearn.log_model(sk_model=model, artifact_path=f"trial_{trial.number}")

    return mean_rmse, std_rmse