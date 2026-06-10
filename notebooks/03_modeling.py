import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import os
    import sys
    import pandas as pd

    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error
    from lightgbm import LGBMRegressor

    import optuna
    import mlflow

    return (
        ColumnTransformer,
        LGBMRegressor,
        OneHotEncoder,
        Pipeline,
        RandomForestRegressor,
        SimpleImputer,
        cross_val_score,
        mlflow,
        optuna,
        os,
        pd,
        root_mean_squared_error,
        sys,
        train_test_split,
    )


@app.cell
def _(os, sys):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from source.modeling.optuna import objective_lgbm

    return (objective_lgbm,)


@app.cell
def _():
    from sklearn import set_config
    set_config(transform_output="pandas")
    # removes warnings from optuna params tuning
    return


@app.cell
def _(pd):
    df = pd.read_csv('../data/processed/spotify_clean.csv')
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.columns[df.isna().any()].to_list()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _():
    num_col = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence', 'speechiness',
    'instrumentalness', 'duration_ms', 'acousticness', 'year_of_release', 'month_of_release']
    cat_col = ['playlist_genre', 'mode', 'key', 'playlist_subgenre', 'is_popular_genre']
    return cat_col, num_col


@app.cell
def _(OneHotEncoder, Pipeline, SimpleImputer):
    num_pipe = Pipeline([
        #since the only missing values are dates i'm using most frequent strategy
        ("impute", SimpleImputer(strategy='most_frequent'))
    ])

    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return cat_pipe, num_pipe


@app.cell
def _(ColumnTransformer, cat_col, cat_pipe, num_col, num_pipe):
    preprocess = ColumnTransformer([
        ("numeric", num_pipe, num_col),
        ("categorical", cat_pipe, cat_col)
    ])
    return (preprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(df):
    X = df.drop(columns=['track_popularity'])
    y = df['track_popularity']
    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        train_size=0.8, 
        random_state=78, 
        shuffle=True, 
        # dataset is 1.6k of popular rows and 3.2k of unpopular ones going after each other -> shuffle must be maden
        stratify=y
    )
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Baseline - `Random Forest`
    """)
    return


@app.cell
def _(mlflow):
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("song-lgbm")
    return


@app.cell
def _(
    Pipeline,
    RandomForestRegressor,
    X_train,
    cross_val_score,
    mlflow,
    preprocess,
    y_train,
):
    with mlflow.start_run(run_name="baseline_rf"):
        baseline_rf = Pipeline([
            ("preprocess", preprocess),
            ("model", RandomForestRegressor(random_state=4, n_jobs=-1))
        ])  

        score = -cross_val_score(
            baseline_rf,
            X_train,
            y_train,
            cv=5,
            scoring='neg_root_mean_squared_error'
        ).mean()

        mlflow.log_metric("cv_rmse",score)
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("random_state", 4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(
    LGBMRegressor,
    Pipeline,
    X_train,
    cross_val_score,
    mlflow,
    preprocess,
    y_train,
):
    with mlflow.start_run(run_name="baseline_lightgbm"):

        baseline_lightgbm = Pipeline([
            ("preprocess", preprocess),
            ("model", LGBMRegressor(random_state=4, n_jobs=-1, verbose=-1))
        ])  

        score_lgbm = -cross_val_score(
            baseline_lightgbm,
            X_train,
            y_train,
            cv=5,
            scoring='neg_root_mean_squared_error'
        ).mean()

        mlflow.log_metric("cv_rmse",score_lgbm)
        mlflow.log_param("model", "LightGBM")
        mlflow.log_param("random_state", 4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Model params tuning - `LightGBM`
    """)
    return


@app.cell
def _(optuna):
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(), 
    study_name="lgbm_tuning", storage="sqlite:///optuna_lgbm.db", load_if_exists=True)
    return (study,)


@app.cell
def _(X_train, objective_lgbm, preprocess, study, y_train):
    study.optimize(
        lambda trial: objective_lgbm(
            trial,
            preprocess,
            X_train,
            y_train
        ),
        n_trials=50, 
        show_progress_bar=True
    )
    return


@app.cell
def _(study):
    best_params = study.best_params
    return (best_params,)


@app.cell
def _(study):
    print(study.best_value)
    print(study.best_params)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(LGBMRegressor, Pipeline, best_params, preprocess):
    best_model = Pipeline([
        ("preprocess", preprocess),
        ("model", LGBMRegressor(**best_params))
    ])
    return (best_model,)


@app.cell
def _(X_train, best_model, y_train):
    best_model.fit(X_train, y_train)
    return


@app.cell
def _(
    X_test,
    best_model,
    best_params,
    mlflow,
    root_mean_squared_error,
    y_test,
):
    with mlflow.start_run(run_name="best_model"):
        y_pred = best_model.predict(X_test)

        score_rmse = root_mean_squared_error(y_test, y_pred)
        print(f"RMSE: {score_rmse:.4f}")

        mlflow.log_metric("cv_rmse", score_rmse)
        mlflow.log_params(best_params)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_lgbm"
        )

        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/best_lgbm",
            name="LightGBM-spotify"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Saving model
    """)
    return


@app.cell
def _():
    import joblib as jl
    # jl.dump(best_model, "../models/best_lgbm.pkl")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `SUMMARY`

    I compared tuned model with its baseline and rf baseline and this is conclusions that i can make based on that
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `SPEED`
    - LGBM bsl was `125% faster` than RF bsl
    - Tuned model was `66% slower` than RF bsl and `275% slower` than LGBM baseline
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `METRICS`
    - LGBM bsl did `7%` worse than rf baseline (with value `12.33` against `11.47`)
    - Tuned model did `8,7%` better than rf and `16%` than not tuned
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


if __name__ == "__main__":
    app.run()
