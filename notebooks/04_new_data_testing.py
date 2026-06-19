import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import joblib, os, sys, shap
    import pandas as pd
    import joblib as jl
    import matplotlib.pyplot as plt

    from  sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split

    return jl, os, pd, sys, train_test_split


@app.cell
def _(os, sys):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv('../data/processed/spotify_clean.csv')
    return (df,)


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
    return (X_test,)


@app.cell
def _(X_test):
    new_songs = X_test.sample(5)
    return (new_songs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(jl):
    best_model = jl.load('../models/best_lgbm.pkl')
    return (best_model,)


@app.cell
def _(new_songs):
    new_songs.agg({
        'energy':'mean',
        'tempo':'mean',
        'danceability':'mean',
        'loudness':'mean',
        'liveness':'mean',
        'valence':'mean',
        'speechiness':'mean',
        'duration_ms':'mean'
    })
    return


@app.cell
def _(best_model, new_songs):
    best_model.predict(new_songs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


if __name__ == "__main__":
    app.run()
