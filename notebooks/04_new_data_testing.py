import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import joblib, os, sys
    import pandas as pd

    from  sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split

    return (
        ColumnTransformer,
        OneHotEncoder,
        Pipeline,
        SimpleImputer,
        pd,
        train_test_split,
    )


@app.cell
def _():
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return


@app.cell
def _(pd):
    df = pd.read_csv('../data/processed/spotify_clean.csv')
    return (df,)


@app.cell
def _():
    num_col = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence', 'speechiness',
    'instrumentalness', 'duration_ms', 'acousticness', 'year_of_release', 'month_of_release']
    cat_col = ['playlist_genre', 'mode', 'key', 'playlist_subgenre', 'is_popular_genre']
    return cat_col, num_col


@app.cell
def _(OneHotEncoder, Pipeline, SimpleImputer):
    num_pipe = Pipeline([
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
    return


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
    return


if __name__ == "__main__":
    app.run()
