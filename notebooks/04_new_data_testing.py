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

    return jl, os, pd, shap, sys, train_test_split


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
    new_songs = X_test.sample(5, random_state=33)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(best_model, new_songs):
    model = best_model.named_steps['model']
    sample_preprocessed = best_model.named_steps['preprocess'].transform(new_songs)
    return model, sample_preprocessed


@app.cell
def _(model, sample_preprocessed, shap):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_preprocessed)
    return explainer, shap_values


@app.cell
def _(
    best_model,
    explainer,
    new_songs,
    sample_preprocessed,
    shap,
    shap_values,
):
    for i in range(len(new_songs)):
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[i],
            base_values=explainer.expected_value,
            data=sample_preprocessed[i],
            feature_names=best_model.named_steps['preprocess'].get_feature_names_out()
        ))
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
    #### Conclusions:
    - As predicted `pop` music doing well with popularity
    - `Blues` increases value on almost the same amount as `pop`
    - `Regional` or `Mood-depend` music is not popular (e.g. French / Mediative)
    - Extremely `quiet` songs are not popular either
    - `is_popular_genre` most of the time impacts final output the most
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
