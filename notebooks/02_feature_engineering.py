import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np

    high_pop = pd.read_csv('../data/raw/spotify/high_popularity_spotify_data.csv')
    low_pop = pd.read_csv('../data/raw/spotify/low_popularity_spotify_data.csv')
    return high_pop, low_pop, pd


@app.cell
def _(high_pop, low_pop):
    (high_pop.columns == low_pop.columns).all()
    return


@app.cell
def _(high_pop, low_pop):
    set(high_pop.columns) == set(low_pop.columns)
    return


@app.cell
def _(high_pop, low_pop):
    low_fixed_order = low_pop[high_pop.columns]
    return (low_fixed_order,)


@app.cell
def _(high_pop, low_fixed_order):
    (high_pop.columns == low_fixed_order.columns).all()
    return


@app.cell
def _(high_pop, low_fixed_order, pd):
    df = pd.concat([high_pop, low_fixed_order], ignore_index=True)
    return (df,)


@app.cell
def _(df, pd):
    df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce')
    return


@app.cell
def _(df):
    Q1 = df['duration_ms'].quantile(0.25)
    Q3 = df['duration_ms'].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_no_outliers = df[
        (df['duration_ms'] >= lower) & (df['duration_ms'] <= upper)
    ]
    return (df_no_outliers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(df_no_outliers):
    df_clean = df_no_outliers.drop(columns=['track_artist', 'track_href', 'uri', 'track_album_name', 'playlist_name', 'track_id',
    'track_name', 'track_album_id', 'id', 'type', 'playlist_id', 'analysis_url', 'time_signature', 'track_album_release_date'])
    return (df_clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(df_clean, df_no_outliers):
    df_clean['year_of_release'] = df_no_outliers['track_album_release_date'].dt.year
    df_clean['month_of_release'] = df_no_outliers['track_album_release_date'].dt.month
    return


@app.cell
def _(df_clean):
    popular_genres = ['rock', 'pop', 'hip-hop']
    df_clean['is_popular_genre'] = df_clean['playlist_genre'].isin(popular_genres)
    return


@app.cell
def _(df_clean):
    df_clean
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
    If you want to recreate project -> uncommment line below and run it, this will save clean dataset
    """)
    return


@app.cell
def _():
    # df_clean.to_csv('../data/processed/spotify_clean.csv', index=False
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


if __name__ == "__main__":
    app.run()
