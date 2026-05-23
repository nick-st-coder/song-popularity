import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    high_pop = pd.read_csv("../data/raw/spotify/high_popularity_spotify_data.csv")
    low_pop = pd.read_csv("../data/raw/spotify/low_popularity_spotify_data.csv")
    return (high_pop,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By executing command below, we see, that the most popular genres are rock, pop and hip-hop with significant difference from other genres below them.
    """)
    return


@app.cell
def _(high_pop):
    # this line may be underlined as an error, but it's just static-analysis noise (warning).
    high_pop.groupby(by='playlist_genre').size().sort_values(ascending=False).head(5)  # ty:ignore[no-matching-overload]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _():
    target_genres = ['rock', 'pop', 'hip-hop']
    return (target_genres,)


@app.cell
def _(high_pop, target_genres):
    result = high_pop[high_pop['playlist_genre'].isin(target_genres)].groupby(by='playlist_genre').agg({
        'energy':'mean',
        'danceability':'mean',
        'playlist_genre':'count',
        'loudness':'mean',
        'liveness':'mean',
        'valence':'mean',
        'speechiness':'mean'
    })
    return (result,)


@app.cell
def _(result):
    result
    return


if __name__ == "__main__":
    app.run()
