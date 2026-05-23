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


@app.cell
def _(high_pop):
    high_pop
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
