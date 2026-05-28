import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import pandas as pd

    high_pop = pd.read_csv('../data/raw/high_popularity_spotify_data.csv')
    low_pop = pd.read_csv('../data/raw/low_popularity_spotify_data.csv')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
