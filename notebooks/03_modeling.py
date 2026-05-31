import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import pandas as pd

    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv('../data/processed/spotify_clean.csv')
    return


if __name__ == "__main__":
    app.run()
