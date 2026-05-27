import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from source.utils.plot import show_plot_4x4

    return pd, plt, show_plot_4x4, sns


@app.cell
def _(pd):
    high_pop = pd.read_csv("../data/raw/spotify/high_popularity_spotify_data.csv")
    return (high_pop,)


@app.cell
def _(high_pop):
    high_pop.describe()
    return


@app.cell
def _(high_pop):
    high_pop.shape
    return


@app.cell
def _(high_pop):
    high_pop.info()
    return


@app.cell
def _(high_pop):
    high_pop.isna().sum()
    return


@app.cell
def _(high_pop):
    high_pop.duplicated().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(high_pop):
    high_pop.groupby(by='playlist_genre').size().sort_values(ascending=False).head(5) #ty:ignore[no-matching-overload]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see, that the most popular genre is `pop` with significant difference from other genres.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(high_pop):
    high_pop.groupby("playlist_genre")["track_popularity"].mean().sort_values(ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While popular songs are most likely to be `gaming` or `pop`
    """)
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
    top_three = high_pop[high_pop['playlist_genre'].isin(target_genres)].groupby(by='playlist_genre').agg({
        'track_popularity':'mean',
        'energy':'mean',
        'danceability':'mean',
        'playlist_genre':'count',
        'loudness':'mean',
        'liveness':'mean',
        'valence':'mean',
        'tempo':'mean',
        'duration_ms':'mean'
    })

    top_three
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The most popular music must be energetic, danceable, not loud, can be perfomed live, lasts about 2 minutes and it's happy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(high_pop):
    high_pop["track_popularity"].hist(bins=30)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Left-skewed distribution` - most of the songs popularity parameter is between 65-75
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _():
    numeric_features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness',
    'valence', 'speechiness', 'track_popularity', 'instrumentalness']
    return (numeric_features,)


@app.cell
def _(high_pop, numeric_features, plt, sns):
    corr = high_pop[numeric_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')

    plt.title("Correlation in music")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Highly `energetic` tracks tend to be loud and positive.
    - `Instrumental` songs tend to be quiet.
    - `Loud` music usually not instrumental, danceable, can be performed live, positive and energetic.
    - `Danceable` music is usually has a lot of lyrics in it, positive and loud.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(high_pop):
    high_pop.columns.to_list()[:11]
    return


@app.cell
def _(high_pop):
    y = high_pop['track_popularity']

    x_vars = [high_pop['energy'], high_pop['danceability'], 
                            high_pop['loudness'], high_pop['valence']]

    title = ['Energy', 'Danceability', 'Loudness', 'Valence', 'Popularity']
    return title, x_vars, y


@app.cell
def _(show_plot_4x4, title, x_vars, y):
    show_plot_4x4(True, True, x_vars, y, title)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Based  on `scatter plot`:
    - `energy` values are spreaded all across the plot, it has `no relationship` with `popularity` and most of the songs located between 0.5-0.9 -> energy `does not effect popularity` of the song

    - the same with `danceability` except the range if values are starting from 0.2 -> it has `zero impact on popularity`

    - `loudness` on the other side has slightly `positive correlation with popularity`, vast majority of values located between -15 and -3, it also hase some outliers (very quiet soings which are not popular) -> louder tracks are often more popular

    - `valence` just like first two features has no relationship, but data spreaded almost the same all across the plot -> song mood has `no impact`
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
    Based on `boxplot`:
    - `Energy` Box Plot - Sits high at approximately 0.69. Ranges from 0.55 to 0.81. Main Range: Most tracks fall between 0.18 and 1.0. Outliers: A long tail of low-energy tracks extends down to 0.0.

    - `Danceability` Box Plot - Median is on 0.7. Ranges from 0.15 to 0.9.Main Range: Most tracks fall between 0.55 and 0.78. Outliers: some of them are almost 0.1

    - `Loudness` Box Plot - Very tight main range: -8 to -5, range: -13 to -1. Outliers: ranges from -12 all across to -43, while most of them are just after left whisker. The only case to have outliers on the right side -> rare cases of very loud music

    - `Valence` Box Plot - Median is on 0.5 Main range: 0.38 to 0.7. No outliers. Range: almost 0.0 to almost 1.0
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(high_pop):
    vars = [high_pop["acousticness"], high_pop["instrumentalness"], 
                            high_pop["duration_ms"], high_pop["liveness"]]
                        
    name = ['Acoustic', 'Instrumental', 'Duration', 'Live', 'Popularity']
    return name, vars


@app.cell
def _(name, show_plot_4x4, vars, y):
    show_plot_4x4(True, True, vars, y, name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - `very long` songs are tend to be `less popular`

    - most values on instrumentalness are located on 0.0 which explains why `there're so much outliers`

    - `liveness and acousticness does not affect popularity`. Some of the songs which are perfect for playing live are as popular as regular songs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(high_pop, plt, sns):
    sns.regplot(x=high_pop['loudness'], y=high_pop['acousticness'])

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we can see that those features are highly negatively correlated with each other -> `as music become less acoustic, it also starts to be more loud`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(high_pop):
    high_pop.groupby(['playlist_subgenre'])['track_popularity'].mean().sort_values(ascending=False)
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
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Low popularity music
    """)
    return


@app.cell
def _(pd):
    low_pop = pd.read_csv('../data/raw/low_popularity_spotify_data.csv')
    return (low_pop,)


@app.cell
def _(low_pop):
    low_pop
    return


@app.cell
def _(low_pop):
    low_pop.describe()
    return


@app.cell
def _(low_pop):
    low_pop.isna().sum()
    return


@app.cell
def _(low_pop):
    low_pop.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(low_pop):
    low_pop.columns.to_list()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(low_pop):
    low_pop.groupby(by='playlist_genre').size().sort_values().head(10) #ty:ignore[no-matching-overload]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(low_pop):
    low_pop['track_popularity'].hist(bins=30)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Just as expected non of tracks gained over 68 points of popularity (moved to pop_high dataset)

    Vast majority of values are between 45 and 60, overall data is spreaded more left-skewed, even comparing to full dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _():
    num_feat = ['energy', 'tempo', 'danceability', 'loudness', 'liveness',
    'valence', 'speechiness', 'track_popularity', 'instrumentalness']
    return (num_feat,)


@app.cell
def _(low_pop, num_feat, plt, sns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(low_pop[num_feat].corr(), annot=True, cmap='coolwarm')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Energy` has high positive correlation with `loudness, valence and danceability`

    `Danceability` related to the `valence and loudness`

    `Loudness` highly correlated with `valence`

    Every feature has negative correlation with `instrumenalness` but the leaders of all of them are `energy, loudness and valence` -> energy related features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(low_pop):
    y_low = low_pop['track_popularity']

    x_list = [low_pop['energy'], low_pop['danceability'], 
                            low_pop['loudness'], low_pop['valence']]

    titles = ['Energy', 'Danceability', 'Loudness', 'Valence', 'Popularity']
    return titles, x_list, y_low


@app.cell
def _(show_plot_4x4, titles, x_list, y_low):
    show_plot_4x4(True, True, x_list, y_low, titles)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now in this plots data is spreaded even more evenly. There's `no relationship` between those values

    The only big difference comparing to high popular dataset is that in this one loudness has even more outliers (because of much more cases of unpopular quite music)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(low_pop):
    var = [low_pop["acousticness"], low_pop["instrumentalness"], 
                            low_pop["duration_ms"], low_pop["liveness"]]
                        
    names = ['Acoustic', 'Instrumental', 'Duration', 'Live', 'Popularity']
    return names, var


@app.cell
def _(names, show_plot_4x4, var, y_low):
    show_plot_4x4(True, True, var, y_low, names)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first thing that you've may noticed is outliers on `duration_ms and liveness` which is eg `13552 seconds long` (3.8 hours). This are full concert audios and it could distort our predictions with their perfect 1.0 `liveness` and high duration. And just like in high_pop dataset, the longer the music the lower the popularity


    What's interesting is that here `instrumentalness` mostly divided into 2 clusters  0.1 values (almost no instrumental part) and 0.9 (almost full song is instrumental) and both of clusters are spreaded all across y value, but still `no relationship`

    So let's analyse our outliers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _():
    useful_features = ["track_popularity", "speechiness", "danceability", "track_artist",
    "duration_ms", "energy", "playlist_genre", "playlist_subgenre", "track_name","instrumentalness",
    "valence", "loudness","liveness", "acousticness", "playlist_name"]
    return (useful_features,)


@app.cell
def _(low_pop):
    low_pop['duration_ms'].hist(bins=30, log=True)
    return


@app.cell
def _(low_pop, useful_features):
    low_pop[useful_features][(low_pop['duration_ms'] >= 600000)] #over 10 minutes long
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As expected most of those songs are live sesions which can distort out model.

    Those outliers are useless because are ultimate goal is to find best music algorithm not how popular concerts are -> will drop them
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(low_pop):
    low_pop['liveness'].hist(bins=30, log=True)
    return


@app.cell
def _(low_pop, useful_features):
    low_pop[useful_features][(low_pop['liveness'] > 0.8)] 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Those outliers are just normal music with good live perfomance rate, nothing special, which is pretty strange for me since by logic the more liveness song is the more chance it's gonna be live recordings.
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
    #### SUMMARY
    - `genre: pop` is the most popluar one, `subgenre: global`

    - only `loudness` has positive correlation with popularity, sweetspot: -10db to -2db

    - features like: energy, danceability, valence, acousticness, liveness has `no impact on popularity`

    - `duration of song` and `instrumentalness` has negative correlation with popularity. The sweetspot is around `1,5-3 minutes` long song and almost no instrumental parts.

    ---

    - dataset contained some live recordings which were bad outliers

    - i can write the same conclusions as for the popular music for the rest of the features most of the time there are no relationship at all
    """)
    return


if __name__ == "__main__":
    app.run()
