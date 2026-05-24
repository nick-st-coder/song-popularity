import seaborn as sns
import matplotlib.pyplot as plt

def show_plot_4x4(scatter: bool, box: bool, y, x1, x2, x3, x4):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    if(scatter):
        sns.scatterplot(x=x1, y=y, ax=axes[0, 0])
        sns.scatterplot(x=x2, y=y, ax=axes[0, 1])
        sns.scatterplot(x=x3, y=y, ax=axes[1, 0])
        sns.scatterplot(x=x4, y=y, ax=axes[1, 1])

        axes[0, 0].set_title("Energy vs Popularity")
        axes[0, 1].set_title("Danceability vs Popularity")
        axes[1, 0].set_title("Loudness vs Popularity")
        axes[1, 1].set_title("Valence vs Popularity")

    if(box):
        sns.boxplot(x=x1, y=y, ax=axes[0, 0])
        sns.boxplot(x=x2, y=y, ax=axes[0, 1])
        sns.boxplot(x=x3, y=y, ax=axes[1, 0])
        sns.boxplot(x=x4, y=y, ax=axes[1, 1])

        axes[0, 0].set_title("Energy vs Popularity")
        axes[0, 1].set_title("Danceability vs Popularity")
        axes[1, 0].set_title("Loudness vs Popularity")
        axes[1, 1].set_title("Valence vs Popularity")

    plt.tight_layout()
    plt.show()