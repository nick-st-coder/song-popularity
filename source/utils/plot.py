import seaborn as sns
import matplotlib.pyplot as plt


def show_plot_4x4(scatter: bool, box: bool, x_vars: list, y, title: list):
    """
    Parameters
    ----------
    x_vars : list
        List containing 4 x variables:
        [x1, x2, x3, x4]

    title : list
        List containing 5 titles:
        [x1, x2, x3, x4, y]
    """

    if len(title) != 5:
        raise ValueError(
            "title must contain 5 elements: [x1, x2, x3, x4, y]"
        )

    if len(x_vars) != 4:
        raise ValueError(
            "x_vars must contain 4 elements: [x1, x2, x3, x4]"
        )

    if scatter:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        sns.regplot(x=x_vars[0], y=y, ax=axes[0, 0])
        sns.regplot(x=x_vars[1], y=y, ax=axes[0, 1])
        sns.regplot(x=x_vars[2], y=y, ax=axes[1, 0])
        sns.regplot(x=x_vars[3], y=y, ax=axes[1, 1])

        axes[0, 0].set_title(f"{title[0]} vs {title[4]}")
        axes[0, 1].set_title(f"{title[1]} vs {title[4]}")
        axes[1, 0].set_title(f"{title[2]} vs {title[4]}")
        axes[1, 1].set_title(f"{title[3]} vs {title[4]}")

        plt.tight_layout()
        plt.show()

    if box:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        sns.boxplot(x=x_vars[0], ax=axes[0, 0])
        sns.boxplot(x=x_vars[1], ax=axes[0, 1])
        sns.boxplot(x=x_vars[2], ax=axes[1, 0])
        sns.boxplot(x=x_vars[3], ax=axes[1, 1])

        axes[0, 0].set_title(f"{title[0]}")
        axes[0, 1].set_title(f"{title[1]}")
        axes[1, 0].set_title(f"{title[2]}")
        axes[1, 1].set_title(f"{title[3]}")

        plt.tight_layout()
        plt.show()