import seaborn as sns
import matplotlib.pyplot as plt


def show_plot_4x4(scatter: bool, box: bool, x_vars: list, y, title: list):
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

        for ax, x, t in zip(
            axes.flat,
            x_vars,
            title[:4]
        ):
            sns.regplot(
                x=x,
                y=y,
                ax=ax,
                scatter_kws={
                    "alpha": 0.3,
                    "s": 25
                },
                line_kws={
                    "color": "red"
                }
            )

            ax.set_title(f"{t} vs {title[4]}")

        plt.tight_layout()
        plt.show()

    if box:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        for ax, x, t in zip(
            axes.flat,
            x_vars,
            title[:4]
        ):
            sns.boxplot(x=x, ax=ax)
            ax.set_title(t)

        plt.tight_layout()
        plt.show()