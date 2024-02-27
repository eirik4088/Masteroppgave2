import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_dist_hist(
    values: np.array,
    title: str,
    x_label: str,
    color_codes="muted",
    color="b",
    mean=True,
    sd=True,
    median=True,
    mad=True,
    kurtosis=True,
    skew=True,
):
    _, ax = plt.subplots()
    sns.set_color_codes(color_codes)
    t = sns.histplot(
        values,
        kde=True,
        color=color,
        bins=9,
        stat="density",
        kde_kws={"cut": 3},
        alpha=0.4,
        edgecolor=(1, 1, 1, 0.4),
    )
    max_y = t.dataLim.get_points()[-1][-1] * 0.975
    min_x = t.dataLim.get_points()[0][0] * 1.2
    ax.set_ylim([0.0, max_y * 1.01])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")

    if mean:
        ax.text(
            min_x,
            max_y,
            f"Mean: {round(np.mean(values), 1)}",
            fontsize=6,
        )
    if sd:
        ax.text(
            min_x,
            max_y * 0.97,
            f"SD: {round(np.std(values), 1)}",
            fontsize=6,
        )
    if median:
        ax.text(
            min_x,
            max_y * 0.97 * 0.97,
            f"Median: {round(np.median(values), 1)}",
            fontsize=6,
        )
    if mad:
        ax.text(
            min_x,
            max_y * 0.97 * 0.97 * 0.97,
            f"MAD: {round(scipy.stats.median_abs_deviation(values), 1)}",
            fontsize=6,
        )
    if kurtosis:
        ax.text(
            min_x,
            max_y * 0.97 * 0.97 * 0.97 * 0.97,
            f"Kurtosis: {round(scipy.stats.kurtosis(values), 1)}",
            fontsize=6,
        )
    if skew:
        ax.text(
            min_x,
            max_y * 0.97 * 0.97 * 0.97 * 0.97,
            f"Kurtosis: {round(scipy.skew.kurtosis(values), 1)}",
            fontsize=6,
        )
