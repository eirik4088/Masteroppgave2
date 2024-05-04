import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_dist_hist(
    values: np.array,
    title= None,
    x_label= None,
    color_codes="muted",
    color="b",
    mean=True,
    sd=True,
    median=True,
    mad=True,
    kurtosis=True,
    skew=True,
    higher=1,
    lefter=1,
):
    style = sns.axes_style()
    style['axes.spines.top'] = False
    style['axes.spines.right'] = False
    style['font'] = "New Times Roman"
    sns.set_theme(font_scale=1.3, style=style)
    _, ax = plt.subplots()
    sns.set_color_codes(color_codes)
    t = sns.histplot(
        values,
        kde=True,
        color=color,
        bins=np.unique(values).size,
        stat="density",
        kde_kws={"cut": 3},
        alpha=0.4,
        edgecolor=(1, 1, 1, 0.4),
    )
    max_y = t.dataLim.get_points()[-1][-1] * 0.975 * higher
    min_x = t.dataLim.get_points()[0][0] * 1.2 * lefter
    ax.set_ylim([0.0, max_y * 1.1])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(None)

    if mean:
        ax.text(
            min_x,
            max_y,
            f"Mean: {round(np.mean(values), 1)}",
            fontsize=14,
            font="Times New Roman",
        )
    if sd:
        ax.text(
            min_x,
            max_y - (max_y * 0.06),
            f"SD: {round(np.std(values), 1)}",
            fontsize=14,
            font="Times New Roman",
        )
    if median:
        ax.text(
            min_x,
            max_y - (max_y * 0.06)*2,
            f"Median: {round(np.median(values), 1)}",
            fontsize=14,
            font="Times New Roman",
        )
    if mad:
        ax.text(
            min_x,
            max_y - (max_y * 0.06)*3,
            f"MAD: {round(scipy.stats.median_abs_deviation(values), 1)}",
            fontsize=14,
            font="Times New Roman",
        )
    if kurtosis:
        ax.text(
            min_x,
            max_y - (max_y * 0.06)*4,
            f"Kurtosis: {round(scipy.stats.kurtosis(values), 1)}",
            fontsize=14,
            font="Times New Roman",
        )
    if skew:
        ax.text(
            min_x,
            max_y - (max_y * 0.06)*5,
            f"Skew: {round(scipy.stats.skew(values), 1)}",
            fontsize=14,
            font="Times New Roman",
        )


def plot_dens_scatter(
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    dens_on_top = False,
):
    x = x_values.copy().flatten()
    y = y_values.copy().flatten()
    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    if dens_on_top:
        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots()
    plt.scatter(x, y, c=z, s=10)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def plot_n_boxplots(data_lists: list[np.ndarray], colors: list):
    sns.boxplot(
        data=data_lists,
        palette=colors,
        showmeans=True,
    )

if __name__ == "__main__":
    pass
