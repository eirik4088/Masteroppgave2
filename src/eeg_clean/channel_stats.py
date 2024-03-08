"""_summary_

    _extended_summary_
    """

import mne
import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from eeg_clean.epoch_stats import EpochStats


class ChannelStats:
    """_summary_

    _extended_summary_
    """

    def __init__(self, mne_epochs_object: mne.Epochs, **kwargs) -> None:
        self.epoch_obj = mne_epochs_object
        epoch_stats_object = EpochStats(self.epoch_obj, **kwargs)
        epoch_stats_object.calc_stability(**kwargs)
        self._quasi_baseline = np.mean(
            epoch_stats_object.quasi_stability.get_mean_abs_stab()
        )
        self._peak_baseline = np.mean(epoch_stats_object.peak_stability.get_mean_abs_stab())
        self._pca_baseline, _ = epoch_stats_object.pca_auc(**kwargs)
        self.ch_name_list = np.array(self.epoch_obj.info["ch_names"])
        self.n_channels = len(self.ch_name_list)
        self.quasi_stab_change = np.ndarray((self.n_channels))
        self.peak_stab_change = np.ndarray((self.n_channels))
        self.pca_auc_change = np.ndarray((self.n_channels))
        self._leave_one_out(**kwargs)

    def _leave_one_out(self, **kwargs) -> None:
        for c in range(self.n_channels):
            instance = self._drop_channel(self.epoch_obj, self.ch_name_list[c])
            stab_instance = EpochStats(instance, **kwargs)
            stab_instance.calc_stability(**kwargs)
            quasi_stab = np.mean(stab_instance.quasi_stability.get_mean_abs_stab())
            peak_stab = np.mean(stab_instance.peak_stability.get_mean_abs_stab())
            pca_auc, _ = stab_instance.pca_auc(**kwargs)
            self.quasi_stab_change[c] = quasi_stab - self._quasi_baseline
            self.peak_stab_change[c] = peak_stab - self._peak_baseline
            self.pca_auc_change[c] = pca_auc - self._pca_baseline

    def _drop_channel(
        self, mne_epoch_object: mne.Epochs, channel_name: str
    ) -> mne.Epochs:
        epochs_copy = mne_epoch_object.copy()
        epochs_copy.drop_channels(channel_name)
        return epochs_copy

    def plot_quasi_stab(self):
        _, ax = plt.subplots()
        sns.set_color_codes("muted")

        t = sns.histplot(
            self.quasi_stab_change,
            kde=True,
            color="b",
            bins=9,
            stat="density",
            kde_kws=dict(cut=3),
            alpha=0.4,
            edgecolor=(1, 1, 1, 0.4),
        )
        max_y = t.dataLim.get_points()[-1][-1] * 0.975
        min_x = t.dataLim.get_points()[0][0] * 1.2
        ax.set_ylim([0.0, t.dataLim.get_points()[-1][-1] * 1.01])
        # set titles and label some text
        ax.set_title("Change in average quasi stability by removing electrodes")
        ax.set_xlabel(r"$\Delta$ " + "Absolute DIS")
        ax.set_ylabel("Frequency")

        ax.text(
            min_x,
            max_y,
            f"Mean: {round(np.mean(self.quasi_stab_change), 1)}",
            fontsize=6,
        )

        ax.text(
            min_x,
            max_y * 0.97,
            f"SD: {round(np.std(self.quasi_stab_change), 1)}",
            fontsize=6,
        )

        ax.text(
            min_x,
            max_y * 0.97 * 0.97,
            f"Median: {round(np.median(self.quasi_stab_change), 1)}",
            fontsize=6,
        )

        ax.text(
            min_x,
            max_y * 0.97 * 0.97 * 0.97,
            f"MAD: {round(scipy.stats.median_abs_deviation(self.quasi_stab_change), 1)}",
            fontsize=6,
        )

        ax.text(
            min_x,
            max_y * 0.97 * 0.97 * 0.97 * 0.97,
            f"Kurtosis: {round(scipy.stats.kurtosis(self.quasi_stab_change), 1)}",
            fontsize=6,
        )

        ax.text(
            min_x,
            max_y * 0.97 * 0.97 * 0.97 * 0.97 * 0.97,
            f"Skew: {round(scipy.stats.skew(self.quasi_stab_change), 1)}",
            fontsize=6,
        )
        plt.show()
