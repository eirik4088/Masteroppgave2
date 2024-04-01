"""_summary_

    _extended_summary_
    """

import mne
import scipy
from scipy.stats import iqr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from eeg_clean.epoch_stats import EpochStats


class ChannelStatsNew:
    """_summary_

    _extended_summary_
    """

    def __init__(
        self, mne_epochs_object: mne.Epochs, dist_specifics: dict, **kwargs
    ) -> None:
        self.epoch_obj = mne_epochs_object
        epoch_stats_object = EpochStats(self.epoch_obj, **kwargs)
        epoch_stats_object.calc_stability(**kwargs)
        self.ch_name_list = np.array(self.epoch_obj.info["ch_names"])
        self.n_channels = len(self.ch_name_list)
        self.dists = list(dist_specifics.keys())

        if "quasi" in dist_specifics.keys():
            self._quasi_baseline, _ = self._calc(
                epoch_stats_object.quasi_stability.get_mean_stab(),
                **dist_specifics["quasi"],
            )
            self._quasi_spred_baseline = iqr(self._quasi_baseline)
            self.quasi_stab_change = np.zeros((self.n_channels))
        else:
            self.quasi_stab_change = None

        if "peak" in dist_specifics.keys():
            self._peak_baseline, _ = self._calc(
                epoch_stats_object.peak_stability.get_mean_stab(),
                **dist_specifics["peak"],
            )
            self._peak_spred_baseline = iqr(self._peak_baseline)
            self.peak_stab_change = np.zeros((self.n_channels))
        else:
            self.peak_stab_change = None

        if "n_peaks" in dist_specifics.keys():
            self._n_peaks_baseline, _ = self._calc(
                epoch_stats_object.n_gfp_peaks, **dist_specifics["n_peaks"]
            )
            self._n_peaks_spred_baseline = iqr(self._n_peaks_baseline)
            self.n_peaks_change = np.zeros((self.n_channels))
        else:
            self.n_peaks_change = None

        self._leave_one_out(dist_specifics, **kwargs)

    def _leave_one_out(self, dist_specifics, **kwargs) -> None:
        for c in range(self.n_channels):
            instance = self._drop_channel(self.epoch_obj, self.ch_name_list[c])
            stab_instance = EpochStats(instance, **kwargs)
            stab_instance.calc_stability(**kwargs)

            if "quasi" in dist_specifics.keys():
                quasi_stab, ampl = self._calc(
                    stab_instance.quasi_stability.get_mean_stab(),
                    spred_baseline=self._quasi_spred_baseline,
                    **dist_specifics["quasi"],
                )
                self.quasi_stab_change[c] = (quasi_stab - self._quasi_baseline) * ampl

            if "peak" in dist_specifics.keys():
                peak_stab, ampl = self._calc(
                    stab_instance.peak_stability.get_mean_stab(),
                    spred_baseline=self._peak_spred_baseline,
                    **dist_specifics["peak"],
                )
                self.peak_stab_change[c] = (peak_stab - self._peak_baseline) * ampl

            if "n_peaks" in dist_specifics.keys():
                n_peaks, ampl = self._calc(
                    stab_instance.n_gfp_peaks,
                    spred_baseline=self._n_peaks_spred_baseline,
                    **dist_specifics["n_peaks"],
                )
                self.n_peaks_change[c] = (n_peaks - self._n_peaks_baseline) * ampl

    def _calc(self, values, central, spred_corrected, spred_baseline=None):
        amplifier = 1

        if central == "mean":
            calculated = np.mean(values)
            if spred_baseline is not None:
                if spred_corrected == "IQR":
                    amplifier = np.abs(spred_baseline - iqr(values))
                if spred_corrected == "var":
                    raise NotImplementedError

        elif central == "median":
            calculated = np.median(values)
            if spred_baseline is not None:
                if spred_corrected == "IQR":
                    amplifier = np.abs(spred_baseline - iqr(values))
                if spred_corrected == "var":
                    raise NotImplementedError

        else:
            raise ValueError("Problems with dis_specifics!")

        return calculated, amplifier

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
