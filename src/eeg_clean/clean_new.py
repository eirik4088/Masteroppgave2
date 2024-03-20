"""_summary_

    _extended_summary_
    """

import numpy as np
import mne
import sklearn
from eeg_clean.channel_stats_new import ChannelStatsNew


class CleanNew:
    """_summary_

    _extended_summary_
    """

    def __init__(self, mne_epochs_obj: mne.Epochs, **kwargs) -> None:
        self.bad_channel_index = []
        self.epochs_obj = mne_epochs_obj.copy()
        self.ch_names = np.array(self.epochs_obj.info["ch_names"])

        while True:
            channel_stats = ChannelStatsNew(self.epochs_obj, **kwargs)
            bad_channel = self.find_bad_channel(channel_stats, **kwargs)
            if bad_channel is not None:
                print("Hello?")
                self.bad_channel_index.append(bad_channel)
                self.epochs_obj.drop_channels(self.ch_names[bad_channel])
                self.ch_names = np.array(self.epochs_obj.info["ch_names"])
            else:
                break

        if len(self.bad_channel_index) > 0:
            self.bad_channel_index = np.array(self.bad_channel_index)
        else:
            self.bad_channel_index = None

    def find_bad_channel(
        self,
        channel_stats,
        thresholds,
        **_,
    ) -> np.ndarray:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        quasi : bool
            _description_
        peaks : bool
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        if len(channel_stats.dists) != len(thresholds):
            raise ValueError("The amount of distributions and thresholds do not match.")

        for i, dist in enumerate(channel_stats.dists):
            if dist == "quasi":
                quasi = np.abs(self._scale(channel_stats.quasi_stab_change.copy()))
                order = np.argsort(quasi)
                if quasi[order][-1] > thresholds[i]:
                    return order[-1]
            if dist == "peak":
                peak = np.abs(self._scale(channel_stats.n_peaks_change.copy()))
                order = np.argsort(peak)
                if peak[order][-1] > thresholds[i]:
                    return order[-1]
            if dist == "n_peaks":
                n_peaks = np.abs(self._scale(channel_stats.n_peaks_change.copy()))
                order = np.argsort(n_peaks)
                if n_peaks[order][-1] > thresholds[i]:
                    return order[-1]

        return None

    def _scale(self, vals):
        v = vals.reshape(-1, 1)
        scaler = sklearn.preprocessing.StandardScaler()
        noe = scaler.fit_transform(v)
        return noe.flatten()

    def _thresholding(
        self, values: np.ndarray, threshold: float, exclude: str
    ) -> np.ndarray:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        dis_values : np.ndarray
            _description_
        abs_dis_values : np.ndarray
            _description_
        at_f : np.poly1d
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        if exclude == "bigger":
            return np.where(values > threshold)
        if exclude == "smaller":
            return np.where(values < threshold)

        raise ValueError(f"Value {exclude} is not valid for exclude argument.")

    def _linear_function_thresholding(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        function: np.poly1d,
        exclude: str,
    ) -> np.ndarray:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        dis_values : np.ndarray
            _description_
        abs_dis_values : np.ndarray
            _description_
        between_f : np.poly1d
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        f_of_x = np.empty
        f_of_x = function(x_values)

        if exclude == "bigger":
            exclude_bools = f_of_x < y_values
        elif exclude == "smaller":
            exclude_bools = f_of_x > y_values
        else:
            raise ValueError(f"Value {exclude} is not valid for exclude argument.")

        bad_index = np.where(exclude_bools)[0]

        if bad_index.size == 0:
            bad_index = None

        return bad_index
