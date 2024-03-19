"""_summary_

    _extended_summary_
    """

import numpy as np
import scipy
import mne
import sklearn
import random
from eeg_clean.epoch_stats import EpochStats
from eeg_clean.channel_stats import ChannelStats


class Clean:
    """_summary_

    _extended_summary_
    """

    def __init__(self, mne_epochs_obj: mne.Epochs, av_ref=False, **kwargs) -> None:
        if av_ref:
            mne_epochs_obj.set_eeg_reference(verbose=False)
        self.channel_stats = ChannelStats(mne_epochs_obj, **kwargs)
        self.bad_channel_index = self.find_bad_channels(**kwargs)

        if self.bad_channel_index is not None:
            self.bad_channel_index = np.unique(self.bad_channel_index)

        self.ch_names = np.array(mne_epochs_obj.info["ch_names"])

        if self.bad_channel_index is not None:
            self.epochs_obj = mne_epochs_obj.copy().drop_channels(
                self.ch_names[self.bad_channel_index]
            )
        else:
            self.epochs_obj = mne_epochs_obj.copy()

        self.epoch_stats = EpochStats(self.epochs_obj, **kwargs)
        self.epoch_stats.calc_stability(**kwargs)
        self.bad_epoch_index = None

    def find_bad_epochs(
        self, quasi_args: dict = None, peaks_args: dict = None, find_random=False
    ) -> np.ndarray:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        at_gfp_peaks_method : str, optional
            _description_, by default "function_threshold"
        between_gfp_peaks_method : str, optional
            _description_, by default "function_threshold"

        Returns
        -------
        np.ndarray
            _description_
        """
        if find_random:
            self.bad_epoch_index = self._bad_epochs_random()
            return self.bad_epoch_index

        if quasi_args is not None:
            quasi_stability = self.epoch_stats.quasi_stability
            quasi_stab = quasi_stability.get_mean_stab().copy()
            quasi_abs_stab = quasi_stability.get_mean_abs_stab().copy()
            bad_quasi = self._bad_epochs_by_quasi_stab(
                stab_values=quasi_stab - 2, abs_stab_values=quasi_abs_stab, **quasi_args
            )

        if peaks_args is not None:
            peak_stability = self.epoch_stats.peak_stability
            peak_stab = peak_stability.get_mean_stab().copy()
            peak_abs_stab = peak_stability.get_mean_abs_stab().copy()
            bad_peak = self._bad_epochs_by_peaks_stab(
                stab_values=peak_stab, abs_stab_values=peak_abs_stab, **peaks_args
            )

        if (
            peaks_args is not None
            and quasi_args is not None
            and bad_peak is not None
            and bad_quasi is not None
        ):
            bad_epoch_index = np.concatenate((bad_peak, bad_quasi))
        elif quasi_args is not None and bad_quasi is not None:
            bad_epoch_index = bad_quasi.copy()
        elif peaks_args is not None and bad_peak is not None:
            bad_epoch_index = bad_peak.copy()
        else:
            bad_epoch_index = None

        if bad_epoch_index is not None:
            self.bad_epoch_index = np.unique(np.array(bad_epoch_index).copy())
        else:
            self.bad_epoch_index = None

        return np.unique(bad_epoch_index)

    def find_bad_channels(
        self,
        quasi=False,
        peaks=False,
        corr=False,
        find_random=False,
        top_n = None,
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

        if find_random:
            return self._bad_channels_random()
        
        if top_n is not None:
            return self._top_bad_channels(top_n)

        if quasi:
            quasi_stab = self.channel_stats.quasi_stab_change.copy()
            bad_quasi = self._bad_channels_by_skew(quasi_stab)

        if peaks:
            peak_stab = self.channel_stats.peak_stab_change.copy()
            bad_peak = self._bad_channels_by_skew(peak_stab)

        if quasi and peaks and bad_peak is not None and bad_quasi is not None:
            bad_channel_index = np.concatenate((bad_peak.copy(), bad_quasi.copy()))

        elif quasi and bad_quasi is not None:
            bad_channel_index = bad_quasi.copy()

        elif peaks and bad_peak is not None:
            bad_channel_index = bad_peak.copy()

        else:
            bad_channel_index = None

        if bad_channel_index is not None:
            bad_channel_index = np.unique(np.array(bad_channel_index))

        if corr:
            bad_corr = np.where(self._scale(self.channel_stats.pca_auc_change) > 2)[0]
            if bad_corr.size == 0:
                pass
            else:
                if bad_channel_index is not None:
                    bad_channel_index = np.concatenate((bad_channel_index, bad_corr))
                else:
                    bad_channel_index = bad_corr

        return bad_channel_index

    def _scale(self, vals):
        v = vals.reshape(-1, 1)
        scaler = sklearn.preprocessing.StandardScaler()
        noe = scaler.fit_transform(v)
        return noe

    def _bad_epochs_by_peaks_stab(
        self,
        method: str,
        stab_values: np.ndarray = None,
        abs_stab_values: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        dis_values : np.ndarray
            _description_
        abs_dis_values : np.ndarray
            _description_
        at_method : str
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        if method == "threshold":
            return self._thresholding(stab_values, **kwargs)

        if method == "function_threshold":
            return self._linear_function_thresholding(
                stab_values, abs_stab_values, **kwargs
            )

        raise ValueError(f"Value {method} is not valid for method argument.")

    def _bad_epochs_by_quasi_stab(
        self,
        method: str,
        stab_values: np.ndarray = None,
        abs_stab_values: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        dis_values : np.ndarray
            _description_
        abs_dis_values : np.ndarray
            _description_
        between_method : str
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        if method == "threshold":
            return self._thresholding(stab_values, **kwargs)

        if method == "function_threshold":
            return self._linear_function_thresholding(
                stab_values, abs_stab_values, **kwargs
            )

        raise ValueError(f"Value {method} is not valid for method argument.")

    def _bad_epochs_random(self):
        quasi_stability = self.epoch_stats.quasi_stability
        quasi_stab = self._scale(quasi_stability.get_mean_stab().copy())
        quasi_stab = (quasi_stab / np.max(quasi_stab)) * 0.9
        quasi_abs_stab = self._scale(quasi_stability.get_mean_abs_stab().copy())
        quasi_abs_stab = (quasi_abs_stab / np.max(quasi_abs_stab)) * 0.9

        peak_stability = self.epoch_stats.peak_stability
        peak_stab = self._scale(peak_stability.get_mean_stab().copy())
        peak_stab = (peak_stab / np.max(peak_stab)) * 0.9
        peak_abs_stab = self._scale(peak_stability.get_mean_abs_stab().copy())
        peak_abs_stab = (peak_abs_stab / np.max(peak_abs_stab)) * 0.9

        bad_epochs = []

        for i, qs in enumerate(quasi_stab):
            if self.__decision(qs):
                bad_epochs.append(i)
        for i, qas in enumerate(quasi_abs_stab):
            if self.__decision(qas):
                bad_epochs.append(i)
        for i, ps in enumerate(peak_stab):
            if self.__decision(ps):
                bad_epochs.append(i)
        for i, pas in enumerate(peak_abs_stab):
            if self.__decision(pas):
                bad_epochs.append(i)

        if len(bad_epochs) == 0:
            bad_epochs = None
        else:
            bad_epochs = np.unique(bad_epochs)

        return bad_epochs

    def _bad_channels_by_skew(self, values: np.ndarray):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        values : np.ndarray
            _description_

        Returns
        -------
        _type_
            _description_
        """
        bad_channels = []
        order = np.argsort(values)
        temporary_vals = values.copy()[order]
        skew = np.inf
        while skew >= 0:

            if self.channel_stats.pca_auc_change[order[-1]] > 0:
                bad_channels.append(order[-1])

            order = np.delete(order, -1)
            temporary_vals = np.delete(temporary_vals, -1)
            skew = scipy.stats.skew(temporary_vals)

        bad_channels = np.array(bad_channels)

        if bad_channels.size == 0:
            bad_channels = None

        return bad_channels

    def _top_bad_channels(self, n: tuple[int, int] = None):
        quasi_stab = self.channel_stats.quasi_stab_change.copy()
        peak_stab = self.channel_stats.peak_stab_change.copy()

        bad_channels = []

        quasi_order = np.argsort(quasi_stab)
        peaks_order = np.argsort(peak_stab)

        for i in quasi_order[-n[0]-1:]:
            bad_channels.append(i)
        for q in peaks_order[-n[1]-1:]:
            bad_channels.append(q)

        bad_channels = np.array(bad_channels[0])

        if bad_channels.size == 0:
            bad_channels = None

        else:
            bad_channels = np.unique(bad_channels)

        return bad_channels

    def _bad_channels_random(self):
        quasi_stab = self._scale(self.channel_stats.quasi_stab_change.copy())
        quasi_stab = (quasi_stab / max(quasi_stab)) * 0.9

        peak_stab = self._scale(self.channel_stats.peak_stab_change.copy())
        peak_stab = (peak_stab / max(peak_stab)) * 0.9

        bad_channels = []

        for i, qs in enumerate(quasi_stab):
            if self.__decision(np.abs(qs)):
                bad_channels.append(i)

        for i, ps in enumerate(peak_stab):
            if self.__decision(np.abs(ps)):
                bad_channels.append(i)

        if len(bad_channels) == 0:
            bad_channels = None
        else:
            bad_channels = np.unique(bad_channels)

        return bad_channels

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

    def __decision(self, probability):
        return random.random() < probability
