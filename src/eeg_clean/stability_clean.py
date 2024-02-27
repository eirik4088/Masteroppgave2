"""_summary_

    _extended_summary_
    """

import numpy as np
import mne
from eeg_clean.epoch_stats import EpochStats
from eeg_clean.channel_stats import ChannelStats


class StabilityClean:
    """_summary_

    _extended_summary_
    """

    def __init__(self, mne_epochs_obj: mne.Epochs, **kwargs) -> None:
        self.epoch_quality = EpochStats(mne_epochs_obj, **kwargs)
        self.epoch_quality.calc_stability(**kwargs)
        self.channel_stability = ChannelStats(self.epoch_quality, **kwargs)
        self.bad_epoch_index = None
        self.bad_channel_index = None

    def find_bad_epochs(
        self,
        quasi_args: dict = None,
        peaks_args: dict = None
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
        if quasi_args is not None:
            quasi_stability = self.epoch_quality.get_quasi_stability()
            quasi_dis = quasi_stability.mean_stab
            quasi_abs_dis = quasi_stability.mean_abs_stab
            bad_quasi = self._bad_epochs_by_quasi_stab(
                quasi_dis, quasi_abs_dis, **quasi_args
            )
            self.bad_epoch_index = bad_quasi

        if peaks_args is not None:
            peak_stability = self.epoch_quality.get_peak_stability()
            peak_dis = peak_stability.mean_stab
            peak_abs_dis = peak_stability.mean_abs_stab
            bad_peak = self._bad_epochs_by_peaks_stab(
                peak_dis, peak_abs_dis, **peaks_args
            )
            self.bad_epoch_index = bad_peak

        if peaks_args is not None and quasi_args is not None:
            self.bad_epoch_index = np.concatenate((bad_peak, bad_quasi))

        return self.bad_epoch_index

    def find_bad_channels(
        self,
        at_gfp_peaks_method="function_threshold",
        between_gfp_peaks_method="function_threshold",
        **kwargs,
    ):
        return

    def _bad_epochs_by_peaks_stab(
        self,
        method: str,
        dis_values: np.ndarray = None,
        abs_dis_values: np.ndarray = None,
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
            return self._thresholding(dis_values, **kwargs)
        
        if method == "function_threshold":
            return self._linear_function_thresholding(
                dis_values, abs_dis_values, **kwargs
            )

        raise ValueError(f"Value {method} is not valid for method argument.")

    def _bad_epochs_by_quasi_stab(
        self,
        method: str,
        dis_values: np.ndarray = None,
        abs_dis_values: np.ndarray = None,
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
            return self._thresholding(dis_values, **kwargs)

        if method == "function_threshold":
            return self._linear_function_thresholding(
                dis_values, abs_dis_values, **kwargs
            )

        raise ValueError(f"Value {method} is not valid for method argument.")

    def _thresholding(
        self, values: np.ndarray, threshold: np.poly1d, exclude: str
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
            return values[values < threshold]
        if exclude == "smaller":
            return values[values > threshold]

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
        f_of_x = function(x_values)

        if exclude == "bigger":
            exclude_bools = f_of_x > y_values
        if exclude == "smaller":
            exclude_bools = f_of_x < y_values
        else:
            raise ValueError(f"Value {exclude} is not valid for exclude argument.")

        bad_index = np.where(exclude_bools is True)
        return bad_index
