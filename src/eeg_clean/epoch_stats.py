import numpy as np
import mne
import sklearn
from neurokit2.microstates.microstates_clean import microstates_clean
from stability.epoch_stability import PeakStability, QuasiStability


class EpochStats:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    mne_epoch_obj : mne.Epochs
        _description_
    """

    def __init__(self, mne_epoch_obj: mne.Epochs, **kwargs):
        self.epoch_obj = mne_epoch_obj
        self.n_epochs = self.epoch_obj.get_data(copy=True).shape[0]
        self.peak_stability = PeakStability(self.n_epochs, **kwargs)
        self.quasi_stability = QuasiStability(self.n_epochs, **kwargs)
        self.n_gfp_peaks = np.zeros(self.n_epochs)
        self.data_at_peaks = None

    def calc_stability(
        self,
        robust_gfp=False,
        gfp_method="l2",
        data_at_peaks=True,
        **_,
    ) -> None:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        mne_epochs_obj : mne.Epochs
            _description_

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            _description_
        """
        epochs_array = self.epoch_obj.get_data(copy=True)
        for e in range(self.n_epochs):
            data, peaks, _, _ = microstates_clean(
                epochs_array[e, :, :],
                standardize_eeg=False,
                normalize=False,
                gfp_method=gfp_method,
                sampling_rate=self.epoch_obj.info["sfreq"],
                robust=robust_gfp,
            )

            self.quasi_stability.add_stability_stats(data[:, peaks])
            extended_peaks = self._t_minus_one_indices(peaks)
            self.peak_stability.add_stability_stats(data[:, extended_peaks])
            self.n_gfp_peaks[e] = len(peaks)

            if data_at_peaks:
                self.data_at_peaks = self._data_accumulate(
                    self.data_at_peaks, data, peaks
                )

    def get_peak_stability(self) -> PeakStability:
        """_summary_

        _extended_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self.peak_stability

    def get_quasi_stability(self) -> QuasiStability:
        """_summary_

        _extended_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self.quasi_stability

    def pca_auc(self, sklearn_scaler=None, **_) -> tuple[float, np.array]:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        data_type : str, optional
            _description_, by default 'data'
        sklearn_scaler : _type_, optional
            _description_, by default None

        Returns
        -------
        tuple[float, np.array]
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        data = self.data_at_peaks.T

        if sklearn_scaler is not None:
            scaler = sklearn_scaler
            data = scaler.fit_transform(data)

        pca = sklearn.decomposition.PCA(random_state=9)
        pca.fit_transform(data)
        n_components = pca.components_.shape[0]
        roc_curve = np.ndarray((n_components))
        explained_var = pca.explained_variance_ratio_
        percent = 0

        for v in range(n_components):
            percent += explained_var[v]
            roc_curve[v] = percent

        auc = np.trapz(roc_curve.flatten())
        return auc / n_components, roc_curve

    def _data_accumulate(
        self,
        data_at_peaks: np.ndarray,
        data: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        data_at_peaks : np.ndarray
            _description_
        data : np.ndarray
            _description_
        indices : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        if data_at_peaks is None:
            accumulated = data[:, indices]
        else:
            accumulated = np.concatenate((data_at_peaks, data[:, indices]), axis=1)

        return accumulated

    def _t_minus_one_indices(self, gfp_peaks_idx: np.array) -> np.array:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        gfp_peaks_idx : np.array
            _description_

        Returns
        -------
        np.array
            _description_
        """
        unorderd = np.array([gfp_peaks_idx, (gfp_peaks_idx - 1)]).flatten()
        ordered = np.sort(unorderd)
        return ordered
