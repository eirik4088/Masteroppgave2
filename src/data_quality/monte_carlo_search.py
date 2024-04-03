"""_summary_

    _extended_summary_
    """

import numpy as np
import pandas as pd
import mne


class MonteCarloSearch:
    def __init__(
        self,
        n_samples: np.ndarray,
        n_repetitions: int,
        significance_level: float
    ):
        self.n_samples = n_samples
        self.n_repetitions = n_repetitions
        self.significance = significance_level

    def search(self, condition_1: mne.Epochs, condition_2: mne.Epochs):
        return None
    
    def _repeated_measures_anova(self, n_subjects):
        df = pd.DataFrame({'subject': np.repeat(np.arange(n_subjects))})
        return None
    

    #Taken directly from mne toturial and augmented slightly
    def eeg_power_band(self, epochs):
        """EEG relative power band feature extraction.

        This function takes an ``mne.Epochs`` object and creates EEG features based
        on relative power in specific frequency bands that are compatible with
        scikit-learn.

        Parameters
        ----------
        epochs : Epochs
            The data.

        Returns
        -------
        X : numpy array of shape [n_samples, 5 * n_channels]
            Transformed data.
        """
        # specific frequency bands
        FREQ_BANDS = {
            "delta": [1.5, 3.5],
            "theta": [4, 7.5],
            "alpha": [8, 13],
            "beta": [13.5, 25],
        }

        spectrum = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0, method='welch')
        psds, freqs = spectrum.get_data(return_freqs=True)
        # Normalize the PSDs
        #psds /= np.sum(psds, axis=-1, keepdims=True)

        X = []
        for fmin, fmax in FREQ_BANDS.values():
            psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
            X.append(psds_band.reshape(len(psds), -1))

        return np.concatenate(X, axis=1)
