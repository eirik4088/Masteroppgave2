"""_summary_

    _extended_summary_
    """

import random
import pathlib
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from multiprocessing import Process


class MonteCarloSearch:
    def __init__(
        self,
        epochs_ec,
        epochs_eo,
        n_resamples,
        repetition_list,
        significance_level: float,
    ):
        self.epochs_ec = epochs_ec
        self.epocs_eo = epochs_eo
        self.n_resamples = n_resamples
        self.repetition_list = repetition_list
        self.significance = significance_level
        self.expected_diff_percentage = None
        self.expected_delta_diff = None
        self.expected_theta_diff = None
        self.expected_alpha_diff = None
        self.expected_beta_diff = None

        if len(epochs_ec) != len(epochs_eo):
            raise ValueError("Not equal amount of participants in the two conditions")

    def search(self):
        expected_diff_count = np.zeros(len(self.repetition_list))
        expected_delta_diff = np.zeros(len(self.repetition_list))
        expected_theta_diff = np.zeros(len(self.repetition_list))
        expected_alpha_diff = np.zeros(len(self.repetition_list))
        expected_beta_diff = np.zeros(len(self.repetition_list))

        for _ in range(self.n_resamples):
            for r, rep in enumerate(self.repetition_list):
                delta, theta, alpha, beta = self._generate_data(
                    rep, self.epochs_ec, self.epocs_eo
                )
                if delta and theta and alpha and beta:
                    expected_diff_count[r] += 1
                if delta:
                    expected_delta_diff[r] += 1
                if theta:
                    expected_theta_diff[r] += 1
                if alpha:
                    expected_alpha_diff[r] += 1
                if beta:
                    expected_beta_diff[r] += 1

        expected_diff_percentage = expected_diff_count / self.n_resamples
        self.expected_diff_percentage = expected_diff_percentage

        expected_delta_percentage = expected_delta_diff / self.n_resamples
        self.expected_delta_diff = expected_delta_percentage

        expected_theta_percentage = expected_theta_diff / self.n_resamples
        self.expected_theta_diff = expected_theta_percentage

        expected_alpha_percentage = expected_alpha_diff / self.n_resamples
        self.expected_alpha_diff = expected_alpha_percentage

        expected_beta_percentage = expected_beta_diff / self.n_resamples
        self.expected_beta_diff = expected_beta_percentage

        return expected_diff_percentage

    def _generate_data(self, repetitions: int, ec_marker, eo_marker):
        ec_accumulated_delta = []
        ec_accumulated_theta = []
        ec_accumulated_alpha = []
        ec_accumulated_beta = []

        eo_accumulated_delta = []
        eo_accumulated_theta = []
        eo_accumulated_alpha = []
        eo_accumulated_beta = []

        expected_delta_diff = False
        expected_theta_diff = False
        expected_alpha_diff = False
        expected_beta_diff = False

        for p, _ in enumerate(self.epochs_ec):
            random_list = self._random_generator(
                repetitions, int(self.epochs_ec[p].get_data(copy=True).shape[0])
            )
            ec_means = self._eeg_power_band(self.epochs_ec[p][random_list])
            random_list = self._random_generator(
                repetitions, int(self.epocs_eo[p].get_data(copy=True).shape[0])
            )
            eo_means = self._eeg_power_band(self.epocs_eo[p][random_list])

            ec_accumulated_delta.append(ec_means[0])
            eo_accumulated_delta.append(eo_means[0])

            ec_accumulated_theta.append(ec_means[1])
            eo_accumulated_theta.append(eo_means[1])

            ec_accumulated_alpha.append(ec_means[2])
            eo_accumulated_alpha.append(eo_means[2])

            ec_accumulated_beta.append(ec_means[3])
            eo_accumulated_beta.append(eo_means[3])

        delta_sign = self._repeated_measures_anova(len(self.epochs_ec), ec_accumulated_delta + eo_accumulated_delta).iloc[0]["Pr > F"]
        theta_sign = self._repeated_measures_anova(len(self.epochs_ec), ec_accumulated_theta + eo_accumulated_theta).iloc[0]["Pr > F"]
        alpha_sign = self._repeated_measures_anova(len(self.epochs_ec), ec_accumulated_alpha + eo_accumulated_alpha).iloc[0]["Pr > F"]
        beta_sign = self._repeated_measures_anova(len(self.epochs_ec), ec_accumulated_beta + eo_accumulated_beta).iloc[0]["Pr > F"]
        
        if delta_sign <= self.significance and np.mean(ec_accumulated_delta) > np.mean(
            eo_accumulated_delta
        ):
            expected_delta_diff = True
        if theta_sign <= self.significance and np.mean(ec_accumulated_theta) > np.mean(
            eo_accumulated_theta
        ):
            expected_theta_diff = True
        if alpha_sign <= self.significance and np.mean(ec_accumulated_alpha) > np.mean(
            eo_accumulated_alpha
        ):
            expected_alpha_diff = True
        if beta_sign <= self.significance and np.mean(ec_accumulated_beta) > np.mean(
            eo_accumulated_beta
        ):
            expected_beta_diff = True

        return (
            expected_delta_diff,
            expected_theta_diff,
            expected_alpha_diff,
            expected_beta_diff,
        )

    def _random_generator(self, repetitions, max_index):
        random_list = random.sample(range(0, max_index), repetitions)
        return random_list

    def _repeated_measures_anova(self, n_subjects, power_means):
        df = pd.DataFrame(
            {
                "subject": np.tile(np.arange(0, n_subjects, 1), 2),
                "condition": np.repeat(["ec", "eo"], n_subjects),
                "power": power_means,
            }
        )
        analysis = AnovaRM(
            data=df, depvar="power", subject="subject", within=["condition"]
        ).fit()
        
        return analysis.anova_table

    # Taken directly from mne toturial and augmented slightly
    def _eeg_power_band(self, epochs):
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

        spectrum = epochs.compute_psd(
            picks="eeg", fmin=0.5, fmax=30.0, method="welch", verbose=False, n_jobs=1
        )
        psds, freqs = spectrum.get_data(return_freqs=True)
        # Normalize the PSDs
        # psds /= np.sum(psds, axis=-1, keepdims=True)

        X = []
        for fmin, fmax in FREQ_BANDS.values():
            psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
            X.append(psds_band.reshape(len(psds), -1))
        data = np.concatenate(X, axis=1)
        delta = np.mean(data[:, : len(epochs.ch_names)])
        theta = np.mean(data[:, len(epochs.ch_names) : len(epochs.ch_names) * 2])
        alpha = np.mean(data[:, len(epochs.ch_names) * 2 : len(epochs.ch_names) * 3])
        beta = np.mean(data[:, len(epochs.ch_names) * 3 :])

        return [delta, theta, alpha, beta]
