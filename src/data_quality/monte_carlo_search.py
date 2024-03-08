"""_summary_

    _extended_summary_
    """

import numpy as np
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
