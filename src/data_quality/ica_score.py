"""_summary_

    _extended_summary_
    """

import numpy as np
import mne
from mne_icalabel import label_components


class IcaScore:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    mne_epochs_obj : mne.Epochs
        _description_
    """

    def __init__(self, mne_epochs_obj: mne.Epochs, **kwargs):
        self.epoch_obj = mne_epochs_obj
        self.ica, self.ica_labels = self.__fit_transform(**kwargs)
        self._brain_components, self._bio_artifacts = self.__get_components_idx(
            **kwargs
        )

    def get_n_components(self):
        """_summary_

        _extended_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self._brain_components.size, self._bio_artifacts.size

    def get_explained_var(self, bio_components=False):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        bio_components : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        if bio_components:
            idx = np.concatenate((self._brain_components, self._bio_artifacts))
            if idx.size > 0:
                print(idx)
                return self.ica.get_explained_variance_ratio(self.epoch_obj, components=idx)
            return {'eeg': 0}
        if self._brain_components.size > 0:
            return self.ica.get_explained_variance_ratio(
                self.epoch_obj, components=self._brain_components
            )
        return {'eeg': 0}

    def __fit_transform(self, n_components=16, **_):
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            max_iter="auto",
            method="infomax",
            random_state=97,
            fit_params=dict(extended=True),
        )
        ica.fit(self.epoch_obj)
        ica_labels = label_components(self.epoch_obj, ica, method="iclabel")
        return ica, ica_labels

    def __get_components_idx(self, threshold=0.9, **_):
        brain = []
        bio_artifact = []

        for ic in range(self.ica.n_components):
            if self.ica_labels["y_pred_proba"][ic] > threshold:

                if self.ica_labels["labels"][ic] == "brain":
                    brain.append(ic)

                if (
                    self.ica_labels["labels"][ic] == "eye blink"
                    or self.ica_labels["labels"][ic] == "muscle artifact"
                ):
                    bio_artifact.append(ic)

        return np.array(brain, dtype=int), np.array(bio_artifact, dtype=int)
