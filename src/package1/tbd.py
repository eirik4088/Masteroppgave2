from copy import deepcopy
import numpy as np
import pycrostates

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.parallel import parallel_func
from numpy.random import Generator, RandomState
from numpy.typing import NDArray

from pycrostates._typing import CHData, Picks, RANDomState
from pycrostates.utils import _corr_vectors
from pycrostates.utils._checks import _check_n_jobs, _check_random_state, _check_type
from pycrostates.utils._docs import copy_doc, fill_doc
from pycrostates.utils._logs import logger
from pycrostates.cluster._base import _BaseCluster
from pycrostates.segmentation import RawSegmentation


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import mne
import pathlib
import pycrostates
import sklearn
import neurokit2

from skstab import StadionEstimator


def correct_cluster_labels(
    previous_cluster_centers: np.ndarray,
    new_cluster_centers: np.ndarray,
    new_cluster_labels: np.array,
):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    previous_cluster_centers : np.ndarray
        _description_
    new_cluster_centers : np.ndarray
        _description_
    new_cluster_labels : np.array
        _description_

    Returns
    -------
    _type_
        _description_
    """
    converter = np.array(range(len(new_cluster_centers)))
    eucledian_min = np.array(converter, dtype=float)
    centers_distance = np.ndarray(
        shape=(len(new_cluster_centers), len(new_cluster_centers)), dtype=float
    )

    for pcc, pcc_e in enumerate(previous_cluster_centers):
        for ncc, ncc_e in enumerate(new_cluster_centers):
            centers_distance[pcc][ncc] = np.linalg.norm(pcc_e - ncc_e)

    centers_distance_itt = centers_distance.copy()

    for d in range(len(centers_distance_itt)):
        min_value = min(centers_distance_itt[0])
        indx = np.argmin(centers_distance_itt[0])
        eucledian_min[d] = min_value
        print(min_value)

        centers_distance_itt = np.delete(centers_distance_itt, indx, 1)
        centers_distance_itt = np.delete(centers_distance_itt, 0, 0)

    print(centers_distance)

    print(eucledian_min)

    for m, m_v in enumerate(eucledian_min):
        print(np.where(centers_distance[m] == m_v))
        converter[np.where(centers_distance[m] == m_v)] = m

    print(converter)

    for l, l_v in enumerate(new_cluster_labels):
        new_cluster_labels[l] = converter[l_v]

    return new_cluster_labels, np.mean(eucledian_min)


def scale_data(data: np.ndarray, z_score: bool = True) -> np.ndarray:
    """Normalize observations to unit norm.

    In microstate analysis observations that are linear combiantions of 
    each other in the n(=channels)-dimentional space constitutes the same 
    voltage distribution and belongs to the same microstate. This function 
    therefor normalize each observation to unit norm as to disregard the 
    voltage strength of each topographie. Option to also Z-transform the
    unit normalized data as to construct mean=0 and std=1 for each fature.

    Parameters
    ----------
    data : np.ndarray of shape (n_samples, n_features)
        Input samples
    Z_score : bool, optional
        Wether to Z-tranform data, by default False

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
        Transformed data
    """
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    unit_vectors = data / norms

    if z_score:
        scaler = sklearn.preprocessing.StandardScaler()
        z_unit_vectors = scaler.fit(unit_vectors)
        z_unit_vectors = z_unit_vectors.transform(unit_vectors)
        return z_unit_vectors
    return unit_vectors


class ModKMeansSkstab(pycrostates.cluster.ModKMeans):
    def __init__(
        self,
        n_clusters: int,
        inst: Union[BaseRaw, BaseEpochs, CHData],
        n_init: int = 100,
        max_itr: int = 300,
        tol: Union[int, float] = 1e-6,
        random_state=None,
    ):
        super(pycrostates.cluster.ModKMeans, self).__init__()

        # k-means has a fix number of clusters defined at init
        self._n_clusters = _BaseCluster._check_n_clusters(n_clusters)
        self._cluster_names = [str(k) for k in range(self.n_clusters)]

        # k-means settings
        self._n_init = ModKMeansSkstab._check_n_init(n_init)
        self._max_iter = ModKMeansSkstab._check_max_iter(max_itr)
        self._tol = ModKMeansSkstab._check_tol(tol)
        self._random_state = _check_random_state(random_state)

        # fit variables
        self._GEV_ = None
        self.inst = inst

    def fit(
        self,
        d: np.ndarray,
        picks: Picks = "eeg",
        tmin: Optional[Union[int, float]] = None,
        tmax: Optional[Union[int, float]] = None,
        reject_by_annotation: bool = True,
        n_jobs: int = 1,
        *,
        verbose: Optional[str] = None,
    ) -> None:
        """Compute cluster centers.

        Parameters
        ----------
        inst : Raw | Epochs | ChData
            MNE `~mne.io.Raw`, `~mne.Epochs` or `~pycrostates.io.ChData` object from
            which to extract :term:`cluster centers`.
        picks : str | list | slice | None
            Channels to include. Note that all channels selected must have the same
            type. Slices and lists of integers will be interpreted as channel indices.
            In lists, channel name strings (e.g. ``['Fp1', 'Fp2']``) will pick the given
            channels. Can also be the string values ``“all”`` to pick all channels, or
            ``“data”`` to pick data channels. ``"eeg"`` (default) will pick all eeg
            channels. Note that channels in ``info['bads']`` will be included if their
            names or indices are explicitly provided.
        %(tmin_raw)s
        %(tmax_raw)s
        %(reject_by_annotation_raw)s
        %(n_jobs)s
        %(verbose)s
        """

        n_jobs = _check_n_jobs(n_jobs)
        data = d.T

        inits = self._random_state.randint(
            low=0, high=100 * self._n_init, size=(self._n_init)
        )

        if n_jobs == 1:
            best_gev, best_maps, best_segmentation = None, None, None
            count_converged = 0
            for init in inits:
                gev, maps, segmentation, converged = ModKMeansSkstab._kmeans(
                    data, self._n_clusters, self._max_iter, init, self._tol
                )
                if not converged:
                    continue
                if best_gev is None or gev > best_gev:
                    best_gev, best_maps, best_segmentation = (
                        gev,
                        maps,
                        segmentation,
                    )
                count_converged += 1
        else:
            parallel, p_fun, _ = parallel_func(
                ModKMeansSkstab._kmeans, n_jobs, total=self._n_init
            )
            runs = parallel(
                p_fun(data, self._n_clusters, self._max_iter, init, self._tol)
                for init in inits
            )
            try:
                best_run = np.nanargmax([run[0] if run[3] else np.nan for run in runs])
                best_gev, best_maps, best_segmentation, _ = runs[best_run]
                count_converged = sum(run[3] for run in runs)
            except ValueError:
                best_gev, best_maps, best_segmentation = None, None, None
                count_converged = 0

        if best_gev is not None:
            logger.info(
                "Selecting run with highest GEV = %.2f%% after %i/%i "
                "iterations converged.",
                best_gev * 100,
                count_converged,
                self._n_init,
            )
        else:
            logger.error(
                "All the K-means run failed to converge. Please adapt the tolerance "
                "and the maximum number of iteration."
            )
            self.fitted = False  # reset variables related to fit
            return  # break early

        self._GEV_ = best_gev
        self._cluster_centers_ = best_maps
        self._labels_ = best_segmentation
        self._fitted = True
        self._ignore_polarity = True
        # self._fitted_data = self._fitted_data.T

        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        # data = data.T
        centroids = self.cluster_centers_

        l = []
        for i in range(len(data)):
            m = []
            for j in range(len(centroids)):
                p = np.linalg.norm(data[i, :] - centroids[j])
                m.append(p)
            po = np.argmin(m)
            l.append(po)
        to_return = np.ravel(l)
        return to_return

