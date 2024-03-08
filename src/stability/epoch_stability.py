"""_summary_

    _extended_summary_
    """

from abc import ABC, abstractmethod
import warnings
import numpy as np
from stability.similarity import Similarity


class EpochStability(ABC, Similarity):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    instances : int
        _description_
    stability_method : str
        _description_
    """

    def __init__(self, instances: int, stability_method: str) -> None:
        super().__init__()
        self.__validate_instances(instances)
        self._mean_stab = np.zeros((instances))
        self._mean_abs_stab = np.zeros((instances))
        self._method = stability_method
        self._instances = instances
        self._current_instance = 0
        self.obj_filled = False

    def add_stability_stats(self, epoch_topo_maps: np.ndarray) -> None:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        epoch_topo_maps : np.ndarray
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        self._validate_arraytype(epoch_topo_maps)

        if self.obj_filled:
            raise ValueError(
                "Objects predefined space for stability statistics is allready full."
            )

        self._stability_stats(epoch_topo_maps)
        self.__update()

    def get_mean_stab(self):
        """_summary_

        _extended_summary_

        Returns
        -------
        _type_
            _description_
        """
        if self.obj_filled:
            return self._mean_stab

        warnings.warn("Object is not filled, returning None")
        return None

    def get_mean_abs_stab(self):
        """_summary_

        _extended_summary_

        Returns
        -------
        _type_
            _description_
        """
        if self.obj_filled:
            return self._mean_abs_stab

        warnings.warn("Object is not filled, returning None")
        return None

    def __update(self):
        self._current_instance += 1

        if self._current_instance == self._instances:
            self.obj_filled = True

    def __validate_instances(self, instances):
        if isinstance(instances, int):
            if instances <= 0:
                raise ValueError(f"Only positive integers allowed, got {instances}.")
        else:
            raise TypeError(f"Got {type(instances)}, when it should be an int.")

    @abstractmethod
    def _stability_stats(self, epoch_topo_maps: np.ndarray):
        pass


class QuasiStability(EpochStability):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    instances : int
        _description_
    stability_method : str, optional
        _description_, by default "dis"
    switching_frq_thresholds : np.ndarray, optional
        _description_, by default None

    Examples
    --------
    >>> maps = np.array([
    ... [-1, -1, -1, 1],
    ... [4, 4, 4, -4],
    ... [-2, -2, -2, 2]
    ... ])
    >>> quasi_stabs = QuasiStability(1)
    >>> quasi_stabs.add_stability_stats(maps)
    >>> print(round(quasi_stabs.get_mean_stab()[0], 2))
    0.67
    >>> print(quasi_stabs.get_mean_abs_stab())
    [1.]
    >>> quasi_stabs2 = QuasiStability(2)
    >>> quasi_stabs2.add_stability_stats(maps[:, :2])
    >>> print(quasi_stabs2.get_mean_stab())
    None
    >>> print(quasi_stabs2.get_mean_abs_stab())
    None
    >>> quasi_stabs2.add_stability_stats(maps[:, 2:])
    >>> print(quasi_stabs2.get_mean_stab())
    [0. 2.]
    >>> print(quasi_stabs2.get_mean_abs_stab())
    [1. 1.]
    """

    def __init__(
        self,
        instances: int,
        stability_method: str = "dis",
        switching_frq_thresholds: np.ndarray = None,
        **_,
    ) -> None:
        super().__init__(instances, stability_method)
        self.switching_thresholds = switching_frq_thresholds
        self._n_thresholds = 0

        if self.switching_thresholds is not None:
            self.__validate_switching_thresholds()
            self._n_thresholds = self.switching_thresholds.shape[0]
            self.switching_freqs = np.zeros((instances, self._n_thresholds))

    def _stability_stats(self, epoch_topo_maps: np.ndarray):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        epoch_topo_maps : np.ndarray
            _description_
        """
        n_values = epoch_topo_maps.shape[1] - 1

        for m in range(n_values):
            dis_stats = self.calc_similarity(
                epoch_topo_maps[:, m], epoch_topo_maps[:, m + 1], self._method
            )
            self._mean_stab[self._current_instance] += dis_stats[0]
            self._mean_abs_stab[self._current_instance] += dis_stats[1]

            for t in range(self._n_thresholds):
                if dis_stats[0] > self.switching_thresholds[t]:
                    self.switching_freqs[self._current_instance, t] += 1

        self._mean_stab[self._current_instance] /= n_values
        self._mean_abs_stab[self._current_instance] /= n_values

    def __validate_switching_thresholds(self):
        self._validate_arraytype(self.switching_thresholds)
        self._validate_dim_size(1, self.switching_thresholds)

        if self._method == "dis":
            for v in self.switching_thresholds:
                if v > 2 or v < 0:
                    raise ValueError(
                        "The supplied swithching thresholds should be \
                                     between 0 and 2 for Global Dissimilarity"
                    )

        if self._method == "cosine":
            for v in self.switching_thresholds:
                if v > 1 or v < -1:
                    raise ValueError(
                        "The supplied swithching thresholds should be \
                                     between -1 and 1 for Cosine Similarity"
                    )


class PeakStability(EpochStability):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    instances : int
        _description_
    stability_method : str, optional
        _description_, by default "dis"

    Examples
    --------
    >>> maps = np.array([
    ... [-1, -1, -4, 4],
    ... [4, 4, 1, -1],
    ... [-2, -2, -2, 2]
    ... ])
    >>> peak_stabs = PeakStability(1)
    >>> peak_stabs.add_stability_stats(maps)
    >>> print(peak_stabs.get_mean_stab())
    [1.]
    >>> print(peak_stabs.get_mean_abs_stab())
    [1.]
    >>> peak_stabs2 = PeakStability(2)
    >>> peak_stabs2.add_stability_stats(maps[:, :2])
    >>> print(peak_stabs2.get_mean_stab())
    None
    >>> print(peak_stabs2.get_mean_abs_stab())
    None
    >>> peak_stabs2.add_stability_stats(maps[:, 2:])
    >>> print(peak_stabs2.get_mean_stab())
    [0. 2.]
    >>> print(peak_stabs2.get_mean_abs_stab())
    [1. 1.]
    """

    def __init__(self, instances: int, stability_method: str = "dis", **_) -> None:
        super().__init__(instances, stability_method)

    def _stability_stats(self, epoch_topo_maps):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        epoch_topo_maps : _type_
            _description_
        """
        n_values = epoch_topo_maps.shape[1]

        for m in range(0, n_values, 2):
            dis_stats = self.calc_similarity(
                epoch_topo_maps[:, m], epoch_topo_maps[:, m + 1], self._method
            )
            self._mean_stab[self._current_instance] += dis_stats[0]
            self._mean_abs_stab[self._current_instance] += dis_stats[1]

        self._mean_stab[self._current_instance] /= n_values / 2
        self._mean_abs_stab[self._current_instance] /= n_values / 2
