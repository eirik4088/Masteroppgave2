"""_summary_

    _extended_summary_
    """

from abc import ABC, abstractmethod
import numpy as np
from stability.similarity import Similarity


class EpochStability(ABC, Similarity):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    ABC : _type_
        _description_
    Stability : _type_
        _description_
    """

    def __init__(self, instances: int, stability_method: str) -> None:
        super().__init__()
        self.mean_stab = np.zeros((instances))
        self.mean_abs_stab = np.zeros((instances))
        self._method = stability_method
        self._instances = instances
        self._current_instance = 0
        self._obj_filled = False

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
        if self._obj_filled:
            raise ValueError(
                "Objects predefined space for stability statistics is allready full."
            )

        self._stability_stats(epoch_topo_maps)
        self._update()

    def _update(self):
        """_summary_

        _extended_summary_
        """
        self._current_instance += 1

        if self._current_instance == self._instances:
            self._obj_filled = True

    @abstractmethod
    def _stability_stats(self, epoch_topo_maps):
        pass


class QuasiStability(EpochStability):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    EpochStability : _type_
        _description_

    Examples
    --------
    >>> 2 == 2
    True
    """

    def __init__(
        self,
        instances: int,
        stability_method: str = "dis",
        switching_frq_thresholds: np.ndarray = None,
        **_
    ) -> None:
        super().__init__(instances, stability_method)
        self.switching_thresholds = switching_frq_thresholds
        self._n_thresholds = 0

        if self.switching_thresholds is not None:
            self._n_thresholds = self.switching_thresholds.shape[0]
            self.switching_freqs = np.zeros((instances, self._n_thresholds))

    def _stability_stats(self, epoch_topo_maps):
        n_values = epoch_topo_maps.shape[1] - 1

        for m in range(n_values):
            dis_stats = self.calc_similarity(
                epoch_topo_maps[:, m], epoch_topo_maps[:, m + 1], self._method
            )
            self.mean_stab[self._current_instance] += dis_stats[0]
            self.mean_abs_stab[self._current_instance] += dis_stats[1]

            for t in range(self._n_thresholds):
                if dis_stats[0] > self.switching_thresholds[t]:
                    self.switching_freqs[self._current_instance, t] += 1

        self.mean_stab[self._current_instance] /= n_values
        self.mean_abs_stab[self._current_instance] /= n_values


class PeakStability(EpochStability):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    EpochStability : _type_
        _description_
    """

    def __init__(self, instances: int, stability_method: str = "dis", **_) -> None:
        super().__init__(instances, stability_method)

    def _stability_stats(self, epoch_topo_maps):
        n_values = epoch_topo_maps.shape[1]

        for m in range(0, n_values, 2):
            dis_stats = self.calc_similarity(
                epoch_topo_maps[:, m], epoch_topo_maps[:, m + 1], self._method
            )
            self.mean_stab[self._current_instance] += dis_stats[0]
            self.mean_abs_stab[self._current_instance] += dis_stats[1]

        self.mean_stab[self._current_instance] /= n_values
        self.mean_abs_stab[self._current_instance] /= n_values
