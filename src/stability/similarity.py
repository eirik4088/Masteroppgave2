import dataclasses
import warnings
from typing import Callable
import numpy as np


@dataclasses.dataclass(frozen=True)
class _ImplementedSimilarityMeasures:
    name: str
    description: str
    function: Callable


class Similarity:
    """Enables comparison of topographical maps via various similarity measures

    In EEG topgraphical analysis is central and this class enables comparison of voltage
    distributions. The parent class features a central method, calc_similarity, which acts as a
    gateway for calculating the similarity between two topographical maps using different metrics.
    New similarity functions can easly be incorperated as long as their corresponding data dicts
    are added to the _implemented_measures var. Costum functions can be supplied to the class
    function without intervening in the implementation. The only constrictions on new implemented
    methods are spessified in the smilarity testing module.

    Examples
    --------
    >>> my_sims = Similarity().calc_similarity(np.array([-3, 1, 2]), np.array([-3, 1, 2]), "dis")
    >>> my_sims2 = Similarity().calc_similarity(np.array([-3, 1, 2]), np.array([3, -1, -2]), "dis")
    >>> print(my_sims)
    (0.0, 1.0)
    >>> print(my_sims2)
    (2.0, 1.0)
    >>> my_sims3 = Similarity().calc_similarity(np.array([-3, 1, 2]), np.array([6, -2, -4]), "dis")
    >>> my_sims2 == my_sims3
    True
    >>> my_sims4 = Similarity().calc_similarity(np.array([-6, 2, 4]), np.array([-3, 1, 2]), "dis")
    >>> my_sims == my_sims4
    True
    """

    def __init__(self) -> None:
        self._implemented_measures = [
            _ImplementedSimilarityMeasures(
                name="dis",
                description="Global dissimalarity: \
                The root average of squared differences between two gfp normalized vectors.",
                function=self._dis,
            ),
            _ImplementedSimilarityMeasures(
                name="cosine",
                description="Cosine similarity: \
                    the angel between two vectors in n dimentional space.",
                function=self._cosine,
            ),
        ]

    def calc_similarity(
        self, v1: np.ndarray, v2: np.ndarray, similarity_method: str = "dis", **kwargs
    ) -> tuple[float, float]:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        v1 : np.array
            _description_
        v2 : np.array
            _description_
        stability_method : str
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        self.__same_length(v1, v2)

        for m in self.get_implemented_measures(function=True):
            if m[0] == similarity_method:
                return m[1](v1, v2, **kwargs)

        raise ValueError(
            f'Method "{similarity_method}" does not exist. Call \
                         get_implemented_measures to see all available predefined methods.'
        )

    def get_implemented_measures(
        self, description=False, function=False
    ) -> list[str] | list[tuple[str, str]]:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        description : bool, optional
            _description_, by default False

        Returns
        -------
        list[str] | list[tuple[str, str]]
            _description_
        """
        if description:
            return [(m.name, m.description) for m in self._implemented_measures]

        if function:
            return [(m.name, m.function) for m in self._implemented_measures]

        return (m.name for m in self._implemented_measures)

    def _dis(self, v1: np.ndarray, v2: np.ndarray, **_) -> tuple[float, float]:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        v1 : np.array
            _description_
        v2 : np.array
            _description_

        Returns
        -------
        tuple[float, float]
            _description_
        """
        v1_average_ref = v1 - np.mean(v1)
        v2_average_ref = v2 - np.mean(v2)

        if self.__zero_vectors([v1, v2]):
            warnings.warn(
                "Zero-vector is not suitible for global dissimilarity, \
                          look for nan values in results."
            )

        v1_normed = v1_average_ref / np.sqrt(np.mean(np.square(v1_average_ref)))
        v2_normed = v2_average_ref / np.sqrt(np.mean(np.square(v2_average_ref)))

        diff = v1_normed - v2_normed
        dis = np.sqrt(np.mean(np.square(diff)))
        abs_dis = np.abs(dis - 1)
        return dis, abs_dis

    def _cosine(self, v1: np.ndarray, v2: np.ndarray, **_) -> tuple[float, float]:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        v1 : np.array
            _description_
        v2 : np.array
            _description_

        Returns
        -------
        tuple[float, float]
            _description_
        """
        if self.__zero_vectors([v1, v2]):
            warnings.warn(
                "Zero-vector is not suitible for cosine similarity, \
                          look for nan values in results."
            )

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        unit_v1 = v1 / v1_norm
        unit_v2 = v2 / v2_norm

        cosine_similarity = unit_v1.dot(unit_v2)
        abs_cosine_similarity = abs(cosine_similarity)
        return cosine_similarity, abs_cosine_similarity

    def __same_length(self, v1: np.ndarray, v2: np.ndarray):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        v1 : np.array
            _description_
        v2 : np.array
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if v1.shape[0] != v2.shape[0]:
            raise ValueError("Vectors v1 and v2 is not the same length.")

    def __zero_vectors(self, vector_list: list[np.array, np.array]):
        for v in vector_list:
            if np.all(v == 0):
                return True
        return False
