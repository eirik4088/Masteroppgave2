"""Tesing of the similarity module.
"""

import math
import pytest
import numpy as np
from stability.similarity import Similarity

test = Similarity()

base_array = np.array([-4, 3, -1])


# Obs floating point math is broken, rounding function..
def __rounded(sims: tuple):
    """Tuple rounding function to avoid floating point math rounding errors"""
    rounded_tuple = [np.round(value, decimals=5) for value in sims]
    return rounded_tuple


def test_calc_similarity():
    """Test to make sure all implemented similarity meassures are consistent."""
    for m in test.get_implemented_measures():
        # All pairs of vectors where they are linear combionations of each other should
        # return equal second return value and this value should indicate max similarity.
        # This should also be the case for the first return value, except when the scaler
        # is negative, and then this value should indicate min similarity.
        sim = test.calc_similarity(base_array, base_array, m)
        sim = __rounded(sim)
        sim2 = test.calc_similarity(base_array, base_array * 3, m)
        sim2 = __rounded(sim2)
        sim3 = test.calc_similarity(base_array, base_array * 0.3, m)
        sim3 = __rounded(sim3)
        assert sim == sim2 == sim3

        dissim = test.calc_similarity(base_array, base_array * -1, m)
        dissim = __rounded(dissim)
        dissim2 = test.calc_similarity(base_array, base_array * -5, m)
        dissim2 = __rounded(dissim2)
        dissim3 = test.calc_similarity(base_array, base_array * -0.7, m)
        dissim3 = __rounded(dissim3)
        assert dissim == dissim2 == dissim3
        assert sim[1] == dissim[1]
        assert sim[0] != dissim[0]

        # All other vectors should have the same properties for linear combinations,
        # except for no filed strength vectors.
        # No other vector pair should have should have vlaues outside max min.
        value_range = np.sort([sim[0], dissim[0]])
        random_vecs = np.random.random((100, 2, 3)) - 0.5

        for vecs in random_vecs:
            instance = test.calc_similarity(vecs[0, :], vecs[1, :], m)
            instance = __rounded(instance)
            permutation = test.calc_similarity(
                vecs[0, :] * np.random.randint(1, 10), vecs[1, :], m
            )
            permutation = __rounded(permutation)
            assert permutation == instance
            assert instance[0] >= value_range[0]
            assert instance[0] <= value_range[1]

        zero_vec = np.array([0, 0, 0])
        with pytest.warns():
            zero_case = test.calc_similarity(base_array, zero_vec, m)
        assert math.isnan(zero_case[0])
        assert math.isnan(zero_case[1])

        # Different length vectors are not alowed
        with pytest.raises(ValueError):
            test.calc_similarity(np.array([1, -5, 3]), np.array([3, 1, -3, -2]), m)
