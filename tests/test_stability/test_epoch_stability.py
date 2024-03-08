"""Tesing of the epoch_stability module.
"""

import pytest
import numpy as np
from stability.epoch_stability import QuasiStability, PeakStability, EpochStability

topo_maps = np.random.random((3, 2, 3)) - 0.5


def test_constructors():
    """Testing constructor input"""
    # First abstracts parent class constructor
    with pytest.raises(TypeError):
        base_constructor = PeakStability("ba")
    with pytest.raises(ValueError):
        base_constructor = PeakStability(0)
    with pytest.raises(ValueError):
        base_constructor = PeakStability(-1)

    # Then the specifics for Quasi
    with pytest.raises(TypeError):
        quasi_constructor = QuasiStability(3, switching_frq_thresholds="ba")
    with pytest.raises(TypeError):
        quasi_constructor = QuasiStability(3, switching_frq_thresholds=np.array(["ba"]))
    with pytest.raises(ValueError):
        quasi_constructor = QuasiStability(
            3, switching_frq_thresholds=np.array([[1], [2]])
        )


def test_quasi_stability():
    """Testing QuasiStability class
    """
    _test_epoch_stabilit_functionality(QuasiStability)
    _test_epoch_stabilit_functionality(QuasiStability, stability_method="cosine")
    _test_epoch_stabilit_functionality(
        QuasiStability, switching_frq_thresholds=np.array([1.5, 1.7])
    )

    # Test switching_thrs
    with pytest.raises(TypeError):
        _test_epoch_stabilit_functionality(
            QuasiStability, switching_frq_thresholds="ba"
        )
    with pytest.raises(ValueError):
        _test_epoch_stabilit_functionality(
            QuasiStability, switching_frq_thresholds=np.array([[1.3, 1.2], [0.4, 0.6]])
        )
    with pytest.raises(ValueError):
        _test_epoch_stabilit_functionality(
            QuasiStability, switching_frq_thresholds=np.array([1.3, 2.2])
        )
    with pytest.raises(ValueError):
        _test_epoch_stabilit_functionality(
            QuasiStability, switching_frq_thresholds=np.array([1.3, -1.2])
        )
    with pytest.raises(ValueError):
        _test_epoch_stabilit_functionality(
            QuasiStability,
            stability_method="cosine",
            switching_frq_thresholds=np.array([0.9, -1.2]),
        )
    with pytest.raises(ValueError):
        _test_epoch_stabilit_functionality(
            QuasiStability,
            stability_method="cosine",
            switching_frq_thresholds=np.array([0.9, 1.2]),
        )

def test_peak_stability():
    """Testing PeakStability class
    """
    _test_epoch_stabilit_functionality(PeakStability)
    _test_epoch_stabilit_functionality(PeakStability, stability_method="cosine")


def _test_epoch_stabilit_functionality(clas_inst: EpochStability, **kwargs):
    """Helper test fir EpochStability descendants
    """
    c = clas_inst(4, **kwargs)
    for m in range(3):
        c.add_stability_stats(topo_maps[:, :, m])
        if m == 2:
            c.add_stability_stats(np.array([[-1, 4], [-1, 3], [1, -2]]))
            assert c.obj_filled
            with pytest.raises(ValueError):
                c.add_stability_stats(topo_maps[:, :, m])
            assert c.mean_abs_stab[-1] != 0
            assert c.mean_stab[-1] != 0
            assert c._instances == c._current_instance
