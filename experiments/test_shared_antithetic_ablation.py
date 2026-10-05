"""Mechanism checks for the initialization-only controlled extension."""
import numpy as np
import pytest
from experiments.shared_antithetic_ablation import AntitheticSharedAblation


def test_opposite_midpoints_receive_exactly_reflected_prior_vectors():
    a = AntitheticSharedAblation(3, 100., 1000, 719,
        {"layout": "lanes", "persistence": "warm", "tilts": 16,
         "burn_in": 4, "transitions": 4, "acceptance": "full"})
    prior = np.stack([a._prior() for _ in range(16)])
    np.testing.assert_array_equal(prior, -prior[::-1])
    np.testing.assert_array_equal(prior[:8]+prior[:7:-1], np.zeros((8, 3)))
    assert a.metrics.counters["independent_prior_vectors"] == 8
    assert a.metrics.counters["antithetic_prior_vectors"] == 8
    assert a.metrics.counters["prior_initializations"] == 16
    assert a._paired_prior_bank == []


def test_pairing_rejects_incompatible_control_layouts():
    for cfg in ({"layout": "sweep", "persistence": "warm", "tilts": 16},
                {"layout": "lanes", "persistence": "fresh", "tilts": 16},
                {"layout": "lanes", "persistence": "warm", "tilts": 15}):
        with pytest.raises(ValueError):
            AntitheticSharedAblation(3, 100., 1000, 719, cfg)
