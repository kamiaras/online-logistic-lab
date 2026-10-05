"""Focused distribution and lookup tests for the supplied shared-tangent loop."""
import math

import numpy as np
import pytest
from scipy.stats import truncnorm

from experiments.common import Metrics
from experiments.shared_tangent_primitives import (
    WeightedHistory, box_normal, normal_interval, normal_standard, open_uniform,
)


@pytest.mark.parametrize("alpha,beta", [(-2., 2.), (0., .2), (2., 8.),
                                      (10., 10.03), (-12., -9.), (-.2, .3)])
def test_exact_scalar_envelope_moments(alpha, beta):
    n = 24000
    metrics = Metrics()
    result = normal_interval(np.full(n, alpha), np.full(n, beta),
                             np.random.default_rng(88276), metrics)
    mean, variance = truncnorm.stats(alpha, beta, moments="mv")
    assert np.all((alpha <= result) & (result <= beta))
    assert abs(result.mean() - mean) < 8 * math.sqrt(variance / n) + 1e-10
    assert abs(result.var() - variance) < 10 * variance * math.sqrt(2 / n) + 1e-10
    assert metrics.counters["tn_proposals"] / n < 3
    assert metrics.counters["tn_rejections"] == metrics.counters["tn_proposals"] - n
    assert metrics.counters["uniform_draws"] == 3 * metrics.counters["tn_proposals"]


def test_unrestricted_normal_uses_same_three_uniform_trials():
    metrics = Metrics()
    values = normal_standard((400, 100), np.random.default_rng(911), metrics)
    assert abs(values.mean()) < .035
    assert abs(values.var() - 1) < .05
    assert metrics.counters["uniform_draws"] == 3 * metrics.counters["tn_proposals"]
    assert metrics.counters["tn_proposals"] / values.size < 3


def test_no_gaussian_rng_primitive_needed_and_zero_is_redrawn():
    class UniformOnly:
        def __init__(self):
            self.rng = np.random.default_rng(555)
            self.first = True

        def random(self, size=None):
            values = self.rng.random(size)
            if self.first:
                self.first = False
                if size is None:
                    return 0.
                values.flat[0] = 0.
            return values

    metrics = Metrics()
    scalar = open_uniform(UniformOnly(), metrics=metrics)
    assert 0 < scalar < 1
    assert metrics.counters["uniform_zero_redraws"] == 1
    values = normal_standard(800, UniformOnly(), metrics)
    assert np.all(np.isfinite(values))
    assert metrics.counters["uniform_zero_redraws"] == 2


def test_box_offsets_avoid_large_center_cancellation():
    metrics = Metrics()
    rng = np.random.default_rng(337)
    centers = np.array([-1e20, 1e20, 0., 1.99])
    values = box_normal(centers, 1., 2., rng, metrics)
    assert np.all(np.isfinite(values)) and np.all(np.abs(values) <= 2.)
    assert metrics.counters["normal_offset_rounding_stalls"] == 2
    # Stable physical-offset generation also retains an interval much smaller
    # than the distant center's floating-point spacing.
    values = box_normal(np.full(16000, 1e8), 1., 2e-8, rng)
    rescaled = (2e-8 - values) * 1e8
    # Exponential(rate 1) truncated at 4 is the accurate limiting tail law.
    target_mean = 1 - 4 / np.expm1(4.)
    assert abs(rescaled.mean() - target_mean) < .025
    assert np.all((-2e-8 <= values) & (values <= 2e-8))


def test_narrow_interval_broadcast_and_scalar_output():
    rng = np.random.default_rng(79)
    a, b = 30., 30. + 1e-9
    values = normal_interval(np.full(20000, a), b, rng)
    scaled = (values - a) / (b - a)
    assert abs(scaled.mean() - .5) < .015
    assert abs(scaled.var() - 1 / 12) < .005
    assert isinstance(normal_interval(-2., 2., rng), float)
    assert isinstance(normal_standard(None, rng), float)
    assert isinstance(box_normal(0., 1., 2., rng), float)


@pytest.mark.parametrize("alpha,beta", [(1., 1.), (2., 1.), (-np.inf, 1.),
                                      (0., np.inf), (np.nan, 2.)])
def test_invalid_interval_is_rejected(alpha, beta):
    with pytest.raises(ValueError):
        normal_interval(alpha, beta, np.random.default_rng(9))


def test_weighted_history_matches_direct_categorical_across_blocks():
    history = WeightedHistory(3, block_size=2)
    raw = np.array([[0, 0, 0], [1, 0, 2], [0, 4, 0], [-3, 0, 1], [0, 0, 0]], float)
    for row in raw:
        history.append(row)
    increments = np.abs(raw) * np.abs(raw).sum(axis=1)[:, None]
    prefixes = np.cumsum(increments, axis=0)
    assert history.n == len(raw)
    assert np.array_equal(history.W, prefixes[-1])
    assert np.array_equal(np.concatenate(list(history.batches())), raw)
    assert history.nbytes == 8 * (3 + 3 * 2 * (2 * 3 + 1))
    for i in range(len(raw)):
        assert np.array_equal(history.row(i), raw[i])
        assert history.row_l1(i) == abs(raw[i]).sum()
    for j in range(3):
        thresholds = np.unique(np.r_[0, prefixes[:, j],
                                     np.linspace(0, history.W[j], 101, endpoint=False)])
        for threshold in thresholds[thresholds < history.W[j]]:
            chosen = history.sample_row(j, threshold)
            assert chosen == np.searchsorted(prefixes[:, j], threshold, side="right")
            assert increments[chosen, j] > 0


def test_failed_append_preserves_valid_history():
    metrics = Metrics()
    history = WeightedHistory(1, block_size=2, metrics=metrics)
    history.append(np.array([1e8]))
    before = history.W.copy()
    with pytest.raises(FloatingPointError, match="invisible"):
        history.append(np.array([1e-5]))
    assert history.n == 1 and np.array_equal(history.W, before)
    assert np.array_equal(history.row(0), [1e8])
    assert metrics.counters["numerical_failures"] == 1
    history.append(np.array([1e8]))
    assert history.n == 2 and np.array_equal(history.W, before * 2)


def test_empty_zero_weight_and_invalid_thresholds_are_explicit():
    history = WeightedHistory(2)
    assert history.n == 0 and not list(history.batches())
    with pytest.raises(ValueError):
        history.sample_row(0, 0.)
    history.append([0., 2.])
    for threshold in (-1., 4., np.inf, np.nan):
        with pytest.raises(ValueError):
            history.sample_row(1, threshold)
    assert history.sample_row(1, 0) == 0
    with pytest.raises(IndexError):
        history.row(1)


def test_new_block_allocation_failure_is_transactional(monkeypatch):
    history = WeightedHistory(2, block_size=1)
    history.append([1., 2.])
    before = history.W.copy()
    original_empty = np.empty
    allocations = 0

    def fail_second(*args, **kwargs):
        nonlocal allocations
        allocations += 1
        if allocations == 2:
            raise MemoryError("synthetic second-array allocation failure")
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(np, "empty", fail_second)
    with pytest.raises(MemoryError):
        history.append([2., 1.])
    assert history.n == 1 and len(history.blocks) == 1
    assert np.array_equal(history.W, before)
