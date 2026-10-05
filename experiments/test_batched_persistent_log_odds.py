"""Focused cache, acceptance, density and outer-schedule batching checks."""
import math
import json

import numpy as np
import pytest
from scipy.special import expit

import experiments.batched_persistent_log_odds as module
from experiments.batched_persistent_log_odds import BatchedPersistentLogOdds
from experiments.persistent_log_odds import PersistentLogOdds


def _model(rows=None, d=2, **config):
    model = BatchedPersistentLogOdds(d, 2., 20, 405,
                                     {"lanes": 4, "transitions": 2,
                                      "particle_batch": 4, "history_block_size": 2, **config})
    for x in ([] if rows is None else rows):
        model._append_feature(np.asarray(x, float))
    return model


def _h(z):
    return np.logaddexp(np.asarray(z)/2., -np.asarray(z)/2.)


def test_batched_preflight_preserves_schedule_and_charges_two_lane_arrays():
    cfg = {"lanes": 16, "transitions": 4, "history_block_size": 128}
    old = PersistentLogOdds.preflight(100, 1000., 500, cfg)
    new = BatchedPersistentLogOdds.preflight(100, 1000., 500, {**cfg, "particle_batch": 64})
    assert new["prescribed"] == old["prescribed"]
    assert new["guard_factor"] == old["guard_factor"] == 4
    assert new["extra_lane_cache_bytes"] == 2*16*100*8
    assert new["estimated_state_bytes"] == old["estimated_state_bytes"]+2*16*100*8
    json.dumps(new, allow_nan=False)
    limit = old["estimated_state_bytes"]
    assert not BatchedPersistentLogOdds.preflight(100, 1000., 500,
                                                  {**cfg, "max_state_bytes": limit})["feasible"]
    for extra in ({"particle_batch": 65}, {"history_block_size": 129},
                  {"particle_batch": 2, "lane_batch_size": 4}):
        with pytest.raises(ValueError):
            BatchedPersistentLogOdds.preflight(2, 1., 5, extra)


def test_incremental_lane_gradients_equal_independent_full_history_formula():
    rows = np.array([[.8, -.1], [.2, .7]])
    model = _model(rows)
    model.anchors[:] = [[.1, -.3], [-2., 3.], [3., 2.], [-1., -.8]]
    model._refresh_lanes(np.arange(4))
    before = model.anchors.copy()
    new = np.array([-.3, .6])
    model._append_feature(new)
    X = np.vstack([rows, new])
    expected = (expit(X@before.T)-.5).T@X
    np.testing.assert_allclose(model.gradients, expected, atol=3e-15)
    np.testing.assert_array_equal(model.anchors, before)
    assert model.metrics.counters["cache_append_rows"] == 3*model.m


def test_batched_acceptance_events_match_independent_exact_residual_thresholds():
    rows = np.array([[.8, -.1], [.2, .7], [-.3, .6]])
    model = _model(rows)
    anchors = np.array([[.1, -.3], [-2., 3.], [3., 2.], [-1., -.8]])
    proposed = np.array([[-.4, .8], [.1, .3], [-.8, 1.2], [-1.1, -.7]])
    model.anchors[:] = anchors
    old, new = rows@anchors.T, rows@proposed.T
    exact = np.sum(_h(new)-_h(old)-(expit(old)-.5)*(new-old), axis=0)
    thresholds = exact+np.array([1e-5, -1e-5, 1e-5, -1e-5])
    accepted = model._residual_test_batch(np.arange(4), proposed, thresholds)
    np.testing.assert_array_equal(accepted, [True, False, True, False])
    assert model.metrics.counters["full_tests"] == 4
    assert model.metrics.counters["poisson_tests"] == 0


def test_proposal_times_acceptance_has_same_target_density_for_every_lane():
    rows = np.array([[.8, -.1], [.2, .7], [-.3, .6]])
    model = _model(rows)
    model.anchors[:] = [[.1, -.3], [-2., 3.], [3., 2.], [-1., -.8]]
    model._refresh_lanes(np.arange(4))
    mu = model._schedule()["mu"]
    r = np.array([[100., -100.], [.8, .3], [-1., 2.], [3., -.7]])
    centers = (r-model.gradients)/mu
    assert np.max(np.abs(centers[0])) > model.bound
    for j in range(4):
        z = np.array([[-1.2, .7], [.5, -.8], [1.7, 1.4]])
        old = rows@model.anchors[j]
        new = rows@z.T
        residual = (_h(new)-_h(old)[:, None]
                    -(expit(old)-.5)[:, None]*(new-old[:, None])).sum(axis=0)
        proposal_log = -.5*mu*np.sum((z-centers[j])**2, axis=1)
        target_log = -.5*mu*np.sum(z*z, axis=1)+z@r[j]-_h(new).sum(axis=0)
        difference = proposal_log-residual-target_log
        np.testing.assert_allclose(difference, np.full(3, difference[0]), rtol=0., atol=2e-12)


def test_only_rejected_lanes_repair_and_their_auxiliary_vector_stays_fixed(monkeypatch):
    rows = np.array([[.8, -.1], [.2, .7]])
    model = _model(rows)
    model.anchors[:] = [[.1, -.3], [-.2, .3], [.3, .2], [-.1, -.8]]
    model._refresh_lanes(np.arange(4))
    old_anchors, old_gradients = model.anchors.copy(), model.gradients.copy()
    lanes, r = np.array([0, 2]), np.array([[.2, -.1], [100., -100.]])
    schedule = model._schedule()
    initial_center = (r-model.gradients[lanes])/schedule["mu"]
    centers, tested = [], []
    def normal(center, *args, **kwargs):
        centers.append(center.copy())
        return np.full_like(center, .4)
    def acceptance(active, proposed, thresholds=None):
        tested.append(active.copy())
        return np.array([True, False]) if len(tested) == 1 else np.array([True])
    monkeypatch.setattr(module, "box_normal", normal)
    monkeypatch.setattr(model, "_residual_test_batch", acceptance)
    result = model._conditional_batch(r, lanes, schedule)
    np.testing.assert_array_equal(tested[0], [0, 2])
    np.testing.assert_array_equal(tested[1], [2])
    np.testing.assert_allclose(centers[0], initial_center)
    np.testing.assert_array_equal(model.anchors[[0, 1, 3]], old_anchors[[0, 1, 3]])
    np.testing.assert_array_equal(model.gradients[[0, 1, 3]], old_gradients[[0, 1, 3]])
    expected_anchor = np.clip(initial_center[1], -model.bound, model.bound)
    expected_gradient = rows.T@(expit(rows@expected_anchor)-.5)
    np.testing.assert_allclose(model.anchors[2], expected_anchor)
    np.testing.assert_allclose(model.gradients[2], expected_gradient, atol=2e-15)
    np.testing.assert_allclose(centers[1][0], (r[1]-expected_gradient)/schedule["mu"])
    np.testing.assert_array_equal(result, np.full((2, 2), .4))
    assert model.metrics.counters["cache_misses"] == 1
    assert model.metrics.counters["rejection_repairs"] == 1
    assert model.metrics.counters["cached_first_accepts"] == 1


def test_batched_move_counts_fixed_midpoints_and_individual_state_handoff(monkeypatch):
    model = _model(d=1, lanes=6, transitions=3, particle_batch=2)
    x = np.ones(1)
    model.predict(x)
    model.observe(x, 1)
    model._initialize_lanes()
    initial = model.states.copy()
    calls = []
    monkeypatch.setattr(module, "normal_standard", lambda shape, *args: np.zeros(shape))
    def move(r, lanes, schedule):
        calls.append((lanes.copy(), r.copy()))
        return r/100.
    monkeypatch.setattr(model, "_conditional_batch", move)
    model.predict(x)
    inveta = model._last_schedule["inveta"]
    expected = initial.copy()
    for step in range(3):
        for batch in range(3):
            lanes, actual_r = calls[step*3+batch]
            tilts = -.5+(lanes+.5)/6
            expected_r = inveta*expected[lanes]+.5+tilts[:, None]
            np.testing.assert_allclose(actual_r, expected_r)
            expected[lanes] = expected_r/100.
    np.testing.assert_allclose(model.states, expected)
    assert model._last_schedule["moves_actual"] == 18
    assert model.metrics.counters["proximal_steps"] == 18
    assert model.metrics.counters["prior_cloud_initializations"] == 1


def test_outer_schedule_skip_gap_and_guard_match_frozen_reference(monkeypatch):
    cfg = {"lanes": 4, "transitions": 2, "history_block_size": 2}
    sequential = PersistentLogOdds(1, 2., 6, 1, cfg)
    batched = BatchedPersistentLogOdds(1, 2., 6, 1, {**cfg, "particle_batch": 2})
    monkeypatch.setattr(sequential, "_conditional_draw", lambda r, s: np.zeros(1))
    monkeypatch.setattr(batched, "_conditional_batch", lambda r, lanes, s: np.zeros_like(r))
    for x, y in [(1., 1), (1., -1), (.5, 1), (1., 1), (0., -1)]:
        row = np.array([x])
        assert sequential.predict(row) == batched.predict(row)
        a, b = sequential._last_schedule, batched._last_schedule
        for field in ("reference_round", "reference_gamma", "gap_norm", "H_warm",
                      "N_prescribed", "N_actual", "skipped", "m_actual", "guard", "moves_actual"):
            assert a[field] == b[field]
        sequential.observe(row, y)
        batched.observe(row, y)
    assert batched.diagnostics()["tangent_cache"] == "per_lane"
    assert batched.state_bytes()-sequential.state_bytes() == 2*batched.m*batched.d*8


def test_small_complete_batched_returns_and_guarded_prediction():
    model = _model(d=1, lanes=4, transitions=2, particle_batch=2)
    for y in [1, 1, -1, 1]:
        x = np.ones(1)
        score = model.predict(x)
        assert math.isfinite(score)
        assert abs(score) <= math.log(4*model.history.n*(model.history.n+1))+1e-14
        model.observe(x, y)
    assert model.metrics.counters["conditional_batches"] > 0
    assert model.metrics.counters["rqo_accepts"] == model.metrics.counters["proximal_steps"]
    assert model.metrics.counters["prior_cloud_initializations"] == 1
    assert model.metrics.counters["poisson_tests"] == 0
    json.dumps(model.diagnostics(), allow_nan=False)
