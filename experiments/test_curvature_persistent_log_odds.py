"""Curvature-summary invariants and unchanged exact acceptance events."""
import json
import math

import numpy as np
import pytest
from scipy.special import expit

import experiments.curvature_persistent_log_odds as module
from experiments.curvature_persistent_log_odds import (
    CurvaturePersistentLogOdds, CURVATURE_BIN_EDGES,
)
from experiments.batched_persistent_log_odds import BatchedPersistentLogOdds
from experiments.proximal import logistic_bregman


def _model(rows=None, d=1, B=100., **config):
    model = CurvaturePersistentLogOdds(d, B, 40, 9901,
                                       {"lanes": 4, "transitions": 2,
                                        "particle_batch": 4, "history_block_size": 3, **config})
    for row in ([] if rows is None else rows):
        model._append_feature(np.asarray(row, float))
    return model


def _direct_summaries(X, anchors):
    result = np.zeros((len(anchors), 8, X.shape[1]))
    for lane, anchor in enumerate(anchors):
        for x in X:
            score = abs(float(x@anchor))
            # Explicit inequalities independently check the searchsorted bins.
            bucket = max(j for j, edge in enumerate(CURVATURE_BIN_EDGES) if score >= edge)
            result[lane, bucket] += np.abs(x).sum()*np.abs(x)
    return result


def test_summary_storage_is_additional_eight_diagonals_and_counts_are_unchanged():
    cfg = {"lanes": 16, "transitions": 4, "particle_batch": 64}
    plain = BatchedPersistentLogOdds.preflight(100, 1000., 500, cfg)
    squeeze = CurvaturePersistentLogOdds.preflight(100, 1000., 500, cfg)
    assert squeeze["prescribed"] == plain["prescribed"]
    assert squeeze["guard_factor"] == plain["guard_factor"] == 4
    extra = 8*16*100*8
    assert squeeze["extra_curvature_summary_bytes"] == extra
    assert squeeze["estimated_state_bytes"] == plain["estimated_state_bytes"]+extra
    assert not CurvaturePersistentLogOdds.preflight(
        100, 1000., 500, {**cfg, "max_state_bytes": plain["estimated_state_bytes"]})["feasible"]
    json.dumps(squeeze, allow_nan=False)


def test_zero_anchor_increment_initializes_bin_zero_without_a_history_pass():
    X = np.array([[.7, -.2], [.1, .8], [-.4, .3]])
    model = _model(X, d=2)
    total = (np.abs(X).sum(axis=1)[:, None]*np.abs(X)).sum(axis=0)
    for lane in range(4):
        np.testing.assert_allclose(model.curvature_summaries[lane, 0], total)
        np.testing.assert_array_equal(model.curvature_summaries[lane, 1:], np.zeros((7, 2)))
    assert model.states is None
    assert model.metrics.counters["summary_rebuilds"] == 0
    assert model.metrics.counters["summary_append_lane_rows"] == 12


def test_one_pass_summary_rebuild_and_increment_match_direct_bin_definitions():
    X = np.array([[0.], [.005], [.015], [.03], [.06], [.12], [.24], [.48], [.96]])
    model = _model(X)
    model.anchors[:, 0] = [100., -100., 50., -50.]
    model._refresh_lanes(np.arange(4))
    expected = _direct_summaries(X, model.anchors)
    np.testing.assert_allclose(model.curvature_summaries, expected, atol=2e-15)
    assert np.all(expected[0, :, 0] > 0.)
    np.testing.assert_allclose(model.gradients, (expit(X@model.anchors.T)-.5).T@X, atol=2e-15)
    assert model.metrics.counters["summary_rebuild_history_rows"] == len(X)
    assert model.metrics.counters["summary_rebuild_lane_rows"] == 4*len(X)
    incoming = np.array([.08])
    model._append_feature(incoming)
    X = np.vstack([X, incoming])
    np.testing.assert_allclose(model.curvature_summaries,
                               _direct_summaries(X, model.anchors), atol=2e-15)
    # Rebuilding one rejected lane must not change the other summaries/gradients.
    untouched = model.curvature_summaries[[0, 1, 3]].copy()
    model.anchors[2, 0] = -140.
    model._refresh_lanes(np.array([2]))
    np.testing.assert_array_equal(model.curvature_summaries[[0, 1, 3]], untouched)
    np.testing.assert_allclose(model.curvature_summaries,
                               _direct_summaries(X, model.anchors), atol=2e-15)


def test_majorant_dominates_exact_residual_and_is_below_global_quadratic():
    rng = np.random.default_rng(831)
    X = rng.normal(size=(17, 3))
    X /= np.maximum(1., np.linalg.norm(X, axis=1))[:, None]
    model = _model(X, d=3)
    model.anchors[:] = [[100., 30., -70.], [-90., 55., 60.], [.1, -.3, .2], [120., -80., 90.]]
    model._refresh_lanes(np.arange(4))
    global_weight = (np.abs(X).sum(axis=1)[:, None]*np.abs(X)).sum(axis=0)
    for scale in [0., .01, .5, 2., 20., 180.]:
        proposals = np.clip(model.anchors+scale*rng.normal(size=(4, 3)), -200., 200.)
        C = model._curvature_majorant(np.arange(4), proposals)
        exact = logistic_bregman(X@model.anchors.T, X@proposals.T).sum(axis=0)
        squared = (proposals-model.anchors)**2
        global_bound = squared@global_weight/8.
        assert np.all(exact <= C+1e-10*(1+C))
        assert np.all(C <= global_bound+1e-10*(1+global_bound))
    # Every bin falls back to curvature 1/4 when the displacement radius>=64.
    proposals = -model.anchors
    C = model._curvature_majorant(np.arange(4), proposals)
    radii = np.linalg.norm(proposals-model.anchors, axis=1)
    big = radii >= 64.
    np.testing.assert_allclose(C[big], ((proposals-model.anchors)**2@global_weight/8.)[big])


def test_certified_acceptance_reads_no_historical_rows(monkeypatch):
    X = np.array([[1.], [.9], [.8]])
    model = _model(X)
    model.anchors[:, 0] = [100., 110., -100., -110.]
    model._refresh_lanes(np.arange(4))
    proposals = model.anchors+.01
    C = model._curvature_majorant(np.arange(4), proposals)
    assert np.all(C < 1e-20)
    def forbidden_history():
        pytest.fail("certified summary acceptance accessed history")
    monkeypatch.setattr(model.history, "batches", forbidden_history)
    accepted = model._residual_test_batch(np.arange(4), proposals, thresholds=np.ones(4))
    np.testing.assert_array_equal(accepted, np.ones(4, bool))
    assert model.metrics.counters["summary_certified_accepts"] == 4
    assert model.metrics.counters["history_rows_avoided_by_squeeze"] == 4*len(X)
    assert model.metrics.counters["full_test_rows"] == 0


def test_unresolved_lanes_continue_with_same_threshold_and_no_fresh_random_coin(monkeypatch):
    model = _model([[1.], [.7]])
    proposals = np.array([[1.], [2.], [1.5], [.1]])
    thresholds = np.array([.4, .7, .9, 1.1])
    monkeypatch.setattr(model, "_curvature_majorant", lambda lanes, z: np.array([.2, 2., 3., .1]))
    seen = []
    def full(self, lanes, proposed, thresholds=None):
        seen.append((lanes.copy(), proposed.copy(), thresholds.copy()))
        return np.array([True, False])
    monkeypatch.setattr(BatchedPersistentLogOdds, "_residual_test_batch", full)
    monkeypatch.setattr(module, "open_uniform", lambda *args: pytest.fail("continuation drew a fresh coin"))
    accepted = model._residual_test_batch(np.arange(4), proposals, thresholds)
    np.testing.assert_array_equal(accepted, [True, True, False, True])
    np.testing.assert_array_equal(seen[0][0], [1, 2])
    np.testing.assert_array_equal(seen[0][2], thresholds[[1, 2]])


def test_squeezed_and_full_acceptance_events_agree_at_same_thresholds():
    model = _model([[.8, -.2], [.1, .7], [-.5, .4]], d=2)
    model.anchors[:] = [[100., 20.], [-80., 50.], [.2, -.1], [40., -100.]]
    model._refresh_lanes(np.arange(4))
    proposals = model.anchors+np.array([[.01, -.02], [2., -1.], [-.7, .3], [140., 130.]])
    for thresholds in [np.full(4, .01), np.full(4, 1.), np.full(4, 1000.)]:
        actual = model._residual_test_batch(np.arange(4), proposals, thresholds)
        full = BatchedPersistentLogOdds._residual_test_batch(model, np.arange(4), proposals, thresholds)
        np.testing.assert_array_equal(actual, full)


def test_complete_small_curvature_sampler_keeps_outer_counts_and_finite_guard():
    model = _model(d=1, B=2., lanes=4, particle_batch=2)
    for y in [1, 1, -1, 1]:
        x = np.ones(1)
        score = model.predict(x)
        assert math.isfinite(score)
        assert abs(score) <= math.log(4*model.history.n*(model.history.n+1))+1e-14
        model.observe(x, y)
    assert model.metrics.counters["summary_queries"] == model.metrics.counters["rqo_proposals"]
    assert model.metrics.counters["rqo_accepts"] == model.metrics.counters["proximal_steps"]
    assert model.metrics.counters["summary_certified_accepts"] > 0
    assert model.metrics.counters["poisson_tests"] == 0
    json.dumps(model.diagnostics(), allow_nan=False)
