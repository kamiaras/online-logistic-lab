"""Selected step/schedule checks; no production measurements."""
import json
import math

import numpy as np
import pytest

from experiments.computation_first_log_odds import ComputationFirstLogOdds
from experiments.computation_followup import LegacyStepCurvatureLogOdds
from experiments.curvature_persistent_log_odds import CurvaturePersistentLogOdds
from experiments.shared_tangent import PrescribedBudgetExceeded


def test_selected_practical_trajectory_matches_frozen_original_step_control():
    config = dict(lanes=16, transitions=1, history_block_size=3, particle_batch=4)
    selected = ComputationFirstLogOdds(3, 2., 1000, 731, config)
    control = LegacyStepCurvatureLogOdds(3, 2., 1000, 731, config)
    X = np.array([[.8, .2, -.1], [.1, -.3, .6], [-.7, .1, .3],
                  [.2, .2, -.3], [.6, .4, .1], [-.1, .5, -.2]])
    for x, y in zip(X, [1, 1, -1, 1, -1, 1]):
        assert selected.predict(x) == control.predict(x)
        for key in ["P", "inveta", "mu", "gamma", "H_warm", "N_prescribed",
                    "N_actual", "skipped", "raw_logit", "guarded_logit"]:
            assert selected._last_schedule[key] == control._last_schedule[key]
        for name in ["states", "anchors", "gradients", "curvature_summaries"]:
            np.testing.assert_array_equal(getattr(selected, name), getattr(control, name))
        selected.observe(x, y)
        control.observe(x, y)
    assert selected.metrics.counters == control.metrics.counters
    assert selected.metrics.counters["proximal_steps"] > 0


def test_preflight_replaces_uniform_step_bound_and_rechecks_budget_without_arrays(monkeypatch):
    cfg = dict(mode="prescribed", max_total_moves=10**12)
    old = CurvaturePersistentLogOdds.preflight(3, 2., 10, cfg)
    def forbid(*args, **kwargs):
        pytest.fail("Preflight allocated an ndarray")
    monkeypatch.setattr(np, "zeros", forbid)
    monkeypatch.setattr(np, "empty", forbid)
    pre = ComputationFirstLogOdds.preflight(3, 2., 10, cfg)
    p = pre["prescribed"]
    expected = 1+p["H_max"]+p["H_max"]*pre["d"]*p["kappa"]*pre["T"]/4
    assert p["N_max"] == math.ceil(expected)
    assert p["N_max"] > old["prescribed"]["N_max"]
    assert pre["uniform_step_coefficient"] == 6
    assert pre["estimated_total_moves"] == pre["T"]*p["m"]*p["N_max"]
    assert pre["estimated_state_bytes"] == old["estimated_state_bytes"]
    cap = old["estimated_total_moves"]
    assert CurvaturePersistentLogOdds.preflight(3, 2., 10, {**cfg, "max_total_moves": cap})["feasible"]
    denied = ComputationFirstLogOdds.preflight(3, 2., 10, {**cfg, "max_total_moves": cap})
    assert not denied["feasible"]
    assert denied["budget_reasons"] == [f"worst-case transitions: {pre['estimated_total_moves']} > {cap}"]
    json.dumps(pre, allow_nan=False)


def test_prescribed_small_run_uses_actual_original_step_mixing_count():
    model = ComputationFirstLogOdds(2, .4, 3, 919,
                                    dict(mode="prescribed", lanes=2, transitions=1))
    for x, y in [(np.array([1., 0.]), 1), (np.array([.8, .2]), 1),
                 (np.array([1., 0.]), -1)]:
        assert math.isfinite(model.predict(x))
        s = model._last_schedule
        assert s["P"] == 2*model.d*model.E
        expected = math.ceil(s["H_warm"]/(2*math.log1p(s["gamma"]/s["inveta"])))
        assert s["N_actual"] == s["N_prescribed"] == expected
        assert expected <= model._preflight["prescribed"]["N_max"]
        if not s["skipped"]:
            assert s["moves_actual"] == model.m*expected
        model.observe(x, y)
    assert model.metrics.counters["proximal_steps"] > model.m
    json.dumps(model.diagnostics(), allow_nan=False)


def test_practical_defaults_are_16_lanes_one_transition_with_storage_limits_retained():
    pre = ComputationFirstLogOdds.preflight(10, 100., 1000)
    assert pre["actual_lanes"] == 16 and pre["actual_max_transitions"] == 1
    assert pre["estimated_total_moves"] == 16000
    assert pre["guard_factor"] == 4
    assert pre["prior_pairing"] == "opposite_midpoints"
    model = ComputationFirstLogOdds(10, 100., 1000, 88)
    assert model.m == model.config["lanes"] == 16
    assert model.config["transitions"] == 1
    with pytest.raises(PrescribedBudgetExceeded):
        ComputationFirstLogOdds(10, 100., 1000, 88,
                                dict(max_state_bytes=pre["estimated_state_bytes"]-1))


@pytest.mark.parametrize("mode", ["practical", "prescribed"])
def test_zero_radius_has_zero_scheduled_and_actual_moves(mode):
    model = ComputationFirstLogOdds(2, 1., 3, 122, {"R": 0., "mode": mode})
    assert model._preflight["estimated_total_moves"] == 0
    assert model._preflight["actual_max_transitions"] == 0
    for _ in range(3):
        x = np.zeros(2)
        assert model.predict(x) == 0.
        assert model._last_schedule["N_actual"] == model._last_schedule["N_prescribed"] == 0
        model.observe(x, 1)
    assert model.states is None
    assert model.metrics.counters["proximal_steps"] == 0
    json.dumps(model.diagnostics(), allow_nan=False)
