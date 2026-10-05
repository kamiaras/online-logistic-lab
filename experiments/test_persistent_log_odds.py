"""Schedule, persistence and exact-return identity checks for the new reference.

Numerical checks do not certify float64 arithmetic or fixed-cap posterior mixing.
Run separately after the controlled experiment workers have finished.
"""
import json
import math

import numpy as np
import pytest
from scipy.special import expit

import experiments.persistent_log_odds as module
import experiments.shared_tangent as shared_core
from experiments.persistent_log_odds import FeatureHistory, PersistentLogOdds
from experiments.shared_tangent import PrescribedBudgetExceeded


def _history_model(rows, *, B=1., anchor=None, block_size=2):
    rows = np.asarray(rows, float)
    model = PersistentLogOdds(rows.shape[1], B, 30, 142857,
                              {"lanes": 2, "transitions": 1,
                               "history_block_size": block_size})
    for row in rows:
        model._append_feature(row)
    if anchor is not None:
        model.v[:] = anchor
        model._refresh_cache()
    return model, rows


def _h(z):
    return np.logaddexp(np.asarray(z)/2., -np.asarray(z)/2.)


def test_population_has_no_old_score_range_floor_and_warm_H_is_explicit():
    d, B, R, T = 1000, .1, 1., 2
    pre = PersistentLogOdds.preflight(d, B, T, {"R": R})
    L, kappa = 2*B*R*math.sqrt(d), (B*R)**2
    a = expit(L)*expit(-L)
    Gamma = d*math.log1p(a*kappa*T/d)/(8*a)
    assert pre["prescribed"]["m"] == 2*max(1, math.ceil(2*Gamma)) == 2
    assert pre["prescribed"]["m"] < L
    assert pre["guard_factor"] == 4
    epsilon = 1/(4*(1+L)*T*(T+1))
    assert pre["prescribed"]["epsilon"] == pytest.approx(epsilon)
    H = math.log((1+kappa*T*T/2)/(2*epsilon**2))
    assert pre["prescribed"]["H_max"] == pytest.approx(H)
    json.dumps(pre, allow_nan=False)


def test_log_domain_large_radius_preflight_is_finite_and_allocation_free(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preflight allocated a state array")
    monkeypatch.setattr(np, "empty", forbidden)
    pre = PersistentLogOdds.preflight(1000, 1e6, 10000)
    assert pre["prescribed"]["m"] == pytest.approx(1e12*10000/2, rel=2e-13)
    assert pre["prescribed"]["global_curvature_underflow"]
    assert pre["actual_lanes"] == 256
    assert pre["actual_max_transitions"] == 1
    json.dumps(pre, allow_nan=False)
    with pytest.raises(PrescribedBudgetExceeded):
        PersistentLogOdds(1000, 1e6, 10000, 1, {"mode": "prescribed"})


def test_preflight_memory_uses_raw_features_and_lane_state():
    d, T, block = 3, 11, 4
    pre = PersistentLogOdds.preflight(d, 10., T,
                                      {"lanes": 6, "history_block_size": block})
    assert pre["estimated_state_bytes"] == 8*((12+6)*d+32*d+256)
    refused = PersistentLogOdds.preflight(d, 10., T,
                                          {"lanes": 6, "max_state_bytes": 1})
    assert not refused["feasible"]
    for config in ({"lanes": 0}, {"lanes": 3}, {"transitions": True}, {"tilts": 16}, {"burn_in": 2}):
        with pytest.raises(ValueError):
            PersistentLogOdds.preflight(1, 1., 3, config)


def test_raw_history_commits_only_valid_rows_and_counts_allocated_blocks(monkeypatch):
    history = FeatureHistory(2, 2)
    history.append([1., 2.])
    history.append([3., 4.])
    with pytest.raises(ValueError):
        history.append([np.nan, 0.])
    assert history.n == 2
    def fail_allocation(*args, **kwargs):
        raise MemoryError("simulated")
    monkeypatch.setattr(np, "empty", fail_allocation)
    with pytest.raises(MemoryError):
        history.append([5., 6.])
    assert history.n == 2 and len(history.blocks) == 1
    np.testing.assert_array_equal(np.concatenate(list(history.batches())), [[1., 2.], [3., 4.]])
    assert history.nbytes == 2*2*8
    assert not hasattr(history, "W")


def test_lanes_initialize_once_and_reuse_own_state_at_fixed_midpoints(monkeypatch):
    model = PersistentLogOdds(1, 5., 5, 321, {"lanes": 4, "transitions": 2})
    x = np.ones(1)
    assert model.predict(x) == 0.
    assert model.states is None
    model.observe(x, 1)
    prior_values = iter([-.4, .2])
    priors, recorded = [], []
    def prior(alpha, beta, *args, **kwargs):
        value = next(prior_values)
        priors.append(value)
        return np.array([value])
    monkeypatch.setattr(module, "normal_interval", prior)
    monkeypatch.setattr(module, "normal_standard", lambda *args, **kwargs: np.zeros(1))
    def move(r, schedule):
        recorded.append(r.copy())
        # Deterministic stand-in makes the state handoff independently checkable.
        return r/100.
    monkeypatch.setattr(model, "_conditional_draw", move)
    model.predict(x)
    assert len(priors) == model.m//2 == 2
    inveta = model._last_schedule["inveta"]
    tilts = -.5+(np.arange(4)+.5)/4
    expected_states = []
    paired_priors = priors+[-p for p in reversed(priors)]
    for j, prior_value in enumerate(paired_priors):
        first_r = inveta*model.B*prior_value+.5+tilts[j]
        second_r = inveta*first_r/100.+.5+tilts[j]
        np.testing.assert_allclose(recorded[2*j:2*j+2], [[first_r], [second_r]])
        expected_states.append(second_r/100.)
    np.testing.assert_allclose(model.states[:, 0], expected_states)
    previous = model.states.copy()
    model.observe(x, 1)
    recorded.clear()
    model.predict(x)
    assert len(priors) == 2
    inveta = model._last_schedule["inveta"]
    for j in range(4):
        assert recorded[2*j][0] == pytest.approx(inveta*previous[j, 0]+1.+tilts[j])
    assert model.metrics.counters["prior_cloud_initializations"] == 1
    assert model.metrics.counters["proximal_steps"] == 16


def test_antithetic_prior_cancels_exactly_and_counts_draws_separately():
    model = PersistentLogOdds(3, 1e6, 4, 27, {"lanes": 8})
    model._initialize_lanes()
    assert model.states.shape == (8, 3)
    for j in range(4):
        np.testing.assert_array_equal(model.states[j]+model.states[7-j], np.zeros(3))
    assert model.metrics.counters["prior_initializations"] == 4
    assert model.metrics.counters["prior_coordinates"] == 12
    assert model.metrics.counters["initialized_lanes"] == 8
    assert model.diagnostics()["within_pair_independence_assumed"] is False


def test_gap_uses_old_reference_across_skips_and_resets_to_current_norm(monkeypatch):
    model = PersistentLogOdds(1, 2., 8, 42, {"lanes": 2, "transitions": 1})
    monkeypatch.setattr(model, "_conditional_draw", lambda r, schedule: np.zeros(1))
    x = np.ones(1)
    model.predict(x)
    assert model._last_schedule["reference_round"] == 0
    assert model.gap_norm == 1.
    model.observe(x, 1)
    model.predict(x)
    first = model._last_schedule
    assert not first["skipped"]
    assert first["gap_norm"] == 2.
    assert first["K_gap"] == pytest.approx(4/(2*model.lam))
    assert model.reference_round == 2 and model.gap_norm == 1.
    old_gamma = model.reference_gamma
    saved = model.states.copy()
    model.observe(x, -1)  # a_obs is zero; the next nonzero row is skipped.
    assert model.predict(np.array([.5])) == 0.
    assert model.reference_round == 2 and model.gap_norm == 1.5
    np.testing.assert_array_equal(model.states, saved)
    model.observe(np.array([.5]), 1)
    model.predict(x)
    schedule = model._last_schedule
    assert not schedule["skipped"]
    assert schedule["reference_round"] == 2
    assert schedule["reference_gamma"] == old_gamma
    assert schedule["gap_norm"] == 2.5
    assert schedule["K_gap"] == pytest.approx(2.5**2/(2*old_gamma))
    eps = model._preflight["prescribed"]["epsilon"]
    H = math.log((1+schedule["K_gap"])/(2*eps*eps))
    expected_N = math.ceil(H/(2*math.log1p(schedule["gamma"]/schedule["inveta"])))
    assert schedule["H_warm"] == pytest.approx(H)
    assert schedule["N_prescribed"] == expected_N
    assert model.reference_round == 4 and model.gap_norm == 1.


def test_capped_population_does_not_enlarge_the_skip_budget(monkeypatch):
    model = PersistentLogOdds(1, 1., 1000, 11, {"lanes": 2, "transitions": 1})
    x = np.ones(1)
    model.predict(x)
    model.observe(x, 1)
    M = model._preflight["prescribed"]["m"]
    a = expit(2.)*expit(-2.)
    gamma, lower_b = 1+2*a, 1/(1+2*a)
    allowance = 3*lower_b/(8*M)
    model.a_obs[:] = 4*gamma*allowance  # Delta/2=2 allowance, must sample.
    calls = []
    monkeypatch.setattr(model, "_conditional_draw",
                        lambda r, schedule: calls.append(r.copy()) or np.zeros(1))
    model.predict(x)
    assert len(calls) == 2
    assert not model._last_schedule["skipped"]


def test_full_block_threshold_rejects_early_and_accepts_only_after_all_rows(monkeypatch):
    model, rows = _history_model([[1.], [1.], [1.], [1.]], block_size=1)
    z = np.array([1.5])
    rho = float(_h(1.5)-_h(0.))
    monkeypatch.setattr(model, "_uniform", lambda: math.exp(-rho/2))
    assert not model._residual_test(z)
    assert model.metrics.counters["full_test_rows"] == 1
    monkeypatch.setattr(model, "_uniform", lambda: math.exp(-5*rho))
    assert model._residual_test(z)
    assert model.metrics.counters["full_test_rows"] == 5
    assert model.metrics.counters["poisson_tests"] == 0
    assert model._full_residual(z) == pytest.approx(4*rho, abs=1e-14)


def test_full_test_coin_matches_exp_negative_residual_without_monte_carlo(monkeypatch):
    model, rows = _history_model([[.8], [-.3], [.9]], anchor=[-.4])
    z = np.array([1.1])
    old, new = rows[:, 0]*model.v[0], rows[:, 0]*z[0]
    residual = float(np.sum(_h(new)-_h(old)-(expit(old)-.5)*(new-old)))
    for threshold, expected in [(residual-1e-5, False), (residual+1e-5, True)]:
        monkeypatch.setattr(model, "_uniform", lambda: math.exp(-threshold))
        assert model._residual_test(z) is expected


def test_shared_core_keeps_raw_proposal_center_and_repairs_only_after_rejection(monkeypatch):
    model, rows = _history_model([[1.], [.7]], anchor=[.1])
    r = np.array([100.])
    schedule = model._schedule()
    raw_center = (r-model.g)/schedule["mu"]
    centers = []
    def proposal(center, *args, **kwargs):
        centers.append(center.copy())
        return np.zeros(1)
    accepts = iter([False, True])
    monkeypatch.setattr(shared_core, "box_normal", proposal)
    monkeypatch.setattr(model, "_residual_test", lambda z: next(accepts))
    model._conditional_draw(r, schedule)
    np.testing.assert_array_equal(centers[0], raw_center)
    assert raw_center[0] > model.bound
    assert model.v[0] == model.bound
    exact_g = rows.T@(expit(rows@model.v)-.5)
    np.testing.assert_allclose(model.g, exact_g, atol=2e-15)
    np.testing.assert_allclose(centers[1], (r-exact_g)/schedule["mu"], atol=2e-15)
    assert model.metrics.counters["rejection_repairs"] == 1


def test_guard_is_output_only_and_preserves_pending_prediction_protocol(monkeypatch):
    model = PersistentLogOdds(1, 100., 3, 7, {"lanes": 2, "transitions": 1})
    x = np.ones(1)
    model.predict(x)
    model.observe(x, 1)
    monkeypatch.setattr(model, "_conditional_draw", lambda r, schedule: np.array([150.]))
    guarded = model.predict(x)
    cap = math.log(4*2*3)
    assert guarded == pytest.approx(cap)
    assert model._last_schedule["raw_logit"] == 150.
    np.testing.assert_array_equal(model.states, np.full((2, 1), 150.))
    counters = model.metrics.counters.copy()
    assert model.predict(x.copy()) == guarded
    assert model.metrics.counters == counters
    with pytest.raises(ValueError):
        model.predict(np.zeros(1))
    with pytest.raises(ValueError):
        model.observe(x, 0)
    model.observe(x, -1)
    assert model.a_obs[0] == 0.
    assert np.logaddexp(0., guarded) <= math.log1p(4*2*3)+1e-14
    json.dumps(model.diagnostics(), allow_nan=False)


def test_guard_pointwise_budget_for_both_labels_and_extreme_raw_scores():
    total_budget = 0.
    for t in range(1, 30):
        K = math.log(4*t*(t+1))
        z = np.array([-1e6, -2*K, -K, 0., K, 2*K, 1e6])
        guarded = np.clip(z, -K, K)
        allowance = math.log1p(math.exp(-K))
        total_budget += allowance
        for y in (-1, 1):
            loss = np.logaddexp(0., -y*guarded)
            old = np.logaddexp(0., -y*z)
            assert np.all(loss-old <= allowance+2e-14)
            assert np.max(loss) <= math.log1p(4*t*(t+1))+2e-14
    assert total_budget < 1/4


def test_small_prescribed_schedule_runs_without_silently_reducing_counts():
    model = PersistentLogOdds(1, .4, 3, 777,
                              {"mode": "prescribed", "lanes": 2, "transitions": 1,
                               "history_block_size": 2})
    x = np.ones(1)
    for y in [1, 1, 1]:
        score = model.predict(x)
        assert math.isfinite(score)
        schedule = model._last_schedule
        assert schedule["N_actual"] == schedule["N_prescribed"]
        if not schedule["skipped"]:
            assert schedule["moves_actual"] == model.m*schedule["N_prescribed"]
        model.observe(x, y)
    assert model.metrics.counters["proximal_steps"] > 1
    assert model.metrics.counters["prior_cloud_initializations"] == 1
    assert model.metrics.counters["poisson_tests"] == 0
    assert model.history.n == 3
    with pytest.raises(ValueError):
        model.predict(x)


def test_zero_feature_radius_never_initializes_cloud_or_returns_nonzero():
    model = PersistentLogOdds(2, 1., 3, 919, {"R": 0., "mode": "prescribed"})
    assert model._preflight["estimated_total_moves"] == 0
    for _ in range(3):
        assert model.predict(np.zeros(2)) == 0.
        model.observe(np.zeros(2), 1)
    assert model.states is None
    assert model.metrics.counters["proximal_steps"] == 0
    json.dumps(model.diagnostics(), allow_nan=False)


def test_partially_advanced_failed_prediction_cannot_be_retried(monkeypatch):
    model = PersistentLogOdds(1, 2., 5, 71, {"lanes": 2})
    x = np.ones(1)
    model.predict(x)
    model.observe(x, 1)
    def fail_after_append(r, schedule):
        raise MemoryError("simulated conditional workspace failure")
    monkeypatch.setattr(model, "_conditional_draw", fail_after_append)
    with pytest.raises(MemoryError):
        model.predict(x)
    assert model.history.n == 2
    with pytest.raises(RuntimeError, match="earlier prediction"):
        model.predict(x)
    assert model.history.n == 2
    assert model.diagnostics()["failed"]
