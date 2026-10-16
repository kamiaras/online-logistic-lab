"""Mathematical and integration checks for the fixed-budget sigmoid baseline."""

import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal

from .common import History, potential, sigmoid
from .sigmoid import SigmoidMALA, mala_log_acceptance


def test_shared_tempered_potential_gradient_finite_differences():
    history = History(3, block_size=2)
    for x, y in [([1.0, -0.2, 0.7], 1), ([-0.3, 0.5, 1.0], -1),
                 ([0.8, 0.1, -0.6], 1)]:
        history.append(np.array(x), y)
    point = np.array([0.4, -0.7, 0.2])
    _, gradient = potential(point, 1.7, history, last_weight=0.3)
    numerical = np.empty(3)
    for i in range(3):
        increment = np.zeros(3)
        increment[i] = 1e-6
        plus, _ = potential(point + increment, 1.7, history,
                            last_weight=0.3, want_grad=False)
        minus, _ = potential(point - increment, 1.7, history,
                             last_weight=0.3, want_grad=False)
        numerical[i] = (plus - minus) / 2e-6
    np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-8)


def test_mh_ratio_matches_forward_and_reverse_normal_densities():
    history = History(2)
    history.append(np.array([1.0, -0.5]), 1)
    history.append(np.array([0.3, 1.0]), -1)
    current = np.array([[0.2, -0.6], [1.3, 0.8], [-0.4, 0.1]])
    proposed = np.array([[0.8, -0.4], [0.7, -0.2], [0.1, -1.1]])
    value_x, grad_x = potential(current, 1.4, history, last_weight=0.7)
    value_z, grad_z = potential(proposed, 1.4, history, last_weight=0.7)
    h = 0.23
    actual = mala_log_acceptance(current, proposed, value_x, grad_x,
                                 value_z, grad_z, h)
    expected = []
    for x, z, vx, gx, vz, gz in zip(current, proposed, value_x, grad_x, value_z, grad_z):
        forward = multivariate_normal.logpdf(z, mean=x - h * gx, cov=2 * h * np.eye(2))
        reverse = multivariate_normal.logpdf(x, mean=z - h * gz, cov=2 * h * np.eye(2))
        expected.append(vx - vz + reverse - forward)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    reverse_ratio = mala_log_acceptance(proposed, current, value_z, grad_z,
                                        value_x, grad_x, h)
    np.testing.assert_allclose(actual, -reverse_ratio, rtol=1e-12, atol=1e-12)


def test_first_prediction_prior_and_total_bridge_budget():
    config = {"particles": 9, "transitions": 7, "bridges": 3,
              "lazy": False, "history_block_size": 2}
    model = SigmoidMALA(2, 2.0, 2, 73, config)
    prior = 2.0 * np.random.default_rng(73).standard_normal((9, 2))
    np.testing.assert_array_equal(model.theta, prior)
    x = np.array([0.6, -0.8])
    logit = model.predict(x)
    score = prior @ x
    expected_probability = (1 + np.sum(sigmoid(score))) / 11
    assert float(sigmoid(logit)) == pytest.approx(expected_probability)
    assert model.metrics.counters["mala_trials"] == 0
    model.observe(x, 1)
    assert model.metrics.counters["mala_trials"] == 0
    model.predict(-x)
    assert model.metrics.counters["mala_trials"] == 9 * 7
    assert model.metrics.counters["mala_proposals"] == 9 * 7
    assert model.metrics.counters["history_particle_pairs"] == 2 * 9 * 7
    assert model.diagnostics()["bridge_steps_per_trajectory"] == [3, 2, 2]
    before_observe = model.metrics.counters.copy()
    model.observe(-x, -1)
    assert model.metrics.counters["mala_trials"] == before_observe["mala_trials"]
    with pytest.raises(RuntimeError, match="horizon"):
        model.predict(x)
    json.dumps(model.diagnostics(), allow_nan=False)


def test_tempering_only_weights_latest_history(monkeypatch):
    import experiments.sigmoid as module

    observed = []
    original = module.potential

    def record(*args, **kwargs):
        observed.append(kwargs["last_weight"])
        return original(*args, **kwargs)

    model = SigmoidMALA(1, 1, 2, 51,
                        {"particles": 3, "transitions": 5, "bridges": 2, "lazy": False})
    model.predict(np.array([0.5]))
    model.observe(np.array([0.5]), -1)
    monkeypatch.setattr(module, "potential", record)
    model.predict(np.array([1.0]))
    # Three first-rung moves and two final-rung moves; two potential passes each.
    assert observed == [0.5] * 6 + [1.0] * 4
    assert model.history.n == 1


def test_mala_one_dimensional_posterior_moments_and_prediction():
    B = 1.3
    data = [(1.0, 1), (-0.8, -1), (1.2, -1)]
    model = SigmoidMALA(1, B, 5, 4981,
                        {"particles": 256, "transitions": 1, "lazy": True,
                         "history_block_size": 2})
    for x, y in data:
        model.predict(np.array([x]))
        model.observe(np.array([x]), y)

    def density(z):
        loss = z * z / (2 * B * B)
        for x, y in data:
            loss += np.logaddexp(0.0, -y * x * z)
        return np.exp(-loss)

    normalizer = quad(density, -np.inf, np.inf, epsabs=1e-11)[0]
    mean = quad(lambda z: z * density(z), -np.inf, np.inf, epsabs=1e-11)[0] / normalizer
    second = quad(lambda z: z * z * density(z), -np.inf, np.inf,
                  epsabs=1e-11)[0] / normalizer
    probability = quad(lambda z: float(sigmoid(0.7 * z)) * density(z),
                       -np.inf, np.inf, epsabs=1e-11)[0] / normalizer
    for _ in range(250):
        model._mala_step(0.3)
    samples = []
    for i in range(300):
        model._mala_step(0.3)
        if i % 3 == 0:
            samples.append(model.theta[:, 0].copy())
    samples = np.concatenate(samples)
    assert np.mean(samples) == pytest.approx(mean, abs=0.075)
    assert np.mean(samples * samples) == pytest.approx(second, abs=0.09)
    # This exercises the actual public predictor against an independently
    # integrated posterior, including its finite-population guard.
    result = float(sigmoid(model.predict(np.array([0.7]))))
    assert result == pytest.approx(probability, abs=0.055)
    counters = model.metrics.counters
    assert counters["lazy_skips"] > 0
    assert 0 < counters["mala_accepted"] <= counters["mala_proposals"]
    assert counters["mala_trials"] == counters["lazy_skips"] + counters["mala_proposals"]


@pytest.mark.parametrize("B", [1e6, 1e12])
def test_large_radius_predictions_are_finite_and_guarded(B):
    model = SigmoidMALA(3, B, 4, 993,
                        {"particles": 32, "transitions": 4, "bridges": 2})
    data = [(np.array([1.0, 0.0, 0.0]), 1),
            (np.array([0.0, -1.0, 0.0]), -1),
            (np.array([0.3, 0.4, -0.5]), 1),
            (np.zeros(3), -1)]
    for x, y in data:
        logit = model.predict(x)
        assert np.isfinite(logit)
        assert abs(logit) <= np.log(33) + 1e-12
        model.observe(x, y)
    assert np.all(np.isfinite(model.theta))
    assert model.diagnostics()["schedule_certified"] is False
    assert model.state_bytes() == model.theta.nbytes + model.history.nbytes
    json.dumps(model.diagnostics(), allow_nan=False)


@pytest.mark.parametrize("B,stagnant", [(1.0, False), (1e20, True)])
def test_displacement_diagnostic_distinguishes_accepted_float64_stagnation(B, stagnant):
    model = SigmoidMALA(1, B, 2, 19,
                        {"particles": 64, "transitions": 4, "lazy": False})
    x = np.ones(1)
    model.predict(x)
    model.observe(x, 1)
    before = model.theta.copy()
    model.predict(x)
    diagnostic = model.diagnostics()
    if stagnant:
        np.testing.assert_array_equal(model.theta, before)
        assert diagnostic["acceptance_rate"] == 1.0
        assert diagnostic["accepted_zero_displacement"] == 64 * 4
        assert diagnostic["mean_squared_displacement_per_trial"] == 0.0
        assert diagnostic["max_squared_displacement"] == 0.0
    else:
        assert diagnostic["accepted_zero_displacement"] == 0
        assert 0.0 < diagnostic["mean_squared_displacement_per_trial"] <= diagnostic["max_squared_displacement"]
    assert diagnostic["displacement_squares_finite"]
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("config", [
    {"transitions": 2, "bridges": 3}, {"particles": 0},
    {"step_multiplier": 0}, {"lazy": "yes"}, {"transitions": 1.5},
])
def test_invalid_budgets_fail_explicitly(config):
    with pytest.raises(ValueError):
        SigmoidMALA(1, 1.0, 2, 1, config)


def test_observation_lifecycle_and_pending_buffer_are_checked():
    model = SigmoidMALA(2, 1.0, 2, 9)
    x = np.array([1.0, 0.0])
    initial_bytes = model.state_bytes()
    with pytest.raises(RuntimeError):
        model.observe(x, 1)
    model.predict(x)
    assert model.state_bytes() == initial_bytes + x.nbytes
    with pytest.raises(RuntimeError):
        model.predict(x)
    with pytest.raises(ValueError, match="differs"):
        model.observe(-x, 1)
    with pytest.raises(ValueError, match="-1 or"):
        model.observe(x, 0)
    model.observe(x, 1)
