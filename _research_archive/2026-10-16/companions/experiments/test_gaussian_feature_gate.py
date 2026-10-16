"""Bounded numerical checks for the finite fixed-Gaussian feature gate."""
from fractions import Fraction
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

from .aioli import AIOLI
from .gaussian_feature_gate import (
    GaussianFeatureGate, _gate_log_weights, _newton_cotes_weights,
    _PrescribedGaussianQuadrature,
)


def _independent_normalized_integral(a, variance):
    """Independent adaptive integration, keeping huge negative tails finite."""
    b = math.sqrt(variance)

    def integrand(z):
        shift = b * z
        if a >= 0:
            ratio = np.logaddexp(0.0, -a) - np.logaddexp(0.0, -a - shift)
        else:
            ratio = shift + np.logaddexp(0.0, a) - np.logaddexp(0.0, a + shift)
        return math.exp(float(ratio) - z * z / 2) / math.sqrt(2 * math.pi)

    return quad(integrand, -14, 14, epsabs=2e-13, epsrel=2e-13, limit=100)[0]


@pytest.mark.parametrize("degree", [1, 2, 6, 11, 16])
def test_rational_newton_cotes_moments(degree):
    weights = _newton_cotes_weights(degree)
    assert weights == weights[::-1]
    for power in range(degree + 1):
        assert sum((w * Fraction(i, degree)**power for i, w in enumerate(weights)),
                   Fraction(0)) == Fraction(1, power + 1)


@pytest.mark.parametrize("T,a,variance", [
    (1, -4.0, 1.0), (1, 0.0, 1.0), (1, 4.0, 1.0),
    (10, -30.0, .1), (10, 30.0, .1),
    (1000, -1000.0, .001), (1000, 1000.0, .001),
    (1000000, -10.0, 1e-6), (1000000, 10.0, 1e-6),
])
def test_prescribed_quadrature_against_independent_adaptive_integral(T, a, variance):
    rule = _PrescribedGaussianQuadrature(T)
    predicted = rule.log_likelihood(a, variance)
    reference = -float(np.logaddexp(0.0, -a)) + math.log(
        _independent_normalized_integral(a, variance))
    # Section 7 bounds normalized-integral absolute error by 1/(8T).
    assert abs(predicted - reference) < 1 / (7 * T) + 2e-12
    assert rule.max_condition < 100


@pytest.mark.parametrize("m,z", [
    (1.0, 0.0), (1.0, .7), (1.0, -.7), (1.0, 1.0), (1.0, -1.0),
    (1e-120, 2e-121), (1e-120, -2e-121),
    (1e20, 40.0), (1e20, -40.0), (1e20, 1000.0),
])
def test_stable_gate_matches_center_mixture(m, z):
    log_plus, log_minus = _gate_log_weights(z, m)
    positive = float(np.logaddexp(log_plus - np.logaddexp(0.0, -m),
                                 log_minus - np.logaddexp(0.0, m)))
    negative = float(np.logaddexp(log_plus - np.logaddexp(0.0, m),
                                 log_minus - np.logaddexp(0.0, -m)))
    assert positive - negative == pytest.approx(z, abs=2e-14, rel=2e-14)
    assert np.logaddexp(log_plus, log_minus) == pytest.approx(0, abs=2e-15)
    if z == 1000:
        assert log_minus == pytest.approx(-1000.0)


def test_boundary_and_invalid_anchor_detection():
    assert _gate_log_weights(2.0, 2.0) == (0.0, -math.inf)
    assert _gate_log_weights(-2.0, 2.0) == (-math.inf, 0.0)
    with pytest.raises(FloatingPointError, match="exceeds"):
        _gate_log_weights(2.01, 2.0)
    with pytest.raises(FloatingPointError):
        _gate_log_weights(0.0, 0.0)


def test_online_gate_and_covariance_match_independent_current_surrogate():
    rng = np.random.default_rng(17)
    d, B, T = 4, 3.0, 25
    model = GaussianFeatureGate(d, B, T, 0)
    base = AIOLI(d, B, T, 0, {"matrix_backend": "inverse"})
    A, c, precision = np.eye(d) / B**2, np.zeros(d), float(T)
    for t in range(T):
        x = rng.normal(size=d)
        x /= np.linalg.norm(x)
        y = 1 if t % 3 else -1
        emitted = model.predict(x)
        gate = base.predict(x)
        assert model.last_base_logit == gate
        assert model.last_covariance_variance == pytest.approx(1 / precision)
        assert model.last_projected_variance == pytest.approx(1 / precision)
        assert abs(gate) <= model.last_anchor_score
        assert math.isfinite(emitted)
        p, tail = expit(-y * gate), expit(-abs(gate))
        alpha = tail * (1 - tail) / (2 * (1 + B))
        beta = alpha * gate + y * p / 2
        A += alpha * np.outer(x, x)
        c += beta * x
        precision += float(x @ x) / (4 * d)
        model.observe(x, y)
        base.observe(x, y)
        np.testing.assert_array_equal(model.M, base.M)
        np.testing.assert_array_equal(model.c, base.c)
        np.testing.assert_allclose(model.M @ A, np.eye(d), atol=3e-14, rtol=3e-14)
        np.testing.assert_allclose(model.c, c, atol=3e-14, rtol=3e-14)
        assert model.covariance_precision == precision
    assert model.metrics.counters["gaussian_component_integrals"] == 2 * T
    assert model.diagnostics()["history_rows_retained"] == 0
    json.dumps(model.diagnostics(), allow_nan=False)


def test_actual_gaussian_forecast_differs_from_base_and_matches_direct_integral():
    model = GaussianFeatureGate(1, .8, 5, 0)
    x = np.ones(1)
    model.predict(x)
    model.observe(x, 1)
    emitted = model.predict(x)
    m, variance = model.last_anchor_score, model.last_projected_variance
    a = math.exp(model.last_log_weight_plus)
    component = quad(lambda z: expit(m + math.sqrt(variance) * z)
                     * math.exp(-z*z/2) / math.sqrt(2*math.pi), -14, 14,
                     epsabs=1e-13, epsrel=1e-13)[0]
    exact_positive = a * component + (1 - a) * (1 - component)
    exact_logit = math.log(exact_positive / (1 - exact_positive))
    assert abs(emitted - exact_logit) < 1 / model.T
    assert abs(emitted - model.last_base_logit) > 1e-4
    assert 0 < emitted < model.last_base_logit
    model.observe(x, -1)
    assert model.diagnostics()["max_abs_gaussian_minus_gate_logit"] > 1e-4


def test_zero_features_order_horizon_and_unchanged_state_on_invalid_observe():
    model = GaussianFeatureGate(2, 1.0, 2, 0, {"R": 0})
    x = np.zeros(2)
    initial_M, initial_c, initial_tau = model.M.copy(), model.c.copy(), model.tau
    with pytest.raises(RuntimeError):
        model.observe(x, 1)
    assert model.predict(x) == 0.0
    with pytest.raises(RuntimeError):
        model.predict(x)
    with pytest.raises(ValueError):
        model.observe(x, 0)
    assert model.t == 0
    model.observe(x, 1)
    assert model.predict(x) == 0
    model.observe(x, -1)
    np.testing.assert_array_equal(model.M, initial_M)
    np.testing.assert_array_equal(model.c, initial_c)
    assert model.tau == initial_tau
    with pytest.raises(RuntimeError):
        model.predict(x)
    json.dumps(model.diagnostics(), allow_nan=False)


def test_large_B_tiny_feature_and_saturated_tail_remain_finite():
    for B, x in [(1e6, np.array([1.0, 0.])),
                 (1.0, np.array([1e-100, 0.]))]:
        model = GaussianFeatureGate(2, B, 8, 0)
        for t in range(8):
            value = model.predict(x)
            assert math.isfinite(value)
            assert model.last_covariance_variance > 0
            model.observe(x, 1 if t % 3 else -1)
        json.dumps(model.diagnostics(), allow_nan=False)
    rule = _PrescribedGaussianQuadrature(10)
    # At an enormous negative anchor, the normalized Gaussian integral tends
    # to exp(variance/2). Verify that its noise shift was not lost to a+shift.
    reference = rule.log_likelihood(-1e6, .1) + 1e6
    assert reference == pytest.approx(.05, abs=2e-5)


def test_state_bytes_counts_all_cached_arrays_and_pending_arrays():
    model = GaussianFeatureGate(3, 1.0, 3, 0)
    stable_bytes = 8 * (3*3 + 3) + model.quadrature.state_bytes()
    assert model.state_bytes() == stable_bytes
    model.predict(np.array([1.0, 0., 0.]))
    assert model.state_bytes() == stable_bytes + 8 * 2 * 3
    model.observe(np.array([1.0, 0., 0.]), 1)
    assert model.state_bytes() == stable_bytes


def test_numerical_failure_detector_does_not_replace_quadrature():
    rule = _PrescribedGaussianQuadrature(10)
    rule.signs[:] = -1
    with pytest.raises(FloatingPointError, match="nonpositive"):
        rule.log_likelihood(1.0, .1)
    with pytest.raises(ValueError, match="inverse"):
        GaussianFeatureGate(2, 1.0, 10, 0, {"matrix_backend": "cholesky"})
