"""Independent checks of the probability predictor in SOFT_BAYES_COMPLETE.md.

Numerical distribution checks are short diagnostics, not exact-real or
floating-point correctness certificates. They must run outside timed campaigns.
"""
import gc
import math
import weakref
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, ndtr

from experiments.common import Metrics
from experiments import soft_bayes as sb


def _model(d=1, B=1., T=4, **config):
    return sb.SoftBayes(d, B, T, 984712, config or None)


def _past(model, rows, labels):
    """Populate sufficient statistics without any online sampling."""
    for row, y in zip(rows, labels):
        row = np.asarray(row, dtype=float)
        model.history.append(row)
        model.a_obs += y * row / 2
        model.E += float(row @ row) / 4


@pytest.mark.parametrize("T,B", [(1, 1.), (2, .1), (4, .7), (9, 30.)])
def test_prescribed_static_and_observed_counts(T, B):
    model = _model(d=2, B=B, T=T)
    assert model.m == T
    A = B * B * T * (T - 1) / 2
    H = math.log1p((T + 1) * A)
    np.testing.assert_allclose(model.a_obs, [0., 0.])
    if T > 1:
        _past(model, [[.3, .4]], [1])
    schedule = model._schedule()
    lam = B ** -2
    chi = lam + 2 * model.E
    expected_n = math.ceil(H / math.log1p(lam / chi))
    assert schedule["chi"] == pytest.approx(chi)
    assert schedule["mu"] == pytest.approx(lam + chi)
    assert schedule["H"] == pytest.approx(H)
    assert schedule["N_prescribed"] == expected_n
    assert schedule["N_actual"] == expected_n
    assert expected_n <= 1 + 1.5 * H + 2 * H * B * B * model.E + 1e-10


def test_fresh_prior_per_sample_streams_add_one_probabilities(monkeypatch):
    model = _model(B=2., T=4, mode="practical", samples=3, transitions=2)
    _past(model, [[1.]], [1])
    generated, entering, refs = [], [], []

    def normal_vector(d, rng, metrics=None):
        values = np.array([float(len(generated) - 1)])
        generated.append(values.copy())
        refs.append(weakref.ref(values))
        return values

    def transition(theta, schedule):
        entering.append(theta.copy())
        return theta + .125

    monkeypatch.setattr(sb, "normal_vector", normal_vector)
    monkeypatch.setattr(model, "_transition", transition)
    result = model.predict(np.array([1.]))
    expected_scores = 2 * np.array([-1., 0., 1.]) + .25
    p = (1 + expit(expected_scores).sum()) / 5
    assert result == pytest.approx(math.log(p) - math.log1p(-p))
    assert len(generated) == 3
    assert len(entering) == 6
    np.testing.assert_allclose(np.array(entering)[::2, 0], [-2., 0., 2.])
    np.testing.assert_allclose(np.array(entering)[1::2, 0], [-1.875, .125, 2.125])
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert not hasattr(model, "states")
    assert not hasattr(model, "particles")


def test_predict_uses_only_past_data_and_observe_is_transactional(monkeypatch):
    model = _model(T=3, mode="practical", samples=2, transitions=1)
    x = np.array([.75])
    before_rng = model.rng.bit_generator.state
    assert model.predict(x) == 0.
    assert model.history.n == 0
    assert model.E == 0
    np.testing.assert_array_equal(model.a_obs, [0.])
    assert model.predict(x.copy()) == 0.
    assert model.rng.bit_generator.state == before_rng
    with pytest.raises(ValueError):
        model.predict(np.array([.5]))
    with pytest.raises(ValueError):
        model.observe(x, 0)
    with pytest.raises(ValueError):
        model.observe(np.array([.5]), 1)
    assert model.history.n == 0 and model.E == 0
    model.observe(x, 1)
    np.testing.assert_array_equal(model.a_obs, [.375])
    assert model.E == pytest.approx(.75**2 / 4)
    assert model.history.n == 1
    with pytest.raises(ValueError):
        model.observe(x, 1)

    seen_n = []

    def transition(theta, schedule):
        seen_n.append(model.history.n)
        np.testing.assert_array_equal(model.a_obs, [.375])
        return theta

    monkeypatch.setattr(model, "_transition", transition)
    model.predict(np.array([1.]))
    assert seen_n == [1, 1]
    assert model.history.n == 1


def test_exact_symmetry_after_cancelled_labels_has_no_random_draws(monkeypatch):
    model = _model(T=4)
    _past(model, [[1.], [1.]], [1, -1])

    def forbidden(*args, **kwargs):
        pytest.fail("An exact-symmetry prediction called the normal sampler")

    monkeypatch.setattr(sb, "normal_vector", forbidden)
    np.testing.assert_array_equal(model.a_obs, [0.])
    assert model.predict(np.array([1.])) == 0.
    np.testing.assert_array_equal(model.predict_proba(np.array([1.])), [.5, .5])


@pytest.mark.parametrize("B,R", [(0., 1.), (2., 0.)])
def test_degenerate_radius_runs_neutral_without_sampling(B, R, monkeypatch):
    model = _model(B=B, T=3, R=R)

    def forbidden(*args, **kwargs):
        pytest.fail("A degenerate-radius prediction called the normal sampler")

    monkeypatch.setattr(sb, "normal_vector", forbidden)
    for _ in range(3):
        x = np.array([R])
        assert model.predict(x) == 0.
        model.observe(x, 1)
    assert not model.history.blocks
    assert model.history.n == 3
    assert model.E == 0
    np.testing.assert_array_equal(model.a_obs, [0.])


def test_tiny_score_gradient_retains_first_order_curvature():
    model = _model(T=2)
    _past(model, [[1.]], [1])
    for score in [-1e-15, 1e-15, -1e-10, 1e-10]:
        _, gradient = model._phi_grad(np.array([score]))
        assert gradient[0] == pytest.approx(.25 * score, rel=2e-15, abs=0.)


def test_zero_feature_symmetry_does_not_require_zero_label_summary(monkeypatch):
    model = _model(T=3)
    _past(model, [[1.]], [1])

    def forbidden(*args, **kwargs):
        pytest.fail("A zero-feature prediction called the normal sampler")

    monkeypatch.setattr(sb, "normal_vector", forbidden)
    assert model.a_obs[0] != 0
    assert model.predict(np.array([0.])) == 0.


@pytest.mark.parametrize("score", [-1e6, 1e6])
def test_extreme_scores_keep_both_add_one_label_losses_finite(score, monkeypatch):
    model = _model(T=7, mode="practical", samples=5, transitions=1)
    _past(model, [[1.]], [1])
    monkeypatch.setattr(model, "_transition", lambda theta, schedule: np.array([score]))
    x = np.array([1.])
    logit = model.predict(x)
    probabilities = np.asarray(model.predict_proba(x))
    expected = np.array([1., 6.]) / 7 if score > 0 else np.array([6., 1.]) / 7
    np.testing.assert_allclose(probabilities, expected, rtol=0, atol=2e-16)
    assert probabilities.sum() == pytest.approx(1.)
    assert abs(logit) == pytest.approx(math.log(6.))
    for y in [-1, 1]:
        assert math.isfinite(float(np.logaddexp(0., -y * logit)))
        assert float(np.logaddexp(0., -y * logit)) <= math.log(7.) + 1e-15


@pytest.mark.parametrize("anchor,proposal", [
    (0., 1e-6), (35., 35.01), (-35., -35.01),
    (700., 699.), (-700., -699.), (700., -700.), (-700., 700.),
    (1e6, 1e6 + 1.), (-1e6, -1e6 - 1.),
])
def test_residual_matches_independent_high_precision_scalar_identity(anchor, proposal):
    model = _model(T=2)
    _past(model, [[1.]], [1])
    value = model._full_residual(np.array([proposal]), np.array([anchor]))
    # More than 700 log units of precision resolves saturated logistic tails.
    with localcontext() as context:
        context.prec = 400
        context.Emax, context.Emin = 9999999, -9999999
        a, z = Decimal(anchor), Decimal(proposal)
        h = lambda s: abs(s) / 2 + (1 + (-abs(s)).exp()).ln()
        derivative = (Decimal(".5") - 1 / (1 + abs(a).exp())) * (1 if a >= 0 else -1)
        exact = h(z) - h(a) - derivative * (z - a)
        target = float(exact)
    assert math.isfinite(value) and value >= 0
    assert value == pytest.approx(target, rel=2e-7, abs=2e-320)


def test_conditional_acceptance_identity_cancels_gaussian_and_label_terms():
    model = _model(B=1.7, T=6)
    _past(model, [[1.], [-.7], [.2], [.9]], [1, -1, 1, -1])
    mu = model._schedule()["mu"]
    r = np.array([.8])
    for anchor in [-2., -.2, 1.3]:
        v = np.array([anchor])
        phi_v, grad = model._phi_grad(v)
        c = (r - grad) / mu
        constants = []
        for coordinate in [-3., -1., 0., 1., 3.]:
            z = np.array([coordinate])
            phi_z, _ = model._phi_grad(z)
            log_q = -.5 * mu * float((z - c) @ (z - c))
            log_accept = -model._full_residual(z, v)
            log_target = -.5 * mu * float(z @ z) + float(r @ z) - phi_z
            constants.append(log_q + log_accept - log_target)
        np.testing.assert_allclose(constants, constants[0], rtol=0, atol=3e-14)


def test_uniform_only_normal_has_standard_normal_law_and_three_draw_trials():
    class UniformOnly:
        def __init__(self):
            self.rng = np.random.default_rng(30170)
            self.draws = 0

        def random(self, size=None):
            self.draws += 1 if size is None else int(np.prod(size))
            return self.rng.random(size)

    rng, metrics, size = UniformOnly(), Metrics(), 10000
    values = np.array([sb.normal_standard(rng, metrics) for _ in range(size)])
    assert np.all(np.isfinite(values))
    assert rng.draws % 3 == 0
    assert rng.draws < 9 * size
    assert abs(values.mean()) <= 6 / math.sqrt(size)
    assert abs(values.var() - 1.) <= 6 * math.sqrt(2 / size)
    # Hoeffding plus a union bound: failure probability <= 1e-6 over 3 CDF tests.
    epsilon = math.sqrt(math.log(6e6) / (2 * size))
    for z in [-1., 0., 1.]:
        assert abs(np.mean(values <= z) - ndtr(z)) <= epsilon


def test_uniform_endpoints_are_redrawn_and_invalid_values_fail():
    class Sequence:
        def __init__(self, values):
            self.values = iter(values)

        def random(self):
            return next(self.values)

    metrics = Metrics()
    assert sb._open_uniform(Sequence([0., 1., .25]), metrics) == .25
    assert metrics.counters["uniform_endpoint_redraws"] == 2
    assert metrics.counters["uniform_draws"] == 3
    for invalid in [-.1, 1.1, float("nan"), float("inf")]:
        with pytest.raises(FloatingPointError):
            sb._open_uniform(Sequence([invalid]))


def test_rejections_hold_r_fixed_and_advance_anchor_to_previous_center(monkeypatch):
    model = _model(T=4)
    _past(model, [[1.], [.5]], [1, -1])
    mu = model._schedule()["mu"]
    r = np.array([1.7])
    original_r = r.copy()
    original_phi_grad = model._phi_grad
    anchors, centers = [], []
    gaussian_values = iter([10.] * 9 + [0.])
    uniform_values = iter([1. - 1e-12] * 9 + [1e-12])

    def phi_grad(v):
        phi, grad = original_phi_grad(v)
        anchors.append(v.copy())
        centers.append((r - grad) / mu)
        return phi, grad

    monkeypatch.setattr(model, "_phi_grad", phi_grad)
    monkeypatch.setattr(sb, "normal_vector", lambda *args: np.array([next(gaussian_values)]))
    monkeypatch.setattr(sb, "_open_uniform", lambda *args: next(uniform_values))
    sample = model._conditional(r, mu)
    np.testing.assert_array_equal(r, original_r)
    np.testing.assert_array_equal(anchors[0], r / mu)
    assert len(anchors) == 10
    for index in range(1, len(anchors)):
        np.testing.assert_array_equal(anchors[index], centers[index - 1])
    np.testing.assert_array_equal(sample, centers[-1])
    assert model.metrics.counters["rqo_rejections"] == 9
    assert model.metrics.counters["rqo_accepts"] == 1


def test_combined_auxiliary_gaussian_uses_label_summary_once(monkeypatch):
    model = _model(d=2, B=2., T=4)
    _past(model, [[.5, .5]], [-1])
    schedule = model._schedule()
    theta, gaussian = np.array([.3, -.6]), np.array([-1.2, .4])
    received = []
    monkeypatch.setattr(sb, "normal_vector", lambda *args: gaussian.copy())

    def conditional(r, mu):
        received.append((r.copy(), mu))
        return np.array([4., -5.])

    monkeypatch.setattr(model, "_conditional", conditional)
    result = model._transition(theta, schedule)
    expected_r = schedule["chi"] * theta + model.a_obs + math.sqrt(schedule["chi"]) * gaussian
    np.testing.assert_array_equal(received[0][0], expected_r)
    assert received[0][1] == schedule["mu"]
    np.testing.assert_array_equal(result, [4., -5.])


def test_allocation_free_preflight_reports_formula_and_refuses_without_capping(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Allocation-free preflight allocated an array")

    monkeypatch.setattr(np, "empty", forbidden)
    monkeypatch.setattr(np, "zeros", forbidden)
    T, d, B = 12, 3, 2.
    preflight = sb.SoftBayes.preflight(d, B, T, {"max_total_moves": 1})
    p = preflight["prescribed"]
    H, J, kappa = p["H"], p["J"], B**2
    C = (1000 * d * T**2 + 320 * T * d * (J + 2) * (
        (1 + 1.5 * H) * T * (T - 1) / 2
        + H * kappa * T * (T - 1) * (2 * T - 1) / 12))
    assert p["expected_charged_operations_bound"] == pytest.approx(C, rel=5e-15)
    assert p["theoretical_cells"] == T * d + 16 * d + 128
    assert not preflight["feasible"]
    assert preflight["actual_samples"] == T
    assert preflight["actual_max_transitions"] == p["N_max"]
    assert preflight["estimated_total_moves"] > 1
    assert not preflight["exact_primitive_operation_counter"]
    assert not preflight["section_10_global_operation_cap_implemented"]
    with pytest.raises(ValueError, match="not implemented"):
        sb.SoftBayes.preflight(d, B, T, {"global_operation_cap": 1000})


def test_conditional_samples_match_scalar_quadrature_moments_and_cdf():
    model = _model(B=1.2, T=5)
    rows = np.array([1., -.8, .45, .25])
    _past(model, rows[:, None], [1, -1, 1, -1])
    mu = model._schedule()["mu"]
    r = np.array([1.3])

    # Independent scalar conditional density; no implementation potential used.
    def kernel(z):
        phi = sum(float(np.logaddexp(0., x * z)) - x * z / 2 for x in rows)
        return math.exp(-mu * z * z / 2 + r[0] * z - phi)

    normalizer = quad(kernel, -np.inf, np.inf, epsabs=2e-12)[0]
    mean = quad(lambda z: z * kernel(z), -np.inf, np.inf, epsabs=2e-12)[0] / normalizer
    var = quad(lambda z: (z - mean)**2 * kernel(z), -np.inf, np.inf, epsabs=2e-12)[0] / normalizer
    fourth = quad(lambda z: (z - mean)**4 * kernel(z), -np.inf, np.inf, epsabs=2e-12)[0] / normalizer
    size = 4000
    samples = np.array([model._conditional(r, mu)[0] for _ in range(size)])
    assert abs(samples.mean() - mean) <= 6 * math.sqrt(var / size)
    assert abs(np.mean((samples - mean)**2) - var) <= 6 * math.sqrt((fourth - var**2) / size)
    epsilon = math.sqrt(math.log(6e6) / (2 * size))
    for z in [-.3, .3, .9]:
        expected = quad(kernel, -np.inf, z, epsabs=2e-12)[0] / normalizer
        assert abs(np.mean(samples <= z) - expected) <= epsilon


def test_practical_counts_are_explicit_and_default_remains_prescribed():
    prescribed = _model(T=4)
    practical = _model(T=4, mode="practical", samples=2, transitions=1)
    _past(practical, [[1.]], [1])
    schedule = practical._schedule()
    assert prescribed.m == 4
    assert practical.m == 2
    assert schedule["N_actual"] == 1
    assert schedule["N_prescribed"] > schedule["N_actual"]
    assert practical.diagnostics()["mode"] == "practical"
