"""Distributional and mathematical checks; not a floating-point certificate."""
import json
import math

import numpy as np
import pytest
from scipy.stats import truncnorm

from experiments.common import Metrics, potential
from experiments.proximal import (PrescribedBudgetExceeded, ProximalLogOdds,
                                  _box_normal, _clip_roundoff, logistic_bregman,
                                  truncated_standard_normal)


@pytest.mark.parametrize("a,b", [(-2., 2.), (0., .2), (2., 8.), (10., 10.03),
                                 (-12., -9.), (-.2, .3), (-.1, 1.2)])
def test_truncated_normal_moments_against_scipy(a, b):
    n = 40000
    metrics = Metrics()
    z = truncated_standard_normal(np.full(n, a), np.full(n, b),
                                  np.random.default_rng(162), metrics)
    mean, var = truncnorm.stats(a, b, moments="mv")
    assert np.all((a <= z) & (z <= b))
    assert abs(z.mean()-mean) <= 8*math.sqrt(var/n)+1e-10
    assert abs(z.var()-var) <= 10*var*math.sqrt(2/n)+1e-10
    assert metrics.counters["tn_proposals"]/n < 6
    assert metrics.counters["tn_rejections"] == metrics.counters["tn_proposals"]-n


def test_narrow_tail_and_far_center_do_not_lose_the_interval():
    rng = np.random.default_rng(19)
    a, b = 30., 30.+1e-9
    z = truncated_standard_normal(np.full(20000, a), np.full(20000, b), rng)
    scaled = (z-a)/(b-a)
    assert abs(scaled.mean()-.5) < .012
    assert abs(scaled.var()-1/12) < .004
    # Forming standardized alpha+width loses all width at these centers.
    centers = np.array([[-1e20, 1e20, 0., 1.99]])
    z = _box_normal(centers, 1., 2., rng, Metrics())
    assert np.all(np.isfinite(z))
    assert np.all(np.abs(z) <= 2.)


def test_only_tolerance_checked_roundoff_is_clamped():
    metrics = Metrics()
    z = _clip_roundoff(np.array([np.nextafter(1., 2.)]), -1., 1., metrics)
    assert z[0] == 1.
    assert metrics.counters["roundoff_boundary_clamps"] == 1
    with pytest.raises(FloatingPointError):
        _clip_roundoff(np.array([1.01]), -1., 1., metrics)
    assert metrics.counters["numerical_failures"] == 1


def test_invalid_truncated_intervals():
    rng = np.random.default_rng(0)
    for a, b in [(1., 1.), (2., 1.), (-np.inf, 1.), (0., np.nan)]:
        with pytest.raises(ValueError):
            truncated_standard_normal(a, b, rng)


def test_population_formula_remains_finite_after_curvature_underflow():
    d, B, T = 1000, 1000., 1000
    out = ProximalLogOdds.preflight(d, B, T)
    p = out["prescribed"]
    assert p["log_a"] < -745
    limiting = (1+B*B/12)*B*B*T/8
    assert abs(p["m"]-limiting) <= max(2., limiting*1e-13)
    assert out["actual_particles"] == 16
    assert out["actual_max_transitions"] == 1
    assert out["feasible"]
    json.dumps(out, allow_nan=False)


def test_prescribed_preflight_precedes_rng_and_large_allocations(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("RNG initialized before failed preflight")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    with pytest.raises(PrescribedBudgetExceeded) as failure:
        ProximalLogOdds(1000, 1000., 1000, 0, {"mode": "prescribed"})
    assert failure.value.details["actual_particles"] > 100000
    assert not failure.value.details["feasible"]


def _history_model(d=2, B=1.):
    model = ProximalLogOdds(d, B, 20, 923)
    rng = np.random.default_rng(537)
    for i in range(5):
        x = rng.normal(size=d)
        x /= max(1., np.linalg.norm(x))
        model.history.append(x, (-1)**i)
        model.sum_norm2 += float(x@x)
    return model


def test_conditional_gradient_majorant_and_constrained_residual():
    model = _history_model()
    x = np.array([.8, -.3])
    schedule = model._schedule(x)
    Y = np.array([[.2, -.1], [1e5, -1e5], [-8., 3.]])
    p = np.array([.1, .8, .5])
    env = model._conditional_center(Y, x, p, schedule)
    ratios = np.linalg.norm(env["residual"], axis=1)/math.sqrt(schedule["mu"])
    assert np.max(ratios) <= 1.+1e-8
    rng = np.random.default_rng(16)
    for j in range(len(Y)):
        test_z = rng.uniform(-2., 2., size=(100, model.d))
        assert np.max((test_z-env["v"][j])@env["normal"][j]) < 1e-7
    # A moderate auxiliary value avoids cancellation in this independent oracle.
    j = 0
    v, y, g = env["v"][j], Y[j], env["g"][j]
    def value(z):
        f, _ = potential(z, model.B, model.history, extra_x=x, extra_p=p[j],
                         want_grad=False)
        return f+.5*schedule["inveta"]*np.sum((z-y)**2, axis=-1)
    h = 1e-6
    numerical_gradient = np.array([(value(v+h*e)-value(v-h*e))/(2*h)
                                   for e in np.eye(model.d)])
    np.testing.assert_allclose(g, numerical_gradient, atol=2e-8, rtol=2e-8)
    z = rng.uniform(-2., 2., size=(50, model.d))
    vv = np.broadcast_to(v, z.shape)
    gap = model._rejection_gap(vv, z, x)
    hz = z-v
    q = value(v)+hz@g+.5*schedule["mu"]*np.sum(hz*hz, axis=1)
    np.testing.assert_allclose(gap, value(z)-q, atol=2e-13, rtol=2e-12)
    assert np.min(gap) >= 0.
    assert np.all(gap <= .5*schedule["E"]*np.sum(hz*hz, axis=1)+1e-12)


def test_logistic_remainder_large_scores_and_small_displacements():
    old = np.array([1000., -1000., 1000., -1000., 0., .3])
    new = np.array([1001., -1001., -1000., 1000., 1e-8, .30000001])
    gap = logistic_bregman(old, new)
    np.testing.assert_allclose(gap[:4], [0., 0., 1000., 1000.])
    assert np.all(gap >= 0.)
    np.testing.assert_allclose(gap[4], 1.25e-17, atol=2e-24)


def test_conditional_posterior_samples_against_one_dimensional_quadrature():
    model = _history_model(d=1)
    x, p, Y = np.array([.7]), .65, np.array([.2])
    schedule = model._schedule(x)
    nodes, weights = np.polynomial.legendre.leggauss(256)
    z = 2*nodes
    values, _ = potential(z[:, None], model.B, model.history, extra_x=x,
                          extra_p=p, want_grad=False)
    values += .5*schedule["inveta"]*(z-Y[0])**2
    density = weights*np.exp(-values+np.min(values))
    density /= density.sum()
    mean = float(density@z)
    var = float(density@((z-mean)**2))
    n = 20000
    draws = model._conditional_draw(np.full((n, 1), Y[0]), x, p, schedule)[:, 0]
    assert abs(draws.mean()-mean) < 7*math.sqrt(var/n)
    assert abs(draws.var()-var) < 10*var*math.sqrt(2/n)
    counts = model.metrics.counters
    assert counts["rqo_proposals"]-counts["rqo_rejections"] == n
    assert counts["rqo_proposals"]/n < math.e+.05
    assert counts["numerical_failures"] == 0


@pytest.mark.parametrize("d,B,particles", [(1000, 1000., 16), (10, 10000., 64),
                                          (5000, 100., 16)])
def test_large_scale_high_dimension_cancellation_smoke(d, B, particles):
    model = ProximalLogOdds(d, B, 1000, 182,
                           {"particles": particles, "transitions": 1})
    rng = np.random.default_rng(32)
    for t in range(3):
        x = rng.normal(size=d)
        x /= np.linalg.norm(x)
        logit = model.predict(x)
        assert math.isfinite(logit)
        assert np.all(np.abs(model.theta) <= 2*B)
        model.observe(x, (-1)**t)
    assert model.metrics.counters["proximal_steps"] == 3*particles
    assert model.metrics.counters["numerical_failures"] == 0
    assert model.diagnostics()["max_center_residual_ratio"] < 1.+1e-6
    json.dumps(model.diagnostics(), allow_nan=False)


def test_prescribed_schedule_is_not_silently_capped():
    model = ProximalLogOdds(1, .5, 2, 89,
                           {"mode": "prescribed", "particles": 1, "transitions": 1})
    for t in range(2):
        x = np.array([1.])
        prediction = model.predict(x)
        steps = model.metrics.counters["proximal_steps"]
        assert model.predict(x) == prediction  # No second refresh for the same round.
        assert model.metrics.counters["proximal_steps"] == steps
        assert model._last_schedule["N_actual"] == model._last_schedule["N_prescribed"]
        assert model._last_schedule["N_actual"] > 1
        model.observe(x, (-1)**t)
    assert model.metrics.counters["proximal_steps"] == model.m*model._n_theoretical_sum
    assert model.state_bytes() == model.history.nbytes+model.theta.nbytes+model.tilts.nbytes
    with pytest.raises(ValueError):
        model.predict([1.])


def test_feature_label_validation_and_reproducibility():
    a, b = (ProximalLogOdds(2, 1., 2, 41) for _ in range(2))
    with pytest.raises(ValueError):
        a.predict([2., 0.])
    for x, y in [(np.array([.5, .5]), 1), (np.array([0., 0.]), -1)]:
        assert a.predict(x) == b.predict(x)
        with pytest.raises(ValueError):
            a.observe(x, 0)
        with pytest.raises(ValueError):
            a.observe(x+.01, y)
        a.observe(x, y)
        b.observe(x, y)
    np.testing.assert_array_equal(a.theta, b.theta)
