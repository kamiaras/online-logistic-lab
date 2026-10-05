"""Independent mathematical checks for the shared-tangent implementation.

These check its exact-real identities numerically and the implemented bookkeeping.
The practical short-chain configuration is deliberately not asserted to mix.
"""
import json
import math

import numpy as np
import pytest
from scipy.special import expit

import experiments.shared_tangent as module
from experiments.shared_tangent import SharedTangentLogOdds


def _h(z):
    z = np.asarray(z)
    return np.logaddexp(0., z)-z/2


def _make_history(d=2, B=1., rows=None, anchor=None):
    model = SharedTangentLogOdds(d, B, 20, 94017,
                                {"tilts": 3, "burn_in": 1, "transitions": 1})
    if rows is None:
        rows = [[.7, -.2], [.1, .8], [-.4, .3]] if d == 2 else [[1.], [.7], [-.4]]
    rows = np.asarray(rows, float)
    for x in rows:
        model.history.append(x)
        model.E += float(x@x)/4
        model.E1 += float(np.abs(x).sum())**2/4
    model.v = np.asarray(anchor if anchor is not None else np.full(d, .2), float)
    model._refresh_cache()
    return model, rows


def test_incremental_cache_includes_current_feature_before_label(monkeypatch):
    model = SharedTangentLogOdds(2, 1., 3, 851)
    first = np.array([.6, -.2])
    assert model.predict(first) == 0.
    assert model.history.n == 1
    np.testing.assert_array_equal(model.a_obs, np.zeros(2))
    model.observe(first, 1)
    model.v[:] = [.3, -.5]
    model._refresh_cache()
    anchor = model.v.copy()
    old_label_sum = model.a_obs.copy()
    monkeypatch.setattr(model, "_conditional_draw", lambda r, schedule: np.array([.1, .2]))
    current = np.array([-.4, .7])
    model.predict(current)
    rows = np.stack([first, current])
    scores = rows@anchor
    np.testing.assert_array_equal(model.v, anchor)
    np.testing.assert_array_equal(model.a_obs, old_label_sum)
    np.testing.assert_allclose(model.g, rows.T@(expit(scores)-.5), atol=2e-15)
    assert model.phi == pytest.approx(_h(scores).sum(), abs=2e-15)
    assert model.history.n == 2
    model.observe(current, -1)
    np.testing.assert_allclose(model.a_obs, .5*(first-current), atol=0)


def test_acceptance_keeps_anchor_gradient_and_value(monkeypatch):
    model, _ = _make_history()
    v, g, phi = model.v.copy(), model.g.copy(), model.phi
    proposed = np.array([-.4, .8])
    monkeypatch.setattr(module, "box_normal", lambda *args, **kwargs: proposed.copy())
    monkeypatch.setattr(model, "_residual_test", lambda z: True)
    actual = model._conditional_draw(np.array([.3, -.7]), model._schedule())
    np.testing.assert_array_equal(actual, proposed)
    np.testing.assert_array_equal(model.v, v)
    np.testing.assert_array_equal(model.g, g)
    assert model.phi == phi


def test_rejection_repairs_to_clipped_unclipped_center_with_fixed_r(monkeypatch):
    model, rows = _make_history(anchor=[.1, -.2])
    schedule = model._schedule()
    r = np.array([100., -100.])
    original_r = r.copy()
    first_center = (r-model.g)/schedule["mu"]
    centers, anchors = [], []
    accepted = np.array([.4, -.6])

    def proposal(center, *args, **kwargs):
        centers.append(center.copy())
        return accepted.copy()

    def test(z):
        anchors.append(model.v.copy())
        return len(anchors) == 2

    monkeypatch.setattr(module, "box_normal", proposal)
    monkeypatch.setattr(model, "_residual_test", test)
    draw = model._conditional_draw(r, schedule)
    clipped = np.clip(first_center, -2*model.B, 2*model.B)
    np.testing.assert_array_equal(centers[0], first_center)
    assert np.max(np.abs(centers[0])) > model.bound
    np.testing.assert_array_equal(anchors[1], clipped)
    np.testing.assert_array_equal(model.v, clipped)
    exact_g = rows.T@(expit(rows@clipped)-.5)
    np.testing.assert_allclose(model.g, exact_g, atol=2e-15)
    assert model.phi == pytest.approx(_h(rows@clipped).sum(), abs=2e-15)
    np.testing.assert_allclose(centers[1], (r-exact_g)/schedule["mu"], atol=2e-15)
    np.testing.assert_array_equal(draw, accepted)
    np.testing.assert_array_equal(r, original_r)


def test_full_residual_and_coordinate_envelope_against_independent_formula():
    model, rows = _make_history()
    rng = np.random.default_rng(674)
    for z in rng.uniform(-2., 2., size=(100, 2)):
        displacement = z-model.v
        old = rows@model.v
        new = rows@z
        residuals = _h(new)-_h(old)-(expit(old)-.5)*(rows@displacement)
        row_l1 = np.abs(rows).sum(axis=1)
        envelopes = row_l1*(np.abs(rows)@(displacement**2))/8
        assert np.all(residuals >= -2e-15)
        assert np.all(residuals <= envelopes+2e-15)
        assert model._full_residual(z) == pytest.approx(residuals.sum(), abs=2e-14)
        assert model.history.W@(displacement**2)/8 == pytest.approx(envelopes.sum())


def test_capped_poisson_acceptance_probability_matches_full_residual():
    # t=3,d=1 has Kcap=1, so the continuation is exercised frequently.
    model, rows = _make_history(d=1, anchor=[.2])
    z = np.array([1.4])
    old, new = rows[:, 0]*.2, rows[:, 0]*1.4
    residual = float(np.sum(_h(new)-_h(old)-(expit(old)-.5)*(new-old)))
    expected = math.exp(-residual)
    n = 12000
    observed = sum(model._residual_test(z) for _ in range(n))/n
    assert abs(observed-expected) < 7*math.sqrt(expected*(1-expected)/n)


def test_capped_poisson_continuation_uses_only_remaining_clock_fraction(monkeypatch):
    model, _ = _make_history(d=1, rows=[[1.]], anchor=[0.])
    # C=1/2, u=1/4, one passed mark exhausts the cap. The final uniform
    # lies strictly between exp(-R) and exp(-R/2), detecting double charging.
    residual = float(_h(2.)-_h(0.))
    continuation_coin = .72
    assert math.exp(-residual) < continuation_coin < math.exp(-residual/2)
    uniforms = iter([math.exp(-.25), .5, .5, .99, continuation_coin])
    monkeypatch.setattr(model, "_uniform", lambda: next(uniforms))
    assert model._residual_test(np.array([2.]))
    assert model.metrics.counters["poisson_arrivals"] == 1
    assert model.metrics.counters["poisson_cap_continuations"] == 1
    assert model.metrics.counters["poisson_continuation_accepts"] == 1
    with pytest.raises(StopIteration):
        next(uniforms)


def test_poisson_rejecting_mark_never_evaluates_full_continuation(monkeypatch):
    model, _ = _make_history(d=1, rows=[[1.]], anchor=[0.])
    uniforms = iter([math.exp(-.25), .5, .5, .1])
    monkeypatch.setattr(model, "_uniform", lambda: next(uniforms))
    monkeypatch.setattr(model, "_full_residual", lambda z: pytest.fail("rejected mark continued"))
    assert not model._residual_test(np.array([2.]))
    assert model.metrics.counters["poisson_factor_rejections"] == 1


def test_zero_displacement_residual_test_uses_no_randomness(monkeypatch):
    model, _ = _make_history()
    monkeypatch.setattr(model, "_uniform", lambda: pytest.fail("zero C used randomness"))
    assert model._residual_test(model.v.copy())


def test_curvature_ledger_is_below_full_box_hessian_and_monotone(monkeypatch):
    d, B = 3, .8
    model = SharedTangentLogOdds(d, B, 15, 956)
    rng = np.random.default_rng(1429)
    rows, ledger = [], np.zeros(d)
    previous = 0.
    for t in range(12):
        x = rng.normal(size=d)
        x /= max(1., np.linalg.norm(x))
        # Symmetric label statistic isolates the deterministic ledger updates.
        model.a_obs[:] = 0.
        assert model.predict(x) == 0.
        rows.append(x)
        X = np.asarray(rows)
        row_l1 = np.abs(x).sum()
        curvature = expit(2*B*row_l1)*expit(-2*B*row_l1)
        ledger += curvature*(2*x*x-np.abs(x)*row_l1)
        expected = max(previous, 0., float(ledger.min()))
        np.testing.assert_allclose(model.curvature_diagonal, ledger, atol=2e-15)
        assert model.curvature_low == pytest.approx(expected)
        previous = expected
        for theta in rng.uniform(-2*B, 2*B, size=(5, d)):
            q = expit(X@theta)*expit(-X@theta)
            hessian = np.eye(d)/B**2+X.T@(q[:, None]*X)
            assert model._last_schedule["gamma"] <= np.linalg.eigvalsh(hessian)[0]+1e-13
        model.observe(x, 1)


def test_cyclic_coordinate_curvature_formula():
    d, B, R = 3, .6, .7
    model = SharedTangentLogOdds(d, B, 12, 498, {"R": R})
    curvature = expit(2*B*R)*expit(-2*B*R)
    for t in range(1, 13):
        x = R*np.eye(d)[(t-1) % d]
        model.a_obs[:] = 0.
        assert model.predict(x) == 0.
        schedule = model._last_schedule
        assert schedule["gamma"] == pytest.approx(B**-2+curvature*R*R*(t//d))
        assert schedule["P"] == pytest.approx(t*R*R/2)
        assert schedule["mu"] == pytest.approx(2*B**-2+schedule["P"])
        model.observe(x, (-1)**t)


def test_prescribed_population_and_mixing_formulas_are_explicit():
    d, B, R, T = 2, .7, .8, 8
    preflight = SharedTangentLogOdds.preflight(d, B, T, {"R": R})
    kappa, L = (B*R)**2, 2*B*R*math.sqrt(d)
    a = expit(L)*expit(-L)
    Gamma = d*math.log1p(a*kappa*T/d)/(8*a)
    expected_m = max(1, math.ceil(max(L, 3*Gamma)))
    assert preflight["prescribed"]["m"] == expected_m
    H = math.log(4*(1+kappa*T*T)*(1+L)**2*T*T)
    assert preflight["prescribed"]["H"] == pytest.approx(H)
    model, _ = _make_history(d=2, B=B)
    schedule = model._schedule()
    denominator = math.log1p(schedule["gamma"]/schedule["inveta"])
    expected_H = model.preflight(2, B, 20)["prescribed"]["H"]
    assert schedule["N_prescribed"] == math.ceil(expected_H/(2*denominator))
    assert schedule["ell_prescribed"] == math.ceil(math.log(32)/denominator)
    assert schedule["mu"] == pytest.approx(model.lam+schedule["inveta"])


def test_population_remains_finite_when_statistical_curvature_underflows():
    out = SharedTangentLogOdds.preflight(1000, 1e6, 10000)
    assert out["prescribed"]["m"] >= 1
    assert out["prescribed"]["m"] == pytest.approx(3*(1e6)**2*10000/8, rel=2e-13)
    json.dumps(out, allow_nan=False)


def test_practical_tilt_cap_does_not_relax_the_certified_skip_budget(monkeypatch):
    model = SharedTangentLogOdds(1, 1., 1000, 918,
                                {"tilts": 1, "burn_in": 1, "transitions": 1})
    x = np.array([1.])
    model.predict(x)
    model.observe(x, 1)
    prescribed_m = model.preflight(1, 1., 1000)["prescribed"]["m"]
    assert prescribed_m > 4
    a = expit(2.)*expit(-2.)
    gamma = 1+2*a
    under_b = 1/(1+2*a)
    proved_rhs = 3*under_b/(8*prescribed_m)
    # At the second round Delta/2 = 2*proved_rhs, so sampling is required.
    # Incorrectly using actual m=1 would skip this deliberately chosen case.
    model.a_obs[:] = 4*gamma*proved_rhs
    calls = []

    def draw(r, schedule):
        calls.append(r.copy())
        return np.array([0.])

    monkeypatch.setattr(model, "_conditional_draw", draw)
    model.predict(x)
    assert len(calls) == 1


def test_midpoint_sweep_counts_and_fresh_prior_initialization(monkeypatch):
    config = {"tilts": 4, "burn_in": 2, "transitions": 3}
    model = SharedTangentLogOdds(1, 1., 3, 655, config)
    x = np.array([1.])
    model.predict(x)
    model.observe(x, 1)
    priors, vectors = [], []

    def prior(alpha, beta, *args, **kwargs):
        priors.append((np.asarray(alpha).copy(), np.asarray(beta).copy()))
        return np.zeros(1)

    monkeypatch.setattr(module, "normal_interval", prior)
    monkeypatch.setattr(module, "normal_standard", lambda *args, **kwargs: np.zeros(1))

    def move(r, schedule):
        vectors.append(r.copy())
        return np.zeros(1)

    monkeypatch.setattr(model, "_conditional_draw", move)
    model.predict(x)
    assert len(priors) == 1
    assert len(vectors) == 2+3*(model.m-1)
    tilts = -.5+(np.arange(model.m)+.5)/model.m
    expected = np.repeat(.5+tilts, [2]+[3]*(model.m-1))
    np.testing.assert_allclose(np.asarray(vectors)[:, 0], expected, atol=0)
    model.observe(x, 1)
    model.predict(x)
    assert len(priors) == 2  # Tangent persists; chain prior is refreshed.


def test_conditional_draw_moments_against_independent_one_dimensional_quadrature():
    model, rows = _make_history(d=1, B=1.1, anchor=[-.9])
    schedule = model._schedule()
    r = np.array([.8])
    nodes, weights = np.polynomial.legendre.leggauss(192)
    nodes = model.bound*nodes
    log_density = -.5*schedule["mu"]*nodes**2+r[0]*nodes-_h(rows@nodes[None, :]).sum(axis=0)
    mass = weights*np.exp(log_density-log_density.max())
    mass /= mass.sum()
    mean = float(mass@nodes)
    variance = float(mass@((nodes-mean)**2))
    n = 7000
    draws = np.array([model._conditional_draw(r, schedule)[0] for _ in range(n)])
    assert abs(draws.mean()-mean) < 7*math.sqrt(variance/n)
    assert abs(draws.var()-variance) < 10*variance*math.sqrt(2/n)
    # These are complete exact conditional rejection draws, independent of the
    # changing tangent in exact arithmetic. This is not a short-chain mixing test.


def test_skip_first_round_zero_feature_and_pending_feature_protocol():
    model = SharedTangentLogOdds(2, 10., 3, 651)
    x = np.array([1., 0.])
    assert model.predict(x) == 0.
    before = model.metrics.counters.copy()
    assert model.predict(x.copy()) == 0.
    assert model.metrics.counters == before
    with pytest.raises((ValueError, RuntimeError)):
        model.predict(np.array([0., 1.]))
    with pytest.raises(ValueError):
        model.observe(x, 0)
    with pytest.raises(ValueError):
        model.observe(-x, 1)
    model.observe(x, 1)
    assert model.predict(np.zeros(2)) == 0.
    model.observe(np.zeros(2), -1)
