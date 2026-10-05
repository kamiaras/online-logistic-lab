import json
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from .aioli import AIOLI, _cholesky_rank_one, _score_root
from .common import Metrics, sigmoid


@pytest.mark.parametrize("v,q", [(0., 1.), (3., .2), (-12., 100.), (100., 1e4)])
def test_scalar_root_matches_independent_score_equation(v, q):
    steps = 40
    metrics = Metrics()
    z, rho, residual = _score_root(v, q, steps, metrics)
    exact = brentq(lambda score: score + q * math.tanh(score / 2.) / 2. - v,
                   v - q / 2., v + q / 2., xtol=1e-13)
    assert abs(z - exact) <= q * 2.**(-steps - 1) + 2e-10
    assert abs(residual) <= (1 + q / 4) * 2.**(-steps) + 2e-10
    assert z == v - q * rho / 2
    assert metrics.counters["root_evaluations"] >= 1


def test_cholesky_update_preserves_matrix_identity():
    rng = np.random.default_rng(3)
    A = np.diag(np.geomspace(1e-6, 1., 7))
    L = np.linalg.cholesky(A)
    for _ in range(20):
        x = rng.normal(size=7)
        _cholesky_rank_one(L, x)
        A += np.outer(x, x)
        np.testing.assert_allclose(L @ L.T, A, rtol=3e-13, atol=3e-13)


def test_online_statistics_and_inverse_backend_match():
    rng = np.random.default_rng(4)
    d, B, T = 6, 5., 35
    stable = AIOLI(d, B, T, seed=8, config={"bisections": 45})
    inverse = AIOLI(d, B, T, seed=19,
                    config={"bisections": 45, "matrix_backend": "inverse"})
    A, c = np.eye(d) / B**2, np.zeros(d)
    for t in range(T):
        x = rng.normal(size=d)
        x /= np.linalg.norm(x)
        y = 1 if t % 3 else -1
        z = stable.predict(x)
        other = inverse.predict(x)
        assert z == pytest.approx(other, rel=2e-10, abs=2e-10)
        h = np.linalg.solve(A, x)
        q, v = x @ h, c @ h
        assert stable.diagnostics()["last_q"] == pytest.approx(q, rel=3e-12)
        assert stable.diagnostics()["last_v"] == pytest.approx(v, abs=3e-12)
        p, tail = sigmoid(np.array([-y * z, -abs(z)]))
        alpha = tail * (1 - tail) / (2 * (1 + B))
        A += alpha * np.outer(x, x)
        c += (alpha * z + y * p / 2) * x
        stable.observe(x, y)
        inverse.observe(x, y)
        np.testing.assert_allclose(stable.L @ stable.L.T, A, rtol=4e-12, atol=1e-13)
        np.testing.assert_allclose(stable.c, c, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(inverse.M @ A, np.eye(d), rtol=5e-11, atol=5e-11)
    assert stable.metrics.counters["observations"] == T
    assert stable.metrics.counters["triangular_solves"] == 2 * T
    assert stable.metrics.counters["cholesky_rank1_updates"] == T
    assert stable.state_bytes() == 8 * (d * d + d)
    json.dumps(stable.diagnostics(), allow_nan=False)


def test_zero_features_and_input_contract():
    model = AIOLI(2, 1., 2, seed=0, config={"R": 0})
    x = np.zeros(2)
    assert model.predict(x) == 0
    with pytest.raises(RuntimeError):
        model.predict(x)
    with pytest.raises(ValueError):
        model.observe(x, 0)
    model.observe(x, 1)
    np.testing.assert_array_equal(model.L, np.eye(2))
    np.testing.assert_array_equal(model.c, np.zeros(2))
    with pytest.raises(ValueError):
        model.predict(np.ones(2))
    assert model.predict(x) == 0
    model.observe(x, -1)
    with pytest.raises(RuntimeError):
        model.predict(x)


def test_large_logits_are_finite_and_curvature_uses_small_tail():
    model = AIOLI(1, 1., 1, seed=0)
    model.c[0] = -40.
    x = np.ones(1)
    z = model.predict(x)
    model.observe(x, 1)
    assert math.isfinite(z)
    assert model.c[0] > -40.
    # p rounds near 1, but the independently computed small tail is retained.
    assert model.metrics.counters["curvature_underflows"] == 0
    assert model.metrics.counters["cholesky_rank1_updates"] == 1


@pytest.mark.parametrize("b,k", [(.1, 8), (1., 15), (10., 26)])
def test_prescribed_bisection_counts(b, k):
    assert AIOLI(10, b, 10000, 0).bisections == k
