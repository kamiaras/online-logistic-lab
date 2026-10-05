import numpy as np
import pytest
from scipy.optimize import minimize

from .common import sigmoid
from .comparator import fit_comparator


@pytest.mark.parametrize("positive,negative,B,optimum", [
    (50, 0, 2., 2.),
    (30, 10, 3., np.log(3.)),
    (20, 10, .2, .2),
    (10, 30, 3., -np.log(3.)),
])
def test_one_dimensional_analytic_comparator(positive, negative, B, optimum):
    X = np.ones((positive + negative, 1))
    y = np.r_[np.ones(positive), -np.ones(negative)]
    result = fit_comparator(X, y, B, tolerance=1e-6, max_seconds=3.)
    expected = float(np.logaddexp(0., -y * optimum).sum())
    assert result["converged"], result
    assert result["theta_norm"] <= B
    assert abs(result["loss"] - expected) < 2e-6
    assert result["loss_lower_bound"] <= expected + 2e-9
    assert result["loss"] + 2e-9 >= expected
    assert result["theta"][0] == pytest.approx(optimum, abs=2e-6)


def test_certificate_brackets_independent_constrained_solver():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(60, 4))
    y = np.where(X[:, 0] + .4 * rng.normal(size=60) > 0, 1., -1.)
    B = 1.3
    def value_gradient(u):
        signed = -y * (X @ u)
        return float(np.logaddexp(0., signed).sum()), X.T @ (-y * sigmoid(signed))
    reference = minimize(value_gradient, np.zeros(4), jac=True, method="SLSQP",
                         constraints=[{"type": "ineq", "fun": lambda u: B**2 - u @ u,
                                       "jac": lambda u: -2 * u}],
                         options={"ftol": 1e-12, "maxiter": 300})
    assert reference.success
    result = fit_comparator(X, y, B, initial=np.ones(4) * 100,
                            tolerance=1e-5, max_seconds=3.)
    value, gradient = value_gradient(result["theta"])
    assert result["theta_norm"] <= B
    assert result["loss"] == pytest.approx(value, abs=1e-12)
    assert result["gap"] == pytest.approx(max(0., gradient @ result["theta"]
                                             + B * np.linalg.norm(gradient)), abs=1e-10)
    assert result["loss_lower_bound"] <= reference.fun + 1e-8
    assert result["loss"] >= reference.fun - 1e-8
    assert result["converged"], result


def test_zero_budget_returns_complete_initial_certificate():
    X, y = np.ones((20, 1)), np.ones(20)
    result = fit_comparator(X, y, 2., max_seconds=0.)
    assert result["status"] == "time_limit"
    assert not result["converged"]
    assert result["timed_out"]
    assert result["evaluations"] == 1
    assert result["loss"] == pytest.approx(20 * np.log(2))
    assert result["gap"] == pytest.approx(20.)
    assert result["history_rows_evaluated"] == 20
    assert result["finite_precision_certified"] is False


def test_empty_data_zero_radius_and_warm_start():
    result = fit_comparator(np.empty((0, 3)), np.empty(0), 2., initial=np.ones(3) * 10)
    assert result["loss"] == 0 and result["gap"] == 0 and result["converged"]
    assert result["theta_norm"] <= 2.
    result = fit_comparator(np.ones((12, 2)), np.ones(12), 0., initial=np.ones(2))
    np.testing.assert_array_equal(result["theta"], np.zeros(2))
    assert result["gap"] == 0 and result["converged"]
    first = fit_comparator(np.ones((12, 1)), np.ones(12), 2.)
    warm = fit_comparator(np.ones((12, 1)), np.ones(12), 2., initial=first["theta"])
    assert warm["evaluations"] == 1 and warm["converged"]


@pytest.mark.parametrize("kwargs", [
    {"B": -1.}, {"B": 1., "tolerance": -1.}, {"B": 1., "max_seconds": -1.},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        fit_comparator(np.ones((2, 1)), np.ones(2), **kwargs)
