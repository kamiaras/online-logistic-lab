"""Independent mathematical checks and small numerical experiments.

These floating-point checks support the derivation; they do not prove the
theorem or execute its very conservative all-round cubature schedule.
"""
import json
import math
from pathlib import Path
import numpy as np
from disjoint_simplex_ftrl import DisjointSimplexFTRL, sigmoid, theorem_degrees


def check_cubature():
    errors = []
    for d in range(1, 5):
        model = DisjointSimplexFTRL(d, 3, degree=3)
        nodes = model.cubature_nodes()
        cov_u = ((d + 1) * np.eye(d) - np.ones((d, d))) / ((d + 1)**2 * (d + 2))
        expected = (1 + (d + 1) / 3) * np.diag(model.h) @ cov_u @ np.diag(model.h)
        error = max(float(np.max(abs(nodes.mean(axis=0)))),
                    float(np.max(abs(nodes.T @ nodes / len(nodes) - expected))))
        assert error < 2e-13
        errors.append(error)
    # Compare the one-dimensional continuous integral to the independently
    # constructed upper cubature and its Hessian/Taylor error bound.
    model = DisjointSimplexFTRL(1, 2, degree=7)
    gl_x, gl_w = np.polynomial.legendre.leggauss(160)
    continuous_nodes = gl_x[:, None] * model.h / 2
    discrete = model.cubature_nodes()
    maximum_slack_violation = 0.0
    for mean in [-1.3, -0.2, 0.0, 0.9]:
        for scale in [0.1, 0.6, 1.0]:
            for feature in [-1.0, 0.4, 1.2]:
                exact = float(gl_w @ np.logaddexp(0, -feature * (mean + scale * continuous_nodes[:, 0])) / 2)
                upper = float(np.logaddexp(0, -feature * (mean + scale * discrete[:, 0])).mean())
                bound = model.C**2 * feature**2 / (8 * 7)
                maximum_slack_violation = max(maximum_slack_violation, exact - upper, upper - exact - bound)
                assert -1e-13 <= upper - exact <= bound + 1e-13
    return {"moment_errors": errors, "maximum_upper_bound_violation": maximum_slack_violation}


def check_projection_and_gradient():
    rng = np.random.default_rng(1986)
    model = DisjointSimplexFTRL(3, 3, degree=3)
    model.X = rng.normal(size=(5, 3))
    model.y = np.array([1, -1, 1, 1, -1], dtype=float)
    worst_normal = 0.0
    for cell in range(3):
        for _ in range(25):
            point = rng.normal(size=4) * 4
            projected = model.project(point, cell)
            for _ in range(20):
                comparison = model.project(rng.normal(size=4) * 3, cell)
                normal = float((point - projected) @ (comparison - projected))
                worst_normal = max(worst_normal, normal)
                assert normal <= 2e-10
    point = np.append(model.center(1), 0.7)
    value, gradient = model.objective_gradient(point)
    numerical = []
    for j in range(4):
        offset = np.zeros(4); offset[j] = 1e-6
        plus = model.objective_gradient(point + offset)[0]
        minus = model.objective_gradient(point - offset)[0]
        numerical.append((plus - minus) / 2e-6)
    gradient_error = float(np.max(abs(gradient - numerical)))
    assert gradient_error < 3e-7
    return {"maximum_projection_normal_violation": worst_normal,
            "gradient_finite_difference_error": gradient_error}


def check_two_cell_comparator():
    rng = np.random.default_rng(4096)
    worst_mean_error, worst_covariance_violation = 0.0, 0.0
    for d in [1, 2, 5]:
        for M in [1, 2, 7]:
            C, B = 2.0, 1.0
            h = np.full(d, C); h[0] /= M
            cov_u = ((d + 1) * np.eye(d) - np.ones((d, d))) / ((d + 1)**2 * (d + 2))
            covariance = np.diag(h) @ cov_u @ np.diag(h)
            for _ in range(100):
                u = rng.normal(size=d); u *= rng.uniform(0, B) / np.linalg.norm(u)
                s = rng.uniform(1e-4, .5)
                lower = -C + 2 * C * np.arange(M) / M + s * h[0] / (d + 1)
                upper = -C + 2 * C * (np.arange(M) + 1) / M - s * h[0] * d / (d + 1)
                owning = np.where((u[0] >= lower) & (u[0] <= upper))[0]
                if len(owning):
                    centers, weights = [u.copy()], [1.0]
                else:
                    i = int(np.searchsorted(upper, u[0]) - 1)
                    assert 0 <= i < M - 1
                    a, b = upper[i], lower[i + 1]
                    left, right = u.copy(), u.copy(); left[0] = a; right[0] = b
                    centers, weights = [left, right], [(b - u[0]) / (b - a), (u[0] - a) / (b - a)]
                mean = sum(w * c for w, c in zip(weights, centers))
                cov = s*s*covariance + sum(w * np.outer(c - u, c - u) for w, c in zip(weights, centers))
                cap = s*s * (C*C / ((d + 1)*(d + 2)) + h[0]**2 / 4)
                mean_error = float(np.linalg.norm(mean - u))
                violation = float(np.linalg.eigvalsh(cov).max() - cap)
                worst_mean_error = max(worst_mean_error, mean_error)
                worst_covariance_violation = max(worst_covariance_violation, violation)
                assert mean_error < 2e-13 and violation < 2e-13
    return {"mean_error": worst_mean_error, "covariance_bound_violation": worst_covariance_violation}


def experiment(d, M, length, mode):
    model = DisjointSimplexFTRL(d, M, degree=20 if d == 1 else 6,
                                optimization_tolerance=1e-8)
    learner_loss = 0.0
    numerical_allowance = 0.0
    max_iterations, max_gap = 0, 0.0
    predictions = []
    for t in range(length):
        x = np.zeros(d); x[t % d] = 1.0
        p = model.predict(x)
        fit_error = max(f.total_gap_bound for f in model.fits)
        numerical_allowance += fit_error + (d + 1) * model.tau * math.sqrt(2 * fit_error)
        pmin = float(sigmoid(-model.C * math.sqrt(d)))
        numerical_allowance += model.C**2 / (8 * model.degree * pmin)
        if mode == "positive":
            y = 1
        elif mode == "reversal":
            y = 1 if t < length // 2 else -1
        else:
            y = -1 if p >= 0.5 else 1
        learner_loss -= math.log(p if y == 1 else 1 - p)
        predictions.append(p)
        model.observe(x, y)
        max_iterations = max(max_iterations, max(f.iterations for f in model.fits))
        max_gap = max(max_gap, max(f.optimization_gap_bound for f in model.fits))
        for i, fit in enumerate(model.fits):
            lower, upper = model.cell(i)
            vertices = model.vertices(fit)
            assert np.all(vertices >= lower - 1e-10) and np.all(vertices <= upper + 1e-10)
    if d == 1:
        # Independent exhaustive comparator check on the original B-ball.
        points = np.linspace(-1, 1, 20001)[:, None]
        losses = np.logaddexp(0, -model.y[:, None] * (model.X @ points.T)).sum(axis=0)
        comparator_loss = float(losses.min())
    else:
        # A named feasible comparator, not a claim of global minimization.
        comparator = np.full(d, 1 / math.sqrt(d)) if mode == "positive" else np.zeros(d)
        comparator_loss = float(np.logaddexp(0, -model.y * (model.X @ comparator)).sum())
    return {"dimension": d, "components": M, "rounds": length, "sequence": mode,
            "learner_loss": learner_loss, "comparator_loss": comparator_loss,
            "regret": learner_loss - comparator_loss,
            "ideal_fit_theorem_bound_without_error_allowance": model.regret_bound(1.0),
            "conservative_error_allowance_for_executed_degrees": numerical_allowance,
            "bound_with_executed_degree_error_allowance": model.regret_bound(1.0, numerical_allowance),
            "first_prediction": predictions[0], "last_prediction": predictions[-1],
            "max_inner_iterations": max_iterations, "max_reported_optimization_gap": max_gap,
            "final_weights": model.weights.tolist(),
            "final_scales": [f.scale for f in model.fits],
            "qualification": "Practical cubature degrees; not the theorem's full accuracy schedule."}


if __name__ == "__main__":
    result = {"cubature": check_cubature(),
              "projection_and_gradient": check_projection_and_gradient(),
              "two_cell_comparator": check_two_cell_comparator(),
              "experiments": [experiment(1, 2, 40, mode) for mode in
                              ["positive", "reversal", "adversarial"]]
                             + [experiment(2, 3, 24, "adversarial")],
              "conservative_schedule_example": theorem_degrees(10, 2, 1., 1., 1., 10.),
              "status": "Numerical checks passed; ordinary floating point, not interval arithmetic."}
    destination = Path(__file__).with_name("DISJOINT_SIMPLEX_NUMERICAL_CHECKS.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
