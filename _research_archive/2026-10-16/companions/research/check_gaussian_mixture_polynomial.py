"""Finite algebra checks; no performance campaign or existing-source changes."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog
from scipy.special import expit, log_expit, logsumexp, betaln


ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(9112602)


def frame(d):
    forbidden_cap = sum(math.comb(d - 1, j) for j in range(min(3, d - 1) + 1))
    N = 1 << forbidden_cap.bit_length()
    columns = []
    for _ in range(d):
        forbidden = {0}
        for size in range(1, min(3, len(columns)) + 1):
            for subset in itertools.combinations(columns, size):
                val = 0
                for column in subset:
                    val ^= column
                forbidden.add(val)
        columns.append(next(val for val in range(N) if val not in forbidden))
    signs = np.array([[1 - 2 * ((z & a).bit_count() % 2) for a in columns] for z in range(N)], dtype=np.int64)
    return columns, signs


def scalar_c(m):
    return 0.125 if m == 0 else math.tanh(m / 2) / (4 * m)


def true_log_probability(m, variance):
    if variance == 0:
        return float(log_expit(m))
    b = math.sqrt(variance)
    baseline = float(log_expit(m))
    value, error = quad(
        lambda z: math.exp(float(log_expit(m + b * z)) - baseline - z * z / 2 - 0.5 * math.log(2 * math.pi)),
        -math.inf, math.inf, epsabs=3e-12, epsrel=3e-12,
    )
    assert error / value < 1e-9
    return baseline + math.log(value)


checks = {}
frame_reports = []
for d in [1, 2, 3, 4, 6, 8, 10, 12, 16]:
    columns, signs = frame(d)
    N = len(signs)
    assert N <= 2 * d ** 3
    assert np.array_equal(signs.T @ signs, N * np.eye(d, dtype=np.int64))
    for size in range(1, min(4, d) + 1):
        for indices in itertools.combinations(range(d), size):
            assert np.prod(signs[:, indices], axis=1).sum() == 0
    min_absolute_ratio = math.inf
    max_fourth_residual = 0.0
    for _ in range(12):
        b = RNG.normal(size=d)
        b /= np.linalg.norm(b)
        projections = signs @ b
        min_absolute_ratio = min(min_absolute_ratio, float(np.mean(np.abs(projections))))
        fourth_exact = 3 - 2 * np.sum(b ** 4)
        max_fourth_residual = max(max_fourth_residual, abs(float(np.mean(projections ** 4)) - fourth_exact))
    assert min_absolute_ratio >= 1 / math.sqrt(3) - 1e-12
    assert max_fourth_residual < 2e-12
    frame_reports.append({'d': d, 'N': N, 'L': 2 * N,
                          'minimum_tested_absolute_projection': min_absolute_ratio,
                          'fourth_moment_max_residual': max_fourth_residual})
checks['exact_binary_frames'] = frame_reports

atom_reports = []
for d in [2, 3, 4, 8]:
    _, signs = frame(d)
    N = len(signs)
    B = 1.7
    u = RNG.normal(size=d)
    u *= B / np.linalg.norm(u)
    # This feasibility solve checks a proof-only representation. The learner never solves it.
    fit = linprog(np.zeros(N), A_eq=signs.T / N,
                  b_eq=u / (math.sqrt(3) * B), bounds=[(-1, 1)] * N, method='highs')
    assert fit.success, fit.message
    alpha = fit.x
    atoms = np.concatenate([math.sqrt(3) * B * signs, -math.sqrt(3) * B * signs])
    probs = np.concatenate([(1 + alpha) / (2 * N), (1 - alpha) / (2 * N)])
    assert probs.min() >= -1e-12
    probs = np.maximum(probs, 0)
    probs /= probs.sum()
    positive = probs > 0
    kl = float(probs[positive] @ np.log(probs[positive] * 2 * N))
    mean_error = float(np.max(np.abs(probs @ atoms - u)))
    second_error = float(np.max(np.abs(atoms.T @ (probs[:, None] * atoms) - 3 * B * B * np.eye(d))))
    assert mean_error < 2e-9
    assert second_error < 2e-9
    assert kl <= math.log(2) + 1e-10
    if d == 3:
        h = 2
        tuples = list(itertools.product(range(2 * N), repeat=h))
        centers = np.array([atoms[list(row)].mean(axis=0) for row in tuples])
        rho = np.array([np.prod(probs[list(row)]) for row in tuples])
        positive_rho = rho > 0
        tuple_kl = float(rho[positive_rho] @ np.log(rho[positive_rho] * len(rho)))
        covariance = centers.T @ (rho[:, None] * centers) - np.outer(u, u)
        assert abs(tuple_kl - h * kl) < 1e-10
        assert np.max(np.abs(covariance - (3 * B * B * np.eye(d) - np.outer(u, u)) / h)) < 2e-9
        for _ in range(30):
            x = RNG.normal(size=d)
            x /= np.linalg.norm(x)
            y = RNG.choice([-1, 1])
            actual_gap = float(rho @ np.logaddexp(0, -y * (centers @ x)) - np.logaddexp(0, -y * (u @ x)))
            variance = float((3 * B * B * (x @ x) - (u @ x) ** 2) / h)
            assert actual_gap <= min(variance / 8, math.sqrt(variance) / 2) + 1e-10
    atom_reports.append({'d': d, 'KL_to_uniform': kl, 'mean_max_error': mean_error, 'second_moment_max_error': second_error})
checks['proof_only_atom_representations_and_tuple_law'] = atom_reports

max_count_cases = 0
for d in [1, 2, 10, 100, 1000]:
    F = sum(math.comb(d - 1, j) for j in range(min(3, d - 1) + 1))
    L = 2 * (1 << F.bit_length())
    for T in [1, 2, 10, 1000, 1000000]:
        for p in [1, 2, 4]:
            h = max(1, math.floor(p * math.log(max(2, T)) / math.log(L)))
            K = L ** h
            assert K <= max(4 * d ** 3, max(2, T) ** p)
            max_count_cases += 1
checks['polynomial_component_count'] = {'parameter_cases': max_count_cases, 'passed': True}

omd_residuals = []
for d in [1, 3, 10]:
    x = RNG.normal(size=d)
    m = 1.3
    c = scalar_c(m)
    old_q = RNG.uniform(0.05, 0.5, size=d)
    new_q = 1 / (1 / old_q + 2 * c * x * x)
    diagonal_objective = c * (new_q @ (x * x)) + 0.5 * np.sum(new_q / old_q - 1 - np.log(new_q / old_q))
    diagonal_value = 0.5 * np.sum(np.log1p(2 * c * old_q * x * x))
    tau = 0.3
    D = 1 + 2 * c * tau * (x @ x) / d
    new_tau = tau / D
    isotropic_objective = c * new_tau * (x @ x) + d / 2 * (new_tau / tau - 1 - math.log(new_tau / tau))
    iso_value = d / 2 * math.log(D)
    assert abs(diagonal_objective - diagonal_value) < 1e-12
    assert abs(isotropic_objective - iso_value) < 1e-12
    actual_loss = -true_log_probability(m, tau * (x @ x))
    a = float(np.logaddexp(0, -m)) + iso_value
    assert actual_loss <= a + 1e-10
    omd_residuals.append({'d': d, 'diagonal_minimum_residual': abs(diagonal_objective - diagonal_value),
                          'isotropic_minimum_residual': abs(isotropic_objective - iso_value)})
checks['restricted_omd_minimum_and_prediction_dominance'] = omd_residuals

max_log_error = 0.0
max_fraction_of_budget = 0.0
quadrature_cases = 0
largest_N = 0
for m in [-1000, -50, -3, 0, 3, 50, 1000]:
    for variance in [0, 0.01, 1, 10]:
        references = np.array([true_log_probability(m, variance), true_log_probability(-m, variance)])
        for epsilon in [0.5, 0.1, 0.02]:
            delta = epsilon / 4
            b = math.sqrt(variance)
            A = b + math.sqrt(5 * variance / 4 + 2 * math.log(2 / delta))
            N = math.ceil(A * (b + A) / delta)
            largest_N = max(largest_N, N)
            assert N <= 1 + ((53 / 2) * variance + 20 * math.log(8 / epsilon)) / epsilon
            width = 2 * A / N
            nodes = -A + (np.arange(N) + 0.5) * width
            common = -nodes * nodes / 2 + math.log(width)
            log_q = np.array([logsumexp(log_expit(m + b * nodes) + common),
                              logsumexp(log_expit(-m - b * nodes) + common)])
            log_approx = log_q - logsumexp(log_q)
            error = float(np.max(np.abs(log_approx - references)))
            max_log_error = max(max_log_error, error)
            max_fraction_of_budget = max(max_fraction_of_budget, error / epsilon)
            assert error <= epsilon + 2e-8
            quadrature_cases += 1
checks['relative_quadrature_extreme_means'] = {
    'cases': quadrature_cases, 'largest_node_count': largest_N,
    'maximum_absolute_log_probability_error': max_log_error,
    'largest_fraction_of_allowed_error': max_fraction_of_budget,
    'mean_range': [-1000, 1000], 'variance_range': [0, 10],
}

spherical_reports = []
for d in [3, 4, 10, 100, 1000]:
    a = 32 * math.sqrt(d)
    log_c = -float(betaln(0.5, (d - 1) / 2))
    assert math.exp(log_c) <= math.sqrt(d)
    def integrand(t):
        z = t / a
        if z >= 1:
            return 0.0
        density = math.exp(log_c + (d - 3) / 2 * math.log1p(-z * z))
        return 2 * density * float(np.logaddexp(0, -t)) / a
    comparator_loss = quad(integrand, 0, min(a, 100), epsabs=1e-12)[0]
    assert comparator_loss <= 1 / 16
    spherical_reports.append({'d': d, 'comparator_expected_loss': comparator_loss,
                               'proved_upper_bound': 1 / 16})
assert 0.75 * math.log(6 / 5) - 1 / 16 >= 1 / 16
checks['spherical_lower_bound_constants'] = spherical_reports

report = {
    'status': 'all_checks_passed', 'seed': 9112602,
    'scope': 'Supplementary finite algebra and scalar integration checks, not performance experiments or a finite-bit certificate. Comparator linear programs verify proof objects and are not part of the learner.',
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'checks': checks,
}
output = ROOT / 'GAUSSIAN_MIXTURE_POLYNOMIAL_CHECKS.json'
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'check_groups': len(checks), 'output': str(output)}, indent=2))
