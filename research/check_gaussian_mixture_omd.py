"""Supplementary algebra checks for the fixed-mean mixture theorem, not a benchmark."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, logsumexp


ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(9112026)


def loss(m, y=1):
    return np.logaddexp(0.0, -y * np.asarray(m))


def curvature(m):
    m = np.asarray(m, dtype=float)
    result = np.full_like(m, 0.125)
    np.divide(np.tanh(m / 2), 4 * m, out=result, where=m != 0)
    return result


def probability(m, variance):
    if variance == 0:
        return float(expit(m))
    sd = math.sqrt(variance)
    value, error = quad(
        lambda z: expit(m + sd * z) * math.exp(-z * z / 2) / math.sqrt(2 * math.pi),
        -math.inf, math.inf, epsabs=2e-13, epsrel=2e-13,
    )
    assert error < 3e-11
    return value


def logdet(matrix):
    sign, result = np.linalg.slogdet(matrix)
    assert sign > 0
    return float(result)


checks = {}
min_surrogate_gap = math.inf
for m, z in RNG.normal(size=(1500, 2)) * [10, 20]:
    for y in (-1, 1):
        upper = loss(m, y) - y * expit(-y * m) * z + curvature(m) * z * z
        min_surrogate_gap = min(min_surrogate_gap, float(upper - loss(m + z, y)))
assert min_surrogate_gap >= -2e-12
checks['global_quadratic_3000_cases'] = {'minimum_slack': min_surrogate_gap}

min_evidence_gap = math.inf
min_strong_gap = math.inf
for m in [-15, -5, -1, 0, 0.3, math.log(3), 3, 15]:
    for variance in [0, 1e-4, 0.05, 0.5, 3, 25]:
        p = probability(m, variance)
        denom = 1 + 2 * curvature(m) * variance
        a = loss(m) + 0.5 * math.log(denom)
        strong_log_lower = -a + expit(-m) ** 2 * variance / (2 * denom)
        min_evidence_gap = min(min_evidence_gap, math.log(p) + float(a))
        min_strong_gap = min(min_strong_gap, math.log(p) - float(strong_log_lower))
assert min_evidence_gap > -2e-10
assert min_strong_gap > -2e-10
checks['scalar_gaussian_evidence_48_cases'] = {
    'minimum_log_slack': min_evidence_gap,
    'minimum_stronger_log_slack': min_strong_gap,
}

d = 3
raw = RNG.normal(size=(d, d))
old_sigma = raw @ raw.T + np.eye(d)
old_precision = np.linalg.inv(old_sigma)
x = RNG.normal(size=d)
c = float(curvature(1.5))
variance = x @ old_sigma @ x
denom = 1 + 2 * c * variance
new_sigma = old_sigma - 2 * c / denom * np.outer(old_sigma @ x, old_sigma @ x)
new_precision = old_precision + 2 * c * np.outer(x, x)
inverse_residual = np.max(np.abs(new_precision @ new_sigma - np.eye(d)))
assert inverse_residual < 1e-12
minimum = 0.5 * math.log(denom)


def cov_objective(sigma):
    return float(c * (x @ sigma @ x) + 0.5 * (
        np.trace(old_precision @ sigma) - d - logdet(sigma) + logdet(old_sigma)))


assert abs(cov_objective(new_sigma) - minimum) < 1e-12
for _ in range(50):
    perturb = RNG.normal(size=(d, d))
    sigma = perturb @ perturb.T + 0.1 * np.eye(d)
    assert cov_objective(sigma) >= minimum - 1e-12
checks['covariance_minimum_and_rank_one_identity'] = {
    'inverse_residual': float(inverse_residual),
    'minimum_value_residual': abs(cov_objective(new_sigma) - minimum),
    'perturbations_checked': 50,
}

old_w = np.array([0.2, 0.3, 0.5])
new_w = np.array([0.1, 0.7, 0.2])
old_cov = []
new_cov = []
for _ in range(3):
    raw1, raw2 = RNG.normal(size=(2, d, d))
    old_cov.append(raw1 @ raw1.T + np.eye(d))
    new_cov.append(raw2 @ raw2.T + np.eye(d))
old_cov, new_cov = np.array(old_cov), np.array(new_cov)
old_v = old_w[:, None, None] * old_cov
new_v = new_w[:, None, None] * new_cov


def psi(w, cov):
    return float(np.sum(w * np.log(w)) - 0.5 * sum(wi * logdet(si) for wi, si in zip(w, cov)))


grad_w = np.log(old_w) + 1 - 0.5 * np.array([logdet(si) for si in old_cov]) + d / 2
grad_v = -0.5 * np.linalg.inv(old_cov)
bregman = psi(new_w, new_cov) - psi(old_w, old_cov) - grad_w @ (new_w - old_w)
bregman -= np.sum(grad_v * (new_v - old_v))
joint_kl = float(new_w @ np.log(new_w / old_w))
for wi, old, new in zip(new_w, old_cov, new_cov):
    joint_kl += 0.5 * wi * (np.trace(np.linalg.solve(old, new)) - d - logdet(new) + logdet(old))
assert abs(bregman - joint_kl) < 2e-12
checks['lifted_bregman_equals_joint_kl'] = {'residual': abs(bregman - joint_kl)}

toy_results = []
for scenario in range(3):
    d, B, R, T = 2, 1.0, 1.0, 48
    n = max(1, math.ceil(B * R * math.sqrt(T / (4 * d))))
    s = B / n
    centers = np.array(list(itertools.product(np.linspace(-B, B, n + 1), repeat=d)))
    frozen_centers = centers.copy()
    K = len(centers)
    covariances = np.tile(s * s * np.eye(d), (K, 1, 1))
    log_w = np.full(K, -math.log(K))
    sum_a = np.zeros(K)
    center_losses = np.zeros(K)
    log_det_increments = np.zeros(K)
    history_hessian = np.zeros((d, d))
    u = np.array([0.37, -0.61])
    actual_loss = virtual_loss = comparator_loss = 0.0
    min_dominance = math.inf
    for t in range(T):
        if scenario == 0:
            x = np.array([1.0, 0.0]) if t % 2 else np.array([0.0, 1.0])
        else:
            x = RNG.normal(size=d)
            x /= np.linalg.norm(x)
        m = centers @ x
        variance = np.einsum('i,kij,j->k', x, covariances, x)
        component_p = np.array([probability(mi, vi) for mi, vi in zip(m, variance)])
        plus = np.exp(log_w) @ component_p
        if scenario == 0:
            y = 1 if t < T // 2 else -1
        elif scenario == 1:
            y = 1 if RNG.random() < expit(x @ u) else -1
        else:
            y = 1 if plus < 0.5 else -1
        actual_p = plus if y == 1 else 1 - plus
        curv = curvature(m)
        denom = 1 + 2 * curv * variance
        a = loss(m, y) + 0.5 * np.log(denom)
        log_z = logsumexp(log_w - a)
        min_dominance = min(min_dominance, math.log(actual_p) - log_z)
        actual_loss -= math.log(actual_p)
        virtual_loss -= log_z
        comparator_loss += float(loss(x @ u, y))
        sum_a += a
        center_losses += loss(m, y)
        log_det_increments += 0.5 * np.log(denom)
        for k in range(K):
            b = covariances[k] @ x
            covariances[k] -= (2 * curv[k] / denom[k]) * np.outer(b, b)
        log_w += -a - log_z
        history_hessian += np.outer(x, x)
    evidence = -logsumexp(-math.log(K) - sum_a)
    det_final = np.array([0.5 * (d * math.log(s * s) - logdet(si)) for si in covariances])
    assert np.array_equal(centers, frozen_centers)
    assert np.linalg.eigvalsh(covariances).min() > 0
    assert abs(float(logsumexp(log_w))) < 2e-12
    assert min_dominance > -1e-11
    assert actual_loss <= virtual_loss + 1e-11
    assert abs(virtual_loss - evidence) < 1e-10
    det_residual = float(np.max(np.abs(det_final - log_det_increments)))
    assert det_residual < 1e-10
    grid = np.linspace(-B, B, n + 1)
    rho = np.ones(K)
    for j in range(d):
        left = min(n - 1, np.searchsorted(grid, u[j], side='right') - 1)
        left = max(0, left)
        right_prob = (u[j] - grid[left]) / (grid[left + 1] - grid[left])
        factors = np.where(centers[:, j] == grid[left], 1 - right_prob,
                           np.where(centers[:, j] == grid[left + 1], right_prob, 0))
        rho *= factors
    assert np.max(np.abs(rho @ centers - u)) < 1e-12
    S = float(np.trace(history_hessian))
    interpolation_budget = B * B * S / (8 * n * n)
    assert rho @ center_losses <= comparator_loss + interpolation_budget + 1e-11
    grid_bound = math.log(K) + interpolation_budget + 0.5 * logdet(np.eye(d) + s * s * history_hessian / 4)
    regret = actual_loss - comparator_loss
    assert regret <= grid_bound + 1e-10
    simplified_bound = d / 2 * math.log1p(B * B * R * R * T / (4 * d)) + d / 2 * (1 + math.log(10))
    assert grid_bound <= simplified_bound + 1e-10
    toy_results.append({
        'scenario': ['label_switch', 'seeded_labels', 'adaptive_labels'][scenario],
        'd': d, 'T': T, 'n': n, 'K': K, 'actual_loss': actual_loss,
        'virtual_loss': virtual_loss, 'regret_to_fixed_u': regret,
        'grid_bound': grid_bound, 'minimum_prediction_dominance_slack': min_dominance,
        'evidence_telescoping_residual': abs(virtual_loss - evidence),
        'determinant_telescoping_residual': det_residual,
    })
checks['tiny_sequences_144_total_rounds'] = toy_results

m = math.log(3)
p = expit(m)
analytical_second = p * (1 - p) * (20 * p * p - 28 * p + 9) / 4
step = 0.0005
numeric_second = (-math.log(probability(m, 2 * step)) + 2 * math.log(probability(m, step)) - math.log(probability(m, 0))) / step ** 2
assert abs(analytical_second + 9 / 256) < 1e-14
assert abs(analytical_second - numeric_second) < 1e-4
checks['actual_covariance_loss_nonconvexity'] = {
    'analytical_second_derivative_at_zero': float(analytical_second),
    'one_sided_finite_difference': numeric_second,
}

for d in [1, 2, 10, 100]:
    for A in [1e-12, 0.1, 1, 10, 1e4, 1e12]:
        n = max(1, math.ceil(math.sqrt(A / (4 * d))))
        explicit = d * math.log(n + 1) + A / (8 * n * n) + d / 2 * math.log1p(A / (4 * d * n * n))
        simple = d / 2 * math.log1p(A / (4 * d)) + d / 2 * (1 + math.log(10))
        assert explicit <= simple + 1e-10
checks['parameter_bound_24_cases'] = {'passed': True}

report = {
    'status': 'all_checks_passed', 'seed': 9112026,
    'scope': 'Supplementary numerical algebra and tiny-sequence checks; not performance experiments, proof replacement, or a floating-point certificate.',
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'checks': checks,
}
output = ROOT / 'GAUSSIAN_MIXTURE_OMD_CHECKS.json'
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'output': str(output), 'check_groups': len(checks)}, indent=2))
