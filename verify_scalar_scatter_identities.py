"""Deterministic algebra/formula checks, not a learning or runtime benchmark."""
import json
import math
from pathlib import Path

import numpy as np

rng = np.random.default_rng(912512)
n, d = 31, 5
X = rng.normal(size=(n, d)) * .04
X[:, 0] += .7
y = np.where(np.arange(n) % 3, 1., -1.)
c = X.mean(axis=0)
kappa = float(np.linalg.norm(c))
e = c / kappa
Z = X - c
di = Z @ e
W = Z.T @ Z
omega = float(e @ W @ e)
label = X.T @ y / 2
lam, young, m, radius = .03, .13, .23, .71
C = kappa / (12 * math.sqrt(3)) + young / 8
A = 1 / (8 * young)


def psi(z):
    return np.logaddexp(z / 2, -z / 2) - math.log(2)


def dpsi(z):
    return np.tanh(z / 2) / 2


def G(t):
    return lam * t * t / 2 - float(e @ label) * t + n * psi(kappa * t)


def J(t, v):
    residual = Z @ (t * e + v)
    return psi(kappa * t + residual) - psi(kappa * t) - dpsi(kappa * t) * residual


max_cancellation_error = max_B_error = max_endpoint_error = 0.
min_row_slack = min_static_slack = math.inf
min_derivative_slack = math.inf
for scale in [0., .3, 3., 13.]:
    v = rng.normal(size=d)
    v = scale * (v - e * float(e @ v))
    reference = m * e + v
    sm = float(reference @ W @ reference)
    qm = float(e @ W @ reference)
    K = radius * (C * (sm + radius * abs(qm) + omega * radius**2 / 3) + A * omega)
    Jv = float(v @ v) + m*m + abs(m) * radius + radius**2 / 3
    bi0 = radius * (C * Jv + A) * np.sum(Z*Z, axis=1)
    values = []
    for t in np.linspace(m-radius, m+radius, 61):
        delta = float(t - m)
        r0 = Z @ reference
        ei = J(t, v) - J(m, v)
        bi = abs(delta) * (C * (r0*r0 + di*r0*delta + di*di*delta*delta/3) + A*di*di)
        B = abs(delta) * (C * (sm + qm*delta + omega*delta*delta/3) + A*omega)
        values.append(B)
        max_B_error = max(max_B_error, abs(float(bi.sum()) - B))
        min_row_slack = min(min_row_slack, float(np.min(bi-np.abs(ei))))
        min_static_slack = min(min_static_slack, float(np.min(bi0-bi)))
        theta = t * e + v
        F = lam * float(theta @ theta)/2 - float(label @ theta) + float(psi(X @ theta).sum())
        reference_F = lam * float(reference @ reference)/2 - float(label @ reference) + float(psi(X @ reference).sum())
        max_cancellation_error = max(max_cancellation_error, abs((F-reference_F)-(G(t)-G(m))-float(ei.sum())))
        residual = Z @ theta
        score = kappa*t
        wi = .25 / math.cosh(score/2)**2
        Ji_derivative = kappa*(dpsi(score+residual)-dpsi(score)-wi*residual)+di*(dpsi(score+residual)-dpsi(score))
        min_derivative_slack = min(min_derivative_slack, float(np.min(C*residual**2+A*di**2-np.abs(Ji_derivative))))
    max_endpoint_error = max(max_endpoint_error, abs(max(values)-K))

# A noncommuting covariance/scatter example verifies the closed Gaussian formulas
# independently of any posterior-sampling implementation.
k = 4
raw = rng.normal(size=(k, k))
covariance = raw @ raw.T
covariance *= .8 / float(np.linalg.eigvalsh(covariance).max())
raw = rng.normal(size=(k, k))
D = raw @ raw.T
D *= .16 / float(np.linalg.eigvalsh(D).max())
mean = rng.normal(size=k) * .2
lam2, nu, s = 1., .16, .7
precision = np.linalg.inv(covariance)
tilted_covariance = np.linalg.inv(precision - 2*s*D)
tilted_mean = tilted_covariance @ precision @ mean
_, logdet_cov = np.linalg.slogdet(covariance)
_, logdet_tilt = np.linalg.slogdet(tilted_covariance)
log_mgf = .5*(logdet_tilt-logdet_cov) + .5*float(mean @ precision @ tilted_covariance @ precision @ mean - mean @ precision @ mean)
h_s = 1-2*s*nu/lam2
M2 = float(mean @ mean)
log_bound = s*(nu*M2+float(np.trace(D))/lam2)/h_s
tilted_second = float(np.trace(tilted_covariance)+tilted_mean @ tilted_mean)
second_bound = k/(lam2*h_s)+M2/h_s**2

out = {
    "conditional_constant_cancellation_max_error": max_cancellation_error,
    "summed_row_envelope_max_error": max_B_error,
    "endpoint_maximum_max_error": max_endpoint_error,
    "minimum_row_envelope_slack": min_row_slack,
    "minimum_static_row_envelope_slack": min_static_slack,
    "minimum_derivative_envelope_slack": min_derivative_slack,
    "noncommuting_covariance_commutator_norm": float(np.linalg.norm(covariance @ D - D @ covariance)),
    "exact_Gaussian_log_mgf": log_mgf,
    "Gaussian_log_mgf_upper_bound": log_bound,
    "exact_tilted_Gaussian_second_moment": tilted_second,
    "tilted_Gaussian_second_moment_upper_bound": second_bound,
}
assert max_cancellation_error < 1e-11
assert max_B_error < 1e-11 and max_endpoint_error < 1e-11
assert min_row_slack > -1e-11 and min_static_slack > -1e-11
assert min_derivative_slack > -1e-11
assert log_mgf <= log_bound and tilted_second <= second_bound
Path("SCALAR_SCATTER_ROOT_FORMULA_CHECK.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
