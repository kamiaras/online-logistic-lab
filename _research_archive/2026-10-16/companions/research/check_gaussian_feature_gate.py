"""Bounded mathematical checks for the feature gate, not a performance study."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(9112603)
REPORT = {}


def nc_weights(n):
    weights = []
    for i in range(n + 1):
        coeff = [1]
        den = 1
        for j in range(n + 1):
            if j == i:
                continue
            new = [0] * (len(coeff) + 1)
            for k, c in enumerate(coeff):
                new[k] -= j * c
                new[k + 1] += n * c
            coeff = new
            den *= i - j
        weights.append(sum((Fraction(c, k + 1) for k, c in enumerate(coeff)), Fraction(0)) / den)
    return weights


def rule(T):
    delta = 1 / (8 * T)
    L = 1 + math.sqrt(2 * math.log(64 * T))
    J = math.ceil(16 * L)
    n = math.ceil(math.log(512 * L * T, 4))
    rational = nc_weights(n)
    assert sum(rational) == 1
    for k in range(n + 1):
        assert sum(w * Fraction(i, n) ** k for i, w in enumerate(rational)) == Fraction(1, k + 1)
    width = 2 * L / J
    nodes = -L + width * (np.arange(J)[:, None] + np.arange(n + 1)[None, :] / n)
    return delta, L, J, n, nodes, width * np.array([float(w) for w in rational])


def normalized_reference(a, variance):
    b = math.sqrt(variance)
    base = float(log_expit(a))
    val, err = quad(lambda z: math.exp(float(log_expit(a + b * z)) - base - z*z/2 - math.log(2*math.pi)/2),
                    -math.inf, math.inf, epsabs=2e-12, epsrel=2e-12)
    assert err < 1e-8 * val
    return val


def normalized_quadrature(a, variance, rr):
    nodes, weights = rr[-2:]
    values = np.exp(log_expit(a + math.sqrt(variance) * nodes) - log_expit(a)
                    - nodes*nodes/2 - math.log(2*math.pi)/2)
    return math.fsum(float(v) for v in (values @ weights))


max_quad_relative = 0.0
min_smoothing_slack = math.inf
quad_cases = 0
grids = []
for T in [1, 10, 1000, 1000000]:
    rr = rule(T)
    grids.append({"T": T, "intervals": rr[2], "degree": rr[3], "nodes_per_integral": rr[2]*(rr[3]+1)})
    for a in [-10000., -100., -8., -0.000000000001, 0., 0.3, 8., 100., 10000.]:
        for variance in [0., 1e-12, 0.1, 1.]:
            ref = normalized_reference(a, variance)
            estimate = normalized_quadrature(a, variance, rr)
            relative = abs(estimate / ref - 1)
            assert estimate > 0 and abs(estimate-ref) <= rr[0] + 2e-11
            assert relative <= 1/(7*T) + 2e-11
            slack = math.log(ref) + variance/8
            assert slack >= -2e-11
            min_smoothing_slack = min(min_smoothing_slack, slack)
            max_quad_relative = max(max_quad_relative, relative)
            quad_cases += 1
REPORT["relative_quadrature"] = {"cases": quad_cases, "max_observed_relative_error": max_quad_relative,
    "minimum_smoothing_slack": min_smoothing_slack, "grids": grids, "passed": True}

gate_cases = 0
max_gate_error = 0.0
for d in [1, 2, 7, 100]:
    for scale in [1e-14, .1, 1., 100.]:
        x = RNG.normal(size=d)
        x *= scale / np.linalg.norm(x)
        W = 3.
        m = W * math.sqrt(d) * np.max(np.abs(x))
        for fraction in [-1., -.99, -.3, 0., .7, 1.]:
            z = fraction * W * np.linalg.norm(x)
            r = math.tanh(z/2) / math.tanh(m/2)
            assert abs(r) <= 1 + 1e-14
            w = (1+r)/2
            p = w*expit(m)+(1-w)*expit(-m)
            err = abs(p-expit(z))
            assert err < 5e-15
            max_gate_error = max(max_gate_error, err)
            gate_cases += 1
REPORT["fixed_anchor_gate"] = {"cases": gate_cases, "max_probability_residual": max_gate_error, "passed": True}

min_scalar_slack = math.inf
for C in [.01, .1, 1., 10., 100.]:
    for _ in range(1000):
        a = RNG.uniform(-C, C)
        z = RNG.uniform(-200, 200)
        f_a, f_z = np.logaddexp(0, -a), np.logaddexp(0, -z)
        p = expit(-z)
        H = math.exp(float(log_expit(z)+log_expit(-z)))
        slack = f_a-f_z+p*(a-z)-H*(a-z)**2/(2*(1+C))
        assert slack >= -1e-10
        min_scalar_slack = min(min_scalar_slack, slack)
REPORT["global_logistic_lower_surrogate"] = {"cases": 5000, "minimum_slack": min_scalar_slack, "passed": True}

mirror_residual = 0.0
radius_excess = 0.0
regret_cases = []
for d in [1, 3, 7]:
    for B in [.1, 1., 10.]:
        T, R = 16, 1.
        lam = B**-2
        M = np.eye(d)/lam
        A = np.eye(d)*lam
        c = np.zeros(d)
        tau = 1/(T*R*R)
        K = math.ceil(math.log2(max(1, 1.5*T*B*B*(1+B*B/4))))
        X, Y, gate_loss = [], [], 0.
        for t in range(1,T+1):
            x = RNG.normal(size=d)
            x /= max(1., np.linalg.norm(x))
            y = int(RNG.choice([-1,1]))
            h = M@x
            q, v = x@h, c@h
            lo, hi = v-q/2, v+q/2
            for _ in range(K):
                mid = (lo+hi)/2
                phi = mid+q/2*math.tanh(mid/2)-v
                if phi <= 0:
                    lo = mid
                else:
                    hi = mid
            z = (lo+hi)/2
            theta = M@c+(z-v)/q*h
            p = expit(-y*z)
            alpha = p*(1-p)/(2*(1+B*R))
            beta = alpha*z+y*p/2
            Mnew = M-alpha/(1+alpha*q)*np.outer(h,h)
            Anew = A+alpha*np.outer(x,x)
            cnew = c+beta*x
            anew = Mnew@cnew
            g = -y*p*x
            grad = 2*A@(anew-M@c)+g+2*alpha*x*(x@(anew-theta))
            mirror_residual = max(mirror_residual, float(np.linalg.norm(grad)))
            assert np.linalg.norm(grad) < 2e-10
            assert np.allclose(Mnew@Anew, np.eye(d), rtol=1e-10, atol=1e-10)
            radius_excess = max(radius_excess, np.linalg.norm(theta)-t*R/(2*lam))
            assert np.linalg.norm(theta) <= t*R/(2*lam)+1e-10
            assert tau*(x@x) <= 1/T+1e-15
            tau = 1/(1/tau+(x@x)/(4*d))
            M, A, c = Mnew, Anew, cnew
            X.append(x)
            Y.append(y)
            gate_loss += np.logaddexp(0,-y*z)
        bound = 2+d*(1+B)*math.log1p(T*B*B/(8*d*(1+B)))
        worst_regret = -math.inf
        X, Y = np.array(X), np.array(Y)
        for _ in range(100):
            u = RNG.normal(size=d)
            u *= B/np.linalg.norm(u)
            comparator = np.logaddexp(0,-Y*(X@u)).sum()
            regret = gate_loss-comparator
            assert regret <= bound+1e-9
            worst_regret = max(worst_regret, float(regret))
        regret_cases.append({"d": d, "B": B, "rounds": T, "largest_checked_regret": worst_regret, "bound": bound})
REPORT["finite_gate_mirror_and_regret"] = {"settings": regret_cases, "max_mirror_gradient_residual": mirror_residual,
    "max_radius_bound_excess": radius_excess, "passed": True}

max_cov_relative = 0.
for d in [1,3,100]:
    for tau in [.001,.1,10.]:
        for norm2 in [.01,1.,100.]:
            expected = 1/(1/tau+norm2/(4*d))
            def objective(log_s):
                s = math.exp(log_s)
                return norm2*s/8+d/2*(s/tau-1-math.log(s/tau))
            fit = minimize_scalar(objective, bounds=(math.log(expected)-3,math.log(expected)+3), method="bounded",
                                  options={"xatol": 1e-12})
            relative = abs(math.exp(fit.x)/expected-1)
            assert relative < 2e-7
            max_cov_relative = max(max_cov_relative, relative)
REPORT["scalar_covariance_kl_minimum"] = {"cases": 27, "max_relative_argmin_residual": max_cov_relative, "passed": True}

REPORT["scope"] = "Supplementary numerical/algebra checks only; not proof by testing, a performance benchmark, or a finite-bit certificate."
REPORT["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for name in ["GAUSSIAN_FEATURE_GATE_COMPLETE.md", "GAUSSIAN_FEATURE_GATE_REPRESENTATION_AUDIT.md", "GAUSSIAN_FEATURE_GATE_REGRET_AUDIT.md"]:
    REPORT.setdefault("source_sha256", {})[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
(ROOT/"GAUSSIAN_FEATURE_GATE_CHECKS.json").write_text(json.dumps(REPORT, indent=2)+"\n", encoding="utf-8")
print(json.dumps({"passed_groups": sum(isinstance(v,dict) and v.get("passed",False) for v in REPORT.values()),
                  "quadrature_cases": quad_cases, "gate_identity_cases": gate_cases,
                  "max_quad_relative_error": max_quad_relative,
                  "max_mirror_gradient_residual": mirror_residual}, indent=2))
