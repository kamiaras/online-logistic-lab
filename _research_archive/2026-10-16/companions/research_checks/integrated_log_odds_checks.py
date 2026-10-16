"""Supplementary numerical checks, not a floating-point theorem or a benchmark."""
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20260910)
results = {"seed": 20260910, "scope": "Supplementary algebra and acceptance-law checks; no full algorithm benchmark or finite-bit certificate."}


def capped_coin(C, residual, cap, size=200000):
    clock = np.zeros(size)
    alive = np.ones(size, dtype=bool)
    accepted = np.zeros(size, dtype=bool)
    for _ in range(cap):
        ids = np.flatnonzero(alive)
        clock[ids] += RNG.exponential(size=len(ids))
        over = ids[clock[ids] >= C]
        accepted[over] = True
        alive[over] = False
        ids = np.flatnonzero(alive)
        alive[ids[RNG.random(len(ids)) < residual / C]] = False
    ids = np.flatnonzero(alive)
    accepted[ids] = RNG.random(len(ids)) < np.exp(-(1 - clock[ids] / C) * residual)
    actual = float(np.mean(accepted))
    target = math.exp(-residual)
    tolerance = 6 * math.sqrt(target * (1-target) / size) + 1 / size
    assert abs(actual-target) <= tolerance
    return dict(C=C, residual=residual, cap=cap, draws=size,
                observed_acceptance=actual, exact_acceptance=target, tolerance=tolerance)


results["capped_coin"] = [capped_coin(*case) for case in [(0.2, 0.1, 1), (4., 1., 2), (10., 7., 3), (8., 0., 2), (4., 4., 2)]]

X = RNG.normal(size=(100, 13))
X[RNG.random(X.shape) < .4] = 0
X[::11] = 0
norms = np.sum(np.abs(X), axis=1)
prefix = np.cumsum(norms[:, None] * np.abs(X), axis=0)
encoded = np.where(X < 0, -prefix, prefix)
diff = np.diff(np.vstack([np.zeros((1, X.shape[1])), np.abs(encoded)]), axis=0)
decoded = np.divide(np.where(encoded < 0, -diff, diff), norms[:, None],
                    out=np.zeros_like(X), where=norms[:, None] > 0)
decode_error = float(np.max(np.abs(X-decoded)))
assert decode_error < 1e-12
D = RNG.normal(size=X.shape[1])
row_weights = norms * (np.abs(X) @ (D * D)) / 8
W = prefix[-1]
C = float(np.dot(W, D * D) / 8)
two_stage = ((W * D * D / (8*C))[None, :] *
             np.divide(norms[:, None] * np.abs(X), W[None, :],
                       out=np.zeros_like(X), where=W[None, :] > 0)).sum(axis=1)
categorical_error = float(np.max(np.abs(two_stage - row_weights/C)))
assert categorical_error < 1e-14
results["history_encoding"] = {"max_absolute_decode_error": decode_error,
                               "two_stage_probability_error": categorical_error}

cases = 0
max_residual_ratio = 0.
max_gamma_ratio = 0.
max_step_ratio = 0.
max_count_ratio = 0.
c0 = (19 + math.sqrt(37)) / 36
for d in [1, 2, 8, 16]:
    for t in [1, 5, 20, 80]:
        for B in [.25, 1., 4.]:
            lam = B**-2
            R = 1.
            X = RNG.normal(size=(t, d))
            X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
            X *= RNG.uniform(.1, 1, size=(t, 1))
            E = float(np.sum(X*X)/4)
            L1 = np.sum(np.abs(X), axis=1)
            E1 = float(np.sum(L1*L1)/4)
            P = max(E1, 2*E)
            mu = 2*lam+P
            assert P <= (d+1)*E + 1e-12
            max_step_ratio = max(max_step_ratio, P/((d+1)*E))
            feature_a = 1/(np.exp(2*B*L1)+2+np.exp(-2*B*L1))
            ledger = np.zeros(d)
            glow = 0.
            for row, l1, ar in zip(X, L1, feature_a):
                ledger += ar * (2*row*row-np.abs(row)*l1)
                glow = max(glow, float(np.min(ledger)))
            gamma = lam+glow
            theta = RNG.uniform(-2*B, 2*B, size=d)
            curvature = 1/(np.exp(X@theta)+2+np.exp(-(X@theta)))
            H = lam*np.eye(d)+(X.T*curvature)@X
            eig = float(np.linalg.eigvalsh(H)[0])
            assert gamma <= eig+1e-10
            max_gamma_ratio = max(max_gamma_ratio, gamma/eig)
            r = RNG.normal(size=d) * mu * 6*B
            v = RNG.uniform(-2*B, 2*B, size=d)
            def grad(z):
                return X.T @ (.5*np.tanh((X@z)/2))
            k = math.ceil(math.log2(max(1., 9*t*R/(4*math.sqrt(mu)))))
            for n in range(1, k+3):
                previous = v
                v = np.clip((r-grad(previous))/mu, -2*B, 2*B)
                residual = grad(v)-grad(previous)
                if n >= 2:
                    bound = .75*t*R*2.**(-(n-2))
                    assert np.linalg.norm(residual) <= bound+1e-10
            ratio = float(np.linalg.norm(residual)/(math.sqrt(mu)/3))
            max_residual_ratio = max(max_residual_ratio, ratio)
            assert ratio <= 1+1e-10
            horizon = 100
            L = 2*B*R*math.sqrt(d)
            a = 1/(math.exp(L)+2+math.exp(-L))
            Gamma = d/(8*a)*math.log1p(a*(B*R)**2*horizon/d)
            m = max(1, math.ceil(max(L, 3*Gamma)))
            bigH = math.log(4*(1+(B*R)**2*horizon**2)*(1+L)**2*horizon**2)
            AA = bigH+2*(m-1)*math.log(32)
            lograte = math.log1p(gamma/(lam+P))
            nt = math.ceil(bigH/(2*lograte))+(m-1)*math.ceil(math.log(32)/lograte)
            bound = m+AA/2+AA*(lam+P)/(2*gamma)
            assert nt <= bound+1e-7
            max_count_ratio = max(max_count_ratio, nt/bound)
            cases += 1
results["one_loop_curvature_and_counts"] = dict(cases=cases,
    maximum_certified_residual_ratio=max_residual_ratio,
    maximum_gamma_to_actual_minimum_curvature=max_gamma_ratio,
    maximum_new_to_old_inverse_step_energy=max_step_ratio,
    maximum_actual_moves_to_analytic_bound=max_count_ratio, c0=c0)

skip_cases = 0
for B in [.25, 1., 4.]:
    z = np.linspace(-2*B, 2*B, 20001)
    X = RNG.uniform(-1, 1, size=12)
    y = RNG.choice([-1, 1], size=12)
    for t in range(1, 13):
        lam = B**-2
        phi = np.logaddexp((X[:t, None]*z)/2, -(X[:t, None]*z)/2).sum(axis=0)
        aobs = float(np.dot(X[:t-1], y[:t-1])/2)
        feature_a = 1/(np.exp(2*B*np.abs(X[:t]))+2+np.exp(-2*B*np.abs(X[:t])))
        gamma = lam+float(np.dot(feature_a, X[:t]**2))
        def log_integral(b):
            potential = -.5*lam*z*z-phi+b*z
            maximum = float(potential.max())
            return maximum+math.log(float(np.trapezoid(np.exp(potential-maximum), z)))
        odds = log_integral(aobs+X[t-1]/2)-log_integral(aobs-X[t-1]/2)
        delta = min(2*B*abs(X[t-1]), abs(aobs)*abs(X[t-1])/gamma)
        assert abs(odds) <= delta+1e-9
        skip_cases += 1
results["one_dimensional_log_odds_bound"] = {"cases": skip_cases, "method": "20,001-point trapezoid checks only"}

text = (ROOT/"INTEGRATED_LOG_ODDS_COMPLETE.md").read_text(encoding="utf-8")
tags = re.findall(r"\\tag\{([^}]+)\}", text)
assert len(tags) == len(set(tags))
assert text.count(r"\(") == text.count(r"\)")
assert text.count(r"\[") == text.count(r"\]")
assert not any(ord(c)<32 and c not in "\n\r\t" for c in text)
results["manuscript_checks"] = {"equations": len(tags), "inline_pairs": text.count(r"\("),
                                "display_pairs": text.count(r"\["), "no_bad_controls": True}
results["status"] = "all checks passed"
(ROOT/"INTEGRATED_LOG_ODDS_CHECKS.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
print(json.dumps(results, indent=2))
