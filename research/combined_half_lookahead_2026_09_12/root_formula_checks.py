"""Deterministic algebra checks; no learning-performance experiment."""
import json
import math
import random
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rng = random.Random(12092026)
report = {"scope": "Finite formula and prototype checks, not a regret proof or learning benchmark", "assertions": 0}

def check(value, label):
    report["assertions"] += 1
    if not value:
        raise AssertionError(label)

def sig(s):
    e = math.exp(-abs(s))
    return 1/(1+e) if s >= 0 else e/(1+e)

def sp(s):
    return max(s, 0)+math.log1p(math.exp(-abs(s)))

def ball(d, radius):
    v = np.array([rng.gauss(0, 1) for _ in range(d)])
    return v*(radius*rng.random()/np.linalg.norm(v))

identity_error = 0.0
clip_error = 0.0
max_local_scaled_violation = -math.inf
min_psd_eigenvalue = math.inf
clipped_cases = 0
for i in range(6000):
    d = (1, 2, 5, 7)[i % 4]
    b = (0.001, 0.3, 3, 30, 1000)[i % 5]
    delta = (0.001, 0.3, 1)[i % 3]
    eps = (0.0001, 0.1, 0.5)[(i // 3) % 3]
    D = math.sqrt(2)*b
    zstar = D*delta
    tau = 2*delta**2/((d+2+zstar)*(math.sqrt(2*(1+zstar))+zstar)**2)
    kappa = 1/(6*(1+b))
    a = eps/4
    M = math.log((1-a)/a)
    w = ball(d, D)
    u = ball(d, b)
    h = ball(d, 1)
    if i % 10 == 0:
        h[:] = 0
        h[0] = 1
        w[:] = 0
        w[0] = D
    v0 = float(w@h)
    s = min(M, max(-M, v0))
    p0 = sig(s)
    c = p0*(1-p0)
    m = p0-0.5
    j = int(np.argmax(abs(h)))
    A = 1+100*rng.random()
    beta = 100*rng.random()
    P = np.full(d, A)
    P[j] += beta
    q = float(np.sum(h*h/P))
    tilde = s-q*m
    p1 = sig(tilde)
    Y = i % 2
    pi = eps+(1-2*eps)*rng.random()
    r = float(u@h)
    raw_residual = sig(v0-q*m)-Y
    lo, hi = sig(-M-q*m)-Y, sig(M-q*m)-Y
    clipped_residual = min(hi, max(lo, raw_residual))
    clip_error = max(clip_error, abs(clipped_residual-(p1-Y)))
    check(abs(clipped_residual-(p1-Y)) <= 3e-16, "paired gradient clipping")
    id_lhs = 0.5*(p1-Y)**2-m*(p1-Y)
    id_rhs = 0.5*p1*(1-p1)+(p1-Y)*(p1-p0)
    identity_error = max(identity_error, abs(id_lhs-id_rhs))
    check(abs(id_lhs-id_rhs) < 2e-16, "optimism identity")
    z = w-(p1-Y)*h/P
    wplus = z.copy()
    wplus[j] = min(b, max(-b, z[j]))
    mask = np.arange(d) != j
    transverse = np.linalg.norm(wplus[mask])
    if transverse > b:
        wplus[mask] *= b/transverse
    check(np.linalg.norm(wplus) <= D*(1+1e-14), "cylinder state bound")
    e, enext = w-u, wplus-u
    potential = 0.5*float(np.sum(P*(e*e-enext*enext)))
    Z = (Y-pi)*(v0-s)
    L1 = math.exp(0.5)/2
    L2 = L1+3*kappa/4
    rhs = potential+L1*c*q+L2*c*q*q-kappa*c*float(h@e)**2+Z
    lhs = (sp(tilde)-Y*tilde)-(sp(r)-Y*r)
    scale = 1+abs(potential)+abs(lhs)+abs(Z)
    violation = (lhs-rhs)/scale
    max_local_scaled_violation = max(max_local_scaled_violation, violation)
    check(violation <= 2e-11, "complete projected one-round inequality")
    if v0 != s:
        clipped_cases += 1
        check((p1-pi)*(v0-s)+1e-12 >= eps*abs(v0-s)/2, "clipping inward drift")
    r0 = tau/2
    zeta = r0*(r0+float(h@h))/(r0+float(h[mask]@h[mask]))
    minorant = np.eye(d)*r0
    minorant[j, j] += zeta
    gap = tau*np.eye(d)+np.outer(h, h)-minorant
    mineig = float(np.linalg.eigvalsh(gap)[0])
    min_psd_eigenvalue = min(min_psd_eigenvalue, mineig)
    check(mineig >= -2e-15, "safe diagonal Loewner bound")
    if d > 1:
        oldj = (j+1) % d
        oldP = np.eye(d)*A
        oldP[oldj, oldj] += beta
        increment = 0.5*kappa*c*minorant
        newP = np.eye(d)*(A+0.5*kappa*c*r0)
        newP[j, j] += 0.5*kappa*c*zeta
        resetgap = increment-(newP-oldP)
        check(float(np.linalg.eigvalsh(resetgap)[0]) >= -2e-14,
              "discarded boost gives a negative semidefinite metric increment")

for omega in (1e-16, 0.001, 1, 100):
    C = S1 = S2 = 0.0
    for c in (0.25, 1e-8, 0.1, 0.24, 0.0001):
        C += c
        S1 += c/(1+omega*C)
        S2 += c/(1+omega*C)**2
        check(S1 <= math.log1p(omega*C)/omega+1e-15, "first self-normalized integral")
        check(S2 <= C/(1+omega*C)+1e-15, "second self-normalized integral")

report.update(identity_max_abs_error=identity_error, clipping_max_abs_error=clip_error,
              local_max_scaled_violation=max_local_scaled_violation,
              safe_diagonal_min_numeric_eigenvalue=min_psd_eigenvalue,
              clipped_one_round_cases=clipped_cases)

def bound(d, b, delta, T):
    D = math.sqrt(2)*b
    z = D*delta
    tau = 2*delta*delta/((d+2+z)*(math.sqrt(2*(1+z))+z)**2)
    kappa = 1/(6*(1+b))
    omega = kappa*tau/4
    L1 = math.exp(0.5)/2
    L2 = L1+3*kappa/4
    out = (b*b/2+L1/omega*math.log1p(omega*T/4)
           +L2*(T/4)/(1+omega*T/4)
           +kappa*D*D*d/2*math.log1p(2*T)+kappa*D*b)
    return out, L1/omega+kappa*D*D*d/2

comparison = []
for eps in (0.1, 0.01, 0.001, 0.0001):
    d, b, delta, T = 1000, 1000, 0.001, 100000
    new, new_coeff = bound(d, b, delta, T)
    oldq = (eps/2)*(1-eps/2)
    oldk = oldq/(4*(1+2*b))
    nu = delta*delta/(d+2)
    old_coeff = 1/(2*oldk*nu)+4*oldk*b*b*d
    comparison.append(dict(epsilon=eps, new_bound=new, old_to_new_asymptotic_log_coefficient=old_coeff/new_coeff))
report["analytic_parameter_comparison"] = comparison

# Compare the in-place prototype to separate, out-of-place matrix equations.
spec = importlib.util.spec_from_file_location("combined_checked", Path(__file__).with_name("combined_half_lookahead.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prototype_error = 0.0
for d in (1, 2, 5):
    for b in (0.01, 0.3, 30, 1000):
        for selected in range(d):
            alg = module.CombinedHalfLookahead(d, b/2, 2, 0.1, 0.001)
            alg.w[0] = 0.9*b
            if d > 1:
                alg.w[1] = 0.7*b
            alg.boost = 0.7
            alg.coordinate = 0
            w = np.array(alg.w)
            h = np.full(d, 0.1/math.sqrt(d))
            h[selected] = 0.8
            x = (2*h).tolist()
            j = int(np.argmax(abs(h)))
            s = min(alg.M, max(-alg.M, float(w@h)))
            p0 = sig(s)
            c = p0*(1-p0)
            mask = np.arange(d) != j
            r0 = alg.tau/2
            zeta = r0*(r0+float(h@h))/(r0+float(h[mask]@h[mask]))
            A = 1+alg.omega*c
            beta = (0.7 if j == 0 else 0)+alg.gamma*c*zeta
            P = np.full(d, A)
            P[j] += beta
            q = float(np.sum(h*h/P))
            expected_p = sig(s-q*(p0-0.5))
            actual_p = alg.predict(x)
            check(abs(actual_p-expected_p) < 4e-16, "prototype forecast")
            check(abs(alg.baseline-A) < 1e-15 and abs(alg.boost-beta) < 1e-14,
                  "prototype adaptive metric and reset")
            Y = selected % 2
            ref = w-(expected_p-Y)*h/P
            ref[j] = min(b, max(-b, ref[j]))
            norm = np.linalg.norm(ref[mask])
            if norm > b:
                ref[mask] *= b/norm
            alg.observe(x, Y)
            error = float(np.max(abs(np.array(alg.w)-ref)))
            prototype_error = max(prototype_error, error)
            check(error < 2e-13*(1+b), "prototype projected update")
            check(abs(alg.sum_curvature_q-c*q) < 3e-16, "curvature diagnostic")
            check(abs(alg.sum_curvature_q_squared-c*q*q) < 3e-16, "squared diagnostic")
            check(alg.stability_diagnostic()["rounds"] == 1, "completed diagnostic rounds")
            expected_bound, _ = bound(d, b, 0.1, 1234)
            check(math.isclose(alg.expected_regret_bound(1234), expected_bound, rel_tol=3e-14),
                  "implementation full theorem constant")
            vector_values = [v for v in vars(alg).values() if isinstance(v, list)]
            check(len(vector_values) == 1 and len(vector_values[0]) == d, "one retained learned vector")

for B, R in ((0, 2), (2, 0), (0, 0)):
    alg = module.CombinedHalfLookahead(2, B, R, 0.1, 0.001)
    check(alg.predict([0, 0]) == 0.5, "neutral forecast")
    alg.observe([0, 0], 1)
    check(alg.expected_regret_bound(10) == 0, "neutral regret")

mix = module.SafeguardedHalfLookahead(2, 30, 1, 0.1, 0.001)
lossmix = lossnew = lossold = 0.0
for x, Y in (([0.8, 0.1], 1), ([0.1, 0.8], 0), ([-0.4, 0.2], 1)):
    p = mix.predict(x)
    pn, po = mix.pending_combined, mix.pending_original
    lossmix -= math.log(p if Y else 1-p)
    lossnew -= math.log(pn if Y else 1-pn)
    lossold -= math.log(po if Y else 1-po)
    mix.observe(x, Y)
    target = -np.logaddexp(-lossnew, -lossold)+math.log(2)
    check(abs(lossmix-float(target)) < 3e-14, "pathwise mixture evidence identity")
    check(lossmix <= min(lossnew, lossold)+math.log(2)+1e-14, "pathwise two-expert bound")
check(mix.expected_regret_bound(3) == min(mix.combined.expected_regret_bound(3), mix.original.expected_regret_bound(3))+math.log(2),
      "mixture theorem constants")
check(math.isfinite(mix.combined.expected_regret_bound(10**1000)), "bound evaluation has no horizon-length loop")
report["prototype_max_parameter_error"] = prototype_error
report["prototype_status"] = "PASS"
report["status"] = "PASS"
(ROOT/"COMBINED_HALF_LOOKAHEAD_ROOT_FORMULA_CHECK.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
print(json.dumps(report, indent=2))
