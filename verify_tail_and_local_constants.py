"""Evaluate proved conditional-envelope formulas; no learning benchmark."""
import json
import math
from pathlib import Path

def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(x-m) for x in values))

def local_tail(d, radius, error=.5):
    slope = radius-2*math.sqrt(error)
    lb = (d-1)*math.log(radius)-math.log(slope)
    terms = [lb]
    for j in range(d-1):
        lb += math.log(d-1-j)-math.log(slope*radius)
        terms.append(lb)
    log_tail = ((1-d/2)*math.log(2)-radius**2/2-
                math.lgamma(d/2)+logsumexp(terms))
    tail = math.exp(log_tail)
    q = -math.expm1(-(radius**2-d-d*math.log(radius**2/d))/2)
    nbar = math.exp(2*error)/q
    return {"dimension": d, "radius": radius, "tail_integral": tail,
            "posterior_tail_bound": math.exp(2*error)*tail/q,
            "expected_exact_trials": nbar*(1+tail),
            "expected_local_gaussians": nbar,
            "TV_1e9_gaussian_cap": math.ceil(nbar*math.log(2e9))}

def local_K(radius, tau, alpha):
    x = tau*radius
    term = x*x*x/6
    rest = term
    for j in range(4, 100):
        term *= x/j
        rest += term
        if term < rest*1e-18:
            break
    return alpha*radius+rest/tau**2

n, dimension, lam, delta = 166, 1000, .001, .001
amax, gmin = 2*delta, 1-4*delta
threshold = (math.log(2*n)+amax*math.sqrt((dimension-1)/lam)+amax**2/(2*lam))/gmin
q = 7.838
rate = n*math.exp(-gmin*q+amax*math.sqrt((dimension-1)/lam)+amax**2/(2*lam))
acceptance_budget = n*math.exp(-gmin*q+amax**2/(2*lam))
out = {"exact_saturation_tail": {
    "all_positive_prefix_probability": .999**n,
    "uniform_threshold": threshold,
    "rate_bound_at_7_838": rate,
    "acceptance_budget_at_7_838": acceptance_budget,
    "gaussian_bound_at_7_838": math.exp(acceptance_budget),
    "row_bound_at_7_838": rate*math.exp(acceptance_budget),
    "uniform_gaussian_bound": math.exp(.5),
    "uniform_row_bound": .5*math.exp(.5),
    "TV_1e9_gaussian_cap": math.ceil(math.log(1e9)/(-math.log1p(-math.exp(-.5))))},
    "local_tail_examples": [local_tail(1, 7), local_tail(1000, 40)],
    "local_K_examples": [local_K(7, .0072, 1e-4), local_K(40, 4.5e-5, 1e-4)],
    "scope": "Deterministic bound arithmetic only, not an implementation or learning benchmark."}
assert threshold < 7.838 and rate < .5
assert out["local_K_examples"][0] < .418
assert out["local_K_examples"][1] < .485
Path("TAIL_AND_LOCAL_ROOT_FORMULA_CHECK.json").write_text(json.dumps(out, indent=2)+"\n", encoding="utf-8")
print(json.dumps(out, indent=2))
