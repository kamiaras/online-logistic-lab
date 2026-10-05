"""Finite analytic probability/gap ledger; no sampling benchmark."""
import json
import math
from pathlib import Path

d, T, delta, R, lam, budget = 1000, 100000, .001, 1., .001, 202.
noise_square = (delta*R)**2
tau = 1/((T+1)*math.log(budget+1))
inverse_gap = 3e13*d*max(1, R*R/lam)**2
M = 1+d/2*math.log1p(T*R*R/(4*lam))
fallback_events = 1+inverse_gap*(M/2+math.log(1/tau)+budget)
rows = []
for nu in [.00016, .00025, .00030]:
    a = nu*(d+2)/(T*noise_square)
    log_failure = math.log(d)-T/(d+2)*(a*math.log(a)-a+1)
    rho = nu/(4*lam)
    c = 1-math.sqrt(rho/(1+rho))
    rows.append({
        "residual_threshold": nu,
        "noise_threshold_ratio": a,
        "log_all_prefix_failure_bound": log_failure,
        "all_prefix_failure_bound": min(1, math.exp(log_failure)),
        "hessian_comparison": c,
        "coordinate_gap_bound": c/d,
        "expected_fallback_events_over_horizon_upper_bound": T*fallback_events*math.exp(log_failure),
    })
out = {
    "parameters": {"d":d,"T":T,"delta":delta,"R":R,"lambda":lam},
    "mean_noise_Gram_bound": T*noise_square/(d+2),
    "fallback_initializer_log_chi_bound": M,
    "per_forecast_fallback_event_upper_bound": fallback_events,
    "observable_exact_axis_mean_magnitude_threshold": 2*delta*R/(1-delta),
    "table": rows,
}
assert fallback_events < 1.35e26
assert rows[-1]["hessian_comparison"] > .735
assert rows[-1]["all_prefix_failure_bound"] < 1e-53
assert rows[-1]["expected_fallback_events_over_horizon_upper_bound"] < 1e-22
Path("ADAPTIVE_AXIS_COVERAGE_FORMULA_CHECK.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
