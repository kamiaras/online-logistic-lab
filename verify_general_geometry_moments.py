"""Evaluate the finite angular-obstruction proof, not a learning experiment."""
import json
import math
from pathlib import Path

d, n, lam, H0 = 1000, 100000, .001, 25.001
log_event = (math.log(2/math.sqrt(2*math.pi))-.5-31**2/2
             -.5*math.log(2*math.pi)
             +998*(math.log(1.5/math.sqrt(2*math.pi))-.75**2/2))
log_normalizer = math.log(2*1.13)+999*math.log(1.1)
log_moment_lower = 20*(218.7-.014)+log_event-20*log_normalizer
scalar_precision = lam+100*5/24
normalizer_balanced = math.sqrt(H0/scalar_precision)
normalizer_unbalanced = normalizer_balanced*math.exp(1/(2*scalar_precision))
S2 = 999/H0+1/24
S3 = 999/H0**2+1/24**2
out = {"event_radius_squared": 1+31**2+998*.75**2,
       "ball_radius_squared": 40**2,
       "log_Gaussian_event_lower_bound": log_event,
       "log_conditioned_normalizer_upper_bound": log_normalizer,
       "log_normalized_twentieth_moment_lower_bound": log_moment_lower,
       "balanced_scalar_central_normalizer_upper_bound": normalizer_balanced,
       "unbalanced_scalar_central_normalizer_upper_bound": normalizer_unbalanced,
       "weighted_S2_bound": S2, "weighted_S3_bound": S3,
       "weighted_cubic_skew_bound": .0205**2/24,
       "scope": "Deterministic angular-moment obstruction and strict analytic arithmetic. No expected-runtime lower bound or typical-noise claim."}
assert out["event_radius_squared"] < out["ball_radius_squared"]
assert log_moment_lower > 1100 and S2 < 40.01 and S3 < 1.601
assert normalizer_balanced < 1.1 and normalizer_unbalanced < 1.13
Path("GENERAL_GEOMETRY_ROOT_FORMULA_CHECK.json").write_text(json.dumps(out, indent=2)+"\n", encoding="utf-8")
print(json.dumps(out, indent=2))
