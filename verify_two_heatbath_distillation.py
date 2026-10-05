"""Evaluate proved chain and correction budgets; no learning benchmark."""
import json
import math
from pathlib import Path

d, horizon, radius, B, h = 1000, 100000, 1., 1000., 202.
lam = d/B**2
nu = lam/6
rho = nu/(4*lam)
c0 = 1-math.sqrt(rho/(1+rho))
mode_bound = (.5*math.sqrt(horizon*nu) + math.sqrt(horizon*nu/4+4*lam*horizon/3))/(2*lam)
M = -d/2*math.log1p(-rho)+nu*(mode_bound+lam**-.5)**2/(8*(1-rho))
v = 1/((horizon+1)*math.log(h+1))
# log(expm1(M))/2 without overflow.
log_chi = (M+math.log1p(-math.exp(-M)))/2

def schedule(gap, block_error):
    warm = math.ceil(max(0., log_chi-math.log(v))/(-math.log1p(-gap)))
    events = warm+h/gap
    return {"gap": gap, "warm_events": warm, "occupation_expected_events": h/gap,
            "total_expected_events": events, "per_block_expected_calls": events/2,
            "per_call_TV_budget": v/(2*events),
            "used_per_call_TV": block_error,
            "whole_path_TV_bound": v/2+events*block_error}

def affine_constants(tol):
    p, ap, U, R0, L0 = 20, 40, 1/96, (.001+1/36)**2, 1/24
    charge = tol/4
    log_z0 = -U+math.log1p(-charge)
    log_up = p*U+p*p*R0/(2*(1-p*L0))
    log_cp = (p-1)*math.log(p-1)-p*math.log(p)
    log_K = (log_cp+log_up+math.e*p*(p-1)/(2*ap)-log_z0-math.log(charge))/(p-1)
    Nbar = math.exp(log_K-log_z0-math.log1p(-charge))
    cap = math.ceil(Nbar*math.log(1/charge))
    log_q = min((math.log(charge)-U-log_up/p)*p/(p-1), math.log(charge/cap))
    z = math.sqrt(2*(math.log(2*horizon)-log_q))
    mass1, mass2, mass3 = math.sqrt(200), math.sqrt((25/24)*(2/6)*100), math.sqrt(40)
    B1 = (mass1+mass2+z*mass3/3)/(12*math.sqrt(3))
    B0 = ((z*z+1)*mass1/2+z*z*mass2+z**3*mass3/3)/(12*math.sqrt(3))
    return {"TV": tol, "gaussian_expected_bound": Nbar, "gaussian_cap": cap,
            "zG": z, "B1": B1, "B0": B0,
            "row_expected_bound": Nbar*(ap*B1+p)*B0}

new_schedule = schedule(c0/2, 5e-10)
scalar_K = .467777141462
scalar_cap = math.ceil(math.exp(2*scalar_K)*math.log(4e9))
reference = d/2+d/2*math.log1p(horizon*(B*radius)**2/(4*d*d))
result = {"M": M, "whole_path_TV_allocation": v, "equal_heatbaths": new_schedule,
          "affine_at_new_tolerance": affine_constants(5e-10),
          "scalar_at_new_tolerance": {"proposal_cap": scalar_cap,
                    "exhaustion_TV_upper_bound": math.exp(-scalar_cap*math.exp(-2*scalar_K)),
                    "original_tail_TV_upper_bound": 1.57e-26},
          "exact_tail_at_new_tolerance": {"proposal_cap": math.ceil(math.log(2e9)/(-math.log1p(-math.exp(-.5))))},
          "aggregation_reference": reference,
          "new_regret_bound": reference+horizon*math.log1p(2/h)+1,
          "scope": "Conditional proved schedules and deterministic arithmetic, not observed runtime or a full T^2 theorem."}
assert new_schedule["whole_path_TV_bound"] <= v
assert new_schedule["per_call_TV_budget"] >= 5e-10
assert result["scalar_at_new_tolerance"]["exhaustion_TV_upper_bound"]+1.57e-26 < 5e-10
Path("TWO_HEATBATH_ROOT_FORMULA_CHECK.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
print(json.dumps(result, indent=2))
