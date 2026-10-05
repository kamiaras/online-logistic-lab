"""Deterministic algebra/formula checks; not a learning benchmark."""
import json
import math
from pathlib import Path
import numpy as np

rng = np.random.default_rng(912406)
n, k, lam = 13, 3, .4
a = rng.normal(size=(n, k)) * .025
gamma = rng.normal(size=n)
labels = rng.choice([-1., 1.], size=n)
bp = a.T @ labels / 2
b0 = gamma @ labels / 2
anchor = .37
m = rng.normal(size=k) * .2

def h(z):
    return np.logaddexp(z/2, -z/2) - math.log(2)

z = gamma*anchor + a@m
hp = .5*np.tanh(z/2)
w = .25/np.cosh(z/2)**2
h3 = -w*np.tanh(z/2)
B = lam*np.eye(k) + a.T@(w[:, None]*a)
L = np.linalg.cholesky(B)
grad = lam*m-bp+a.T@hp
cross = a.T@(w*gamma)
slope = -np.linalg.solve(B, cross)
alpha = np.linalg.norm(np.linalg.solve(L, grad))
row_slope = gamma+a@slope
si = np.sum(np.linalg.solve(L, a.T)**2, axis=0)
bar = np.sum(a*a, axis=1)/lam
cs = np.sqrt(bar)
v0 = np.linalg.solve(L, grad)
beta = np.linalg.eigvalsh(np.linalg.solve(L, a.T) @
                         np.linalg.solve(L, a.T).T).max()
def potential(t, v):
    return lam*(t*t+v@v)/2-b0*t-bp@v+h(gamma*t+a@v).sum()

errors = {"raw_density_identity": 0., "constant_free_density_identity": 0.,
          "raw_gram_identity": 0., "constant_free_gram_identity": 0.,
          "mean_polynomial_identity": 0.}
for delta in [-.4, -.03, 0., .02, .3]:
    mean = m+slope*delta
    di = delta*row_slope
    for _ in range(12):
        Y = rng.normal(size=k)
        u = np.linalg.solve(L.T, Y)
        ri = a@u
        def rem(q):
            return h(z+q)-h(z)-hp*q-w*q*q/2
        raw = -v0@Y-rem(di+ri).sum()
        raw0 = -rem(di).sum()
        free = -v0@Y-(rem(di+ri)-rem(di)).sum()
        truth = potential(anchor+delta, mean)-potential(anchor+delta, mean+u)+Y@Y/2
        errors["raw_density_identity"] = max(errors["raw_density_identity"], abs(raw-raw0-truth))
        errors["constant_free_density_identity"] = max(errors["constant_free_density_identity"], abs(free-truth))
        zg = 9.
        eta = abs(di)+zg*cs
        aug = np.column_stack([row_slope, a])
        g1 = aug.T@(abs(row_slope)[:, None]*aug)
        g2 = aug.T@(cs[:, None]*aug)
        vec = np.r_[delta, u]
        raw_b = np.sum(eta*(di+ri)**2)/(36*math.sqrt(3))
        raw_gram = vec@(abs(delta)*g1+zg*g2)@vec/(36*math.sqrt(3))
        errors["raw_gram_identity"] = max(errors["raw_gram_identity"], abs(raw_b-raw_gram))
        q1 = a.T@((row_slope**2/cs)[:, None]*a)
        q2 = a.T@(abs(row_slope)[:, None]*a)
        q3 = a.T@(cs[:, None]*a)
        free_b = np.sum((di**2/(2*cs)+abs(di)+zg*cs/3)*ri**2+di**2*cs/2)/(12*math.sqrt(3))
        free_gram = (u@(delta**2*q1/2+abs(delta)*q2+zg*q3/3)@u + delta**2*np.sum(row_slope**2*cs)/2)/(12*math.sqrt(3))
        errors["constant_free_gram_identity"] = max(errors["constant_free_gram_identity"], abs(free_b-free_gram))
        c_raw = np.sum(h3*(di**3+3*di*si))/6
        c_poly = delta**3*np.sum(h3*row_slope**3)/6+delta*np.sum(h3*row_slope*si)/2
        errors["mean_polynomial_identity"] = max(errors["mean_polynomial_identity"], abs(c_raw-c_poly))

def rejection_constants(U, R0, L0, p=20, ap=40, tol=1e-9, rows=100000):
    charge = tol/4
    log_zminus = -U
    log_z0 = log_zminus+math.log1p(-charge)
    log_up = p*U+p*p*R0/(2*(1-p*L0))
    log_cp = (p-1)*math.log(p-1)-p*math.log(p)
    log_k = (log_cp+log_up+math.e*p*(p-1)/(2*ap)-log_z0-math.log(charge))/(p-1)
    nbar = math.exp(log_k-log_z0-math.log1p(-charge))
    cap = math.ceil(nbar*math.log(1/charge))
    log_q = min((math.log(charge)+log_zminus-log_up/p)*p/(p-1), math.log(charge/cap))
    zg = math.sqrt(2*(math.log(2*rows)-log_q))
    return {"Nbar": nbar, "cap": cap, "zG": zg}

M4, frame, anchor_tol = 2., 1/6, .001
R0 = (anchor_tol+math.sqrt(frame*M4/432))**2
raw_constants = rejection_constants(M4/192, R0, frame/4)
# With D4=1 and M4=2, subtracting row constants removes D4/192.
free_constants = rejection_constants((M4-1)/192, R0, frame/4)
zg = raw_constants["zG"]
mass1 = math.sqrt(2*100)
mass2 = math.sqrt((25/24)*(2/6)*100)
mass3 = math.sqrt(40)
env1 = (mass1+mass2+zg*mass3/3)/(12*math.sqrt(3))
env0 = ((zg*zg+1)*mass1/2+zg*zg*mass2+zg**3*mass3/3)/(12*math.sqrt(3))
row_bound = raw_constants["Nbar"]*(40*env1+20)*env0
target_reference = 500+500*math.log1p(100000*1000**2/(4*1000**2))
target_total = target_reference+100000*math.log1p(2/202)+1
report = {"all_identity_errors_below_1e-12": all(v < 1e-12 for v in errors.values()),
          "identity_errors": errors, "raw_M4_two_constants": raw_constants,
          "constant_free_D4_one_constants": free_constants,
          "constant_free_uniform_row_bound": row_bound,
          "target_aggregation_reference": target_reference,
          "target_complete_regret_bound": target_total,
          "scope": "Small finite algebra checks and deterministic bound evaluation; no learning benchmark."}
assert report["all_identity_errors_below_1e-12"]
Path("AFFINE_CACHE_ROOT_FORMULA_CHECK.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
print(json.dumps(report, indent=2))
