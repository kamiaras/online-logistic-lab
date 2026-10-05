"""Small deterministic identity and analytic ledger checks; no learning benchmark."""
import json
import math
from pathlib import Path
import numpy as np

rng = np.random.default_rng(912509)
n, d = 37, 5
X = rng.normal(size=(n, d))*.08
X[:, 0] += np.linspace(-.15, .35, n)
anchor = X[:3].mean(axis=0)
c = X.mean(axis=0)
s = float(np.linalg.norm(c-anchor))
weights = np.linalg.norm(X-anchor, axis=1)
Z = X-c
W = Z.T@Z
raw_second = X.T@(weights[:, None]*X)
raw_first = X.T@weights
weighted_centered = (raw_second-np.outer(raw_first,c)-np.outer(c,raw_first)
                     +weights.sum()*np.outer(c,c)+s*W)
direct = Z.T@((weights+s)[:, None]*Z)
gram_error = float(np.max(np.abs(direct-weighted_centered)))
true_weight = np.linalg.norm(Z, axis=1)
dom_min = float(np.min(weights+s-true_weight))
inflation_max = float(np.max(weights+s-true_weight))
moment_errors = []
for left, right in [(0, 37), (0, 12), (12, 24), (24, 37)]:
    rows = X[left:right]
    w = weights[left:right]
    masses = ((w[:,None]*rows**2).sum(axis=0)
              -2*c*(w[:,None]*rows).sum(axis=0)+c*c*w.sum()
              +s*((rows**2).sum(axis=0)-2*c*rows.sum(axis=0)+len(rows)*c*c))
    truth = ((w+s)[:,None]*(rows-c)**2).sum(axis=0)
    moment_errors.append(float(np.max(np.abs(masses-truth))))

anova_error = 0.
for m in [1, 3, 11, 26]:
    old = X[:m]
    new = X[m:]
    cm = old.mean(axis=0)
    cn = new.mean(axis=0)
    Wm = (old-cm).T@(old-cm)
    Wnew = (new-cn).T@(new-cn)
    increment = Wnew+(m*n/(n-m))*np.outer(c-cm,c-cm)
    anova_error = max(anova_error, float(np.max(np.abs(W-Wm-increment))))

rho = .045
builds = [1]
a = X[0].copy()
for end in range(2, n+1):
    now = X[:end].mean(axis=0)
    if np.linalg.norm(now-a) > rho:
        builds.append(end)
        a = now
ledger_left = rho**2*sum(builds[:-1])
scatter_trace = float(np.trace(W))
total_build_sizes = sum(builds)
ledger_right = n+scatter_trace/rho**2

T, dimension, block_size = 100000, 1000, 1000
leaves = math.ceil(T/block_size)
nodes = 2*leaves-1
tree_cells = nodes*(4*dimension+2)
out = {
    "weighted_centered_Gram_identity_error": gram_error,
    "all_tree_mass_identity_errors": moment_errors,
    "ANOVA_matrix_identity_error": anova_error,
    "row_weight_domination_min_slack": dom_min,
    "row_weight_inflation_max": inflation_max,
    "permitted_inflation_2s": 2*s,
    "drift_ledger": {"build_indices": builds, "charged_trace": ledger_left,
                      "actual_trace": scatter_trace, "sum_build_sizes": total_build_sizes,
                      "trace_bound": ledger_right},
    "target_analytic_counts": {"block_tree_nodes": nodes, "tree_cells": tree_cells,
        "tree_bytes_at_8_bytes_per_cell": 8*tree_cells,
        "naive_rebuilt_history_rows": T*(T+1)//2,
        "trace_point1_rho_point0005_rebuilt_rows_bound": T+.1/(.0005**2),
        "matrix_gate_rho_point0005_rebuilt_rows_bound": T+dimension*(.001/6)/(.0005**2)},
    "scope": "Algebraic identities and analytic resource bounds; not an online prediction benchmark."}
assert gram_error < 1e-12 and max(moment_errors) < 1e-12 and anova_error < 1e-12
assert dom_min >= -1e-12 and inflation_max <= 2*s+1e-12
assert ledger_left <= scatter_trace+1e-12 and total_build_sizes <= ledger_right+1e-12
Path("ONLINE_SUMMARY_ROOT_FORMULA_CHECK.json").write_text(json.dumps(out, indent=2)+"\n", encoding="utf-8")
print(json.dumps(out, indent=2))
