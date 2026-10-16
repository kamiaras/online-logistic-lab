"""Time-limited convex comparator fitting on the Euclidean norm ball.

The reported first-order gap is evaluated in float64, not interval arithmetic.
No ridge or other penalty is added to the comparator's cumulative logistic loss.
"""
import math
from time import perf_counter

import numpy as np
from scipy.linalg import norm

from .common import sigmoid


class _TimeLimit(Exception):
    pass


def _project_ball(theta, B):
    theta = np.array(theta, dtype=np.float64, copy=True)
    radius = float(norm(theta, check_finite=False))
    if radius > B:
        if B == 0:
            theta.fill(0.0)
        else:
            # Leave a small numerical feasibility margin after projection.
            theta *= (B / radius) * (1.0 - 4.0 * np.finfo(float).eps)
    return theta


def fit_comparator(X, y, B, initial=None, tolerance=1e-2, max_seconds=10):
    """Fit the unregularized cumulative logistic loss subject to ||theta||<=B.

    Projected accelerated gradient with backtracking and adaptive restart uses
    bounded-size full-history batches. The deadline is cooperative: an initial
    complete evaluation is mandatory for a valid returned certificate, and an
    individual BLAS call cannot be interrupted. A timeout returns the best
    completely evaluated feasible point, never a partial objective.
    """
    start = perf_counter()
    B, tolerance, max_seconds = float(B), float(tolerance), float(max_seconds)
    if not math.isfinite(B) or B < 0:
        raise ValueError("B must be nonnegative and finite")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative and finite")
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError("max_seconds must be nonnegative and finite")
    X, y = np.asarray(X), np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 1 or y.shape != (X.shape[0],):
        raise ValueError("X must have shape (n,d), y shape (n,), with d>=1")
    if not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        raise ValueError("X must contain real numbers")
    if not np.all((y == -1) | (y == 1)):
        raise ValueError("y must contain only -1 and 1")
    n, d = X.shape
    if initial is None:
        theta = np.zeros(d)
    else:
        initial = np.asarray(initial, dtype=np.float64)
        if initial.shape != (d,) or not np.isfinite(initial).all():
            raise ValueError("initial must be a finite vector of length d")
        theta = _project_ball(initial, B)
    deadline = start + max_seconds
    # Bound input temporaries even for very wide matrices or memory-mapped data.
    batch_size = max(1, min(8192, 1_048_576 // d))
    evaluations = rows_evaluated = iterations = restarts = backtracks = 0
    estimated_flops = 0

    def evaluate(u, enforce_deadline=True):
        nonlocal evaluations, rows_evaluated, estimated_flops
        value = 0.0
        gradient = np.zeros(d)
        for lo in range(0, n, batch_size):
            if enforce_deadline and perf_counter() >= deadline:
                raise _TimeLimit
            hi = min(n, lo + batch_size)
            block, labels = X[lo:hi], y[lo:hi]
            signed = -labels * (block @ u)
            value += float(np.logaddexp(0.0, signed).sum())
            gradient += block.T @ (-labels * sigmoid(signed))
            rows_evaluated += hi - lo
            estimated_flops += 4 * (hi - lo) * d + 8 * (hi - lo)
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            raise ValueError("Non-finite comparator evaluation; check input magnitudes")
        evaluations += 1
        return value, gradient

    def gap_of(u, gradient):
        raw = float(gradient @ u + B * norm(gradient, check_finite=False))
        return max(0.0, raw), raw

    value, gradient = evaluate(theta, enforce_deadline=False)
    gap, raw_gap = gap_of(theta, gradient)
    best = (value, theta.copy(), gradient.copy(), gap, raw_gap)
    z, z_value, z_gradient = theta.copy(), value, gradient.copy()
    acceleration = 1.0
    lipschitz = max(float(norm(gradient, check_finite=False)) / max(B, 1e-15), 1e-12)
    status = "converged" if gap <= tolerance else "iteration_limit"
    if gap > tolerance:
        try:
            for iteration in range(10000):
                if perf_counter() >= deadline:
                    raise _TimeLimit
                trial_lipschitz = max(lipschitz / 1.25, 1e-15)
                for _ in range(80):
                    candidate = _project_ball(z - z_gradient / trial_lipschitz, B)
                    trial_value, trial_gradient = evaluate(candidate)
                    change = candidate - z
                    quadratic_bound = (z_value + z_gradient @ change
                                       + 0.5 * trial_lipschitz * (change @ change))
                    slack = 32 * np.finfo(float).eps * max(1.0, abs(z_value),
                                                         abs(trial_value))
                    if trial_value <= quadratic_bound + slack:
                        break
                    trial_lipschitz *= 2.0
                    backtracks += 1
                else:
                    status = "line_search_stalled"
                    break
                iterations = iteration + 1
                trial_gap, trial_raw_gap = gap_of(candidate, trial_gradient)
                if trial_value < best[0] or (trial_value == best[0] and trial_gap < best[3]):
                    best = (trial_value, candidate.copy(), trial_gradient.copy(),
                            trial_gap, trial_raw_gap)
                if trial_gap <= tolerance:
                    best = (trial_value, candidate.copy(), trial_gradient.copy(),
                            trial_gap, trial_raw_gap)
                    status = "converged"
                    break
                if np.array_equal(candidate, theta):
                    status = "numerical_stall"
                    break
                previous = theta
                theta, value, gradient = candidate, trial_value, trial_gradient
                lipschitz = trial_lipschitz
                # Gradient-mapping restart: discard momentum when it points back.
                if float((z - candidate) @ (candidate - previous)) > 0.0:
                    acceleration = 1.0
                    z, z_value, z_gradient = theta.copy(), value, gradient.copy()
                    restarts += 1
                else:
                    next_acceleration = (1.0 + math.sqrt(1.0 + 4.0 * acceleration**2)) / 2.0
                    momentum = (acceleration - 1.0) / next_acceleration
                    acceleration = next_acceleration
                    if momentum == 0.0:
                        z, z_value, z_gradient = theta.copy(), value, gradient.copy()
                    else:
                        z = theta + momentum * (theta - previous)
                        z_value, z_gradient = evaluate(z)
        except _TimeLimit:
            status = "time_limit"
    value, theta, gradient, gap, raw_gap = best
    elapsed = perf_counter() - start
    return {
        "theta": theta,
        "loss": float(value),
        "gap": float(gap),
        "raw_gap": float(raw_gap),
        "loss_lower_bound": float(value - gap),
        "elapsed": float(elapsed),
        "converged": bool(gap <= tolerance),
        "status": status,
        "timed_out": status == "time_limit",
        "time_limit_is_soft": True,
        "max_seconds": max_seconds,
        "tolerance": tolerance,
        "theta_norm": float(norm(theta, check_finite=False)),
        "gradient_norm": float(norm(gradient, check_finite=False)),
        "iterations": iterations,
        "evaluations": evaluations,
        "history_rows_evaluated": rows_evaluated,
        "estimated_flops": estimated_flops,
        "restarts": restarts,
        "backtracks": backtracks,
        "batch_size": batch_size,
        "solver": "projected_accelerated_gradient",
        "certificate": "numerical_first_order_gap",
        "finite_precision_certified": False,
        "gap_clipped_for_rounding": bool(raw_gap < 0),
    }
