"""Offline streamed Newton refinement of cached low-dimensional comparators.

Run only after measured campaigns and ``experiments.analyze deep_long`` finish.
The objective is the same unregularized logistic sum on ||theta||<=B. Newton
steps are an evaluation aid, not a change to any online algorithm. The reported
first-order bound is calculated in float64, not certified interval arithmetic.
"""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve, norm
from threadpoolctl import threadpool_limits

from .common import sigmoid

BASE = Path(__file__).resolve().parent


class _Deadline(Exception):
    pass


def streamed_evaluate(X, y, theta, deadline=math.inf, stats=None):
    """Value, gradient and Hessian; allocate at most one bounded data block."""
    n, d = X.shape
    batch_size = max(1, min(8192, 1_048_576 // d))
    value, gradient, hessian = 0.0, np.zeros(d), np.zeros((d, d))
    for lo in range(0, n, batch_size):
        if perf_counter() >= deadline:
            raise _Deadline
        hi = min(n, lo + batch_size)
        block = np.asarray(X[lo:hi], dtype=np.float64)
        labels = np.asarray(y[lo:hi], dtype=np.float64)
        if not np.isfinite(block).all() or not np.all((labels == -1) | (labels == 1)):
            raise ValueError("Nonfinite features or labels outside {-1,+1}")
        score = block @ theta
        signed = -labels * score
        value += float(np.logaddexp(0.0, signed).sum())
        gradient += block.T @ (-labels * sigmoid(signed))
        # Using the smaller sigmoid tail retains curvature for positive scores.
        tail = sigmoid(-np.abs(score))
        curvature = tail * (1.0 - tail)
        hessian += block.T @ (curvature[:, None] * block)
        if stats is not None:
            stats["history_rows_evaluated"] += hi - lo
    if not math.isfinite(value) or not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise FloatingPointError("Nonfinite Newton comparator evaluation")
    if stats is not None:
        stats["evaluations"] += 1
    return value, gradient, (hessian + hessian.T) / 2.0


def _gap(theta, gradient, B):
    raw = float(gradient @ theta + B * norm(gradient, check_finite=False))
    if not math.isfinite(raw):
        raise FloatingPointError("Nonfinite comparator first-order gap")
    return max(0.0, raw), raw


def _improves(candidate, old):
    return (candidate["loss"] < old["loss"] or
            (candidate["loss"] == old["loss"] and candidate["gap"] < old["gap"]))


def refine_fit(X, y, B, old, max_iterations=15, max_seconds=120.0, tolerance=1e-4):
    """Return (candidate-or-None, audit), leaving the old dictionary untouched.

    A timeout returns only a completely evaluated feasible point. Boundary
    obstructions, nonpositive Hessians and numerical solver failures defer to
    the old cache. A boundary-limited search must reach the gap tolerance before
    its candidate is eligible, since this routine is intended for interior fits.
    """
    start = perf_counter()
    if X.ndim != 2 or not 1 <= X.shape[1] <= 100 or y.shape != (X.shape[0],):
        raise ValueError("Expected X=(n,d), y=(n,), with 1<=d<=100")
    if not 0 <= max_iterations <= 15 or int(max_iterations) != max_iterations:
        raise ValueError("max_iterations must be an integer between 0 and 15")
    if not math.isfinite(max_seconds) or not 0 <= max_seconds <= 120:
        raise ValueError("max_seconds must lie in [0,120]")
    if not math.isfinite(B) or B <= 0 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid radius or tolerance")
    theta = np.array(old["theta"], dtype=np.float64, copy=True)
    if theta.shape != (X.shape[1],) or not np.isfinite(theta).all():
        raise ValueError("Invalid cached comparator vector")
    deadline = start + max_seconds
    margin_radius = B * (1.0 - 64.0 * np.finfo(float).eps)
    stats = dict(iterations=0, evaluations=0, history_rows_evaluated=0,
                 backtracks=0, boundary_backtracks=0)
    status, best, boundary_limited, error = "iteration_limit", None, False, None
    initial_evaluation = None
    try:
        if float(norm(theta, check_finite=False)) >= margin_radius:
            status = "boundary_deferred"
        else:
            value, gradient, hessian = streamed_evaluate(X, y, theta, deadline, stats)
            gap, raw = _gap(theta, gradient, B)
            initial_evaluation = dict(loss=value, gap=gap, raw_gap=raw)
            best = (theta.copy(), value, gradient.copy(), gap, raw)
            for iteration in range(int(max_iterations)):
                if gap <= tolerance:
                    status = "converged"
                    break
                if perf_counter() >= deadline:
                    raise _Deadline
                # No ridge, damping penalty, or change to the comparator objective.
                direction = -cho_solve(cho_factor(hessian, lower=True, check_finite=False),
                                       gradient, check_finite=False)
                descent = float(gradient @ direction)
                if not np.isfinite(direction).all() or not math.isfinite(descent) or descent >= 0:
                    status, best = "newton_direction_failure", None
                    break
                alpha, accepted = 1.0, False
                for _ in range(40):
                    if perf_counter() >= deadline:
                        raise _Deadline
                    trial = theta + alpha * direction
                    if (not np.isfinite(trial).all()
                            or float(norm(trial, check_finite=False)) >= margin_radius):
                        boundary_limited = True
                        stats["boundary_backtracks"] += 1
                    elif np.array_equal(trial, theta):
                        status = "numerical_stall"
                        break
                    else:
                        new_value, new_gradient, new_hessian = streamed_evaluate(
                            X, y, trial, deadline, stats)
                        new_gap, new_raw = _gap(trial, new_gradient, B)
                        slack = 32.0 * np.finfo(float).eps * max(1.0, abs(value), abs(new_value))
                        armijo = new_value <= value + 1e-4 * alpha * descent + slack
                        monotone = (new_value < value or
                                    (new_value == value and new_gap < gap))
                        if armijo and monotone:
                            theta, value, gradient, hessian = trial, new_value, new_gradient, new_hessian
                            gap, raw = new_gap, new_raw
                            best = (theta.copy(), value, gradient.copy(), gap, raw)
                            stats["iterations"] = iteration + 1
                            accepted = True
                            break
                    stats["backtracks"] += 1
                    alpha *= 0.5
                if not accepted:
                    if status != "numerical_stall":
                        status = "line_search_stalled"
                    break
            if best is not None and best[3] <= tolerance:
                status = "converged"
    except _Deadline:
        status = "time_limit"
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        status, best, error = "numerical_failure_deferred", None, repr(exc)
    elapsed = perf_counter() - start
    if boundary_limited and status != "converged":
        status, best = "boundary_deferred", None
    audit = dict(status=status, elapsed=elapsed, max_seconds=max_seconds,
                 max_iterations=max_iterations, tolerance=tolerance,
                 time_limit_is_soft=True, boundary_limited=boundary_limited,
                 initial_reevaluation=initial_evaluation, error=error, **stats)
    if best is None:
        return None, audit
    theta, value, gradient, gap, raw = best
    candidate = dict(theta=theta.tolist(), loss=float(value), gap=float(gap), raw_gap=float(raw),
                     loss_lower_bound=float(value-gap),
                     elapsed=float(old.get("elapsed", 0.0) + elapsed),
                     refinement_elapsed=elapsed, previous_elapsed=old.get("elapsed", 0.0),
                     converged=bool(gap <= tolerance), status=status,
                     timed_out=status == "time_limit", time_limit_is_soft=True,
                     max_seconds=max_seconds, tolerance=tolerance,
                     theta_norm=float(norm(theta, check_finite=False)),
                     gradient_norm=float(norm(gradient, check_finite=False)),
                     batch_size=max(1, min(8192, 1_048_576 // X.shape[1])),
                     solver="streamed_interior_newton_refinement",
                     certificate="numerical_first_order_gap", finite_precision_certified=False,
                     gap_clipped_for_rounding=bool(raw < 0), **stats)
    return candidate, audit


def _no_measured_workers():
    """A guard against ordinary campaign overlap; scheduling is still sequential."""
    import psutil
    modules = {"experiments.worker", "experiments.runner", "experiments.memory_frontier",
               "experiments.history_frontier"}
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if modules.intersection(proc.info["cmdline"] or []):
                found.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if found:
        raise RuntimeError("Measured campaign processes are active; defer offline refinement: " + str(found))


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".newton.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def refine_stage(stage="deep_long", max_iterations=15, max_seconds=120.0, tolerance=1e-4):
    _no_measured_workers()
    folder = BASE / "results" / stage
    if not (folder / "summary.json").exists() or not (folder / "curves.json").exists():
        raise RuntimeError("Run the ordinary stage analysis first to establish the comparator cache")
    specs = json.loads((folder / "manifest.json").read_text())
    if any(not (Path(s["output"]) / "supervisor.json").exists() for s in specs):
        raise RuntimeError("Complete every measured stage run before comparator refinement")
    by_case = {}
    for spec in specs:
        by_case.setdefault(spec["case_id"], spec)
    audit_path = folder / "newton_refinement_audit.json"
    archive = json.loads(audit_path.read_text()) if audit_path.exists() else {
        "purpose": "Offline comparator evaluation only; online results and algorithms are unchanged.",
        "certificate": "G=gradient@theta+B*||gradient|| is a real-arithmetic convexity bound, evaluated here in float64.",
        "finite_precision_certified": False, "attempts": []}
    with threadpool_limits(limits=1):
        for cid, spec in sorted(by_case.items()):
            if spec["d"] > 100:
                continue
            cache_path = folder / (cid + "__comparators.json")
            if not cache_path.exists():
                raise RuntimeError("Missing established cache: " + str(cache_path))
            cache = json.loads(cache_path.read_text())
            endpoints = sorted(int(n) for n in cache if int(n) > 100000)
            if not endpoints:
                continue
            X = np.memmap(Path(spec["data_path"]) / "X.bin", dtype="<f8", mode="r",
                          shape=(spec["T"], spec["d"]))
            y = np.memmap(Path(spec["data_path"]) / "y.bin", dtype="i1", mode="r", shape=(spec["T"],))
            for n in endpoints:
                _no_measured_workers()
                old = cache[str(n)]
                candidate, detail = refine_fit(X[:n], y[:n], float(spec["B"]), old,
                                              max_iterations, max_seconds, tolerance)
                accepted = candidate is not None and _improves(candidate, old)
                event = dict(case_id=cid, round=n, accepted=accepted, cache_replaced=False, old=old,
                             candidate=candidate, refinement=detail)
                archive["attempts"].append(event)
                # Save provenance before mutating the cache. Interrupted propagation
                # is recoverable by rerunning the ordinary analysis.
                _atomic_json(audit_path, archive)
                if accepted:
                    cache[str(n)] = candidate
                    _atomic_json(cache_path, cache)
                    event["cache_replaced"] = True
                    _atomic_json(audit_path, archive)
                print(json.dumps(dict(case_id=cid, round=n, accepted=accepted,
                                      old_gap=old["gap"], new_gap=candidate["gap"] if candidate else None,
                                      elapsed=detail["elapsed"], status=detail["status"])), flush=True)
            del X, y
    _no_measured_workers()
    # Zero requested fitting time makes the existing analyzer reuse all its
    # established cache entries, propagating revised loss/gap values to reports.
    from .analyze import analyze
    analyze(stage, comparator_seconds=0.0)
    return archive


def self_test():
    """Tiny mathematical checks, with no campaign reads, writes, or large fits."""
    start = perf_counter()
    X = np.array([[.3, -.7], [.8, .2], [-.4, .5], [.1, .9], [-.6, -.2]])
    y, theta = np.array([1, -1, 1, 1, -1]), np.array([.2, -.4])
    value, gradient, hessian = streamed_evaluate(X, y, theta)
    epsilon = 1e-5
    numerical_gradient, numerical_hessian = np.zeros(2), np.zeros((2, 2))
    for j in range(2):
        shift = np.eye(2)[j] * epsilon
        plus, gp, _ = streamed_evaluate(X, y, theta + shift)
        minus, gm, _ = streamed_evaluate(X, y, theta - shift)
        numerical_gradient[j] = (plus - minus) / (2 * epsilon)
        numerical_hessian[:, j] = (gp - gm) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(hessian, numerical_hessian, rtol=1e-8, atol=1e-9)
    # Cross an actual 8192-row boundary without constructing a large problem.
    repeated = streamed_evaluate(np.tile(X, (1641, 1)), np.tile(y, 1641), theta)
    for actual, expected in zip(repeated, (value*1641, gradient*1641, hessian*1641)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
    # Three positive and one negative identical feature: optimum theta=log(3).
    xx, yy = np.ones((4, 1)), np.array([1, 1, 1, -1])
    old = dict(theta=[0.0], loss=4*math.log(2), gap=5.0, elapsed=0.0)
    candidate, _ = refine_fit(xx, yy, 5.0, old, max_seconds=.2, tolerance=1e-10)
    assert candidate is not None and _improves(candidate, old)
    np.testing.assert_allclose(candidate["theta"], [math.log(3)], rtol=0, atol=1e-8)
    optimum = 4*math.log(4) - 3*math.log(3)
    assert candidate["loss"] - candidate["gap"] <= optimum + 1e-12
    assert optimum <= candidate["loss"] + 1e-12
    # Explicitly constrained optima are deferred, not mistaken for interior ones.
    constrained = dict(theta=[.1], loss=1.0, gap=1.0, elapsed=0.0)
    failed, audit = refine_fit(xx, yy, .1, constrained, max_seconds=.2)
    assert failed is None and audit["status"] == "boundary_deferred"
    untouched, audit = refine_fit(xx, yy, 5., old, max_seconds=0.)
    assert untouched is None and audit["status"] == "time_limit"
    assert old == dict(theta=[0.0], loss=4*math.log(2), gap=5.0, elapsed=0.0)
    assert not _improves(dict(loss=old["loss"]+1., gap=0.), old)
    assert _improves(dict(loss=old["loss"], gap=old["gap"]-1.), old)
    return dict(passed=True, elapsed=perf_counter()-start,
                checks=["finite-difference gradient", "finite-difference Hessian", "streamed block boundary", "known interior optimum",
                        "first-order bound", "boundary fallback", "deadline", "cache acceptance"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="deep_long")
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--max-seconds", type=float, default=120.)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        with threadpool_limits(limits=1):
            print(json.dumps(self_test()))
    else:
        refine_stage(args.stage, args.max_iterations, args.max_seconds, args.tolerance)


if __name__ == "__main__":
    main()
