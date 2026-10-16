"""Fixed-capture Poisson/full acceptance replay; never edits historical code.

Run only after shared_ablation_campaign and its workers have exited:
  python -m experiments.shared_acceptance_replay --max-captures 24 --repeats 20

A gated child has one BLAS thread and a hard 2 GiB Windows commit limit.
Setup and R/Q/C/M diagnostics are outside acceptance timing. Both methods see
the same frozen history, tangent and proposal, but use independent test streams.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from time import perf_counter, sleep
import traceback

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments/results/shared_ablation"
DEFAULT_OUTPUT = ROOT / "experiments/results/shared_acceptance_replay.json"
HISTORICAL = [
    "experiments/shared_tangent.py",
    "experiments/shared_tangent_primitives.py",
    "experiments/proximal.py",
    "experiments/common.py",
    "experiments/data.py",
    "experiments/sigmoid.py",
    "experiments/aioli.py",
]
SOURCES = HISTORICAL + [
    "experiments/shared_ablation.py",
    "experiments/shared_ablation_worker.py",
    "experiments/shared_ablation_campaign.py",
    "experiments/shared_acceptance_replay.py",
    "experiments/memory.py",
    "SHARED_ABLATION_PLAN.md",
    "INTEGRATED_LOG_ODDS_COMPLETE.md",
]
LIMIT_BYTES = 2 * 1024**3


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes():
    return {name: _hash(ROOT / name) for name in SOURCES}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _active_campaigns():
    import psutil

    active = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if process.pid != os.getpid() and any(
                name in command
                for name in (
                    "experiments.shared_ablation_worker",
                    "experiments.shared_ablation_campaign",
                )
            ):
                active.append({"pid": process.pid, "command": command})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return active


def _inventory(directory):
    import numpy as np

    entries, rejected = [], []
    for capture_path in sorted(Path(directory).glob("*/captures.npz")):
        spec_path = capture_path.parent / "spec.json"
        result_path = capture_path.parent / "result.json"
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
            if result.get("status") not in ("completed", "time_budget"):
                rejected.append({"path": str(capture_path), "reason": "source run not completed/time_budget"})
                continue
            with np.load(capture_path, allow_pickle=False) as saved:
                times = saved["t"]
                required = ("v", "z", "g", "r")
                if any(saved[key].shape != (len(times), spec["d"]) for key in required):
                    raise ValueError("capture vector dimensions do not match its source spec")
                for index, time in enumerate(times):
                    if int(time) != time or not 1 <= int(time) <= spec["T"]:
                        raise ValueError("invalid capture round")
                    entries.append({
                        "capture_path": str(capture_path.resolve()),
                        "capture_index": index,
                        "t": int(time),
                        "spec_path": str(spec_path.resolve()),
                        "source_result_path": str(result_path.resolve()),
                        "source_status": result["status"],
                        "source_completed_rounds": result.get("round"),
                        "case": spec["case"],
                        "d": spec["d"],
                        "B": spec["B"],
                        "seed": spec["seed"],
                        "layout": spec["config"].get("layout", "sweep"),
                        "persistence": spec["config"].get("persistence", "fresh"),
                        "acceptance": spec["config"].get("acceptance", "full"),
                    })
        except Exception as exc:
            rejected.append({"path": str(capture_path), "reason": repr(exc)})
    entries.sort(key=lambda row: (
        row["case"], row["B"], row["layout"], row["persistence"],
        row["acceptance"], row["seed"], row["t"], row["capture_index"],
    ))
    return entries, rejected


def _select(entries, count):
    if count == 0 or count >= len(entries):
        return entries
    if count == 1:
        return [entries[len(entries) // 2]]
    return [entries[index * (len(entries) - 1) // (count - 1)] for index in range(count)]


def _ratio(numerator, denominator):
    if denominator == 0:
        return None, "zero_denominator"
    value = numerator / denominator
    return (value, "finite") if math.isfinite(value) else (None, "overflow")


def _reset_metrics(algorithm):
    from .common import Metrics

    algorithm.metrics = Metrics()
    algorithm.history.metrics = algorithm.metrics


def _prepare(entry, file_hash_cache):
    import numpy as np
    from .shared_ablation import SharedAblation

    started = perf_counter()
    source_spec = json.loads(Path(entry["spec_path"]).read_text(encoding="utf-8-sig"))
    d, t = entry["d"], entry["t"]
    feature_path = Path(source_spec["data_path"]) / "X.bin"
    raw = np.fromfile(feature_path, dtype="<f8", count=t * d)
    if raw.size != t * d:
        raise ValueError("source feature file is shorter than captured history")
    features = raw.reshape(t, d)
    with np.load(entry["capture_path"], allow_pickle=False) as saved:
        index = entry["capture_index"]
        vectors = {key: saved[key][index].copy() for key in ("v", "z", "g", "r")}
    if not all(np.all(np.isfinite(value)) for value in vectors.values()):
        raise ValueError("nonfinite capture vector")

    config = dict(source_spec["config"])
    config.update(mode="practical", capture_checkpoints=[], acceptance="full")
    algorithm = SharedAblation(d, source_spec["B"], source_spec["T"], 0, config)
    # Reproduce rowwise energy accumulation and prefix construction, outside timing.
    for feature in features:
        algorithm._append_feature(feature)
    schedule = algorithm._schedule()
    algorithm.v[:] = vectors["v"]
    algorithm.g[:] = vectors["g"]
    algorithm._current_r = vectors["r"]
    if np.any(np.abs(algorithm.v) > algorithm.bound) or np.any(np.abs(vectors["z"]) > algorithm.bound):
        raise ValueError("capture lies outside the sampler box")
    algorithm.phi = 0.0
    for block in algorithm.history.batches():
        score = block @ algorithm.v
        algorithm.phi += float(np.logaddexp(score / 2, -score / 2).sum())
    setup_seconds = perf_counter() - started

    started = perf_counter()
    residual = algorithm._full_residual(vectors["z"])
    difference = vectors["z"] - algorithm.v
    # Match the archived test's summation order for its actual floating C.
    majorant = float(np.cumsum(algorithm.history.W * np.square(difference))[-1]) / 8.0
    quadratic, gradient_reference = 0.0, np.zeros(d)
    for block in algorithm.history.batches():
        delta = block @ difference
        quadratic += float(delta @ delta) / 8.0
        old_score = block @ algorithm.v
        gradient_reference += block.T @ (0.5 * np.tanh(old_score / 2.0))
    mu = schedule["mu"]
    force = mu * algorithm.v - vectors["r"] + algorithm.g
    normal_residual = force.copy()
    upper = algorithm.v == algorithm.bound
    lower = algorithm.v == -algorithm.bound
    normal_residual[upper] = np.maximum(force[upper], 0.0)
    normal_residual[lower] = np.minimum(force[lower], 0.0)
    normalized = np.abs(normal_residual) / math.sqrt(mu)
    with np.errstate(over="ignore", invalid="ignore"):
        moment_factor = (normalized + np.hypot(normalized, 2.0)) / 2.0
        M = float(algorithm.history.W @ np.square(moment_factor)) / (8.0 * mu)
    if not all(math.isfinite(value) and value >= 0 for value in (residual, quadratic, majorant)):
        raise ValueError("invalid R/Q/C diagnostic")
    c_over_r, cr_status = _ratio(majorant, residual)
    c_over_q, cq_status = _ratio(majorant, quadratic)
    q_over_r, qr_status = _ratio(quadratic, residual)
    norm_ratio = float(np.linalg.norm(normal_residual)) / math.sqrt(mu)
    max_ratio = float(np.max(normalized))
    if not all(math.isfinite(value) for value in (norm_ratio, max_ratio)):
        raise ValueError("normal-residual diagnostic is not representable")
    query_overhead = t.bit_length() + 4
    mark_cap = max(1, (t * d + d + query_overhead - 1) // (d + query_overhead))
    diagnostics = {
        "R": residual,
        "Q": quadratic,
        "C": majorant,
        "archived_mark_cap": mark_cap,
        "C_minus_R": majorant - residual,
        "C_over_R": c_over_r,
        "C_over_R_status": cr_status,
        "C_over_Q": c_over_q,
        "C_over_Q_status": cq_status,
        "Q_over_R": q_over_r,
        "Q_over_R_status": qr_status,
        "acceptance_probability_float64": math.exp(-residual),
        "normal_residual_l2_over_sqrt_mu": norm_ratio,
        "normal_residual_linf_over_sqrt_mu": max_ratio,
        "certified_by_l2_sufficient_condition": norm_ratio <= 1.0 / 3.0,
        "certified_by_coordinate_condition": max_ratio <= 1.0 / 3.0,
        "M": M if math.isfinite(M) else None,
        "M_nonfinite": not math.isfinite(M),
        "M_is_proposal_mean_bound_not_pointwise": True,
        "mu": mu,
        "E": schedule["E"],
        "E1": schedule["E1"],
        "P": schedule["P"],
        "displacement_l2": float(np.linalg.norm(difference)),
        "cached_gradient_recompute_max_abs_difference": float(np.max(np.abs(
            gradient_reference - algorithm.g))),
        "uncapped_expected_processed_marks_at_fixed_proposal": (
            majorant if residual == 0 else majorant * ((-math.expm1(-residual)) / residual)),
        "diagnostic_seconds": perf_counter() - started,
    }
    hash_paths = [feature_path, Path(entry["capture_path"]), Path(entry["spec_path"])]
    hashes = {}
    for path in hash_paths:
        name = str(path.resolve())
        if name not in file_hash_cache:
            file_hash_cache[name] = _hash(path)
        hashes[name] = file_hash_cache[name]
    _reset_metrics(algorithm)
    return algorithm, vectors["z"], diagnostics, setup_seconds, hashes


def _measure(algorithm, proposal, method, repeats, warmup, seed, deadline):
    import numpy as np

    algorithm.acceptance = method
    algorithm.rng = np.random.default_rng(seed)
    warmed = 0
    try:
        for _ in range(warmup):
            if perf_counter() >= deadline:
                return {"status": "time_budget", "completed_repeats": 0, "warmup_completed": warmed}
            algorithm._residual_test(proposal)
            warmed += 1
    except Exception as exc:
        return {"status": "warmup_error", "error": repr(exc), "completed_repeats": 0,
                "warmup_completed": warmed}
    _reset_metrics(algorithm)
    timings, accepted = [], 0
    error = None
    loop_start = perf_counter()
    for _ in range(repeats):
        if perf_counter() >= deadline:
            break
        call_start = perf_counter()
        try:
            outcome = algorithm._residual_test(proposal)
        except Exception as exc:
            error = repr(exc)
            break
        timings.append(perf_counter() - call_start)
        accepted += int(outcome)
    elapsed = perf_counter() - loop_start
    count = len(timings)
    status = "error" if error else ("completed" if count == repeats else "time_budget")
    return {
        "status": status,
        "error": error,
        "seed": seed,
        "requested_repeats": repeats,
        "completed_repeats": count,
        "warmup_completed": warmed,
        "accepts": accepted,
        "acceptance_fraction": accepted / count if count else None,
        "timed_loop_seconds": elapsed,
        "sum_call_seconds": sum(timings),
        "mean_call_seconds": statistics.mean(timings) if timings else None,
        "median_call_seconds": statistics.median(timings) if timings else None,
        "call_seconds": timings,
        "metrics": algorithm.metrics.as_dict(),
    }


def _run_worker(spec):
    # This import and all numerical work occur after the supervisor gate.
    import numpy as np
    from threadpoolctl import threadpool_limits, threadpool_info
    from .memory import memory_snapshot

    started = perf_counter()
    deadline = started + spec["seconds"]
    report = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "fixed-capture acceptance replay, separate from online timings",
        "configuration": spec,
        "captures": [],
        "floating_point_certificate": False,
        "sampling_note": "First two proposals at selected source rounds; deterministic inventory selection, not representative random proposals.",
        "comparison_note": "Same frozen history/v/z/g/r; distinct acceptance streams. Equal kernels do not imply identical decisions or same-seed online paths.",
        "timing_note": "Setup, diagnostics and input hashing are excluded from acceptance timing. Both tests retain archived instrumentation. Timer scopes are nested; fallback_rows counts all full-scan rows even for the direct method.",
        "source_hashes_before": _source_hashes(),
    }
    file_hash_cache = {}
    with threadpool_limits(limits=1):
        try:
            for capture_number, entry in enumerate(spec["selected"]):
                if perf_counter() >= deadline:
                    report["status"] = "time_budget"
                    break
                row = dict(entry)
                try:
                    algorithm, proposal, diag, setup, hashes = _prepare(entry, file_hash_cache)
                    row.update(diagnostics=diag, setup_seconds=setup, input_hashes=hashes)
                    order = ["poisson", "full"]
                    random.Random(spec["seed"] + 31 * capture_number).shuffle(order)
                    row["method_order"] = order
                    row["methods"] = {}
                    for method in order:
                        stream = 1 if method == "poisson" else 2
                        method_seed = spec["seed"] + 1009 * capture_number + stream
                        measured = _measure(
                            algorithm, proposal, method, spec["repeats"], spec["warmup"],
                            method_seed, deadline,
                        )
                        count = measured["completed_repeats"]
                        probability = diag["acceptance_probability_float64"]
                        measured["expected_accepts_from_R"] = count * probability
                        measured["binomial_standard_deviation_from_R"] = math.sqrt(
                            count * probability * (1 - probability))
                        row["methods"][method] = measured
                    complete = all(row["methods"][name]["status"] == "completed"
                                   for name in ("poisson", "full"))
                    row["paired_timing_complete"] = complete
                    if complete:
                        poisson_mean = row["methods"]["poisson"]["mean_call_seconds"]
                        full_mean = row["methods"]["full"]["mean_call_seconds"]
                        row["poisson_over_full_time_ratio"] = poisson_mean / full_mean
                    row["replay_state_bytes"] = algorithm.state_bytes()
                    del algorithm, proposal
                except Exception as exc:
                    row.update(status="capture_error", error=repr(exc), traceback=traceback.format_exc())
                report["captures"].append(row)
                report["worker_elapsed_seconds"] = perf_counter() - started
                _write_json(spec["output"], report)
                print(json.dumps({"capture": capture_number + 1, "t": entry["t"],
                                  "case": entry["case"],
                                  "paired": row.get("paired_timing_complete", False)}), flush=True)
            else:
                report["status"] = "completed"
        except Exception as exc:
            report.update(status="worker_error", error=repr(exc), traceback=traceback.format_exc())
        report["threadpools"] = threadpool_info()
        report["memory"] = memory_snapshot()
    ratios = [row["poisson_over_full_time_ratio"] for row in report["captures"]
              if row.get("paired_timing_complete")]
    report["summary"] = {
        "selected_captures": len(spec["selected"]),
        "attempted_captures": len(report["captures"]),
        "complete_timing_pairs": len(ratios),
        "capture_errors": sum(row.get("status") == "capture_error" for row in report["captures"]),
        "method_errors": sum(method["status"] in ("error", "warmup_error")
                             for row in report["captures"]
                             for method in row.get("methods", {}).values()),
        "method_time_budgets": sum(method["status"] == "time_budget"
                                  for row in report["captures"]
                                  for method in row.get("methods", {}).values()),
        "median_poisson_over_full_time_ratio": statistics.median(ratios) if ratios else None,
        "geometric_mean_poisson_over_full_time_ratio": (
            math.exp(statistics.mean(math.log(value) for value in ratios)) if ratios else None),
        "no_distributional_pass_claim": "Finite acceptance counts are diagnostics, not a high-power kernel certificate.",
    }
    report["source_hashes_after"] = _source_hashes()
    report["historical_sources_unchanged"] = all(
        report["source_hashes_before"][name] == report["source_hashes_after"][name]
        for name in HISTORICAL
    )
    report["all_replay_sources_unchanged"] = (
        report["source_hashes_before"] == report["source_hashes_after"])
    if report["status"] == "completed":
        if report["summary"]["capture_errors"] or report["summary"]["method_errors"]:
            report["status"] = "completed_with_errors"
        elif report["summary"]["method_time_budgets"]:
            report["status"] = "time_budget"
    if not report["historical_sources_unchanged"]:
        report["status"] = "historical_source_changed"
    elif not report["all_replay_sources_unchanged"]:
        report["status"] = "replay_source_changed"
    report["worker_elapsed_seconds"] = perf_counter() - started
    _write_json(spec["output"], report)


def _launch(spec):
    import psutil
    from .memory import ProcessLimit

    output = Path(spec["output"])
    spec_path = output.with_suffix(".spec.json")
    stdout_path = output.with_suffix(".stdout.txt")
    stderr_path = output.with_suffix(".stderr.txt")
    for path in (output, spec_path, stdout_path, stderr_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing replay artifact: {path}")
    _write_json(spec_path, spec)
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    started, process, job, peak = perf_counter(), None, None, 0
    reason = None
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-m", "experiments.shared_acceptance_replay",
                 "--worker-spec", str(spec_path), "--gate"],
                cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            job = ProcessLimit(process._handle if os.name == "nt" else 0, LIMIT_BYTES)
            process.stdin.write("GO\n")
            process.stdin.flush()
            process.stdin.close()
            monitor = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    info = monitor.memory_info()
                    peak = max(peak, getattr(info, "private", info.vms))
                except psutil.NoSuchProcess:
                    pass
                if perf_counter() - started > spec["seconds"] + 45:
                    reason = "external_timeout"
                    process.kill()
                    break
                sleep(0.1)
            process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if job is not None:
            job.close()
    supervisor = {
        "returncode": process.returncode if process else None,
        "reason": reason,
        "wall_seconds": perf_counter() - started,
        "peak_private_bytes": peak,
        "hard_process_commit_limit": LIMIT_BYTES,
        "os_hard_limit_active": os.name == "nt",
    }
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8-sig"))
        if reason or supervisor["returncode"]:
            report["status_before_supervisor"] = report.get("status")
            report["status"] = reason or "abnormal_exit"
    else:
        report = {"status": reason or "no_worker_result", "configuration": spec}
    report["supervisor"] = supervisor
    _write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-captures", type=int, default=24,
                        help="Evenly spaced deterministic inventory entries; zero uses all.")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=914401)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--gate", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_spec:
        if not args.gate or sys.stdin.readline().strip() != "GO":
            raise SystemExit("Missing replay supervisor gate")
        _run_worker(json.loads(args.worker_spec.read_text(encoding="utf-8-sig")))
        return
    if (args.max_captures < 0 or args.repeats < 1 or args.warmup < 0
            or not math.isfinite(args.seconds) or args.seconds <= 0 or args.seed < 0):
        parser.error("Counts and time allowance must be positive (warmup/max-captures may be zero).")
    active = _active_campaigns()
    if active:
        raise SystemExit("Refusing acceptance replay while measured ablation is active:\n"
                         + json.dumps(active, indent=2))
    entries, rejected = _inventory(args.input)
    if not entries:
        raise SystemExit("No completed/time-budget source runs with usable captures were found.")
    selected = _select(entries, args.max_captures)
    campaign_hash_path = args.input / "source_hashes_before.json"
    campaign_expected = {}
    if campaign_hash_path.exists():
        campaign_expected = json.loads(campaign_hash_path.read_text(encoding="utf-8-sig"))
        for name in HISTORICAL:
            if name in campaign_expected and _hash(ROOT / name) != campaign_expected[name]:
                raise SystemExit(f"Historical source no longer matches campaign provenance: {name}")
    spec = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "selected": selected,
        "available_capture_count": len(entries),
        "inventory_rejections": rejected,
        "selection": "Evenly spaced entries in deterministic case/layout/persistence/acceptance/seed/round order.",
        "repeats": args.repeats,
        "warmup": args.warmup,
        "seconds": args.seconds,
        "seed": args.seed,
        "campaign_source_hashes": campaign_expected,
        "hard_process_commit_limit": LIMIT_BYTES,
        "one_blas_thread": True,
    }
    report = _launch(spec)
    print(json.dumps({"output": spec["output"], "status": report["status"],
                      "summary": report.get("summary")}, indent=2))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
