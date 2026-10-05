"""Audit completed deep-study artifacts without replaying any algorithm.

Run only after all workers, classification repair/resume, ordinary analyses,
Newton refinement, and precision diagnostics finish:
    python -m experiments.deep_audit

The process guard runs before numerical imports and artifact reads. Only raw
loss/time arrays are streamed, in bounded chunks; feature datasets are neither
read nor hashed. This checks saved-record consistency, not floating-point regret.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter


BASE = Path(__file__).resolve().parent
CAP = 2 * 1024 ** 3
STAGES = {"deep_scale": 36, "deep_budget": 28, "deep_long": 1}
HISTORY_MIB = (1536, 1856, 1920, 1968, 2000, 2048)
CORE = ("aioli.py", "common.py", "proximal.py", "sigmoid.py", "data.py")
CHUNK_VALUES = 262144  # 2 MiB of float64 input per stream chunk.
LOSS_RTOL, LOSS_ATOL = 1e-10, 1e-6
BLOCKED_MODULES = {
    "experiments.worker", "experiments.runner", "experiments.memory_frontier",
    "experiments.history_frontier", "experiments.deep_campaign", "experiments.deep_prepare",
    "experiments.analyze", "experiments.deep_pairwise", "experiments.deep_precision",
    "experiments.deep_comparator_refine", "experiments.deep_report",
}


def assert_idle():
    """Refuse concurrent measurement or mutation of audit inputs."""
    import psutil
    found = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.info["pid"] == os.getpid():
            continue
        try:
            command = proc.info["cmdline"] or []
            modules = BLOCKED_MODULES.intersection(command)
            scripts = {Path(arg).name for arg in command
                       if "experiments" in arg.lower() and arg.lower().endswith(".py")}
            matched_scripts = scripts.intersection(m.rsplit(".", 1)[-1] + ".py"
                                                   for m in BLOCKED_MODULES)
            if modules or matched_scripts:
                found.append({"pid": proc.info["pid"], "modules": sorted(modules),
                              "scripts": sorted(matched_scripts)})
            elif not command and "python" in (proc.info["name"] or "").lower():
                found.append({"pid": proc.info["pid"], "uninspectable_python_process": True})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            if "python" in (proc.info["name"] or "").lower():
                found.append({"pid": proc.info["pid"], "access_denied": True})
    if found:
        raise RuntimeError("Audit deferred: active experiment processes " + json.dumps(found))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def text_if_present(path):
    return Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""


def key(row):
    return tuple(row[k] for k in ("case_id", "algorithm", "label", "sampler_seed"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Audit:
    def __init__(self):
        self.errors, self.warnings, self.flags = [], [], {}
        self.interval_count = 0
        self.input_paths = set()
        self.input_stamps = {}

    def check(self, condition, label, **detail):
        if not condition:
            self.errors.append({"check": label, **detail})
        return bool(condition)

    def load(self, path):
        path = Path(path).resolve()
        self.input_paths.add(str(path))
        stat = path.stat()
        self.input_stamps[str(path)] = (stat.st_size, stat.st_mtime_ns)
        return read(path)

    def section(self, name, fn):
        initial = len(self.errors)
        try:
            value = fn()
        except Exception as exc:
            self.errors.append({"check": name, "exception": repr(exc)})
            value = None
        self.flags[name] = len(self.errors) == initial
        return value

    def intervals(self, value, source):
        if isinstance(value, dict):
            if "regret_lower" in value or "regret_upper" in value:
                lo, hi = value.get("regret_lower"), value.get("regret_upper")
                self.check(finite(lo) and finite(hi) and lo <= hi,
                           "finite_ordered_regret_interval", source=str(source), lower=lo, upper=hi)
                self.interval_count += 1
            for child in value.values():
                self.intervals(child, source)
        elif isinstance(value, list):
            for child in value:
                self.intervals(child, source)


def _scan_raw(path, points, *, times=False):
    """Bounded 2 MiB streaming reads; selected prefixes, no dataset access."""
    import numpy as np
    path = Path(path)
    size = path.stat().st_size if path.exists() else 0
    if size % 8:
        raise ValueError(f"Raw float64 file has a partial element: {path}")
    count = size // 8
    requested = sorted({int(n) for n in points if 0 < n <= count})
    selected, sums, last, position, cursor = {0: 0.}, [], None, 0, 0
    if count:
        with path.open("rb") as stream:
            while position < count:
                block = np.fromfile(stream, dtype="<f8", count=min(CHUNK_VALUES, count - position))
                if not len(block):
                    raise ValueError(f"Raw file truncated during audit: {path}")
                if not np.isfinite(block).all() or np.any(block < 0.):
                    raise ValueError(f"Nonfinite or negative raw values: {path}")
                if times and ((last is not None and block[0] < last)
                              or np.any(block[1:] < block[:-1])):
                    raise ValueError(f"Nonmonotone online times: {path}")
                previous_sum = math.fsum(sums) if not times else 0.
                while cursor < len(requested) and requested[cursor] <= position + len(block):
                    point = requested[cursor]
                    selected[point] = (float(block[point - position - 1]) if times else
                                       previous_sum + float(block[:point - position].sum()))
                    cursor += 1
                if not times:
                    sums.append(float(block.sum()))
                last = float(block[-1])
                position += len(block)
    return dict(path=str(path), exists=path.exists(), bytes=size, values=count,
                sum=math.fsum(sums) if not times else None, last=last,
                selected=selected, finite_nonnegative=True, monotone=True if times else None)


def _check_isolation(audit, spec, sup, worker, source, *, censored=False):
    audit.check(spec.get("memory_bytes") == CAP and sup.get("hard_process_commit_limit") == CAP
                and sup.get("os_hard_limit_active") is True,
                "active_two_gib_job_limit", source=source)
    pools = worker.get("threadpools")
    if pools:
        audit.check(all(p.get("num_threads") == 1 for p in pools),
                    "worker_one_blas_thread", source=source, threadpools=pools)
        return "verified_worker_final_snapshot"
    audit.check(censored, "worker_threadpool_snapshot_required", source=source)
    if censored:
        audit.warnings.append({"check": "worker_blas_snapshot_unavailable", "source": source,
                               "scope": "Censored process; one-thread configuration is retained, actual pool snapshot unavailable."})
    return "unavailable_censored"


def audit_online(audit, results):
    stages, runs = {}, []
    for stage, expected_count in STAGES.items():
        assert_idle()
        folder = results / stage
        specs = audit.load(folder / "manifest.json")
        summaries = audit.load(folder / "summary.json")
        curves = audit.load(folder / "curves.json")
        cases = audit.load(folder / "cases.json")
        environment = audit.load(folder / "environment.json")
        audit.check(len(specs) == expected_count, "stage_manifest_count", stage=stage,
                    actual=len(specs), expected=expected_count)
        audit.check(environment.get("worker_blas_threads") == 1, "stage_blas_configuration", stage=stage)
        identities = {key(s) for s in specs}
        summary_by_key = {key(r): r for r in summaries}
        audit.check(len(identities) == len(specs) and len(summary_by_key) == len(summaries),
                    "unique_online_identities", stage=stage)
        audit.check(set(summary_by_key) <= identities, "summary_identities_in_manifest", stage=stage)
        audit.intervals(summaries, folder / "summary.json")
        audit.intervals(curves, folder / "curves.json")
        cases_by_id = {c["id"]: c for c in cases}
        counts = Counter()
        zero_failures = []
        for spec in specs:
            out = Path(spec["output"]).resolve()
            def one():
                raw_spec, sup = audit.load(out / "spec.json"), audit.load(out / "supervisor.json")
                audit.check(raw_spec == spec, "manifest_raw_spec_equality", output=str(out))
                worker = sup["worker"]
                result_path = out / "result.json"
                has_final = result_path.exists()
                if has_final:
                    final = audit.load(result_path)
                    audit.check(final == worker, "result_supervisor_worker_equality", output=str(out))
                censored = not has_final or sup.get("termination") is not None
                threads = _check_isolation(audit, spec, sup, worker, str(out), censored=censored)
                status = worker.get("status")
                audit.check(status not in (None, "running"), "terminal_worker_status", output=str(out))
                n = worker.get("round", 0)
                audit.check(isinstance(n, int) and 0 <= n <= spec["T"], "valid_completed_rounds", output=str(out))
                if not isinstance(n, int) or n < 0:
                    return
                if status == "completed":
                    audit.check(n == spec["T"], "completed_means_full_horizon", output=str(out))
                if status == "evaluation_completed":
                    audit.check(n == spec.get("stop_after_round", spec["T"]),
                                "evaluation_endpoint_matches", output=str(out))
                summary = summary_by_key.get(key(spec))
                audit.check(summary is not None or n == 0, "positive_run_retained_in_summary", output=str(out))
                if summary:
                    audit.check(summary.get("completed_rounds") == n and summary.get("status") == status,
                                "summary_matches_terminal_status", output=str(out))
                points = {n}
                if summary:
                    points.update(p["round"] for name in ("common", "final")
                                  if (p := summary.get(name)))
                losses = _scan_raw(out / "losses.bin", points)
                times = _scan_raw(out / "online_times.bin", points, times=True)
                raw_valid = (losses["values"] >= n and times["values"] >= n if censored else
                             losses["values"] == times["values"] == n)
                audit.check(raw_valid, "raw_array_completed_lengths", output=str(out),
                            rounds=n, losses=losses["values"], times=times["values"], censored=censored)
                if has_final:
                    audit.check(losses["exists"] and times["exists"], "final_raw_arrays_exist", output=str(out))
                reported_loss = worker.get("cumulative_loss")
                observed_loss = losses["selected"].get(n)
                audit.check(finite(reported_loss) and observed_loss is not None
                            and math.isclose(observed_loss, reported_loss, rel_tol=LOSS_RTOL, abs_tol=LOSS_ATOL),
                            "raw_loss_sum_matches_completed_prefix", output=str(out),
                            reported=reported_loss, streamed=observed_loss)
                if n and n in times["selected"]:
                    last_time, online = times["selected"][n], worker.get("online_seconds")
                    audit.check(finite(online) and online + 1e-9 >= last_time,
                                "online_time_covers_completed_prefix", output=str(out))
                    if status in ("completed", "evaluation_completed", "time_budget") and not censored:
                        audit.check(math.isclose(last_time, online, rel_tol=1e-12, abs_tol=1e-9),
                                    "last_completed_time_matches_final_online_time", output=str(out))
                for name in ("common", "final"):
                    point = summary.get(name) if summary else None
                    if point:
                        q = point["round"]
                        audit.check(q in losses["selected"] and math.isclose(
                            point["loss"], losses["selected"][q], rel_tol=LOSS_RTOL, abs_tol=LOSS_ATOL),
                            "summary_prefix_loss_matches_raw", output=str(out), point=name)
                counts[status] += 1
                memory_failure = (status == "memory_error" or sup.get("termination") == "memory_limit"
                                  or "MemoryError" in str(worker.get("error")))
                if n == 0:
                    zero_failures.append(dict(output=str(out), status=status, memory_failure=memory_failure,
                                              retained_in_manifest_denominator=True))
                audit.check(spec["case_id"] in cases_by_id, "case_metadata_present", output=str(out))
                runs.append(dict(stage=stage, output=str(out), algorithm=spec["algorithm"],
                                 case_id=spec["case_id"], d=spec["d"], B=spec["B"], T=spec["T"],
                                 status=status, completed_rounds=n, zero_round_memory_failure=n == 0 and memory_failure,
                                 censored=censored, blas_verification=threads,
                                 raw_loss_values=losses["values"], raw_time_values=times["values"],
                                 raw_loss_sum_completed=observed_loss, reported_loss=reported_loss,
                                 loss_sum_absolute_difference=abs(observed_loss - reported_loss)
                                 if observed_loss is not None and finite(reported_loss) else None,
                                 raw_loss_bytes=losses["bytes"], raw_time_bytes=times["bytes"]))
            audit.section("online_run:" + str(out), one)
        stages[stage] = dict(manifest_runs=len(specs), audited_runs=sum(r["stage"] == stage for r in runs),
                             statuses=dict(counts), zero_round_runs_retained=zero_failures)
    audit.check(sum(v["manifest_runs"] for v in stages.values()) == 65, "total_online_manifest_runs_65")
    return dict(stages=stages, runs=runs)


def audit_frontier(audit, results):
    from .memory_frontier import classify_attempt, OPENBLAS_ALLOCATION_FAILURE, Axis, Frontier
    folder = results / "memory_frontier_deep"
    specs = audit.load(folder / "manifest.json")
    policy = audit.load(folder / "policy.json")
    summary = audit.load(folder / "frontier_summary.json")
    repair = audit.load(folder / "classification_audit.json")
    records_path = folder / "frontier_records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit.input_paths.add(str(records_path.resolve()))
    audit.check(repair.get("status") == "completed" and repair.get("raw_worker_artifacts_modified") is False,
                "classification_repair_completed_preserving_raw")
    original_sources = audit.load(folder / "source_hashes.json")
    controls = audit.load(results / "controls.json")["algorithm_source_sha256"]
    audit.check(all(original_sources.get(name) == controls.get(name) for name in CORE),
                "frontier_measured_core_matches_original_controls")
    # The classifier/reporting code may change in the audited repair. Only the
    # measured core must remain fixed; policy and derived records are read in
    # their final repaired/resumed form, not compared with obsolete summaries.
    resumed_sources = repair.get("authorized_resume_source_hashes", {})
    audit.check(all(resumed_sources.get(name) == controls.get(name) for name in CORE),
                "frontier_resumed_core_matches_original_controls")
    audit.check(summary.get("status") not in (None, "running", "reclassified_pending_resume"),
                "frontier_resume_finished", status=summary.get("status"))
    audit.check(len(records) == len(specs) == summary.get("attempts"), "frontier_manifest_record_counts")
    by_out = {str(Path(s["output"]).resolve()): s for s in specs}
    audit.check(len(by_out) == len(specs), "frontier_unique_outputs")
    states = {item["name"]: Frontier(Axis(**item), policy["refinements_per_axis"],
                                       policy["relative_tolerance"]) for item in policy["axes"]}
    native, counts = [], Counter()
    for record in records:
        out = Path(record["output"]).resolve()
        def one():
            spec, sup = audit.load(out / "spec.json"), audit.load(out / "supervisor.json")
            audit.check(by_out.get(str(out)) == spec, "frontier_raw_manifest_spec", output=str(out))
            tb, stderr = text_if_present(out / "traceback.txt"), text_if_present(out / "stderr.txt")
            expected = classify_attempt(spec, sup, tb, stderr)
            different = [k for k, value in expected.items() if record.get(k) != value]
            audit.check(not different, "frontier_derived_fields_match_raw_classifier", output=str(out),
                        different_fields=different)
            audit.check(expected["memory_limit_bytes"] == CAP and expected["os_hard_limit_active"],
                        "frontier_two_gib_limit", output=str(out))
            if expected["native_allocation_failure"]:
                audit.check(OPENBLAS_ALLOCATION_FAILURE in [line.strip() for line in stderr.splitlines()]
                            and sup.get("returncode") not in (None, 0),
                            "native_failure_explicit_stderr_and_exit", output=str(out))
                audit.check(record["failure_phase"] == "unknown" and record["native_failure_phase_not_inferred"]
                            and record["memory_peak_is_lower_bound"] and not record["worker_final_snapshot_available"],
                            "native_phase_unknown_peak_censored", output=str(out))
                audit.check(record["peak_private_commit"] >= sup.get("monitored_peak_private", 0),
                            "native_monitor_peak_retained", output=str(out))
                native.append(dict(output=str(out), phase=record["failure_phase"],
                                   timing_status=record["timing_measurement_status"],
                                   peak_source=record["memory_peak_source"], lower_bound=True))
            if (out / "result.json").exists():
                final = audit.load(out / "result.json")
                audit.check(final == sup.get("worker"), "frontier_result_supervisor_equality", output=str(out))
                _check_isolation(audit, spec, sup, final, str(out))
            else:
                _check_isolation(audit, spec, sup, sup.get("worker", {}), str(out), censored=True)
            states[record["axis"]].add(record)
            counts[record["outcome"]] += 1
        audit.section("frontier_run:" + str(out), one)
    for saved in summary["axes"]:
        expected = json.loads(json.dumps(states[saved["axis"]].summary()))
        audit.check(saved == expected, "frontier_axis_bracket_matches_records", axis=saved["axis"])
    return dict(attempts=len(records), outcomes=dict(counts), native_failures=native,
                status=summary.get("status"), classification_version=repair.get("classification_version"),
                bracket_scope="Separate observed parameter-axis brackets at each raw spec's actual stopping rule; dimension and population maxima are not jointly attainable guarantees.")


def audit_history(audit, results):
    rows = []
    label = "history kernel probe, not online regret experiment"
    for mib in HISTORY_MIB:
        for algorithm in ("ours", "sigmoid"):
            out = results / "history_frontier_deep" / f"{algorithm}_{mib}"
            def one():
                spec, sup = audit.load(out / "spec.json"), audit.load(out / "supervisor.json")
                worker = sup["worker"]
                censored = bool(sup.get("checkpoint_is_censored"))
                _check_isolation(audit, spec, sup, worker, str(out), censored=censored)
                if not censored:
                    audit.check(audit.load(out / "result.json") == worker,
                                "history_result_supervisor_equality", output=str(out))
                audit.check(spec.get("label") == sup.get("label") == worker.get("label") == label,
                            "history_probe_label", output=str(out))
                audit.check(worker.get("executed_online_rounds") == 0
                            and worker.get("preload_is_executed_online_trajectory") is False,
                            "history_not_online_trajectory", output=str(out))
                n, capacity = worker.get("retained_history_rows"), worker.get("allocated_history_capacity_rows")
                audit.check(isinstance(n, int) and isinstance(capacity, int)
                            and 0 <= n <= capacity and n <= spec["history_rows"],
                            "history_valid_rows_vs_capacity", output=str(out))
                audit.check(worker.get("history_ndarray_bytes") == 8 * capacity * (spec["d"] + 1),
                            "history_exact_owned_capacity_bytes", output=str(out))
                audit.check(worker.get("configured_horizon") == spec["history_rows"] + 1,
                            "history_configured_horizon_unchanged", output=str(out))
                status = worker.get("predict_status")
                allowed = {"completed", "memory_error", "error", "not_started", "skipped_time_budget"}
                audit.check(status in allowed or (censored and status == "running"),
                            "history_predict_status_explicit", output=str(out), status=status)
                if status == "completed":
                    before = worker.get("metrics_before_predict", {}).get("counters", {})
                    after = worker.get("metrics_after_predict", {}).get("counters", {})
                    audit.check(worker.get("prediction_calls") == 1 and finite(worker.get("prediction_logit"))
                                and worker.get("predict_seconds", 0) > 0
                                and after.get("history_particle_pairs", 0) > before.get("history_particle_pairs", 0),
                                "history_completed_predict_has_real_kernel_work", output=str(out))
                    validity = worker.get("history_validity_before_predict") or {}
                    audit.check(validity.get("only_valid_rows_exposed") is True
                                and validity.get("valid_batch_rows") == n,
                                "history_prediction_uses_valid_rows", output=str(out))
                if algorithm == "ours":
                    overrides = spec.get("advisory_constructor_overrides", {})
                    audit.check(overrides == worker.get("advisory_constructor_overrides")
                                and overrides.get("max_state_bytes", 0) > CAP
                                and overrides.get("max_particles", 0) >= spec["particles"],
                                "history_advisory_override_recorded", output=str(out))
                rows.append(dict(output=str(out), algorithm=algorithm, requested_mib=mib,
                                 valid_rows=n, allocated_capacity_rows=capacity,
                                 append_status=worker.get("append_status"), predict_status=status,
                                 censored=censored, executed_online_rounds=0,
                                 prediction_counter_scope="Attempt marker when interrupted; completed status plus kernel counters validates completed execution.",
                                 partial_work_scope="Counters can omit partial work in an interrupted kernel."))
            audit.section("history_run:" + str(out), one)
    audit.check(len(rows) == 12, "all_twelve_history_results_retained", count=len(rows))
    return rows


def audit_report_inputs(audit, results, online):
    # Reuse only the report's read-only loader, never its figure writer or an
    # analyzer. It verifies stale comparator-cache propagation and paired keys.
    from .deep_report import load_artifacts
    bundle = load_artifacts(results)
    audit.input_paths.update(str(Path(p).resolve()) for p in bundle["sources"])
    precision = audit.load(results / "deep_precision_audit.json")
    audit.check(precision.get("decimal_precision") == 80, "precision_diagnostic_present")
    records = precision.get("runs", [])
    eligible = {(r["case_id"], r["completed_rounds"]) for r in online["runs"]
                if r["stage"] == "deep_scale" and r["algorithm"] == "aioli"
                and r["d"] <= 100 and r["B"] >= 10000 and r["status"] == "completed"}
    actual = {(r["case_id"], r["rounds"]) for r in records}
    audit.check(actual == eligible and len(actual) == len(records), "precision_run_coverage")
    audit.check(all(r.get("exact_replay_of_recorded_losses") is True
                    and r.get("max_loss_difference") == 0 for r in records),
                "saved_precision_replay_agrees")
    newton = audit.load(results / "deep_long" / "newton_refinement_audit.json")
    audit.check(bool(newton.get("attempts")) and newton.get("finite_precision_certified") is False,
                "newton_refinement_record_present")
    for attempt in newton.get("attempts", []):
        audit.check(not attempt.get("accepted") or attempt.get("cache_replaced") is True,
                    "accepted_newton_cache_propagated", case_id=attempt.get("case_id"), round=attempt.get("round"))
    audit.intervals(bundle["long_curves"], results / "deep_long" / "curves.json")
    return dict(report_loader_succeeded=True, precision_runs=len(records),
                newton_attempts=len(newton.get("attempts", [])), paired_complete=bundle["pairwise"]["complete"])


def run_audit(results):
    assert_idle()  # No imports or reads from numerical studies before this guard.
    started = perf_counter()
    results = Path(results).resolve()
    audit = Audit()
    original = audit.load(results / "controls.json")["algorithm_source_sha256"]
    hashes = {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in CORE}
    unchanged = {name: hashes[name] == original.get(name) for name in CORE}
    audit.check(all(unchanged.values()), "core_algorithm_hashes_match_original_controls", unchanged=unchanged)
    audit.flags["core_algorithm_hashes_unchanged"] = all(unchanged.values())
    online = audit.section("online_artifacts", lambda: audit_online(audit, results))
    assert_idle()
    frontier = audit.section("memory_frontier_artifacts", lambda: audit_frontier(audit, results))
    history = audit.section("history_frontier_artifacts", lambda: audit_history(audit, results))
    report_inputs = audit.section("report_required_inputs", lambda: audit_report_inputs(audit, results, online))
    assert_idle()
    for path, previous in audit.input_stamps.items():
        current = Path(path).stat()
        audit.check((current.st_size, current.st_mtime_ns) == previous,
                    "audit_input_unchanged_during_read", path=path)
    audit.flags["recorded_regret_intervals_ordered"] = not any(
        e["check"] == "finite_ordered_regret_interval" for e in audit.errors)
    result = dict(schema_version=1, status="passed" if not audit.errors else "failed",
                  passed=not audit.errors, utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=perf_counter() - started, check_flags=audit.flags,
                  source_sha256=hashes, core_matches_control_snapshot=unchanged,
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  online=online, memory_frontier=frontier, history_frontier=history,
                  counts=dict(online_manifest_runs=sum(s["manifest_runs"] for s in online["stages"].values()) if online else None,
                              online_audited_runs=len(online["runs"]) if online else None,
                              online_zero_round_memory_failures=sum(r["zero_round_memory_failure"] for r in online["runs"]) if online else None,
                              memory_frontier_attempts=frontier["attempts"] if frontier else None,
                              history_probes=len(history) if history is not None else None),
                  report_inputs=report_inputs, checked_intervals=audit.interval_count,
                  required_inputs=sorted(audit.input_paths), errors=audit.errors, warnings=audit.warnings,
                  raw_stream_chunk_bytes=8 * CHUNK_VALUES,
                  loss_sum_tolerance=dict(relative=LOSS_RTOL, absolute=LOSS_ATOL),
                  scope="Saved artifact consistency only; no predictor replay, comparator fitting, dataset read/hash, or floating-point certificate.")
    target = results / "deep_artifact_audit.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=BASE / "results")
    args = parser.parse_args()
    try:
        result = run_audit(args.results)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(dict(status=result["status"], errors=len(result["errors"]),
                          elapsed_seconds=result["elapsed_seconds"],
                          output=str(args.results / "deep_artifact_audit.json"))))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
