"""Read-only consistency audit of the completed shared-tangent comparison.

Run only after all measurements, comparator analyses, and report generation:
    python -m experiments.shared_artifact_audit

The process guard precedes numerical imports. Loss/time files are streamed in
bounded chunks; datasets are identified by their unchanged paths, metadata and
sizes, not replayed or rehashed. Failures remain in all manifest denominators.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from time import perf_counter


BASE = Path(__file__).resolve().parent
CAP = 2 * 1024**3
COUNTS = {
    "smoke": 2, "prescribed": 3, "calibration": 24, "screen": 63,
    "deep_scale": 12, "confirmation": 30, "deep_budget": 2,
    "long_horizon": 1, "deep_long": 1,
}
PRIMARY = ("screen", "deep_scale", "confirmation", "deep_budget", "long_horizon", "deep_long")
COMPARISONS = ("prescribed",) + PRIMARY
GRID = ((16, 1), (16, 4), (16, 16), (64, 1), (64, 4), (256, 1))
CORE = ("aioli.py", "common.py", "proximal.py", "sigmoid.py", "data.py")
SNAPSHOT = ("SHARED_TANGENT_INPUT.md", "shared_tangent.py", "shared_tangent_primitives.py",
            "shared_worker.py", "shared_runner.py", "common.py", "proximal.py", "sigmoid.py",
            "aioli.py", "data.py", "memory.py")
RTOL, ATOL = 1e-10, 1e-6
TERMINAL = {"completed", "evaluation_completed", "time_budget", "memory_error", "error",
            "memory_limit", "external_timeout", "abnormal_exit"}


def assert_idle():
    """Refuse active experiment workers, orchestration, analysis, or reports."""
    import psutil
    matches = []
    words = ("worker", "runner", "campaign", "frontier", "analyze", "prepare", "report",
             "refine", "precision", "pairwise")
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if proc.info["pid"] == os.getpid():
            continue
        try:
            command = proc.info["cmdline"] or []
            matched = [arg for arg in command if "experiments" in arg.lower()
                       and any(word in arg.lower() for word in words)
                       and (arg.startswith("experiments.") or arg.lower().endswith(".py"))]
            if matched:
                matches.append({"pid": proc.info["pid"], "commands": matched})
            elif not command and "python" in (proc.info["name"] or "").lower():
                matches.append({"pid": proc.info["pid"], "uninspectable_python": True})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            if "python" in (proc.info["name"] or "").lower():
                matches.append({"pid": proc.info["pid"], "access_denied": True})
    if matches:
        raise RuntimeError("Artifact audit deferred until experiment processes finish: " + json.dumps(matches))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(first, second):
    return finite(first) and finite(second) and math.isclose(first, second, rel_tol=RTOL, abs_tol=ATOL)


def key(row):
    return tuple(row.get(name) for name in ("case_id", "algorithm", "label", "sampler_seed"))


def resolved(path):
    return str(Path(path).resolve())


class Audit:
    def __init__(self):
        self.errors, self.warnings, self.flags = [], [], {}
        self.stamps, self.runs, self.manifests, self.prefixes = {}, {}, {}, {}
        self.interval_count = 0

    def check(self, condition, check_name, **details):
        if not condition:
            self.errors.append(dict(check=check_name, **details))
        return bool(condition)

    def stamp(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        old = self.stamps.setdefault(str(path), stamp)
        self.check(old == stamp, "input_unchanged_on_repeated_read", path=str(path))
        return path

    def load(self, path):
        return read(self.stamp(path))

    def section(self, name, function):
        previous = len(self.errors)
        try:
            result = function()
        except Exception as exc:
            self.errors.append(dict(check=name, exception=repr(exc)))
            result = None
        self.flags[name] = len(self.errors) == previous
        return result

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

    def prefix(self, output, n):
        """Small random-access prefix read, cached; no feature-data access."""
        import numpy as np
        output = resolved(output)
        n = int(n)
        cache_key = (output, n)
        if cache_key not in self.prefixes:
            if n == 0:
                self.prefixes[cache_key] = (0., 0.)
            else:
                sums, remaining = [], n
                with (Path(output)/"losses.bin").open("rb") as stream:
                    while remaining:
                        block = np.fromfile(stream, dtype="<f8", count=min(262144, remaining))
                        if not len(block) or not np.isfinite(block).all():
                            raise ValueError("Incomplete/nonfinite requested loss prefix: " + output)
                        sums.append(float(block.sum()))
                        remaining -= len(block)
                with (Path(output)/"online_times.bin").open("rb") as stream:
                    stream.seek(8*(n-1))
                    online = struct.unpack("<d", stream.read(8))[0]
                self.prefixes[cache_key] = (math.fsum(sums), online)
        return self.prefixes[cache_key]


def audit_sources(audit, results):
    controls = audit.load(results/"controls.json")["algorithm_source_sha256"]
    current = {name: hashlib.sha256(audit.stamp(BASE/name).read_bytes()).hexdigest()
               for name in SNAPSHOT}
    for name in CORE:
        audit.check(current[name] == controls.get(name), "original_algorithm_source_unchanged", name=name)
    snapshots = {}
    for stage in COUNTS:
        saved = audit.load(results/("shared_"+stage)/"source_hashes.json")
        audit.check(saved == current, "measured_source_snapshot_matches_current", stage=stage,
                    differing=[name for name in set(saved)|set(current) if saved.get(name) != current.get(name)])
        snapshots[stage] = saved
    return dict(current=current, controls_original_core={name: controls[name] for name in CORE},
                all_measured_stages=list(snapshots), old_algorithms_reused_unchanged=True)


def audit_manifest(audit, results, source_stage, selected):
    stage, folder = "shared_"+source_stage, results/("shared_"+source_stage)
    specs = audit.load(folder/"manifest.json")
    source = audit.load(results/source_stage/"manifest.json")
    cases = audit.load(folder/"cases.json")
    audit.check(cases == audit.load(results/source_stage/"cases.json"), "copied_case_metadata_unchanged", stage=stage)
    environment = audit.load(folder/"environment.json")
    audit.check(environment.get("worker_blas_threads") == 1, "one_thread_worker_configuration", stage=stage)
    audit.check(len(specs) == COUNTS[source_stage], "required_new_manifest_count", stage=stage,
                expected=COUNTS[source_stage], actual=len(specs))
    audit.check(len({key(s) for s in specs}) == len(specs), "unique_manifest_identities", stage=stage)
    audit.check(len({resolved(s["output"]) for s in specs}) == len(specs), "unique_manifest_outputs", stage=stage)
    originals = {}
    for spec in source:
        originals.setdefault(spec["case_id"], spec)
    groups = defaultdict(list)
    data_identities = {}
    changed = {"algorithm", "config", "label", "stage", "output", "source_stage", "source_manifest", "sampler_seed"}
    for spec in specs:
        cid = spec["case_id"]
        original = originals[cid]
        audit.check(spec.get("algorithm") == "shared" and spec.get("stage") == stage,
                    "shared_algorithm_stage_identity", output=spec["output"])
        audit.check(Path(spec["output"]).resolve().is_relative_to(folder.resolve()),
                    "output_within_own_stage", output=spec["output"])
        audit.check(spec.get("source_stage") == source_stage and
                    resolved(spec.get("source_manifest", "")) == resolved(results/source_stage/"manifest.json"),
                    "source_manifest_reference", output=spec["output"])
        preserved = {k: v for k, v in original.items() if k not in changed}
        audit.check(all(spec.get(k) == v for k, v in preserved.items()),
                    "same_data_horizon_stopping_and_resource_budget", output=spec["output"],
                    differing=[k for k, v in preserved.items() if spec.get(k) != v])
        audit.check(spec.get("memory_bytes") == CAP, "manifest_two_gib_limit", output=spec["output"])
        repeat = 0
        if source_stage == "confirmation":
            match = re.fullmatch(re.escape(selected["label"])+r"_rep([012])", spec["label"])
            audit.check(match is not None, "confirmation_repetition_label", output=spec["output"])
            repeat = int(match.group(1)) if match else -1
        expected_seed = spec["data_seed"]+89173+(40000003+1000003*repeat if source_stage == "confirmation" else 0)
        audit.check(spec["sampler_seed"] == expected_seed, "predeclared_sampler_seed", output=spec["output"])
        if source_stage == "calibration":
            allowed = {(f"m{m}n{n}", json.dumps(dict(mode="practical", tilts=m, burn_in=n, transitions=n), sort_keys=True))
                       for m, n in GRID}
            audit.check((spec["label"], json.dumps(spec["config"], sort_keys=True)) in allowed,
                        "predeclared_calibration_configuration", output=spec["output"])
        elif source_stage == "prescribed":
            audit.check(spec["config"] == {"mode": "prescribed"}, "uncapped_prescribed_config", output=spec["output"])
        elif source_stage == "smoke":
            audit.check(spec["config"] == dict(mode="practical", tilts=4, burn_in=1, transitions=1),
                        "predeclared_smoke_config", output=spec["output"])
        else:
            audit.check(spec["config"] == selected["config"], "frozen_calibration_choice_used", output=spec["output"])
        groups[cid].append(spec)
        if cid not in data_identities:
            data_folder = Path(spec["data_path"])
            meta = audit.load(data_folder/"data.json")
            case = meta["case"]
            audit.check(all(case[k] == spec[k] for k in ("d", "B", "T", "data_seed")),
                        "data_metadata_matches_spec", case_id=cid)
            audit.check(case["id"] == cid and len(meta.get("sha256_interleaved_blocks", "")) == 64,
                        "data_identifier_and_saved_hash", case_id=cid)
            xstat, ystat = audit.stamp(data_folder/"X.bin").stat(), audit.stamp(data_folder/"y.bin").stat()
            audit.check(xstat.st_size == 8*case["T"]*case["d"] and ystat.st_size == case["T"],
                        "data_file_expected_sizes", case_id=cid)
            data_identities[cid] = dict(data_path=resolved(data_folder), metadata_sha256=meta["sha256_interleaved_blocks"],
                                       configured_rounds=case["T"], dimension=case["d"],
                                       identity_check="Same raw data path and metadata; sizes checked; content not rehashed.")
    multiplier = 6 if source_stage == "calibration" else 3 if source_stage == "confirmation" else 1
    audit.check(set(groups) == set(originals) and all(len(v) == multiplier for v in groups.values()),
                "all_source_cases_and_repetitions_covered", stage=stage)
    audit.manifests[stage] = specs
    return dict(stage=stage, specs=specs, data_identities=data_identities)


def audit_run(audit, spec, *, new=False, points=()):
    from .deep_audit import _scan_raw
    out = Path(spec["output"]).resolve()
    audit.check(audit.load(out/"spec.json") == spec, "manifest_raw_spec_equality", output=str(out))
    sup = audit.load(out/"supervisor.json")
    worker = sup["worker"]
    has_final = (out/"result.json").exists()
    if has_final:
        audit.check(audit.load(out/"result.json") == worker, "supervisor_matches_worker_result", output=str(out))
    censored = not has_final or sup.get("termination") is not None
    status = worker.get("status")
    audit.check(status in TERMINAL or isinstance(status, str) and status.startswith("supervisor_error:"),
                "terminal_worker_status", output=str(out), status=status)
    n = worker.get("round")
    if not audit.check(isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= spec["T"],
                       "valid_completed_round_count", output=str(out), rounds=n):
        return None
    if status == "completed":
        audit.check(n == spec["T"], "completed_is_full_configured_horizon", output=str(out))
    if status == "evaluation_completed":
        audit.check(n == spec.get("stop_after_round", spec["T"]), "completed_fixed_evaluation_endpoint", output=str(out))
    audit.check(sup.get("hard_process_commit_limit") == CAP and sup.get("os_hard_limit_active") is True,
                "active_two_gib_os_cap", output=str(out))
    pools = worker.get("threadpools")
    if pools:
        audit.check(all(p.get("num_threads") == 1 for p in pools), "actual_worker_one_blas_thread", output=str(out))
    else:
        audit.check(censored, "uncensored_final_threadpool_snapshot_exists", output=str(out))
        audit.warnings.append(dict(check="censored_threadpool_snapshot_unavailable", output=str(out),
                                   scope="One-thread configuration retained; actual final pool snapshot unavailable."))
    points = {n, *points}
    for name in ("losses.bin", "online_times.bin"):
        if (out/name).exists():
            audit.stamp(out/name)
    losses = _scan_raw(out/"losses.bin", points)
    times = _scan_raw(out/"online_times.bin", points, times=True)
    lengths_ok = (losses["values"] >= n and times["values"] >= n if censored else
                  losses["values"] == times["values"] == n)
    audit.check(lengths_ok, "raw_completed_prefix_lengths", output=str(out), rounds=n,
                losses=losses["values"], times=times["values"], censored=censored)
    if has_final:
        audit.check(losses["exists"] and times["exists"], "final_raw_arrays_exist", output=str(out))
    observed_loss = losses["selected"].get(n)
    audit.check(close(observed_loss, worker.get("cumulative_loss")), "raw_loss_sum_matches_completed_prefix",
                output=str(out), raw=observed_loss, reported=worker.get("cumulative_loss"))
    if n:
        final_time = times["selected"].get(n)
        audit.check(finite(final_time) and finite(worker.get("online_seconds"))
                    and worker["online_seconds"]+1e-9 >= final_time,
                    "online_time_covers_completed_prefix", output=str(out))
        if has_final and not censored and status in ("completed", "evaluation_completed", "time_budget"):
            audit.check(close(final_time, worker["online_seconds"]), "terminal_time_matches_last_completed_round", output=str(out))
    for point in points:
        if point in losses["selected"] and point in times["selected"]:
            audit.prefixes[(str(out), point)] = (losses["selected"][point], times["selected"][point])
    if new and worker.get("diagnostics"):
        diag = worker["diagnostics"]
        audit.check(diag.get("skip_uses_prescribed_tilts") is True and diag.get("floating_point_certificate") is False,
                    "skip_and_floating_point_scope_recorded", output=str(out))
        preflight = diag.get("preflight", {})
        audit.check(preflight.get("mode") == spec["config"].get("mode", "practical") and
                    preflight.get("memory_representation") == "raw features plus separate weighted prefix",
                    "implemented_mode_and_history_representation", output=str(out))
        if not censored and status in ("completed", "evaluation_completed", "time_budget"):
            audit.check(diag.get("history_length") == n, "finished_round_history_length", output=str(out))
        else:
            audit.check(diag.get("history_length") in (n, n+1), "failed_attempt_history_can_include_current_feature", output=str(out))
    failure = status not in ("completed", "evaluation_completed", "time_budget")
    if failure:
        audit.warnings.append(dict(check="measured_failure_retained", output=str(out), status=status,
                                   completed_rounds=n, error=worker.get("error"), termination=sup.get("termination")))
    row = dict(output=str(out), stage=spec.get("stage"), case_id=spec["case_id"], algorithm=spec["algorithm"],
               label=spec["label"], sampler_seed=spec["sampler_seed"], status=status, completed_rounds=n,
               zero_round_run=n == 0, measured_failure=failure, censored=censored,
               raw_loss_values=losses["values"], raw_time_values=times["values"],
               completed_prefix_loss=observed_loss, reported_loss=worker.get("cumulative_loss"),
               worker_blas_verified=bool(pools), online_seconds=worker.get("online_seconds"),
               process_wall_seconds=sup.get("process_wall_seconds"), manifest_denominator_retained=True)
    audit.runs[str(out)] = row
    return row


def audit_selection(audit, results, selected):
    specs = audit.manifests["shared_calibration"]
    groups = defaultdict(list)
    for spec in specs:
        groups[spec["case_id"]].append(spec)
    common, values = {}, defaultdict(list)
    for cid, rows in groups.items():
        n = min(audit.runs[resolved(s["output"])]["completed_rounds"] for s in rows)
        common[cid] = n
        audit.check(n > 0, "positive_calibration_common_prefix", case_id=cid)
        for spec in rows:
            run = audit.runs[resolved(spec["output"])]
            audit.check(run["status"] in ("completed", "time_budget"), "calibration_terminal_eligibility", output=spec["output"])
            loss, _ = audit.prefix(spec["output"], n)
            values[spec["label"]].append(loss/n)
    scores = {label: math.fsum(v)/len(v) for label, v in values.items()}
    best = min(scores, key=lambda label: (scores[label], label))
    audit.check(selected.get("common_prefixes") == common, "saved_calibration_prefixes_recomputed")
    audit.check(selected.get("label") == best, "calibration_selection_recomputed", expected=best, actual=selected.get("label"))
    saved_scores = {row["label"]: row for row in selected.get("scores", [])}
    audit.check(set(saved_scores) == set(scores), "all_calibration_configurations_retained")
    for label, score in scores.items():
        audit.check(close(saved_scores.get(label, {}).get("mean_loss_per_round"), score),
                    "saved_calibration_score_recomputed", label=label)
    calibration_paths = {resolved(s["data_path"]) for s in specs}
    primary_paths = {resolved(s["data_path"]) for stage in PRIMARY
                     for s in audit.manifests["shared_"+stage]}
    audit.check(not calibration_paths.intersection(primary_paths), "calibration_data_held_out_from_primary_cases")
    return dict(selected_label=best, common_prefixes=common, mean_loss_per_round=scores,
                held_out_choice_recomputed=True)


def audit_comparison(audit, results, source_stage):
    folder = results/("shared_compare_"+source_stage)
    specs = audit.load(folder/"manifest.json")
    expected = audit.load(results/source_stage/"manifest.json") + audit.manifests["shared_"+source_stage]
    saved_map = {resolved(s["output"]): s for s in specs}
    expected_map = {resolved(s["output"]): s for s in expected}
    audit.check(len(saved_map) == len(specs) and saved_map == expected_map,
                "combined_manifest_is_exact_old_plus_new_union", stage=source_stage)
    provenance = audit.load(folder/"comparison_provenance.json")
    audit.check(provenance.get("source_spec_values_and_raw_paths_unchanged") is True
                and provenance.get("numerical_jobs_executed") is False,
                "comparison_preparation_preserves_archived_runs", stage=source_stage)
    audit.check(provenance.get("run_counts") == dict(
        archived=len(expected)-len(audit.manifests["shared_"+source_stage]),
        shared=len(audit.manifests["shared_"+source_stage]), total=len(expected)),
        "comparison_preparation_counts", stage=source_stage)
    provenance_rows = provenance.get("runs", {}).get("archived", [])+provenance.get("runs", {}).get("shared", [])
    audit.check({resolved(row["output"]) for row in provenance_rows} == set(expected_map),
                "comparison_preparation_covers_every_raw_output", stage=source_stage)
    for record in [provenance.get("manifest", {}), provenance.get("cases", {})]+[
            record for row in provenance_rows for record in (row.get("spec", {}), row.get("supervisor", {}))]:
        path = audit.stamp(record["path"])
        audit.check(record.get("exists") is True and record.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest(),
                    "preparation_raw_evidence_hash_matches", path=str(path))
    summaries = audit.load(folder/"summary.json")
    curves = audit.load(folder/"curves.json")
    audit.intervals(summaries, folder/"summary.json")
    audit.intervals(curves, folder/"curves.json")
    by_key = {key(row): row for row in summaries}
    audit.check(len(by_key) == len(summaries), "unique_combined_summary_identities", stage=source_stage)
    by_curve = defaultdict(list)
    for row in curves:
        by_curve[key(row)].append(row)
    identities = {key(spec) for spec in specs}
    audit.check(set(by_key) <= identities and set(by_curve) <= identities,
                "combined_summaries_and_curves_belong_to_manifest", stage=source_stage)
    groups = defaultdict(list)
    for spec in specs:
        groups[spec["case_id"]].append(spec)
    for cid, rows in groups.items():
        paths = {resolved(s["data_path"]) for s in rows}
        dimensions = {(s["d"], s["B"], s["T"], s["data_seed"]) for s in rows}
        audit.check(len(paths) == len(dimensions) == 1, "paired_comparison_identical_dataset", stage=source_stage, case_id=cid)
        comparator = audit.load(folder/(cid+"__comparators.json")) if (folder/(cid+"__comparators.json")).exists() else {}
        ns = []
        for spec in rows:
            identity, output = key(spec), resolved(spec["output"])
            summary = by_key.get(identity)
            points = {p["round"] for p in by_curve.get(identity, [])}
            if summary:
                points.update(summary[name]["round"] for name in ("common", "final") if summary.get(name))
            # This also streams reused baseline arrays; no algorithm is rerun.
            run = audit_run(audit, spec, new=spec["algorithm"] == "shared", points=points)
            if run is None:
                continue
            n = run["completed_rounds"]
            ns.append(n)
            audit.check(summary is not None or n == 0, "all_positive_runs_in_combined_summary", output=output)
            if summary:
                audit.check(summary.get("completed_rounds") == n and summary.get("status") == run["status"],
                            "summary_matches_worker_terminal_prefix", output=output)
            records = [(name, summary[name]) for name in ("common", "final") if summary and summary.get(name)]
            records += [("curve", row) for row in by_curve.get(identity, [])]
            for name, point in records:
                q = point["round"]
                audit.check(isinstance(q, int) and 1 <= q <= n, "comparison_prefix_within_completed_trajectory", output=output, point=q)
                loss, online = audit.prefix(output, q)
                audit.check(close(loss, point.get("loss")), "summary_or_curve_raw_loss", output=output, point=q, kind=name)
                if "online_seconds" in point:
                    audit.check(close(online, point["online_seconds"]), "summary_raw_prefix_time", output=output, point=q)
                fit = comparator.get(str(q))
                audit.check(fit is not None, "comparison_has_cached_comparator", case_id=cid, point=q)
                if fit:
                    width = min(fit["gap"], max(0., fit["loss"]))
                    lower = loss-fit["loss"]
                    audit.check(close(point.get("regret_lower"), lower) and close(point.get("regret_upper"), lower+width),
                                "regret_interval_matches_current_comparator", output=output, point=q)
        common = min(ns) if ns else 0
        for spec in rows:
            row = by_key.get(key(spec))
            if row:
                audit.check(row.get("common_round") == common, "all_method_common_prefix_recomputed", output=spec["output"])
    return dict(source_stage=source_stage, combined_manifest_runs=len(specs), summary_rows=len(summaries),
                curve_rows=len(curves), zero_round_rows_can_be_omitted_only_from_summaries=True,
                all_manifest_failures_retained_in_audit=True)


def _repeat(label):
    match = re.search(r"_rep(\d+)$", label)
    return int(match.group(1)) if match else None


def _mean(values):
    values = list(values)
    return math.fsum(values)/len(values) if values else None


def audit_report(audit, results, selected):
    report = audit.load(results/"shared_comparison.json")
    provenance = audit.load(results/"shared_comparison_provenance.json")
    audit.check(report.get("complete") is True and report.get("schema_version") == 1,
                "completed_supported_report_schema")
    audit.check(report.get("selected_config") == selected, "report_uses_frozen_selection")
    audit.check(set(report.get("stages", {})) == set(COMPARISONS), "report_covers_every_required_comparison_stage")
    audit.check(set(report.get("auxiliary_stages", {})) == {"calibration", "smoke"},
                "report_covers_auxiliary_attempts")
    for stage, saved in report.get("auxiliary_stages", {}).items():
        expected = {resolved(s["output"]): audit.runs[resolved(s["output"])] for s in audit.manifests["shared_"+stage]}
        rows = saved.get("rows", [])
        audit.check(saved.get("new_runs") == len(rows) == len(expected)
                    and {resolved(row["output"]) for row in rows} == set(expected),
                    "report_auxiliary_attempt_denominator", stage=stage)
        audit.check(saved.get("new_status_counts") == dict(Counter(row["status"] for row in expected.values())),
                    "report_auxiliary_status_counts", stage=stage)
        for row in rows:
            raw = expected[resolved(row["output"])]
            audit.check(all(row.get(k) == raw.get(k) for k in ("case_id", "label", "status", "completed_rounds")),
                        "report_auxiliary_terminal_records", output=row["output"])
    audit.check(provenance.get("limits") == dict(worker_private_commit_bytes=CAP,
                                               worker_blas_threads=1, measured_workers_at_once=1),
                "report_isolation_scope")
    audit.check(provenance.get("report_source_sha256") == hashlib.sha256(
        audit.stamp(BASE/"shared_comparison_report.py").read_bytes()).hexdigest(),
        "report_source_snapshot_matches_generator")
    records = [(provenance["report"], provenance["report_sha256"]),
               (provenance["comparison_json"], provenance["comparison_json_sha256"]),
               (results/"shared_prescribed_preflight.json", provenance["prescribed_preflight_sha256"])]
    records += list(provenance.get("figures", {}).items())+list(provenance.get("input_artifacts", {}).items())
    audit.check(len(provenance.get("figures", {})) == 4, "four_report_figures_present")
    audit.check({Path(path).name for path in provenance.get("figures", {})} == {
        "shared_loss_difference.png", "shared_time_ratio.png", "shared_memory.png", "shared_diagnostics.png"},
        "expected_report_figure_names")
    for path, digest in records:
        path = audit.stamp(path)
        audit.check(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                    "report_provenance_content_hash", path=str(path))
    preflight = audit.load(results/"shared_prescribed_preflight.json")
    audit.check(preflight == report.get("prescribed_preflight") and preflight.get("allocation_free") is True,
                "prescribed_feasibility_artifact_propagated")
    expected_settings = {(s["d"], s["B"], s["T"], s["config"].get("R", 1.))
                         for stage in COMPARISONS for s in audit.manifests["shared_"+stage]}
    audit.check({tuple(row[k] for k in ("d", "B", "T", "R")) for row in preflight["rows"]} == expected_settings,
                "prescribed_feasibility_covers_measured_settings")
    pairs, grouped = [], defaultdict(list)
    for source, saved in report["stages"].items():
        specs = audit.load(results/("shared_compare_"+source)/"manifest.json")
        spec_by_out = {resolved(s["output"]): s for s in specs}
        public = saved["rows"]
        audit.check(len(public) == saved.get("expected_runs") == len(specs), "report_stage_run_denominator", stage=source)
        audit.check({resolved(row["output"]) for row in public} == set(spec_by_out),
                    "report_all_raw_runs_including_zero_failures", stage=source)
        audit.check(saved.get("new_runs") == COUNTS[source] and saved.get("old_runs") == len(specs)-COUNTS[source],
                    "report_new_old_counts", stage=source)
        actual_counts = Counter(audit.runs[output]["status"] for output in spec_by_out)
        new_counts = Counter(audit.runs[output]["status"] for output, spec in spec_by_out.items() if spec["algorithm"] == "shared")
        audit.check(saved.get("all_status_counts") == dict(actual_counts)
                    and saved.get("new_status_counts") == dict(new_counts), "report_all_terminal_status_counts", stage=source)
        for row in public:
            output = resolved(row["output"])
            raw = audit.runs[output]
            audit.check(row["status"] == raw["status"] and row["completed_rounds"] == raw["completed_rounds"],
                        "public_run_matches_raw_status", output=output)
            audit.check(all(row.get(k) == v for k, v in spec_by_out[output].items()),
                        "public_run_preserves_raw_spec", output=output)
        expected_pairs = set()
        for new in specs:
            if new["algorithm"] != "shared":
                continue
            for old in specs:
                if old["algorithm"] == "shared" or old["case_id"] != new["case_id"]:
                    continue
                first, second = _repeat(new["label"]), _repeat(old["label"])
                if first is not None and second is not None and first != second:
                    continue
                expected_pairs.add((resolved(new["output"]), resolved(old["output"])))
        actual_pairs = [(resolved(pair["new"]["output"]), resolved(pair["old"]["output"])) for pair in saved["pairwise"]]
        audit.check(len(actual_pairs) == len(set(actual_pairs)) and set(actual_pairs) == expected_pairs,
                    "all_predeclared_pairs_retained", stage=source)
        for pair in saved["pairwise"]:
            new_out, old_out = resolved(pair["new"]["output"]), resolved(pair["old"]["output"])
            new, old = audit.runs[new_out], audit.runs[old_out]
            new_spec, old_spec = spec_by_out[new_out], spec_by_out[old_out]
            for identity, spec in ((pair["new"], new_spec), (pair["old"], old_spec)):
                audit.check(all(identity.get(k) == spec[k] for k in ("algorithm", "label", "config", "sampler_seed"))
                            and identity.get("run_id") == Path(spec["output"]).name,
                            "report_pair_algorithm_and_config_identity", output=spec["output"])
            audit.check(pair.get("case_id") == new_spec["case_id"] == old_spec["case_id"]
                        and pair.get("data_seed") == new_spec["data_seed"] == old_spec["data_seed"]
                        and resolved(pair["data_path"]) == resolved(new_spec["data_path"]) == resolved(old_spec["data_path"]),
                        "report_pair_uses_identical_data", new=new_out, old=old_out)
            case = audit.load(Path(new_spec["data_path"])/"data.json")["case"]
            audit.check(pair["setting"] == {k: v for k, v in case.items() if k not in ("id", "data_seed")},
                        "reported_parameter_regime_matches_data", new=new_out, old=old_out)
            audit.check(pair["common"]["round"] == min(new["completed_rounds"], old["completed_rounds"]),
                        "pair_common_attained_prefix", new=new_out, old=old_out)
            audit.check([p["round"] for p in pair["checkpoints"]] == [100, 500, 1000],
                        "fixed_checkpoint_coverage", new=new_out, old=old_out)
            identity = dict(source=source, setting=pair["setting"],
                            new_label=re.sub(r"_rep\d+$", "", pair["new"]["label"]), new_config=pair["new"]["config"],
                            old_algorithm=pair["old"]["algorithm"],
                            old_label=re.sub(r"_rep\d+$", "", pair["old"]["label"]), old_config=pair["old"]["config"])
            for name, point in [("pair_common", pair["common"])]+[(str(p["round"]), p) for p in pair["checkpoints"]]:
                n = point["round"]
                available = n > 0 and n <= min(new["completed_rounds"], old["completed_rounds"])
                audit.check(point["available"] == available, "pair_availability_from_completed_prefix", new=new_out, old=old_out, round=n)
                if available:
                    a, ta = audit.prefix(new_out, n)
                    b, tb = audit.prefix(old_out, n)
                    expected = dict(new_loss=a, old_loss=b, loss_difference=a-b, average_loss_difference=(a-b)/n,
                                    new_online_seconds=ta, old_online_seconds=tb)
                    audit.check(all(close(point.get(k), v) for k, v in expected.items()),
                                "reported_pair_values_match_raw_recordings", new=new_out, old=old_out, round=n)
                    audit.check(close(point.get("time_ratio"), ta/tb) if tb > 0 else point.get("time_ratio") is None,
                                "reported_pair_time_ratio", new=new_out, old=old_out, round=n)
                else:
                    audit.check(not any(k in point for k in ("new_loss", "old_loss", "loss_difference", "time_ratio")),
                                "unavailable_pair_never_reported_as_zero_loss", new=new_out, old=old_out, round=n)
                grouped[json.dumps({**identity, "checkpoint": name}, sort_keys=True)].append((pair, point))
            pairs.append(pair)
    saved_aggregates = report["paired_seed_summaries"]
    aggregate_map = {json.dumps({k: row[k] for k in ("source", "setting", "new_label", "new_config", "old_algorithm",
                                                     "old_label", "old_config", "checkpoint")}, sort_keys=True): row
                     for row in saved_aggregates}
    audit.check(len(aggregate_map) == len(saved_aggregates) and set(aggregate_map) == set(grouped),
                "all_seed_aggregate_groups_retained")
    for identity, rows in grouped.items():
        by_seed = defaultdict(list)
        for pair, point in rows:
            by_seed[pair["data_seed"]].append(point)
        saved = aggregate_map.get(identity, {})
        dataset_rows = {row["data_seed"]: row for row in saved.get("dataset_values", [])}
        audit.check(set(dataset_rows) == set(by_seed), "seed_aggregate_dataset_denominator", group=identity)
        good_values = []
        for seed, points in by_seed.items():
            good = [point for point in points if point["available"]]
            row = dataset_rows.get(seed, {})
            audit.check(row.get("available") == bool(good) and row.get("repetitions_expected") == len(points)
                        and row.get("repetitions_available") == len(good), "seed_repetition_availability", group=identity, seed=seed)
            if good:
                new_time, old_time = _mean(p["new_online_seconds"] for p in good), _mean(p["old_online_seconds"] for p in good)
                expected = dict(loss_difference=_mean(p["loss_difference"] for p in good),
                                average_loss_difference=_mean(p["average_loss_difference"] for p in good),
                                new_online_seconds=new_time, old_online_seconds=old_time,
                                time_ratio=new_time/old_time if old_time > 0 else None)
                audit.check(all(close(row.get(k), v) if v is not None else row.get(k) is None for k, v in expected.items()),
                            "within_dataset_repetitions_averaged", group=identity, seed=seed)
                good_values.append(expected)
        audit.check(saved.get("datasets_expected") == len(by_seed) and saved.get("datasets_available") == len(good_values),
                    "seed_summary_availability", group=identity)
        for field, item in (("mean_loss_difference", "loss_difference"),
                            ("mean_average_loss_difference", "average_loss_difference"), ("mean_time_ratio", "time_ratio")):
            expected = _mean(v[item] for v in good_values if finite(v[item]))
            audit.check(close(saved.get(field), expected) if expected is not None else saved.get(field) is None,
                        "across_dataset_average_recomputed", group=identity, field=field)
    confirmation_rows = report["confirmation"]
    audit.check({(row["round"], row["old_algorithm"]) for row in confirmation_rows} == {
        (n, algo) for n in (100, 500) for algo in ("ours", "sigmoid", "aioli")}, "confirmation_checkpoint_algorithm_coverage")
    source_specs = audit.load(results/"shared_compare_confirmation"/"manifest.json")
    groups = defaultdict(list)
    for spec in source_specs:
        groups[spec["case_id"]].append(spec)
    for saved in confirmation_rows:
        n, algorithm = saved["round"], saved["old_algorithm"]
        dataset_map = {row["case_id"]: row for row in saved["dataset_values"]}
        audit.check(set(dataset_map) == set(groups), "confirmation_all_datasets_retained", round=n, algorithm=algorithm)
        differences = []
        for cid, specs in groups.items():
            new = [s for s in specs if s["algorithm"] == "shared"]
            old = [s for s in specs if s["algorithm"] == algorithm]
            ready = all(audit.runs[resolved(s["output"])]["completed_rounds"] >= n for s in new+old)
            row = dataset_map[cid]
            audit.check(row["available"] == ready, "confirmation_requires_all_repetitions", case_id=cid, round=n)
            if ready:
                a = _mean(audit.prefix(s["output"], n)[0] for s in new)
                b = _mean(audit.prefix(s["output"], n)[0] for s in old)
                audit.check(close(row.get("new_mean_loss"), a) and close(row.get("old_mean_loss"), b)
                            and close(row.get("loss_difference"), a-b), "confirmation_dataset_mean_from_raw", case_id=cid, round=n)
                differences.append(a-b)
        complete = len(differences) == len(groups)
        audit.check(saved.get("complete") == complete and saved.get("datasets_expected") == len(groups)
                    and saved.get("datasets_available") == len(differences), "confirmation_complete_availability", round=n, algorithm=algorithm)
        if complete:
            from scipy.stats import t as student_t
            mean = _mean(differences)
            audit.check(close(saved.get("mean_loss_difference"), mean), "confirmation_overall_mean", round=n, algorithm=algorithm)
            variance = math.fsum((v-mean)**2 for v in differences)/(len(differences)-1)
            half = float(student_t.ppf(.975, len(differences)-1))*math.sqrt(variance/len(differences))
            interval = saved.get("cluster_t_interval_95")
            audit.check(isinstance(interval, list) and len(interval) == 2 and close(interval[0], mean-half)
                        and close(interval[1], mean+half), "confirmation_dataset_cluster_interval_recomputed", round=n, algorithm=algorithm)
        else:
            audit.check(saved.get("mean_loss_difference") is None and saved.get("cluster_t_interval_95") is None,
                        "incomplete_confirmation_has_no_selected_subset_interval", round=n, algorithm=algorithm)
    return dict(report_complete=True, paired_rows=len(pairs), seed_aggregate_rows=len(saved_aggregates),
                confirmation_rows=len(confirmation_rows), figures=len(provenance.get("figures", {})),
                values_independently_recomputed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=BASE/"results")
    parser.add_argument("--output", type=Path, default=BASE.parent/"SHARED_TANGENT_ARTIFACT_AUDIT.json")
    args = parser.parse_args()
    assert_idle()
    start = perf_counter()
    audit = Audit()
    results = args.results.resolve()
    selected = audit.load(results/"shared_calibration"/"selected_config.json")
    sources = audit.section("source_provenance", lambda: audit_sources(audit, results))
    stages = {}
    for source_stage in COUNTS:
        assert_idle()
        bundle = audit.section("manifest:"+source_stage, lambda s=source_stage: audit_manifest(audit, results, s, selected))
        statuses = Counter()
        if bundle:
            for spec in bundle["specs"]:
                run = audit.section("new_run:"+spec["output"], lambda s=spec: audit_run(audit, s, new=True))
                if run:
                    statuses[run["status"]] += 1
            stages[source_stage] = dict(manifest_runs=len(bundle["specs"]), terminal_statuses=dict(statuses),
                                        data_identities=bundle["data_identities"])
    calibration = audit.section("held_out_selection", lambda: audit_selection(audit, results, selected))
    comparisons = {}
    for stage in COMPARISONS:
        assert_idle()
        comparisons[stage] = audit.section("comparison:"+stage, lambda s=stage: audit_comparison(audit, results, s))
    report = audit.section("final_report", lambda: audit_report(audit, results, selected))
    assert_idle()
    changed = [path for path, stamp in audit.stamps.items()
               if not Path(path).exists() or (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) != stamp]
    audit.check(not changed, "inputs_unchanged_during_audit", paths=changed)
    new_runs = [row for row in audit.runs.values() if row["algorithm"] == "shared"]
    audit.check(len(new_runs) == sum(COUNTS.values()), "all_138_new_runs_audited", actual=len(new_runs))
    result = dict(status="passed" if not audit.errors else "failed", audit_version=1,
                  generated_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=perf_counter()-start,
                  source_hashes=sources, required_new_stages=COUNTS, required_primary_stages=list(PRIMARY),
                  new_attempts=len(new_runs), new_statuses=dict(Counter(r["status"] for r in new_runs)),
                  zero_round_new_attempts=[r for r in new_runs if r["zero_round_run"]],
                  measured_failures=[r for r in audit.runs.values() if r["measured_failure"]],
                  reused_baseline_attempts=sum(r["algorithm"] != "shared" for r in audit.runs.values()),
                  stages=stages, calibration=calibration, comparisons=comparisons,
                  report=report,
                  intervals_checked=audit.interval_count, checks=audit.flags,
                  errors=audit.errors, warnings=audit.warnings, runs=list(audit.runs.values()),
                  inputs_checked=len(audit.stamps),
                  scope="Saved artifact consistency and unchanged-source/data identity; no algorithm replay, no feature-file rehash, no floating-point regret certificate.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=result["status"], new_attempts=len(new_runs),
                          baseline_attempts=result["reused_baseline_attempts"],
                          errors=len(audit.errors), warnings=len(audit.warnings), output=str(args.output))))
    if audit.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
