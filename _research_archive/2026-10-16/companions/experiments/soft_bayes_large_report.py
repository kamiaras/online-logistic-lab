"""Audit the terminal d=1000, B=1000, T=100000 prescribed Soft-Bayes pair.

This is reporting code only. It starts no sampler and imports no numerical
library. Both supervisor records must exist before any source/data hashing or
trajectory reading. Historical studies and their report files are untouched.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "experiments/results/soft_bayes_full_d1000_B1000_T100000"
CAP = 2 * 1024**3
COMMIT = "22c6ec54cd45908c0a39cbfa931a60e55646308f"
FROZEN = {
    "experiments/soft_bayes.py": "6a45ab2ba3d3d3007366a8448072021ef9329b6930b972b627a5b0c4d1c933e4",
    "SOFT_BAYES_COMPLETE.md": "a95bd46c1c544fcb6797a255754e30d2ee6471d3d24e44ea51e3d4df8b245b2b",
    "research/author_code_comparison_2026-09-11/a9a_experiment.py": "2632499867210720d9b0156678994597ce3cd5c09a079b692cebf5de706ab9cc",
}
COLUMNS = ["raw_logit", "output_logit", "loss", "online_seconds",
           "state_bytes", "author_clipped_loss"]
RECORD = struct.Struct("<6d")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def logistic_loss(score, label):
    value = -label * score
    return max(0., value) + math.log1p(math.exp(-abs(value)))


def native_clipped_loss(score, label):
    if score >= 0.:
        probability = 1. / (1. + math.exp(-score))
    else:
        exponential = math.exp(score)
        probability = exponential / (1. + exponential)
    probability = min(1. - 1e-12, max(1e-12, probability))
    return -math.log(probability if label == 1 else 1. - probability)


def number(value, digits=6):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def audit_trajectory(path, n, labels, author, prefix, check, run_id):
    """Read exactly the committed records; exclude any tail left by a kill."""
    losses, native_losses = [], []
    maximum_residual = maximum_native_residual = 0.
    finite = equal_scores = nonnegative = monotone = integral_state = True
    sentinel = author_state = True
    previous_time, last_state, prefix_time, prefix_state = 0., None, 0., None
    expected_author_state = 8 * (1000**2 + 2*1000)
    with Path(path).open("rb") as stream:
        for row in range(n):
            payload = stream.read(RECORD.size)
            if len(payload) != RECORD.size:
                raise RuntimeError(f"Truncated committed trajectory: {run_id}, row {row+1}")
            raw, output, loss, online, state, clipped = RECORD.unpack(payload)
            row_finite = all(math.isfinite(v) for v in (raw, output, loss, online, state))
            finite &= row_finite
            equal_scores &= raw == output
            nonnegative &= min(loss, online, state) >= 0.
            monotone &= online >= previous_time
            integral_state &= math.isfinite(state) and state == int(state) and int(state) % 8 == 0
            if not row_finite:
                raise RuntimeError(f"Nonfinite committed primary record: {run_id}, row {row+1}")
            label = 1 if labels[row] == 1 else -1
            maximum_residual = max(maximum_residual, abs(loss - logistic_loss(output, label)))
            losses.append(loss)
            if author:
                sentinel &= math.isfinite(clipped) and clipped >= 0.
                if not math.isfinite(clipped):
                    raise RuntimeError(f"Nonfinite native loss: {run_id}, row {row+1}")
                maximum_native_residual = max(maximum_native_residual,
                                             abs(clipped-native_clipped_loss(output, label)))
                native_losses.append(clipped)
                author_state &= state == expected_author_state
            else:
                sentinel &= math.isnan(clipped)
            previous_time, last_state = online, int(state)
            if row+1 == prefix:
                prefix_time, prefix_state = online, int(state)
    check(f"committed_values:{run_id}", finite and equal_scores and nonnegative
          and monotone and integral_state)
    check(f"all_committed_stable_losses:{run_id}", maximum_residual <= 5e-12,
          {"maximum_absolute_residual": maximum_residual, "records_checked": n})
    check(f"author_loss_or_non_author_sentinel:{run_id}", sentinel
          and (not author or maximum_native_residual <= 2e-8),
          {"maximum_native_absolute_residual": maximum_native_residual if author else None})
    if author:
        check(f"author_retained_arrays:{run_id}", author_state,
              {"expected_bytes": expected_author_state})
    total, common_total = math.fsum(losses), math.fsum(losses[:prefix])
    return dict(cumulative_loss=total, mean_loss=total/n if n else None,
                native_clipped_cumulative_loss=math.fsum(native_losses) if author else None,
                native_clipped_mean_loss=math.fsum(native_losses)/n if author and n else None,
                completed_prefix_online_seconds=previous_time,
                final_committed_state_bytes=last_state,
                common_prefix=dict(rounds=prefix, cumulative_loss=common_total,
                                   mean_loss=common_total/prefix if prefix else None,
                                   online_seconds=prefix_time, state_bytes=prefix_state,
                                   native_clipped_cumulative_loss=math.fsum(native_losses[:prefix])
                                   if author else None))


def generate(study=DEFAULT_STUDY):
    study = Path(study).resolve()
    manifest_path = study / "manifest.json"
    manifest = read(manifest_path)
    planned = manifest["runs"]
    if len(planned) != 2 or {s["algorithm"] for s in planned} != {"soft_bayes", "author_aioli"}:
        raise ValueError("Expected exactly one prescribed Soft-Bayes and one released-AIOLI attempt")
    supervisors = {}
    for spec in planned:
        path = Path(spec["output"]) / "supervisor.json"
        if not path.exists():
            raise RuntimeError("Report deferred: both measured attempts must be terminal")
        supervisor = read(path)
        if supervisor.get("status") in (None, "running"):
            raise RuntimeError("Report deferred: a measured attempt is still running")
        supervisors[spec["algorithm"]] = supervisor
    # The existing read-only process guard excludes this process and its parents,
    # allowing the sequential campaign to invoke us after its last worker exits.
    from .soft_bayes_supervisor import assert_idle
    assert_idle()
    started = perf_counter()
    checks, records = [], []
    audit_path = study / "artifact_audit.json"

    def check(name, passed, detail=None):
        item = {"check": name, "passed": bool(passed)}
        if detail is not None:
            item["detail"] = detail
        checks.append(item)
        return bool(passed)

    def audit_document():
        return dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
                    manifest=str(manifest_path), manifest_sha256=digest(manifest_path),
                    report_source_sha256=digest(Path(__file__)),
                    passed=all(c["passed"] for c in checks),
                    attempted_runs=len(planned), checked_runs=len(records),
                    checks=checks, elapsed_seconds=perf_counter()-started,
                    scope="Source/data identity, committed losses, recorded schedule and resource evidence; not a sampler mixing or finite-bit regret certificate")

    def require_checks():
        dump(audit_path, audit_document())
        failed = [item["check"] for item in checks if not item["passed"]]
        if failed:
            raise RuntimeError("Artifact audit failed: " + "; ".join(failed))

    check("requested_single_pair", len(planned) == 2
          and all(s["d"] == 1000 and s["B"] == 1000 and s["T"] == 100000
                  and s.get("rounds", s["T"]) == 100000 for s in planned))
    check("three_hour_allowance_per_method", all(s["seconds"] == 10800 for s in planned))
    check("unique_run_ids", len({s.get("id", s["algorithm"]) for s in planned}) == 2)
    for path, expected_hash in FROZEN.items():
        check("frozen_source:"+path, digest(ROOT/path) == expected_hash)

    dataset = Path(manifest["data_path"]).resolve()
    data_hashes = {name: digest(dataset/name) for name in ("X.bin", "y.bin", "data.json")}
    check("frozen_dataset_hashes", data_hashes == manifest["data_hashes"])
    metadata = read(dataset / "data.json")
    case = metadata["case"]
    check("dataset_dimensions_and_radius", case["d"] == 1000 and case["B"] == 1000
          and case["T"] == 100000 and case.get("B_generator") == 3.)
    check("dataset_file_sizes", (dataset/"X.bin").stat().st_size == 8*1000*100000
          and (dataset/"y.bin").stat().st_size == 100000)
    labels = (dataset / "y.bin").read_bytes()
    check("all_dataset_labels", len(labels) == 100000 and all(v in (1, 255) for v in labels))
    check("stored_feature_norm_bound", 0. <= metadata["feature_norm_min"]
          <= metadata["feature_norm_max"] <= 1.+1e-12)
    prefix = min(int(s.get("worker", {}).get("committed_round", 0)) for s in supervisors.values())
    source_versions = defaultdict(set)

    for plan in planned:
        algorithm = plan["algorithm"]
        run_id = plan.get("id", algorithm)
        output = Path(plan["output"]).resolve()
        spec = read(output/"spec.json")
        supervisor = supervisors[algorithm]
        worker = supervisor.get("worker", {})
        n = int(worker.get("committed_round", 0))
        check(f"planned_specification:{run_id}", all(spec.get(k) == v for k, v in plan.items()))
        check(f"same_frozen_input:{run_id}", Path(spec["data_path"]).resolve() == dataset
              and spec.get("data_hashes") == data_hashes)
        check(f"prefix_bounds:{run_id}", worker.get("committed_round") == n and 0 <= prefix <= n <= 100000)
        check(f"complete_status:{run_id}", supervisor["status"] != "completed"
              or (n == 100000 and not supervisor.get("checkpoint_is_censored")))
        before = read(output/"source_hashes_before.json")
        after = read(output/"source_hashes_after.json")
        check(f"sources_unchanged:{run_id}", before == after and supervisor.get("sources_unchanged") is True)
        check(f"current_measured_source_identity:{run_id}", all(digest(ROOT/p) == value for p, value in after.items()))
        check(f"frozen_core_sources_recorded:{run_id}", all(before.get(p) == value for p, value in FROZEN.items()))
        for path, value in before.items():
            source_versions[path].add(value)
        verification = supervisor.get("job_verification") or {}
        check(f"verified_actual_process_2GiB_cap:{run_id}",
              supervisor.get("hard_process_commit_limit") == CAP
              and supervisor.get("os_hard_limit_active") is True
              and verification.get("process_memory_limit") == CAP
              and verification.get("verified_inside_actual_interpreter") is True
              and verification.get("pid") == supervisor.get("actual_worker_pid"))
        pools = worker.get("threadpools", [])
        check(f"one_worker_and_one_BLAS_thread:{run_id}", supervisor.get("workers_at_once") == 1
              and supervisor.get("blas_threads") == 1 and bool(pools)
              and all(p.get("num_threads") == 1 for p in pools))
        trajectory, path = worker.get("trajectory", {}), output/"trajectory.bin"
        size = path.stat().st_size
        check(f"committed_trajectory_schema:{run_id}", trajectory.get("columns") == COLUMNS
              and trajectory.get("record_bytes") == RECORD.size
              and trajectory.get("committed_bytes") == n*RECORD.size
              and size >= n*RECORD.size)
        require_checks()
        author = algorithm == "author_aioli"
        values = audit_trajectory(path, n, labels, author, prefix, check, run_id)
        check(f"committed_cumulative_loss:{run_id}", abs(values["cumulative_loss"]-worker["cumulative_loss"]) <= 1e-8)
        check(f"committed_online_time:{run_id}", abs(values["completed_prefix_online_seconds"]
              -worker.get("completed_prefix_online_seconds", 0.)) <= 1e-9)
        diagnostic = worker.get("diagnostics", {})
        if author:
            check(f"committed_native_loss:{run_id}", abs(values["native_clipped_cumulative_loss"]
                  -worker["author_reported_cumulative_loss"]) <= 1e-8)
            configuration = diagnostic.get("configuration", {})
            check(f"unchanged_released_author_configuration:{run_id}", diagnostic.get("pinned_commit") == COMMIT
                  and diagnostic.get("native_prediction_and_update_unchanged") is True
                  and configuration.get("R") == 1.
                  and configuration.get("lam") == 1e-6
                  and configuration.get("T_inner") == 80
                  and configuration.get("omega_solver") == "lbfgs")
        else:
            check(f"fully_prescribed_main_schedule:{run_id}", spec["config"].get("mode") == "prescribed"
                  and spec["config"].get("max_total_moves", 10000000) is None
                  and diagnostic.get("mode") == "prescribed"
                  and diagnostic.get("samples") == 100000
                  and diagnostic.get("prescribed_samples") == 100000
                  and diagnostic.get("history_length") == n
                  and "conditional_rejection_cap" in diagnostic and diagnostic["conditional_rejection_cap"] is None
                  and "normal_rejection_cap" in diagnostic and diagnostic["normal_rejection_cap"] is None
                  and diagnostic.get("section_10_global_operation_cap_implemented") is False)
            check(f"worker_preflight_prescribed_counts:{run_id}",
                  diagnostic.get("preflight", {}).get("actual_samples") == 100000
                  and diagnostic.get("preflight", {}).get("mode") == "prescribed"
                  and diagnostic.get("preflight", {}).get("feasible") is True)
        censored = not supervisor.get("worker_final_available", False)
        checkpoint_censored = bool(supervisor.get("checkpoint_is_censored"))
        record = dict(id=run_id, algorithm=algorithm, d=spec["d"], B=spec["B"], T=spec["T"],
                      seed=spec["seed"], configuration=spec.get("config", {}),
                      status=supervisor["status"], worker_status=worker.get("status"),
                      error=worker.get("error"), failure_phase=worker.get("failure_phase"),
                      committed_round=n, full_completion=supervisor["status"] == "completed" and n == 100000,
                      recorded_online_seconds=worker.get("online_seconds"),
                      recorded_online_is_lower_bound=censored,
                      process_wall_seconds=supervisor.get("process_wall_seconds"),
                      external_wall_budget_seconds=supervisor.get("external_wall_budget_seconds"),
                      peak_private_bytes=max(worker.get("memory", {}).get("peak_private_commit", 0),
                                             supervisor.get("monitored_peak_private", 0)),
                      peak_private_is_lower_bound=checkpoint_censored,
                      checkpoint_is_censored=checkpoint_censored,
                      uncommitted_trajectory_bytes=size-n*RECORD.size,
                      trajectory_sha256=digest(path),
                      completed_round_diagnostics=diagnostic,
                      diagnostic_scope="Last worker snapshot; after an external kill, unfinished next-round work and its counters are absent",
                      metrics=worker.get("metrics", {}),
                      native_prediction_seconds=worker.get("native_prediction_seconds"),
                      native_update_seconds=worker.get("native_update_seconds"),
                      output=str(output), **values)
        records.append(record)
    check("single_measured_source_version", all(len(v) == 1 for v in source_versions.values()))
    check("equal_prefix", all(r["common_prefix"]["rounds"] == prefix <= r["committed_round"] for r in records))
    require_checks()
    by_algorithm = {r["algorithm"]: r for r in records}
    soft = by_algorithm["soft_bayes"]["common_prefix"]
    author = by_algorithm["author_aioli"]["common_prefix"]
    summary = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
                   manifest=str(manifest_path), manifest_sha256=digest(manifest_path),
                   study=str(study), requested=dict(d=1000, B=1000, T=100000),
                   author_commit=COMMIT, frozen_core_source_hashes=FROZEN,
                   measured_source_hashes={p: sorted(v) for p, v in source_versions.items()},
                   data_path=str(dataset), data_hashes=data_hashes, data_metadata=metadata,
                   runs=records, common_prefix_rounds=prefix,
                   common_prefix_cumulative_loss_difference=soft["cumulative_loss"]-author["cumulative_loss"] if prefix else None,
                   common_prefix_mean_loss_difference=soft["mean_loss"]-author["mean_loss"] if prefix else None,
                   common_prefix_online_time_ratio=soft["online_seconds"]/author["online_seconds"] if prefix and author["online_seconds"] > 0. else None,
                   equal_prefix_interpretation=("Only initial neutral prediction is shared; no meaningful evidence about posterior learning"
                                                if prefix == 1 else "No shared completed prediction" if not prefix else
                                                "Single fixed dataset and sampler seed at an attained, potentially runtime-selected prefix"),
                   numerical_comparator_fit=None,
                   regret_difference_scope="At the equal prefix, loss difference equals regret difference against any common comparator; no absolute-regret estimate is made",
                   neutral_mean_loss=math.log(2), preflight=manifest.get("preflight"),
                   limits=dict(allowance_seconds_per_method=10800, workers_at_once=1,
                               blas_threads=1, hard_private_commit_bytes=CAP),
                   audit_passed=True, audit_checks=len(checks), audit_path=str(audit_path),
                   report_elapsed_seconds=perf_counter()-started)
    dump(study/"summary.json", summary)
    (study/"REPORT.md").write_text(make_markdown(summary), encoding="utf-8")
    dump(audit_path, audit_document())
    return summary


def make_markdown(summary):
    by_algorithm = {r["algorithm"]: r for r in summary["runs"]}
    soft, author = by_algorithm["soft_bayes"], by_algorithm["author_aioli"]
    prefix = summary["common_prefix_rounds"]
    lines = ["# Prescribed Soft Bayes versus released AIOLI: d=1000, B=1000, T=100000", "",
             f"Generated {summary['generated_utc']}. One run of each method on the same fixed stored dataset, with a three-hour allowance per method.", "",
             f"Soft Bayes committed **{soft['committed_round']:,}/{soft['T']:,} rounds** ({soft['status']}); released AIOLI committed **{author['committed_round']:,}/{author['T']:,} rounds** ({author['status']}). Their common completed prefix is **{prefix:,} rounds**.", ""]
    if prefix <= 1:
        lines += ["The shared prefix contains only the initial neutral prediction, or no complete prediction. It gives no meaningful comparison of learning or posterior tracking. The unfinished Soft-Bayes prediction has no published loss. A timeout is evidence about this implementation's progress under the allowance, not a refutation of the exact-real regret theorem.", ""]
    lines += ["## Same-prefix comparison", "",
              "Losses use the actual returned logit and stable logistic loss. Online time includes construction, prediction and label update; it excludes module imports, input streaming and recording. The time below stops at the last completed round in this common prefix and excludes an unfinished later prediction. Neutral loss is 0.693147181 per round.", "",
              "| Method | Common completed rounds | Cumulative loss | Loss/round | Completed-prefix online seconds |",
              "|---|---:|---:|---:|---:|"]
    for run, name in ((soft, "Prescribed Soft Bayes"), (author, "Released AIOLI")):
        common = run["common_prefix"]
        lines.append(f"| {name} | {prefix:,} | {number(common['cumulative_loss'], 9)} | {number(common['mean_loss'], 9)} | {number(common['online_seconds'])} |")
    lines += ["", f"At this same prefix, Soft Bayes minus AIOLI cumulative loss is {number(summary['common_prefix_cumulative_loss_difference'], 12)}. This is also the regret difference against any common comparator. No offline comparator was fitted here and no absolute regret is reported.", "",
              "## Each method at its own attained endpoint", "",
              "The rows below have their stated individual endpoints. They are not a paired full-horizon loss or memory comparison. A ≥ mark on recorded online time denotes the last checkpoint's lower bound after external termination; the unknown time inside the interrupted prediction is absent. Process-wall time is separately measured and includes imports, input, recording and supervision.", "",
              "| Method | Status | Completed rounds | Cumulative loss | Loss/round | Recorded online seconds | Process-wall seconds |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for run, name in ((soft, "Prescribed Soft Bayes"), (author, "Released AIOLI")):
        online = ("≥" if run["recorded_online_is_lower_bound"] else "")+number(run["recorded_online_seconds"])
        lines.append(f"| {name} | {run['status']} | {run['committed_round']:,} | {number(run['cumulative_loss'], 9)} | {number(run['mean_loss'], 9)} | {online} | {number(run['process_wall_seconds'])} |")
    lines += ["", f"AIOLI's unchanged native probability-clipped loss totals {number(author['native_clipped_cumulative_loss'], 9)} at its {author['committed_round']:,}-round endpoint ({number(author['native_clipped_mean_loss'], 9)} per round). The stable-score total is primary; the native reporting clip does not change its updates.", "",
              "| Method | Completed rounds | Owned persistent array bytes at completed endpoint | Peak process private MiB |",
              "|---|---:|---:|---:|"]
    for run, name in ((soft, "Prescribed Soft Bayes"), (author, "Released AIOLI")):
        peak = ("≥" if run["peak_private_is_lower_bound"] else "")+number(run["peak_private_bytes"]/1024**2, 3)
        state = "unavailable" if run["final_committed_state_bytes"] is None else f"{run['final_committed_state_bytes']:,}"
        lines.append(f"| {name} | {run['committed_round']:,} | {state} | {peak} |")
    lines += ["", "Owned state counts the algorithm's retained arrays. Process memory also includes runtime, imports, input buffers and scratch. A censored peak is a lower bound from available monitor samples and the latest worker OS snapshot. No memory measurement at an uncompleted horizon is inferred.", "",
              "## Prescribed schedule and provenance", "",
              "Soft Bayes uses the unchanged main manuscript implementation: m=T=100000 independent fresh Gaussian-prior chains on each nonsymmetric round, the full N=ceil(H/log1p(lambda/chi)) transitions per chain, chi=lambda+2E, uniform-only scalar normal generation, uncapped conditional tangent rejection, and add-one smoothed mean probabilities. The default preflight refusal was explicitly disabled; sample and transition counts were not shortened. Round one is exactly neutral by symmetry. Section 10's optional primitive-operation budget is not implemented, and external wall-clock termination is not that algorithm.", "",
              "The frozen preflight is retained in [manifest.json](manifest.json) and [summary.json](summary.json). Its sufficient transition and exact-real operation bounds are metadata, not executed counts, runtime predictions, measured hardware instructions or complexity lower bounds. Worker diagnostics after a kill describe the last checkpoint and cannot establish how far the unfinished next prediction progressed.", "",
              f"AIOLI is the sigmoid/MALA paper authors' archived comparison code at commit `{COMMIT}`. Its actual initial precision is B² and its linear-statistic increment differs from the defining AIOLI surrogate. Both behaviors are retained. This is not a verified release by the original AIOLI authors. Configuration: R=1, named lambda=B^-2, T_inner=80, omega_solver=lbfgs; native prediction and update are unchanged.", "",
              f"Data: `{summary['data_path']}`. Both methods read these same stored rows and labels in order; generator radius B_generator=3 is distinct from algorithm B=1000. Dataset seed: {summary['data_metadata']['case'].get('data_seed', 'unavailable')}; Soft-Bayes sampler seed: {soft['seed']}; AIOLI seed: {author['seed']} (the released AIOLI path is deterministic).", "",
              "Each worker uses one BLAS thread and a verified Windows Job Object cap of 2 GiB private commit. The two measured workers run sequentially. Exact external process-wall deadlines are in the supervisor records and summary; the online allowance and process-wall deadline are different timing scopes.", "",
              "The [artifact audit](artifact_audit.json) checks frozen sources/data, recorded configuration, actual-process memory-cap and thread evidence, all committed stable losses, native AIOLI reporting losses, committed totals and equal-prefix selection. Hashing and report generation occur only after both measured attempts terminate. Sources, manuscript, trajectory records, specifications, supervisor evidence and preflight metadata are retained.", "",
              "This single fixed-data run provides no multi-seed uncertainty estimate. No loss or timing is extrapolated to an uncompleted horizon. Float64 arithmetic has no inherited exact-real regret certificate. The measured implementation and any censored observations must remain distinct from the manuscript's fully completed exact-real algorithm.", "",
              f"Audit: **{summary['audit_checks']} checks passed**. Full machine-readable results: [summary.json](summary.json).", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    summary = generate(args.study)
    print(json.dumps({"report": str(args.study.resolve()/"REPORT.md"),
                      "common_prefix_rounds": summary["common_prefix_rounds"],
                      "statuses": {r["algorithm"]: r["status"] for r in summary["runs"]},
                      "audit_checks": summary["audit_checks"]}))


if __name__ == "__main__":
    main()
